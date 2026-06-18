import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image

from models.losses import HyperbolaGradientPatchLoss
from models.model import GaussianVolumeCodec
from utils.data import VolumeBatchSampler, VolumePatchSampler, make_volume_bundle
from utils.optim import get_lr, get_optim
from utils.utils import (
    TimeCalculator,
    build_coordinate_grid,
    compute_reconstruction_metrics,
    denormalize_from_training_scale,
    derive_num_gaussians,
    flatten_volume,
    normalize_volume,
    read_volume,
    scale_volume_for_training,
    seed_everything,
)
from utils import logger as exp_logger


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _configure_runtime(config):
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    if config.use_deterministic_algorithms:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def _load_volume_bundle(config):
    volume, raw_bytes = read_volume(config.data.path)
    normalized_volume, normalization = normalize_volume(volume, config.data.normalize)
    training_volume = scale_volume_for_training(normalized_volume, config.data.scale_max)
    coord_volume = build_coordinate_grid(normalized_volume.shape, config.data.coord_norm).reshape(
        *normalized_volume.shape, 3
    )
    coords = coord_volume.reshape(-1, 3)
    values = flatten_volume(training_volume)
    return make_volume_bundle(
        coords=coords,
        coord_volume=coord_volume,
        values=values,
        value_volume=torch.from_numpy(training_volume.astype(np.float32)),
        shape=normalized_volume.shape,
        normalization=normalization,
        raw_bytes=raw_bytes,
        raw_volume=volume.astype(np.float32),
        normalized_volume=normalized_volume.astype(np.float32),
        training_volume=training_volume.astype(np.float32),
        scale_max=float(config.data.scale_max),
    )


def _compute_psnr(mse):
    return 10.0 * math.log10(1.0 / mse)


def _save_grayscale_image(image: np.ndarray, path: Path):
    img = np.asarray(image, dtype=np.float32)
    img = img - float(img.min())
    denom = float(img.max())
    if denom > 0.0:
        img = img / denom
    img_u8 = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(img_u8, mode="L").save(path)


def _evaluate_model(model, bundle, device, chunk_size):
    model.eval()
    with torch.inference_mode():
        outputs = []
        flat_coords = bundle.coords.reshape(-1, 3)
        for start in range(0, flat_coords.shape[0], chunk_size):
            stop = min(start + chunk_size, flat_coords.shape[0])
            outputs.append(model(flat_coords[start:stop].to(device)).cpu())
        preds = torch.cat(outputs, dim=0)
        targets = bundle.values
        mse = nn.functional.mse_loss(preds, targets).item()
        payload = model.get_payload_stats()
        achieved_ratio = bundle.raw_bytes * 8 / payload["total_bits_with_overhead"]
        return {
            "eval_loss": mse,
            "eval_psnr": _compute_psnr(mse),
            "payload_total_bits": payload["total_bits"],
            "payload_overhead_bits": payload["overhead_bits"],
            "payload_total_bits_with_overhead": payload["total_bits_with_overhead"],
            "payload_bits_per_gaussian": payload["bits_per_gaussian"],
            "payload_center_bits": payload["centers"]["total_bits"],
            "payload_scale_bits": payload["scales"]["total_bits"],
            "payload_intensity_bits": payload["intensities"]["total_bits"],
            "achieved_compression_ratio": achieved_ratio,
        }


def _predict_flat_training_volume(model, bundle, device, chunk_size):
    outputs = []
    flat_coords = bundle.coords.reshape(-1, 3)
    for start in range(0, flat_coords.shape[0], chunk_size):
        stop = min(start + chunk_size, flat_coords.shape[0])
        outputs.append(model(flat_coords[start:stop].to(device)).cpu())
    return torch.cat(outputs, dim=0)


def _reconstruct_volume(model, bundle, device, chunk_size):
    model.eval()
    with torch.inference_mode():
        pred_training = _predict_flat_training_volume(model, bundle, device, chunk_size).reshape(bundle.shape)
        pred_training = pred_training.numpy().astype(np.float32)
    return pred_training


def _should_run_reallocation(step, training_config):
    if not training_config.realloc_enable:
        return False
    if step < training_config.realloc_start_step:
        return False
    if step > training_config.realloc_end_step:
        return False
    return step % training_config.realloc_interval == 0


def _make_zero_reallocation_metrics():
    return {
        "realloc_event": 0,
        "realloc_num_pruned": 0,
        "realloc_num_respawned": 0,
        "realloc_score_mean_pruned": 0.0,
        "realloc_score_mean_kept": 0.0,
        "realloc_residual_mean_selected": 0.0,
    }


def _compute_reallocation_prune_count(num_gaussians, realloc_fraction):
    num_gaussians = int(num_gaussians)
    if num_gaussians <= 1:
        raise ValueError("Reallocation requires at least 2 Gaussians to preserve the fixed budget.")
    prune_count = int(math.floor(float(realloc_fraction) * float(num_gaussians)))
    prune_count = max(1, prune_count)
    prune_count = min(prune_count, num_gaussians - 1)
    if prune_count <= 0:
        raise ValueError("Reallocation prune count must be positive.")
    return prune_count


def _reset_optimizer_rows_(optimizer, parameter, row_indices):
    state = optimizer.state.get(parameter)
    if not state:
        return
    for tensor in state.values():
        if tensor is None or not torch.is_tensor(tensor):
            continue
        if tensor.ndim == 0:
            continue
        if tensor.shape[0] != parameter.shape[0]:
            continue
        tensor_row_indices = row_indices.to(device=tensor.device, dtype=torch.long)
        tensor[tensor_row_indices] = 0


def _run_reallocation_step(model, optimizer, bundle, device, config):
    params = model.get_activated_parameter_tensors()
    intensities = params["intensities"].abs().reshape(-1)
    scales = params["scales"]
    scores = intensities * scales.prod(dim=-1)

    num_gaussians = model.num_gaussians
    prune_count = _compute_reallocation_prune_count(num_gaussians, config.training.realloc_fraction)

    prune_order = torch.argsort(scores, descending=False)
    prune_indices = prune_order[:prune_count]
    keep_indices = prune_order[prune_count:]

    model.eval()
    with torch.inference_mode():
        preds = _predict_flat_training_volume(model, bundle, device, config.eval.chunk_size)
    model.train()

    targets = bundle.values.reshape(-1, 1)
    residuals = (preds - targets).abs().reshape(-1)
    selected_coord_indices = torch.topk(residuals, k=prune_count, largest=True, sorted=True).indices
    respawn_coords = bundle.coords[selected_coord_indices].to(device=device, dtype=model.center_logits.dtype)
    gt_values = targets[selected_coord_indices].to(device=device, dtype=model.intensity_logits.dtype)
    pred_values = preds[selected_coord_indices].to(device=device, dtype=model.intensity_logits.dtype)
    signed_residuals = (gt_values - pred_values).clamp(
        min=-model.intensity_range,
        max=model.intensity_range,
    )

    init_scale_value = min(max(model.init_scale, model.min_scale), model.max_scale)
    respawn_scales = torch.full(
        (prune_count, 3),
        fill_value=init_scale_value,
        device=device,
        dtype=model.raw_scales.dtype,
    )

    center_logits = model.inverse_center_parameterization(respawn_coords)
    raw_scales = model.inverse_scale_parameterization(respawn_scales)
    intensity_logits = model.inverse_intensity_parameterization(signed_residuals)
    model.overwrite_gaussian_rows_(prune_indices, center_logits, raw_scales, intensity_logits)

    optimizer_indices = prune_indices.to(device=model.center_logits.device, dtype=torch.long)
    _reset_optimizer_rows_(optimizer, model.center_logits, optimizer_indices)
    _reset_optimizer_rows_(optimizer, model.raw_scales, optimizer_indices)
    _reset_optimizer_rows_(optimizer, model.intensity_logits, optimizer_indices)

    pruned_scores = scores[prune_indices]
    kept_scores = scores[keep_indices]
    selected_residuals = residuals[selected_coord_indices]
    return {
        "realloc_event": 1,
        "realloc_num_pruned": int(prune_count),
        "realloc_num_respawned": int(prune_count),
        "realloc_score_mean_pruned": float(pruned_scores.mean().item()),
        "realloc_score_mean_kept": float(kept_scores.mean().item()),
        "realloc_residual_mean_selected": float(selected_residuals.mean().item()),
    }


def _run_and_save_evaluation(model, bundle, config, device, eval_dir, model_weights_path):
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    pred_training = _reconstruct_volume(model, bundle, device, config.eval.chunk_size)
    gt = bundle.raw_volume.astype(np.float32)
    pred = denormalize_from_training_scale(pred_training, bundle.normalization, bundle.scale_max)

    info_metrics, gt_normed, pred_normed = compute_reconstruction_metrics(gt, pred)
    metrics = {"info": info_metrics}
    data_task = str(getattr(config.data, "task", "pam")).lower()
    if data_task != "pam":
        raise ValueError(f"Unsupported data task for evaluation: {data_task}")

    map_gt_raw = np.max(gt, axis=-1)
    map_pred_raw = np.max(pred, axis=-1)
    map_metrics, map_gt, map_pred = compute_reconstruction_metrics(map_gt_raw, map_pred_raw)
    metrics["map"] = map_metrics

    np.save(eval_dir / "gt_volume.npy", gt)
    np.save(eval_dir / "pred_volume.npy", pred)
    np.save(eval_dir / "gt_map.npy", map_gt)
    np.save(eval_dir / "pred_map.npy", map_pred)
    _save_grayscale_image(map_gt, eval_dir / "gt_map.png")
    _save_grayscale_image(map_pred, eval_dir / "pred_map.png")

    results_payload = {
        "pred_info": pred,
        "gt_info": gt,
        "metrics": metrics,
        "gt_map": map_gt,
        "pred_map": map_pred,
    }
    np.savez(eval_dir / "results.npz", **results_payload)

    metrics_json = {
        "psnr": metrics["info"]["PSNR"],
        "ssim": metrics["info"]["SSIM"],
        "mse": metrics["info"]["MSE"],
        "mae": metrics["info"]["MAE"],
        "nrmse": metrics["info"]["NRMSE"],
        "psnr_map": metrics["map"]["PSNR"],
        "ssim_map": metrics["map"]["SSIM"],
        "mse_map": metrics["map"]["MSE"],
        "mae_map": metrics["map"]["MAE"],
        "nrmse_map": metrics["map"]["NRMSE"],
    }
    with open(eval_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=4)

    torch.save(model.state_dict(), eval_dir / Path(model_weights_path).name)

    return {
        "eval_loss": metrics["info"]["MSE"],
        "eval_psnr": metrics["info"]["PSNR"],
        "eval_ssim": metrics["info"]["SSIM"],
        "map_psnr": metrics["map"]["PSNR"],
        "map_ssim": metrics["map"]["SSIM"],
        "map_mse": metrics["map"]["MSE"],
        "pred_volume_path": str(eval_dir / "pred_volume.npy"),
        "metrics_path": str(eval_dir / "metrics.json"),
    }


def _save_checkpoint(model, optimizer, scheduler, step, ckpt_dir, name):
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        os.path.join(ckpt_dir, name),
    )


def _resolve_eval_checkpoint_path(train_workdir, max_steps):
    ckpt_dir = Path(train_workdir) / "ckpt"
    best_path = ckpt_dir / "best.pt"
    if best_path.exists():
        return best_path

    final_path = ckpt_dir / f"step_{int(max_steps)}.pt"
    if final_path.exists():
        return final_path

    step_candidates = []
    for path in ckpt_dir.glob("step_*.pt"):
        stem = path.stem
        try:
            step_candidates.append((int(stem.split("_", 1)[1]), path))
        except (IndexError, ValueError):
            continue
    if step_candidates:
        return max(step_candidates, key=lambda item: item[0])[1]

    raise ValueError(f"Checkpoint not found in {ckpt_dir}")


def train(config, workdir, train_dir="train"):
    _configure_runtime(config)
    seed_everything(config.seed)
    device = _get_device()

    run_root = Path(workdir)
    workdir = os.path.join(workdir, train_dir)
    ckpt_dir = os.path.join(workdir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    exp_logger.configure(dir=workdir, format_strs=["stdout", "log", "json"])
    exp_logger.info(f"Loading volume from {config.data.path}")
    bundle = _load_volume_bundle(config)
    budget = derive_num_gaussians(bundle.raw_bytes, config)
    exp_logger.info(
        f"Loaded volume shape={bundle.shape}, raw_bytes={bundle.raw_bytes}, num_gaussians={budget['num_gaussians']}"
    )

    model = GaussianVolumeCodec(
        config=config,
        num_gaussians=budget["num_gaussians"],
        dense_chunk_size=config.eval.chunk_size,
    ).to(device)
    optimizer, scheduler = get_optim(model, config)
    criterion = nn.MSELoss()
    sampler = VolumeBatchSampler(bundle.coords, bundle.values, config.training.batch_size)
    patch_loss_fn = HyperbolaGradientPatchLoss(
        lambda_grad=config.training.patch_lambda_grad,
        alpha_x=config.training.patch_grad_alpha_x,
        alpha_y=config.training.patch_grad_alpha_y,
        alpha_z=config.training.patch_grad_alpha_z,
        delta=config.training.patch_grad_delta,
    )
    patch_sampler = None
    if config.training.patch_loss_enable:
        patch_sampler = VolumePatchSampler(
            bundle.coord_volume,
            bundle.value_volume,
            config.training.patch_size,
            config.training.patch_batch_size,
        )
    timer = TimeCalculator()

    best_eval = float("inf")
    for step in range(1, config.training.max_steps + 1):
        model.train()
        batch_coords, batch_values = sampler.next()
        batch_coords = batch_coords.to(device)
        batch_values = batch_values.to(device)

        preds = model(batch_coords)
        loss_3d = criterion(preds, batch_values)
        loss_grad_patch = loss_3d.new_zeros(())
        if patch_sampler is not None:
            patch_coords, patch_values = patch_sampler.next()
            patch_coords = patch_coords.to(device)
            patch_values = patch_values.to(device)
            patch_preds = model.reconstruct_dense(
                patch_coords,
                chunk_size=config.model.forward_query_chunk_size,
            ).permute(0, 4, 1, 2, 3)
            loss_grad_patch = patch_loss_fn.gradient_loss(patch_preds, patch_values)
        loss = loss_3d + config.training.patch_lambda_grad * loss_grad_patch

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.model.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), config.model.clip_grad_norm)
        optimizer.step()
        scheduler.step()

        realloc_metrics = _make_zero_reallocation_metrics()
        if _should_run_reallocation(step, config.training):
            realloc_metrics = _run_reallocation_step(model, optimizer, bundle, device, config)

        train_log = {
            "step": step,
            "train_loss": float(loss.item()),
            "loss_3d": float(loss_3d.item()),
            "loss_grad_patch": float(loss_grad_patch.item()),
            "loss_total": float(loss.item()),
            "lr": get_lr(optimizer),
            "elapsed": timer.period(),
            **realloc_metrics,
        }
        if step % config.training.log_freq == 0 or step == 1:
            payload = model.get_payload_stats()
            train_log.update(
                {
                    "raw_bytes": budget["raw_bytes"],
                    "target_bytes": budget["target_bytes"],
                    "target_bits": budget["target_bits"],
                    "budget_overhead_bits": budget["overhead_bits"],
                    "budget_bits_per_gaussian": budget["bits_per_gaussian"],
                    "num_gaussians": budget["num_gaussians"],
                    "payload_total_bits": payload["total_bits"],
                    "payload_overhead_bits": payload["overhead_bits"],
                    "payload_total_bits_with_overhead": payload["total_bits_with_overhead"],
                    "payload_bits_per_gaussian": payload["bits_per_gaussian"],
                    "payload_center_bits": payload["centers"]["total_bits"],
                    "payload_scale_bits": payload["scales"]["total_bits"],
                    "payload_intensity_bits": payload["intensities"]["total_bits"],
                    "achieved_compression_ratio": (
                        bundle.raw_bytes * 8 / payload["total_bits_with_overhead"]
                    ),
                }
            )
        exp_logger.logkvs(train_log)
        exp_logger.dumpkvs()

        if config.training.eval_freq > 0 and step % config.training.eval_freq == 0:
            metrics = _evaluate_model(model, bundle, device, config.eval.chunk_size)
            exp_logger.logkvs({"step": step, **metrics})
            exp_logger.dumpkvs()
            if metrics["eval_loss"] < best_eval:
                best_eval = metrics["eval_loss"]
                _save_checkpoint(model, optimizer, scheduler, step, ckpt_dir, "best.pt")

        if config.training.ckpt_freq > 0 and step % config.training.ckpt_freq == 0:
            _save_checkpoint(model, optimizer, scheduler, step, ckpt_dir, f"step_{step}.pt")

    _save_checkpoint(
        model,
        optimizer,
        scheduler,
        config.training.max_steps,
        ckpt_dir,
        f"step_{config.training.max_steps}.pt",
    )

    final_eval = _run_and_save_evaluation(
        model=model,
        bundle=bundle,
        config=config,
        device=device,
        eval_dir=run_root / "eval",
        model_weights_path=Path(ckpt_dir) / "model_final_weights.pth",
    )
    final_payload_metrics = _evaluate_model(model, bundle, device, config.eval.chunk_size)
    if final_payload_metrics["eval_loss"] < best_eval:
        _save_checkpoint(
            model,
            optimizer,
            scheduler,
            config.training.max_steps,
            ckpt_dir,
            "best.pt",
        )
    exp_logger.logkvs(
        {
            "phase": "post_train_eval",
            **final_payload_metrics,
            **final_eval,
            "post_train_eval_loss_train_domain": final_payload_metrics["eval_loss"],
        }
    )
    exp_logger.dumpkvs()


def eval(config, workdir, train_dir="train", eval_dir="eval"):
    _configure_runtime(config)
    device = _get_device()
    train_workdir = os.path.join(workdir, train_dir)
    eval_workdir = os.path.join(workdir, eval_dir)
    os.makedirs(eval_workdir, exist_ok=True)
    ckpt_path = _resolve_eval_checkpoint_path(train_workdir, config.training.max_steps)

    exp_logger.configure(dir=eval_workdir, format_strs=["stdout", "log", "json"])
    bundle = _load_volume_bundle(config)
    budget = derive_num_gaussians(bundle.raw_bytes, config)
    model = GaussianVolumeCodec(
        config=config,
        num_gaussians=budget["num_gaussians"],
        dense_chunk_size=config.eval.chunk_size,
    ).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    eval_metrics = _run_and_save_evaluation(
        model=model,
        bundle=bundle,
        config=config,
        device=device,
        eval_dir=eval_workdir,
        model_weights_path=Path(ckpt_path),
    )
    payload_metrics = _evaluate_model(model, bundle, device, config.eval.chunk_size)
    exp_logger.logkvs({**payload_metrics, **eval_metrics})
    exp_logger.dumpkvs()
