import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image

from models.model import GaussianVolumeCodec
from utils.data import VolumeBatchSampler, make_volume_bundle
from utils.optim import get_lr, get_optim
from utils.utils import (
    TimeCalculator,
    build_coordinate_grid,
    compute_reconstruction_metrics,
    denormalize_from_training_scale,
    derive_num_gaussians,
    flatten_volume,
    map_gradient_consistency_loss,
    map_projection_loss,
    normalize_volume,
    project_map,
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
    coords = build_coordinate_grid(normalized_volume.shape, config.data.coord_norm)
    values = flatten_volume(training_volume)
    return make_volume_bundle(
        coords=coords,
        values=values,
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


def _reconstruct_volume(model, bundle, device, chunk_size):
    model.eval()
    with torch.inference_mode():
        outputs = []
        flat_coords = bundle.coords.reshape(-1, 3)
        for start in range(0, flat_coords.shape[0], chunk_size):
            stop = min(start + chunk_size, flat_coords.shape[0])
            outputs.append(model(flat_coords[start:stop].to(device)).cpu())
        pred_training = torch.cat(outputs, dim=0).reshape(bundle.shape).numpy().astype(np.float32)
    return pred_training


def _sample_map_training_patch(model, bundle, config, device):
    sample_mode = str(config.training.map_column_sample_mode).lower()
    if sample_mode != "patch":
        raise ValueError(f"Unsupported MAP column sample mode: {sample_mode}")

    sample_height = int(config.training.map_column_sample_height)
    sample_width = int(config.training.map_column_sample_width)
    if sample_height <= 0 or sample_width <= 0:
        raise ValueError("MAP column patch dimensions must be positive")

    height, width, depth = bundle.shape
    if sample_height > height or sample_width > width:
        raise ValueError(
            "MAP column patch dimensions exceed the training volume shape: "
            f"patch=({sample_height}, {sample_width}), shape=({height}, {width}, {depth})"
        )

    start_x_max = height - sample_height
    start_y_max = width - sample_width
    start_x = 0 if start_x_max == 0 else int(torch.randint(0, start_x_max + 1, (1,)).item())
    start_y = 0 if start_y_max == 0 else int(torch.randint(0, start_y_max + 1, (1,)).item())

    coords_grid = bundle.coords.reshape(height, width, depth, 3)
    coord_patch = coords_grid[
        start_x : start_x + sample_height,
        start_y : start_y + sample_width,
        :,
        :,
    ]
    pred_patch = model(coord_patch.reshape(-1, 3).to(device)).reshape(
        sample_height, sample_width, depth
    )
    target_patch = torch.from_numpy(
        bundle.training_volume[
            start_x : start_x + sample_height,
            start_y : start_y + sample_width,
            :,
        ]
    ).to(device=device, dtype=pred_patch.dtype)
    return pred_patch, target_patch


def _compute_map_loss_terms(model, bundle, config, device):
    if str(getattr(config.data, "task", "pam")).lower() != "pam":
        raise ValueError("MAP-aware loss is currently only supported for PAM data.")

    pred_volume, target_volume = _sample_map_training_patch(model, bundle, config, device)

    projection_mode = str(config.training.map_loss_type).lower()
    tau = float(config.training.map_softmax_tau)
    topk = int(config.training.map_topk)

    map_loss = map_projection_loss(
        pred_volume,
        target_volume,
        mode=projection_mode,
        tau=tau,
        topk=topk,
        loss_type="mse",
    )
    map_grad_loss = map_gradient_consistency_loss(
        pred_volume,
        target_volume,
        projection_mode=projection_mode,
        tau=tau,
        topk=topk,
        gradient_mode="sobel",
        loss_type="mse",
    )
    pred_map = project_map(pred_volume, mode=projection_mode, tau=tau, topk=topk)
    target_map = project_map(target_volume, mode=projection_mode, tau=tau, topk=topk)
    return {
        "map_loss": map_loss,
        "map_grad_loss": map_grad_loss,
        "pred_map_mean": float(pred_map.detach().mean().item()),
        "target_map_mean": float(target_map.detach().mean().item()),
        "sampled_columns": int(pred_volume.shape[0] * pred_volume.shape[1]),
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
    timer = TimeCalculator()

    best_eval = float("inf")
    for step in range(1, config.training.max_steps + 1):
        model.train()
        batch_coords, batch_values = sampler.next()
        batch_coords = batch_coords.to(device)
        batch_values = batch_values.to(device)

        preds = model(batch_coords)
        loss_3d = criterion(preds, batch_values)

        map_loss_active = bool(config.training.map_loss_enable) and (
            step >= int(config.training.map_loss_start_step)
        )
        map_loss = preds.new_zeros(())
        map_grad_loss = preds.new_zeros(())
        map_aux_stats = {
            "pred_map_mean": 0.0,
            "target_map_mean": 0.0,
            "sampled_columns": 0,
        }
        if map_loss_active:
            map_terms = _compute_map_loss_terms(model, bundle, config, device)
            map_loss = map_terms["map_loss"]
            map_grad_loss = map_terms["map_grad_loss"]
            map_aux_stats = {
                "pred_map_mean": map_terms["pred_map_mean"],
                "target_map_mean": map_terms["target_map_mean"],
                "sampled_columns": map_terms["sampled_columns"],
            }

        loss = (
            loss_3d
            + float(config.training.map_loss_weight) * map_loss
            + float(config.training.map_grad_loss_weight) * map_grad_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.model.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), config.model.clip_grad_norm)
        optimizer.step()
        scheduler.step()

        if step % config.training.log_freq == 0 or step == 1:
            payload = model.get_payload_stats()
            exp_logger.logkvs(
                {
                    "step": step,
                    "train_loss": float(loss.item()),
                    "loss_3d": float(loss_3d.item()),
                    "loss_map": float(map_loss.item()),
                    "loss_map_grad": float(map_grad_loss.item()),
                    "loss_total": float(loss.item()),
                    "map_loss_active": int(map_loss_active),
                    "pred_map_mean": float(map_aux_stats["pred_map_mean"]),
                    "target_map_mean": float(map_aux_stats["target_map_mean"]),
                    "map_sampled_columns": int(map_aux_stats["sampled_columns"]),
                    "lr": get_lr(optimizer),
                    "elapsed": timer.period(),
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
