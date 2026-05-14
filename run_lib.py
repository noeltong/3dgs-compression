import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from models.model import GaussianVolumeCodec
from utils.data import VolumeBatchSampler, make_volume_bundle
from utils.optim import get_lr, get_optim
from utils.utils import (
    TimeCalculator,
    build_coordinate_grid,
    derive_num_gaussians,
    flatten_volume,
    normalize_volume,
    read_volume,
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
    coords = build_coordinate_grid(normalized_volume.shape, config.data.coord_norm)
    values = flatten_volume(normalized_volume)
    return make_volume_bundle(
        coords=coords,
        values=values,
        shape=normalized_volume.shape,
        normalization=normalization,
        raw_bytes=raw_bytes,
        raw_volume=volume.astype(np.float32),
        normalized_volume=normalized_volume.astype(np.float32),
    )


def _compute_psnr(mse):
    return 10.0 * math.log10(1.0 / mse)


def _compute_eval_metrics(gt_arr: np.ndarray, pred_arr: np.ndarray):
    mse = float(((pred_arr - gt_arr) ** 2).mean())
    mae = float(np.abs(pred_arr - gt_arr).mean())
    nrmse = float(np.linalg.norm(pred_arr - gt_arr) / (np.linalg.norm(gt_arr) + 1e-8))
    psnr = float(peak_signal_noise_ratio(gt_arr, pred_arr, data_range=1.0))
    ssim = float(structural_similarity(gt_arr, pred_arr, data_range=1.0))
    return {
        "MSE": mse,
        "MAE": mae,
        "NRMSE": nrmse,
        "SSIM": ssim,
        "PSNR": psnr,
    }


def _save_grayscale_image(image: np.ndarray, path: Path):
    img = np.asarray(image, dtype=np.float32)
    img = img - float(img.min())
    denom = float(img.max())
    if denom > 0.0:
        img = img / denom
    img_u8 = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(img_u8, mode="L").save(path)


def _denormalize_prediction(pred_volume: np.ndarray, bundle):
    if bundle.normalization["mode"] != "minmax":
        return pred_volume
    vmin = float(bundle.normalization["vmin"])
    vmax = float(bundle.normalization["vmax"])
    return pred_volume * (vmax - vmin + 1e-6) + vmin


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
        pred_norm = torch.cat(outputs, dim=0).reshape(bundle.shape).numpy().astype(np.float32)
    return pred_norm


def _run_and_save_evaluation(model, bundle, config, device, eval_dir, model_weights_path):
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    pred_norm = _reconstruct_volume(model, bundle, device, config.eval.chunk_size)
    gt = bundle.raw_volume.astype(np.float32)
    pred = _denormalize_prediction(pred_norm, bundle)

    gt_normed = gt / max(float(gt.max()), 1e-8)
    pred_normed = pred.clip(float(gt.min()), float(gt.max())) / max(float(gt.max()), 1e-8)

    metrics = {"info": _compute_eval_metrics(gt_normed, pred_normed)}
    data_task = str(getattr(config.data, "task", "pam")).lower()
    if data_task != "pam":
        raise ValueError(f"Unsupported data task for evaluation: {data_task}")

    map_gt = np.max(gt_normed, axis=-1)
    map_pred = np.max(pred_normed, axis=-1)
    metrics["map"] = _compute_eval_metrics(map_gt, map_pred)

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
        loss = criterion(preds, batch_values)

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

        if step % config.training.eval_freq == 0 or step == config.training.max_steps:
            metrics = _evaluate_model(model, bundle, device, config.eval.chunk_size)
            exp_logger.logkvs({"step": step, **metrics})
            exp_logger.dumpkvs()
            if metrics["eval_loss"] < best_eval:
                best_eval = metrics["eval_loss"]
                _save_checkpoint(model, optimizer, scheduler, step, ckpt_dir, "best.pt")

        if step % config.training.ckpt_freq == 0 or step == config.training.max_steps:
            _save_checkpoint(model, optimizer, scheduler, step, ckpt_dir, f"step_{step}.pt")

    final_eval = _run_and_save_evaluation(
        model=model,
        bundle=bundle,
        config=config,
        device=device,
        eval_dir=run_root / "eval",
        model_weights_path=Path(ckpt_dir) / "model_final_weights.pth",
    )
    exp_logger.logkvs({"phase": "post_train_eval", **final_eval})
    exp_logger.dumpkvs()


def eval(config, workdir, train_dir="train", eval_dir="eval"):
    _configure_runtime(config)
    device = _get_device()
    train_workdir = os.path.join(workdir, train_dir)
    eval_workdir = os.path.join(workdir, eval_dir)
    os.makedirs(eval_workdir, exist_ok=True)
    ckpt_path = os.path.join(train_workdir, "ckpt", "best.pt")
    if not os.path.exists(ckpt_path):
        raise ValueError(f"Checkpoint not found: {ckpt_path}")

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
