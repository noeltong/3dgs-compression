import random
import time
import os

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import (
    mean_squared_error,
    normalized_root_mse,
    peak_signal_noise_ratio,
    structural_similarity,
)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class MinMaxScaler:
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.vmin = None
        self.vmax = None

    def scale(self, data):
        self.vmin = float(np.min(data))
        self.vmax = float(np.max(data))
        return (data - self.vmin) / (self.vmax - self.vmin + self.eps)

    def unscale(self, data):
        return data * (self.vmax - self.vmin + self.eps) + self.vmin


class TimeCalculator:
    def __init__(self):
        self.start_time = time.time()

    def period(self):
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed - 3600 * hours) // 60)
        seconds = elapsed - 3600 * hours - 60 * minutes
        return f"{hours}H {minutes}M {seconds:.2f}S"


def read_volume(raw_data_path):
    if not raw_data_path.endswith(".npy"):
        raise ValueError(f"Unsupported volume format: {raw_data_path}")
    volume = np.load(raw_data_path)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")
    raw_bytes = os.path.getsize(raw_data_path)
    return volume, raw_bytes


def normalize_volume(volume, mode):
    if mode in (None, "none"):
        metadata = {"mode": "none", "vmin": None, "vmax": None}
        return volume.astype(np.float32), metadata
    if mode != "minmax":
        raise ValueError(f"Unsupported normalization mode: {mode}")
    scaler = MinMaxScaler()
    scaled = scaler.scale(volume.astype(np.float32))
    metadata = {"mode": "minmax", "vmin": scaler.vmin, "vmax": scaler.vmax}
    return scaled.astype(np.float32), metadata


def scale_volume_for_training(volume, scale_max):
    scale_max = float(scale_max)
    if scale_max <= 0:
        raise ValueError("data.scale_max must be positive")
    return (volume.astype(np.float32) * scale_max).astype(np.float32)


def build_coordinate_grid(shape, coord_norm=1.0):
    if abs(float(coord_norm) - 1.0) > 1e-6:
        raise ValueError(
            "coord_norm must be 1.0 for this codec because the Gaussian model only "
            "supports the representable coordinate domain [-1, 1]."
        )
    h, w, d = shape
    xs = torch.linspace(-1.0, 1.0, h, dtype=torch.float32)
    ys = torch.linspace(-1.0, 1.0, w, dtype=torch.float32)
    zs = torch.linspace(-1.0, 1.0, d, dtype=torch.float32)
    mesh = torch.meshgrid(xs, ys, zs, indexing="ij")
    coords = torch.stack(mesh, dim=-1).reshape(-1, 3)
    return coords * float(coord_norm)


def flatten_volume(volume):
    return torch.from_numpy(volume.reshape(-1, 1).astype(np.float32))


def denormalize_from_training_scale(pred_training, normalization, scale_max):
    pred_unit = pred_training.astype(np.float32) / float(scale_max)
    if normalization["mode"] != "minmax":
        return pred_unit
    vmin = float(normalization["vmin"])
    vmax = float(normalization["vmax"])
    return pred_unit * (vmax - vmin + 1e-6) + vmin


def compute_reconstruction_metrics(gt_arr, pred_arr):
    gt = np.asarray(gt_arr, dtype=np.float32)
    pred = np.asarray(pred_arr, dtype=np.float32)
    pred = np.clip(pred, float(gt.min()), float(gt.max()))
    scaler = MinMaxScaler()
    gt_norm = scaler.scale(gt)
    pred_norm = (pred - scaler.vmin) / (scaler.vmax - scaler.vmin + scaler.eps)
    pred_norm = np.clip(pred_norm, 0.0, 1.0)

    metrics = {
        "MSE": float(mean_squared_error(gt_norm, pred_norm)),
        "MAE": float(np.mean(np.abs(gt_norm - pred_norm))),
        "NRMSE": float(normalized_root_mse(gt_norm, pred_norm)),
        "PSNR": float(peak_signal_noise_ratio(gt_norm, pred_norm, data_range=1.0)),
        "SSIM": float(structural_similarity(gt_norm, pred_norm, data_range=1.0)),
    }
    return metrics, gt_norm.astype(np.float32), pred_norm.astype(np.float32)


def estimate_bits_per_gaussian(config):
    quantized = bool(config.model.quantization_enabled)

    def effective_bits(bit_width):
        bit_width = int(bit_width)
        if quantized and bit_width > 0:
            return bit_width
        return 32

    center_bits = 3 * effective_bits(config.model.coordinate_bits)
    scale_bits = 3 * effective_bits(config.model.scale_bits)
    intensity_bits = effective_bits(config.model.intensity_bits)
    total_bits = center_bits + scale_bits + intensity_bits
    return {
        "center_bits": center_bits,
        "scale_bits": scale_bits,
        "intensity_bits": intensity_bits,
        "total_bits": total_bits,
    }


def estimate_quantizer_overhead_bits(config):
    return int(config.model.quantizer_overhead_bits)


def derive_num_gaussians(raw_bytes, config):
    if config.model.target_compression_ratio <= 0:
        raise ValueError("target_compression_ratio must be positive")
    target_bytes = raw_bytes / float(config.model.target_compression_ratio)
    target_bits = int(target_bytes * 8)
    bits = estimate_bits_per_gaussian(config)
    overhead_bits = estimate_quantizer_overhead_bits(config)
    remaining_bits = target_bits - overhead_bits
    if remaining_bits < bits["total_bits"]:
        raise ValueError(
            "Compression target is too small for one quantized Gaussian plus overhead"
        )
    derived = remaining_bits // bits["total_bits"]
    num_gaussians = int(config.model.num_gaussians) or int(derived)
    if num_gaussians <= 0:
        raise ValueError("Derived number of Gaussians must be positive")
    return {
        "raw_bytes": int(raw_bytes),
        "target_bytes": float(target_bytes),
        "target_bits": int(target_bits),
        "overhead_bits": int(overhead_bits),
        "bits_per_gaussian": int(bits["total_bits"]),
        "center_bits": int(bits["center_bits"]),
        "scale_bits": int(bits["scale_bits"]),
        "intensity_bits": int(bits["intensity_bits"]),
        "num_gaussians": num_gaussians,
    }


def _validate_map_volume_tensor(volume, name="volume"):
    if not torch.is_tensor(volume):
        raise TypeError(f"{name} must be a torch.Tensor")
    if volume.ndim not in (3, 4):
        raise ValueError(
            f"{name} must have shape [H, W, D] or [B, H, W, D], got {tuple(volume.shape)}"
        )
    if volume.shape[-1] <= 0:
        raise ValueError(f"{name} depth dimension must be positive")
    if not volume.is_floating_point():
        raise TypeError(f"{name} must use a floating point dtype")
    return volume


def _validate_map_image_tensor(map_image, name="map_image"):
    if not torch.is_tensor(map_image):
        raise TypeError(f"{name} must be a torch.Tensor")
    if map_image.ndim not in (2, 3):
        raise ValueError(
            f"{name} must have shape [H, W] or [B, H, W], got {tuple(map_image.shape)}"
        )
    if map_image.shape[-2] <= 0 or map_image.shape[-1] <= 0:
        raise ValueError(f"{name} spatial dimensions must be positive")
    if not map_image.is_floating_point():
        raise TypeError(f"{name} must use a floating point dtype")
    return map_image


def hard_map_projection(volume):
    volume = _validate_map_volume_tensor(volume, name="volume")
    return torch.amax(volume, dim=-1)


def soft_map_projection(volume, tau=1.0):
    volume = _validate_map_volume_tensor(volume, name="volume")
    tau = float(tau)
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    return tau * torch.logsumexp(volume / tau, dim=-1)


def topk_map_projection(volume, topk, tau=1.0):
    volume = _validate_map_volume_tensor(volume, name="volume")
    tau = float(tau)
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    topk = int(topk)
    depth = int(volume.shape[-1])
    if topk <= 0 or topk > depth:
        raise ValueError(f"topk must be in [1, {depth}], got {topk}")
    topk_vals, _ = torch.topk(volume, k=topk, dim=-1)
    return tau * torch.logsumexp(topk_vals / tau, dim=-1)


def project_map(volume, mode="hard", tau=1.0, topk=None):
    volume = _validate_map_volume_tensor(volume, name="volume")
    if mode == "hard":
        return hard_map_projection(volume)
    if mode == "soft":
        return soft_map_projection(volume, tau=tau)
    if mode == "topk":
        if topk is None:
            raise ValueError("topk projection requires a positive topk value")
        return topk_map_projection(volume, topk=topk, tau=tau)
    raise ValueError(f"Unsupported MAP projection mode: {mode}")


def map_projection_loss(
    pred_volume,
    target_volume,
    mode="hard",
    tau=1.0,
    topk=None,
    loss_type="l1",
):
    pred_map = project_map(pred_volume, mode=mode, tau=tau, topk=topk)
    target_map = project_map(target_volume, mode=mode, tau=tau, topk=topk)
    if pred_map.shape != target_map.shape:
        raise ValueError(
            f"MAP projections must have matching shapes, got {tuple(pred_map.shape)} "
            f"and {tuple(target_map.shape)}"
        )
    if loss_type == "l1":
        return F.l1_loss(pred_map, target_map)
    if loss_type == "mse":
        return F.mse_loss(pred_map, target_map)
    raise ValueError(f"Unsupported MAP projection loss type: {loss_type}")


def _compute_map_spatial_gradients(map_image, mode="sobel"):
    map_image = _validate_map_image_tensor(map_image, name="map_image")
    image = map_image.unsqueeze(0).unsqueeze(0) if map_image.ndim == 2 else map_image.unsqueeze(1)

    if mode == "finite_difference":
        grad_y = image[..., 1:, :] - image[..., :-1, :]
        grad_x = image[..., :, 1:] - image[..., :, :-1]
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
    elif mode == "sobel":
        kernel_x = image.new_tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
        ).unsqueeze(0)
        kernel_y = image.new_tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
        ).unsqueeze(0)
        grad_x = F.conv2d(image, kernel_x, padding=1)
        grad_y = F.conv2d(image, kernel_y, padding=1)
    else:
        raise ValueError(f"Unsupported MAP gradient mode: {mode}")

    grad_x = grad_x.squeeze(1)
    grad_y = grad_y.squeeze(1)
    if map_image.ndim == 2:
        grad_x = grad_x.squeeze(0)
        grad_y = grad_y.squeeze(0)
    return grad_x, grad_y


def map_gradient_consistency_loss(
    pred_volume,
    target_volume,
    projection_mode="hard",
    tau=1.0,
    topk=None,
    gradient_mode="sobel",
    loss_type="l1",
):
    pred_map = project_map(pred_volume, mode=projection_mode, tau=tau, topk=topk)
    target_map = project_map(target_volume, mode=projection_mode, tau=tau, topk=topk)
    if pred_map.shape != target_map.shape:
        raise ValueError(
            f"MAP projections must have matching shapes, got {tuple(pred_map.shape)} "
            f"and {tuple(target_map.shape)}"
        )

    pred_grad_x, pred_grad_y = _compute_map_spatial_gradients(pred_map, mode=gradient_mode)
    target_grad_x, target_grad_y = _compute_map_spatial_gradients(
        target_map, mode=gradient_mode
    )

    if loss_type == "l1":
        loss_fn = F.l1_loss
    elif loss_type == "mse":
        loss_fn = F.mse_loss
    else:
        raise ValueError(f"Unsupported MAP gradient loss type: {loss_type}")

    return loss_fn(pred_grad_x, target_grad_x) + loss_fn(pred_grad_y, target_grad_y)
