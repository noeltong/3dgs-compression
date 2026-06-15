import random
import time
import os

import numpy as np
import torch
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
