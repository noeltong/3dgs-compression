from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from models.quantization import UniformFakeQuantizer


def _get_config_value(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        if name in config:
            return config.get(name, default)
        nested_model = config.get("model")
        if nested_model is not None:
            return _get_config_value(nested_model, name, default)
        return default
    if hasattr(config, name):
        return getattr(config, name)
    nested_model = getattr(config, "model", None)
    if nested_model is not None:
        return _get_config_value(nested_model, name, default)
    return default


def _inverse_softplus(x: float) -> float:
    if x <= 0:
        raise ValueError("inverse softplus is only defined for positive values.")
    return math.log(math.expm1(x))


class GaussianVolumeCodec(nn.Module):
    """Axis-aligned 3D Gaussian codec for single-volume reconstruction."""

    def __init__(self, config: Optional[Any] = None, **overrides: Any) -> None:
        super().__init__()

        self.num_gaussians = int(overrides.get("num_gaussians", _get_config_value(config, "num_gaussians", 128)))
        if self.num_gaussians <= 0:
            raise ValueError("num_gaussians must be positive.")

        self.coordinate_bits = int(
            overrides.get("coordinate_bits", _get_config_value(config, "coordinate_bits", 12))
        )
        self.scale_bits = int(overrides.get("scale_bits", _get_config_value(config, "scale_bits", 10)))
        self.intensity_bits = int(
            overrides.get("intensity_bits", _get_config_value(config, "intensity_bits", 12))
        )
        self.quantization_enabled = bool(
            overrides.get("quantization_enabled", _get_config_value(config, "quantization_enabled", True))
        )
        self.quantizer_overhead_bits = int(
            overrides.get(
                "quantizer_overhead_bits",
                _get_config_value(config, "quantizer_overhead_bits", 0),
            )
        )
        self.min_scale = float(overrides.get("min_scale", _get_config_value(config, "min_scale", 1e-3)))
        self.max_scale = float(overrides.get("max_scale", _get_config_value(config, "max_scale", 2.0)))
        self.intensity_range = float(
            overrides.get("intensity_range", _get_config_value(config, "intensity_range", 1.0))
        )
        self.dense_chunk_size = int(
            overrides.get("dense_chunk_size", _get_config_value(config, "dense_chunk_size", 65536))
        )
        self.forward_query_chunk_size = int(
            overrides.get("forward_query_chunk_size", _get_config_value(config, "forward_query_chunk_size", 8192))
        )
        self.forward_gaussian_chunk_size = int(
            overrides.get(
                "forward_gaussian_chunk_size",
                _get_config_value(config, "forward_gaussian_chunk_size", 4096),
            )
        )
        self.eps = float(overrides.get("eps", _get_config_value(config, "eps", 1e-8)))

        if self.min_scale <= 0:
            raise ValueError("min_scale must be positive.")
        if self.max_scale <= self.min_scale:
            raise ValueError("max_scale must be larger than min_scale.")
        if self.intensity_range <= 0:
            raise ValueError("intensity_range must be positive.")
        if self.dense_chunk_size <= 0:
            raise ValueError("dense_chunk_size must be positive.")
        if self.forward_query_chunk_size <= 0:
            raise ValueError("forward_query_chunk_size must be positive.")
        if self.forward_gaussian_chunk_size <= 0:
            raise ValueError("forward_gaussian_chunk_size must be positive.")
        if self.quantizer_overhead_bits < 0:
            raise ValueError("quantizer_overhead_bits must be non-negative.")

        init_scale = float(overrides.get("init_scale", _get_config_value(config, "init_scale", 0.15)))
        init_scale = min(max(init_scale, self.min_scale), self.max_scale)
        scale_offset = max(init_scale - self.min_scale, self.eps)
        raw_scale_init = _inverse_softplus(scale_offset)

        center_init = torch.empty(self.num_gaussians, 3).uniform_(-0.5, 0.5)
        raw_scale_tensor = torch.full((self.num_gaussians, 3), raw_scale_init)
        intensity_init = torch.zeros(self.num_gaussians, 1)

        self.center_logits = nn.Parameter(center_init)
        self.raw_scales = nn.Parameter(raw_scale_tensor)
        self.intensity_logits = nn.Parameter(intensity_init)

        self.center_quantizer = UniformFakeQuantizer(
            bit_width=self.coordinate_bits,
            min_value=-1.0,
            max_value=1.0,
            enabled=self.quantization_enabled,
        )
        self.scale_quantizer = UniformFakeQuantizer(
            bit_width=self.scale_bits,
            min_value=self.min_scale,
            max_value=self.max_scale,
            enabled=self.quantization_enabled,
        )
        self.intensity_quantizer = UniformFakeQuantizer(
            bit_width=self.intensity_bits,
            min_value=-self.intensity_range,
            max_value=self.intensity_range,
            enabled=self.quantization_enabled,
        )

    def _activated_parameters(self) -> Dict[str, torch.Tensor]:
        centers = torch.tanh(self.center_logits)
        scales = self.min_scale + F.softplus(self.raw_scales)
        scales = scales.clamp(max=self.max_scale)
        intensities = torch.tanh(self.intensity_logits) * self.intensity_range
        return {
            "centers": centers,
            "scales": scales,
            "intensities": intensities,
        }

    def _quantized_parameters(self, use_ste: bool) -> Dict[str, torch.Tensor]:
        params = self._activated_parameters()
        quantize = (lambda q, x: q(x)) if use_ste else (lambda q, x: q.quantize(x))
        return {
            "centers": quantize(self.center_quantizer, params["centers"]),
            "scales": quantize(self.scale_quantizer, params["scales"]),
            "intensities": quantize(self.intensity_quantizer, params["intensities"]),
        }

    def get_parameter_tensors(self, quantized: bool = False, use_ste: bool = True) -> Dict[str, torch.Tensor]:
        if quantized:
            return self._quantized_parameters(use_ste=use_ste)
        return self._activated_parameters()

    def get_payload_stats(self) -> Dict[str, Any]:
        params = self._activated_parameters()
        center_stats = self.center_quantizer.stats(params["centers"])
        scale_stats = self.scale_quantizer.stats(params["scales"])
        intensity_stats = self.intensity_quantizer.stats(params["intensities"])
        total_bits = center_stats.total_bits + scale_stats.total_bits + intensity_stats.total_bits
        total_bits_with_overhead = total_bits + self.quantizer_overhead_bits
        return {
            "num_gaussians": self.num_gaussians,
            "quantization_enabled": self.quantization_enabled,
            "bits_per_gaussian": float(total_bits) / float(self.num_gaussians),
            "centers": {
                "encoding": center_stats.encoding,
                "bit_width": center_stats.bit_width,
                "num_values": center_stats.num_values,
                "total_bits": center_stats.total_bits,
                "quantized": center_stats.quantized,
                "min_value": center_stats.min_value,
                "max_value": center_stats.max_value,
            },
            "scales": {
                "encoding": scale_stats.encoding,
                "bit_width": scale_stats.bit_width,
                "num_values": scale_stats.num_values,
                "total_bits": scale_stats.total_bits,
                "quantized": scale_stats.quantized,
                "min_value": scale_stats.min_value,
                "max_value": scale_stats.max_value,
            },
            "intensities": {
                "encoding": intensity_stats.encoding,
                "bit_width": intensity_stats.bit_width,
                "num_values": intensity_stats.num_values,
                "total_bits": intensity_stats.total_bits,
                "quantized": intensity_stats.quantized,
                "min_value": intensity_stats.min_value,
                "max_value": intensity_stats.max_value,
            },
            "total_bits": total_bits,
            "overhead_bits": self.quantizer_overhead_bits,
            "total_bits_with_overhead": total_bits_with_overhead,
        }

    def _export_payload_group(
        self,
        name: str,
        tensor: torch.Tensor,
        quantizer: UniformFakeQuantizer,
    ) -> Dict[str, Any]:
        stats = quantizer.stats(tensor)
        payload_tensor: torch.Tensor
        payload_key: str
        if stats.quantized:
            payload_tensor = quantizer.codes(tensor).detach().cpu()
            payload_key = "codes"
        else:
            payload_tensor = tensor.detach().to(dtype=torch.float32).cpu()
            payload_key = "values"

        return {
            "name": name,
            "encoding": stats.encoding,
            "quantized": stats.quantized,
            "bit_width": stats.bit_width,
            payload_key: payload_tensor,
            "num_values": stats.num_values,
            "total_bits": stats.total_bits,
            "value_range": (stats.min_value, stats.max_value),
        }

    def get_quantized_payload(self) -> Dict[str, Any]:
        params = self._activated_parameters()
        center_payload = self._export_payload_group("centers", params["centers"], self.center_quantizer)
        scale_payload = self._export_payload_group("scales", params["scales"], self.scale_quantizer)
        intensity_payload = self._export_payload_group(
            "intensities", params["intensities"], self.intensity_quantizer
        )
        return {
            "centers": center_payload,
            "scales": scale_payload,
            "intensities": intensity_payload,
            "stats": self.get_payload_stats(),
        }

    def forward(self, query_coordinates: torch.Tensor) -> torch.Tensor:
        if query_coordinates.shape[-1] != 3:
            raise ValueError("query_coordinates must have shape [N, 3] or [..., 3].")

        original_shape = query_coordinates.shape[:-1]
        params = self._quantized_parameters(use_ste=self.training)
        centers = params["centers"]
        scales = params["scales"]
        intensities = params["intensities"]
        flat_coords = query_coordinates.reshape(-1, 3).to(device=centers.device, dtype=centers.dtype)
        if torch.any(flat_coords < -1.0) or torch.any(flat_coords > 1.0):
            raise ValueError("query_coordinates must lie within the representable range [-1, 1].")

        reconstruction = flat_coords.new_zeros(flat_coords.shape[0], 1)
        for query_start in range(0, flat_coords.shape[0], self.forward_query_chunk_size):
            query_stop = min(query_start + self.forward_query_chunk_size, flat_coords.shape[0])
            coord_chunk = flat_coords[query_start:query_stop]
            chunk_output = coord_chunk.new_zeros(coord_chunk.shape[0], 1)

            for gaussian_start in range(0, self.num_gaussians, self.forward_gaussian_chunk_size):
                gaussian_stop = min(gaussian_start + self.forward_gaussian_chunk_size, self.num_gaussians)
                center_chunk = centers[gaussian_start:gaussian_stop]
                scale_chunk = scales[gaussian_start:gaussian_stop]
                intensity_chunk = intensities[gaussian_start:gaussian_stop]

                deltas = coord_chunk[:, None, :] - center_chunk[None, :, :]
                scaled = deltas / (scale_chunk[None, :, :] + self.eps)
                exponents = -0.5 * scaled.square().sum(dim=-1)
                weights = torch.exp(exponents)
                chunk_output = chunk_output + weights @ intensity_chunk

            reconstruction[query_start:query_stop] = chunk_output
        return reconstruction.reshape(*original_shape, 1)

    def reconstruct_dense(
        self,
        coordinate_tensor: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        if coordinate_tensor.shape[-1] != 3:
            raise ValueError("coordinate_tensor must have shape [..., 3].")

        flat_coords = coordinate_tensor.reshape(-1, 3)
        chunk_size = self.dense_chunk_size if chunk_size is None else int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if flat_coords.shape[0] == 0:
            return flat_coords.new_empty(*coordinate_tensor.shape[:-1], 1)
        outputs = []
        for start in range(0, flat_coords.shape[0], chunk_size):
            stop = min(start + chunk_size, flat_coords.shape[0])
            outputs.append(self.forward(flat_coords[start:stop]))
        reconstructed = torch.cat(outputs, dim=0)
        return reconstructed.reshape(*coordinate_tensor.shape[:-1], 1)
