from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    rounded = torch.round(x)
    return x + (rounded - x).detach()


@dataclass(frozen=True)
class QuantizerStats:
    encoding: str
    bit_width: int
    num_values: int
    total_bits: int
    quantized: bool
    min_value: float
    max_value: float


class UniformFakeQuantizer(nn.Module):
    """Uniform fake quantizer with a straight-through estimator."""

    def __init__(
        self,
        bit_width: int,
        min_value: float,
        max_value: float,
        enabled: bool = True,
        fallback_bit_width: int = 32,
    ) -> None:
        super().__init__()
        if bit_width < 0:
            raise ValueError("bit_width must be non-negative.")
        if max_value <= min_value:
            raise ValueError("max_value must be larger than min_value.")

        self.bit_width = int(bit_width)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.enabled = bool(enabled)
        self.fallback_bit_width = int(fallback_bit_width)

    @property
    def is_active(self) -> bool:
        return self.enabled and self.bit_width > 0

    @property
    def active_bit_width(self) -> int:
        return self.bit_width if self.is_active else self.fallback_bit_width

    @property
    def encoding(self) -> str:
        return "uniform_quantized" if self.is_active else "float32"

    def _quantize_impl(self, x: torch.Tensor, use_ste: bool) -> torch.Tensor:
        if not self.is_active:
            return x

        x_clamped = x.clamp(self.min_value, self.max_value)
        num_levels = (1 << self.bit_width) - 1
        if num_levels <= 0:
            return x_clamped

        scale = (self.max_value - self.min_value) / num_levels
        zero_based = (x_clamped - self.min_value) / scale
        quantized = _ste_round(zero_based) if use_ste else torch.round(zero_based)
        dequantized = quantized * scale + self.min_value
        return dequantized

    def _codes_impl(self, x: torch.Tensor) -> torch.Tensor:
        x_clamped = x.clamp(self.min_value, self.max_value)
        num_levels = (1 << self.active_bit_width) - 1
        if num_levels <= 0:
            return torch.zeros_like(x_clamped, dtype=torch.int64)

        # Use float64 here so large fallback bit widths still produce stable export codes.
        scale = (self.max_value - self.min_value) / float(num_levels)
        zero_based = (x_clamped.to(torch.float64) - self.min_value) / scale
        return torch.round(zero_based).to(torch.int64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantize_impl(x, use_ste=True)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantize_impl(x, use_ste=False)

    def codes(self, x: torch.Tensor) -> torch.Tensor:
        return self._codes_impl(x)

    def stats(self, x: torch.Tensor) -> QuantizerStats:
        num_values = int(x.numel())
        return QuantizerStats(
            encoding=self.encoding,
            bit_width=self.active_bit_width,
            num_values=num_values,
            total_bits=num_values * self.active_bit_width,
            quantized=self.is_active,
            min_value=self.min_value,
            max_value=self.max_value,
        )
