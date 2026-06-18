from __future__ import annotations

from typing import Dict

import torch
from torch import nn


def _validate_patch_tensor(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim != 5:
        raise ValueError(f"{name} must have shape [B, C, H, W, D].")


def hyperbola_phi(residual: torch.Tensor, delta: float) -> torch.Tensor:
    delta_tensor = torch.as_tensor(delta, dtype=residual.dtype, device=residual.device)
    if torch.any(delta_tensor <= 0):
        raise ValueError("delta must be positive.")
    scaled = residual / delta_tensor
    return delta_tensor.square() * (torch.sqrt(1.0 + scaled.square()) - 1.0)


def diff_x(tensor: torch.Tensor) -> torch.Tensor:
    _validate_patch_tensor(tensor, "tensor")
    return tensor[:, :, 1:, :, :] - tensor[:, :, :-1, :, :]


def diff_y(tensor: torch.Tensor) -> torch.Tensor:
    _validate_patch_tensor(tensor, "tensor")
    return tensor[:, :, :, 1:, :] - tensor[:, :, :, :-1, :]


def diff_z(tensor: torch.Tensor) -> torch.Tensor:
    _validate_patch_tensor(tensor, "tensor")
    return tensor[:, :, :, :, 1:] - tensor[:, :, :, :, :-1]


class HyperbolaGradientPatchLoss(nn.Module):
    def __init__(
        self,
        lambda_grad: float = 0.1,
        alpha_x: float = 1.0,
        alpha_y: float = 1.0,
        alpha_z: float = 1.0,
        delta: float = 1.0,
    ) -> None:
        super().__init__()
        if lambda_grad < 0:
            raise ValueError("lambda_grad must be non-negative.")
        if alpha_x < 0 or alpha_y < 0 or alpha_z < 0:
            raise ValueError("Gradient axis weights must be non-negative.")
        if delta <= 0:
            raise ValueError("delta must be positive.")
        self.lambda_grad = float(lambda_grad)
        self.alpha_x = float(alpha_x)
        self.alpha_y = float(alpha_y)
        self.alpha_z = float(alpha_z)
        self.delta = float(delta)

    def mse_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        self._validate_pair(pred, gt)
        return torch.mean((pred - gt).square())

    def gradient_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        self._validate_pair(pred, gt)
        # TODO: Consider an optional 13-offset 3D neighborhood variant here.
        # The current loss uses only the 3 forward axis differences, which is
        # the standard first-order discrete gradient. A future extension could
        # evaluate the robust phi penalty over the 13 unique directed offsets
        # that represent the full 26-neighbor voxel neighborhood without
        # double-counting.
        grad_x = torch.mean(hyperbola_phi(diff_x(pred) - diff_x(gt), self.delta))
        grad_y = torch.mean(hyperbola_phi(diff_y(pred) - diff_y(gt), self.delta))
        grad_z = torch.mean(hyperbola_phi(diff_z(pred) - diff_z(gt), self.delta))
        return (
            self.alpha_x * grad_x
            + self.alpha_y * grad_y
            + self.alpha_z * grad_z
        )

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        loss_mse = self.mse_loss(pred, gt)
        loss_grad = self.gradient_loss(pred, gt)
        loss_total = loss_mse + self.lambda_grad * loss_grad
        return {
            "loss_total": loss_total,
            "loss_mse": loss_mse,
            "loss_grad": loss_grad,
        }

    @staticmethod
    def _validate_pair(pred: torch.Tensor, gt: torch.Tensor) -> None:
        _validate_patch_tensor(pred, "pred")
        _validate_patch_tensor(gt, "gt")
        if pred.shape != gt.shape:
            raise ValueError(f"pred and gt must match exactly, got {pred.shape} and {gt.shape}.")
