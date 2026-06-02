# MAP Loss Experiment Note

**Date:** 2026-05-26

## Goal

Improve PAM MAP metrics without materially degrading 3D raw-volume metrics for the 3D Gaussian compression model.

## First Ablation Matrix

### Baseline

- 3D reconstruction loss only
- current reference for:
  - 3D PSNR / SSIM / MSE / MAE / NRMSE
  - MAP PSNR / SSIM / MSE / MAE / NRMSE

### Variant A: `+ soft-MAP`

- total loss:
  - `L_total = L_3d + lambda_map * L_soft_map`
- purpose:
  - improve projection fidelity with smoother gradients than hard max

### Variant B: `+ soft-MAP + MAP gradient`

- total loss:
  - `L_total = L_3d + lambda_map * L_soft_map + lambda_grad * L_map_grad`
- purpose:
  - improve MAP edge continuity and peak structure

### Variant C: `+ top-k MAP`

- replace soft-MAP branch with top-k projection surrogate
- purpose:
  - distribute gradient to several strong depth responses instead of a single argmax

## Initial Hyperparameters

- `map_loss_enable = True`
- `map_loss_start_step = 200`
- `map_loss_weight = 0.05`
- `map_grad_loss_weight = 0.01`
- `map_softmax_tau = 10.0`
- `map_topk = 4`

## Success Criteria

- MAP PSNR and SSIM improve over baseline
- 3D PSNR and SSIM do not regress materially
- 3D MSE / MAE / NRMSE do not worsen beyond acceptable tolerance
- qualitative MAP images preserve sharper vessel or high-intensity structures

## Failure Modes To Watch

- MAP improves but 3D fidelity drops
- exaggerated peaks that game MAP while harming the full volume
- unstable optimization when MAP loss is enabled too early or too strongly

## Run Order

1. Baseline 3D-only reference
2. `+ soft-MAP`
3. `+ soft-MAP + MAP gradient`
4. optional top-k surrogate
