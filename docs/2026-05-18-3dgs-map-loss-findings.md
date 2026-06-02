# 3DGS MAP Improvement Findings

**Date:** 2026-05-18  
**Question:** The current 3D Gaussian compression model reconstructs 3D PAM volumes well, but MAP quality is poor. What literature-backed training changes are most promising for improving both raw-volume and MAP metrics?

## Executive Summary

The literature suggests that the current failure mode is expected: optimizing only voxel-wise 3D reconstruction does not sufficiently emphasize the sparse, high-intensity structures that dominate MAP. The strongest next step is to keep the current 3D reconstruction loss as the anchor objective and add a differentiable MAP-aware auxiliary loss with smoother gradients than a hard max projection.

## Relevant Literature

### 1. 3DGS compression increasingly optimizes rate-distortion, not pure reconstruction

- **GETA-3DGS** argues that pruning and mixed-precision quantization must be optimized jointly to improve rate-distortion rather than by heuristic stages.  
  Link: https://arxiv.org/abs/2605.02086
- **MesonGS++** shows post-training 3DGS compression is highly sensitive to coupled hyperparameters and that better rate-distortion control matters.  
  Link: https://arxiv.org/abs/2604.26799
- **GeoHCC** emphasizes preserving local geometric structure during compression because naive compression harms structural fidelity.  
  Link: https://arxiv.org/abs/2603.28431

**Relevance here:** MAP quality is effectively a structure-sensitive target. If the training objective only rewards average voxel reconstruction, the codec can preserve bulk fidelity while underfitting projection-critical peaks.

### 2. Projection-aware losses are an established way to improve projection metrics

- **SPOCKMIP** uses maximum intensity projection as an auxiliary loss to improve vessel continuity in 3D medical segmentation.  
  Link: https://arxiv.org/abs/2407.08655
- **Top-K Maximum Intensity Projection Priors** replaces a hard MIP with top-k projection priors to preserve more informative gradient flow along projection rays.  
  Link: https://arxiv.org/abs/2503.03367

**Relevance here:** A hard MAP loss is aligned with the target metric, but it is gradient-sparse. A soft or top-k MAP surrogate is more trainable.

### 3. 3DGS benefits from auxiliary geometric supervision without replacing the main reconstruction objective

- **DRGSplat** improves 3DGS geometry using auxiliary depth, normal, and curvature losses while preserving the main rendering objective.  
  Link: https://openreview.net/forum?id=BpwRgbmTW9

**Relevance here:** This is a strong analogy for adding MAP-aware supervision: keep the base reconstruction loss, add structured auxiliary losses, and tune the weight schedule carefully.

## Ranked Ideas

### 1. Recommended: Soft-MAP auxiliary loss

Use:

`L = L_3d + lambda_map * L_soft_map`

where:

- `L_3d` is the current voxel reconstruction loss
- `soft_map(z)` is a smooth max surrogate, e.g. `logsumexp(tau * x, dim=depth) / tau`

**Why it is strong:** It targets the exact failure mode while avoiding the sparse gradients of hard max.

### 2. Recommended: Soft-MAP + MAP gradient consistency

Use:

`L = L_3d + lambda_map * L_soft_map + lambda_grad * L_map_grad`

where `L_map_grad` matches horizontal and vertical gradients on the MAP image.

**Why it is strong:** MAP quality is often degraded by broken vessel edges and peak mislocalization rather than pure intensity drift.

### 3. Backup: Top-k MAP loss

Replace pure max with a top-k projection loss along depth.

**Why it is strong:** It more directly follows the top-k MIP literature and gives gradient to several strong responses instead of only one.

## Recommended First Experiment

Start with:

- `L_3d = MSE(pred_volume, gt_volume)`
- `L_soft_map = MSE(soft_map(pred), map(gt))`
- `L_map_grad = MSE(grad(soft_map(pred)), grad(map(gt)))`

Initial weights:

- `lambda_map = 0.05`
- `lambda_grad = 0.01`

Schedule:

- warm up with `L_3d` only
- enable MAP losses after early stabilization

## Main Risk

If `lambda_map` is too large, the model may learn to game MAP by overstating peaks while harming 3D fidelity. The experiment must therefore compare:

- 3D metrics
- MAP metrics
- qualitative MAP images
- compressed bitrate consistency

## Recommendation

Proceed with **soft-MAP + MAP-gradient auxiliary supervision** as the primary next idea. It is the best combination of literature support, implementation feasibility in the current codebase, and likely improvement to both MAP and 3D fidelity.
