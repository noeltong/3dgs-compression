# MAP-Aware 3DGS Loss Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a MAP-aware auxiliary loss to the 3D Gaussian PAM codec so that MAP metrics improve without sacrificing volumetric reconstruction quality.

**Architecture:** Keep the current voxel reconstruction loss as the base objective and add a differentiable projection branch that computes a soft MAP image from the reconstructed volume. Train with a weighted sum of 3D reconstruction loss, soft-MAP loss, and MAP gradient consistency loss, with a configurable warmup and loss schedule.

**Tech Stack:** Python, PyTorch, NumPy, existing training/eval pipeline in `run_lib.py`

---

### Task 1: Extend config for MAP-aware supervision

**Files:**
- Modify: `configs/default_config.py`
- Modify: `configs/gaussian3d.py`

**Step 1: Add config keys**

Add fields for:

- `training.map_loss_enable`
- `training.map_loss_start_step`
- `training.map_loss_type`
- `training.map_loss_weight`
- `training.map_grad_loss_weight`
- `training.map_softmax_tau`
- `training.map_topk`

**Step 2: Set conservative defaults**

Defaults should keep the current behavior unless explicitly enabled.

**Step 3: Commit**

```bash
git add configs/default_config.py configs/gaussian3d.py
git commit -m "feat: add map-aware loss config"
```

### Task 2: Add reusable MAP loss helpers

**Files:**
- Modify: `utils/utils.py`

**Step 1: Implement soft MAP projection**

Add a helper that computes:

- hard MAP via `amax`
- soft MAP via `logsumexp`
- optional top-k MAP surrogate

**Step 2: Implement MAP gradient loss helper**

Add Sobel-based or finite-difference gradient matching for 2D MAP images.

**Step 3: Add small shape/value validation**

Fail fast on unsupported projection modes or bad tensor ranks.

**Step 4: Commit**

```bash
git add utils/utils.py
git commit -m "feat: add map projection loss helpers"
```

### Task 3: Integrate auxiliary losses into training

**Files:**
- Modify: `run_lib.py`

**Step 1: Keep current 3D reconstruction loss unchanged**

Preserve the existing 3D data-fidelity path as `L_3d`.

**Step 2: Add MAP-aware loss computation**

When enabled and after warmup:

- reconstruct the predicted volume batch
- compute predicted MAP surrogate
- compute GT MAP target
- compute `L_soft_map`
- compute `L_map_grad`

**Step 3: Form the total loss**

Use:

`L_total = L_3d + lambda_map * L_soft_map + lambda_grad * L_map_grad`

**Step 4: Log all components**

Add `logkvs()` entries for:

- `loss_3d`
- `loss_map`
- `loss_map_grad`
- `loss_total`
- `map_loss_active`

**Step 5: Commit**

```bash
git add run_lib.py
git commit -m "feat: add map-aware training losses"
```

### Task 4: Add a dedicated experiment override

**Files:**
- Create: `configs/gaussian3d_pam_map.py`

**Step 1: Create a runnable MAP-loss config**

Set:

- MAP loss enabled
- nonzero `map_loss_weight`
- nonzero `map_grad_loss_weight`
- delayed `map_loss_start_step`

**Step 2: Commit**

```bash
git add configs/gaussian3d_pam_map.py
git commit -m "feat: add map-aware pam config"
```

### Task 5: Verify no regression in eval/export

**Files:**
- Modify: `run_lib.py` only if needed

**Step 1: Ensure eval still writes**

- `pred_volume.npy`
- `pred_map.npy`
- `metrics.json`

**Step 2: Run smoke training**

Run:

```bash
conda run -n ip python main.py --mode=train --config=configs/gaussian3d_smoke.py --workdir=map_loss_smoke
```

Expected:

- training completes
- post-train eval still saves artifacts
- no shape/rank errors in MAP loss path when disabled

**Step 3: Run MAP-enabled smoke**

Run:

```bash
conda run -n ip python main.py --mode=train --config=configs/gaussian3d_pam_map.py --workdir=map_loss_enabled
```

Expected:

- MAP loss terms appear in logs
- training remains stable

**Step 4: Commit**

```bash
git add run_lib.py
git commit -m "test: verify map-aware loss training path"
```

### Task 6: Write experiment note

**Files:**
- Create: `docs/2026-05-18-map-loss-experiment-note.md`

**Step 1: Document the first ablation matrix**

Include:

- baseline 3D-only
- `+ soft-MAP`
- `+ soft-MAP + grad`
- optional top-k variant

**Step 2: Define success criteria**

- MAP PSNR/SSIM improve
- 3D PSNR/SSIM do not regress materially

**Step 3: Commit**

```bash
git add docs/2026-05-18-map-loss-experiment-note.md
git commit -m "docs: add map-loss experiment note"
```
