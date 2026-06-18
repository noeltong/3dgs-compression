# Strict Reallocation V1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a deterministic fixed-budget prune-and-regrow training schedule for the 3D Gaussian codec without changing compression accounting or evaluation semantics.

**Architecture:** Keep the parameter count fixed by overwriting selected Gaussian rows in-place. Put inverse-parameterization and row-overwrite helpers in `models/model.py`, and orchestrate trigger/score/residual selection/state reset/logging in `run_lib.py` using full training-volume residuals with chunked inference.

**Tech Stack:** Python, PyTorch, ml_collections configs

---

### Task 1: Add model-side row overwrite helpers

**Files:**
- Modify: `models/model.py`

**Step 1:** Expose activated parameters via a simple public helper.

**Step 2:** Add inverse parameterization helpers for centers, scales, and intensities that match current tanh/softplus parameterization.

**Step 3:** Add an in-place row overwrite helper for `center_logits`, `raw_scales`, and `intensity_logits` only.

### Task 2: Implement strict training-time reallocation

**Files:**
- Modify: `run_lib.py`

**Step 1:** Add helpers for the trigger condition, prune-score computation, chunked full-volume prediction, top-k residual selection, and row-wise optimizer-state reset.

**Step 2:** Run reallocation only on the exact configured schedule, using `k = floor(realloc_fraction * N)` clamped to `[1, N - 1]`.

**Step 3:** Prune lowest `abs(intensity) * scale_x * scale_y * scale_z` scores, respawn at top-k residual coordinates, overwrite rows in-place, and zero optimizer state only for affected rows.

**Step 4:** Emit `realloc_*` logs every step, with zero values on non-event steps.

### Task 3: Add config knobs and smoke coverage

**Files:**
- Modify: `configs/default_config.py`
- Modify: `configs/gaussian3d_smoke.py`

**Step 1:** Add the five requested default training realloc fields.

**Step 2:** Enable aggressive smoke overrides so the 2-step run exercises the path.

### Task 4: Validate invariants

**Files:**
- Validate only

**Step 1:** Run `compileall` for syntax checking.

**Step 2:** Run the smoke train command and confirm realloc events happen.

**Step 3:** Optionally run smoke eval and confirm MAP remains eval-only.

**Step 4:** Verify fixed Gaussian count, reconstruction-only training loss, and row-local optimizer reset behavior from code and logs.
