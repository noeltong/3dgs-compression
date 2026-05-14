# 3D Gaussian PAM/PAT Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a single-volume 3D Gaussian codec for PAM/PAT raw volumes with compression-ratio-driven Gaussian count, QAT-style parameter quantization, and single-GPU reconstruction-only training.

**Architecture:** The pipeline loads one `[H, W, D]` volume, normalizes intensities and coordinates, derives a Gaussian budget from raw file size and target compression ratio, then fits a set of axis-aligned 3D Gaussians whose quantized parameters define the codec size. The training loop is rewritten to use the repository logger and to remove all DDP, AMP, and EMA assumptions from the template.

**Tech Stack:** Python, PyTorch, ml_collections, NumPy, existing `utils/logger.py`

---

### Task 1: Define the new config surface

**Files:**
- Modify: `configs/default_config.py`
- Create: `configs/gaussian3d.py`

**Step 1: Add 3D Gaussian config sections**

Add fields for:

- `model.name = "gaussian3d"`
- `model.target_compression_ratio`
- `model.num_gaussians = 0`
- `model.truncation_sigma`
- `model.init_scale_bias`
- `model.coordinate_bits`
- `model.scale_bits`
- `model.intensity_bits`
- `training.max_steps`
- `training.log_freq`
- `training.eval_freq`
- `training.ckpt_freq`
- `data.path`
- `data.normalize`
- `data.coord_norm`

**Step 2: Remove template-only DDP/EMA config fields**

Delete or stop using:

- `cfg.distributed`
- `model.ema`
- `model.ema_rate`
- `model.ema_steps`

**Step 3: Create the first override config**

Create `configs/gaussian3d.py` that imports the default config and sets the method-specific values for the first experiment.

### Task 2: Build volume IO and sampling utilities

**Files:**
- Modify: `utils/data.py`
- Modify: `utils/utils.py`

**Step 1: Add a volume reader helper**

Implement a helper that:

- loads `.npy` volume data
- returns the volume array
- returns raw file bytes
- records the original shape and dtype

**Step 2: Add intensity normalization**

Implement normalization logic compatible with the existing PAM conventions:

- no normalization
- min-max normalization to `[0, 1]`

Return the normalized volume and normalization metadata.

**Step 3: Add coordinate generation**

Implement helpers to:

- generate normalized voxel-center coordinates in `[-1, 1]^3`
- flatten volume intensities and coordinates for optimization

**Step 4: Add a dataset or sampler for a single volume**

Implement the simplest useful training input path:

- full flattened coordinate-intensity tensors for evaluation
- random coordinate batches for training

### Task 3: Add bitrate accounting helpers

**Files:**
- Modify: `utils/utils.py`

**Step 1: Add per-Gaussian bit estimator**

Implement a helper that computes:

- bits for center
- bits for scales
- bits for intensity
- total bits per Gaussian

**Step 2: Add global-overhead estimator**

Implement a helper that computes coarse metadata overhead for the active quantizers.

**Step 3: Add Gaussian-count derivation**

Implement a helper that computes:

- `target_bytes`
- `target_bits`
- `num_gaussians`

Clamp to at least `1` Gaussian and log all derived values.

### Task 4: Implement parameter fake quantization

**Files:**
- Modify: `models/model.py`
- Optionally create: `models/quantization.py`

**Step 1: Add a reusable STE fake-quant function**

Implement a straight-through fake quantizer for uniform quantization with configurable bit width and value range.

**Step 2: Split quantization by parameter group**

Implement separate quantization calls for:

- centers
- scales
- intensities

**Step 3: Add export-ready quantized payload stats**

Return the quantized tensors and estimated payload bits from the model or helper functions so the training loop can report them.

### Task 5: Implement the 3D Gaussian model

**Files:**
- Modify: `models/model.py`

**Step 1: Replace the template model placeholder with a 3D Gaussian codec model**

Add model parameters for:

- learnable centers
- learnable raw scales
- learnable intensities

**Step 2: Add stable parameter activations**

Use bounded or positive parameterizations:

- center parameters mapped into `[-1, 1]`
- scales mapped to positive values

**Step 3: Implement local-support Gaussian evaluation**

Add a forward path that:

- quantizes parameters when QAT is enabled
- evaluates Gaussian contributions at query coordinates
- returns reconstructed intensities

**Step 4: Add dense-volume reconstruction**

Implement a helper to reconstruct the full volume for evaluation from all voxel coordinates.

### Task 6: Rewrite the training loop

**Files:**
- Modify: `run_lib.py`
- Modify: `main.py`

**Step 1: Remove DDP, AMP, EMA, and TensorBoard assumptions**

Delete imports and code paths for:

- `torch.distributed`
- `DistributedDataParallel`
- `torch.cuda.amp`
- `SummaryWriter`
- `ExponentialMovingAverage`

**Step 2: Build a single-process `train()` pipeline**

The new `train()` should:

- configure the logger with `stdout`, `log`, and `json`
- load the volume
- derive the Gaussian count
- construct the model
- create optimizer and scheduler
- run reconstruction-only training

**Step 3: Add periodic metric logging**

Use `exp_logger.logkvs({...})` and `exp_logger.dumpkvs()` for:

- step
- loss
- lr
- elapsed time
- target bytes
- estimated payload bits
- derived Gaussian count

**Step 4: Add checkpointing and evaluation**

Save model checkpoints at the configured frequency and periodically evaluate dense reconstruction on the full volume.

### Task 7: Update optimizer plumbing

**Files:**
- Modify: `utils/optim.py`

**Step 1: Make the optimizer config compatible with step-based training**

Keep existing optimizer support but update scheduler handling so it works cleanly with the new loop and no DDP assumptions.

**Step 2: Make LR reporting explicit**

Add a small helper or usage pattern so the loop can log the current LR every reporting step.

### Task 8: Add minimal validation checks

**Files:**
- Modify: `run_lib.py`
- Modify: `utils/data.py`

**Step 1: Validate input shape**

Raise a clear error if the input is not 3D.

**Step 2: Validate compression settings**

Raise a clear error if:

- `target_compression_ratio <= 0`
- derived target bits are too small for one Gaussian plus overhead

**Step 3: Validate normalization mode**

Reject unknown normalization modes with a clear message.

### Task 9: Wire the first experiment config

**Files:**
- Create: `configs/gaussian3d_pam_first.py`

**Step 1: Add a small runnable override**

Set:

- method name
- data path placeholder
- target compression ratio
- step count
- logging cadence
- quantization bits

This file should override the default config rather than redefining everything.

### Task 10: Smoke-test the pipeline

**Files:**
- No code changes required if the earlier tasks are correct

**Step 1: Run a short training smoke test**

Run:

```bash
python main.py --mode=train --config=configs/gaussian3d_pam_first.py --workdir=smoke_gaussian3d
```

**Expected:**

- logger initializes
- volume loads
- Gaussian count is derived
- training steps run
- checkpoints and JSON logs are written

**Step 2: Inspect logged metrics**

Confirm that the log contains:

- loss
- Gaussian count
- target bytes
- payload-bit estimate

### Task 11: Document known limitations

**Files:**
- Modify: `README.md`

**Step 1: Add a short method section**

Document:

- first-pass 3D Gaussian codec scope
- supported input format
- compression-ratio-driven Gaussian count
- current limitations

**Step 2: Add an example launch command**

Include one example using the new config override.
