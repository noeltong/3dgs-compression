# 3D Gaussian PAM/PAT Compression Design

**Date:** 2026-05-12

## Goal

Implement a first-pass 3D Gaussian compression method for PAM/PAT raw volume data inside this template repository. The method should follow the released GaussianImage paper's quantization-aware training spirit, but target true 3D volumetric reconstruction instead of 2D image fitting.

## Scope

This first implementation is intentionally narrow:

- Single-volume fitting from a `.npy` or tensor file with shape `[H, W, D]`
- Reconstruction-only training
- True 3D Gaussian representation
- Single-GPU training only
- Quantization-aware training for Gaussian parameters
- Compression-ratio-driven Gaussian-count derivation

This first implementation explicitly excludes:

- MAP supervision
- DAS supervision
- Multi-volume dataset training
- AMP
- DDP
- EMA
- TensorBoard-first logging workflows
- Rotated 3D Gaussians

## Data Model

The input is one PAM/PAT volume stored as a tensor-like file, starting with `.npy` support. The volume shape is `[H, W, D]`, consistent with the existing PAM INR repo. Raw file bytes are preserved and used for compression-ratio accounting.

Training uses normalized coordinates in `[-1, 1]^3`. A voxel index `(i, j, k)` maps to normalized coordinates:

- `x = 2 * i / (H - 1) - 1`
- `y = 2 * j / (W - 1) - 1`
- `z = 2 * k / (D - 1) - 1`

Voxel intensities are normalized for optimization, while the original file size remains unchanged for bitrate calculations.

## Representation

The model is a global set of 3D Gaussians. Each Gaussian contains:

- center `(x, y, z)`
- axis-aligned scales `(sx, sy, sz)`
- scalar intensity `a`

The first version uses axis-aligned anisotropic Gaussians rather than rotated ellipsoids. This reduces parameterization complexity and keeps the first implementation focused on the codec pipeline.

## Reconstruction Operator

The reconstructed volume is the sum of Gaussian contributions evaluated at voxel centers:

`V_hat(p) = sum_i a_i * G_i(p)`

where `p` is a normalized voxel-center coordinate and `G_i` is a 3D Gaussian defined by the Gaussian's center and scales.

For efficiency, each Gaussian is evaluated only over a bounded local support region instead of the full volume. The support window should be derived from the scales and a configurable truncation factor.

## Quantization Strategy

The quantization path follows the same high-level idea as GaussianImage:

- train with fake quantization in the loop
- estimate compressed payload from quantized parameters
- size the model from target compression ratio

The first version should quantize Gaussian parameter groups separately:

- center parameters
- scale parameters
- intensity parameters

The default starting point is uniform fake quantization per parameter group with configurable bit widths. Learned scale/offset quantizers can be introduced where useful, but the code structure should keep the quantization logic modular so the paper's exact mixed scheme can be tightened later.

## Compression Budget and Gaussian Count

The config provides a target compression ratio against the raw input file:

- `target_bytes = raw_file_bytes / target_compression_ratio`

The runtime estimates:

- per-Gaussian quantized payload bits
- global overhead bits for quantizer metadata

Then it computes:

- `num_gaussians = floor((target_bits - overhead_bits) / bits_per_gaussian)`

The derived Gaussian count is part of the run metadata and should be logged at startup.

## Training Loop

Training is reconstruction-only. The loss is computed between the reconstructed volume and the normalized ground-truth volume. The initial implementation should prefer a simple loss such as MSE, with the code structured so additional losses can be added later.

The training pipeline should be rewritten around:

- single process
- single GPU if available, CPU fallback otherwise
- periodic logging via `utils/logger.py`
- periodic checkpointing
- periodic evaluation on the same fitted volume

AMP, DDP, EMA, and TensorBoard-specific training assumptions should be removed from the template code.

## Config Design

`configs/default_config.py` remains the base config. Sub-config overrides should be added for 3D Gaussian experiments.

The config should include:

- data path and normalization settings
- target compression ratio
- Gaussian quantization bit widths
- truncation radius factor
- training iterations or epochs
- optimizer settings
- logging and checkpoint frequencies

The design should keep config overrides small and composable rather than duplicating the full default config.

## Logging

Training and evaluation metrics should use the existing logger in `utils/logger.py`.

The logger should be configured to emit:

- stdout
- text log file
- JSON progress log

Periodic metrics should be reported with `logkvs()` and flushed with `dumpkvs()`. These metrics should include at least:

- step or epoch
- loss
- learning rate
- derived Gaussian count
- target bytes
- estimated payload bits
- achieved compression ratio estimate
- wall-clock timing

## Main Code Changes

Expected repository changes:

- Replace the placeholder data pipeline in `utils/data.py`
- Replace the placeholder model in `models/model.py`
- Rewrite `run_lib.py` around single-GPU reconstruction-only training
- Update `configs/default_config.py` for 3D Gaussian settings
- Add new config override files under `configs/`
- Keep `main.py` as the main entrypoint unless a cleaner mode split is needed

## Risks and Follow-Up

Main risks in the first pass:

- naive Gaussian field evaluation may be slow for large volumes
- payload estimation may need refinement after real quantizer implementation
- axis-aligned Gaussians may underfit structures that rotated Gaussians could model better

Planned follow-ups after the baseline is stable:

- rotated 3D Gaussians
- tighter paper-style mixed quantization
- support for task-driven PAM/PAT losses
- improved sparse/local evaluation kernels
