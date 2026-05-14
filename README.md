# 3D Gaussian Compression for PAM/PAT Volumes

This repository fits a quantization-aware 3D Gaussian representation to a single PAM/PAT raw volume for compression experiments.

## Current Scope

- Input format: `.npy` volumes with shape `[H, W, D]`
- Representation: global set of axis-aligned 3D Gaussians
- Training: reconstruction-only
- Budgeting: derive Gaussian count from raw file bytes and target compression ratio
- Runtime: single-process, single-GPU if available

## Current Limitations

- No MAP supervision
- No DAS supervision
- No DDP
- No AMP
- No EMA
- No rotated 3D Gaussians yet

## Configs

- Base config: `configs/default_config.py`
- Method override: `configs/gaussian3d.py`
- PAM template override: `configs/gaussian3d_pam_first.py`
- CLI smoke config: `configs/gaussian3d_smoke.py`

## Example

```bash
python main.py --mode=train --config=configs/gaussian3d_smoke.py --workdir=gaussian3d_smoke
```

For a real PAM volume, update `configs/gaussian3d_pam_first.py` so `data.path` points to your local `.npy` file, then run:

```bash
python main.py --mode=train --config=configs/gaussian3d_pam_first.py --workdir=gaussian3d_first
```

`coord_norm` must stay at `1.0` in this first-pass codec because the Gaussian centers are defined only over `[-1, 1]^3`.
