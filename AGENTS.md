# Repository Guidelines

## Project Structure & Module Organization

Core training code lives in the repo root and a few focused subdirectories:

- `main.py`: CLI entrypoint for `train` and `eval`
- `run_lib.py`: training loop, post-train evaluation, checkpointing, metric export
- `configs/`: base and override configs such as `gaussian3d.py` and `gaussian3d_smoke.py`
- `models/`: 3D Gaussian codec model and quantization helpers
- `utils/`: data loading, logger, optimizer, and compression-budget utilities
- `data/`: local sample inputs such as `smoke_volume.npy`
- `workspace/`: run outputs, logs, checkpoints, and evaluation artifacts
- `docs/plans/`: design and implementation notes

## Build, Test, and Development Commands

Use the `ip` conda environment.

- `conda run -n ip python main.py --mode=train --config=configs/gaussian3d_smoke.py --workdir=smoke`
  Runs a short end-to-end training smoke test.
- `conda run -n ip python main.py --mode=eval --config=configs/gaussian3d_smoke.py --workdir=smoke`
  Evaluates the saved best checkpoint and writes artifacts to `workspace/smoke/eval/`.
- `conda run -n ip python -m compileall main.py run_lib.py configs models utils`
  Fast syntax check for the repository.

## Coding Style & Naming Conventions

- Follow Python 4-space indentation and keep code ASCII unless a file already requires otherwise.
- Prefer descriptive names tied to the domain, e.g. `GaussianVolumeCodec`, `derive_num_gaussians`.
- Keep config names explicit: `gaussian3d_<dataset>.py`.
- Use small helpers for shared logic instead of duplicating evaluation or budget code.
- Fail fast on invalid configs or unsupported modes; do not hide bugs behind defensive fallbacks.

## Testing Guidelines

There is no dedicated test suite yet. Treat smoke runs as the minimum gate:

- run the smoke train command above after meaningful changes
- rerun `eval` when touching metrics, artifact saving, or checkpoint loading
- use `compileall` after editing configs, models, or utilities

When adding tests later, place them under `tests/` and name files `test_<feature>.py`.

## Commit & Pull Request Guidelines

- Use short, imperative commit subjects such as `feat: add post-train eval export` or `fix: align payload accounting`.
- Keep each commit focused on one logical change.
- PRs should include: purpose, key files changed, commands run, and any metric or artifact changes.

## Configuration & Runtime Notes

- `data.coord_norm` must remain `1.0` for the current codec.
- `configs/gaussian3d_pam_first.py` is a template; update `data.path` to your local PAM volume before running.
- Large real-volume runs may still be computationally expensive even with chunked Gaussian evaluation.
