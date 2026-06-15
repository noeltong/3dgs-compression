Validate the recent removal of MAP supervision loss from this repository end to end.

Repo:
`/home/shangqing.tong/pam/pam-compression-3dgs`

Your job:
Act as an independent validation/review agent. Do not implement new features unless required for minimal validation instrumentation. Focus on correctness, regressions, dead references, and whether the MAP-supervision removal is complete and consistent.

Context:
The intended change was to remove MAP supervision loss from training entirely, while keeping standard voxelwise 3D reconstruction training and keeping MAP evaluation/reporting artifacts if they are still used for reconstructed volume quality reporting.

MAP supervision that should be gone:
- `map_loss_enable`
- `map_loss_start_step`
- `map_loss_type`
- `map_loss_weight`
- `map_grad_loss_weight`
- `map_softmax_tau`
- `map_topk`
- `map_column_sample_mode`
- `map_column_sample_height`
- `map_column_sample_width`
- Any helper used only for MAP training loss
- Any MAP-only training patch sampling
- Any MAP-loss-only config files such as `configs/gaussian3d_pam_map.py` and related smoke configs

Important scope boundary:
- MAP supervision must be removed from training.
- MAP evaluation artifacts may remain if they are still used for reporting reconstructed volume quality.
- Normal non-MAP train/eval flows must still work.

What to validate:
1. Training path
- Confirm `run_lib.py` no longer computes or logs MAP supervision losses.
- Confirm total training loss is pure voxelwise 3D reconstruction loss.
- Confirm there are no stale config reads for removed MAP fields.

2. Utility/helper layer
- Confirm MAP-loss-specific helpers were fully removed if no longer used.
- Confirm no dead imports or unused branches remain because of the removal.

3. Config surface
- Confirm defaults and derived configs no longer define MAP-loss fields.
- Confirm deleted MAP-only config files were truly obsolete.
- Confirm standard configs still load correctly.

4. Evaluation path
- Confirm MAP evaluation/reporting that remains is evaluation-only, not training supervision.
- Confirm eval artifacts like MAP projections/metrics are still internally consistent if retained.

5. Documentation / stale references
- Find docs or comments that still describe MAP supervision as an active feature.
- Distinguish between:
  - acceptable historical/reference docs
  - stale docs that now misrepresent the current repo behavior

6. Validation commands
Run at least:
- `conda run -n ip python -m compileall main.py run_lib.py configs models utils`
- `conda run -n ip python main.py --mode=train --config=configs/gaussian3d_smoke.py --workdir=smoke_validation`

If useful, also run eval:
- `conda run -n ip python main.py --mode=eval --config=configs/gaussian3d_smoke.py --workdir=smoke_validation`

7. Repository-wide checks
Use `rg` aggressively to verify removal of MAP supervision references.
Search for:
- `map_loss`
- `map_grad`
- `map_softmax_tau`
- `map_topk`
- `map_column_sample`
- `project_map`
- `MAP-aware`
- `soft-MAP`
- `top-k MAP`

Expected output format:
- Start with findings only, ordered by severity.
- Each finding must include file path and line reference when possible.
- Prioritize actual bugs, regressions, stale references, broken commands, or inconsistencies.
- After findings, include:
  - `Open questions / assumptions`
  - `Validation run`
  - `Residual MAP eval code intentionally kept`
- If no issues are found, say so explicitly.

Constraints:
- Use `rg` for search.
- Do not revert unrelated user changes.
- Keep the task focused on validation, not refactoring.
- If you make any edits, keep them minimal and explain why they were necessary.
