# Repository Guidelines

## Project Structure & Module Organization
- `TI_runner_multi-core.py`, `TI_runner_single-core.py`: run SimNIBS TI simulations (meshing → TDCS pairs → TImax → msh2nii export).
- `post_process.py` + `post_functions.py`: subject-level TI mapping to atlases, ROI masks/CSVs/overlays, region stats, robustness metrics.
- `post_population.py`: aggregates per-subject outputs into population hotspot/robustness summaries.
- `make_atlas.sh`, `run_atlasMaker.py`: FastSurfer/FreeSurfer atlas generation.
- `create3dmesh.py`: convert TI volumes + masks to VTK/PLY/STL.
- Slurm wrappers: `my_jobArray.slurm`, `ti_multi.slurm`.
- Docs/diagrams: `README.md`, `Updated_TI_Pipeline.drawio`.

## Build, Test, and Development Commands
- Subject post-process (example): `python post_process.py` (edit cfg at bottom or import `run_post_process`).
- Population aggregation: `python post_population.py --root <root> --peak-threshold 0.2 --target-roi Hippocampus`.
- Simulation (single subject): `python TI_runner_multi-core.py --subject sub-XXX` (expects SimNIBS env and input data).
- Atlas generation: `./make_atlas.sh <DATA_DIR> <THREADS> <LICENSE_PATH>` (Docker required).

## Coding Style & Naming Conventions
- Python 3; prefer PEP8 with 4-space indent; concise, purposeful comments only for non-obvious logic.
- Use snake_case for variables/functions; UpperCamelCase for classes.
- Keep outputs under `<root>/<subject>/anat/post/` for discoverability by population scripts.

## Testing Guidelines
- No formal test suite present; validate by running small-scope jobs:
  - Post-process dry run on MNI152 or one subject and confirm outputs (ROI CSVs, overlays).
  - Population script on a few subjects to ensure summaries are generated without errors.
- For mesh/sim changes, spot-check a single subject end-to-end.

## Commit & Pull Request Guidelines
- Commits: concise imperative subject lines (e.g., “Add population aggregation script”), group related changes.
- PRs: include purpose, key changes, run commands/output checked (e.g., post_process on subject X), and any screenshots of overlays/plots if relevant. Link issues when applicable.

## Environment & Execution Tips
- SimNIBS/Charm/msh2nii must be available in PATH (see Slurm scripts for module usage).
- FastSurfer/FreeSurfer atlases expected at `<root>/FastSurfer_out/<sub>/mri/aparc.DKTatlas+aseg.deep.nii.gz` or via `--fs-mri-path`.
- Avoid destructive git commands; do not overwrite user outputs under subject folders without backups.***
