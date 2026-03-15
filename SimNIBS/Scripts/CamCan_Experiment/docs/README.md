# Temporal Interference Pipeline (CamCan Experiment)

This repository runs temporal interference (TI) simulations on CamCan subjects, maps fields to individual anatomy, and aggregates variability/robustness metrics across the cohort.

## What was added/changed
- Subject-level post-processing (`post_process.py`) now:
  - Builds ROI masks on the TI grid (Harvard–Oxford or FastSurfer) and saves per-ROI masks/TI volumes/overlap CSVs.
  - Writes full FastSurfer region stats (`region_stats_fastsurfer.csv`) with mean/median/max/pXX/CV/volume per label.
  - Emits per-subject robustness metrics (`subject_metrics.json`) including ROI overlap fractions vs top-percentile TI voxels.
  - Generates overlay PNGs for a chosen ROI.
- Population aggregation (`post_population.py`) now:
  - Merges all per-subject region stats and robustness JSONs.
  - Reports variability/robustness/hotspot tables (fraction above threshold, IQR, CV, worst-case peaks, volume–intensity correlation).
- Helper utilities (`post_functions.py`) gained atlas resampling and region summarization helpers to support the above.
- FastSurfer ROI alias handling now lives in `utils/roi_registry.py`, so dataset roots such as `Left_Hippocampus_Data_test` or `rigth_m1_project` can be resolved automatically.

- **Shared utilities**: `ti_utils.py` (ROI name helpers, TI scalar loading, atlas resampling, region summaries) and `utils/roi_registry.py` (FastSurfer ROI labels, aliases, dataset-name matching).
- **Atlas generation**: `atlas/make_atlas.sh`, `atlas/run_atlasMaker.py` (FastSurfer+FreeSurfer Docker; post-processing expects per-subject atlas files under `<fastsurfer_root>/<subject>.nii.gz`).
- **Simulation**: `simulation/TI_runner_multi-core.py` (Slurm array/local multi-subject) and `simulation/TI_runner_single-core.py` (sequential) create meshes, run SimNIBS TDCS pairs, compute TImax, export TI volumes (`ti_brain_only.nii.gz`).
- **Subject post-processing**: `post/post_process.py` + `post/post_functions.py` consume TI volume + T1 + atlas; write ROI masks, CSVs, overlays, region stats, and subject-level metrics.
- **Population analysis**: `post/post_population.py` aggregates subject outputs into cohort-wide variability/robustness/hotspot tables.
- **Pipeline entrypoint**: `post/run_post_processing.py` runs subject post-processing and optional population aggregation from one config.
- **Mesh export**: `viz/create3dmesh.py` converts TI volumes + masks to VTK/PLY/STL for visualization.
- **Job wrappers**: `my_jobArray.slurm`, `ti_multi.slurm`, and `run_post_processing.slurm` help launch simulation/post-processing steps on HPC.
- **Docs/diagrams**: `README.md`, `PIPELINE_OVERVIEW.md`, `Updated_TI_Pipeline.drawio`.

## Directory layout
- `atlas/`: FastSurfer/FreeSurfer atlas scripts.
- `simulation/`: SimNIBS TI runners (single/multi-subject).
- `post/`: Subject and population post-processing.
- `utils/`: Shared helpers (TI/atlas utilities, simulation helpers).
- `viz/`: Geometry export utilities.
- Root: Slurm wrappers, legacy `functions.py`, docs/diagrams.

## Post-processing pipeline
1) Edit the pipeline config in `post/run_post_processing.py`:
   - `post.root`: dataset root.
   - `post.subjects`: list of subject IDs or `None` for all.
   - `post.atlas_mode`: `auto` (prefer FastSurfer if present), `fastsurfer`, or `mni`.
   - `post.fastsurfer_root` / `post.fs_mri_path`: where to find the subject atlas NIfTI. `fastsurfer_root` is resolved as `<fastsurfer_root>/<subject>.nii.gz`.
   - `post.plot_roi`: set to `None` to infer the ROI from `post.root`, or provide an alias/canonical FastSurfer ROI name directly.
   - `population.target_roi`: set to `None` to reuse the resolved post ROI.
   - `population.enabled`: toggle population aggregation.
   - FastSurfer alias matching expects snake_case names such as `left_hippocampus`, `right_m1`, or `ctx_rh_precentral`.
   - If no alias can be resolved, the pipeline exits before any subject analysis starts for that dataset.
2) Run:
```bash
python post/run_post_processing.py
```
HPC launch (single-node parallel batch):
```bash
sbatch HPC_scripts/run_post_processing.slurm
```
On HPC, the job needs a Python environment with the post-processing stack installed.
If interactive setup is inconvenient, do the one-time setup itself via Slurm:
```bash
sbatch HPC_scripts/bootstrap_post_conda.slurm
```
That job creates or repairs a `ti-post` conda env by default. After it completes,
launch the actual post-processing job with:
```bash
sbatch --export=ALL,POST_CONDA_ENV=ti-post HPC_scripts/run_post_processing.slurm
```
On Stanage, the Sheffield docs recommend loading an `Anaconda3` module and using
`source activate` for your conda environment. The bootstrap script follows that
pattern. If you still want to set the env up manually, one working pattern is:
```bash
module load Anaconda3/2022.05
conda create -n ti-post python=3.11 numpy pandas nibabel scipy nilearn matplotlib
source activate ti-post
sbatch --export=ALL,POST_CONDA_ENV=ti-post HPC_scripts/run_post_processing.slurm
```
If you keep a personal miniconda install instead of the Stanage module, submit with
`POST_CONDA_ENV=<name-or-prefix>` and `POST_CONDA_SH=/path/to/conda.sh`.
If you prefer `venv`, create it first, install `requirements-post.txt`, then submit with
`POST_VENV=/path/to/venv`.
Outputs go to `<root>/<subject>/anat/post/`:
- ROI masks/overlaps (`atlas_<ROI>_mask.nii.gz`, `<ROI>_overlap_topXXpct_mask.nii.gz`).
- TI masked volumes (`TI_in_<ROI>.nii.gz`, `TI_in_<ROI>_TopXX.nii.gz`, `TI_in_TopXX.nii.gz`).
- CSVs of voxel values (`<ROI>_values.csv`, `TopXX_values.csv`, `<ROI>_TopXX_overlap_values.csv`).
- FastSurfer region stats (`region_stats_fastsurfer.csv`).
- Subject metrics (`subject_metrics.json`).
- Overlays for selected ROI (`<ROI>_TI_overlay_topXX.png`, `..._aboveHHH.png` if T1 available).

## Population aggregation
Population aggregation runs from `post/run_post_processing.py` when `population.enabled=True`.
You can still run it directly:
```bash
python post/post_population.py \
  --root /path/to/root \
  --peak-threshold 0.2 \
  --target-roi Left-Hippocampus \
  --template-region-csv /path/to/MNI152/region_stats_fastsurfer.csv
```

## Tests
- Minimal smoke tests for utils and ROI alias resolution: `pytest tests/test_ti_utils.py tests/test_roi_registry.py` (requires pytest + nibabel).
Outputs in `/path/to/root/population_analysis/`:
- `all_region_values.csv` (concatenated per-subject region stats).
- `population_region_summary.csv` (variability/robustness per label).
- `volume_intensity_correlation.csv` (volume vs mean/max TI correlations).
- `subject_robustness.csv` (target ROI peaks/drops, overlap fractions).

## Current pipeline diagram (mermaid)
```mermaid
flowchart TD
    A[CamCan T1/T2 + manual corrections] --> B[FastSurfer + FreeSurfer atlases\n(make_atlas.sh / run_atlasMaker.py)]
    A --> C[Subject CHARM mesh + manual seg merge\n(TI_runner_*)]
    T[MNI152 template TI montage optimization] --> C
    C --> D[SimNIBS TI simulations per subject\n(TI_runner_multi-core/single-core)]
    D --> E[TI volumes + labels (msh2nii)\n ti_brain_only.nii.gz]
    E --> F[Subject post-processing\n(post_process.py)]
    F --> G[ROI masks, CSVs, overlays,\nregion_stats_fastsurfer.csv,\nsubject_metrics.json]
    G --> H[Population aggregation\n(post_population.py)]
    H --> I[Population hotspot & robustness tables\n(IQR, CV, peak frac > thr, worst cases)]
    G --> J[Optional VTK/PLY/STL exports\n(create3dmesh.py)]
```

## Notes
- `post_process.py` automatically chooses FastSurfer atlas when available (`atlas_mode=auto`); otherwise defaults to Harvard–Oxford (MNI).
- `run_post_processing.py` can infer the target ROI from the dataset directory name via `utils/roi_registry.py`; use hemisphere-specific aliases for FastSurfer targets.
- If a dataset directory name does not match any known ROI alias, the pipeline aborts early and does not process that dataset.
- Thresholds are configurable: ROI overlap percentile (`percentile`), hard cutoff (`hard_threshold`), population peak threshold (`--peak-threshold`).
- Keep outputs per subject under `<root>/<subject>/anat/post/` so population aggregation can auto-discover them.
