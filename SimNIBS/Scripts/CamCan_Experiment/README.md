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

- **Shared utilities**: `ti_utils.py` (ROI name helpers, TI scalar loading, atlas resampling, region summaries).
- **Atlas generation**: `atlas/make_atlas.sh`, `atlas/run_atlasMaker.py` (FastSurfer+FreeSurfer Docker, produces aparc.DKTatlas+aseg.deep.nii.gz).
- **Simulation**: `simulation/TI_runner_multi-core.py` (Slurm array/local multi-subject) and `simulation/TI_runner_single-core.py` (sequential) create meshes, run SimNIBS TDCS pairs, compute TImax, export TI volumes (`ti_brain_only.nii.gz`).
- **Subject post-processing**: `post/post_process.py` + `post/post_functions.py` consume TI volume + T1 + atlas; write ROI masks, CSVs, overlays, region stats, and subject-level metrics.
- **Population analysis**: `post/post_population.py` aggregates subject outputs into cohort-wide variability/robustness/hotspot tables.
- **Mesh export**: `viz/create3dmesh.py` converts TI volumes + masks to VTK/PLY/STL for visualization.
- **Job wrappers**: `my_jobArray.slurm`, `ti_multi.slurm` submit simulations on HPC.
- **Docs/diagrams**: `README.md`, `PIPELINE_OVERVIEW.md`, `Updated_TI_Pipeline.drawio`.

## Directory layout
- `atlas/`: FastSurfer/FreeSurfer atlas scripts.
- `simulation/`: SimNIBS TI runners (single/multi-subject).
- `post/`: Subject and population post-processing.
- `utils/`: Shared helpers (TI/atlas utilities, simulation helpers).
- `viz/`: Geometry export utilities.
- Root: Slurm wrappers, legacy `functions.py`, docs/diagrams.

## Subject-level post-processing
1) Set config in `post/post_process.py` (or instantiate `PostProcessConfig` in your own script):
   - `root_dir`: dataset root.
   - `subject`: subject ID (e.g., `sub-CC110037` or `MNI152`).
   - `atlas_mode`: `auto` (prefer FastSurfer if present), `fastsurfer`, or `mni`.
   - `fastsurfer_root` / `fs_mri_path`: where to find `aparc.DKTatlas+aseg.deep.nii.gz`.
2) Run:
```bash
python post/post_process.py
# or import and call run_post_process(cfg)
```
Outputs go to `<root>/<subject>/anat/post/`:
- ROI masks/overlaps (`atlas_<ROI>_mask.nii.gz`, `<ROI>_overlap_topXXpct_mask.nii.gz`).
- TI masked volumes (`TI_in_<ROI>.nii.gz`, `TI_in_<ROI>_TopXX.nii.gz`, `TI_in_TopXX.nii.gz`).
- CSVs of voxel values (`<ROI>_values.csv`, `TopXX_values.csv`, `<ROI>_TopXX_overlap_values.csv`).
- FastSurfer region stats (`region_stats_fastsurfer.csv`).
- Subject metrics (`subject_metrics.json`).
- Overlays for selected ROI (`<ROI>_TI_overlay_topXX.png`, `..._aboveHHH.png` if T1 available).

## Population aggregation
Run after all subjects have post-processing outputs:
```bash
python post/post_population.py \
  --root /path/to/root \
  --peak-threshold 0.2 \
  --target-roi Hippocampus \
  --template-region-csv /path/to/MNI152/region_stats_fastsurfer.csv
```

## Tests
- Minimal smoke tests for utils: `pytest tests/test_ti_utils.py` (requires pytest + nibabel).
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
- Thresholds are configurable: ROI overlap percentile (`percentile`), hard cutoff (`hard_threshold`), population peak threshold (`--peak-threshold`).
- Keep outputs per subject under `<root>/<subject>/anat/post/` so population aggregation can auto-discover them.
