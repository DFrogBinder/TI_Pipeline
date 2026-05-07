# Temporal Interference Pipeline (CamCan Experiment)

This repository runs temporal interference (TI) simulations on CamCan subjects, maps fields to individual anatomy, and aggregates variability/robustness metrics across the cohort.

## What was added/changed
- Subject-level post-processing (`post_process.py`) now:
  - Builds ROI masks on the TI grid (Harvard–Oxford or FastSurfer) and saves per-ROI masks/TI volumes/overlap CSVs.
  - Writes full FastSurfer region stats (`region_stats_fastsurfer.csv`) with mean/median/max/pXX/CV/volume per label.
  - Emits per-subject robustness metrics (`subject_metrics.json`) including ROI overlap fractions vs top-percentile TI voxels.
  - Generates seven overlay PNGs per chosen ROI by default when a T1 is available: three `context` views, three `roi_focus` views, and one true whole-brain-max reference view.
- Population aggregation (`post_population.py`) now:
  - Merges all per-subject region stats and robustness JSONs.
  - Reports variability/robustness/hotspot tables (fraction above threshold, IQR, CV, worst-case peaks, volume–intensity correlation).
- Helper utilities (`post_functions.py`) gained atlas resampling and region summarization helpers to support the above.
- FastSurfer ROI alias handling now lives in `utils/roi_registry.py`, so dataset roots such as `Left_Hippocampus_Data_test` or `rigth_m1_project` can be resolved automatically.

- **Shared utilities**: `ti_utils.py` (ROI name helpers, TI scalar loading, atlas resampling, region summaries) and `utils/roi_registry.py` (FastSurfer ROI labels, aliases, dataset-name matching).
- **Atlas generation**: `atlas/make_atlas.sh`, `atlas/run_atlasMaker.py` (FastSurfer+FreeSurfer Docker; post-processing expects per-subject atlas files under `<fastsurfer_root>/<subject>.nii.gz`).
- **Simulation**: `simulation/TI_runner_multi-core.py` (Slurm array/local multi-subject), `simulation/TI_runner_single-core.py` (sequential/local debug), and `simulation/TI_runner_MNI152.py` (dedicated MNI152 template runner) create meshes or use the built-in MNI mesh, run SimNIBS TDCS pairs, compute TImax, and export TI volumes (`ti_brain_only.nii.gz`).
- **Subject post-processing**: `post/post_process.py` + `post/post_functions.py` consume TI volume + T1 + atlas; write ROI masks, CSVs, overlays, region stats, and subject-level metrics.
- **Population analysis**: `post/post_population.py` aggregates subject outputs into cohort-wide variability/robustness/hotspot tables.
- **Pipeline entrypoint**: `post/run_post_processing.py` runs subject post-processing and optional population aggregation from one config.
- **Metric dictionary**: `docs/METRIC_DICTIONARY.md` explains how every subject-level, population-level, and repeatability metric is computed.
- **Mesh export**: `viz/create3dmesh.py` converts TI volumes + masks to VTK/PLY/STL for visualization.
- **Job wrappers**: `my_jobArray.slurm`, `ti_multi.slurm`, and `run_post_processing.slurm` help launch simulation/post-processing steps on HPC.
- **Docs/diagrams**: `README.md`, `PIPELINE_OVERVIEW.md`, source diagrams in `drawio/`, and publication exports in `docs/figures/post_pipeline/`.

## Directory layout
- `atlas/`: FastSurfer/FreeSurfer atlas scripts.
- `simulation/`: SimNIBS TI runners (single-subject, multi-subject, and dedicated MNI152 template).
- `post/`: Subject and population post-processing.
- `utils/`: Shared helpers (TI/atlas utilities, simulation helpers).
- `viz/`: Geometry export utilities.
- Root: Slurm wrappers, legacy `functions.py`, docs/diagrams.

## MNI152 template simulation
Use `simulation/TI_runner_MNI152.py` when you want to run only the built-in SimNIBS `MNI152` head model without toggling `runMNI152` inside the subject-oriented runners.

The script:
- uses the built-in SimNIBS MNI152 mesh and reference T1 by default
- writes outputs to `<root>/MNI152/anat/SimNIBS/`
- exposes named montage presets such as `left-thalamus`, `right-thalamus`, `hippocampus`, `m1`, `left-pallidum`, and `right-pallidum`
- lets you override electrode centres, current amplitudes, size, thickness, conductivity, and mesh element size from the CLI

List the available presets:
```bash
simnibs_python simulation/TI_runner_MNI152.py --list-presets
```

Run one MNI152 simulation:
```bash
simnibs_python simulation/TI_runner_MNI152.py \
  --root-dir simulation/example_experiment_root \
  --preset left-thalamus
```

Optional overrides:
```bash
simnibs_python simulation/TI_runner_MNI152.py \
  --root-dir simulation/example_experiment_root \
  --preset hippocampus \
  --pair1-anode F10 \
  --pair1-cathode P8 \
  --pair1-current-a 0.002 \
  --pair2-anode T7 \
  --pair2-cathode P7 \
  --pair2-current-a 0.001588656
```

The MNI152 runner does not perform subject-specific CHARM meshing or segmentation replacement. It is intended for fast switching between template-only montage configurations while keeping the same downstream output layout expected by the post-processing code.

## Post-processing pipeline
1) Edit the pipeline config in `post/run_post_processing.py`:
   - `post.root`: dataset root.
   - `post.subjects`: list of subject IDs or `None` for all.
   - `post.atlas_mode`: `auto` (prefer FastSurfer if present), `fastsurfer`, or `mni`.
   - `post.fastsurfer_root` / `post.fs_mri_path`: where to find the subject atlas NIfTI. `fastsurfer_root` is resolved as `<fastsurfer_root>/<subject>.nii.gz`.
   - `--atlas-filename` or `post.fastsurfer_atlas_filename`: optional atlas override for all subjects. Use an absolute atlas path to reuse one atlas for every subject, or a relative path such as `mri/aparc.DKTatlas+aseg.deep.nii.gz` to resolve `<fastsurfer_root>/<subject>/mri/aparc.DKTatlas+aseg.deep.nii.gz`.
   - `post.plot_roi`: set to `None` to infer the ROI from `post.root`, or provide an alias/canonical FastSurfer ROI name directly.
   - `post.overlay_full_field`: defaults to `True`, so each subject now gets `full`, `topXX`, and `aboveHHH` overlays for both scale modes, plus one whole-brain reference full-field overlay.
   - `population.target_roi`: set to `None` to reuse the resolved post ROI.
   - `population.enabled`: toggle population aggregation.
   - FastSurfer alias matching expects snake_case names such as `left_hippocampus`, `right_m1`, or `ctx_rh_precentral`.
   - If no alias can be resolved, the pipeline exits before any subject analysis starts for that dataset.
2) Run:
```bash
python post/run_post_processing.py
```
Example with per-subject atlas filename override:
```bash
python post/run_post_processing.py --atlas-filename mri/aparc.DKTatlas+aseg.deep.nii.gz
```
HPC launch (single-node parallel batch):
```bash
sbatch HPC_scripts/run_post_processing.slurm
```
Interactive worker shell with matching default resources:
```bash
./HPC_scripts/start_interactive_post.sh
```
You can override the defaults with environment variables such as
`SLURM_INTERACTIVE_TIME=02:00:00` or pass extra `srun` flags like
`./HPC_scripts/start_interactive_post.sh --account=<account>`.
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
- Overlays for selected ROI when T1 is available:
  - Context scale: `<ROI>_TI_overlay_context_<subject>_full.png`, `<ROI>_TI_overlay_context_<subject>_topXX.png`, `<ROI>_TI_overlay_context_<subject>_aboveHHH.png`
  - ROI-focus scale: `<ROI>_TI_overlay_roi_focus_<subject>_full.png`, `<ROI>_TI_overlay_roi_focus_<subject>_topXX.png`, `<ROI>_TI_overlay_roi_focus_<subject>_aboveHHH.png`
  - Whole-brain reference: `<ROI>_TI_overlay_whole_brain_reference_<subject>_full.png`
  - `context` uses a robust upper colorbar limit from labeled non-CSF / non-ventricular tissue when a FastSurfer atlas is available, which reduces distortion from small CSF hotspots.
  - `roi_focus` uses a robust upper colorbar limit computed from voxels inside the selected ROI, which improves contrast within the target.
  - `whole_brain_reference` uses the true maximum positive TI value across the whole brain, giving you the old unscaled reference view alongside the two robustly scaled modes.

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
- `population_cohort_manifest.csv` (complete subject cohort used by within-run aggregation).
- `population_region_summary.csv` (variability/robustness per label).
- `volume_intensity_correlation.csv` (legacy pooled volume vs mean/max TI correlations).
- `regional_volume_intensity_correlation.csv` (per-region volume vs mean/max TI correlations when at least three subjects are available).
- `subject_robustness.csv` (target ROI peaks/drops, overlap fractions).
- `subject_metric_values.csv` and `population_subject_metric_summary.csv` (flattened subject metrics and cohort summaries).
- `subject_neighbor_metrics.csv` and `population_neighbor_summary.csv` (raw and summarized neighbor exposure metrics).
- `population_anatomy_correlations.csv` and `worst_case_subjects.csv` (anatomy/performance associations and lowest ROI-peak subjects).

Across-repeat outputs in `<batch_root>/subject_metrics_analysis/`:
- `subject_metrics_long.csv` and `run_subject_coverage.csv` (flattened subject-run records and cohort coverage).
- `repeat_level_population_statistics.csv` and `repeat_level_population_statistics_complete_subjects.csv` (per-repeat population summaries).
- `experiment_level_population_statistics.csv`, `variation_analysis_metrics.csv`, `pairwise_run_differences.csv`, and `within_subject_repeatability.csv` (repeatability and drift summaries).
- `subject_repeat_metric_means.csv`, `subject_repeat_metric_sds.csv`, `subject_level_variation.csv`, `subject_level_variation_summary.csv`, `subject_level_top_variable_subjects.csv`, and `subject_cross_metric_instability.csv` (subject-level repeat variation).
- `image_repeatability_*.csv`, image repeatability reports, optional log-audit outputs, Markdown reports, and `figures/*.png`.

## Current pipeline diagram (mermaid)
```mermaid
flowchart TD
    A[CamCan T1/T2 + manual corrections] --> B[FastSurfer + FreeSurfer atlases\n(make_atlas.sh / run_atlasMaker.py)]
    A --> C[Subject CHARM mesh + manual seg merge\n(TI_runner_*)]
    T[MNI152 template TI montage optimization] --> C
    C --> D[SimNIBS TI simulations per subject or MNI152 template\n(TI_runner_multi-core/single-core/MNI152)]
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
- Overlay scaling is separate from thresholding: the percentile and hard cutoff decide which voxels are shown in the thresholded views, while the `context` and `roi_focus` modes decide how the colorbar is scaled.
- Keep outputs per subject under `<root>/<subject>/anat/post/` so population aggregation can auto-discover them.
