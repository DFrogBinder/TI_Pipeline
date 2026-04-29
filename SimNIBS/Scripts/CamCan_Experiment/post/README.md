# TI Post-Processing Pipeline

This directory contains the post-processing and repeatability-analysis code for the TI simulation workflow used in the CamCAN experiment pipeline.

The code answers three different analysis questions:

1. **Subject level**
   For one subject in one simulation run, what field metrics should be extracted, what masks and images should be saved, and what should be recorded for downstream analysis?
2. **Within-run population level**
   For one dataset run, how do those subject-level metrics vary across subjects?
3. **Across-repeat repeatability level**
   When the same ROI is simulated repeatedly with identical parameters, how stable are the subject-level metrics across the 10 reruns?

This README is intended to bring a new developer or analyst up to speed on:

- what the pipeline is for
- what is being measured
- how the data moves through the code
- which scripts own which responsibilities
- what files are written at each stage
- how to run and extend the pipeline safely

## Purpose

The pipeline is designed for repeated TI simulations where:

- one ROI is targeted per dataset family
- the montage and stimulation parameters are fixed for that ROI
- each subject is simulated repeatedly
- each repeat regenerates the mesh
- the analysis must separate true subject differences from repeat-to-repeat numerical or meshing variability

The current implementation supports the analysis concepts discussed for the ROI repeatability study:

- overlap between the target ROI and the top field distribution
- target ROI peak and mean field strength
- absolute delta from an MNI baseline
- focality above a fixed threshold
- field intensity in anatomically neighboring regions
- anatomy-linked metrics such as CSF distance, skull distance, and electrode-to-ROI distance
- subject-level and population-level repeatability statistics across reruns

## High-Level Pipeline

The pipeline has four layers.

### 1. Per-subject post-processing

Implemented in [post_process.py](./post_process.py).

This stage:

- loads the TI image for one subject
- resolves the target ROI mask on the TI grid
- computes percentile-threshold and overlap products
- preserves the existing overlay image generation
- appends extended metrics to `subject_metrics.json`

### 2. Within-run population aggregation

Implemented in [post_population.py](./post_population.py).

This stage:

- reads per-subject `subject_metrics.json`
- reads per-subject FastSurfer region summary tables when available
- aggregates subject metrics across all subjects in one dataset run
- summarizes neighboring-region and anatomy-linked metrics
- produces per-run robustness tables

### 3. Repeat-batch orchestration

Implemented in [run_post_processing_batch.py](./run_post_processing_batch.py) and [run_post_processing_batch_env.py](./run_post_processing_batch_env.py).

This stage:

- discovers repeated dataset folders such as `*_Data_01` to `*_Data_10`
- runs the per-subject and per-run stages for each repeat
- optionally launches the across-repeat repeatability stage automatically after the batch completes

### 4. Across-repeat repeatability analysis

Implemented in [repeatability/analyze_subject_metrics.py](./repeatability/analyze_subject_metrics.py).

This stage:

- loads `subject_metrics.json` across all repeats for the selected ROI
- computes run-level and experiment-level population statistics
- computes subject-level variation across repeats
- writes tables, figures, and narrative reports
- writes per-subject mean and SD summaries across repeats for the extended metrics

## Data Model

There are two different concepts that matter throughout this codebase.

### Dataset run

A dataset run is one concrete directory such as:

```text
/root/Left_Hippocampus_Data_01
```

This is one execution of the simulation pipeline for one ROI.

### Repeat batch

A repeat batch is a parent directory containing the repeated runs for the same ROI:

```text
/root/
├── Left_Hippocampus_Data_01
├── Left_Hippocampus_Data_02
├── ...
└── Left_Hippocampus_Data_10
```

The repeatability layer compares subjects across these sibling directories.

### Subject output directory

Per-subject post-processing outputs are written beneath:

```text
<dataset_root>/<subject>/anat/post/
```

The most important file in that directory is `subject_metrics.json`.

Visual overview:

- Summary structure: [draw.io](../docs/post_pipeline_metric_flowchart.drawio), [SVG](../docs/post_pipeline_metric_flowchart.svg)
- Inputs and subject-level metrics: [draw.io](../docs/post_pipeline_metric_flowchart_inputs_subject_level.drawio), [SVG](../docs/post_pipeline_metric_flowchart_inputs_subject_level.svg)
- Cohort definition across repeats: [draw.io](../docs/post_pipeline_metric_flowchart_cohort_definition_across_repeats.drawio), [SVG](../docs/post_pipeline_metric_flowchart_cohort_definition_across_repeats.svg)
- Within-run population metrics: [draw.io](../docs/post_pipeline_metric_flowchart_within_run_population_metrics.drawio), [SVG](../docs/post_pipeline_metric_flowchart_within_run_population_metrics.svg)
- Across-repeat repeatability metrics: [draw.io](../docs/post_pipeline_metric_flowchart_across_repeat_repeatability_metrics.drawio), [SVG](../docs/post_pipeline_metric_flowchart_across_repeat_repeatability_metrics.svg)

## What Is Measured

The pipeline currently measures five analysis families.

### 1. Overlap and percentile metrics

These are the original post-processing metrics and remain the core compatibility layer for the pipeline.

They quantify how the top field distribution overlaps the target ROI.

Main fields:

- `percentile_value`
- `region_percentile`
- `top_percentile_voxels`
- `rois[ROI].roi_voxels`
- `rois[ROI].overlap_top_voxels`
- `rois[ROI].roi_volume_mm3`
- `rois[ROI].overlap_volume_mm3`
- `rois[ROI].overlap_fraction`
- `rois[ROI].roi_percentile_value`

Interpretation:

- `percentile_value` is the whole-brain threshold used to define the top field mask
- `top_percentile_voxels` is the size of that high-field mask
- `overlap_fraction` is the fraction of the target ROI occupied by the top field
- `roi_percentile_value` is the same percentile computed only inside the target ROI for one subject and one repeat
- `region_percentile` records which percentile level was used for the ROI-internal percentile calculation

### 2. Target ROI intensity metrics

These describe the magnitude of the field inside the target ROI.

Main fields in `extended_metrics`:

- `roi_peak`
- `roi_mean`
- `roi_peak_abs_delta_mni`
- `roi_mean_abs_delta_mni`
- `mni_baseline_roi_peak`
- `mni_baseline_roi_mean`

Interpretation:

- `roi_peak` is the maximum field inside the ROI
- `roi_mean` is the average field inside the ROI
- the `*_abs_delta_mni` fields are absolute differences relative to the configured MNI baseline, not signed drops

### 3. Focality metrics

These quantify how much brain volume exceeds the chosen threshold.

Main fields:

- `focality_threshold_v_per_m`
- `focality_voxels_gt_threshold`
- `focality_volume_mm3_gt_threshold`
- `focality_voxels_abs_delta_mni`
- `focality_volume_mm3_abs_delta_mni`

Interpretation:

- these metrics estimate how widespread the strong field is
- they are useful as a quality or spread measure relative to the MNI baseline

### 4. Neighbor-region metrics

These quantify field intensity in the regions that are anatomically adjacent to the target ROI.

Important implementation detail:

- neighboring regions are defined from a **fixed MNI adjacency template per ROI**
- they are not re-derived independently for each subject

Main fields:

- `neighbors`
- `neighbor_mean_of_means`
- `neighbor_max_of_max`
- `neighbor_min_of_max`
- flattened fields such as `neighbor_mean__<label>` and `neighbor_peak__<label>`

Interpretation:

- these metrics describe how strongly field leaks into the immediately surrounding anatomy
- the fixed-template design ensures every subject is compared against the same surrounding-region definition

### 5. Anatomical context metrics

These characterize subject-specific anatomical context that may explain variability.

Main fields:

- `roi_centroid_x`
- `roi_centroid_y`
- `roi_centroid_z`
- `csf_distance_mm`
- `skull_distance_mm`
- `electrode_distances`
- `electrode_distance_mean_mm`
- `electrode_distance_min_mm`
- `electrode_distance_max_mm`

Interpretation:

- these are candidate covariates for explaining between-subject or between-repeat differences
- they are recomputed per repeat at the subject-processing stage
- the repeatability analysis then summarizes them across the 10 repeats per subject

## Core Design Decisions

### Overlay generation is preserved

The existing overlay image generation in `post_process.py` was intentionally left untouched because it is operationally useful for visual QC.

### `subject_metrics.json` stores raw per-repeat values

Each `subject_metrics.json` corresponds to one subject in one repeat.

That means:

- raw repeat-specific values are appended directly into the JSON
- final per-subject mean values across the 10 repeats are not written back into each per-repeat JSON
- those across-repeat summaries are written by the repeatability stage into downstream tables

This keeps the data model logically correct.

### Neighboring regions are fixed by ROI

Neighbor labels come from a configured MNI atlas template and are reused across subjects and repeats for that ROI.

This avoids subject-by-subject changes in adjacency definition.

### MNI comparisons use absolute delta

The pipeline uses absolute differences relative to the configured MNI baseline for the new baseline-comparison metrics.

## Directory Structure

```text
post/
├── README.md
├── metric_extensions.py
├── post_functions.py
├── post_population.py
├── post_process.py
├── run_full_post_pipeline.py
├── repeatability/
│   ├── analyze_subject_metrics.py
│   ├── README.md
│   └── __init__.py
├── robustness_analysis.py
├── run_post_processing.py
├── run_post_processing_batch.py
├── run_post_processing_batch_env.py
├── configs/
│   ├── electrode_centers_placeholder.csv
│   └── mni_baseline_placeholder.csv
└── utils/
    ├── cleanup_post_dirs.py
    ├── collect_overlay_pngs.py
    ├── collect_post_outputs.py
    └── sanity_check.py
```

## Script Responsibilities

### `post_process.py`

Library-style subject-level processing.

Responsibilities:

- load TI image and T1 image
- resolve the target ROI mask
- build percentile-threshold masks
- build overlap masks and tables
- write masks and masked TI volumes
- write overlay PNGs
- append extended metrics to `subject_metrics.json`

Main output contract:

- one subject processed
- one `subject_metrics.json`
- one set of ROI masks, overlays, and optional region tables

### `metric_extensions.py`

Shared helper module for the new metrics.

Responsibilities:

- derive MNI baseline metrics from the configured MNI simulation root
- derive fixed neighbor templates from the configured MNI atlas
- compute ROI peak/mean metrics
- compute the ROI-internal percentile value written to `rois[ROI].roi_percentile_value`
- compute focality metrics
- compute anatomy-linked distances
- compute electrode distance summaries
- flatten nested JSON fields for downstream aggregation

This module is the main bridge between subject-level post-processing and repeatability analysis.

### `post_population.py`

Within-run population aggregation.

Responsibilities:

- concatenate per-subject region tables
- aggregate region statistics across subjects
- flatten the extended `subject_metrics.json` payload
- summarize subject-level metrics across subjects
- summarize neighboring-region metrics
- compute anatomy-to-performance correlations
- write within-run population tables

### `robustness_analysis.py`

Standalone robustness analysis.

Responsibilities:

- compute many of the same extended metrics outside the main pipeline
- produce per-subject and population robustness tables

This script is useful for focused analysis, but the main production path is now through `post_process.py` and `post_population.py`.

### `run_post_processing.py`

Single-dataset pipeline runner.

Responsibilities:

- discover subjects under one dataset root
- call `post_process.py` for each subject
- optionally call `post_population.py`

This is the main entrypoint for one dataset run.

### `run_full_post_pipeline.py`

Single-command CLI wrapper for the whole pipeline.

Responsibilities:

- auto-detect whether `--root` points to one dataset or a repeat batch root
- build the single-dataset and batch configuration objects from CLI arguments
- run subject-level processing
- optionally run within-run population aggregation
- optionally run across-repeat repeatability analysis

This is the recommended entrypoint for most interactive use.

### `run_post_processing_batch.py`

Repeat-batch pipeline runner.

Responsibilities:

- discover repeated dataset folders
- run the single-dataset pipeline for each repeat
- write a batch summary JSON
- optionally run across-repeat repeatability analysis per ROI after the batch completes

This is the main end-to-end entrypoint when testing the full repeated-run pipeline.

### `run_post_processing_batch_env.py`

Environment-driven wrapper for HPC or Slurm execution.

Responsibilities:

- build pipeline configuration from environment variables
- avoid the stdin-launch pattern that interferes with multiprocessing on HPC

### `repeatability/analyze_subject_metrics.py`

Across-repeat repeatability analysis.

Responsibilities:

- read all `subject_metrics.json` files for one ROI across repeats
- flatten overlap and extended metrics into a long table
- compute complete-case cohorts
- optionally restrict the analysis to the complete-case cohort present in every selected repeat
- compute per-run and experiment-level statistics
- compute subject-level repeat variation
- compute image-level repeatability on saved ROI masks, top-5% masks, overlap masks, and within-ROI field images
- quantify hotspot localization stability from peak displacement and overlap-mask center of mass
- compute mean and SD across repeats per subject
- generate figures and reports

## `subject_metrics.json` Schema

The subject-level JSON now contains the original overlap structure plus `extended_metrics`.

Simplified shape:

```json
{
  "schema_version": 2,
  "subject": "sub-CCxxxxxx",
  "target_roi": "Left-Hippocampus",
  "percentile": 95.0,
  "percentile_value": 0.214,
  "region_percentile": 95.0,
  "voxel_volume_mm3": 1.0,
  "top_percentile_voxels": 53100,
  "rois": {
    "Left-Hippocampus": {
      "roi_voxels": 4200,
      "overlap_top_voxels": 1700,
      "roi_volume_mm3": 4200.0,
      "overlap_volume_mm3": 1700.0,
      "overlap_fraction": 0.40,
      "roi_percentile": 95.0,
      "roi_percentile_value": 0.267
    }
  },
  "extended_metrics": {
    "roi_peak": 0.31,
    "roi_mean": 0.18,
    "roi_peak_abs_delta_mni": 0.02,
    "roi_mean_abs_delta_mni": 0.01,
    "focality_voxels_gt_threshold": 51000,
    "csf_distance_mm": 4.8,
    "electrode_distance_mean_mm": 63.2,
    "neighbors": [...],
    "electrode_distances": [...]
  }
}
```

## Output Files by Stage

### Subject-level outputs

Typical files in `<dataset>/<subject>/anat/post/`:

- `subject_metrics.json`
- `region_stats_fastsurfer.csv`
- percentile-threshold NIfTI masks
- ROI masks and overlap masks
- ROI voxel tables
- overlay PNGs
- `<roi>_fixed_neighbors.json`
- `<roi>_electrode_distances.json`

### Within-run outputs

Typical files in `<dataset>/population_analysis/`:

- `all_region_values.csv`
- `population_region_summary.csv`
- `volume_intensity_correlation.csv`
- `subject_robustness.csv`
- `subject_metric_values.csv`
- `population_subject_metric_summary.csv`
- `subject_neighbor_metrics.csv`
- `population_neighbor_summary.csv`
- `population_anatomy_correlations.csv`
- `worst_case_subjects.csv`

### Across-repeat outputs

Typical files in `<batch_root>/repeatability_analysis/<roi>/` or `<dataset_root>/subject_metrics_analysis/`:

- `subject_metrics_long.csv`
- `repeat_level_population_statistics.csv`
- `repeat_level_population_statistics_complete_subjects.csv`
- `experiment_level_population_statistics.csv`
- `within_subject_repeatability.csv`
- `subject_repeat_metric_means.csv`
- `subject_repeat_metric_sds.csv`
- `subject_level_variation.csv`
- `image_repeatability_run_level.csv`
- `image_repeatability_pairwise_subject_run_pairs.csv`
- `image_repeatability_subject_level.csv`
- `image_repeatability_pairwise_run_summary.csv`
- `image_repeatability_cohort_summary.csv`
- `image_repeatability_issues.csv`
- `image_repeatability_report.md`
- `image_repeatability_methodology.md`
- figures
- Markdown interpretation and methodology reports

## Repeatability Logic

The repeatability stage always uses all available repeats for the chosen ROI.

The main ideas are:

- each `subject_metrics.json` contributes one row for one subject in one repeat
- a complete-case cohort is built from subjects that are present in every repeat
- by default, repeat-level summaries are restricted to that complete-case cohort
- experiment-level summaries always use the complete-case cohort for subject-matched repeated measures
- experiment-level summaries quantify run-to-run variability relative to subject-to-subject variability
- subject-level summaries identify which subjects are especially unstable across repeats
- within-run population summaries in batch mode are rerun on the complete-case cohort after all repeats finish

Important defaults:

- batch mode now discards subjects that do not complete all selected repeats when it produces within-run population summaries and repeatability outputs
- the strict cohort can be relaxed only when you explicitly opt in with `--allow-incomplete-repeat-subjects`, `--allow-incomplete-subjects`, or `PIPELINE_COMPLETE_REPEAT_SUBJECTS_ONLY=0`

Important consequence:

- the per-subject final repeatability summaries, such as the mean ROI peak across the 10 reruns, are computed downstream from the repeated measurements
- the batch root also receives one complete-case subject manifest per ROI so the exact analysis cohort is auditable

## Configuration

The code supports both Python configuration objects and environment-driven configuration.

### Important subject/batch configuration fields

Useful fields in `PostBatchConfig`:

- `root`
- `atlas_mode`
- `fastsurfer_root`
- `fastsurfer_atlas_filename`
- `plot_roi`
- `percentile`
- `offtarget_threshold`
- `mni_baseline_root`
- `mni_fixed_atlas_path`
- `neighbor_dilation_iter`
- `csf_labels`
- `skull_labels`
- `electrode_csv`
- `electrode_names`
- `eeg_positions_path_template`

### Important HPC environment variables

Useful environment variables in `run_post_processing_batch_env.py`:

- `BATCH_ROOT`
- `BATCH_DATASET_GLOB`
- `BATCH_REPEATS`
- `PIPELINE_FASTSURFER_ROOT`
- `PIPELINE_FASTSURFER_ATLAS_FILENAME`
- `PIPELINE_PLOT_ROI`
- `PIPELINE_PERCENTILE`
- `PIPELINE_OFFTARGET_THRESHOLD`
- `PIPELINE_MNI_BASELINE_ROOT`
- `PIPELINE_MNI_FIXED_ATLAS_PATH`
- `PIPELINE_NEIGHBOR_DILATION_ITER`
- `PIPELINE_CSF_LABELS`
- `PIPELINE_SKULL_LABELS`
- `PIPELINE_ELECTRODE_CSV`
- `PIPELINE_ELECTRODE_NAMES`
- `PIPELINE_EEG_POSITIONS_PATH_TEMPLATE`
- `PIPELINE_POPULATION_ENABLED`
- `PIPELINE_REPEATABILITY_ENABLED`
- `PIPELINE_REPEATABILITY_OUTPUT_DIR`
- `PIPELINE_REPEATABILITY_LOGS_ROOT`
- `PIPELINE_COMPLETE_REPEAT_SUBJECTS_ONLY`

## Quick Start

These examples are meant to show the intended entrypoints, not to lock you into one specific filesystem layout.

Replace the placeholder paths before running them.

### 1. Process one dataset run

This is the best choice when you want to test subject-level processing and optional within-run aggregation on a single dataset such as `Left_Hippocampus_Data_01`.

```bash
python3 post/run_full_post_pipeline.py \
  --root /path/to/Left_Hippocampus_Data_01 \
  --mode single \
  --fastsurfer-root /path/to/subject_atlases \
  --atlas-filename mri/aparc.DKTatlas+aseg.deep.nii.gz \
  --roi Left-Hippocampus \
  --mni-baseline-root /path/to/MNI152_Hippocampus \
  --mni-fixed-atlas-path /path/to/mni_fastsurfer_atlas.nii.gz \
  --electrode-csv /path/to/electrode_centers.csv
```

Use this mode when you want:

- `subject_metrics.json` for each subject
- overlays and mask outputs
- `population_analysis/` for that one dataset run

### 2. Process a full 10-repeat batch

This is the main end-to-end test path for the repeatability study.

```bash
python3 post/run_full_post_pipeline.py \
  --root /path/to/repeat_batch_root \
  --mode batch \
  --dataset-glob '*_Data_*' \
  --repeats 01 02 03 04 05 06 07 08 09 10 \
  --fastsurfer-root /path/to/subject_atlases \
  --atlas-filename mri/aparc.DKTatlas+aseg.deep.nii.gz \
  --mni-baseline-root /path/to/MNI152_Hippocampus \
  --mni-fixed-atlas-path /path/to/mni_fastsurfer_atlas.nii.gz \
  --electrode-csv /path/to/electrode_centers.csv \
  --repeatability-output-dir repeatability_analysis
```

Use this mode when you want:

- every repeat dataset processed
- per-run `population_analysis/` outputs restricted to subjects that complete all selected repeats
- one batch summary JSON
- automatic across-repeat analysis after the batch finishes
- one complete-case subject manifest per ROI

### 3. Run repeatability analysis only

Use this when all subject-level outputs already exist and you only want the cross-repeat statistics and figures.

```bash
python3 post/repeatability/analyze_subject_metrics.py \
  /path/to/repeat_batch_root \
  --roi Left-Hippocampus \
  --output-dir /path/to/repeat_batch_root/repeatability_analysis/left_hippocampus
```

Use this mode when you want:

- repeat-level and experiment-level summary tables
- subject-level mean and SD tables across repeats
- ROI-mask, top-5% mask, overlap-mask, and within-ROI field repeatability metrics
- hotspot localization stability metrics from peak and overlap center-of-mass displacement
- repeatability figures and narrative reports
- complete-case-only outputs by default

## Typical Execution Modes

### 1. Process one dataset run

Use [run_full_post_pipeline.py](./run_full_post_pipeline.py) when you want a single command.

Use [run_post_processing.py](./run_post_processing.py) when you want direct Python-level control.

Typical goals:

- subject-level outputs
- optional within-run population summaries

### 2. Process a full repeat batch

Use [run_full_post_pipeline.py](./run_full_post_pipeline.py) when you want a single command.

Use [run_post_processing_batch.py](./run_post_processing_batch.py) when you want the lower-level batch orchestration directly.

Typical goals:

- every repeat dataset processed
- one batch summary
- optional repeatability analysis after the batch

### 3. Run on HPC

Use [run_post_processing.slurm](../HPC_scripts/run_post_processing.slurm) when:

- you want to process one dataset root on Slurm
- config should live in the Slurm file, not in Python defaults

Use [run_post_processing_batch.slurm](../HPC_scripts/run_post_processing_batch.slurm) when:

- you want to process a repeat batch on Slurm
- the repeatability stage should run automatically after the batch

Use [run_post_processing_batch_env.py](./run_post_processing_batch_env.py) when:

- the batch is launched from Slurm
- config is provided via exported environment variables

### 4. Run repeatability analysis only

Use [repeatability/analyze_subject_metrics.py](./repeatability/analyze_subject_metrics.py) when:

- subject-level `subject_metrics.json` files already exist
- you want only the cross-repeat statistical analysis

## Implementation Notes

### Why the new metrics live in `subject_metrics.json`

Keeping the extended metrics in the same JSON as the overlap metrics makes downstream aggregation much simpler:

- one file per subject per repeat
- one loading path for repeatability analysis
- one compatibility point for future tooling

### Why the extended computations were moved into a shared helper

Without [metric_extensions.py](./metric_extensions.py), the code would have duplicated the same metric logic in:

- subject-level post-processing
- robustness analysis
- downstream flattening code

The helper module reduces divergence and keeps metric definitions consistent.

### Why the repeatability stage still starts from `subject_metrics.json`

This keeps the repeated-run comparison decoupled from the raw NIfTI processing. The repeatability layer should compare standardized outputs, not rerun field extraction logic.

## Common Pitfalls

### Missing MNI baseline

If `mni_baseline_root` is not configured, the absolute-delta baseline fields will be `NaN`.

The baseline root should point to the SimNIBS output root for the MNI run of the current ROI. The extractor accepts either:

- a direct subject-style root with `anat/SimNIBS/ti_brain_only.nii.gz`
- a parent directory containing one baseline subject folder such as `MNI152/anat/SimNIBS/ti_brain_only.nii.gz`

### Missing fixed MNI atlas

If `mni_fixed_atlas_path` is not configured, neighboring-region metrics cannot be defined from the fixed-template design and will be absent.

### Old `subject_metrics.json` files

If a subject already has an older JSON without `extended_metrics`, the single-dataset runner may need `force=True` so the subject is reprocessed.

### ROI naming

The batch runner infers ROI aliases from dataset directory names. If the directory name is not recognized by the ROI registry, the pipeline will stop before analysis.

## Recommended Mental Model

The easiest way to think about the code is:

- `post_process.py` creates subject measurements
- `post_population.py` summarizes one run
- `run_post_processing_batch.py` orchestrates many repeated runs
- `repeatability/analyze_subject_metrics.py` turns repeated measurements into repeatability statistics

If you keep that separation in mind, the rest of the implementation becomes much easier to follow.
