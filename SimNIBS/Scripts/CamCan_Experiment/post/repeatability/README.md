# Repeatability Analysis

This folder contains the across-repeat analysis layer for repeated TI simulation experiments.

By default, this layer now analyses only the complete-case cohort: subjects that
have valid post-processing outputs in every selected repeat for the ROI being
analysed.

## What belongs here

- `analyze_subject_metrics.py`
  Compares `subject_metrics.json` outputs across repeated dataset runs such as
  `Left_Hippocampus_Data_01` to `Left_Hippocampus_Data_10`.
  It performs repeat-level summaries, experiment-level repeatability analysis,
  subject-level variation analysis, image-level repeatability analysis on the
  saved NIfTI masks and ROI field volumes, figure generation, optional
  log-based failure auditing, and complete-case cohort filtering.

## How this differs from the rest of `post/`

- `post_process.py`
  Generates per-subject post-processing outputs for one dataset run.

- `post_population.py`
  Aggregates subjects within one dataset run.

- `robustness_analysis.py`
  Computes anatomical and targeting robustness measures within one dataset.

- `run_post_processing_batch.py`
  Runs the post-processing pipeline across repeated dataset folders, computes the
  per-ROI complete-case cohort shared across all selected repeats, reruns
  within-run population summaries on that cohort, and then launches the
  repeatability analysis.

## Recommended workflow

1. Run `run_post_processing_batch.py` or the standard `post` pipeline to generate
   `subject_metrics.json` for each repeat dataset.
2. Run `repeatability/analyze_subject_metrics.py` on the repeated-dataset root or
   on a single repeat-batch dataset root that contains all repeat folders.
3. Review the complete-case subject manifest written for the ROI.
4. Review outputs under `<dataset_root>/subject_metrics_analysis`.

## Cohort logic

- One row is loaded for each subject in each repeat from `subject_metrics.json`.
- A complete-case cohort is built from the intersection of subjects present in
  every selected repeat.
- By default, repeat-level descriptive outputs and all experiment-level
  repeatability outputs are restricted to that complete-case cohort.
- In batch mode, within-run population summaries are rerun on that same
  complete-case cohort after all repeats finish.
- The image-level repeatability layer always uses the complete-case cohort so
  the same subject and ROI support are compared across all repeated runs.

The default can be relaxed only when explicitly requested:

- batch orchestration: `--allow-incomplete-repeat-subjects`
- standalone repeatability script: `--allow-incomplete-subjects`
- HPC/env mode: `PIPELINE_COMPLETE_REPEAT_SUBJECTS_ONLY=0`

## Typical use

```bash
python3 repeatability/analyze_subject_metrics.py /path/to/Left_Hippocampus_Post_Data
```

To include incomplete subjects in the descriptive repeat-level outputs:

```bash
python3 repeatability/analyze_subject_metrics.py /path/to/Left_Hippocampus_Post_Data \
  --allow-incomplete-subjects
```

If matching execution logs exist, the script can also audit failure patterns and
paired run transitions.

To disable the image-level layer and run only the scalar `subject_metrics.json`
analysis:

```bash
python3 repeatability/analyze_subject_metrics.py /path/to/Left_Hippocampus_Post_Data \
  --skip-image-repeatability
```
