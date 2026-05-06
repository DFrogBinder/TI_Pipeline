# Repeatability Analysis

This folder contains the across-repeats-level metrics layer for repeated TI simulation experiments.

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
  Generates the subject-level metrics layer for one dataset run.

- `post_population.py`
  Generates the population (within run)-level metrics layer for one dataset run.

- `robustness_analysis.py`
  Computes anatomical and targeting robustness measures within one dataset.

- `run_post_processing_batch.py`
  Orchestrates the first two layers across repeated dataset folders, computes
  the per-ROI complete-case cohort shared across all selected repeats, reruns
  the within-run population summaries on that cohort when requested, and then
  launches this third layer.

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
  every selected repeat whose `subject_metrics.json` is marked
  `extended_metrics_meta.status == "complete"`.
- By default, repeat-level descriptive outputs and all experiment-level
  repeatability outputs are restricted to that complete-case cohort.
- In batch mode, a repeat dataset may finish as `partial` if some subjects fail.
  Within-run population summaries are then rerun on the shared complete-case
  cohort after all repeats finish.
- The image-level repeatability layer always uses the complete-case cohort so
  the same subject and ROI support are compared across all repeated runs.
- Image-level mask lookup uses the stored `percentile` field from each
  `subject_metrics.json` entry, so the repeatability layer no longer assumes
  that the high-field masks are always `top95`.

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

## Image-level naming

The image-level repeatability CSV outputs now use generic top-percentile column
names such as:

- `top_percentile_mask_dice`
- `top_percentile_mask_jaccard`
- `top_percentile_mask_dice_mean`
- `top_percentile_mask_jaccard_mean`

This avoids silently mislabeling non-95th-percentile runs as `top95`.
