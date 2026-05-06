# Repeatability Analysis

This folder contains the across-repeats-level metrics layer for repeated TI simulation experiments.
For exact metric formulas and output field definitions, see
`../../docs/METRIC_DICTIONARY.md`.

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

## Metric checklist

### Scalar and subject-level repeatability

The across-repeat scalar layer computes:

- per-run `n`, mean, SD, SEM, and 95% CI
- median, IQR, min, max, and CV
- mean and max absolute pairwise difference between repeat runs
- SD of run means and CV of run means
- subject repeat mean, SD, and CV
- subject-specific drift slope, drift p-value, drift R-squared, SD ratio versus pooled repeatability, and top unstable subject rankings

These are mainly written to:

- `repeat_level_population_statistics.csv`
- `repeat_level_population_statistics_complete_subjects.csv`
- `experiment_level_population_statistics.csv`
- `variation_analysis_metrics.csv`
- `pairwise_run_differences.csv`
- `within_subject_repeatability.csv`
- `subject_repeat_metric_means.csv`
- `subject_repeat_metric_sds.csv`
- `subject_level_variation.csv`
- `subject_level_variation_summary.csv`
- `subject_level_top_variable_subjects.csv`
- `subject_cross_metric_instability.csv`

### Image-level repeatability

The image-level layer computes:

- pairwise Dice and Jaccard for ROI masks, configured top-percentile masks, and overlap masks
- within-ROI voxelwise field correlations across repeat pairs
- voxel SD and CV summaries inside the ROI
- ROI mean, ROI P95, and ROI peak field repeatability
- peak displacement and overlap-mask center-of-mass displacement

For the normal `percentile=95` configuration, the configured top-percentile
mask is the top 5% field mask. The CSV field names use `top_percentile_*`
rather than `top5_*` so non-95th-percentile analyses are not mislabeled.

These are mainly written to:

- `image_repeatability_run_level.csv`
- `image_repeatability_pairwise_subject_run_pairs.csv`
- `image_repeatability_subject_level.csv`
- `image_repeatability_pairwise_run_summary.csv`
- `image_repeatability_cohort_summary.csv`
- `image_repeatability_issues.csv`

## Written outputs and reports

### Scalar CSV tables

- `subject_metrics_long.csv`
  Flattened subject-run records loaded from complete `subject_metrics.json`
  files. This is the base table for scalar repeatability.
- `run_subject_coverage.csv`
  Per-repeat subject availability, including missing-from-union and
  missing-from-complete-case counts.
- `repeat_level_population_statistics.csv`
  Per-repeat descriptive statistics for every scalar metric in the analysis
  cohort.
- `repeat_level_population_statistics_complete_subjects.csv`
  Per-repeat descriptive statistics restricted to the complete-case cohort.
- `experiment_level_population_statistics.csv`
  Across-repeat stability, drift, ICC, repeatability coefficient, and variance
  decomposition metrics.
- `variation_analysis_metrics.csv`
  A compact review table for the primary endpoints:
  `percentile_value`, `top_percentile_voxels`, `overlap_top_voxels`, and
  `overlap_fraction`.
- `pairwise_run_differences.csv`
  Paired run-to-run difference summaries for each metric and repeat pair.
- `within_subject_repeatability.csv`
  Cohort summaries of subject-level repeat means, within-subject SD, and
  within-subject CV.

### Subject-repeat mean/SD and instability tables

- `subject_repeat_metric_means.csv`
  One row per complete-case subject with the mean of each scalar metric across
  repeats.
- `subject_repeat_metric_sds.csv`
  One row per complete-case subject with the sample SD of each scalar metric
  across repeats.
- `subject_level_variation.csv`
  Per subject and metric: repeat mean, SD, CV, median, MAD, range, pairwise
  absolute differences, drift, and SD ratio versus pooled repeatability.
- `subject_level_variation_summary.csv`
  Per metric summaries of subject CV, SD ratio, pairwise differences, and
  drift.
- `subject_level_top_variable_subjects.csv`
  Top unstable subjects for primary metrics, ranked by relative CV and by SD
  ratio.
- `subject_cross_metric_instability.csv`
  Per-subject instability across the primary metrics.

### Image repeatability tables

- `image_repeatability_run_level.csv`
  One row per subject/run with mask voxel counts, ROI field mean/P95/peak, peak
  location, and overlap-mask center of mass.
- `image_repeatability_pairwise_subject_run_pairs.csv`
  One row per subject/run pair with Dice, Jaccard, within-ROI field
  correlation, peak displacement, COM displacement, and ROI field absolute
  differences.
- `image_repeatability_subject_level.csv`
  One row per subject summarizing pairwise image metrics, voxelwise ROI field
  SD/CV, and ROI mean/P95/peak repeat-vector stability.
- `image_repeatability_pairwise_run_summary.csv`
  Pairwise image metrics summarized across subjects by run pair.
- `image_repeatability_cohort_summary.csv`
  Cohort-level summaries of subject image-repeatability metrics.
- `image_repeatability_issues.csv`
  Subjects for which image-level repeatability could not be computed, with
  failure type and details.
- `image_repeatability_methodology.md`
  Methodology notes specific to the image-level analysis.
- `image_repeatability_report.md`
  Narrative interpretation of image-level repeatability.

### Narrative reports

- `analysis_summary.md`
  High-level summary of the repeatability run, cohort sizes, key scalar
  results, output inventory, and image-repeatability status.
- `analysis_methodology.md`
  Detailed methodology for scalar repeatability, complete-case logic, drift,
  variance decomposition, and optional image/log analysis.
- `results_interpretation.md`
  Narrative interpretation of scalar repeatability, run stability, drift, and
  variance fractions.
- `subject_level_variation_report.md`
  Narrative interpretation of subject-specific instability and top-variable
  subjects.
- `failure_report.md`
  Optional log-failure interpretation when `--logs-root` is provided.

### Figures 01-08

- `figures/01_subject_coverage.png`
  Subject availability and complete-case reference.
- `figures/02_repeat_level_distributions.png`
  Per-repeat distributions for primary scalar metrics.
- `figures/03_repeat_level_mean_ci.png`
  Repeat-level means and confidence intervals.
- `figures/04_pairwise_run_differences.png`
  Pairwise repeat-difference heatmap.
- `figures/05_variation_summary.png`
  Experiment-level variation and repeatability summary.
- `figures/06_subject_variation_summary.png`
  Subject-level instability summary.
- `figures/07_failure_summary.png`
  Optional log-failure summary when `--logs-root` is provided.
- `figures/08_image_repeatability_summary.png`
  Optional image-repeatability summary when image analysis succeeds.

Optional log-audit outputs when `--logs-root` is provided:

- `log_subject_run_details.csv`
- `log_failure_summary_by_category.csv`
- `log_failure_category_by_run.csv`
- `log_failure_stage_by_run.csv`
- `log_run_transition_summary.csv`
- `failure_report.md`

## Image-level naming

The image-level repeatability CSV outputs now use generic top-percentile column
names such as:

- `top_percentile_mask_dice`
- `top_percentile_mask_jaccard`
- `top_percentile_mask_dice_mean`
- `top_percentile_mask_jaccard_mean`

This avoids silently mislabeling non-95th-percentile runs as `top95`.
