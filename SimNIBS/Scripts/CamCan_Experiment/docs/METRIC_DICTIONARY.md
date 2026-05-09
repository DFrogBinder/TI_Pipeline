# Post-Processing Metric Dictionary

This dictionary documents how metrics are computed across the post-processing
pipeline. It covers the subject-level outputs, population within-run outputs,
and across-repeat repeatability outputs.

Primary source files:

- `post/post_process.py`
- `post/metric_extensions.py`
- `post/post_population.py`
- `post/repeatability/analyze_subject_metrics.py`
- `utils/ti_utils.py`

## Shared Conventions

- TI field data are loaded as a scalar volume. If the NIfTI has shape
  `(X, Y, Z, 3)`, the scalar field is the Euclidean vector norm across the last
  axis. If the NIfTI is already 3D, values are used directly.
- `finite` means `np.isfinite(ti_data)`.
- `voxel_volume_mm3` is the product of the first three voxel zooms in the TI
  NIfTI header.
- Atlas masks are resampled to the TI grid with nearest-neighbor interpolation.
- World coordinates in millimeters are computed by applying the image affine to
  voxel indices.
- Unless a metric name says otherwise, field intensity values are in the same
  units as the input TI field, usually V/m.
- `IQR` is `Q3 - Q1` using pandas quantiles.
- In population summaries, `std` is the sample standard deviation (`ddof=1`)
  when at least two values are present. `cv` uses population standard deviation
  (`ddof=0`) divided by the mean, matching the existing CSV compatibility
  behavior.
- In repeatability summaries, `std` is sample standard deviation (`ddof=1`) and
  `cv_percent` is `std / mean * 100`.
- Subject-level outputs are considered complete only when
  `subject_metrics.json` has `extended_metrics_meta.status == "complete"`.

## Subject-Level Outputs

Subject-level post-processing writes one output directory per subject under
`<dataset>/<subject>/anat/post/`.

### Per-Voxel CSV Columns

These columns appear in ROI value CSVs, top-percentile value CSVs, and ROI/top
overlap value CSVs written by `post.post_functions.write_csv`.

| Metric | Where | Computation |
| --- | --- | --- |
| `I`, `J`, `K` | `*_values.csv` | Integer voxel indices from `np.argwhere(mask)`. |
| `X_mm`, `Y_mm`, `Z_mm` | `*_values.csv` | World coordinates from the NIfTI affine applied to `I,J,K`. |
| `Value` | `*_values.csv` | TI scalar value at the voxel. |

### Global Top-Percentile Metrics

| Metric | Where | Computation |
| --- | --- | --- |
| `percentile` | `subject_metrics.json` | Configured percentile, for example `95.0`. |
| `percentile_value` | `subject_metrics.json` | `np.nanpercentile(ti_data[finite], percentile)`. |
| `top_percentile_voxels` | `subject_metrics.json` | Count of voxels where `finite and ti_data >= percentile_value`. |
| `voxel_volume_mm3` | `subject_metrics.json` | Product of TI voxel sizes. |
| `efield_top<P>pct_mask` | NIfTI mask | Binary mask where `finite and ti_data >= percentile_value`. |
| `TI_in_Top<P>` | NIfTI field | TI data inside `efield_top<P>pct_mask`, zero elsewhere. |

### ROI Overlap Metrics

These metrics are stored under `rois[ROI]` in `subject_metrics.json`.

| Metric | Computation |
| --- | --- |
| `roi_voxels` | Count of voxels in the ROI mask. |
| `overlap_top_voxels` | Count of voxels in `ROI mask and top_percentile_mask`. |
| `roi_volume_mm3` | `roi_voxels * voxel_volume_mm3`. |
| `overlap_volume_mm3` | `overlap_top_voxels * voxel_volume_mm3`. |
| `overlap_fraction` | `overlap_top_voxels / roi_voxels`; `0.0` when the ROI is empty. |
| `roi_percentile` | Configured within-ROI percentile, usually `region_percentile`. |
| `roi_percentile_value` | `np.nanpercentile(ti_data[ROI and finite], roi_percentile)`, or `NaN` if no finite ROI values exist. |

Supporting ROI NIfTI outputs:

- `atlas_<ROI>_mask.nii.gz`: ROI mask on the TI grid.
- `<ROI>_overlap_top<P>pct_mask.nii.gz`: intersection of the ROI mask and top
  percentile mask.
- `TI_in_<ROI>.nii.gz`: TI field inside the ROI mask, zero elsewhere.
- `TI_in_<ROI>_Top<P>.nii.gz`: TI field inside the ROI/top-percentile overlap,
  zero elsewhere.

### FastSurfer Region Table

`region_stats_fastsurfer.csv` is written when a FastSurfer atlas is available
and region-table output is enabled. For each atlas label with at least
`min_voxels` voxels and at least one finite TI value:

| Metric | Computation |
| --- | --- |
| `label_id` | Integer atlas label. |
| `label_name` | Label name from the FastSurfer DKT label map. |
| `voxels` | Count of voxels in that atlas label mask. |
| `volume_mm3` | `voxels * voxel_volume_mm3`. |
| `mean` | Mean TI value over finite voxels in the label. |
| `median` | Median TI value over finite voxels in the label. |
| `max` | Maximum TI value over finite voxels in the label. |
| `p<P>` | Configured percentile of TI values in the label, for example `p95`. |
| `std` | Population standard deviation of TI values in the label (`np.std`, `ddof=0`). |
| `cv` | `std / mean`, or `NaN` when `mean == 0`. |

### Target ROI Intensity Metrics

These fields are stored in `extended_metrics` in `subject_metrics.json`.

| Metric | Computation |
| --- | --- |
| `roi_peak` | Maximum finite TI value inside the selected target ROI. |
| `roi_mean` | Mean finite TI value inside the selected target ROI. |

### Focality Metrics

Focality is computed over the whole finite TI volume, not just the target ROI.

| Metric | Computation |
| --- | --- |
| `focality_threshold_v_per_m` | Configured focality threshold, `offtarget_threshold`. |
| `focality_voxels_gt_threshold` | Count of voxels where `finite and ti_data > focality_threshold_v_per_m`. |
| `focality_volume_mm3_gt_threshold` | `focality_voxels_gt_threshold * voxel_volume_mm3`. |

### MNI Baseline Comparison Metrics

Baseline comparison requires both `mni_baseline_root` and
`mni_fixed_atlas_path`. The baseline ROI intensity and focality metrics are
computed from the configured MNI baseline TI field using the same formulas as
the subject metrics.

| Metric | Computation |
| --- | --- |
| `mni_baseline_roi_peak` | Baseline ROI peak. |
| `mni_baseline_roi_mean` | Baseline ROI mean. |
| `mni_baseline_focality_voxels_gt_threshold` | Baseline focality voxel count. |
| `mni_baseline_focality_volume_mm3_gt_threshold` | Baseline focality volume. |
| `roi_peak_abs_delta_mni` | `abs(roi_peak - mni_baseline_roi_peak)`. |
| `roi_mean_abs_delta_mni` | `abs(roi_mean - mni_baseline_roi_mean)`. |
| `focality_voxels_abs_delta_mni` | `abs(focality_voxels_gt_threshold - mni_baseline_focality_voxels_gt_threshold)`. |
| `focality_volume_mm3_abs_delta_mni` | `abs(focality_volume_mm3_gt_threshold - mni_baseline_focality_volume_mm3_gt_threshold)`. |

If baseline comparison is not configured, these fields remain in the scaffold
as null/NaN-compatible values and are marked `not_configured` in
`extended_metric_status`.

### Neighbor-Region Metrics

Neighbor regions are defined from a fixed MNI atlas template:

1. Resolve target ROI label IDs from the FastSurfer label map.
2. Build a target ROI mask in the fixed MNI atlas.
3. Dilate that mask by `neighbor_dilation_iter`.
4. Define the border as `dilated_target_mask and not target_mask`.
5. Neighbor labels are nonzero labels in the border, excluding target ROI labels.

The fixed neighbor label list is then evaluated in each subject's FastSurfer
atlas on the subject TI grid.

| Metric | Computation |
| --- | --- |
| `neighbor_template_count` | Number of fixed-template neighbor labels. |
| `neighbors[].label_id` | Neighbor atlas label ID. |
| `neighbors[].label_name` | Neighbor atlas label name. |
| `neighbors[].voxels` | Count of voxels in the subject atlas with that neighbor label. |
| `neighbors[].volume_mm3` | `voxels * voxel_volume_mm3`. |
| `neighbors[].mean` | Mean finite TI value in that neighbor label. |
| `neighbors[].max` | Maximum finite TI value in that neighbor label. |
| `neighbor_mean_of_means` | Mean of all finite `neighbors[].mean` values. |
| `neighbor_max_of_max` | Maximum of all finite `neighbors[].max` values. |
| `neighbor_min_of_max` | Minimum of all finite `neighbors[].max` values. |
| `neighbor_mean__<label_slug>` | Flattened per-neighbor mean for repeatability analysis. |
| `neighbor_peak__<label_slug>` | Flattened per-neighbor peak for repeatability analysis. |
| `neighbor_voxels__<label_slug>` | Flattened per-neighbor voxel count for repeatability analysis. |

When configured, `<roi>_fixed_neighbors.json` stores the `neighbors` rows.
The subject-level post-processing stage also writes visualization artefacts from
the same fixed neighbor label list:

| Output | Meaning |
| --- | --- |
| `<roi>_fixed_neighbor_union_mask.nii.gz` | Binary subject-space union of all fixed-template neighbor labels. |
| `<roi>_fixed_neighbor_categorical_mask.nii.gz` | Categorical subject-space neighbor mask retaining FastSurfer label IDs. |
| `<roi>_fixed_neighbor_visualization.json` | Audit metadata: target ROI IDs, neighbor label IDs/names, dilation setting, and union voxel count. |
| `<roi>_fixed_neighbor_union_overlay.png` | Visual QC overlay: cyan neighbor union on subject anatomy with the target ROI contour in red. Written only when a T1 background is available. |

### ROI Centroid And Anatomy-Distance Metrics

| Metric | Computation |
| --- | --- |
| `roi_centroid_x`, `roi_centroid_y`, `roi_centroid_z` | Mean ROI voxel index converted to world coordinates with the TI affine. |
| `csf_distance_mm` | Distance from the rounded ROI centroid voxel to the nearest configured CSF label voxel, using `distance_transform_edt` with image zooms as sampling. |
| `skull_distance_mm` | Same method as CSF distance, but using configured skull labels. |

`csf_labels` default to `[24]` in the main subject-processing pipeline when
anatomy distances are requested. `skull_labels` must be configured to compute
skull distance.

### Electrode-Distance Metrics

Electrode positions come from either `electrode_csv` with columns
`subject,electrode,x,y,z` or from configured electrode names resolved from
`eeg_positions.csv`.

| Metric | Computation |
| --- | --- |
| `electrode_distances[].electrode` | Electrode name. |
| `electrode_distances[].distance_mm` | Euclidean distance from electrode coordinate to ROI centroid world coordinate. |
| `electrode_distance_count` | Number of electrode distances available. |
| `electrode_distance_mean_mm` | Mean electrode-to-centroid distance. |
| `electrode_distance_min_mm` | Minimum electrode-to-centroid distance. |
| `electrode_distance_max_mm` | Maximum electrode-to-centroid distance. |
| `electrode_distance_mm__<electrode_slug>` | Flattened per-electrode distance for repeatability analysis. |

When configured, `<roi>_electrode_distances.json` stores the per-electrode
rows.

### Extended Metric Status Fields

`subject_metrics.json` also records audit metadata:

| Field | Meaning |
| --- | --- |
| `extended_metric_status` | Per-field status: `ok`, `not_configured`, `error`, or `pending` before finalization. |
| `extended_metric_messages` | Per-field diagnostic messages. |
| `extended_metrics_meta.schema_version` | Extended metric schema version. |
| `extended_metrics_meta.status` | `complete` if no extended metric field has status `error`; otherwise `partial`. |
| `extended_metrics_meta.config_fingerprint` | Hash of the relevant metric configuration and input paths. |
| `extended_metrics_meta.group_statuses` | Status by metric group, such as `roi_intensity`, `baseline`, or `neighbors`. |
| `extended_metrics_meta.group_messages` | Diagnostic messages by metric group. |

## Population Within-Run Outputs

Population within-run outputs are written under
`<dataset>/population_analysis/` by `post/post_population.py`.

### Cohort And Raw Tables

| Output | Metric | Computation |
| --- | --- | --- |
| `population_cohort_manifest.csv` | `subject` | Subject ID considered by the population loader. |
| `population_cohort_manifest.csv` | `has_region_table` | Whether the subject has the configured region table file. |
| `population_cohort_manifest.csv` | `has_complete_subject_metrics` | Whether `subject_metrics.json` exists and has `extended_metrics_meta.status == "complete"`. |
| `population_cohort_manifest.csv` | `included` | `has_region_table and has_complete_subject_metrics`. |
| `all_region_values.csv` | all columns | Concatenation of included subjects' `region_stats_fastsurfer.csv` tables, with a leading `subject` column. |
| `subject_metric_values.csv` | all columns | Flattened included `subject_metrics.json` payloads for the selected target ROI. |
| `subject_neighbor_metrics.csv` | all columns | One row per subject and neighbor label from `extended_metrics.neighbors`. |

Only included subjects contribute to downstream summaries.

### Population Region Summary

`population_region_summary.csv` groups `all_region_values.csv` by
`label_id,label_name`.

| Metric | Computation |
| --- | --- |
| `label_id`, `label_name` | Atlas label identity. |
| `subjects` | Number of unique included subjects contributing rows for the label. |
| `mean_of_mean` | Mean across subjects of region-table `mean`. |
| `median_of_mean` | Median across subjects of region-table `mean`. |
| `iqr_mean` | IQR across subjects of region-table `mean`. |
| `cv_mean` | Population CV across subjects of region-table `mean`. |
| `mean_peak` | Mean across subjects of region-table `max`. |
| `median_peak` | Median across subjects of region-table `max`. |
| `iqr_peak` | IQR across subjects of region-table `max`. |
| `cv_peak` | Population CV across subjects of region-table `max`. |
| `min_peak` | Minimum region-table `max` across subjects. |
| `max_peak` | Maximum region-table `max` across subjects. |
| `frac_peak_gt_thr` | Fraction of rows where region-table `max > peak_threshold`. |
| `mean_volume_mm3` | Mean region volume across subjects. |
| `median_volume_mm3` | Median region volume across subjects. |

Documentation that refers to a generic population `CV` maps to the explicit
fields `cv_mean` and `cv_peak`.

### Volume-Intensity Correlations

| Output | Metric | Computation |
| --- | --- | --- |
| `volume_intensity_correlation.csv` | `metric` | Either `mean` or `max`. |
| `volume_intensity_correlation.csv` | `pearson_r` | Pearson correlation between `volume_mm3` and the selected metric across all pooled atlas-region rows. This legacy file is intentionally pooled. |
| `regional_volume_intensity_correlation.csv` | `label_id`, `label_name` | Atlas region identity. |
| `regional_volume_intensity_correlation.csv` | `subjects` | Number of unique subjects with finite `volume_mm3` and metric values for the region. |
| `regional_volume_intensity_correlation.csv` | `metric` | Either `mean` or `max`. |
| `regional_volume_intensity_correlation.csv` | `pearson_r` | Per-region Pearson correlation between `volume_mm3` and the selected metric; only written for region/metric pairs with at least three subjects. |

### Subject Robustness Table

`subject_robustness.csv` starts from rows in `all_region_values.csv` whose
`label_name` matches the configured target ROI, case-insensitively. It then
merges selected flattened subject metrics.

| Metric | Computation |
| --- | --- |
| region columns | Target ROI row from `all_region_values.csv`. |
| `drop_vs_template` | `(template_peak - subject_target_peak) / template_peak`, where `subject_target_peak` is the target region `max`; present only when `template_region_csv` contains the target ROI. |
| `roi_overlap_fraction` | Flattened target ROI `overlap_fraction`. |
| `focality_voxels_gt_threshold` | Flattened subject focality voxel count. |
| `focality_volume_mm3_gt_threshold` | Flattened subject focality volume. |
| `roi_peak_abs_delta_mni`, `roi_mean_abs_delta_mni` | Flattened MNI baseline absolute deltas. |
| `csf_distance_mm`, `skull_distance_mm` | Flattened anatomy-distance covariates. |
| `electrode_distance_*_mm` | Flattened electrode-distance summaries. |
| `neighbor_mean_of_means`, `neighbor_max_of_max` | Flattened neighbor summary metrics. |

### Population Subject Metric Summary

`population_subject_metric_summary.csv` summarizes each numeric column in
`subject_metric_values.csv`, excluding `subject` and `target_roi`.

| Metric | Computation |
| --- | --- |
| `metric` | Flattened subject metric column name. |
| `subjects` | Count of finite numeric values. |
| `mean` | Mean across subjects. |
| `median` | Median across subjects. |
| `iqr` | IQR across subjects. |
| `std` | Sample standard deviation across subjects (`ddof=1`; `0.0` for one value). |
| `cv` | Population standard deviation (`ddof=0`) divided by the mean. |
| `min` | Minimum across subjects. |
| `max` | Maximum across subjects. |

### Population Neighbor Summary

`population_neighbor_summary.csv` groups `subject_neighbor_metrics.csv` by
`label_id,label_name`.

| Metric | Computation |
| --- | --- |
| `subjects` | Number of unique subjects contributing the neighbor label. |
| `mean_of_mean` | Mean across subjects of neighbor `mean`. |
| `median_of_mean` | Median across subjects of neighbor `mean`. |
| `iqr_mean` | IQR across subjects of neighbor `mean`. |
| `std_mean` | Sample standard deviation across subjects of neighbor `mean`. |
| `cv_mean` | Population CV across subjects of neighbor `mean`. |
| `min_mean` | Minimum neighbor `mean` across subjects. |
| `max_mean` | Maximum neighbor `mean` across subjects. |
| `mean_peak` | Mean across subjects of neighbor `max`. |
| `median_peak` | Median across subjects of neighbor `max`. |
| `iqr_peak` | IQR across subjects of neighbor `max`. |
| `std_peak` | Sample standard deviation across subjects of neighbor `max`. |
| `cv_peak` | Population CV across subjects of neighbor `max`. |
| `min_peak` | Minimum neighbor `max` across subjects. |
| `max_peak` | Maximum neighbor `max` across subjects. |
| `mean_volume_mm3` | Mean neighbor `volume_mm3` across subjects. |

### Anatomy Correlations And Worst Cases

| Output | Metric | Computation |
| --- | --- | --- |
| `population_anatomy_correlations.csv` | `performance_metric` | One of `roi_peak`, `roi_mean`, baseline delta metrics, or focality metrics when present. |
| `population_anatomy_correlations.csv` | `anatomy_metric` | One of CSF, skull, or electrode-distance metrics when present. |
| `population_anatomy_correlations.csv` | `subjects` | Number of complete subject pairs used. |
| `population_anatomy_correlations.csv` | `pearson_r` | Pearson correlation between the performance and anatomy metric; only emitted for pairs with at least three complete values. |
| `worst_case_subjects.csv` | `subject`, `roi_peak` | Lowest 10 subjects by flattened `roi_peak`, ascending. |

## Across-Repeat Repeatability Outputs

Across-repeat outputs are written by
`post/repeatability/analyze_subject_metrics.py` under
`<dataset_root>/subject_metrics_analysis/` unless an output directory is
provided.

### Across-Repeat Metric Checklist

Scalar and subject-level repeatability:

- per-run `n`, mean, SD, SEM, and 95% CI
- median, IQR, min, max, and CV
- mean and max absolute pairwise difference
- SD of run means and CV of run means
- subject repeat mean, SD, and CV
- subject-specific drift, SD ratio versus pooled repeatability, and top
  unstable subject rankings

Image-level repeatability:

- pairwise Dice and Jaccard for ROI masks, configured top-percentile masks, and
  overlap masks
- within-ROI voxelwise field correlations across repeats
- voxel SD and CV summaries inside the ROI
- ROI mean, ROI P95, and ROI peak field repeatability
- peak displacement and overlap-mask center-of-mass displacement

For the default `percentile=95`, the configured top-percentile mask is the top
5% field mask. Output fields use `top_percentile_*` naming so runs with a
different configured percentile remain correctly labeled.

### Loaded Scalar Metric Table

| Output | Metric | Computation |
| --- | --- | --- |
| `subject_metrics_long.csv` | `run_label` | Repeat dataset folder label inferred from the metric file path. |
| `subject_metrics_long.csv` | `repeat_id` | Numeric repeat ID inferred from `run_label`. |
| `subject_metrics_long.csv` | `run_short` | `R<repeat_id>` with two-digit padding. |
| `subject_metrics_long.csv` | `roi` | ROI key loaded from `subject_metrics.json`. |
| `subject_metrics_long.csv` | `source_path` | Relative path to the source `subject_metrics.json`. |
| `subject_metrics_long.csv` | metric columns | Flattened ROI overlap fields and numeric `extended_metrics` fields. Incomplete subject payloads are skipped. |

`numeric_metrics()` excludes identifiers and static/configuration fields such as
`focality_threshold_v_per_m`, MNI baseline constants,
`neighbor_template_count`, and `electrode_distance_count`.

### Coverage Metrics

`run_subject_coverage.csv` reports cohort coverage.

| Metric | Computation |
| --- | --- |
| `n_subjects` | Number of unique subjects in a repeat. |
| `n_missing_from_union` | Size of all-subject union minus subjects present in the repeat. |
| `n_missing_from_complete_case` | Number of complete-case subjects absent from the repeat. |

### Shared Summary Statistics

`summarise_series()` is used throughout repeatability outputs.

| Metric | Computation |
| --- | --- |
| `n` | Count of finite numeric values. |
| `mean` | Mean of finite values. |
| `std` | Sample standard deviation (`ddof=1`; `0.0` for one value). |
| `sem` | `std / sqrt(n)` when `n > 1`, otherwise `0.0`. |
| `ci95_low`, `ci95_high` | Two-sided t interval around the mean. |
| `median` | Median of finite values. |
| `q1`, `q3` | 25th and 75th percentiles. |
| `iqr` | `q3 - q1`. |
| `min`, `max` | Minimum and maximum finite values. |
| `cv_percent` | `std / mean * 100`, or `NaN` when mean is zero. |

### Repeat-Level Population Statistics

`repeat_level_population_statistics.csv` is computed on the selected analysis
frame. By default, that is the complete-case cohort. If incomplete subjects are
allowed, it includes all available complete subject JSONs per repeat.

`repeat_level_population_statistics_complete_subjects.csv` is always computed
on the complete-case cohort.

For each repeat and each numeric metric:

- identifiers: `repeat_id`, `run_label`, `run_short`, `metric`,
  `metric_label`
- summary fields from `summarise_series()`

### Pairwise Run Differences

`pairwise_run_differences.csv` compares every pair of repeats for every numeric
metric on subjects that have both repeats.

| Metric | Computation |
| --- | --- |
| `repeat_a`, `repeat_b`, `run_a`, `run_b`, `comparison` | Pair identity. |
| summary fields | `summarise_series(repeat_b_value - repeat_a_value)` across subjects. |

### Within-Subject Repeatability

`within_subject_repeatability.csv` pivots each metric to a
subject-by-repeat matrix and drops subjects missing any repeat for that metric.

| Metric | Computation |
| --- | --- |
| `n_subjects` | Number of complete subjects for the metric. |
| `subject_mean_mean` | Mean across subjects of each subject's mean across repeats. |
| `subject_mean_std` | Sample SD across subject-level repeat means. |
| `subject_mean_median` | Median of subject-level repeat means. |
| `subject_within_run_sd_mean` | Mean across subjects of each subject's repeat SD. |
| `subject_within_run_sd_median` | Median across subjects of each subject's repeat SD. |
| `subject_within_run_cv_percent_mean` | Mean across subjects of each subject's repeat CV percent. |
| `subject_within_run_cv_percent_median` | Median across subjects of each subject's repeat CV percent. |

### Experiment-Level Population Statistics

`experiment_level_population_statistics.csv` summarizes run-to-run and
subject-to-subject variation for each numeric metric.

| Metric | Computation |
| --- | --- |
| `n_runs` | Number of repeat-level rows for the metric. |
| `n_complete_subjects` | Number of complete subjects in the metric pivot. |
| `mean_of_run_means` | Mean of repeat-level means. |
| `sd_of_run_means` | Sample SD of repeat-level means. |
| `ci95_low_of_run_means`, `ci95_high_of_run_means` | t interval around repeat-level means. |
| `min_run_mean`, `max_run_mean`, `range_run_mean` | Minimum, maximum, and range of repeat-level means. |
| `cv_percent_run_means` | CV percent of repeat-level means. |
| `mean_within_run_sd` | Mean of repeat-level subject SD values. |
| `mean_run_subject_count` | Mean repeat-level `n`. |
| `mean_abs_pairwise_diff` | Mean absolute value of pairwise difference means. |
| `max_abs_pairwise_diff` | Maximum absolute value of pairwise difference means. |
| `grand_mean_complete_case` | Mean of all values in the complete subject-by-repeat pivot. |
| `between_subject_sd` | Square root of ANOVA subject variance component. |
| `run_effect_sd` | Square root of ANOVA run variance component. |
| `pooled_within_subject_sd` | Square root of ANOVA residual variance component. |
| `standard_error_of_measurement` | Equal to `pooled_within_subject_sd`. |
| `repeatability_coefficient` | `1.96 * sqrt(2) * pooled_within_subject_sd`. |
| `pooled_within_subject_cv_percent` | `pooled_within_subject_sd / grand_mean_complete_case * 100`. |
| `icc_absolute_agreement` | Two-way absolute agreement ICC from subject, run, and residual mean squares. |
| `mean_pairwise_correlation` | Mean upper-triangle Pearson correlation between repeat columns. |
| `drift_slope_per_repeat` | Linear regression slope of run mean versus repeat ID. |
| `drift_slope_percent_per_repeat` | `drift_slope_per_repeat / grand_mean_complete_case * 100`. |
| `drift_pvalue` | p-value for the run-mean drift slope. |
| `drift_r_squared` | Squared correlation from the drift regression. |
| `subject_variance_fraction_percent` | Subject variance component divided by total variance, times 100. |
| `run_variance_fraction_percent` | Run variance component divided by total variance, times 100. |
| `residual_variance_fraction_percent` | Residual variance component divided by total variance, times 100. |
| `friedman_statistic`, `friedman_pvalue` | Friedman test across repeats when at least three repeats and within-subject changes exist. |
| `kendall_w` | `friedman_statistic / (n_subjects * (n_runs - 1))`. |
| `subject_within_run_*` | Values copied from `within_subject_repeatability.csv` for the metric. |

The ANOVA components use:

- `ss_subject = n_runs * sum((subject_mean - grand_mean)^2)`
- `ss_run = n_subjects * sum((run_mean - grand_mean)^2)`
- `ss_residual = sum((value - subject_mean - run_mean + grand_mean)^2)`
- `subject_variance = max((ms_subject - ms_residual) / n_runs, 0)`
- `run_variance = max((ms_run - ms_residual) / n_subjects, 0)`
- `residual_variance = max(ms_residual, 0)`

### Variation Analysis Metrics

`variation_analysis_metrics.csv` is a concise subset of
`experiment_level_population_statistics.csv` for the primary plot metrics:

- `percentile_value`
- `top_percentile_voxels`
- `overlap_top_voxels`
- `overlap_fraction`

It retains the experiment-level repeatability, drift, correlation, and variance
fraction fields most useful for quick review.

### Subject Repeat Means And SDs

| Output | Computation |
| --- | --- |
| `subject_repeat_metric_means.csv` | For each complete-case subject, mean of each numeric metric across repeats. |
| `subject_repeat_metric_sds.csv` | For each complete-case subject, sample SD of each numeric metric across repeats. |

### Subject-Level Variation

`subject_level_variation.csv` computes per-subject repeat variation for every
numeric metric.

| Metric | Computation |
| --- | --- |
| `n_runs` | Number of repeated values for the subject and metric. |
| `mean`, `std`, `median`, `min`, `max`, `range` | Basic within-subject summaries across repeats. |
| `cv_percent` | `std / mean * 100`. |
| `mad` | Median absolute deviation from the subject median. |
| `mean_abs_pairwise_diff` | Mean absolute difference over all repeat pairs for the subject. |
| `max_abs_pairwise_diff` | Maximum absolute pairwise repeat difference for the subject. |
| `pooled_within_subject_sd` | Experiment-level pooled within-subject SD for the metric. |
| `sd_vs_pooled_repeatability` | Subject `std / pooled_within_subject_sd`. |
| `drift_slope_per_repeat` | Linear regression slope of the subject's metric values versus repeat number. |
| `drift_pvalue` | p-value for the subject drift slope. |
| `drift_r_squared` | Squared correlation from the subject drift regression. |

`subject_level_variation_summary.csv` aggregates the subject-level variation
table per metric:

- mean, median, 95th percentile, and maximum of subject `cv_percent`
- mean, median, 95th percentile, and maximum of `sd_vs_pooled_repeatability`
- mean, median, and maximum pairwise absolute-difference summaries
- count of subjects with `drift_pvalue < 0.05`
- median and 95th percentile of absolute drift slopes

`subject_level_top_variable_subjects.csv` ranks the primary plot metrics by:

- `relative_variation_cv`: highest `cv_percent`
- `absolute_variation_sd_ratio`: highest `sd_vs_pooled_repeatability`

`subject_cross_metric_instability.csv` pivots the primary plot metrics to one
row per subject and reports:

- `<metric>_sd_ratio` for each primary metric
- `mean_sd_ratio_across_metrics`
- `max_sd_ratio_across_metrics`

### Image-Level Repeatability

Image-level repeatability uses saved NIfTI outputs for complete-case subjects.
The image mask filenames are resolved from the stored subject percentile, so
non-95th-percentile runs are not mislabeled.

`image_repeatability_run_level.csv` has one row per subject/run.

| Metric | Computation |
| --- | --- |
| `same_grid_as_reference` | Whether each image header matches the first run's shape and affine; mismatches are reported as issues. |
| `same_roi_mask_as_reference` | Whether the run ROI mask equals the first run ROI mask. |
| `roi_mask_voxels` | Count of ROI mask voxels. |
| `roi_field_finite_voxels` | Count of finite ROI field values inside the reference ROI support. |
| `top_percentile_mask_voxels` | Count of top-percentile mask voxels. |
| `overlap_mask_voxels` | Count of overlap mask voxels. |
| `roi_mean_field` | Mean of ROI field values inside reference ROI support. |
| `roi_p95_field` | 95th percentile of ROI field values inside reference ROI support. |
| `roi_peak_field` | Maximum ROI field value inside reference ROI support. |
| `peak_x_mm`, `peak_y_mm`, `peak_z_mm` | World coordinate of maximum finite ROI field value. |
| `overlap_com_x_mm`, `overlap_com_y_mm`, `overlap_com_z_mm` | Center of mass of overlap-mask voxel coordinates in world space, or NaN when overlap is empty. |

`image_repeatability_pairwise_subject_run_pairs.csv` has one row per
subject/run pair.

| Metric | Computation |
| --- | --- |
| `roi_mask_dice` | `2 * |A intersect B| / (|A| + |B|)` for ROI masks. |
| `roi_mask_jaccard` | `|A intersect B| / |A union B|` for ROI masks. |
| `top_percentile_mask_dice`, `top_percentile_mask_jaccard` | Same formulas for top-percentile masks. |
| `overlap_mask_dice`, `overlap_mask_jaccard` | Same formulas for overlap masks. |
| `within_roi_field_correlation` | Pearson correlation between paired ROI field vectors over common finite support. |
| `peak_displacement_mm` | Euclidean distance between pairwise peak coordinates. |
| `overlap_com_displacement_mm` | Euclidean distance between pairwise overlap center-of-mass coordinates. |
| `roi_mean_field_abs_diff` | Absolute difference in pairwise ROI mean field. |
| `roi_p95_field_abs_diff` | Absolute difference in pairwise ROI p95 field. |
| `roi_peak_field_abs_diff` | Absolute difference in pairwise ROI peak field. |

`image_repeatability_subject_level.csv` summarizes image metrics per subject.

| Metric | Computation |
| --- | --- |
| `grid_consistent_all_runs` | True if all runs passed grid consistency checks. |
| `roi_mask_identical_all_runs` | True if all ROI masks equal the first run ROI mask. |
| `reference_roi_voxels` | Voxels in first-run ROI mask. |
| `reference_roi_voxels_common_finite_support` | ROI voxels finite in every run. |
| `reference_roi_voxels_excluded_nonfinite` | Reference ROI voxels minus common finite support. |
| `within_roi_voxel_sd_*` | Mean, median, p95, and max of per-voxel SD across repeats. |
| `within_roi_voxel_cv_percent_*` | Mean, median, p95, and max of per-voxel CV percent across repeats. |
| `<pairwise_metric>_mean`, `_median`, `_min`, `_max` | Subject-level summaries over all pairwise run comparisons. |
| `roi_mean_field_*`, `roi_p95_field_*`, `roi_peak_field_*` | Repeat-vector summaries: mean, SD, CV percent, min, max, range, mean absolute pairwise difference, and max absolute pairwise difference. |

`image_repeatability_pairwise_run_summary.csv` groups pairwise image metrics by
run pair and applies `summarise_series()` across subjects.

`image_repeatability_cohort_summary.csv` summarizes each numeric column in
`image_repeatability_subject_level.csv` across subjects using
`summarise_series()` plus a 95th percentile.

`image_repeatability_issues.csv` lists subjects for which image-level
repeatability could not be computed, with `issue_type` and `details`.

### Optional Log-Audit Outputs

When execution logs are provided, the repeatability script can emit log-audit
tables. These are operational quality-control metrics rather than TI field
metrics.

| Output | Meaning |
| --- | --- |
| `log_subject_run_details.csv` | Per subject/run log status details. |
| `log_failure_summary_by_category.csv` | Counts grouped by failure category. |
| `log_failure_category_by_run.csv` | Failure-category counts per run. |
| `log_failure_stage_by_run.csv` | Failure-stage counts per run. |
| `log_run_transition_summary.csv` | Paired transition summaries across runs. |

### Narrative And Figure Outputs

These files summarize the CSV metrics for review. They do not introduce new TI
field formulas, but they are part of the across-repeat layer's output contract.

| Output | Meaning |
| --- | --- |
| `analysis_summary.md` | Narrative summary of cohort size, key repeatability statistics, output inventory, and image-repeatability availability. |
| `analysis_methodology.md` | Methodology document generated from the current run settings and computed fields. |
| `results_interpretation.md` | Interpretation of repeat-level and experiment-level scalar repeatability results. |
| `subject_level_variation_report.md` | Narrative report for subject-level repeat variation and top-variable subjects. |
| `failure_report.md` | Optional narrative report for log-derived failures when logs are provided. |
| `image_repeatability_methodology.md` | Methodology document for image-level repeatability when image analysis is enabled. |
| `image_repeatability_report.md` | Narrative image-repeatability report when image analysis is enabled. |
| `figures/01_subject_coverage.png` | Subject coverage figure. |
| `figures/02_repeat_level_distributions.png` | Repeat-level metric distribution figure. |
| `figures/03_repeat_level_mean_ci.png` | Repeat-level mean and CI figure. |
| `figures/04_pairwise_run_differences.png` | Pairwise repeat-difference heatmap. |
| `figures/05_variation_summary.png` | Experiment-level variation summary figure. |
| `figures/06_subject_variation_summary.png` | Subject-level repeat variation figure. |
| `figures/07_failure_summary.png` | Optional failure summary figure when logs are provided. |
| `figures/08_image_repeatability_summary.png` | Image-level repeatability summary figure when image analysis succeeds. |

## Legacy Standalone Robustness Analysis

`post/robustness_analysis.py` is a standalone compatibility script. It computes
many of the same extended subject metrics through `compute_extended_subject_metrics`
and writes:

- `per_subject_metrics.csv`: selected ROI intensity, baseline delta, focality,
  centroid, anatomy-distance, and electrode-distance fields per subject.
- `subjects/<subject>_neighbors.csv`: raw neighbor rows when available.
- `subjects/<subject>_electrode_distances.csv`: raw electrode rows when
  available.
- `population_summary.csv`: mean, median, IQR, CV, min, and max for
  `roi_peak`, `roi_mean`, and `focality_voxels_gt_threshold`.
- `worst_case_subjects.csv`: lowest 10 finite `roi_peak` values.

The main pipeline path is `post_process.py` plus `post_population.py`; use the
standalone robustness script only when that older workflow is explicitly needed.
