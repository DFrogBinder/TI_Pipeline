# Whole-Brain Coverage >= 0.2 V/m Design

## Goal

Add a subject-level post-processing metric that reports the percentage of finite whole-brain TI voxels whose field is at least `0.2 V/m`, and add a downstream repeat-level aggregation that summarizes this metric across the 10 repeats for each subject and ROI.

## Semantics

- The cutoff is inclusive: `field >= 0.2`.
- The denominator is the same whole-brain denominator used by existing focality metrics: all finite TI voxels.
- The metric is fixed to `0.2 V/m` for this analysis, not dependent on a caller-provided threshold.
- Existing configurable focality metrics should also use inclusive cutoff semantics when the configured threshold is `0.2`.

## Subject-Level Output

Add a new extended metric family for fixed whole-brain coverage. The subject payload should include scalar fields such as:

- `whole_brain_coverage_threshold_v_per_m`
- `whole_brain_coverage_voxels_ge_threshold`
- `whole_brain_coverage_volume_mm3_ge_threshold`
- `whole_brain_coverage_percent_ge_threshold`

These fields live in `extended_metrics`, participate in metric status tracking, and are flattened by the existing metric-flattening path so population and repeatability code can consume them without custom JSON parsing.

## Repeat-Level Aggregation

Add a downstream function in `post/repeatability/analyze_subject_metrics.py` that groups loaded repeat rows by `roi`, `subject`, and metric name, then computes the distribution of the subject's repeat values:

- `n_repeats`
- `mean`
- `median`
- `std`
- `cv_percent`
- `q1`
- `q3`
- `iqr`
- `min`
- `max`
- `range`
- `mean_abs_pairwise_diff`
- `max_abs_pairwise_diff`

The analysis should write a dedicated CSV, `whole_brain_coverage_repeat_distribution.csv`, alongside the other repeatability outputs.

## Compatibility

Keep the older focality field names for existing downstream consumers, but adjust their cutoff comparator from `>` to `>=`. The new fixed-threshold fields make the requested metric explicit and avoid relying on the older `gt` naming.

## Testing

Use test-first changes for:

- A voxel exactly equal to `0.2` is included in both the existing focality calculation and the new fixed coverage metric.
- Flattening includes the new coverage fields.
- The repeat-level aggregation function produces per-subject distribution rows across repeats.
