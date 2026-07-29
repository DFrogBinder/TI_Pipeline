# Optimizer-matched CamCan ROI analysis

## Why this layer exists

The original manuscript post-processing treated the complete anatomical atlas
parcel as the target ROI. The individualized optimization did not. It used a
small spherical target centred on the parcel and clipped to the parcel. The two
methods therefore summarized different tissue volumes, which made a successful
optimization appear worse when judged by the full-parcel median.

Analysis schema 4 makes the post-processing target match the target constructed
by the supervisor's `MakeROIs.m`.

## MATLAB-to-Python mapping

For each subject and anatomical ROI:

1. Resolve the subject-space FreeSurfer/Destrieux parcel.
2. Calculate its volumetric centroid in world coordinates.
3. Start with a sphere radius of 3.00 mm.
4. Select locations satisfying both:
   - Euclidean distance from the centroid `< radius`;
   - membership in the anatomical parcel.
5. Increase radius by 0.01 mm until the selected physical volume reaches:
   - 100 mm³ for cortical targets (`Left_M1`, `Right_DLPC`);
   - 200 mm³ for subcortical targets (`Left_Hippocampus`,
     `Right_Thalamus`);
   or until the next radius would reach 10 mm.

`MakeROIs.m` applies this procedure to tetrahedral element centres and sums
tetrahedral element volumes. The post-processing layer applies the equivalent
procedure to NIfTI voxel centres and sums constant NIfTI voxel volumes. The
world-space mean of voxel centres is volume weighted because every voxel in a
given NIfTI has the same physical volume.

Every repeat record stores the centroid, requested and achieved volume, voxel
volume, selected voxel count, final radius, growth constants, and whether the
requested volume was reached.

## Metric scopes

- Unprefixed metrics are the primary optimizer-matched target analysis.
- `anatomical_`-prefixed metrics repeat the calculation over the complete
  anatomical parcel as a secondary compatibility/QC analysis.
- `roi_mean_v_per_m` and `roi_median_v_per_m` are the primary target-field
  summaries used in manuscript figures. The mean is included explicitly to
  validate against the supervisor's mean-field optimization objective.
- `roi_min_v_per_m` and robust-maximum metrics remain available as validation
  and QC summaries rather than primary outcome panels.
- Threshold analyses are calculated at 0.20, 0.18, and 0.15 V/m.
- Every metric is calculated separately in each remeshing repeat and then
  arithmetic-mean aggregated across the ten repeats.

Earlier schema-3 CSVs cannot be reaggregated into schema 4 because they do not
contain the target-ROI mean. The image-level extraction must be rerun; the FEM
simulations remain read-only.

## Production scopes

### Full CamCan cohort

- 132 subjects × four ROIs × ten repeats = 5,280 image-level records.
- Forty resumable ROI/repeat array elements, at the established 40-way
  concurrency.
- Existing FEM fields and atlases are read-only.
- Output is isolated under
  `campaigns/final_132/post_processing/optimizer_matched_analysis_schema4/`.

### Personalized versus generic

- Seven optimized subjects × four ROIs × two conditions × ten repeats = 560
  image-level records.
- All 280 personalized simulations are in scope because the individualized
  target table contains a distinct correct montage for every subject/ROI
  configuration.
- Each personalized field is compared with the generic montage simulated on
  the same subject head and ROI.
- Output is isolated under
  `campaigns/optimized_best_worst_7/post_processing/`
  `optimizer_matched_personalized_vs_generic_schema3/`.
- This is descriptive rather than population-inferential because the seven
  subjects were originally selected as outcome extremes.
