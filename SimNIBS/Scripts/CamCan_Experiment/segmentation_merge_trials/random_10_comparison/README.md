# Random 10-Subject Segmentation Merge Comparison

This comparison reuses the two merge definitions from
`../run_merge_mesh_trial.py` on a reproducible random sample of ten CamCAN
subjects. It consumes already generated CHARM tissue maps, so it does not run
CHARM segmentation, meshing, or electric-field simulations.

## Inputs

- CHARM maps: `/home/boyan/sandbox/Jake_Data/all-seg-maps`
- Manual maps: `/home/boyan/sandbox/Jake_Data/Archive/ti_dataset`
- Default output: `/home/boyan/sandbox/Jake_Data/segmentation-merge-random-10`

All 175 CHARM maps have a matching manual map. The default sample uses Python's
seeded random sampler with seed `20260714`. The selected IDs are recorded in
`sampled_subjects.txt` and the selection rule is recorded in
`sample_manifest.json`.

## Merge Definitions

Approach A starts from the complete CHARM tissue map and overwrites it with every
positive manual label except skin label `5`:

```text
final = CHARM tissue map
final[manual != 0 and manual != 5] = manual
```

Approach B fills the CHARM head envelope, labels the solid envelope as skin, and
then applies the same manual non-skin overlay:

```text
solid_head = fill_holes(CHARM != 0)
final = background
final[solid_head] = 5
final[manual != 0 and manual != 5] = manual
```

The manual map is nearest-neighbour resampled into the corresponding CHARM grid
before either merge.

## Run

```bash
/home/boyan/anaconda3/envs/simnibs_post/bin/python3.12 run_random_10_merge_comparison.py
/home/boyan/anaconda3/envs/simnibs_post/bin/python3.12 render_random_10_presentation_assets.py
/home/boyan/anaconda3/envs/simnibs_post/bin/python3.12 build_random_10_presentation.py
soffice --headless --convert-to pdf --outdir \
  /home/boyan/sandbox/Jake_Data/segmentation-merge-random-10 \
  /home/boyan/sandbox/Jake_Data/segmentation-merge-random-10/random_10_segmentation_merge_A_vs_B_comparison.pptx
```

The merge runner is resumable. Completed subjects are reused unless `--force`
is supplied. A fixed subject list can be provided with `--subjects-file`.

## Outputs

```text
segmentation-merge-random-10/
  sampled_subjects.txt
  sample_manifest.json
  summary.csv
  summary.json
  subjects/<subject>/
    <subject>_manual_resampled_to_CHARM.nii.gz
    candidate_A_full_charm_fallback/
      <subject>_candidate_a_merged.nii.gz
      <subject>_candidate_a_preview.png
    candidate_B_solid_charm_head_skin_base/
      <subject>_candidate_b_merged.nii.gz
      <subject>_candidate_b_preview.png
    <subject>_candidate_A_vs_B.png
    <subject>_voxel_change_large.png
    merge_qc.json
  presentation_assets/
  random_10_segmentation_merge_A_vs_B_comparison.pptx
  random_10_segmentation_merge_A_vs_B_comparison.pdf
```

The presentation uses two slides per subject: one A/B segmentation slide with
the same large-panel layout as the original three-subject deck, followed by one
full-width voxel-change slide for detailed inspection.

QC verifies that both candidates preserve the manual non-skin overlay exactly,
reports label counts and physical volumes, and records which Approach A tissue
labels are converted to skin by Approach B.

## Candidate A For All Subjects

`run_candidate_a_all_subjects.py` generates only Approach A for every paired
subject. It is resumable, validates each saved grid against its CHARM source,
continues past individual failures, and updates `summary.csv` after every
subject.
