# Segmentation Merge Mesh Trials

This directory contains a mesh-only trial pipeline for comparing two ways to combine
manual/ROAST-derived segmentations with CHARM segmentations. It does not run
SimNIBS electric-field simulations.

## Trial Layout

For each subject, `run_merge_mesh_trial.py` writes:

```text
<out-root>/<subject>/
  charm_base/
    anat/
      <subject>_T1w.nii[.gz]
      <subject>_T2w.nii[.gz]
      <subject>_T1w_ras_1mm_T1andT2_masks.nii[.gz]
      m2m_*/
      <subject>_CHARM_original_tissue_labeling_upsampled.nii.gz
    BASE_COMPLETE.json
    source_manifest.json
  candidate_A_full_charm_fallback/
    anat/
    <subject>_candidate_a_merged.nii.gz
    <subject>_manual_resampled_to_CHARM.nii.gz
    <subject>_CHARM_original_tissue_labeling_upsampled.nii.gz
    <subject>_candidate_a_preview.png
    merge_qc.json
    COMPLETE.json
  candidate_B_solid_charm_head_skin_base/
    anat/
    <subject>_candidate_b_merged.nii.gz
    <subject>_manual_resampled_to_CHARM.nii.gz
    <subject>_CHARM_original_tissue_labeling_upsampled.nii.gz
    <subject>_candidate_b_solid_charm_head_mask.nii.gz
    <subject>_candidate_b_preview.png
    merge_qc.json
    COMPLETE.json
```

The source dataset is copied into the trial output tree before CHARM runs. The
original archived/source `anat` directory is not modified.

## Shared Preprocessing

Both candidates use the same setup:

1. Run CHARM once from copied T1/T2 inputs to create a clean CHARM baseline.
2. Keep a copy of CHARM's original `label_prep/tissue_labeling_upsampled.nii.gz`.
3. Resample the manual segmentation into CHARM voxel space using nearest-neighbor
   interpolation.
4. Ignore manual label `5` in the final overlay.
5. Replace the copied CHARM `tissue_labeling_upsampled.nii.gz` with the candidate
   merged segmentation.
6. Run `charm <subject> --mesh`.

## Candidate A: Full CHARM Fallback

Candidate A starts from the complete CHARM tissue map:

```text
final = CHARM full tissue map
final[manual label is positive and not 5] = manual label
```

This keeps every positive manual label except skin. CHARM supplies the skin and
also supplies fallback labels wherever the manual segmentation has background or
the removed skin label.

This is the more conservative candidate because gaps left by removing manual skin
are filled by CHARM's tissue labels rather than being forced to skin.

## Candidate B: Solid CHARM Head Skin Base

Candidate B starts from a solid CHARM-derived head envelope:

```text
solid_head = fill_holes(CHARM label > 0)
final = background everywhere
final[solid_head] = skin label 5
final[manual label is positive and not 5] = manual label
```

This keeps every positive manual label except skin, but any remaining space inside
the CHARM head envelope becomes skin/scalp. This is closer to the remembered
"solid binary mask, then paste manual segmentation on top" approach.

This candidate is more aggressive: it can hide gaps that break the mesh, but it
may assign skin conductivity to regions that CHARM would have labelled as skull,
CSF, or another tissue.

## HPC Usage

Create a subject list on Stanage, one subject ID per line:

```bash
cp /users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/segmentation_merge_trials/merge_mesh_trial_subjects.example.txt /users/cop23bi/scripts/merge_mesh_trial_subjects.txt
```

Edit `/users/cop23bi/scripts/merge_mesh_trial_subjects.txt` to contain the ten
worst subjects.

Submit using:

```bash
TI_MERGE_TRIAL_SOURCE_ROOT=/mnt/parscratch/users/cop23bi/ZIPs/Analised-Data/Left_Hippocampus_Runs/Left_Hippocampus_Data_01 \
TI_MERGE_TRIAL_OUT_ROOT=/mnt/parscratch/users/cop23bi/merge_mesh_trials/Left_Hippocampus_Data_01 \
/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/segmentation_merge_trials/submit_merge_mesh_trial_array.sh
```

Defaults:

- `TI_MERGE_TRIAL_MAX_RETRIES=0`: unlimited self-requeue until validation passes.
- `TI_MERGE_TRIAL_TIMEOUT_HOURS=7`: internal timeout before the 8-hour Slurm limit.
- `TI_MERGE_TRIAL_MAX_CONCURRENT=4`: conservative concurrency because each subject
  may run one CHARM baseline plus two candidate remeshes.

Validation requires both candidate meshes, merged NIfTI files, QC JSON files, and
completion markers to exist.
