# Corrected-label SimNIBS 3.2.6 repeatability workflow

This package implements the new v3 arm of the balanced-10 comparison. It uses
the existing manually corrected v4 categorical labels as anatomical input,
projects them deterministically into a v3-compatible geometry, then creates
new SimNIBS 3.2.6 surfaces and tetrahedral meshes. It does **not** rerun
SPM/CAT12 and does not reuse the old v3 masks, surfaces, or meshes.

No HPC work was submitted during implementation. The harmonization and direct
v3 meshing stages remain candidates until the real Stanage module probe and
ten-subject head-model campaign produce evidence.

## Scientific boundary

The label mapping is:

| Corrected source | v3-compatible result |
|---|---|
| 1 WM, 2 GM, 3 CSF | unchanged |
| 7 compact bone, 8 spongy bone | 4 generic bone |
| 10 muscle | 5 external soft tissue |
| 9 blood | nearest eligible WM/GM/CSF/soft-tissue voxel in physical mm |
| 5 scalp, 6 eyes, optional 11 internal air | unchanged |

Exact blood-distance ties resolve as WM, GM, CSF, then external soft tissue.
Background, bone, eyes, and air are never blood recipients. Source label 4,
unknown labels, non-finite values, and fractional labels fail closed.

Only one derived NIfTI is retained for each participant:

```text
<headmodel-root>/harmonized_labels/<subject>_v3_harmonized_t1fs_grid.nii.gz
```

The original corrected map is not copied. Its absolute path and SHA-256 are in
the subject receipt. Temporary mutually exclusive masks are deleted after
surface generation.

The old v3 tree is used only for coordinate/reference files. Atomic staging
allows `T1fs_conform.nii.gz`, `T1fs_nu_conform.nii.gz`, regular files under
`eeg_positions/` and `toMNI/`, and the old ready marker as provenance. Meshes,
surfaces, tissue masks, `segment/`, and `mask_prep/` are excluded.

The reference collector first re-hashes both the legacy files and their staged
copies. After it writes the complete ten-subject cohort receipt, head-model
tasks validate staged files against the receipt hashes and no longer require
the legacy root. Deleting that old root is still a separate user action.

The simulation runner still links the existing corrected-root T1, T2, and
source label into each task workspace because that is the validated legacy
runner contract. Preflight checks all three; the new mesher consumes the
retained harmonized geometry, while T1 remains the field-export reference.

## Gates and interpretation

The workflow has separate gates:

1. read-only ten-subject input preflight;
2. real SimNIBS 3.2.6 module/API probe;
3. verified legacy-reference staging;
4. ten newly constructed v3 head models and rendered QC panels;
5. human review and content-addressed promotion;
6. initialization of configs bound to that exact promotion receipt;
7. remesh, median fixed-mesh selection, fixed runs, and paired ROI analysis.

Promotion records authorization; it does not alter scientific artifacts. A
changed manifest, promotion receipt, map, surface, mesh, or config blocks the
simulation boundary.

Every nominal remesh repeat records its mesh hash. Byte-identical repeats are
valid solver/reproducibility controls, but they are not evidence of
mesh-induced variability and must not be described that way.

## HPC handoff

After pulling the implementation branch:

```bash
cd ~/Repos/TI_Pipeline/SimNIBS/Scripts
PIPELINE=simnibs_326_corrected_label_repeatability/hpc/submit.sh
```

The default is a read-only preflight and submits nothing:

```bash
bash "$PIPELINE" preflight
bash "$PIPELINE" status
```

Run the real module probe, then stage and build:

```bash
bash "$PIPELINE" module-check
bash "$PIPELINE" stage-references
bash "$PIPELINE" headmodels
```

`headmodels` requires the completed module and reference-collection receipts.
It uses the renderer at
`SIMNIBS326H_RENDER_PYTHON` (default:
`$HOME/.conda/envs/ti-post/bin/python`) and submits only the 10 build tasks plus
their collector. It never releases simulations.

Inspect all ten PNG panels and the manifest under:

```text
<headmodel-root>/_pipeline/headmodels/
```

Promote only the hash actually printed by the collected manifest:

```bash
HEADMODEL_ROOT=/mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_simnibs326_harmonized_v2
MANIFEST_SHA256=<64-character-manifest-hash>
bash "$PIPELINE" promote \
  --confirm-root="$HEADMODEL_ROOT" \
  --manifest-sha256="$MANIFEST_SHA256" \
  --confirm-reviewed
```

Initialization is deliberately after promotion because each config is bound to
the promoted hashes. It also requires exact confirmation of both new result
roots:

```bash
LEFT_ROOT=/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_simnibs326_harmonized_left_hippocampus_v2
RIGHT_ROOT=/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_simnibs326_harmonized_right_m1_v2
bash "$PIPELINE" init --confirm-roots="$LEFT_ROOT,$RIGHT_ROOT"
```

The simulation stages are explicit:

```bash
bash "$PIPELINE" all-remesh
# After remesh completion, ROI extraction and fixed-mesh preparation:
bash "$PIPELINE" all-fixed
# After both final receipts validate 800/800 per target:
bash "$PIPELINE" analysis
bash "$PIPELINE" analysis-status
```

For individual target control:

```bash
bash "$PIPELINE" remesh left-hippocampus
bash "$PIPELINE" post-remesh left-hippocampus --dependency=<remesh-array-job-id>
bash "$PIPELINE" fixed left-hippocampus --dependency=<optional-prerequisite-job-id>
```

`all-remesh` and `all-fixed` keep the target arrays sequential so no more than
50 simulation tasks run concurrently. Simulation tasks retain the established
`sheffield`, 8 CPU, 32 GB, 8-hour profile, one retry, append-only Slurm output,
immutable attempt logs, and exact array-element requeue. Worker jobs use
parscratch only.

Every successful launcher call appends its UTC time, mode, target, requested
dependency, returned Slurm job IDs, and Git commit to the campaign-wide ledger:

```text
<headmodel-root>/_pipeline/submissions.tsv
```

`status` and `analysis-status` report artifact/receipt state. They do not
replace Slurm accounting: use the site job-history tool or `sacct` to confirm
terminal job states, including failed or requeued attempts that are no longer
visible in `squeue`.

## Environment overrides

Every path override is namespaced with `SIMNIBS326H_`. The principal variables
are `CORRECTED_ROOT`, `LEGACY_REFERENCE_ROOT`, `ATLAS_ROOT`, `HEADMODEL_ROOT`,
`LEFT_ROOT`, `RIGHT_ROOT`, and `TARGETS_CSV`; for example,
`SIMNIBS326H_HEADMODEL_ROOT`. The launcher additionally accepts
`SIMNIBS326H_RENDER_PYTHON` and `SIMNIBS326H_MAX_RETRIES`.

The protected completed v3/v4 roots and corrected-source root are checked for
equality, ancestor, and descendant collisions before writes.
