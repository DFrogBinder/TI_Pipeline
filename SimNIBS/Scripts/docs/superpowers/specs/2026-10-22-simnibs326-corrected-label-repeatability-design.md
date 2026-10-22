# SimNIBS 3.2.6 corrected-label repeatability pipeline design

## Status

Approved architecture captured for implementation. This document defines the
scientific and operational contract; it does not authorize an HPC submission.

## Purpose

Re-run the balanced-10 SimNIBS 3.2.6 repeatability arm using the exact
manually corrected label image used by the SimNIBS 4 arm, while retaining the
SimNIBS 3.2.6 surface and tetrahedral meshing approach. This removes the old
segmentation difference from the version comparison without making the two
versions share a mesh.

The resulting comparison asks:

> When both versions start from the same corrected tissue classification, how
> do the SimNIBS 3.2.6 and SimNIBS 4 meshing and simulation workflows compare?

The new campaign is isolated from all completed v3 and v4 campaigns. Existing
results remain immutable evidence.

## Scientific boundary

### What is held constant

- the ten balanced-cohort participants;
- the manually corrected v4 source label image for each participant;
- the two targets: left hippocampus and right M1;
- the stimulation/current inputs already frozen by the completed comparison;
- the remesh and fixed-mesh conditions;
- forty repeats per participant, target, and condition;
- the optimizer-matched spherical-ROI post-processing definition.

### What remains version specific

- conversion of the corrected labels into a v3-compatible tissue model;
- v3 surface generation;
- v3 tetrahedral volume meshing;
- SimNIBS 3.2.6 FEM execution and field export.

The new v3 meshes are generated from the harmonized corrected labels. No old
v3 tissue mask, surface, or mesh is reused.

### Non-goals

- Do not rerun SPM or CAT12.
- Do not rerun or alter the completed v4 campaign.
- Do not overwrite or delete the completed legacy v3 campaign.
- Do not treat repeated byte-identical meshes as mesh-induced variability.
- Do not submit jobs while implementing or locally testing this pipeline.

## Package and isolation

Implement a new package:

`simnibs_326_corrected_label_repeatability/`

The package may reuse stable, read-only utilities from the repository, but it
must not change the scientific meaning or outputs of
`simnibs_326_repeatability/`.

Default roots:

```text
corrected source
  /mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_corrected

legacy v3 reference source (read only)
  /mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_simnibs326_headreco

new harmonized head models
  /mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_simnibs326_harmonized_v2

new left-hippocampus results
  /mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_simnibs326_harmonized_left_hippocampus_v2

new right-M1 results
  /mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_simnibs326_harmonized_right_m1_v2
```

Every mutating command must resolve and compare real paths before writing.
The three new roots must be distinct from one another and must not overlap any
source, completed v3, or completed v4 root.

Environment overrides use a single `SIMNIBS326H_` prefix and are recorded in
every plan and receipt.

## Input contract

For participant `<subject>`, the corrected source is:

```text
<corrected-root>/<subject>/anat/<subject>_T1w_ras_1mm_T1andT2_masks.nii
```

The preflight must validate that all ten participants have:

- the corrected label image;
- the original staged T1 image used by field export at
  `<corrected-root>/<subject>/anat/<subject>_T1w.nii`;
- the legacy v3 ready marker;
- `m2m_<subject>/T1fs_conform.nii.gz`;
- `m2m_<subject>/T1fs_nu_conform.nii.gz`;
- `m2m_<subject>/eeg_positions/`;
- `m2m_<subject>/toMNI/`;
- the target and stimulation records required by the completed v3 workflow.

Preflight is read only. It emits a plan containing file sizes, image grids,
observed label sets, resolved paths, and source hashes. An unknown source label,
missing required artifact, unsafe root relationship, or wrong SimNIBS version
must fail closed.

The corrected source label image is never copied into the new head-model root.
Its absolute path and SHA-256 hash are recorded in provenance so the retained
harmonized map remains traceable without duplicating the existing dataset.

## Legacy v3 reference staging

The old v3 scaffold is used only to avoid repeating SPM/CAT12 and to preserve
the already established v3 coordinate infrastructure. Before the old root can
be removed, the new workflow copies and verifies this explicit allowlist:

- `T1fs_conform.nii.gz`;
- `T1fs_nu_conform.nii.gz`;
- all regular files below `eeg_positions/`;
- all regular files below `toMNI/`;
- the old scaffold-ready marker, stored as provenance rather than as a new
  completion marker.

The staging code must reject symlinks and path escapes. It must not copy:

- `segment/`;
- `mask_prep/`;
- tissue masks;
- `.stl` or `.off` surfaces;
- `.msh` meshes;
- mesh-derived control images.

Each participant gets a staging receipt with source and destination SHA-256
hashes. A cohort receipt is complete only when the allowlist is present and
hash-identical for all ten participants and the exclusion audit finds no
forbidden files. That receipt is the evidence that the legacy scaffold root is
no longer a dependency of the new workflow; deletion remains a separate,
explicit user action.

## Corrected-label projection

### Accepted source labels

| Source label | Corrected-v4 meaning | v3-compatible result |
|---:|---|---|
| 0 | background | 0 background |
| 1 | white matter | 1 white matter |
| 2 | grey matter | 2 grey matter |
| 3 | CSF | 3 CSF |
| 5 | scalp | 5 external soft tissue |
| 6 | eyes | 6 eyes |
| 7 | compact bone | 4 generic bone |
| 8 | spongy bone | 4 generic bone |
| 9 | blood | local eligible enclosing tissue |
| 10 | muscle | 5 external soft tissue |
| 11 | internal air, when present | 11 geometry/control label; void in FEM tissues |

Label 4 is not an accepted corrected-source label. Any nonzero label not listed
above is an error.

### Blood assignment

Blood is not globally converted to CSF or scalp. For every source blood voxel:

1. Build physical-distance transforms, in millimetres, to the nonblood voxels
   of white matter, grey matter, CSF, and external soft tissue.
2. Assign the blood voxel to the class with the smallest Euclidean distance.
3. Resolve exact ties deterministically in this order: white matter, grey
   matter, CSF, external soft tissue.
4. Never assign blood to bone, eyes, background, or internal air.

The implementation records blood-voxel recipient counts, tie counts, physical
voxel spacing, and whether each eligible recipient class was present. It fails
if blood exists but no eligible recipient class exists.

Nonblood labels follow the mapping table exactly and are not modified by the
blood calculation.

### Saved label products

For each participant, retain only one harmonized categorical label image:

```text
harmonized_labels/<subject>_v3_harmonized_t1fs_grid.nii.gz
harmonized_labels/qc/<subject>.json
harmonized_labels/figures/<subject>.png
```

The corrected source label image stays in its existing location and is
referenced by path and hash. Harmonization is calculated on the source grid in
memory or task-local temporary storage. That categorical geometry map is then
resampled with nearest-neighbour interpolation onto `T1fs_conform.nii.gz` and
saved as the single retained harmonized map. Tissue labels are 1-6; internal
air, when present, remains label 11 in this geometry map so `MASK_AIR` can be
derived without retaining another NIfTI.

The implementation must not independently resample overlapping binary tissue
masks. A single categorical resampling prevents overlaps and gaps caused by
mask-by-mask interpolation.

The transient source-grid harmonized image and separate air images are not
retained. The per-participant JSON records the existing source path and hash,
output hash, affines, shapes, voxel sizes, original and final label counts,
blood assignment counts, and software version. A cohort TSV, manifest JSON,
and checksum file collect the ten retained maps for later inspection.

## v3 masks, surfaces, and tetrahedral mesh

From the retained v3-grid geometry map, derive mutually exclusive temporary
binary images named for the SimNIBS 3.2.6 headreco interface:

- `MASK_WM.nii.gz`;
- `MASK_GM.nii.gz`;
- `MASK_CSF.nii.gz`;
- `MASK_BONE.nii.gz`;
- `MASK_SKIN.nii.gz`;
- `MASK_EYES.nii.gz`;
- `MASK_AIR.nii.gz`.

These masks are task-local working products used to generate the surfaces.
After the mask validation report and new surfaces have been successfully
written and hashed, the binary masks are removed. They remain exactly
reconstructable from the retained categorical map and therefore need not
consume permanent parscratch space.

Use the SimNIBS 3.2.6 implementation of `make_surface_meshes` to generate new
surfaces from those masks. Use the SimNIBS 3.2.6 implementation of
`make_volume_mesh`, followed by the same v3 mesh orientation, thin-tetrahedron,
and tissue-tag cleanup performed by `headreco volumemesh`, to create a new
participant mesh.

This is a version-pinned adapter around the v3 meshing API, not a call to
`headreco all`. It deliberately avoids SPM/CAT12 and avoids the unrelated
post-meshing MNI/control-volume work performed by the full CLI.

The HPC preflight loads `SimNIBS/3.2.6-foss-2023a` and verifies:

- the reported SimNIBS version;
- the exact callable locations and signatures;
- availability of the v3 mesh cleanup functions;
- a non-mutating import/API probe under the real module environment.

If the installed API differs from the pinned adapter, preflight fails instead
of guessing. The adapter logs the exact call sequence and package paths in the
head-model receipt.

## Head-model validation and promotion gate

One array cell builds one participant head model. No simulation array is
submitted by the head-model launcher.

Automated validation requires:

- the retained harmonized label map and all requested surface/mesh artifacts;
- a successful temporary-mask validation recorded in the receipt;
- mutually exclusive tissues and an allowed final label set;
- nonempty required tissue compartments;
- finite, valid NIfTI geometry;
- a readable v3 mesh with expected tissue tags;
- positive tetrahedral volumes after cleanup;
- no old mesh or surface hash reused as the new artifact;
- complete per-file hashes and immutable logs;
- a rendered cross-sectional QC panel per participant.

The collector writes a cohort QC report and a content-addressed promotion
candidate. Simulations require a separate promotion receipt that names the
collector-manifest SHA-256 and confirms that the ten panels were reviewed.
Promotion never modifies source artifacts; it only authorizes the verified
manifest for downstream use.

## Simulation workflow and numerical scope

The full experiment is:

```text
10 participants x 2 targets x 2 conditions x 40 repeats
= 1,600 simulations
= 3,200 component FEM solves
```

The two targets are submitted as separate, versioned result roots. For each
target:

1. validate the promoted head-model manifest;
2. run 400 remesh cases (10 participants x 40 repeats), regenerating the v3
   volume mesh from the new v3 surfaces for every repeat;
3. collect mesh hashes and select the fixed mesh using the already established
   deterministic median-selection rule;
4. run 400 fixed-mesh cases;
5. export fields using the staged original T1 reference, as in the completed
   v3 workflow;
6. calculate the optimizer-matched spherical-ROI metrics;
7. validate 400/400 completion markers for each condition and write immutable
   receipts.

The existing stimulation records and target definitions are copied by value
into the plan and hashed. Electrode names continue to resolve through the
staged v3 `eeg_positions` cap. No target, current, electrode, ROI, or repeat
count is silently inferred from directory contents.

## Remesh interpretation

Every remesh result records its mesh hash before FEM interpretation. The
collector reports the number of unique meshes per participant and target.

If forty remesh cases are byte-identical, the workflow may still complete as a
solver/reproducibility control, but downstream reports must label the condition
deterministic. They must not report remesh CV, rank stability, or SD reduction
as mesh-induced uncertainty. This is a scientific interpretation gate, not a
reason to fabricate stochastic perturbations or silently change the mesher.

## Scheduler and failure handling

Preserve the operational profile already validated for the v3 campaign unless
the preflight plan explicitly reports a change:

- partition `sheffield`;
- eight CPUs and 32 GB per simulation task;
- eight-hour simulation wall time;
- bounded concurrency of 50;
- at most one retry after the initial attempt;
- append-only Slurm output;
- timestamped immutable attempt logs;
- exact array-element requeue using
  `${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}`.

Head-model resources are declared separately after the API smoke test and are
printed by `show-plan`. Login-node planning commands must remain dependency
light. Shared-storage paths are not used by worker jobs.

## Command surface

The package exposes explicit, non-overloaded operations:

```text
python -m simnibs_326_corrected_label_repeatability preflight
python -m simnibs_326_corrected_label_repeatability init
python -m simnibs_326_corrected_label_repeatability stage-references
python -m simnibs_326_corrected_label_repeatability build-headmodel --subject <id>
python -m simnibs_326_corrected_label_repeatability collect-headmodels
python -m simnibs_326_corrected_label_repeatability promote-headmodels ...
python -m simnibs_326_corrected_label_repeatability show-plan
python -m simnibs_326_corrected_label_repeatability status
```

HPC wrapper modes are `preflight`, `init`, `headmodels`, `status`, and the
existing full target-chain modes. Mutating modes require explicit confirmation
of their resolved new output root. Submission is never the default action.

## Receipts and reproducibility

Every stage writes machine-readable JSON plus a compact TSV or Markdown summary
where useful. Receipts include:

- UTC timestamps and Git commit;
- resolved configuration and roots;
- module and Python executable;
- SimNIBS version and API probe;
- input and output hashes;
- subject/target/condition/repeat scope;
- Slurm job IDs, dependencies, exit states, and retry counts;
- marker and artifact counts;
- mesh distinctness summaries;
- failure details without overwriting earlier attempts.

Completion requires terminal Slurm states, expected markers, expected artifact
counts, and valid hashes. `squeue` absence alone is not completion evidence.

## Test strategy

Implementation proceeds test first. Required tests include:

- exact label mapping and preservation of every nonblood class;
- physical-distance blood assignment with anisotropic voxels;
- deterministic blood tie priority and missing-recipient failure;
- internal-air separation;
- categorical nearest-neighbour grid conversion;
- mutual exclusivity and allowed labels of generated masks;
- unknown-label and unsafe-path failure;
- reference allowlist copying, hash validation, symlink rejection, and
  forbidden-file exclusion;
- mocked v3 API call order and signature-gate failures;
- head-model QC and manifest-bound promotion;
- exact full-scope task counts and root separation;
- bounded retry and exact-element requeue semantics;
- deterministic-remesh interpretation flags;
- shell syntax and Python unit/integration tests;
- regression tests proving the legacy pipeline remains unchanged.

Local tests use small synthetic NIfTI volumes and mocked SimNIBS calls. A real
v3 API probe and a one-subject head-model smoke build occur on the HPC only
after implementation review; neither authorizes the 1,600-simulation campaign.

## Delivery boundary

Code completion means the package, tests, documentation, Slurm launchers, and
preflight commands are committed and locally verified. It does not mean that
head models or simulations have been submitted or completed. The user receives
separate copy-paste commands for HPC pull, read-only preflight, smoke build,
inspection, promotion, and eventual full submission.
