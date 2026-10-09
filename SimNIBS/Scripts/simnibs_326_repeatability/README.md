# SimNIBS 3.2.6 repeatability pipeline

This is an isolated, opt-in implementation of the SimNIBS 3.2.6 half of the
final-132 balanced-10 repeatability experiment. It does not change the
completed SimNIBS 4.0.1 roots and it does not delete transferred data.

## Scientific scope

- SimNIBS module: `SimNIBS/3.2.6-foss-2023a`
- `headreco` dependency: `MATLAB/2023b` for SPM12/CAT12 segmentation; the FEM
  simulations themselves remain SimNIBS 3.2.6 Python jobs
- Participants: the same 10-member balanced final-132 cohort
- Targets: left hippocampus and right M1
- Conditions: 40 fresh volume meshes and 40 fixed-mesh solves per
  participant-target cell
- Main simulation tasks: 1,600 (800 per target)
- Component FEM solves: 3,200 (two tDCS fields per TI simulation)
- One-time support work: 10 `headreco all` scaffold tasks and two 10-subject
  optimizer-ROI extraction arrays
- Nested 40 by 40 work is not part of this campaign

The stimulation values are resolved fail-closed from the repository's
checksum-pinned `utils/targets.csv`. The left-hippocampus pairs are F8-P8 at
2.0 mA and T7-P7 at 1.5886564694485628 mA. The right-M1 pairs are Fp2-F6 at
2.0 mA and C4-CP2 at 0.6324555320336759 mA.

The conductivity handling follows the established runner: WM 0.126 S/m, GM
0.276 S/m, CSF 1.65 S/m, average bone 0.01 S/m, scalp 0.465 S/m, eye 0.5 S/m,
muscle 0.16 S/m, and the 1.4 S/m electrode layer. SimNIBS 3.2.6 supplies its
standard compact-bone (0.008 S/m), spongy-bone (0.025 S/m), and blood
(0.6 S/m) entries. The module preflight records the observed table before any
production array is released. It also launches a short headless MATLAB probe
and verifies the legacy NumPy scalar aliases and NiBabel accessor required by
SimNIBS 3.2.6 before releasing the ten scaffold jobs.

SimNIBS 3.2.6 `msh2nii` supports mask and field interpolation but predates the
`--create_label` option. For the combined tissue-label volume used to mask the
TI field, the v3 backend applies the same tetrahedra-only element-tag
assignment implemented by later SimNIBS versions, using the installed v3
`mesh_io.ElementData.to_nifti(..., method="assign")` API. The preflight checks
and records both this internal API and the actual v3 CLI capabilities.

## Version boundary

SimNIBS 4 CHARM models cannot be used by SimNIBS 3.2.6. The new pipeline
therefore runs `headreco all --noclean` once per participant using the same T1
and T2 images, then copies the immutable v3 scaffold for each repeat and runs a
fresh `headreco volumemesh --noclean`.

The corrected v4 label image remains checksum-recorded input provenance but is
not consumed by `headreco`. Consequently, this is an end-to-end version arm:
the v3 and v4 meshes differ in segmentation/head-model construction as well as
software version. They must not be described as the same-label remeshing of an
identical head model.

The v3 native mesh is written beside `m2m_<subject>`. After validation, the
runner makes a physical, checksum-matched copy inside `m2m_<subject>` so the
existing task validator and optimizer-matched spherical ROI selection can use
the established layout. Max-TI is calculated with the Grossman maximal-envelope
equation inside the adapter because `simnibs.utils.TI_utils` is not guaranteed
to exist in 3.2.6.

## Production roots

Defaults can be overridden with the corresponding `SIMNIBS326_*` environment
variables.

```text
source:
  /mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_corrected
v3 scaffolds:
  /mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_simnibs326_headreco
left hippocampus:
  /mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_simnibs326_left_hippocampus_v1
right M1:
  /mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_simnibs326_right_m1_v1
```

The v3 pipeline treats these existing CHARM locations as protected read-only
roots and refuses to start if any writable v3 root equals, contains, or is
nested within one of them:

```text
/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Scaffolds
/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10
/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_right_m1
```

The staged T1/T2/label input root and atlas root are protected in the same way.
Only the dedicated `simnibs326` roots can be rebuilt or overwritten.

## SimNIBS 3.2.6 CAT12 compatibility

The CAT12 revision bundled with SimNIBS 3.2.6 can write its MAT quality report
but fails in its legacy XML serializer under MATLAB R2023b. This was confirmed
uniformly across the balanced ten-subject scaffold array. Stanage does not
currently expose an older compatible MATLAB release.

The pipeline therefore builds a temporary, per-process compatibility overlay
from the installed SimNIBS 3.2.6 MATLAB files. It first verifies their official
v3.2.6 SHA-256 hashes, then changes only the two fatal XML-write branches in
`cat_io_xml.m` to warnings. The MAT report is still written, XML is still
attempted, and all segmentation/meshing operations are unchanged. CAT12's
`tbx_cfg_cat.m` re-adds the installed toolbox directory while
`spm_jobman('initcfg')` runs, so a patched `segment_CAT.m` reasserts the
overlay, clears cached `cat_io_xml` resolution, and verifies the selected file
again immediately after batch initialization. The overlay is deleted when
headreco exits; `/opt/apps` and all CHARM locations are never modified.

The module-preflight job now reproduces the `spm_jobman('initcfg')` path
mutation, reasserts the overlay, executes it in MATLAB, and records the patch
ID, source hashes, replacement counts, post-init resolution checks, and
successful MAT-report probe. Scaffold tasks reject old or incompatible
module-preflight receipts before starting headreco.

Live computation remains on `/mnt/parscratch`. The Shared partition can hold
transferred archives, but it is not used as a worker-node input or output path.

## Safe launch sequence (later, after storage cleanup)

From the repository `Scripts` directory on Stanage:

```bash
cd simnibs_326_repeatability
```

Run the read-only full-scope preflight. It checks every T1, T2, provenance-only
label image, subject atlas, and the confirmed montage file. It submits nothing
and writes nothing.

```bash
bash hpc/submit.sh preflight
```

Initialize only the two new output roots. This writes configs/manifests but
submits nothing.

```bash
bash hpc/submit.sh init
```

Submit the module check, ten scaffold tasks, both 400-task remesh arrays, their
completion gates, optimizer-sphere extraction, and fixed-mesh preparation:

```bash
bash hpc/submit.sh all-remesh
```

The two target chains are sequential, so the campaign keeps a global maximum
of 50 simulation tasks. This first release stays below Stanage's 1,000
submitted-array-element limit. It intentionally does not submit fixed-mesh
arrays.

After both fixed configs have been prepared and the preceding jobs have left
the queue, submit the two sequential 400-task fixed-mesh arrays and their final
validation gates:

```bash
bash hpc/submit.sh all-fixed
```

Read-only status is available at any time:

```bash
bash hpc/submit.sh status
```

Individual stages are also available as `module-check`, `scaffold`, `remesh`,
`post-remesh`, and `fixed`. Use `--dependency=<job-id>` when a required prior
stage is still queued.

## Scheduler and retry contract

- Partition: `sheffield`
- Simulation/scaffold task: 8 CPUs, 32 GB, 8 hours
- Main arrays: `0-399%50`
- Optimizer-ROI arrays: `0-9%10`
- Completed tasks are skipped only when output validation and SimNIBS 3.2.6
  provenance both pass
- Incomplete scaffold and simulation tasks receive at most one
  validation-gated requeue after the initial attempt; simulations retain the
  existing four-hour task timeout
- Requeues always name the exact array element; one failing placeholder task
  cannot restart its siblings
- Slurm output is append-only and each attempt has a timestamped log, so the
  first failure is retained for diagnosis
- Missing required inputs exit with code 126 and are logged without an
  infinite requeue loop
- The CAT12 compatibility overlay is hash-verified and fail-closed; unexpected
  module source files or an old module-preflight receipt prevent scaffold work

`SIMNIBS326_MAX_RETRIES` can override the retry cap with a non-negative
integer. `0` disables automatic retries; it no longer means unlimited retries.

The optimizer-matched spherical ROI is reconstructed on each output grid. Its
mesh- and voxel-based realizations are aligned/equivalent analysis definitions,
not literally identical masks.
