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
and verifies the legacy NiBabel accessor required by SimNIBS 3.2.6 before
releasing the ten scaffold jobs.

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
- Incomplete simulation tasks use the existing four-hour task timeout and
  unlimited validation-gated requeue policy
- Missing required inputs exit with code 126 and are logged without an
  infinite requeue loop

The optimizer-matched spherical ROI is reconstructed on each output grid. Its
mesh- and voxel-based realizations are aligned/equivalent analysis definitions,
not literally identical masks.
