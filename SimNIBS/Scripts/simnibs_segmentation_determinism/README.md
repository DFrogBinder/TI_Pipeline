# Single-subject segmentation determinism experiment

This package tests whether two segmentation implementations reproduce their
own voxel outputs when run repeatedly with unchanged inputs:

- SimNIBS 3.2.6 `headreco` with SPM12/CAT12; and
- SimNIBS 4.0.1 CHARM.

The pinned participant is `sub-CC110174`, selected from the tracked balanced
ten with seed `20261106`. Each implementation runs ten independent
segmentations. The full scope is **20 scientific segmentation tasks** and
**23 scheduler tasks**: two compute-node module preflights, twenty array
elements and one collector. There are no surface, tetrahedral-meshing,
electrode-placement or FEM tasks.

This implementation does not authorize HPC submission. The `run`
command must not be used until the active multi-terabyte dataset checksum has
finished, its state has been checked, and the user has explicitly approved
this experiment's submission.

## Scientific boundary

Both arms read the same checksum-pinned T1w and T2w images from:

```text
/mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_corrected
```

They write only below:

```text
/mnt/parscratch/users/cop23bi/simnibs_segmentation_determinism_v1
```

The v3 arm runs only:

```text
headreco preparevols --noclean sub-CC110174 <T1> <T2>
```

It reuses the hash-verified CAT12/MATLAB compatibility overlay and the bounded
NumPy/NiBabel activation shim already used by the v3 repeatability pipeline.
`preparecat`, `cleanvols`, `surfacemesh`, `volumemesh` and `headreco all` are
forbidden.

The v4 arm runs only:

```text
charm sub-CC110174 <T1> <T2> --registerT2 --initatlas --segment --forceqform
```

CHARM surface and mesh options are forbidden. Only
`tissue_labeling_upsampled.nii.gz` is promoted from the temporary task
workspace.

Every repeat uses the same canonical subject ID. Repeat identity exists only
in the isolated parent directory, Slurm array index and receipts. An invalid
or partial run never receives `complete.json`.

## Candidate Stanage resources

The initial settings are pinned to the nearest established profiles but remain
candidate settings until the compute-node preflight receipts exist:

- partition: `sheffield`;
- segmentation array: `0-9%10` for each arm;
- 8 CPU, 32G and 08:00:00 per segmentation task;
- `SimNIBS/3.2.6-foss-2023a` plus `MATLAB/2023b` for v3; and
- `SimNIBS/4.0.1-foss-2023a` for v4.

One exact failed array element may be requeued once. A retry starts from a new
attempt-specific work directory, preserves the first failure's logs and is
identified in the scientific manifest. There is no unbounded retry loop.
Incomplete publication from a failed attempt is quarantined inside that repeat
before the one permitted retry. The submitter also uses an atomic lock and
refuses a second campaign for an output root that is already submitted.
If a submission command fails and leaves `submission.lock`, do not remove it
or rerun automatically: audit `job_ids.tsv`, `submission.json`, `squeue` and
`sacct` first so a partially submitted campaign is not duplicated.

## Safe operator sequence

From `~/Repos/TI_Pipeline/SimNIBS/Scripts` on Stanage:

```bash
SEGDET=simnibs_segmentation_determinism/hpc/submit.sh

bash "$SEGDET" scope
bash "$SEGDET" preflight
bash "$SEGDET" init
bash "$SEGDET" status
```

These four commands do not call `sbatch`. `preflight` is read-only and leaves
an absent output root absent; `init` creates configuration and source manifests
but submits nothing.

Before any submission, independently confirm that the dataset checksum is no
longer running and obtain explicit approval. Only after both gates are met is
the submission command:

```bash
bash "$SEGDET" run
```

`run` submits two module checks, then one ten-element array after each matching
module check succeeds, followed by an `afterany` collector. The collector can
record an incomplete audit after a task failure, but it creates the terminal
success receipt only after validating all twenty repeat receipts.

Progress remains read-only:

```bash
bash "$SEGDET" status
```

After all twenty valid repeats exist, analysis can be rerun without `sbatch`:

```bash
bash "$SEGDET" analyze
```

If the login-node Python lacks NiBabel, set `ANALYSIS_PYTHON` to an approved
analysis environment before this local re-analysis. The scheduled collector
uses the SimNIBS 4.0.1 module environment.

## Outputs and interpretation

The analysis directory contains:

- `determinism_manifest.json`: complete machine-readable classifications;
- `artifact_inventory.tsv`: repeat, attempt, size and SHA-256 provenance;
- `repeat_comparisons.tsv`: header, geometry and voxel comparisons;
- `tissue_summary.tsv`: discrete-label voxel counts and physical volumes;
- `changed_voxel_summary.tsv`: comparisons with changed voxels; and
- `report.md`: compact human-readable conclusions.

Determinism is assessed **within** each version. The package does not treat v3
and v4 tissue taxonomies as voxel-equivalent and does not test electric-field
differences.

The report separates:

- operational determinism: consistent artifacts and successful bounded stages;
- geometric determinism: shape, voxel sizes, qform, sform and affine;
- discrete determinism: exact hard-label/binary arrays; and
- probability determinism: exact floating arrays plus descriptive absolute
  differences above `1e-7`, `1e-6` and `1e-5`.

Artifact classes are assigned from the pinned output contract, not merely the
on-disk dtype. This matters because SPM posterior probabilities may be stored
as scaled integers. The analysis records both raw stored-array hashes and
decoded-array hashes, reads effective scaling from the NiBabel proxy, and uses
decoded values for label, probability and intensity comparisons. CHARM has no
retained probability-map output in this experiment, so its probability result
is reported as not applicable rather than as vacuous success.

Different `.nii.gz` file SHA-256 values can be caused by gzip metadata and do
not by themselves prove a voxel difference. Canonical uncompressed array hashes
and explicit header comparisons determine the scientific classification.

## Completion gate

`_pipeline/complete.json` is written last and only when:

- the cohort, selection audit and both MRI hashes match their pins;
- both module preflights succeeded;
- exactly ten valid repeat receipts exist for each arm;
- every receipt proves the approved stage and module boundary;
- artifact inventories validate within each arm;
- the comparison outputs and report were written and hashed; and
- no source output was modified.

Scheduler state alone is never treated as scientific completion.
