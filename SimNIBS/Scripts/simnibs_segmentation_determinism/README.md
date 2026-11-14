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

This implementation does not authorize automatic HPC submission. The
experiment may run while dataset consolidation continues because it uses a
separate source and staging root. It does not read from or write to the active
combined dataset. The `run` command remains an explicit operator action.

## Scientific boundary

The canonical MRI pair is read once from:

```text
/mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_corrected
```

Initialization verifies the tracked cohort and the selected subject, then
atomically creates checksum-identical staged copies of the T1w, T2w, and cohort
file below:

```text
/mnt/parscratch/users/cop23bi/segmentation_determinism_staging_v1/inputs
```

Both arms read only those staged copies. All repeat artifacts, Slurm logs,
retry state, workflow receipts, and analysis products are written below:

```text
/mnt/parscratch/users/cop23bi/segmentation_determinism_staging_v1
```

After both the experiment and consolidation have terminal receipts, a separate
checksum-gated installer can materialize the staging package at
`final_132_repeatability_combined_dataset_v1/extensions/segmentation_determinism_v1`.
The installer has its own ownership and completion receipts. It does not alter
the staging package, delete data, or rewrite the original combined-dataset
completion receipt.

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
SEGDET_STAGING_ROOT=/mnt/parscratch/users/cop23bi/segmentation_determinism_staging_v1
SEGDET_DATASET_ROOT=/mnt/parscratch/users/cop23bi/final_132_repeatability_combined_dataset_v1
export SEGDET_STAGING_ROOT SEGDET_DATASET_ROOT

bash "$SEGDET" scope
bash "$SEGDET" preflight
bash "$SEGDET" init
bash "$SEGDET" status
```

These four commands do not call `sbatch`. `preflight` is read-only and does not
require the combined dataset to be complete. `init` creates the staging root,
copies and hashes the three pinned input files, and submits nothing.

After reviewing the preflight and obtaining explicit approval, submit the fixed
two-arm experiment with:

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

After the terminal analysis receipt exists, generate the standalone figure
package without submitting jobs or changing the completed analysis:

```bash
bash "$SEGDET" figures
```

`figures` uses `ANALYSIS_PYTHON` and therefore requires NumPy, NiBabel and
Matplotlib in that environment. It writes only to
`segmentation_determinism_staging_v1/visualizations_v1`. Re-running the command
reuses the package only after verifying its source binding and checksums; a
foreign, partial or stale destination is rejected.

For the journal-ready three-dimensional CHARM comparison, use the separate
versioned exporter:

```bash
bash "$SEGDET" journal-spatial-figures
```

This mode is read-only with respect to the completed experiment and submits no
Slurm jobs. It reads the ten retained CHARM hard-label maps and the staged T1,
then writes only to
`segmentation_determinism_staging_v1/journal_spatial_figures_v1`. The output
contains a global projection/focal-slice composite and a whole-head slice atlas,
each as 600-dpi PNG, PDF and SVG, plus captions, a spatial summary, provenance
manifest and `SHA256SUMS`. An existing package is reused only after its source
binding and every checksum validate.

If this renderer is wrapped in one explicit Stanage job, retain the already
validated profile: `SimNIBS/4.0.1-foss-2023a`, `sheffield`, 4 CPU, 16G and
02:00:00. That job performs no segmentation, surface reconstruction, meshing or
FEM simulation.

## Later installation into the combined dataset

Do not install while consolidation is running. Once the staging terminal
receipt and the combined dataset's `dataset_manifest.json` and
`_consolidation/complete.json` all validate, run the read-only gate:

```bash
bash "$SEGDET" install-preflight
```

Only if it reports `status: ready`, install with the literal dataset-root
confirmation:

```bash
bash "$SEGDET" install-extension \
    --confirm-dataset-root="$SEGDET_DATASET_ROOT"
```

Both install commands submit zero Slurm jobs. The copy is resumable only when
the destination's ownership receipt matches the staging and dataset receipts.
A checksum dry-run with destination deletion detection must be empty before
`extensions/segmentation_determinism_v1/_import/complete.json` is written.
The installer never deletes the staging package or any prior campaign data.

## Outputs and interpretation

The analysis directory contains:

- `determinism_manifest.json`: complete machine-readable classifications;
- `artifact_inventory.tsv`: repeat, attempt, size and SHA-256 provenance;
- `repeat_comparisons.tsv`: header, geometry and voxel comparisons;
- `tissue_summary.tsv`: discrete-label voxel counts and physical volumes;
- `changed_voxel_summary.tsv`: comparisons with changed voxels; and
- `report.md`: compact human-readable conclusions.

The separate `visualizations_v1` directory contains six figures in both PNG
and PDF format:

- a two-arm determinism summary;
- the complete pairwise changed-label heatmap;
- the distribution of changed-voxel percentages across 45 CHARM pairs;
- per-label Dice distributions;
- per-label volume variation across the ten CHARM runs; and
- orthogonal slices showing the spatial frequency of disagreement with the
  modal label.

It also contains `summary_statistics.json`, `figure_captions.md`, a
source-bound `figure_manifest.json`, and `SHA256SUMS`. Tissue labels remain
numeric because the exporter does not infer an unverified lookup table. These
are descriptive results for one participant and do not establish population
variability or an electric-field effect.

The separate `journal_spatial_figures_v1` directory focuses only on where the
ten CHARM hard-label maps differ. Its discrete scale reports the number of runs
not matching the modal tissue label at each voxel. This is a repeat-disagreement
count, not segmentation error probability. Anatomical direction labels and RAS
coordinates are shown, but the evidence remains one participant under one
SimNIBS 4.0.1/HPC configuration.

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
