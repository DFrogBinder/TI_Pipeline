# Segmentation Determinism Consolidated Paths Design

## Intent

Keep the complete single-subject segmentation-determinism experiment inside
the final combined repeatability dataset so one dataset transfer or Shared
backup includes the scientific inputs, both ten-repeat arms, logs, receipts,
and analysis products.

## Selected approach

Use an append-only experiment extension below the existing consolidated
dataset:

```text
/mnt/parscratch/users/cop23bi/final_132_repeatability_combined_dataset_v1/
  extensions/segmentation_determinism_v1/
    inputs/sub-CC110174/anat/
    simnibs326_headreco/
    simnibs401_charm/
    analysis/
    _pipeline/
```

The consolidated dataset remains the single pinned parent. The canonical MRI
source is the materialized `sub-CC110174` T1/T2 pair already present in the
copied SimNIBS 4 left-hippocampus remesh repeat 001. Initialization copies
those files into the extension `inputs/` tree, verifies their SHA-256 values,
and all scientific tasks read only the staged copies.

The experiment uses the dataset-root `subjects.txt`, `dataset_manifest.json`,
and `_consolidation/complete.json` instead of repository-local selection
files. It verifies the known consolidation plan SHA-256 and repeats the seeded
subject selection before creating the extension.

## Alternatives considered

- Place the experiment under `supplemental/`: rejected because it blurs the
  original consolidated supplemental material with a later experiment.
- Keep a sibling parscratch experiment root: rejected because it would not be
  carried by a whole-dataset transfer or backup.
- Create a second full dataset version: rejected because duplicating or
  restructuring roughly 9 TB is disproportionate for this small extension.

## Safety and provenance

- This change edits code only; it does not run or submit the experiment.
- Initialization requires the consolidated dataset completion receipt and
  refuses a prepared or incomplete dataset.
- The output root must be exactly the code-owned extension below the dataset.
- Canonical and staged MRI paths must both remain below the dataset root.
- Initialization is atomic: a failed staging operation cannot publish a
  partially owned experiment root.
- Source MRI files remain read-only and their before/after hashes are recorded.
- Existing scientific scope is unchanged: one subject, two implementations,
  ten repeats each, 20 segmentation tasks, no surfaces, meshing, or FEM.
- Existing module and resource settings are unchanged.
- The extension has its own terminal receipt. The base dataset completion
  receipt continues to describe the original consolidated package rather than
  silently claiming the later extension was part of its initial inventory.

## Execution boundary

The operator must not initialize or submit the extension while the current
consolidation/readiness workflow is active. Code and documentation will state
that execution begins only after consolidation and retirement cleanup are
complete. No HPC command is executed as part of this path migration.
