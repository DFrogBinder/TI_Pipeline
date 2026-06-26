# CamCan In-Place Mesh-Reuse Rerun Design

## Context

The existing CamCan repeat datasets were generated with incorrect electrode
positions and/or injection currents. The corrected montage values are read from
`utils/targets.csv` through `simulation/target_montages.py`.

Rerunning the full experiment from scratch is too expensive because meshing is
the dominant runtime cost and the current incorrect dataset already occupies
roughly 11 TB on the HPC filesystem. The rerun must therefore preserve existing
anatomical inputs and SimNIBS `m2m` mesh products, delete generated outputs, and
rerun only the corrected simulations in place.

The workflow must support any ROI montage preset present in `targets.csv`. It
must not encode which targets are considered finalized; the operator is
responsible for choosing which ROI roots to submit.

## Goals

- Reuse existing subject meshes from the current CamCan repeat dataset.
- Avoid creating a second full dataset tree.
- Free storage before rerun by deleting generated outputs that are no longer
  valid.
- Submit one Slurm array per ROI root, covering all subjects across all 10
  repeat dataset folders.
- Keep downstream paths compatible by writing corrected outputs back into the
  same repeat dataset directories.
- Make destructive cleanup explicit, auditable, and dry-run by default.

## Non-Goals

- Do not optimize or validate montage target rows in `utils/targets.csv`.
- Do not create a symlinked mirror dataset.
- Do not remesh subjects unless the operator separately chooses to run the old
  full pipeline.
- Do not delete anatomy, segmentation, `m2m_*`, or `m2m_sub-*` directories.
- Do not require 10 separate array submissions for 10 repeats.

## Approach

Implement an in-place CamCan rerun workflow with three stages:

1. Prepare and optionally clean the ROI root.
2. Submit one manifest-driven Slurm array for the ROI.
3. Validate corrected outputs for each task and summarize batch status.

The preflight command discovers all repeat datasets under one ROI root, writes a
task manifest with one row per repeat/subject, and reports cleanup actions. It
does not delete anything unless called with an explicit apply flag.

The Slurm array reads the manifest. Each array task corresponds to one subject
inside one repeat dataset. The task sets `TI_SIM_ROOT` to that repeat dataset
root and calls `TI_runner_multi-core.py` with `--reuse-existing-mesh`,
`--subject`, and `--montage-preset`.

## Files and Responsibilities

### `CamCan_Experiment/simulation/prepare_inplace_rerun.py`

New command-line utility for discovery, cleanup planning, cleanup application,
and manifest writing.

Responsibilities:

- Accept an ROI root such as
  `/mnt/parscratch/users/cop23bi/ZIPs/Analised-Data/Left_M1_Runs`.
- Discover repeat dataset directories matching `*_Data_*`.
- Optionally filter repeat IDs and subjects.
- For each repeat/subject, validate required preserved inputs:
  - `<dataset>/<subject>/anat/<subject>_T1w.nii` or `.nii.gz`
  - `<dataset>/<subject>/anat/<subject>_T2w.nii` or `.nii.gz`
  - `<dataset>/<subject>/anat/<subject>_T1w_ras_1mm_T1andT2_masks.nii`
  - at least one usable mesh path:
    - `<dataset>/<subject>/anat/m2m_<subject>/<subject>.msh`
    - or `<dataset>/<subject>/anat/m2m_sub-<suffix>/<subject>.msh`
- Write a tab-separated task manifest with:
  - `task_id`
  - `dataset_name`
  - `repeat_id`
  - `dataset_root`
  - `subject`
  - `anat_dir`
  - `mesh_path`
  - `status`
  - `message`
- Dry-run cleanup by default.
- With `--apply`, delete generated output paths and write a cleanup manifest.

Cleanup must use a conservative generated-output denylist. The initial deletion
set is:

- Per subject:
  - `<subject>/anat/SimNIBS`
  - `<subject>/anat/post`
  - known generated scratch files outside `m2m*`:
    - `<subject>_T1w_ras_1mm_T1andT2_masks_clipped.nii`
    - `<subject>_T1w_ras_1mm_T1andT2_masks_merged.nii`
    - `skin_mask.nii.gz`
- Per repeat dataset root:
  - `population_analysis`
  - `subject_metrics_analysis`
  - `_analysis`
  - generated collection/compression manifests from previous output collection,
    if present and explicitly listed in code.

Cleanup must not recurse through a broad "delete everything except" rule. It
must delete only known generated paths. Missing paths are recorded as skipped,
not errors.

### `CamCan_Experiment/simulation/TI_runner_multi-core.py`

Modify the existing CamCan runner.

Responsibilities:

- Add `--reuse-existing-mesh`.
- In reuse mode:
  - validate normal subject anatomy inputs;
  - resolve an existing head mesh before simulation;
  - skip `charm` and `charm --mesh`;
  - skip meshing cleanup on failure;
  - clean only generated SimNIBS outputs for that subject before rerun.
- Keep the default behavior unchanged when `--reuse-existing-mesh` is absent.
- Continue using `--montage-preset` and `target_montages.py`, so all ROI
  presets in `targets.csv` work without extra code.

Mesh resolution should accept the existing convention
`m2m_<subject>/<subject>.msh`. If necessary for old CamCan outputs, it should
also handle `m2m_sub-<suffix>/<subject>.msh` or `m2m_sub-<suffix>/sub-<suffix>.msh`
with an explicit log event naming the selected mesh.

### `CamCan_Experiment/HPC_scripts/camcan_inplace_rerun_array.slurm`

New Slurm array script.

Responsibilities:

- Require `TI_INPLACE_RERUN_MANIFEST`.
- Read one manifest row per `SLURM_ARRAY_TASK_ID`.
- Export `TI_SIM_ROOT` as the row's `dataset_root`.
- Call `TI_runner_multi-core.py` with:
  - `--subject "$SUBJECT"`
  - `--montage-preset "$TI_MONTAGE_PRESET"`
  - `--reuse-existing-mesh`
- Validate output with `validate_simulation_outputs.py`.
- Use existing SimNIBS module/shim conventions from `my_jobArray.slurm`.
- Keep retry/requeue behavior for incomplete simulations.
- Log task metadata: dataset, repeat, subject, selected montage, mesh path, and
  dataset root.

### `CamCan_Experiment/HPC_scripts/submit_camcan_inplace_rerun.sh`

New submission wrapper for one ROI root.

Responsibilities:

- Accept environment variables:
  - `ROI_ROOT`
  - `MONTAGE_PRESET`
  - `MANIFEST`
  - `LOG_DIR`
  - optional `MAX_CONCURRENT_TASKS`
  - optional `MAX_ARRAY_TASKS`
- Count runnable manifest rows.
- Submit one logical ROI job through one command.
- Split into multiple Slurm chunks only if `MAX_ARRAY_TASKS` requires it.
- Pass task offset to the array script so chunks still cover the single
  manifest.

### Validation Summary

Extend the preflight utility with a `validate` subcommand. The validator should
read the manifest and call the existing `validate_subject_outputs()` logic for
each row, producing:

- total task count
- complete count
- incomplete count
- skipped/missing-input count
- CSV or TSV summary with missing output names and reasons

## Operator Workflow

For one ROI:

1. Run dry-run preflight:

```bash
python CamCan_Experiment/simulation/prepare_inplace_rerun.py \
  preflight \
  --roi-root "$ROI_ROOT" \
  --manifest "$MANIFEST"
```

2. Review the cleanup and task manifests.

3. Apply cleanup:

```bash
python CamCan_Experiment/simulation/prepare_inplace_rerun.py \
  preflight \
  --roi-root "$ROI_ROOT" \
  --manifest "$MANIFEST" \
  --apply
```

4. Submit one array for that ROI:

```bash
ROI_ROOT="$ROI_ROOT" \
MANIFEST="$MANIFEST" \
MONTAGE_PRESET="$MONTAGE_PRESET" \
LOG_DIR="$LOG_DIR" \
bash CamCan_Experiment/HPC_scripts/submit_camcan_inplace_rerun.sh
```

5. Validate all rows from the manifest after completion.

The same command pattern is used for all ROI presets. The code must not restrict
examples or execution to a subset of targets.

## Safety Rules

- Destructive cleanup is never the default.
- Cleanup requires `--apply`.
- Cleanup writes a manifest of deleted and skipped paths.
- Cleanup deletes only known generated outputs, not all unknown files.
- The runner in mesh-reuse mode never invokes `cleanup_subject_mesh_outputs()`.
- The runner validates an existing mesh before starting SimNIBS.
- The Slurm task validates final outputs before considering a task complete.
- Existing unrelated git changes, especially `utils/targets.csv`, are not
  reverted or included in unrelated commits.

## Testing

Unit tests should cover:

- repeat dataset discovery and sorting;
- subject task manifest generation;
- missing input and missing mesh reporting;
- dry-run cleanup reports paths without deleting them;
- apply cleanup deletes only generated paths and preserves anatomy and mesh
  paths;
- runner argument parsing for `--reuse-existing-mesh`;
- mesh resolution across supported `m2m` naming conventions;
- Slurm wrapper task counting and manifest header validation.

Integration or smoke tests should use a temporary miniature dataset with two
repeat folders and two subjects. The test should run preflight dry-run, apply
cleanup, and validate the generated manifest without requiring SimNIBS.

## Open Operational Notes

- The operator decides which `MONTAGE_PRESET` values to run.
- The workflow supports every montage preset currently resolvable from
  `targets.csv`; it does not distinguish finalized and pending ROI targets.
- If a repeat dataset lacks a mesh for a subject, that row should be marked
  blocked in the manifest and not silently remeshed.
