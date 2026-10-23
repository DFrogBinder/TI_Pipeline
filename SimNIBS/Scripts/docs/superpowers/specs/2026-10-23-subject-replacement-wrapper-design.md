# Generic SimNIBS Replacement-Subject Wrapper Design

## Intent

Provide one clean, subject-agnostic workflow for replacing an excluded participant
in the balanced repeatability study. The operator supplies a subject ID once; the
workflow validates it, derives all paths and task scopes from it, and reuses the
existing harmonized SimNIBS 3.2.6 and corrected SimNIBS 4 repeatability pipelines.

## Goals

- Accept any canonical `sub-CC...` ID present in the authoritative final-132
  cohort, without hard-coded subject occurrences in launch code.
- Fail closed unless the required corrected inputs, atlas, and legacy v3
  coordinate-reference scaffold exist for that subject.
- Build, collect, checksum, render, and manually promote the harmonized v3 head
  before any replacement FEM simulation can be released.
- Run the existing two-target, two-condition, 40-repeat design for one subject in
  each SimNIBS version: 160 tasks and 320 component FEM solves per version.
- Keep every replacement input staging area, head model, result root, log,
  receipt, and manifest isolated from the existing full10, base9, and historical
  v3/v4 roots.
- Make preflight read-only and make every state-changing or submitting command
  explicit.

## Non-goals

- Automatically choose which final-132 participant should be the replacement.
- Modify, delete, or append data inside the existing nine-subject or historical
  ten-subject roots.
- Automatically continue after failed v3 head construction or bypass human QC.
- Aggregate the reusable nine subjects with the replacement for manuscript
  analysis in this change; that will be a later provenance-safe 9+1 analysis
  task.
- Support IDs outside the authoritative final-132 cohort.

## Constraints

- The authoritative eligibility list is
  `CamCan_Experiment/cohort_pipeline/cohorts/final_132/subjects.txt`.
- Subject IDs must match `^sub-CC[0-9]{6}$` and occur exactly once in that file;
  path fragments are never accepted as subjects. The candidate must differ from
  the subject it replaces and must not duplicate a retained base-nine subject.
- The existing `full10` and `base9` v3 profiles, defaults, roots, task counts, and
  submission behavior must not change when the new external profile is unused.
- The replacement wrapper must call existing pipeline modules and Slurm workers;
  it must not copy their simulation, meshing, ROI, retry, or fixed-mesh logic.
- Existing SimNIBS modules, resources, retry limit, stimulation definitions, and
  `targets.csv` provenance remain unchanged.
- The corrected-v4 source scaffold may default to the established
  `/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Scaffolds` root. The broader
  legacy-v3 reference root has no verified arbitrary-final-132 default and is
  therefore a mandatory command-line option or environment value until an HPC
  audit establishes one.
- A missing input, mismatched receipt, overlapping root, stale subject manifest,
  or incomplete v3 promotion receipt is a hard error.

## Approaches Considered

### Selected: thin wrapper plus an opt-in external v3 cohort interface

Create `simnibs_subject_replacement/` for the subject manifest, validation,
derived path contract, HPC commands, documentation, and tests. Add a small
generic external-cohort interface to the existing v3 package so it can obtain an
active singleton subject from a validated manifest and calculate its arrays
dynamically. The v4 side calls the existing staged median/fixed pipeline with the
same singleton subject.

This maximizes code reuse, keeps subject-specific policy together, and avoids
changing established profiles.

### Rejected: duplicate the v3 and v4 pipelines

Copying workers and submitters into the replacement folder would isolate the
feature physically but duplicate retry, meshing, ROI, stimulation, and validation
logic. The copies would drift from the production workflows.

### Rejected: hard-code a new subject profile into both pipelines

This would be quick for one candidate but would require code edits after every
failed candidate and would spread one-off subject policy through generic code.

## Architecture

### Subject-specific orchestration folder

`simnibs_subject_replacement/` will contain:

- `replacement.py`: validates `--subject`, authoritative membership, source
  inputs, root isolation, receipts, and stage gates; emits a versioned resolved
  manifest and numerical scope.
- `settings.py`: central path templates and constants; every writable root
  includes the canonical subject ID.
- `hpc/submit.sh`: one copy-paste entry point with explicit modes for preflight,
  staging, v3 head construction/collection/promotion, v3 simulations, v4
  initialization/simulations, and status.
- `README.md`: ordered runbook and evidence required at each gate.
- `tests/`: subject validation, path isolation, task-count, gate, and launcher
  regression tests.

No source file contains `sub-CC510415` or any other candidate ID as executable
configuration. The selected subject and the subject it replaces are runtime
arguments, so changing candidate requires changing one shell variable rather
than editing source. Examples may use placeholders such as `$SUBJECT` and
`$REPLACES`.

Read-only preflight constructs and prints the candidate manifest in memory.
The explicit preparation command persists that exact manifest before any
staging or submission; all state-changing modes require and revalidate it.

### Generic v3 extension

The harmonized v3 package will gain an opt-in `external1` cohort profile. It
requires the replacement wrapper's resolved manifest and permits exactly one
validated final-132 subject. Subject lists, collectors, promotion directories,
array bounds, ROI extraction arrays, and task counts will use the active cohort
rather than fixed constants.

The external profile will be rejected when invoked directly without a valid
manifest. Existing `full10` and `base9` code paths retain their current defaults
and roots.

### Existing v4 reuse

The wrapper will stage the selected subject into an isolated corrected-input
root using the existing final-132 staging utility, then initialize the existing
`ti_current_repair/pipeline/staged_median_fixed_experiment.py` once for each
target. Each v4 experiment receives a singleton subject list and isolated root.

### Gates and execution order

1. `preflight`: read-only membership, input, atlas, legacy-reference, root, hash,
   and numerical-scope audit; submits nothing.
2. `stage`: stage the corrected singleton input and v3 reference allowlist with
   receipts.
3. `v3-headmodel`: one harmonized-label/head-model task and collector.
4. `v3-promote`: requires manifest hash plus explicit confirmation of human QC.
5. Only after promotion, initialize and release the v3 replacement simulations.
6. Only after the same promotion gate, initialize and release the matching v4
   replacement simulations.
7. Within each version, right-M1 submission is a second explicit operator
   command. The wrapper accepts it only after validating the left-hippocampus
   terminal receipt, preserving the established maximum of 50 simultaneous
   simulation tasks without hidden scheduler-side submission.
8. Status reports each version and target independently; it never interprets a
   submitted job as completed work.

Singleton post-processing retains the existing per-run and paired ROI metrics
needed by the later 9+1 aggregation. It does not treat one-subject summaries or
figures as cohort-level inference.

For each version the full replacement scope is two targets multiplied by two
conditions multiplied by 40 repeats: 160 simulation tasks, 160 expected field
meshes, and 320 component FEM solves. Across both versions the scope is 320
simulation tasks and 640 component FEM solves.

## Failure Behavior

- A candidate that lacks any required input is rejected before staging.
- A v3 meshing failure records evidence but cannot release simulations or v4.
- Re-running with another eligible subject only changes `--subject`; all derived
  roots and manifests change with it automatically.
- Existing artifacts for the same subject are reusable only when their receipts,
  roots, subject ID, configuration, and hashes match exactly.
- Existing non-empty, unmanifested output roots are never adopted or overwritten.

## Success Criteria

- The wrapper accepts a valid final-132 subject and rejects malformed,
  non-member, duplicated, and path-traversal values.
- Preflight reports one reference task, one v3 head-model task, 160 tasks per
  version, 320 tasks overall, and no submission.
- All generated paths contain the supplied subject and are disjoint from every
  protected production root.
- V3 head promotion is mandatory before either version's FEM submission mode.
- Existing v3 full10/base9 and v4 pipeline tests remain green.
- Shell launchers pass `bash -n`, and dry-run launcher tests prove the expected
  arrays, roots, dependencies, and zero unintended submissions.
