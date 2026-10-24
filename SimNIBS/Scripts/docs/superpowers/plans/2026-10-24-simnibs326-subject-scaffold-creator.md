# SimNIBS 3.2.6 Subject Scaffold Creator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a generic, auditable one-subject SimNIBS 3.2.6 `headreco` scaffold creator, initially for `sub-CC510415`, while retaining the complete diagnostic output.

**Architecture:** Extract the existing tested scaffold build into a request-driven shared core and keep the balanced-10 runner as a compatible adapter. Add a separate subject-facing package that validates final-132 corrected-source evidence, owns an isolated output tree, and submits a one-element Slurm chain using the existing SimNIBS 3.2.6 module and CAT12 preflight.

**Tech Stack:** Python 3.11+, pytest, Bash, Slurm, SimNIBS 3.2.6 `headreco`, MATLAB 2023b, SHA-256 JSON receipts

**Spec:** `docs/superpowers/specs/2026-10-24-simnibs326-subject-scaffold-creator-design.md`

## Global Constraints

- Pin the runtime to `SimNIBS/3.2.6-foss-2023a`; never load or invoke SimNIBS 4.
- Use `MATLAB/2023b` through the existing `SIMNIBS326_MATLAB_MODULE` override.
- Reuse the existing hash-pinned CAT12 compatibility overlay; do not duplicate or weaken it.
- Accept a canonical runtime subject ID and never hardcode `sub-CC510415` into executable code.
- Retain the complete successful `headreco all --noclean` output for diagnosis.
- Keep preflight and status read-only; require exact output-root confirmation before submission.
- Preserve protected source and production roots; fail closed on overlap or unowned output.
- Use a one-element Slurm array, one exact-element retry, immutable attempt logs, and append-only scheduler logs.
- Do not submit any HPC jobs while implementing or testing this feature.

## Review Focus

- Corrected-v4 source files may be gzip-compressed and named through a receipt; resolve the receipt rather than assuming flat filenames, and test both copy and gzip-backed records.
- A prior failed run may leave a populated subject tree; only a matching creator ownership receipt may authorize replacement, and test both owned and unowned partial trees.
- A ready marker can exist beside changed inputs or meshes; validation must detect stale input and mesh hashes rather than trusting marker presence.
- Module preflight evidence can be stale or from another runtime; the worker must validate the existing receipt immediately before `headreco`.
- Scheduler output can include cluster suffixes or warnings; parse only a valid numeric `sbatch --parsable` job ID and fail on empty or malformed responses.

---

### Task 1: Extract a request-driven shared scaffold core

**Files:**
- Create: `simnibs_326_repeatability/scaffold_core.py`
- Modify: `simnibs_326_repeatability/scaffold_runner.py`
- Modify: `simnibs_326_repeatability/tests/test_simnibs_326_pipeline.py`
- Test: `simnibs_326_repeatability/tests/test_scaffold_core.py`

**Interfaces:**
- Consumes: Existing `simnibs_326_repeatability/cat12_compat.py`, its `PATCH_ID`, and the balanced-10 settings adapter.
- Produces:
  - `ScaffoldRequest(subject: str, t1: Path, t2: Path, corrected_labels: Path, scaffold_root: Path, simnibs_module: str, cat12_compat_script: Path, cat12_patch_id: str, head_model_strategy: str, segmentation_provenance: str, workflow: str)`;
  - `ScaffoldPaths(anat: Path, m2m: Path, native_mesh: Path, compatibility_mesh: Path, ready_marker: Path)`;
  - `scaffold_paths(request: ScaffoldRequest) -> ScaffoldPaths`;
  - `validate_scaffold(request: ScaffoldRequest) -> dict[str, object]`; and
  - `build_scaffold(request: ScaffoldRequest) -> dict[str, object]`.

- [ ] **Step 1: Write failing shared-core tests**

  Add tests proving that an arbitrary canonical subject request constructs the exact
  `headreco all --noclean` command, retains generated diagnostic files, creates equal
  native/compatibility meshes, writes the pinned module and input hashes, and reports
  idempotent reuse. Mock only the external `headreco` process and SimNIBS version lookup.

- [ ] **Step 2: Write failure-path tests for the shared core**

  Cover missing or empty inputs, fewer than five STL surfaces, empty `toMNI` or
  `eeg_positions`, a stale marker input hash, differing mesh hashes, and a partial build
  that never produces its native mesh.

- [ ] **Step 3: Run the new tests and verify they fail**

  Run: `pytest -q simnibs_326_repeatability/tests/test_scaffold_core.py`

  Expected: FAIL because `scaffold_core.py` and its interfaces do not exist.

- [ ] **Step 4: Implement the shared core**

  Define immutable `ScaffoldRequest` and `ScaffoldPaths` dataclasses. Accept explicit
  subject, T1, T2, corrected-label, scaffold root, module, CAT12 runner/patch, strategy,
  provenance, and workflow values. Keep the existing layout and marker schema compatible,
  use non-empty-file and non-empty-directory checks, and bind validation to marker hashes.

- [ ] **Step 5: Convert the balanced-10 runner into a backwards-compatible adapter**

  Preserve `plan`, `build --subject-index`, `build --subject`, `validate`, existing output
  roots, and existing marker fields. Replace the fixed `choices=SUBJECTS` restriction with
  canonical subject parsing while keeping subject-index behavior pinned to the original ten.

- [ ] **Step 6: Run shared-core and historical campaign tests**

  Run: `pytest -q simnibs_326_repeatability/tests/test_scaffold_core.py simnibs_326_repeatability/tests/test_simnibs_326_pipeline.py`

  Expected: PASS with no changed campaign counts or runtime module values.

- [ ] **Step 7: Commit the shared core**

  ```bash
  git add simnibs_326_repeatability/scaffold_core.py \
      simnibs_326_repeatability/scaffold_runner.py \
      simnibs_326_repeatability/tests/test_scaffold_core.py \
      simnibs_326_repeatability/tests/test_simnibs_326_pipeline.py
  git commit -m "refactor: share SimNIBS 3.2.6 scaffold builder"
  ```

### Task 2: Implement generic subject and source validation

**Files:**
- Create: `simnibs_326_subject_scaffold_creator/__init__.py`
- Create: `simnibs_326_subject_scaffold_creator/__main__.py`
- Create: `simnibs_326_subject_scaffold_creator/settings.py`
- Create: `simnibs_326_subject_scaffold_creator/creator.py`
- Create: `simnibs_326_subject_scaffold_creator/tests/__init__.py`
- Create: `simnibs_326_subject_scaffold_creator/tests/conftest.py`
- Test: `simnibs_326_subject_scaffold_creator/tests/test_creator.py`

**Interfaces:**
- Consumes: Task 1's `ScaffoldRequest`, `build_scaffold`, and `validate_scaffold`; `simnibs_subject_replacement.staging.audit_corrected_sources` for the established final-132 corrected-source receipt format.
- Produces:
  - `CreatorConfig(subject: str, corrected_source_root: Path, output_root: Path, final132_subjects_file: Path)`;
  - `CreatorPaths(subject_anat: Path, module_receipt: Path, owner_receipt: Path, completion_receipt: Path, log_dir: Path)`;
  - `creator_paths(config: CreatorConfig) -> CreatorPaths`;
  - `validate_subject(config: CreatorConfig) -> tuple[str, ...]`;
  - `preflight(config: CreatorConfig) -> dict[str, object]`;
  - `build(config: CreatorConfig) -> dict[str, object]`;
  - `validate(config: CreatorConfig) -> dict[str, object]`;
  - `status(config: CreatorConfig) -> dict[str, object]`; and
  - `main(argv: list[str] | None = None) -> int` through `python -m simnibs_326_subject_scaffold_creator`.

- [ ] **Step 1: Write failing subject-validation tests**

  Test canonical `sub-CC######` parsing, exactly-once membership in the authoritative
  final-132 list, rejection of duplicates and invalid list entries, and acceptance of
  `sub-CC510415` only through runtime input.

- [ ] **Step 2: Write failing corrected-source and isolation tests**

  Test exact T1/T2/label resolution from valid copy and gzip source receipts, missing and
  hash-mismatched files, output/source overlap, protected production-root overlap, and the
  default isolated output root.

- [ ] **Step 3: Write failing ownership and validation tests**

  Assert preflight creates no files; unowned or mismatched existing output fails closed;
  an owned partial tree can be rebuilt; a complete request writes bound `owner.json` and
  `complete.json`; ready reuse is idempotent; and stale inputs, marker, or meshes invalidate
  completion.

- [ ] **Step 4: Run the creator tests and verify they fail**

  Run: `pytest -q simnibs_326_subject_scaffold_creator/tests/test_creator.py`

  Expected: FAIL because the creator package does not exist.

- [ ] **Step 5: Implement settings and path isolation**

  Define `CreatorConfig` with subject, corrected source, output root, and authoritative
  subjects file. Define `CreatorPaths` for subject output, module receipt, owner receipt,
  completion receipt, and logs. Pin exact default paths and protected roots from the spec.

- [ ] **Step 6: Implement preflight, owned build, validation, and status**

  Reuse the corrected-source audit and Task 1 shared core. Use atomic JSON writes. Create
  the ownership receipt outside the rebuildable `anat` tree before the first build, and
  require an exact match before deleting/rebuilding any partial subject output.

- [ ] **Step 7: Implement the CLI**

  Add `preflight`, worker-only `build`, `validate`, and `status` subcommands with common
  runtime path flags. Emit structured JSON and return nonzero for incomplete or invalid
  state without mutating during read-only commands.

- [ ] **Step 8: Run creator and replacement tests**

  Run: `pytest -q simnibs_326_subject_scaffold_creator/tests/test_creator.py simnibs_subject_replacement/tests`

  Expected: PASS, including replacement legacy-reference validation.

- [ ] **Step 9: Commit the creator core**

  ```bash
  git add simnibs_326_subject_scaffold_creator
  git commit -m "feat: add generic SimNIBS 3.2.6 scaffold creator"
  ```

### Task 3: Add guarded one-subject Slurm submission

**Files:**
- Create: `simnibs_326_subject_scaffold_creator/hpc/scaffold_array.slurm`
- Create: `simnibs_326_subject_scaffold_creator/hpc/submit.sh`
- Create: `simnibs_326_subject_scaffold_creator/operational_profile.json`
- Test: `simnibs_326_subject_scaffold_creator/tests/test_hpc_launcher.py`

**Interfaces:**
- Consumes: Task 2's CLI; existing `simnibs_326_repeatability/hpc/module_preflight.slurm`, `activate_compat.sh`, and `pipeline.py validate-module-receipt` command.
- Produces: User-facing `preflight`, `module-check`, `submit`, `status`, and `validate` shell modes, plus an auditable candidate HPC profile.

- [ ] **Step 1: Write failing launcher contract tests**

  Assert exact module strings, `0-0%1`, Sheffield/8 CPU/32G/8h resources, 7-hour timeout,
  one retry, exact array-element requeue, append-only Slurm output, immutable attempt logs,
  required subject/output confirmation flags, and absence of SimNIBS 4 strings.

- [ ] **Step 2: Write failing stub-scheduler tests**

  With fake `sbatch`, test numeric `--parsable` responses including a cluster suffix,
  malformed/empty responses, dependency wiring, module-check-to-scaffold ordering, and that
  `preflight`, `status`, and `validate` never call `sbatch`.

- [ ] **Step 3: Run launcher tests and verify they fail**

  Run: `pytest -q simnibs_326_subject_scaffold_creator/tests/test_hpc_launcher.py`

  Expected: FAIL because the HPC launchers do not exist.

- [ ] **Step 4: Implement the singleton scaffold worker**

  Load only `MATLAB/2023b` and `SimNIBS/3.2.6-foss-2023a`, activate the existing
  compatibility environment, validate the module receipt immediately before build, and
  invoke the Task 2 worker with all paths passed explicitly through the environment.

- [ ] **Step 5: Implement the guarded submission wrapper**

  Parse subject and path flags once. Require `--confirm-output-root` for `module-check` and
  `submit`. Run read-only preflight before `sbatch`. Reuse the existing module-preflight
  Slurm script, then submit the scaffold array with `afterok` dependency and robust job-ID
  parsing.

- [ ] **Step 6: Record the candidate operational profile**

  Pin the module, MATLAB dependency, resources, roots, subject-runtime contract, retry
  semantics, validation commands, and `not_submitted` status. Do not include a fabricated
  job ID or claim runtime success before Stanage evidence exists.

- [ ] **Step 7: Verify shell and launcher behavior**

  Run: `bash -n simnibs_326_subject_scaffold_creator/hpc/submit.sh simnibs_326_subject_scaffold_creator/hpc/scaffold_array.slurm`

  Run: `pytest -q simnibs_326_subject_scaffold_creator/tests/test_hpc_launcher.py`

  Expected: both commands PASS.

- [ ] **Step 8: Commit HPC support**

  ```bash
  git add simnibs_326_subject_scaffold_creator/hpc \
      simnibs_326_subject_scaffold_creator/operational_profile.json \
      simnibs_326_subject_scaffold_creator/tests/test_hpc_launcher.py
  git commit -m "feat: submit one-subject SimNIBS 3.2.6 scaffolds"
  ```

### Task 4: Document the diagnostic-retention and replacement handoff

**Files:**
- Create: `simnibs_326_subject_scaffold_creator/README.md`
- Modify: `simnibs_subject_replacement/README.md`
- Test: `simnibs_326_subject_scaffold_creator/tests/test_hpc_launcher.py`

**Interfaces:**
- Consumes: Tasks 2 and 3 command names, defaults, receipts, and output layout.
- Produces: Copy-paste-safe commands for generating and validating a subject scaffold and passing it to the existing v3/v4 replacement workflow.

- [ ] **Step 1: Add documentation assertions**

  Require the README to name `SimNIBS/3.2.6-foss-2023a`, `MATLAB/2023b`, full diagnostic
  retention, no automatic FEM submission, the default isolated root, and the exact
  `--legacy-v3-root` handoff.

- [ ] **Step 2: Write the creator runbook**

  Document read-only preflight, guarded submission, job/accounting checks, validation,
  diagnostic paths, and the exact `sub-CC510415` example as shell input rather than code.
  State explicitly that scheduler submission is not completion.

- [ ] **Step 3: Update the replacement runbook**

  Add the scaffold-creator prerequisite and set `LEGACY_V3_ROOT` to the validated creator
  output root. Keep the replacement pipeline's v3 QC/promotion gate and v3/v4 simulation
  commands unchanged.

- [ ] **Step 4: Run documentation and focused integration tests**

  Run: `pytest -q simnibs_326_subject_scaffold_creator/tests simnibs_326_repeatability/tests/test_scaffold_core.py simnibs_subject_replacement/tests`

  Expected: PASS.

- [ ] **Step 5: Commit the runbook**

  ```bash
  git add simnibs_326_subject_scaffold_creator/README.md \
      simnibs_326_subject_scaffold_creator/tests/test_hpc_launcher.py \
      simnibs_subject_replacement/README.md
  git commit -m "docs: add SimNIBS 3.2.6 scaffold runbook"
  ```

### Task 5: Run complete verification without submitting work

**Files:**
- Modify only files required by failures uncovered during verification.

**Interfaces:**
- Consumes: All prior tasks.
- Produces: A tested implementation ready to push and a separate set of Stanage commands for the user to run.

- [ ] **Step 1: Run all affected Python tests**

  Run: `pytest -q simnibs_326_subject_scaffold_creator/tests simnibs_326_repeatability/tests simnibs_subject_replacement/tests`

  Expected: PASS.

- [ ] **Step 2: Run shell syntax and static checks**

  Run: `bash -n simnibs_326_subject_scaffold_creator/hpc/submit.sh simnibs_326_subject_scaffold_creator/hpc/scaffold_array.slurm`

  Run: `git diff --check`

  Expected: both commands PASS with no whitespace errors.

- [ ] **Step 3: Exercise a local fixture-only dry run**

  Run the creator `preflight`, `status`, and `validate` commands against test fixtures and
  a fake scheduler. Confirm no real `/mnt/parscratch` path is created and no real `sbatch`
  command is invoked.

- [ ] **Step 4: Review the final diff against the spec**

  Confirm generic runtime subject handling, exact SimNIBS 3.2.6 labeling, full diagnostic
  retention, source/root isolation, bounded retries, no hidden submission, and unchanged
  replacement v3/v4 scientific scope.

- [ ] **Step 5: Commit any verification-only corrections**

  ```bash
  git add <only-the-corrected-files>
  git commit -m "test: verify SimNIBS 3.2.6 scaffold workflow"
  ```
