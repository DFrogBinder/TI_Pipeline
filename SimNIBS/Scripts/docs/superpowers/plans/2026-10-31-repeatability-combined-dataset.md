# Combined Repeatability Dataset Consolidation and Retirement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed, resumable workflow that creates and checksum-validates one ten-subject physical SimNIBS 3.2.6/4 repeatability dataset, archives obsolete v3 references to Shared, and separately retires only exact verified source roots.

**Architecture:** Add a dependency-free Python package that converts code-owned source definitions into an immutable copy plan, merges only subject-condition scientific subtrees into canonical experiment roots, and preserves historical metadata separately. A thin login-node shell wrapper exposes read-only preflight/status, copy/verification, Shared archive, and dry-run-by-default retirement modes; permanent deletion is delegated to the existing hardened one-directory-per-job launcher only after receipt gates pass.

**Tech Stack:** Python 3.9+ standard library, Bash, rsync, Slurm submission through the existing deletion launcher, pytest.

**Spec:** `docs/superpowers/specs/2026-10-30-repeatability-combined-dataset-design.md`

## Global Constraints

- Production dataset root: `/mnt/parscratch/users/cop23bi/final_132_repeatability_combined_dataset_v1`.
- Production Shared archive root: `/shared/boyan/Shared/retired_repeatability_v3_sources_v1`.
- Canonical cohort is the retained base nine with `sub-CC510415` replacing `sub-CC520083`; `sub-CC520083` is forbidden from the canonical scientific tree.
- Full scientific scope is 10 subjects, 2 versions, 2 targets, 2 conditions, 40 repeats, 80 subject-condition cells, and 3,200 simulation records.
- Copying and Shared archival are login-node operations; no SimNIBS module or Slurm simulation task is introduced.
- Source roots are read-only; copy and checksum modes never delete source data.
- Dataset completion requires full-content equality, per-file destination SHA-256 inventory, terminal scientific receipts, exact counts, and zero scientific-tree symlinks.
- Retirement is a separate dry-run-by-default stage and can select only code-owned allowlist keys.
- Shared remains login-node-only and is never substituted into SimNIBS worker paths.
- The global `CamCan_Corrected_v4_Scaffolds` root is protected; only the ten required subject subtrees and provenance records are copied.
- Existing migration, deletion, repeatability, and replacement defaults remain unchanged.

## Review Focus

- A source condition with 39 or 41 repeats, a missing `TI.msh`, or a non-terminal completion receipt must block plan creation; Task 2 tests this.
- A collision where two sources map to one canonical destination must fail before creating the dataset root; Task 1 tests this.
- A symlink escaping approved source roots, a broken link, or a link left in the canonical destination must block completion; Tasks 1 and 3 test these cases.
- A valid-looking but stale/tampered dataset or Shared completion receipt must block retirement; Task 5 tests both receipt hashes and live revalidation.
- Successful deletion submission without terminal receipts or independent target absence must remain `submitted`/`incomplete`, never `complete`; Task 5 tests this.

---

### Task 1: Code-owned dataset scope and immutable copy-plan model

**Files:**
- Create: `repeatability_dataset_consolidation/__init__.py`
- Create: `repeatability_dataset_consolidation/settings.py`
- Create: `repeatability_dataset_consolidation/planning.py`
- Create: `repeatability_dataset_consolidation/tests/conftest.py`
- Create: `repeatability_dataset_consolidation/tests/test_settings.py`
- Create: `repeatability_dataset_consolidation/tests/test_planning.py`

**Interfaces:**
- Produces: immutable `DatasetConfig`, `SourceRoots`, `CopyItem`, and `ConsolidationPlan` dataclasses; `production_config() -> DatasetConfig`; `build_copy_plan(config: DatasetConfig) -> ConsolidationPlan`; `plan_sha256(plan: ConsolidationPlan) -> str`; `retirement_allowlist(config: DatasetConfig) -> dict[str, Path]`.
- Consumes: the exact subject order, source groups, destinations, and retirement policy from the spec.

- [ ] **Step 1: Write failing settings tests**

  Assert the exact ten-subject order, output/archive roots, protected v4 root, four v3 archive roots, consolidated retirement keys, 80 cells, 3,200 records, and absence of `sub-CC520083`.

- [ ] **Step 2: Run settings tests and verify RED**

  Run: `python3 -m pytest -q repeatability_dataset_consolidation/tests/test_settings.py`

  Expected: import failure because the package does not exist.

- [ ] **Step 3: Implement immutable production settings**

  Add frozen dataclasses and code-owned source/retirement definitions. Accept alternate roots and repeat counts only through direct dataclass construction for tests; the production CLI uses `production_config()`.

- [ ] **Step 4: Write failing copy-plan tests**

  Build a miniature two-repeat fixture and assert unique source/destination mappings for all version-target-condition-subject cells, v3/v4 head-model bundles, ten v4 provenance records, complete nested/paper-analysis trees, and root-level metadata. Test duplicate destinations, source/destination overlap, traversal, broken links, and symlinks resolving outside approved source roots.

- [ ] **Step 5: Run planning tests and verify RED**

  Run: `python3 -m pytest -q repeatability_dataset_consolidation/tests/test_planning.py`

  Expected: failures because copy-plan construction and validation are absent.

- [ ] **Step 6: Implement deterministic plan construction**

  Implement typed plan items with exact role, source, destination, required state, subject/version/target/condition metadata, copy policy, and source identity. Enumerate non-subject root metadata as separate provenance items; materialize only validated in-scope symlinks in canonical scientific items.

- [ ] **Step 7: Run Task 1 tests**

  Run: `python3 -m pytest -q repeatability_dataset_consolidation/tests/test_settings.py repeatability_dataset_consolidation/tests/test_planning.py`

  Expected: all tests pass.

- [ ] **Step 8: Commit**

  Run: `git add SimNIBS/Scripts/repeatability_dataset_consolidation && git commit -m "Add repeatability dataset consolidation plan model"`

### Task 2: Read-only scientific preflight and owned dataset preparation

**Files:**
- Create: `repeatability_dataset_consolidation/validation.py`
- Create: `repeatability_dataset_consolidation/workflow.py`
- Create: `repeatability_dataset_consolidation/tests/test_validation.py`
- Create: `repeatability_dataset_consolidation/tests/test_workflow_prepare.py`

**Interfaces:**
- Consumes: Task 1 `DatasetConfig` and `ConsolidationPlan`.
- Produces: `preflight(config: DatasetConfig) -> dict[str, object]`; `prepare(config: DatasetConfig) -> dict[str, object]`; `validate_scientific_scope(plan: ConsolidationPlan) -> ScopeValidation`; atomic `_consolidation/owner.json`, `planned_items.json`, `dataset_manifest.json`, `subjects.txt`, and `README.md`.

- [ ] **Step 1: Write failing scientific validation tests**

  Assert exact repeat tags `repeat_001` through `repeat_040` in production semantics, final `TI.msh` presence, terminal source receipt binding, complete 80-cell scope, selected v3 head-model artifacts, selected v4 scaffold/provenance artifacts, and explicit rejection of excluded-subject scientific paths. Include 39/41-repeat, duplicate-tag, missing-mesh, incomplete-receipt, and missing-head-model cases.

- [ ] **Step 2: Run validation tests and verify RED**

  Run: `python3 -m pytest -q repeatability_dataset_consolidation/tests/test_validation.py`

  Expected: failures because scope validation is missing.

- [ ] **Step 3: Implement read-only preflight validation**

  Validate real paths, source identities, completion receipts, repeat trees, head-model/scaffold requirements, analysis roots, canonical mapping collisions, and free-space reporting without creating the destination.

- [ ] **Step 4: Write failing preparation/ownership tests**

  Assert `prepare` creates the exact skeleton and atomic plan/owner records only after preflight; reuses an exact owned state; rejects unowned, non-empty, differently planned, symlinked, source-overlapping, and partially owned roots; and makes no source changes.

- [ ] **Step 5: Run preparation tests and verify RED**

  Run: `python3 -m pytest -q repeatability_dataset_consolidation/tests/test_workflow_prepare.py`

  Expected: failures because owned preparation is missing.

- [ ] **Step 6: Implement atomic preparation**

  Bind owner and plan receipts to schema version, Git commit, production root, subject list, source identities, and plan hash. Generate the human-readable dataset README and manifest without copying scientific data.

- [ ] **Step 7: Run Task 2 tests**

  Run: `python3 -m pytest -q repeatability_dataset_consolidation/tests/test_validation.py repeatability_dataset_consolidation/tests/test_workflow_prepare.py`

  Expected: all tests pass.

- [ ] **Step 8: Commit**

  Run: `git add SimNIBS/Scripts/repeatability_dataset_consolidation && git commit -m "Add fail-closed dataset preparation"`

### Task 3: Resumable physical copy, verification, and dataset completion

**Files:**
- Modify: `repeatability_dataset_consolidation/workflow.py`
- Create: `repeatability_dataset_consolidation/rsync.py`
- Create: `repeatability_dataset_consolidation/inventory.py`
- Create: `repeatability_dataset_consolidation/tests/test_copy_verify.py`
- Create: `repeatability_dataset_consolidation/tests/test_completion.py`

**Interfaces:**
- Consumes: Task 2 owned plan and destination skeleton.
- Produces: `copy_dataset(config: DatasetConfig) -> dict[str, object]`; `verify_dataset(config: DatasetConfig, *, checksum: bool) -> dict[str, object]`; `complete_dataset(config: DatasetConfig) -> dict[str, object]`; `build_file_inventory(root: Path) -> FileInventory`; atomic copy/verification receipts and `_consolidation/complete.json`.

- [ ] **Step 1: Write failing copy/verification tests**

  Assert per-item resumable rsync, physical symlink materialization, no destination-root `--delete`, exact preservation of file bytes, safe reuse of matching items, detection of changed/missing/extra destination files, and no source mutation. Verify size/mtime and full-checksum modes are distinct.

- [ ] **Step 2: Run copy tests and verify RED**

  Run: `python3 -m pytest -q repeatability_dataset_consolidation/tests/test_copy_verify.py`

  Expected: failures because rsync execution and verification are absent.

- [ ] **Step 3: Implement per-item resumable copy and comparison**

  Build argument arrays without shell evaluation, use `rsync -rltH --copy-links --partial --protect-args` only after preflight has validated every link target, compare each item read-only with itemized changes, and write atomic per-item status. Revalidate source identity immediately before each copy and verification.

- [ ] **Step 4: Write failing completion tests**

  Assert completion requires all plan items checksum-matched, 10 subjects, 80 cells, 3,200 final `TI.msh` records, no canonical symlinks, no excluded subject, all manifests/receipts, and a sorted per-file SHA-256 inventory. Test tampered copy status, changed source, stale plan, missing replacement, excluded-subject injection, and inventory mismatch.

- [ ] **Step 5: Run completion tests and verify RED**

  Run: `python3 -m pytest -q repeatability_dataset_consolidation/tests/test_completion.py`

  Expected: failures because completion and inventory binding are absent.

- [ ] **Step 6: Implement inventory and atomic completion receipt**

  Hash every destination regular file, bind the inventory and source mapping hashes into `complete.json`, and refuse to overwrite an existing nonmatching completion receipt.

- [ ] **Step 7: Run Task 3 tests**

  Run: `python3 -m pytest -q repeatability_dataset_consolidation/tests/test_copy_verify.py repeatability_dataset_consolidation/tests/test_completion.py`

  Expected: all tests pass.

- [ ] **Step 8: Commit**

  Run: `git add SimNIBS/Scripts/repeatability_dataset_consolidation && git commit -m "Add checksummed dataset consolidation"`

### Task 4: Login-node archive for obsolete v3 references

**Files:**
- Create: `repeatability_dataset_consolidation/archive.py`
- Create: `repeatability_dataset_consolidation/tests/test_archive.py`

**Interfaces:**
- Consumes: four exact legacy-v3 source roots from Task 1.
- Produces: `archive_preflight(config: DatasetConfig) -> dict[str, object]`; `archive_copy(config: DatasetConfig) -> dict[str, object]`; `archive_verify(config: DatasetConfig, *, checksum: bool) -> dict[str, object]`; immutable `/shared/boyan/Shared/.repeatability_retirement_receipts/legacy_v3_archive_v1.complete.json`.

- [ ] **Step 1: Write failing archive tests**

  Assert exact four-root scope, resumable whole-tree copying, symlink/hard-link preservation suitable for an archive, size/mtime then checksum verification, independent receipt directory, exact destination confirmation, and rejection of missing/unowned/different archive roots.

- [ ] **Step 2: Run archive tests and verify RED**

  Run: `python3 -m pytest -q repeatability_dataset_consolidation/tests/test_archive.py`

  Expected: failures because archive operations are absent.

- [ ] **Step 3: Implement the archive lifecycle**

  Reuse the established login-node rsync profile (`-rltH`, no owner/group/mode verification, partial transfers), validate source identities, and write completion only when all four roots checksum-match.

- [ ] **Step 4: Run archive tests**

  Run: `python3 -m pytest -q repeatability_dataset_consolidation/tests/test_archive.py`

  Expected: all tests pass.

- [ ] **Step 5: Commit**

  Run: `git add SimNIBS/Scripts/repeatability_dataset_consolidation && git commit -m "Add legacy v3 Shared archive workflow"`

### Task 5: Receipt-gated retirement and terminal-status proof

**Files:**
- Create: `repeatability_dataset_consolidation/retirement.py`
- Create: `repeatability_dataset_consolidation/tests/test_retirement.py`

**Interfaces:**
- Consumes: dataset and archive completion receipts, Task 1 allowlist keys, current scheduler state, and existing `CamCan_Experiment/HPC_scripts/submit_delete_directory.sh`.
- Produces: `plan_retirement(config: DatasetConfig, selected: Sequence[str] | None) -> RetirementPlan`; `submit_retirement(plan: RetirementPlan, confirmations: RetirementConfirmations) -> dict[str, object]`; `retirement_status(config: DatasetConfig) -> dict[str, object]`; external request and terminal summary receipts.

- [ ] **Step 1: Write failing retirement-plan tests**

  Assert dry-run default, exact allowlist paths, subset selection by key only, protected-root rejection, dataset receipt/hash binding for every source, additional Shared receipt/hash binding for archived sources, source identity revalidation, wrong confirmation rejection, and active-job conflict rejection.

- [ ] **Step 2: Run retirement-plan tests and verify RED**

  Run: `python3 -m pytest -q repeatability_dataset_consolidation/tests/test_retirement.py -k 'plan or gate or confirm'`

  Expected: failures because retirement planning is absent.

- [ ] **Step 3: Implement fail-closed retirement planning**

  Accept allowlist keys rather than paths, rehash immutable receipts, verify copied source coverage, inspect current `squeue` output for selected-root bindings, and emit an exact immutable deletion request without side effects.

- [ ] **Step 4: Write failing submission/status tests**

  Mock the hardened deletion launcher and assert one noninteractive submission per selected target with exact target/confirm/allowed-root/external-log arguments. Assert submitted jobs are not complete; terminal success requires each generic deletion receipt plus independent target absence; failed/missing receipts and still-present paths remain incomplete.

- [ ] **Step 5: Run submission/status tests and verify RED**

  Run: `python3 -m pytest -q repeatability_dataset_consolidation/tests/test_retirement.py -k 'submit or status'`

  Expected: failures because submission and status proof are absent.

- [ ] **Step 6: Implement submission and terminal summary**

  Delegate deletion without duplicating `rm` logic, record job IDs and receipt paths outside all targets, and atomically write a retirement completion summary only after terminal evidence and absence checks pass.

- [ ] **Step 7: Run Task 5 tests**

  Run: `python3 -m pytest -q repeatability_dataset_consolidation/tests/test_retirement.py`

  Expected: all tests pass.

- [ ] **Step 8: Commit**

  Run: `git add SimNIBS/Scripts/repeatability_dataset_consolidation && git commit -m "Add receipt-gated dataset retirement"`

### Task 6: Login-node CLI, runbook, and cross-workflow verification

**Files:**
- Create: `repeatability_dataset_consolidation/cli.py`
- Create: `repeatability_dataset_consolidation/__main__.py`
- Create: `repeatability_dataset_consolidation/hpc/submit.sh`
- Create: `repeatability_dataset_consolidation/README.md`
- Create: `repeatability_dataset_consolidation/tests/test_cli.py`
- Create: `repeatability_dataset_consolidation/tests/test_hpc_launcher.py`
- Modify: `docs/superpowers/specs/2026-10-30-repeatability-combined-dataset-design.md`

**Interfaces:**
- Consumes: Tasks 1-5 APIs.
- Produces: modes `preflight`, `prepare`, `copy`, `verify`, `checksum`, `status`, `archive-preflight`, `archive-copy`, `archive-verify`, `archive-checksum`, `retire-plan`, `retire-submit`, and `retire-status`; copy-paste-safe Stanage runbook.

- [ ] **Step 1: Write failing CLI and shell-wrapper tests**

  Assert all modes dispatch correctly, the production paths and complete 3,200-record scope are printed, read-only modes make no writes, copy requires exact dataset confirmation, archive copy requires exact Shared confirmation, retirement defaults to plan-only, retirement submission requires receipt hashes and `--confirm-retirement`, and no mode silently invokes Slurm simulations.

- [ ] **Step 2: Run interface tests and verify RED**

  Run: `python3 -m pytest -q repeatability_dataset_consolidation/tests/test_cli.py repeatability_dataset_consolidation/tests/test_hpc_launcher.py`

  Expected: failures because CLI and shell wrapper are absent.

- [ ] **Step 3: Implement CLI, wrapper, and runbook**

  Use argument arrays and exact confirmations, print a numerical Scope block before executable operations, document tmux use for long login-node copies, and document that checksum completion precedes—but never automatically triggers—retirement.

- [ ] **Step 4: Run package tests and syntax checks**

  Run: `bash -n repeatability_dataset_consolidation/hpc/submit.sh && python3 -m pytest -q repeatability_dataset_consolidation/tests`

  Expected: syntax success and all consolidation tests pass.

- [ ] **Step 5: Run affected regression suites**

  Run: `python3 -m pytest -q hpc_storage_migration/test_migrate_parscratch_to_shared.py CamCan_Experiment/tests/test_delete_directory_hpc.py simnibs_326_corrected_label_repeatability/tests simnibs_subject_replacement/tests repeatability_dataset_consolidation/tests`

  Expected: all affected tests pass.

- [ ] **Step 6: Run final static and full-suite checks**

  Run: `git diff --check && python3 -m compileall -q repeatability_dataset_consolidation && bash -n repeatability_dataset_consolidation/hpc/submit.sh && python3 -m pytest -q`

  Expected: static checks pass; report every full-suite failure by test name rather than omitting unrelated baseline failures.

- [ ] **Step 7: Commit**

  Run: `git add SimNIBS/Scripts/repeatability_dataset_consolidation SimNIBS/Scripts/docs/superpowers && git commit -m "Document combined repeatability dataset operations"`
