# Generic SimNIBS Replacement-Subject Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a subject-agnostic, fail-closed wrapper that generates new harmonized SimNIBS 3.2.6 and corrected SimNIBS 4 repeatability results for any eligible final-132 replacement subject.

**Architecture:** Add an opt-in `external1` cohort interface to the existing v3 package while preserving `full10` and `base9`, then create `simnibs_subject_replacement/` as a thin orchestration layer. The wrapper validates one runtime subject, stages corrected inputs, gates both simulation versions on a promoted v3 head model, and delegates all meshing, FEM, ROI, retry, median-selection, and fixed-mesh work to existing production modules.

**Tech Stack:** Python 3.11+, Bash/Slurm, pytest, existing SimNIBS 3.2.6 package and `ti_current_repair` pipeline.

**Spec:** `docs/superpowers/specs/2026-10-23-subject-replacement-wrapper-design.md`

## Global Constraints

- Candidate IDs match `^sub-CC[0-9]{6}$`, occur exactly once in final-132, differ from `--replaces`, and do not duplicate the retained base nine.
- No executable configuration hard-codes a candidate replacement ID.
- Existing v3 `full10` and `base9` behavior and existing v4 behavior remain unchanged unless the new wrapper/profile is explicitly selected.
- Preflight is read-only; preparation and submission are separate explicit commands.
- V3 promotion with manifest hash and human-review confirmation gates both v3 and v4 simulations.
- Every writable path is subject-derived and isolated from protected production roots.
- Resources remain Sheffield, 8 CPU, 32G, 08:00:00, maximum 50 simulation tasks, and one exact-element retry.
- One version comprises 160 simulation tasks and 320 component FEM solves; both versions comprise 320 tasks and 640 solves.

## Review Focus

- A valid-looking ID absent from final-132 must fail before any output directory is created; Task 3 tests this.
- A candidate equal to a retained subject or the replaced subject must fail before staging; Task 3 tests both cases.
- A forged or stale external manifest must not activate v3 paths or submissions; Tasks 1 and 2 test manifest binding and rejection.
- Existing non-empty unmanifested roots must never be adopted; Tasks 3 and 4 test root ownership failures.
- Right-M1 submission must fail until the corresponding left-hippocampus terminal receipt is complete; Tasks 4 and 5 test both versions.

---

### Task 1: Opt-in external singleton cohort in the v3 Python package

**Files:**
- Modify: `simnibs_326_corrected_label_repeatability/settings.py`
- Modify: `simnibs_326_corrected_label_repeatability/reference_stage.py`
- Modify: `simnibs_326_corrected_label_repeatability/headmodels.py`
- Modify: `simnibs_326_corrected_label_repeatability/pipeline.py`
- Test: `simnibs_326_corrected_label_repeatability/tests/test_settings.py`
- Test: `simnibs_326_corrected_label_repeatability/tests/test_reference_stage.py`
- Test: `simnibs_326_corrected_label_repeatability/tests/test_headmodels.py`
- Test: `simnibs_326_corrected_label_repeatability/tests/test_pipeline.py`

**Interfaces:**
- Consumes: `SIMNIBS326H_COHORT=external1` and `SIMNIBS326H_EXTERNAL_MANIFEST=/absolute/path/replacement_manifest.json`.
- Produces: `load_external_manifest(environ: Mapping[str, str] | None) -> dict[str, object]`, `cohort_subjects(environ: Mapping[str, str] | None) -> tuple[str, ...]`, dynamic singleton paths/scopes, and external head-model collection/promotion under the isolated head-model root.

- [ ] **Step 1: Write failing tests for external-manifest validation**

  Add tests asserting that `external1` accepts exactly one complete, canonical subject manifest; rejects missing, malformed, multi-subject, relative-path, and stale-profile manifests; and leaves `full10`/`base9` outputs unchanged.

- [ ] **Step 2: Run the focused settings tests and verify RED**

  Run: `python3 -m pytest -q simnibs_326_corrected_label_repeatability/tests/test_settings.py`

  Expected: failures because `external1` and `SIMNIBS326H_EXTERNAL_MANIFEST` are unsupported.

- [ ] **Step 3: Implement the external cohort contract**

  Add `EXTERNAL_MANIFEST_ENV`, validate the JSON binding, make `cohort_subjects()` return its singleton, allow `subject_paths()` only for the active subject, and calculate arrays from the active cohort. Preserve existing default constants and results.

- [ ] **Step 4: Write failing lifecycle tests for singleton reference/head-model collection**

  Add tests asserting one-subject reference collection, validation, QC collection, manifest path, promotion, parser subject choices, and scope counts (`0-39%50`, one reference task, one head-model task).

- [ ] **Step 5: Run lifecycle tests and verify RED**

  Run: `python3 -m pytest -q simnibs_326_corrected_label_repeatability/tests/test_reference_stage.py simnibs_326_corrected_label_repeatability/tests/test_headmodels.py simnibs_326_corrected_label_repeatability/tests/test_pipeline.py`

  Expected: failures at hard-coded `SUBJECTS`/`BASE9_SUBJECTS` checks.

- [ ] **Step 6: Generalize active-cohort lifecycle functions**

  Make collection counts, subject order, output-directory selection, promotion, parser choices, and `_selected_subject()` use the active cohort. Store external collections under `_pipeline/headmodels/external1` while retaining current full10/base9 locations.

- [ ] **Step 7: Run the complete v3 Python suite**

  Run: `python3 -m pytest -q simnibs_326_corrected_label_repeatability/tests`

  Expected: all tests pass, with the existing full10/base9 assertions unchanged.

- [ ] **Step 8: Commit**

  Run: `git add SimNIBS/Scripts/simnibs_326_corrected_label_repeatability && git commit -m "feat: support manifest-bound singleton v3 cohorts"`

### Task 2: Dynamic v3 HPC launcher scope

**Files:**
- Modify: `simnibs_326_corrected_label_repeatability/hpc/submit.sh`
- Test: `simnibs_326_corrected_label_repeatability/tests/test_hpc_launchers.py`

**Interfaces:**
- Consumes: Task 1 `campaign_scope()`, `cohort_subjects()`, and external head-model directory.
- Produces: `--cohort=external1` launcher behavior with reference/head-model arrays `0-0%1`, simulation arrays `0-39%50`, and ROI arrays `0-0%1`.

- [ ] **Step 1: Write failing launcher tests**

  Add mock-`sbatch` tests for external preflight, reference staging, head models, collection, promotion lookup, remesh, fixed, and ROI arrays. Assert no submission on preflight and exact preservation of existing full10/base9 commands.

- [ ] **Step 2: Run the launcher tests and verify RED**

  Run: `python3 -m pytest -q simnibs_326_corrected_label_repeatability/tests/test_hpc_launchers.py`

  Expected: failure because `external1` is rejected and array bounds are hard-coded.

- [ ] **Step 3: Implement dynamic scope in `submit.sh`**

  Resolve counts from the Python settings contract, allow reference/head-model stages for full10 and external1, calculate every array bound, use `_pipeline/headmodels/external1`, and preserve all resource/module/retry defaults.

- [ ] **Step 4: Verify shell syntax and launcher behavior**

  Run: `bash -n simnibs_326_corrected_label_repeatability/hpc/submit.sh && python3 -m pytest -q simnibs_326_corrected_label_repeatability/tests/test_hpc_launchers.py`

  Expected: syntax success and all launcher tests pass.

- [ ] **Step 5: Commit**

  Run: `git add SimNIBS/Scripts/simnibs_326_corrected_label_repeatability && git commit -m "feat: launch dynamic singleton v3 scope"`

### Task 3: Replacement identity, paths, audit, and preparation

**Files:**
- Create: `simnibs_subject_replacement/__init__.py`
- Create: `simnibs_subject_replacement/settings.py`
- Create: `simnibs_subject_replacement/staging.py`
- Create: `simnibs_subject_replacement/replacement.py`
- Create: `simnibs_subject_replacement/tests/conftest.py`
- Create: `simnibs_subject_replacement/tests/test_settings.py`
- Create: `simnibs_subject_replacement/tests/test_replacement.py`

**Interfaces:**
- Consumes: authoritative final-132 list, runtime `--subject`, `--replaces`, corrected-v4 scaffold root, mandatory legacy-v3 root, atlas root, targets CSV, and optional replacement work root.
- Produces: immutable `ReplacementConfig` and `ReplacementPaths` dataclasses, `validate_candidate(config: ReplacementConfig) -> tuple[str, ...]`, `preflight(config: ReplacementConfig) -> dict[str, object]`, `prepare(config: ReplacementConfig) -> dict[str, object]`, and a versioned `_pipeline/replacement_manifest.json` plus singleton staged corrected dataset.

- [ ] **Step 1: Write failing identity and path tests**

  Test canonical/membership validation, duplicate membership, retained-nine rejection, replaced-subject rejection, subject-derived path layout, protected-root overlap, and the exact 320-task/640-solve scope.

- [ ] **Step 2: Run tests and verify RED**

  Run: `python3 -m pytest -q simnibs_subject_replacement/tests/test_settings.py`

  Expected: import failure because the replacement package does not exist.

- [ ] **Step 3: Implement settings and identity validation**

  Add immutable dataclasses for inputs and paths, exact regex/membership checks, retained-subject checks imported from v3 settings, subject-derived root templates, and overlap validation.

- [ ] **Step 4: Write failing audit/preparation tests**

  Build realistic temporary final132 scaffold records, corrected files, legacy allowlist, atlas, and targets fixtures. Assert preflight hashes inputs without writing, rejects missing/mismatched sources, preparation writes a bound manifest and stages exactly three corrected inputs, reuses an exact prepared state, and rejects non-empty unowned roots.

- [ ] **Step 5: Run tests and verify RED**

  Run: `python3 -m pytest -q simnibs_subject_replacement/tests/test_replacement.py`

  Expected: failures because preflight/preparation are absent.

- [ ] **Step 6: Implement audit and atomic preparation**

  Load and call the existing final-132 staging utility rather than copying it. Print the candidate manifest in-memory during preflight; persist and revalidate it only during `prepare`. Keep interrupted prepared roots resumable only when their manifest matches exactly.

- [ ] **Step 7: Run replacement core tests**

  Run: `python3 -m pytest -q simnibs_subject_replacement/tests/test_settings.py simnibs_subject_replacement/tests/test_replacement.py`

  Expected: all tests pass.

- [ ] **Step 8: Commit**

  Run: `git add SimNIBS/Scripts/simnibs_subject_replacement && git commit -m "feat: audit and prepare replacement subjects"`

### Task 4: V3 promotion gates and delegated v4 workflows

**Files:**
- Modify: `simnibs_subject_replacement/replacement.py`
- Create: `simnibs_subject_replacement/v4_workflow.py`
- Modify: `simnibs_subject_replacement/tests/test_replacement.py`
- Create: `simnibs_subject_replacement/tests/test_v4_workflow.py`

**Interfaces:**
- Consumes: Task 3 prepared manifest and existing v3/v4 completion receipts.
- Produces: `require_v3_promotion(manifest: dict[str, object]) -> dict[str, object]`, `require_left_complete(manifest: dict[str, object], version: str) -> dict[str, object]`, `initialize_v4_target(manifest: dict[str, object], target: str) -> int`, `run_v4_target(manifest: dict[str, object], target: str, dry_run: bool) -> int`, and `replacement_status(manifest: dict[str, object]) -> dict[str, object]`.

- [ ] **Step 1: Write failing gate tests**

  Assert both versions reject simulation before a subject/root/hash-bound v3 promotion, accept the exact promotion, and reject right-M1 until the same version's left completion receipt is valid.

- [ ] **Step 2: Run gate tests and verify RED**

  Run: `python3 -m pytest -q simnibs_subject_replacement/tests/test_replacement.py -k gate`

  Expected: failures because gate functions do not exist.

- [ ] **Step 3: Implement receipt-bound gates**

  Validate promotion and terminal receipts against manifest subject, roots, and hashes without accepting scheduler submission state as completion.

- [ ] **Step 4: Write failing v4 delegation tests**

  Assert initialization passes one subject, repeat count 40, confirmed target-specific montage/ROI, singleton corrected root, and isolated target root to `staged_median_fixed_experiment`; assert dry-run submits nothing, real submission delegates once, and right-M1 is gated by left completion.

- [ ] **Step 5: Run delegation tests and verify RED**

  Run: `python3 -m pytest -q simnibs_subject_replacement/tests/test_v4_workflow.py`

  Expected: import/function failures because v4 delegation is absent.

- [ ] **Step 6: Implement v4 delegation and status**

  Call the existing staged median/fixed pipeline directly. Do not copy simulation or controller logic. Treat singleton figures as QC only and retain per-run/paired metrics for later 9+1 aggregation.

- [ ] **Step 7: Run Task 4 tests**

  Run: `python3 -m pytest -q simnibs_subject_replacement/tests`

  Expected: all replacement tests pass.

- [ ] **Step 8: Commit**

  Run: `git add SimNIBS/Scripts/simnibs_subject_replacement && git commit -m "feat: gate and delegate replacement simulations"`

### Task 5: Subject-agnostic HPC entry point and runbook

**Files:**
- Create: `simnibs_subject_replacement/hpc/submit.sh`
- Create: `simnibs_subject_replacement/README.md`
- Create: `simnibs_subject_replacement/tests/test_hpc_launcher.py`

**Interfaces:**
- Consumes: Tasks 1-4 manifests, gates, v3 submitter, and v4 delegation commands.
- Produces: one `submit.sh MODE --subject=...` interface whose modes never require source edits when the candidate changes.

- [ ] **Step 1: Write failing shell-wrapper tests**

  Test required subject/replaces/legacy-root arguments, no default submission, safe manifest-derived v3 environment, explicit preparation, v3 head lifecycle delegation, left then right target gates for v3/v4, numerical scope output, and absence of candidate IDs in executable configuration.

- [ ] **Step 2: Run wrapper tests and verify RED**

  Run: `python3 -m pytest -q simnibs_subject_replacement/tests/test_hpc_launcher.py`

  Expected: failure because `hpc/submit.sh` is missing.

- [ ] **Step 3: Implement the wrapper and runbook**

  Add exact modes `preflight`, `prepare`, `v3-module-check`, `v3-stage-references`, `v3-headmodels`, `v3-collect-headmodels`, `v3-promote`, `v3-init`, `v3-left-remesh`, `v3-left-fixed`, `v3-right-remesh`, `v3-right-fixed`, `v3-analysis`, `v3-status`, `v4-init-left`, `v4-preflight-left`, `v4-submit-left`, `v4-init-right`, `v4-preflight-right`, `v4-submit-right`, and `status`. Export only allowlisted manifest values and delegate to existing launchers/modules. Document copy-paste commands using `$SUBJECT`, `$REPLACES`, and `$LEGACY_V3_ROOT`.

- [ ] **Step 4: Verify syntax and focused integration tests**

  Run: `bash -n simnibs_subject_replacement/hpc/submit.sh && python3 -m pytest -q simnibs_subject_replacement/tests`

  Expected: syntax success and all replacement tests pass.

- [ ] **Step 5: Run cross-package regression suites**

  Run: `python3 -m pytest -q simnibs_326_corrected_label_repeatability/tests ti_current_repair/tests/test_staged_current_repair_pipeline.py simnibs_subject_replacement/tests`

  Expected: all tests pass with the existing single expected skip.

- [ ] **Step 6: Run final static checks**

  Run: `git diff --check && bash -n simnibs_326_corrected_label_repeatability/hpc/submit.sh && bash -n simnibs_subject_replacement/hpc/submit.sh && python3 -m compileall -q simnibs_326_corrected_label_repeatability simnibs_subject_replacement`

  Expected: exit status zero and no output from failure checks.

- [ ] **Step 7: Commit**

  Run: `git add SimNIBS/Scripts/simnibs_subject_replacement SimNIBS/Scripts/simnibs_326_corrected_label_repeatability && git commit -m "feat: orchestrate generic v3 and v4 subject replacement"`
