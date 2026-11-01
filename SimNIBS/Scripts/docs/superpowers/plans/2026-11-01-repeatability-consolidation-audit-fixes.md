# Repeatability Consolidation Audit Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the self-audit safety gaps before the combined dataset is created or any source retirement is submitted.

**Architecture:** Keep the canonical physical-copy design, but bind retirement to an exhaustive source-disposition inventory instead of descendant-existence checks. Make the Shared archive self-contained by materializing validated links, strengthen source completion bindings, and make operator-visible completion state live-validated and internally consistent.

**Tech Stack:** Python 3.9-compatible standard library, rsync, pytest, Bash.

**Spec:** `docs/superpowers/specs/2026-10-30-repeatability-combined-dataset-design.md`

## Global Constraints

- No HPC copy, archive, submission, or deletion is performed while implementing these fixes.
- Dataset and archive copying remain resumable and non-destructive.
- Retirement remains a separate, explicit, whole-allowlist action using external receipts.
- The canonical scope remains 10 subjects, 2 versions, 2 targets, 2 conditions, 40 repeats, and 3,200 simulation records.
- Primary inputs and the global corrected-v4 scaffold root remain outside retirement.

## Review Focus

- Extra files anywhere under a whole-root retirement target must be visible and receipt-bound before deletion.
- Absolute or cross-root symlinks in legacy roots must not become broken Shared archive entries.
- A changed copied file must make live dataset status invalid rather than complete.
- Source completion receipts must bind the expected target, subject count, repeat count, condition counts, and final stage.
- Dataset and Shared preflight must expose required bytes and fail before first creation when capacity is insufficient.

---

### Task 1: Exhaustive retirement disposition

**Files:**
- Create: `repeatability_dataset_consolidation/disposition.py`
- Modify: `repeatability_dataset_consolidation/workflow.py`
- Modify: `repeatability_dataset_consolidation/retirement.py`
- Modify: `repeatability_dataset_consolidation/cli.py`
- Test: `repeatability_dataset_consolidation/tests/test_retirement.py`
- Test: `repeatability_dataset_consolidation/tests/test_completion.py`

**Interfaces:**
- Produces: a deterministic per-file disposition manifest classifying each retirement-root entry as `dataset_copy`, `shared_archive`, or `intentional_retirement`.
- Consumes: the immutable consolidation plan and exact retirement/archive allowlists.

- [x] Write failing tests proving an unplanned file is represented in the disposition and that retirement requires its exact disposition hash.
- [x] Run the focused tests and verify they fail for missing behavior.
- [x] Implement deterministic source inventories, bind them into dataset completion, and revalidate them live during retirement planning.
- [x] Add the exact `--retirement-disposition-sha256` confirmation gate and expose per-root summaries in `retire-plan`.
- [x] Run the focused and task suites.

### Task 2: Self-contained Shared archive

**Files:**
- Modify: `repeatability_dataset_consolidation/archive.py`
- Test: `repeatability_dataset_consolidation/tests/test_archive.py`

**Interfaces:**
- Produces: a physical Shared archive with no symlinks and materialized inventories equivalent to source link semantics.

- [x] Write failing tests showing absolute internal links currently break after retirement and external links must be rejected.
- [x] Run the tests and verify the expected failures.
- [x] Validate all link targets against the four archive roots, copy with `--copy-links`, and compare materialized source inventories to regular destination inventories.
- [x] Run archive and retirement tests.

### Task 3: Honest completion and source-receipt bindings

**Files:**
- Modify: `repeatability_dataset_consolidation/validation.py`
- Modify: `repeatability_dataset_consolidation/workflow.py`
- Test: `repeatability_dataset_consolidation/tests/test_validation.py`
- Test: `repeatability_dataset_consolidation/tests/test_completion.py`

**Interfaces:**
- Produces: scope-bound source receipt validation, a final `dataset_manifest.json`, and a live-validating status result.

- [x] Write failing tests for a scope-less receipt, tampered completed data, and the stale `prepared` top-level manifest.
- [x] Run them and confirm the expected failures.
- [x] Bind source receipts to the grouped experiment scope and factor one read-only live completion validator shared by status and retirement.
- [x] Promote the top-level manifest before final inventory creation and report invalid live state explicitly.
- [x] Run focused and package tests.

### Task 4: Storage-capacity preflight and documentation

**Files:**
- Modify: `repeatability_dataset_consolidation/inventory.py`
- Modify: `repeatability_dataset_consolidation/workflow.py`
- Modify: `repeatability_dataset_consolidation/archive.py`
- Modify: `repeatability_dataset_consolidation/README.md`
- Modify: `docs/superpowers/specs/2026-10-30-repeatability-combined-dataset-design.md`
- Test: `repeatability_dataset_consolidation/tests/test_validation.py`
- Test: `repeatability_dataset_consolidation/tests/test_archive.py`
- Test: `repeatability_dataset_consolidation/tests/test_cli.py`

**Interfaces:**
- Produces: exact materialized byte estimates and an explicit sufficient/insufficient capacity gate before new roots are created.

- [ ] Write failing dataset and archive capacity tests.
- [ ] Run them and verify expected failure.
- [ ] Add link-aware byte estimation and fail only before initial root creation when required bytes exceed free bytes.
- [ ] Update the runbook and design with the disposition confirmation and self-contained archive semantics.
- [ ] Run all package tests, affected integration tests, compile checks, shell syntax, and diff hygiene.

