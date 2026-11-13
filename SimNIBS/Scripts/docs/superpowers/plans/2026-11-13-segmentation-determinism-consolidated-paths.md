# Segmentation Determinism Consolidated Paths Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every scientific input, run output, log, receipt, and analysis path for the two-arm segmentation-determinism experiment live below the combined repeatability dataset.

**Architecture:** Pin one consolidated dataset root, derive a canonical internal MRI source and one owned extension root, then atomically stage T1/T2 copies into the extension before any task can run. Replace repository-local cohort/audit provenance with the dataset's own subjects, manifest, and completion receipt while preserving the existing arm, module, retry, and resource contracts.

**Tech Stack:** Python 3.9+, pathlib, hashlib, atomic filesystem promotion, Bash/Slurm, pytest.

**Spec:** `docs/superpowers/specs/2026-11-13-segmentation-determinism-consolidated-paths-design.md`

## Global Constraints

- Production dataset root is `/mnt/parscratch/users/cop23bi/final_132_repeatability_combined_dataset_v1`.
- Extension root is exactly `extensions/segmentation_determinism_v1` below that dataset.
- Scientific scope remains 1 subject, 2 arms, 10 repeats per arm, 20 segmentation tasks, 0 surface/mesh/FEM tasks.
- SimNIBS modules, MATLAB module, resources, retry semantics, and stage commands do not change.
- No experiment submission or HPC filesystem mutation occurs during implementation.

## Review Focus

- Incomplete or foreign dataset receipts must fail before extension creation.
- Any source/output override outside the dataset must fail closed.
- Partial input staging must not leave a reusable experiment root.
- Reuse must detect staged-input drift even when canonical inputs are unchanged.
- Slurm exports and printed scope must resolve to the same dataset-contained paths as Python.

---

### Task 1: Pin the dataset-contained path contract

**Files:**
- Modify: `simnibs_segmentation_determinism/settings.py`
- Modify: `simnibs_segmentation_determinism/cli.py`
- Modify: `simnibs_segmentation_determinism/tests/conftest.py`
- Modify: `simnibs_segmentation_determinism/tests/test_settings.py`
- Modify: `simnibs_segmentation_determinism/tests/test_preflight.py`

**Interfaces:**
- Produces: dataset-root configuration, canonical input resolution, staged input resolution, and completed-dataset validation.

- [ ] Write failing tests for exact production paths, containment failures, dataset receipt drift, and dataset-contained cohort selection.
- [ ] Run the focused tests and verify the new assertions fail for the old roots.
- [ ] Implement the minimal configuration and preflight changes.
- [ ] Run the focused tests and verify they pass.
- [ ] Commit the task.

### Task 2: Atomically stage inputs and bind workflow receipts

**Files:**
- Modify: `simnibs_segmentation_determinism/workflow.py`
- Modify: `simnibs_segmentation_determinism/tests/test_workflow.py`
- Modify: `simnibs_segmentation_determinism/tests/test_end_to_end.py`

**Interfaces:**
- Consumes: Task 1 canonical/staged input resolvers.
- Produces: atomic extension initialization and repeat requests that use only staged inputs.

- [ ] Write failing tests for atomic staging, origin/staged hashes, staged-input drift, and run-command paths.
- [ ] Run the focused tests and verify expected failures.
- [ ] Implement temporary-root staging, atomic promotion, and staged-input validation.
- [ ] Run the focused tests and verify they pass.
- [ ] Commit the task.

### Task 3: Align the HPC launcher and operator documentation

**Files:**
- Modify: `simnibs_segmentation_determinism/hpc/submit.sh`
- Modify: `simnibs_segmentation_determinism/tests/test_hpc_launchers.py`
- Modify: `simnibs_segmentation_determinism/README.md`

**Interfaces:**
- Consumes: Tasks 1-2 path contract.
- Produces: one dataset-root export and documented post-consolidation execution sequence.

- [ ] Write failing launcher tests for the dataset root, derived source root, extension root, and unchanged scope/resources.
- [ ] Run the launcher tests and verify expected failures.
- [ ] Update the submitter and README without changing submission semantics.
- [ ] Run launcher tests, `bash -n`, and ShellCheck.
- [ ] Commit the task.

### Task 4: Whole-package verification

**Files:**
- Review: `simnibs_segmentation_determinism/**`

**Interfaces:**
- Consumes: all previous tasks.
- Produces: a clean feature branch ready for review and integration.

- [ ] Run the complete segmentation-determinism test suite.
- [ ] Run Ruff, compileall, Bash syntax, ShellCheck, and `git diff --check`.
- [ ] Audit that all production scientific paths resolve beneath the consolidated dataset and scope/module/resource settings did not drift.
- [ ] Commit any review-only corrections and report the exact evidence.
