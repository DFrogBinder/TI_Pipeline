# Segmentation Determinism Staging and Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the fixed two-arm experiment in an isolated staging root and later install its completed package into the finalized combined dataset through a checksum-gated importer.

**Architecture:** Experiment configuration owns independent source, staging, and future dataset roots. A separate importer validates both terminal receipts, performs resumable rsync materialization and checksum verification, and creates an extension-specific receipt without modifying the staging package or original dataset receipt.

**Tech Stack:** Python 3.9+, Bash, Slurm launchers, rsync, pytest, Ruff, ShellCheck.

**Spec:** `docs/superpowers/specs/2026-11-14-segmentation-determinism-staging-import-design.md`

## Global Constraints

- Subject `sub-CC110174`, seed `20261106`, two arms, 10 repeats per arm.
- Preserve 8 CPU, 32G, 08:00:00, `%10` concurrency, and at most one exact retry.
- No surface, meshing, electrode-placement, FEM, Shared-copy, or deletion tasks.
- Staging root: `/mnt/parscratch/users/cop23bi/segmentation_determinism_staging_v1`.
- Future extension: `/mnt/parscratch/users/cop23bi/final_132_repeatability_combined_dataset_v1/extensions/segmentation_determinism_v1`.
- Import may not run before both terminal receipts validate.

## Review Focus

- Missing combined-dataset receipt must block import but not staging preflight.
- Foreign or already-completed extension roots must be rejected.
- Interrupted owned imports must resume without deleting staging data.
- Rsync verification must detect content differences and destination-only files.
- Environment overrides must not decouple source, staging, and extension paths.

---

### Task 1: Split staging execution from dataset installation paths

**Files:**
- Modify: `simnibs_segmentation_determinism/settings.py`
- Modify: `simnibs_segmentation_determinism/workflow.py`
- Test: `simnibs_segmentation_determinism/tests/conftest.py`
- Test: `simnibs_segmentation_determinism/tests/test_settings.py`
- Test: `simnibs_segmentation_determinism/tests/test_preflight.py`
- Test: `simnibs_segmentation_determinism/tests/test_workflow.py`

**Interfaces:**
- Produces: staging-default `ExperimentConfig`, `with_staging_root()`, and dataset-receipt validation callable only by import.
- Consumes: existing input staging, repeat requests, and provenance helpers.

- [ ] Write failing tests pinning exact roots, completion-independent preflight, staged cohort/input copies, and path collision rejection.
- [ ] Run focused tests and confirm failures describe the old dataset-bound behavior.
- [ ] Implement the minimal configuration and workflow split.
- [ ] Run focused tests and the package suite.
- [ ] Commit the task.

### Task 2: Add checksum-gated extension importer

**Files:**
- Create: `simnibs_segmentation_determinism/extension_import.py`
- Modify: `simnibs_segmentation_determinism/cli.py`
- Test: `simnibs_segmentation_determinism/tests/test_extension_import.py`
- Test: `simnibs_segmentation_determinism/tests/test_cli.py`

**Interfaces:**
- Consumes: staging `_pipeline/complete.json`, combined dataset receipts, staging and extension roots.
- Produces: `install_preflight(config)`, `install_extension(config, confirm_dataset_root)`, and `_import/complete.json`.

- [ ] Write failing tests for missing receipts, confirmation mismatch, resumable ownership, foreign destinations, rsync mismatch, successful verification, and source immutability.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement the importer and CLI commands minimally.
- [ ] Run focused tests and the package suite.
- [ ] Commit the task.

### Task 3: Expose safe HPC commands and document handoff

**Files:**
- Modify: `simnibs_segmentation_determinism/hpc/submit.sh`
- Modify: `simnibs_segmentation_determinism/README.md`
- Modify: `simnibs_segmentation_determinism/tests/test_hpc_launchers.py`
- Modify: `simnibs_segmentation_determinism/tests/test_end_to_end.py`

**Interfaces:**
- Consumes: new CLI staging and install modes.
- Produces: copy-paste-safe `install-preflight` and confirmed `install-extension` commands that submit zero Slurm jobs.

- [ ] Write failing launcher/documentation tests for exact staging paths, unchanged job graph, non-submitting install modes, and later-import instructions.
- [ ] Run focused tests and confirm expected failures.
- [ ] Update the launcher and README without changing resources or dependencies.
- [ ] Run package tests, Ruff, Bash syntax, ShellCheck, compileall, and diff checks.
- [ ] Commit the task and perform a whole-branch self-review.

