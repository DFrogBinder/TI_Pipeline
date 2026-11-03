# Replacement v4 Paired-Metric Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the 160 schema-v2 optimizer-ROI rows for the completed replacement-subject SimNIBS 4 campaigns without changing campaign outputs or the active consolidated-dataset copy.

**Architecture:** Extend the existing read-only SimNIBS 4 metric launcher to accept an exact subject count and separate output roots. Add one manifest-bound replacement wrapper command that supplies the two completed v4 experiment roots, uses one-subject arrays, and writes both targets beneath the isolated combined-analysis input root. Rebind the final analysis to those isolated tables.

**Tech Stack:** Bash, Slurm, Python/pytest.

**Spec:** `docs/superpowers/specs/2026-11-03-final-132-combined-repeatability-analysis-design.md`

## Global Constraints

- Reuse exactly 160 completed v4 TI NIfTI fields: one subject, two targets, two conditions, 40 repeats.
- Submit zero meshing tasks and zero FEM simulation tasks.
- Do not write under any completed simulation campaign or the active combined-dataset root.
- Require both replacement terminal receipts before preflight or submission.
- Preserve explicit preflight and submission modes.

## Review Focus

- Array bounds must be `0-0%1`, never the existing ten-subject `0-9` scope.
- Left and right isolated outputs must not collide.
- A missing terminal receipt must fail before `sbatch`.
- Production analysis paths must point to the isolated recovered tables.
- Existing full-ten refresh behavior must remain unchanged by default.

---

### Task 1: Parameterize the read-only metric launcher

**Files:**
- Modify: `ti_current_repair/hpc_scripts/submit_repeatability_optimizer_roi_metric_refresh.sh`
- Create: `ti_current_repair/tests/test_optimizer_roi_metric_refresh_launcher.py`

- [x] Write failing tests for dynamic subject count, array bounds, separate output roots, and default full-ten behavior.
- [x] Run the focused test and confirm it fails because overrides are not implemented.
- [x] Implement validated `SUBJECT_COUNT`, `LEFT_OUTPUT_ROOT`, and `RIGHT_OUTPUT_ROOT` environment overrides.
- [x] Run the focused test and confirm it passes.

### Task 2: Add the manifest-bound replacement recovery command

**Files:**
- Modify: `simnibs_subject_replacement/hpc/submit.sh`
- Modify: `simnibs_subject_replacement/tests/test_hpc_launcher.py`
- Modify: `simnibs_subject_replacement/README.md`

- [x] Write failing tests proving the command requires completed v4 targets and delegates exact roots, one-subject scope, and isolated outputs.
- [x] Run the focused test and confirm it fails because the command is absent.
- [x] Implement `v4-paired-metrics` and `v4-paired-metrics-preflight`.
- [x] Run the focused tests and confirm they pass.

### Task 3: Rebind and verify the combined analysis

**Files:**
- Modify: `repeatability_combined_analysis/settings.py`
- Modify: `repeatability_combined_analysis/tests/test_settings.py`
- Modify: `repeatability_combined_analysis/README.md`

- [x] Write a failing production-path test for the isolated replacement v4 inputs.
- [x] Run the focused test and confirm it fails on the campaign-local path.
- [x] Point both replacement v4 source paths at the isolated recovery root.
- [x] Run the combined focused suites, shell syntax checks, and full affected package suites.
- [ ] Commit, merge into `Analysis`, and push only after verification.
