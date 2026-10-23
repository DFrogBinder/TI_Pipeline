# SimNIBS 3.2.6 Base-Nine Early Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a provenance-safe nine-subject early FEM release and an isolated one-subject `sub-CC520083` diagnostic submission.

**Architecture:** An opt-in cohort profile drives subject membership, roots, artifact paths, task counts, and gates while leaving the default full-ten profile unchanged. The existing head-model worker gains exact cleanup and per-job working-directory isolation; the launcher exposes separate base-nine collection/promotion/simulation and one-subject diagnostic commands.

**Tech Stack:** Python 3, pytest, Bash, Slurm, SimNIBS 3.2.6.

**Spec:** `docs/superpowers/specs/2026-10-23-simnibs326-base9-early-release-design.md`

## Global Constraints

- Default `full10` behavior and paths remain unchanged.
- `base9` excludes only `sub-CC520083` and uses separate result and collection artifacts.
- Preserve SimNIBS module `3.2.6-foss-2023a`, `sheffield`, 8 CPU, 32G, 08:00:00, max 50 simulations, and one retry for production FEM.
- Diagnostic head-model runs use zero retries and submit no FEM work.
- All production behavior follows test-first red-green cycles.

## Review Focus

- Invalid cohort profile must fail before any submission.
- Base-nine jobs must never resolve the full-ten writable result roots.
- Full-ten and base-nine QC/manifest/promotion artifacts must not overwrite one another.
- Dynamic completion gates must reject 359/360 or 719/720 results.
- Cleanup must preserve T1fs files, `eeg_positions`, `toMNI`, and reference receipts.

---

### Task 1: Cohort profiles and isolated paths

**Files:**
- Modify: `simnibs_326_corrected_label_repeatability/settings.py`
- Modify: `simnibs_326_corrected_label_repeatability/tests/test_settings.py`

**Interfaces:**
- Produces: `cohort_subjects(environ=None)`, `cohort_name(environ=None)`, profile-aware `campaign_paths_from_environment`, and dynamic `campaign_scope`.

- [ ] Write failing tests for base-nine membership, counts, roots, and invalid profile rejection.
- [ ] Run the settings tests and confirm the expected failures.
- [ ] Implement the smallest profile resolver and root selection.
- [ ] Run the settings tests to green.

### Task 2: Cohort-specific head-model collection and promotion

**Files:**
- Modify: `simnibs_326_corrected_label_repeatability/headmodels.py`
- Modify: `simnibs_326_corrected_label_repeatability/simulation_runner.py`
- Modify: `simnibs_326_corrected_label_repeatability/tests/test_headmodels.py`
- Modify: `simnibs_326_corrected_label_repeatability/tests/test_simulation_runner.py`

**Interfaces:**
- Produces: `headmodel_output_dir(paths, subjects=None)`, cohort-aware collection, validation, promotion, and simulation binding.

- [ ] Write failing tests showing base-nine uses nine markers and a separate artifact directory.
- [ ] Run the tests and confirm they fail for missing profile behavior.
- [ ] Implement cohort-specific collection, QC panels, manifests, and promotion.
- [ ] Add a failing simulation-boundary test for base-nine promotion and subject order.
- [ ] Implement active-cohort promotion/config validation and run both suites to green.

### Task 3: Dynamic pipeline gates

**Files:**
- Modify: `simnibs_326_corrected_label_repeatability/pipeline.py`
- Modify: `simnibs_326_corrected_label_repeatability/tests/test_pipeline.py`

**Interfaces:**
- Consumes: `cohort_subjects()` and cohort-specific promotion validation.
- Produces: exact dynamic preflight, configs, completion receipts, status counts, and validation thresholds.

- [ ] Write failing tests for 9-subject initialization and 360/720 task gates.
- [ ] Run tests to confirm failures.
- [ ] Replace hard-coded full-ten task thresholds only where active-cohort execution requires it.
- [ ] Run pipeline tests to green and confirm full-ten regression tests remain green.

### Task 4: Diagnostic cleanup and isolated Gmsh logs

**Files:**
- Modify: `simnibs_326_corrected_label_repeatability/headmodels.py`
- Modify: `simnibs_326_corrected_label_repeatability/hpc/headmodel_array.slurm`
- Modify: `simnibs_326_corrected_label_repeatability/tests/test_headmodels.py`
- Modify: `simnibs_326_corrected_label_repeatability/tests/test_hpc_launchers.py`

**Interfaces:**
- Produces: exact failed-attempt cleanup and per-attempt `gmsh_log.txt` isolation.

- [ ] Write a failing cleanup test containing the observed NIfTI/GEO leftovers and allowlisted references.
- [ ] Run it and confirm failure.
- [ ] Implement exact generated-artifact cleanup and run it to green.
- [ ] Write a failing static launcher test for per-job working directories and diagnostic receipts.
- [ ] Implement worker isolation and verify the launcher test passes.

### Task 5: Opt-in Slurm launcher and documentation

**Files:**
- Modify: `simnibs_326_corrected_label_repeatability/hpc/submit.sh`
- Modify: `simnibs_326_corrected_label_repeatability/tests/test_hpc_launchers.py`
- Modify: `simnibs_326_corrected_label_repeatability/README.md`

**Interfaces:**
- Produces: `--cohort=base9`, `collect-headmodels`, and `diagnose-headmodel sub-CC520083` launcher surfaces.

- [ ] Write failing launcher tests for scope text, root isolation, arrays `0-359%50` and `0-8%9`, collector-only release, and one-subject diagnosis.
- [ ] Run them and confirm the failures.
- [ ] Implement dynamic launcher counts, cohort export, collector-only mode, and diagnostic submission.
- [ ] Run launcher tests to green.
- [ ] Document the exact HPC review, promotion, initialization, FEM, diagnosis, and status sequence.

### Task 6: Full verification, commit, and push

**Files:**
- Verify all modified files.

**Interfaces:**
- Produces: a pushed `Analysis` commit and copy-paste HPC commands.

- [ ] Run all corrected-label v3 tests.
- [ ] Run `bash -n` on every modified shell/Slurm file.
- [ ] Run `git diff --check` and inspect the diff for settings drift.
- [ ] Commit only the isolated-worktree changes.
- [ ] Push the fast-forward commit to `origin/Analysis`.
- [ ] Report the commit, verification evidence, and exact submission commands.
