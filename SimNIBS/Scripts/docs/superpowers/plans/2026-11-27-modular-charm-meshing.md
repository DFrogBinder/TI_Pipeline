# Modular CHARM Meshing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a configuration-driven, version-aware CHARM meshing campaign that creates exactly one audited mesh per segmentation while preserving all existing collected-meshing pipelines.

**Architecture:** Add a focused `charm_mesh_campaign` package for configuration, normalized task manifests and orchestration. Reuse the established direct CHARM meshing primitive behind an explicit version/settings contract; retain existing scripts as compatibility surfaces and make new configuration opt-in.

**Tech Stack:** Python 3.11 standard library, NumPy/nibabel/SimNIBS at worker runtime, Bash/Slurm, pytest.

**Spec:** `docs/superpowers/specs/2026-11-27-modular-charm-meshing-design.md`

## Global Constraints

- Generate exactly one mesh per ready normalized task row.
- Never run segmentation, surfaces, FEM simulation or field extraction.
- Never modify source tissue-label images.
- Require explicit expected SimNIBS version, module, backend and settings policy for configuration-driven campaigns.
- Unknown schemas, backends and software versions fail closed.
- Preserve every existing collected-meshing Python API, CLI, shell environment variable, default resource and output layout.
- Do not submit or operate HPC jobs.

## Review Focus

- Two realizations of one subject must resolve to distinct task keys and output paths; Task 2 tests this.
- Relative paths and `..`/absolute template escapes must never write outside the output root; Task 1 and Task 2 test this.
- Version strings such as `4.0.1` versus `4.0.1+local` need an explicit equality policy and mismatches must stop before meshing; Task 3 tests this.
- A changed settings file or campaign configuration must invalidate result reuse; Task 3 tests this.
- Legacy shell calls with no new configuration variables must produce the same module, resources, array and task command; Task 4 tests this.

---

### Task 1: Versioned campaign configuration

**Files:**
- Create: `CamCan_Experiment/charm_mesh_campaign/__init__.py`
- Create: `CamCan_Experiment/charm_mesh_campaign/config.py`
- Create: `CamCan_Experiment/tests/test_charm_mesh_campaign_config.py`

**Interfaces:**
- Produces: `CampaignConfig`, `load_campaign_config(path: Path) -> CampaignConfig`, `CampaignConfig.sha256`, and normalized path/resource/software properties consumed by Tasks 2–4.

- [ ] Write failing tests for a complete literal JSON configuration, relative path resolution, unknown/missing fields, unsafe output templates, unsupported schema/backend, and deterministic configuration hashing.
- [ ] Run the focused tests and confirm import/behaviour failures.
- [ ] Implement immutable dataclasses and strict standard-library JSON parsing with no implicit environment expansion.
- [ ] Run focused tests and the unchanged collected-meshing baseline.

### Task 2: Normalized task manifests with realization identity

**Files:**
- Create: `CamCan_Experiment/charm_mesh_campaign/tasks.py`
- Create: `CamCan_Experiment/tests/test_charm_mesh_campaign_tasks.py`

**Interfaces:**
- Consumes: `CampaignConfig` from Task 1.
- Produces: `TaskRecord`, `build_task_manifest(config: CampaignConfig) -> dict`, `read_task_manifest(path: Path) -> list[TaskRecord]`, and collision-safe mesh/result paths consumed by Tasks 3–4.

- [ ] Write failing tests for one-row-per-map normalization, repeated realizations of one subject, input hash/size/status validation, exact expected count, exclusions, duplicate task/output rejection, and output-root containment.
- [ ] Run the focused tests and confirm the expected failures.
- [ ] Implement the task adapter and atomic TSV/summary writers.
- [ ] Run Tasks 1–2 tests and unchanged legacy preflight tests.

### Task 3: Explicit software contract and audited task execution

**Files:**
- Create: `CamCan_Experiment/charm_mesh_campaign/runner.py`
- Modify: `CamCan_Experiment/charm_segmentation_batch/mesh_collected_segmentations.py`
- Create: `CamCan_Experiment/tests/test_charm_mesh_campaign_runner.py`
- Modify: `CamCan_Experiment/tests/test_mesh_collected_segmentations.py`

**Interfaces:**
- Consumes: `CampaignConfig` and `TaskRecord`.
- Produces: `run_task(config, task_index, staging_root=None) -> dict`, `validate_campaign(config, skip_hashes=False) -> dict`, plus optional version/settings/provenance parameters on the existing `create_mesh_from_label` primitive.

- [ ] Write failing tests proving version mismatch stops before label loading, settings/config hashes enter provenance, changed settings/config invalidate reuse, required tissue tags are enforced, and legacy calls without new parameters retain existing payload behaviour.
- [ ] Run focused tests and confirm the failures.
- [ ] Add the version-aware runner and minimally extend the existing primitive without changing default behaviour.
- [ ] Run new runner tests and all unchanged collected-meshing tests.

### Task 4: Config-driven CLI and Slurm orchestration

**Files:**
- Create: `CamCan_Experiment/charm_mesh_campaign/cli.py`
- Create: `CamCan_Experiment/charm_mesh_campaign/mesh_campaign_array.slurm`
- Create: `CamCan_Experiment/charm_mesh_campaign/submit_mesh_campaign.py`
- Modify: `CamCan_Experiment/charm_segmentation_batch/mesh_collected_segmentations_array.slurm`
- Create: `CamCan_Experiment/tests/test_charm_mesh_campaign_cli.py`
- Modify: `CamCan_Experiment/tests/test_mesh_collected_segmentations.py`

**Interfaces:**
- Consumes: Tasks 1–3 configuration, task and runner interfaces.
- Produces: `preflight`, `describe`, `run-task`, `validate`, and gated `submit --submit` commands; legacy worker gains a defaulted `TI_CHARM_MESH_SIMNIBS_MODULE` only.

- [ ] Write failing tests for dry-run submission, explicit full scope, exact `sbatch` arguments, module/Python propagation, packed-worker arithmetic, and no-submit default.
- [ ] Add a legacy regression test proving the worker still defaults to `SimNIBS/4.0.1-foss-2023a` and its existing task command.
- [ ] Run focused tests and confirm the expected failures.
- [ ] Implement the Python CLI/submitter and generic worker; parameterize the legacy module with an identical default.
- [ ] Run new orchestration tests, legacy launcher tests, `bash -n`, and ShellCheck where available.

### Task 5: Candidate raw-v4.0.1 configuration and documentation

**Files:**
- Create: `CamCan_Experiment/charm_mesh_campaign/configs/uncorrected_v401.example.json`
- Create: `CamCan_Experiment/charm_mesh_campaign/README.md`
- Modify: `CamCan_Experiment/charm_segmentation_batch/README.md`
- Create: `CamCan_Experiment/tests/test_charm_mesh_campaign_docs.py`

**Interfaces:**
- Consumes: all prior interfaces.
- Produces: copyable preflight/describe/submit/validate commands for the candidate 652-map uncorrected campaign, explicitly labelled pending live preflight.

- [ ] Write failing tests that load the example configuration, assert raw/corrected path separation, require SimNIBS 4.0.1 provenance, and verify commands remain submission-gated.
- [ ] Implement the example and documentation without asserting HPC completion.
- [ ] Run documentation/configuration tests.

### Task 6: Backward-compatibility and final verification

**Files:**
- Modify only if a regression is identified: files owned by Tasks 1–5.

**Interfaces:**
- Consumes: complete implementation.
- Produces: verification evidence and a compatibility report.

- [ ] Run all `charm_mesh_campaign` tests.
- [ ] Run unchanged `test_mesh_collected_segmentations.py`, `test_charm_only_remesh.py`, `test_cohort_pipeline.py`, and `test_camcan_inplace_rerun_hpc.py`.
- [ ] Run broader affected CamCan tests selected by imports/references to the legacy mesher and launchers.
- [ ] Run Python compilation, Ruff if available, Bash syntax, ShellCheck if available, and `git diff --check`.
- [ ] Compare the pre-refactor 39-test baseline with final results and report any repository-wide unrelated collection failures separately.
- [ ] Review the diff for unintended changes to default module, paths, resources, retry/requeue behaviour, output layout and scientific meshing settings.
