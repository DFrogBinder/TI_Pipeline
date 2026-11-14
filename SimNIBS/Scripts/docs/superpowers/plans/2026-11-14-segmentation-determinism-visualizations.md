# Segmentation Determinism Visualizations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a provenance-bound, read-only figure exporter for the completed segmentation-determinism experiment.

**Architecture:** A focused statistics module validates and summarizes existing analysis outputs and CHARM label maps. A rendering module creates six PNG/PDF figures and atomically publishes a separately checksummed `visualizations_v1` package. The existing CLI and HPC wrapper expose this without submitting jobs or rewriting completed scientific outputs.

**Tech Stack:** Python 3.9+, NumPy, NiBabel, Matplotlib, standard-library CSV/JSON/hash utilities, pytest.

**Spec:** `docs/superpowers/specs/2026-11-14-segmentation-determinism-visualizations-design.md`

## Global Constraints

- Existing `_pipeline/complete.json` and `analysis/` artifacts are read-only.
- Default output is `<staging-root>/visualizations_v1`.
- The source experiment must validate as complete before rendering.
- All 45 CHARM repeat pairs and all 10 CHARM label maps are required.
- No cross-version voxel equivalence is implied.
- No pandas or seaborn dependency.
- `figures` submits zero Slurm jobs and uses `ANALYSIS_PYTHON` in the HPC wrapper.

## Review Focus

- Missing, duplicated or reversed CHARM repeat pairs must fail before a misleading heatmap is rendered; covered in Task 1.
- Mismatched NIfTI shape or affine must fail before spatial aggregation; covered in Task 1.
- A stale or foreign `visualizations_v1` directory must never be overwritten; covered in Task 2.
- Source analysis or terminal-receipt hash drift must prevent reuse; covered in Task 2.
- Headless Matplotlib execution and zero-Slurm launcher behavior must remain reliable; covered in Task 3.

---

### Task 1: Validated visualization data model

**Files:**
- Create: `simnibs_segmentation_determinism/visualization_data.py`
- Create: `simnibs_segmentation_determinism/tests/test_visualization_data.py`

**Interfaces:**
- Consumes: completed staging root with terminal receipt, manifest, TSV tables and CHARM repeat maps.
- Produces: `VisualizationData`, `load_visualization_data(staging_root: Path) -> VisualizationData`, and JSON-serializable summary statistics.

- [ ] Write failing tests for exact 45-pair validation, aggregate statistics, tissue Dice/volume parsing, modal-label disagreement frequency, and shape/affine rejection.
- [ ] Run `pytest simnibs_segmentation_determinism/tests/test_visualization_data.py -q` and confirm failures are due to the missing module.
- [ ] Implement the minimal validated loaders and pure numerical helpers.
- [ ] Re-run the focused tests and confirm they pass.
- [ ] Commit the task.

### Task 2: Atomic figure package renderer

**Files:**
- Create: `simnibs_segmentation_determinism/visualizations.py`
- Create: `simnibs_segmentation_determinism/tests/test_visualizations.py`

**Interfaces:**
- Consumes: Task 1 `VisualizationData` and a completed staging root.
- Produces: `build_visualizations(config: ExperimentConfig) -> dict` and a complete `visualizations_v1` package with six PNG/PDF pairs, captions, statistics, manifest and checksums.

- [ ] Write failing tests for required outputs, readable image dimensions, source-hash binding, checksums, idempotent reuse, stale-source rejection and foreign/partial destination refusal.
- [ ] Run `pytest simnibs_segmentation_determinism/tests/test_visualizations.py -q` and confirm expected failures.
- [ ] Implement headless Matplotlib rendering, captions, summary files, atomic publication and reuse validation.
- [ ] Re-run focused tests and confirm they pass.
- [ ] Commit the task.

### Task 3: CLI, HPC wrapper and operator documentation

**Files:**
- Modify: `simnibs_segmentation_determinism/cli.py`
- Modify: `simnibs_segmentation_determinism/hpc/submit.sh`
- Modify: `simnibs_segmentation_determinism/README.md`
- Modify: `simnibs_segmentation_determinism/tests/test_cli.py`
- Modify: `simnibs_segmentation_determinism/tests/test_hpc_launchers.py`

**Interfaces:**
- Consumes: Task 2 `build_visualizations(config)`.
- Produces: `python -m simnibs_segmentation_determinism figures` and `bash hpc/submit.sh figures`, both returning a machine-readable receipt and submitting no jobs.

- [ ] Write failing CLI and launcher tests, including `ANALYSIS_PYTHON` forwarding and zero `sbatch` calls.
- [ ] Run the focused CLI/launcher tests and confirm expected failures.
- [ ] Add the command wiring and concise operator documentation.
- [ ] Run focused tests and `bash -n simnibs_segmentation_determinism/hpc/submit.sh`.
- [ ] Run the complete package test suite and `git diff --check`.
- [ ] Commit the task.

### Task 4: End-to-end render verification

**Files:**
- Modify: `simnibs_segmentation_determinism/tests/test_end_to_end.py`

**Interfaces:**
- Consumes: complete figure command and synthetic completed experiment.
- Produces: an end-to-end assertion that scientific collection remains unchanged while the separate visualization package validates.

- [ ] Write the failing end-to-end test before changing fixtures or production code.
- [ ] Run it and confirm the missing visualization fixture behavior is the cause.
- [ ] Add only the fixture support required to render the package.
- [ ] Run the end-to-end test, full package suite, `bash -n`, and `git diff --check`.
- [ ] Commit the task.
