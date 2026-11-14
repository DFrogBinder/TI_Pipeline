# CHARM Journal Spatial Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a versioned, provenance-bound journal figure package that clearly presents the three-dimensional CHARM label disagreement observed across ten completed runs.

**Architecture:** Add a focused spatial-data preparation module and a separate atomic package builder, leaving the validated `visualizations_v1` code and outputs unchanged. Expose the exporter through a new CLI and non-submitting HPC launcher mode that preserves the proven rendering environment.

**Tech Stack:** Python 3.9+, NumPy, SciPy, NiBabel, Matplotlib, pytest, Bash/Slurm wrapper.

**Spec:** `docs/superpowers/specs/2026-11-14-charm-journal-spatial-figures-design.md`

## Global Constraints

- Read exactly 10 completed SimNIBS 4.0.1 CHARM label maps and one staged T1 for `sub-CC110174`.
- Submit zero segmentation, surface, meshing, or FEM tasks.
- Do not modify completed experiment outputs or `visualizations_v1`.
- Write only to `journal_spatial_figures_v1` through an atomic temporary sibling.
- Preserve the working HPC rendering profile: SimNIBS 4.0.1, Sheffield, 4 CPU, 16G, 02:00:00.
- Export two figures as 600-dpi PNG, PDF, and SVG with checksums and source hashes.

## Review Focus

- Non-RAS source geometry must reorient data and affine together without label interpolation.
- A zero-disagreement fixture must still choose valid deterministic fallback slices.
- Small or edge-touching heads must produce clipped, non-empty crop bounds.
- A T1 with no finite positive intensities must fail closed with a clear data error.
- Existing foreign, partial, stale, or checksum-invalid destinations must be rejected rather than overwritten.

---

### Task 1: Spatial data preparation

**Files:**
- Create: `simnibs_segmentation_determinism/journal_spatial_data.py`
- Create: `simnibs_segmentation_determinism/tests/test_journal_spatial_data.py`
- Modify: `simnibs_segmentation_determinism/tests/test_visualizations.py`

**Interfaces:**
- Consumes: `load_visualization_data(staging_root: Path) -> VisualizationData` and `resolve_staged_subject_inputs(config) -> SubjectInputs`.
- Produces: `prepare_journal_spatial_data(config: ExperimentConfig) -> JournalSpatialData`, `weighted_slice_indices(weights: np.ndarray, count: int) -> tuple[int, ...]`, and display/crop metadata used by Task 2.

- [x] **Step 1: Write failing tests for integer disagreement counts, RAS reorientation, T1 resampling/normalisation, crop bounds, weighted slice selection, zero-disagreement fallback, and invalid T1 rejection.**

- [x] **Step 2: Run `test_journal_spatial_data.py` and verify failures are due to the missing module and interfaces.**

- [x] **Step 3: Implement the minimal slab-safe preparation and validation code.**

- [x] **Step 4: Run the new data tests and all existing visualization-data tests; require all to pass.**

### Task 2: Journal renderer and atomic package

**Files:**
- Create: `simnibs_segmentation_determinism/journal_spatial_figures.py`
- Create: `simnibs_segmentation_determinism/tests/test_journal_spatial_figures.py`

**Interfaces:**
- Consumes: `JournalSpatialData` from Task 1 and the existing terminal-source validation contract.
- Produces: `build_journal_spatial_figures(config: ExperimentConfig) -> dict`, `_plot_spatial_overview(data)`, and `_plot_slice_atlas(data)`.

- [x] **Step 1: Write failing tests for intuitive labels, discrete 1-6 colour scale, physical coordinates, two complete three-format figures, 600-dpi PNG dimensions, captions, source hashes, checksums, idempotent reuse, read-only sources, and fail-closed cleanup.**

- [x] **Step 2: Run the new renderer tests and verify the expected missing-interface failures.**

- [x] **Step 3: Implement the two figures, caption/summary writers, provenance manifest, checksums, destination validation, and atomic publication.**

- [x] **Step 4: Render the fixture package and visually inspect both PNGs for crop, orientation, legibility, colour-bar semantics, and clipping.**

- [x] **Step 5: Run the new renderer tests and existing visualization tests; require all to pass.**

### Task 3: CLI, HPC launcher, and documentation

**Files:**
- Modify: `simnibs_segmentation_determinism/cli.py`
- Modify: `simnibs_segmentation_determinism/hpc/submit.sh`
- Modify: `simnibs_segmentation_determinism/README.md`
- Modify: `simnibs_segmentation_determinism/tests/test_cli.py`
- Modify: `simnibs_segmentation_determinism/tests/test_hpc_launchers.py`

**Interfaces:**
- Consumes: `build_journal_spatial_figures(config)` from Task 2.
- Produces: the `journal-spatial-figures` CLI and `submit.sh` modes, both non-submitting by themselves.

- [x] **Step 1: Write failing CLI and launcher tests proving the new mode uses `ANALYSIS_PYTHON`, calls no `sbatch`, and preserves the existing launcher defaults.**

- [x] **Step 2: Run the focused tests and verify expected failures.**

- [x] **Step 3: Implement the lazy CLI import, submitter mode, and README scope/output/submission instructions.**

- [x] **Step 4: Run `bash -n` and the focused CLI/HPC tests; require all to pass.**

### Task 4: Integrated verification

**Files:**
- Verify all files above.

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: one reviewable feature branch with no HPC submission or source mutation.

- [x] **Step 1: Run the complete `simnibs_segmentation_determinism/tests` suite with the verified local SimNIBS Python.**

- [x] **Step 2: Run `bash -n simnibs_segmentation_determinism/hpc/submit.sh`, `git diff --check`, and inspect the diff for module/path/resource/retry drift.**

- [x] **Step 3: Re-render the local fixture, inspect both figures, and verify every `SHA256SUMS` entry.**

- [x] **Step 4: Commit the completed feature without merging or submitting an HPC job.**
