# Repeatability Paper-Figure Clarity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a separate 12-PNG paper handoff whose titles, axes, legends and panel labels are understandable without pipeline terminology.

**Architecture:** Extend the existing source-bound figure renderer so every non-legacy figure is regenerated directly from the combined-analysis CSVs and derived ranking tables. Embed the actual displayed text in PNG metadata for testable accessibility, then use a focused builder to copy only the approved 12 images into the three-folder handoff while leaving all sources unchanged.

**Tech Stack:** Python 3.11, pandas, NumPy, Matplotlib, Pillow, pytest

**Spec:** `docs/superpowers/specs/2026-11-04-repeatability-paper-figure-clarity-design.md`

## Global Constraints

- Preserve the downloaded analysis, manuscript, and `final_132_repeatability_paper_figures_v1` byte-for-byte.
- Keep exactly the 12 figures currently represented in v1; do not restore the removed v3 ranking summary.
- Regenerate all ten non-legacy figures from canonical analysis tables.
- Retain the two legacy structural figures only as pixel-identical copies because their metric tables are unavailable locally.
- The final directory contains PNG files only in `01_simnibs_v3`, `02_simnibs_v4`, and `03_v3_vs_v4`.
- Use `temporal-interference field`, `target region`, `participant`, `New mesh each run`, and `Same mesh reused` in reader-facing text.
- Do not expose `optimizer-ROI`, `|E|`, `within-cell`, `difference-in-differences`, or `Subject IDs` in regenerated figure text.

## Review Focus

- Missing or malformed v4 ranking tables must fail closed instead of producing a misleading ranking panel; Task 1 validates table columns, row counts and targets.
- Exact displayed text must come from the rendered Matplotlib objects, not an unrelated label registry; Task 1 embeds extracted figure text in PNG metadata and tests it.
- Mean-field bootstrap labels must remain distinct from median-field primary-outcome labels; Task 1 asserts both terms in the relevant artifacts.
- The focused builder must not leak CSV, JSON, metadata files, or the removed v3 ranking image into the handoff; Task 2 checks the complete relative-path set.
- Legacy images must remain byte-identical while all source trees remain unchanged; Task 2 checks SHA-256 and tree manifests.

---

### Task 1: Reader-facing renderers and direct v4 paper figures

**Files:**
- Modify: `repeatability_figure_review/audit.py`
- Modify: `repeatability_figure_review/figures.py`
- Modify: `repeatability_figure_review/tests/test_audit.py`
- Modify: `repeatability_figure_review/tests/test_figures.py`

**Interfaces:**
- Consumes: `AuditBundle`, the four canonical 800-row input CSVs, and the final v4 ranking CSVs under `established/within_simnibs4`.
- Produces: `render_revised_figures(...) -> tuple[Artifact, ...]` with ten newly rendered non-legacy PNGs whose `Description` metadata contains the actual visible text.

- [ ] **Step 1: Write failing source-audit and rendering tests**

Add tests that create complete v4 ranking tables, reject a missing ranking table, render the figures, read each PNG `Description`, and assert:

```python
assert "Median temporal-interference field in target region (V/m)" in primary_text
assert "Mean temporal-interference field in target region" in bootstrap_text
assert not any(term in all_text for term in BANNED_READER_TERMS)
```

Also assert that the v4 repeatability and ranking outputs are not pixel copies of the supplied placeholder source PNGs.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `MPLCONFIGDIR=/tmp/repeatability-clarity-mpl /home/boyan/anaconda3/envs/simnibs_env/bin/python -m pytest repeatability_figure_review/tests/test_audit.py repeatability_figure_review/tests/test_figures.py -q`

Expected: FAIL because the ranking tables are not audited, PNG text metadata is absent, and v4 paper figures are still image composites.

- [ ] **Step 3: Implement the reader-facing rendering contract**

In `audit.py`, require and validate the two v4 ranking CSVs. In `figures.py`, add:

```python
def figure_text(figure: plt.Figure) -> tuple[str, ...]
def load_v4_ranking_tables(bundle: AuditBundle) -> tuple[pd.DataFrame, pd.DataFrame]
```

Make `_save_figure(...)` embed the text returned by `figure_text` in PNG `Description` metadata. Replace jargon in every existing title, axis and legend according to the spec.

- [ ] **Step 4: Render v4 repeatability and ranking figures directly**

Add direct renderers using the canonical run-level frame plus validated ranking tables. They must retain the established calculations and use the plain-language condition labels and outcome names from the spec. Remove the image-composition path for these two current paper equivalents.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `MPLCONFIGDIR=/tmp/repeatability-clarity-mpl /home/boyan/anaconda3/envs/simnibs_env/bin/python -m pytest repeatability_figure_review/tests/test_audit.py repeatability_figure_review/tests/test_figures.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add repeatability_figure_review/audit.py repeatability_figure_review/figures.py repeatability_figure_review/tests/test_audit.py repeatability_figure_review/tests/test_figures.py
git commit -m "feat: clarify repeatability figure language"
```

---

### Task 2: Image-only clarity handoff

**Files:**
- Create: `repeatability_figure_review/paper_figures.py`
- Create: `repeatability_figure_review/tests/test_paper_figures.py`
- Modify: `repeatability_figure_review/README.md`

**Interfaces:**
- Consumes: `render_revised_figures(...)` from Task 1, `tree_manifest(...)`, and the approved 12-file mapping in the spec.
- Produces: `build_clarity_directory(source_root, manuscript_root, output_root, *, bootstrap_iterations=20_000, bootstrap_max_repeats=40) -> dict[str, object]` and a CLI runnable as `python -m repeatability_figure_review.paper_figures`.

- [ ] **Step 1: Write the failing focused-builder test**

Build from realistic fixtures and assert the exact 12 relative paths, no non-PNG files, valid Pillow images, byte-identical legacy figures, absent v3 ranking summary, and unchanged analysis/manuscript tree manifests.

- [ ] **Step 2: Run the builder test and verify RED**

Run: `/home/boyan/anaconda3/envs/simnibs_env/bin/python -m pytest repeatability_figure_review/tests/test_paper_figures.py -q`

Expected: FAIL because `paper_figures.py` does not exist.

- [ ] **Step 3: Implement `build_clarity_directory(...)` and its CLI**

Render into an isolated temporary staging directory, copy only the approved paths into the three output folders, reject an existing output root, verify the output set and source manifests, and return counts plus `source_outputs_modified=False`.

- [ ] **Step 4: Run builder and regression tests**

Run: `MPLCONFIGDIR=/tmp/repeatability-clarity-mpl /home/boyan/anaconda3/envs/simnibs_env/bin/python -m pytest repeatability_figure_review/tests repeatability_combined_analysis/tests -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add repeatability_figure_review/paper_figures.py repeatability_figure_review/tests/test_paper_figures.py repeatability_figure_review/README.md
git commit -m "feat: build clarity-reviewed paper figures"
```

---

### Task 3: Production build and visual verification

**Files:**
- No repository files expected.
- Create outside git after verification: `/home/boyan/sandbox/final_132_repeatability_paper_figures_v2_clarity`

**Interfaces:**
- Consumes: Task 2 CLI and the local downloaded combined analysis plus manuscript figures.
- Produces: the final 12-PNG image-only directory.

- [ ] **Step 1: Build into a fresh temporary output root**

Run the Task 2 CLI with the downloaded analysis and current manuscript as sources, 20,000 bootstrap iterations, and 40 maximum repeats.

Expected: status `complete`, 12 PNGs, and `source_outputs_modified=false`.

- [ ] **Step 2: Verify file scope and image integrity**

Confirm exactly 12 files, all ending in `.png`, and open every file with Pillow `verify()`.

- [ ] **Step 3: Perform visual QA**

Render contact sheets for all three folders and inspect each individual figure at full resolution. Check terminology, target names, condition encodings, legends, clipping, whitespace and printed-size readability.

- [ ] **Step 4: Copy the verified directory to the final local path**

Copy only after proving the destination does not exist, then repeat the file-count and Pillow verification at the final path.

- [ ] **Step 5: Run final regression verification**

Run: `MPLCONFIGDIR=/tmp/repeatability-clarity-mpl /home/boyan/anaconda3/envs/simnibs_env/bin/python -m pytest repeatability_figure_review/tests repeatability_combined_analysis/tests -q`

Expected: PASS with no warnings or failures.
