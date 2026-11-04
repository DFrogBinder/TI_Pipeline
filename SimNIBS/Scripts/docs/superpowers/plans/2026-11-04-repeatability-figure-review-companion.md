# Repeatability Figure Review Companion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify an immutable-source companion package that repairs the combined repeatability figures, covers the current paper figure set as far as the downloaded data permit, and provides an illustrated figure guide.

**Architecture:** A small Python package validates and inventories the downloaded analysis, renders revised figures and paper equivalents from canonical tables, then assembles Markdown/PDF documentation and checksums. Generated output is isolated from both source analysis and manuscript.

**Tech Stack:** Python 3.12, pandas, NumPy, matplotlib, Pillow, ReportLab, pytest.

**Spec:** `docs/superpowers/specs/2026-11-04-repeatability-figure-review-companion.md`

## Global Constraints

- The downloaded analysis and manuscript are read-only.
- Participants are the inferential unit; 40 repeats are technical replicates.
- Version-specific ROI masks are aligned, not identical.
- No therapeutic or biological threshold claims.
- Missing mesh/tissue metrics must be reported, never inferred.
- All generated artifacts require SHA-256 coverage and rendered visual QA.

## Review Focus

- Source package missing a required figure or table must fail before partial output is labelled complete.
- A 100th-percentile boundary must remain visibly inside the plotting area.
- Zero technical-SD values must be distinguishable from missing observations.
- Paper-equivalence statuses must not call legacy structural figures current 9+1 equivalents.
- PDF assembly must preserve legible labels and all 20 original-figure explanations.

---

### Task 1: Source audit and inventory

**Files:**
- Create: `repeatability_figure_review/__init__.py`
- Create: `repeatability_figure_review/audit.py`
- Create: `repeatability_figure_review/tests/test_audit.py`

**Interfaces:**
- Consumes: downloaded analysis root and manuscript root paths.
- Produces: `audit_sources(source_root, manuscript_root) -> AuditBundle` and CSV-ready inventory/equivalence records.

- [ ] Write failing tests for the 20-PNG inventory, five manuscript mappings, source-scope validation, and legacy/data-gap statuses.
- [ ] Run the focused tests and confirm the expected missing-module/API failure.
- [ ] Implement source validation, checksums, metadata extraction, and explicit equivalence mapping.
- [ ] Run focused tests and the package test suite.
- [ ] Commit the audited source-boundary implementation.

### Task 2: Revised figures and paper equivalents

**Files:**
- Create: `repeatability_figure_review/figures.py`
- Create: `repeatability_figure_review/tests/test_figures.py`

**Interfaces:**
- Consumes: `AuditBundle`, canonical input tables, and established output tables.
- Produces: `render_revised_figures(bundle, output_root) -> list[Artifact]`.

- [ ] Write failing tests for two-target outputs, visible 100th-percentile points, explicit deterministic-v3 summary, log-floor encoding, and deterministic bootstrap summaries.
- [ ] Run focused tests and confirm failures are caused by absent rendering APIs.
- [ ] Implement combined/repaired figures and updated paper Figures 1-3.
- [ ] Copy paper Figures 4-5 into a legacy directory without changing pixels and record their classifications.
- [ ] Run focused tests and the package test suite.
- [ ] Commit the figure-rendering implementation.

### Task 3: Illustrated guide, manifest, and package build

**Files:**
- Create: `repeatability_figure_review/report.py`
- Create: `repeatability_figure_review/build.py`
- Create: `repeatability_figure_review/__main__.py`
- Create: `repeatability_figure_review/tests/test_report.py`
- Create: `repeatability_figure_review/tests/test_end_to_end.py`
- Create: `repeatability_figure_review/README.md`

**Interfaces:**
- Consumes: inventory, equivalence table, original figures, and revised artifacts.
- Produces: Markdown/PDF guide, manifest, checksums, and a complete companion directory.

- [ ] Write failing tests for all-figure guide coverage, page creation, manifest completeness, SHA-256 verification, and source-tree immutability.
- [ ] Run focused tests and confirm failures are caused by absent report/build APIs.
- [ ] Implement the Markdown guide, ReportLab PDF, package manifest, checksum writer, and CLI.
- [ ] Run the focused tests, all new tests, and the existing combined-analysis test suite.
- [ ] Build the real companion package from the downloaded analysis.
- [ ] Render the guide PDF to page images and visually inspect all revised figures and guide pages.
- [ ] Verify checksums and compare source-tree manifests before and after the build.
- [ ] Commit the documentation and package builder.

