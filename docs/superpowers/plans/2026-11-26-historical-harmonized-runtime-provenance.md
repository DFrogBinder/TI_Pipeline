# Historical Harmonized Runtime Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the historical native-field preflight validate the production SimNIBS 3.2.6 harmonized runtime provenance without changing data or HPC execution settings.

**Architecture:** Keep manifest discovery and stimulation validation unchanged. Replace the legacy runtime-key contract at the existing manifest boundary with the exact harmonized producer identity, protected by production-shaped regression tests.

**Tech Stack:** Python 3, pytest

**Spec:** `docs/superpowers/specs/2026-11-26-historical-harmonized-runtime-provenance-design.md`

## Global Constraints

- The completed combined dataset and task manifests are read-only.
- The full historical scope remains 3,200 simulations and 9,600 metric rows.
- Do not change Slurm settings, launch topology, extraction, ROIs, or metrics.
- Do not dereference retired historical path strings recorded in manifests.

## Review Focus

- A legacy-only runtime key must fail rather than silently select the wrong v3 workflow.
- A missing harmonized runtime mapping must fail before extraction.
- Wrong version, module, or backend values must each fail closed.
- Correct stimulation provenance must remain required after the runtime change.
- SimNIBS 4 manifest behavior must remain unchanged.

---

### Task 1: Production-shaped manifest regression

**Files:**
- Modify: `SimNIBS/Scripts/ti_current_repair/repeatability_v2/tests/test_historical_component_analysis.py`
- Modify: `SimNIBS/Scripts/ti_current_repair/repeatability_v2/historical.py`

**Interfaces:**
- Consumes: `_manifest_anode(manifest, version, target, subject, condition, repeat, manifest_path)`
- Produces: strict validation of `simnibs_326_harmonized_runtime`

- [ ] **Step 1: Replace the v3 fixture runtime with the production harmonized key and exact producer identity**

- [ ] **Step 2: Add parametrized tests for missing, legacy-only, wrong-version, wrong-module, and wrong-backend provenance**

- [ ] **Step 3: Run the focused tests and verify they fail because the validator still expects `simnibs_326_runtime`**

Run: `python3 -m pytest ti_current_repair/repeatability_v2/tests/test_historical_component_analysis.py -q`

Expected: FAIL in harmonized v3 discovery/provenance tests.

- [ ] **Step 4: Add exact harmonized runtime constants and update `_manifest_anode` to enforce them**

- [ ] **Step 5: Run the focused tests and verify they pass**

Run: `python3 -m pytest ti_current_repair/repeatability_v2/tests/test_historical_component_analysis.py -q`

Expected: PASS.

- [ ] **Step 6: Run the repeatability-v2 test package and repository verification available locally**

Run: `python3 -m pytest ti_current_repair/repeatability_v2/tests -q`

Run: `python3 -m compileall -q ti_current_repair/repeatability_v2`

- [ ] **Step 7: Review the diff and commit only the provenance repair, tests, design, and plan**
