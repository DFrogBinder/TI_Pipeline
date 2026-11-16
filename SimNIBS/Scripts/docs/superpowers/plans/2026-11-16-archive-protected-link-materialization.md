# Protected Link Materialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Permit the legacy Shared archive to materialize links from the configured protected corrected-input root without relaxing rejection of arbitrary external targets.

**Architecture:** Extend archive planning's approved link-target boundary with `DatasetConfig.protected_materialization_roots`. Bind that allowlist into the archive plan hash and preflight provenance, and reject overlap with archive or retirement targets.

**Tech Stack:** Python 3, pathlib, pytest, rsync-backed archive workflow

**Spec:** `docs/superpowers/specs/2026-11-16-archive-protected-link-materialization-design.md`

## Global Constraints

- The four archive roots and 14 retirement targets do not change.
- Only configured protected materialization roots are newly approved.
- Protected roots are never retirement targets and are not copied wholesale.
- Arbitrary external and broken symbolic links remain errors.
- Existing finalization checkpoints remain reusable.

## Review Focus

- Protected file links must become regular files in Shared with identical bytes.
- Unapproved external links must still fail archive preflight.
- Protected roots overlapping retirement targets must fail closed.
- The plan hash must change when the protected-root allowlist changes.
- Empty protected-root configuration must retain existing archive behavior.

---

### Task 1: Extend the archive link boundary safely

**Files:**
- Modify: `repeatability_dataset_consolidation/archive.py`
- Modify: `repeatability_dataset_consolidation/tests/test_archive.py`
- Modify: `repeatability_dataset_consolidation/README.md`

**Interfaces:**
- Consumes: `DatasetConfig.protected_materialization_roots` and `retirement_allowlist(config)`
- Produces: unchanged public archive functions with a protected-root-aware plan and preflight payload

- [ ] **Step 1: Write failing regression tests**

Add tests proving protected links materialize and verify, overlap with retirement targets is rejected, arbitrary external links remain rejected, and the protected-root allowlist affects the plan hash.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python3 -m pytest repeatability_dataset_consolidation/tests/test_archive.py -q`

Expected: new protected-link tests fail because `_plan` currently allows only the four archive sources.

- [ ] **Step 3: Implement the minimal protected-root policy**

Resolve and validate configured protected roots, add them only to link-target approval, bind them into the plan hash and preflight output, and retain existing link materialization and rejection behavior.

- [ ] **Step 4: Run focused and full verification**

Run the archive tests, the complete consolidation test directory, `compileall`, and `bash -n repeatability_dataset_consolidation/hpc/submit.sh`.

- [ ] **Step 5: Review the diff**

Confirm no archive source, retirement target, Shared destination, launcher, or destructive behavior changed.
