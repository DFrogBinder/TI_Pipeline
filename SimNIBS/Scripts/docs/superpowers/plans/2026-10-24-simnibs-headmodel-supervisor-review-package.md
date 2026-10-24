# SimNIBS Head-Model Supervisor Review Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only HPC workflow that randomly selects one fully audited base-nine subject and assembles the approved eight-artifact supervisor package.

**Architecture:** A dependency-light Python core resolves and validates receipt-bound sources, persists an immutable random selection, and atomically builds and validates the package. A separate Slurm launcher loads the pinned SimNIBS 3.2.6 module only for the mesh-tag label export and exposes preflight, submit, status, and validate operations.

**Tech Stack:** Python 3 standard library, pytest, Bash, Slurm, SimNIBS 3.2.6 mesh I/O adapter.

**Spec:** `docs/superpowers/specs/2026-10-24-simnibs-headmodel-supervisor-review-package-design.md`

## Global Constraints

- Exactly nine canonical base-nine candidates are audited before selection.
- Exactly one subject is selected with `secrets.choice` and persisted immutably.
- Exactly eight scientific artifacts are physically materialized in the package.
- Every receipt-provided path must remain inside its canonical protected root.
- Scientific source paths must be non-empty regular files and not symlinks.
- Source roots are read-only; only the dedicated output root may be written.
- Original headreco labels are derived from original v3 mesh tissue tags on `T1fs_conform` using SimNIBS 3.2.6.
- The v4 mesh is the receipt-bound median representative mesh from the completed left-hippocampus campaign.
- Slurm uses `sheffield`, `SimNIBS/3.2.6-foss-2023a`, 8 CPUs, 32 GB, and 08:00:00.

## Review Focus

- A receipt names a path outside its protected root: preflight must reject it.
- One of nine subjects is incomplete: selection must not narrow the pool or continue.
- Selection is rerun: the exact original subject and source audit must be reused.
- A copied or derived file differs from its expected/source checksum: atomic promotion must not occur.
- A package contains a wrong subject, wrong artifact count, or stale manifest: validation must fail.

---

### Task 1: Receipt-bound source audit and immutable selection

**Files:**
- Create: `simnibs_headmodel_supervisor_review/__init__.py`
- Create: `simnibs_headmodel_supervisor_review/settings.py`
- Create: `simnibs_headmodel_supervisor_review/package.py`
- Create: `simnibs_headmodel_supervisor_review/tests/__init__.py`
- Create: `simnibs_headmodel_supervisor_review/tests/conftest.py`
- Create: `simnibs_headmodel_supervisor_review/tests/test_package.py`

**Interfaces:**
- Produces: `ReviewPaths`, `SubjectSources`, `audit_subject()`, `preflight()`, `select_subject()` and `load_selection()`.
- Consumes: canonical base-nine subjects and established v3/v4 receipt layouts.

- [x] **Step 1: Write failing tests** for exact eight-source resolution, root containment, all-nine fail-closed audit, and immutable random selection reuse.
- [x] **Step 2: Run the focused tests** and verify failures arise from the missing implementation.
- [x] **Step 3: Implement the settings, receipt parsers, source audit, and selection receipt** with no scientific imports at module import time.
- [x] **Step 4: Run the focused tests** and verify all Task 1 tests pass.

### Task 2: Atomic package build and validation

**Files:**
- Modify: `simnibs_headmodel_supervisor_review/package.py`
- Modify: `simnibs_headmodel_supervisor_review/tests/conftest.py`
- Modify: `simnibs_headmodel_supervisor_review/tests/test_package.py`

**Interfaces:**
- Consumes: `SubjectSources` and immutable selection from Task 1.
- Produces: `build_package()`, `validate_package()`, eight named scientific files, manifest, checksums, archive, and completion receipt.

- [x] **Step 1: Write failing tests** for physical copies, injected mesh-tag export, exact filenames, manifest provenance, checksum verification, atomic promotion, stale-output rejection, and completed-output reuse.
- [x] **Step 2: Run the focused tests** and verify expected failures.
- [x] **Step 3: Implement the atomic builder and validator**, reusing the established SimNIBS 3.2.6 label-export function through a lazy adapter.
- [x] **Step 4: Run the focused tests** and verify all Task 2 tests pass.

### Task 3: CLI, HPC launcher, and operator documentation

**Files:**
- Create: `simnibs_headmodel_supervisor_review/__main__.py`
- Create: `simnibs_headmodel_supervisor_review/hpc/build.slurm`
- Create: `simnibs_headmodel_supervisor_review/hpc/submit.sh`
- Create: `simnibs_headmodel_supervisor_review/README.md`
- Create: `simnibs_headmodel_supervisor_review/tests/test_hpc_launcher.py`
- Modify: `simnibs_headmodel_supervisor_review/tests/test_package.py`

**Interfaces:**
- Consumes: Task 1 preflight/selection and Task 2 build/validate APIs.
- Produces: `python -m simnibs_headmodel_supervisor_review` commands and `hpc/submit.sh` modes `preflight`, `submit`, `status`, and `validate`.

- [x] **Step 1: Write failing CLI and shell integration tests** covering scope output, output-root confirmation, one-job submission, exported variables, module/version pin, and validation exit status.
- [x] **Step 2: Run the focused tests** and verify failures arise from absent interfaces.
- [x] **Step 3: Implement the CLI, one-shot Slurm job, copy-paste-safe launcher, and README** with the exact scope and source/write boundaries.
- [x] **Step 4: Run `bash -n`, focused tests, and Python compile checks** and verify all pass.

### Task 4: Whole-change verification

**Files:**
- Verify all created files above.

**Interfaces:**
- Consumes: completed package creator and launcher.
- Produces: final verification evidence and a clean, reviewable diff limited to the new workflow and its documents.

- [x] **Step 1: Run the new workflow's complete test suite.**
- [x] **Step 2: Run relevant existing settings, head-model, and HPC-launcher regression tests.**
- [x] **Step 3: Run `bash -n`, `compileall`, `git diff --check`, and inspect the exact diff/status.**
- [x] **Step 4: Perform a final self-review against the five Review Focus items** because subagent delegation is unavailable for this task.
