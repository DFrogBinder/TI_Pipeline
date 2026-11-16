# Covered-source Retirement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separately confirmed, receipt-bound deletion workflow for exactly the ten completed-dataset-covered roots while preserving the four archive roots and the existing full retirement behavior.

**Architecture:** Introduce a canonical covered allowlist and make retirement request handling scope-aware through an explicit workflow value. Reuse existing dataset, disposition, scheduler, identity, launcher, resume, and terminal-proof gates, while requiring zero Shared-archive disposition counts for every covered target.

**Tech Stack:** Python 3.12, argparse, pytest, Bash, Slurm launcher wrappers

**Spec:** `docs/superpowers/specs/2026-11-16-covered-retirement-design.md`

## Global Constraints

- Covered scope is exactly the canonical full retirement allowlist minus all `config.archive_sources` keys.
- The four archive roots, corrected input roots, v4 scaffolds, combined dataset, and segmentation staging must not be covered targets.
- Existing full retirement modes and their archive-completion requirement remain unchanged.
- Plan and status are read-only; only explicitly confirmed submit modes may invoke the hardened deletion launcher.
- Covered submission requires exact dataset and disposition hashes and must reject an archive hash.

## Review Focus

- A forged covered request containing one archive key must fail before resume or status.
- A covered disposition with a nonzero Shared-archive count must fail even if all receipts otherwise validate.
- Existing full and covered requests must not collide or resume one another.
- Missing, non-integer, boolean, or negative disposition counters must fail closed.
- Covered submission must preserve target order and submit exactly ten unique launcher calls in production scope.

---

### Task 1: Canonical covered allowlist

**Files:**
- Modify: `repeatability_dataset_consolidation/settings.py`
- Modify: `repeatability_dataset_consolidation/tests/test_settings.py`

**Interfaces:**
- Produces: `covered_retirement_allowlist(config: DatasetConfig) -> dict[str, Path]`
- Consumes: `retirement_allowlist(config)` and `config.archive_sources`

- [ ] Write tests asserting exact ten-key production order, disjoint archive keys, and excluded protected/output/scaffold roots.
- [ ] Run the focused settings test and verify it fails because the helper is absent.
- [ ] Implement the helper as an ordered filter of the existing full allowlist.
- [ ] Run the focused settings tests and verify they pass.
- [ ] Commit the task.

### Task 2: Scope-aware covered planning

**Files:**
- Modify: `repeatability_dataset_consolidation/retirement.py`
- Modify: `repeatability_dataset_consolidation/tests/test_retirement.py`

**Interfaces:**
- Produces: `plan_covered_retirement(config: DatasetConfig) -> RetirementPlan`
- Produces: a scope/workflow binding in `RetirementPlan` and its plan hash
- Consumes: `covered_retirement_allowlist(config)` and the bound disposition summary

- [ ] Write tests proving a covered plan succeeds without an archive receipt, has exactly the covered targets in canonical order, and rejects malformed or nonzero Shared-archive counters.
- [ ] Run the focused tests and verify expected failures.
- [ ] Refactor the internal planner to accept only a code-owned scope definition; keep `plan_retirement` full-only and add `plan_covered_retirement`.
- [ ] Bind the scope in the retirement hash and validate exact disposition summary records and counters.
- [ ] Run the focused retirement tests and verify they pass.
- [ ] Commit the task.

### Task 3: Separate covered request, resume, and terminal status

**Files:**
- Modify: `repeatability_dataset_consolidation/retirement.py`
- Modify: `repeatability_dataset_consolidation/tests/test_retirement.py`

**Interfaces:**
- Produces: `submit_covered_retirement`, `resume_covered_retirement`, and `covered_retirement_status`
- Consumes: the scope-bound plan and exact confirmations from Task 2

- [ ] Write tests for distinct request/complete paths, exactly one launcher call per covered target, cross-scope collision refusal, forged archive-key refusal, partial resume, and terminal receipt plus absence proof.
- [ ] Run the focused tests and verify expected failures.
- [ ] Make request loading, submission, resume, and status scope-aware without accepting operator-selected subsets.
- [ ] Run the focused retirement tests and verify they pass without changing full-mode tests.
- [ ] Commit the task.

### Task 4: CLI, launcher, and operator documentation

**Files:**
- Modify: `repeatability_dataset_consolidation/cli.py`
- Modify: `repeatability_dataset_consolidation/hpc/submit.sh`
- Modify: `repeatability_dataset_consolidation/README.md`
- Modify: `repeatability_dataset_consolidation/tests/test_cli.py`
- Modify: `repeatability_dataset_consolidation/tests/test_hpc_launcher.py`

**Interfaces:**
- Produces: `covered-retire-plan`, `covered-retire-submit`, and `covered-retire-status`
- Consumes: Task 3 covered operations

- [ ] Write CLI and wrapper tests proving plan/status are read-only, submission requires `--confirm-covered-retirement`, archive hashes are rejected, and all modes pass arguments safely.
- [ ] Run the focused tests and verify expected failures.
- [ ] Implement the CLI branches and wrapper allowlist, retaining old modes unchanged.
- [ ] Document the exact two-phase plan/submit/status workflow and preserved-root boundary.
- [ ] Run CLI, launcher, retirement, and settings tests and verify they pass.
- [ ] Commit the task.

### Task 5: Regression and operational verification

**Files:**
- Modify only if a failing test exposes a defect.

**Interfaces:**
- Consumes: all prior task interfaces
- Produces: verified branch ready for independent review

- [ ] Run the full `repeatability_dataset_consolidation/tests` suite.
- [ ] Run `bash -n repeatability_dataset_consolidation/hpc/submit.sh` and inspect the production covered plan scope through unit assertions.
- [ ] Run `git diff --check` and review changed paths for deletion or scope leakage.
- [ ] Commit any TDD-driven fixes, then record final green evidence.

