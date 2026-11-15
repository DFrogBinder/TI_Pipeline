# Streaming Inventory Order Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make bounded-memory inventory streams reproduce the established globally sorted inventory hashes.

**Architecture:** Preserve recursive streaming, but order siblings by the lexical prefix they contribute to complete relative paths: `name` for files and `name/` for directories. Use the same rule for destination and materialized-source traversal.

**Tech Stack:** Python 3, pathlib, pytest

**Spec:** `docs/superpowers/specs/2026-11-15-streaming-inventory-order-compatibility-design.md`

## Global Constraints

- Preserve bounded memory: retain only one directory's children at a time.
- Do not change inventory schemas, plan hashes, receipts, archive scope, or deletion behavior.
- Preserve symlink materialization and special-file rejection semantics.
- Use test-first development and demonstrate the original failure before production edits.

## Review Focus

- A directory whose name prefixes a sibling file must sort by complete logical path.
- Materialized directory symlinks must follow the same ordering contract.
- Broken links, special files, and source-link cycles must still fail closed.
- Existing valid finalization checkpoints must remain reusable.
- The automatic readiness workflow must continue to stop before deletion.

---

### Task 1: Restore legacy-compatible streaming order

**Files:**
- Modify: `repeatability_dataset_consolidation/tests/test_streaming_inventory.py`
- Modify: `repeatability_dataset_consolidation/streaming_inventory.py`

**Interfaces:**
- Consumes: `_records(root, materialize_links, exclude)` and the established `FileInventory.inventory_sha256` contract.
- Produces: globally ordered records for `write_inventory_stream` and `metadata_fingerprint` without changing their public signatures.

- [x] **Step 1: Add failing destination and materialized-source regression tests**

Create fixtures containing sibling paths `a.txt` and `a/x`. Assert that emitted
paths and `inventory_sha256` exactly match `build_file_inventory` and
`build_materialized_inventory`, respectively.

- [x] **Step 2: Run the focused regressions and verify RED**

Run: `python3 -m pytest repeatability_dataset_consolidation/tests/test_streaming_inventory.py -q`

Expected: both new prefix-order assertions fail because the stream emits `a/x`
before `a.txt`.

- [x] **Step 3: Implement per-directory logical-prefix ordering**

In `streaming_inventory.py`, stage only the current directory's validated child
records and sort them using `child.name + "/"` for directory targets and
`child.name` for file targets before recursion/yield. Apply this to both
destination and materialized-source traversal.

- [x] **Step 4: Run the focused tests and verify GREEN**

Run: `python3 -m pytest repeatability_dataset_consolidation/tests/test_streaming_inventory.py -q`

Expected: all streaming-inventory tests pass.

- [x] **Step 5: Run consolidation regression verification**

Run: `python3 -m pytest repeatability_dataset_consolidation/tests -q`

Expected: all consolidation tests pass with no failures.

- [x] **Step 6: Run static and shell checks**

Run: `python3 -m compileall -q repeatability_dataset_consolidation`

Run: `bash -n repeatability_dataset_consolidation/hpc/submit.sh`

Expected: both commands exit zero.

- [x] **Step 7: Commit the tested correction**

Commit the two documentation files, regression tests, and minimal implementation
with a message describing legacy-compatible streaming inventory ordering.

