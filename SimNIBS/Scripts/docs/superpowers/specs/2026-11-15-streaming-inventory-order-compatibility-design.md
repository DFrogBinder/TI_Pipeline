# Streaming Inventory Order Compatibility Design

## Intent

Allow resumable dataset finalization to reproduce the inventory SHA-256 values
created by the established checksum verifier. The correction must preserve the
bounded-memory design and allow the existing 126 valid finalization checkpoints
to remain reusable.

## Confirmed failure

The checksum verifier globally sorts complete relative file paths. The streaming
inventory implementation instead walks directory entries by basename and emits
an entire directory subtree before later siblings. These orders differ when a
directory name is a prefix of a file name, such as `a/x` and `a.txt`. Identical
path, size, and file-SHA records therefore produced different order-sensitive
inventory hashes.

The production failure contained the same 1,208 unique entries as the current
verified destination, with no path, size, or SHA differences and no duplicates.
Only serialized order differed.

## Goals

- Emit streaming inventory records in the same global relative-path order as
  `build_file_inventory` and `build_materialized_inventory`.
- Keep memory bounded by directory fan-out rather than total file count.
- Apply the same ordering contract to destination and materialized-source walks.
- Preserve atomic stream replacement, symlink handling, cycle detection, special
  file rejection, metadata fingerprints, receipts, and deletion guards.

## Non-goals

- Do not recopy or re-checksum the 9.23 TB dataset.
- Do not change the consolidation plan, archive roots, or receipt schemas.
- Do not submit retirement or delete any source data.

## Approaches considered

1. **Collect and globally sort every record in memory.** This exactly matches the
   legacy implementation but abandons the bounded-memory purpose of streaming.
2. **Write unsorted chunks and externally merge-sort them.** This is bounded but
   adds temporary-file complexity and I/O for an ordering property available
   directly from the directory tree.
3. **Sort each directory by its logical lexical prefix.** Treat a directory child
   as `name/` and a file child as `name` before recursively emitting the selected
   child. All descendants share the directory prefix, so this produces the same
   global path order while retaining per-directory memory bounds.

Approach 3 is selected because it is the smallest compatible correction and
preserves streaming behavior.

## Verification

A regression fixture containing `a.txt` and `a/x` must fail against the old
implementation, then pass with identical entry order and inventory SHA-256 under
both destination and materialized-source modes. Existing streaming, finalization,
archive, readiness, completion, and workflow tests must remain green.

