# Resumable Repeatability-Dataset Retirement Readiness Design

## Intent

Finish the interrupted `final_132_repeatability_combined_dataset_v1`
finalization, create and verify the required Shared archive, and produce an
exact deletion-ready plan with one login-node command. The workflow must stop
before submitting any deletion job.

The current scientific copy is already checksum-matched: all 215 planned copy
items matched, with zero content mismatches and no source modification. The
outstanding work is packaging, independently resumable inventory creation,
Shared archival, and receipt-bound retirement planning.

## Goals

- Recover the current production state where `dataset_manifest.json` says
  `complete`, but `file_inventory.json` and `_consolidation/complete.json` are
  absent.
- Build the final per-file destination inventory without repeating the
  completed source/destination checksum campaign.
- Checkpoint long-running work by planned dataset item and Shared archive root.
- Resume safely after SSH loss, tmux loss, process interruption, or login-node
  restart.
- Automatically run dataset finalization, Shared archive preflight/copy/quick
  verification/checksum verification, and retirement planning.
- Write an external `retirement_ready.json` receipt containing all exact hashes
  and the explicit deletion command.
- Provide read-only progress/status output throughout.
- Preserve all source and scientific destination files until a later explicit
  deletion confirmation.

## Non-goals and anti-goals

- Never call `retire-submit`, `sbatch`, `rm`, or the deletion launcher.
- Never create new simulations, meshes, FEM solves, or analysis outputs.
- Never silently weaken an observed checksum mismatch into a metadata match.
- Never describe a copied or checksum-matched dataset as deletion-ready until
  the dataset completion receipt, Shared archive completion receipt, retirement
  disposition, scheduler audit, and retirement plan all validate.
- Never require a clean restart of a completed multi-terabyte stage merely
  because a later stage was interrupted.

## Current constraints

- Production dataset root:
  `/mnt/parscratch/users/cop23bi/final_132_repeatability_combined_dataset_v1`.
- Shared archive root:
  `/shared/boyan/Shared/retired_repeatability_v3_sources_v1`.
- Scope: 10 subjects, 2 SimNIBS versions, 2 targets, 2 conditions, 40
  repeats, 80 cells, 3,200 simulations, and 215 copy-plan items.
- The workflow runs on a Stanage login node, not in Slurm.
- `/shared/boyan/Shared` is login-node-only.
- Existing plan SHA-256:
  `1ab5c604e4d965ae70a63ca9984c878db2631f89a1f99b5927196e25d696a32c`.
- Existing matched checksum evidence must remain immutable and plan-bound.

## Approaches considered

### 1. Resumable finalization and retirement preparation — selected

Reuse the immutable matched checksum receipts, create a checkpointed final
destination inventory, checkpoint Shared archive verification per archive
root, and orchestrate every remaining reversible stage. This preserves the
scientific integrity model and avoids repeating completed work.

### 2. Re-run the current checksum command and then archive

This is a smaller code change, but it repeats multi-terabyte hashing and still
has a non-resumable final inventory stage. It was rejected because another
login-node interruption could discard several days of work again.

### 3. Trust the 215 item receipts and omit the per-file inventory

This would be fast, but removes the standalone per-file catalogue expected by
the completion receipt. It was rejected because the consolidated dataset
should remain independently auditable after source retirement.

## Architecture

### Checkpointed dataset finalization

Add a dedicated finalization component rather than overloading
`verify_dataset()`.

For each of the 215 planned items it will:

1. Validate the immutable top-level checksum receipt and the recorded hash of
   the item receipt.
2. Hash the current destination item once and compare its aggregate inventory
   SHA-256 with the previously recorded `destination_inventory_sha256`.
3. Stream the per-file entries to
   `_consolidation/finalization/items/<item-hash>.jsonl` and write a small
   adjacent receipt containing the entry-stream SHA-256, item inventory
   SHA-256, counts, bytes and metadata fingerprint.
4. On resume, reuse a checkpoint only when its receipt binding and current
   path/type/device/inode/size/mtime/ctime fingerprint remain identical.
   Changed metadata invalidates only that item and forces it to be re-hashed.

Inventory traversal is lexicographically ordered and streaming: neither a
large item nor the complete dataset catalogue is accumulated in memory. After
every item is validated, a k-way streaming merge combines the sorted item
streams and explicitly hashed control files into the final
`file_inventory.json`. Duplicate or overlapping dataset-relative paths fail
closed. Operational checkpoint files are excluded from the scientific
inventory and bound separately in the final completion receipt. This bounds
memory independently of total file count and avoids reproducing the existing
all-entries-in-memory failure risk.

The finalizer recognizes the present partial state. It changes the premature
manifest state from `complete` to `finalizing` before long work begins. Only
after the final inventory is atomically installed does it write a new
`completed_utc`, promote the manifest to `complete`, and atomically write
`_consolidation/complete.json`. The completion receipt is the authoritative
gate; a manifest alone never authorizes retirement.

### Completion validation

Split validation into two explicit levels:

- Normal `status` validates receipt hashes, plan binding, exact scope and saved
  metadata fingerprints without re-reading every file body.
- Explicit `checksum` remains the deep content-validation operation.

This avoids an undocumented multi-day checksum every time an operator asks for
status. Metadata validation is not allowed to replace a failed checksum; it is
used only after a matched cryptographic checkpoint has been recorded.

### Resumable Shared archive

Keep the existing four-root archive plan and copy semantics. Extend checksum
verification so each archive root writes an immutable checkpoint containing
its source and destination inventory hashes plus streamed entry files. A rerun
reuses a root only when its receipt binding and metadata fingerprints remain
unchanged; otherwise only that root is re-hashed. Archive inventories use the
same bounded-memory streaming representation as dataset finalization.

The external archive completion receipt is written only when all four roots
match. Copying remains resumable and never deletes a source.

### Retirement-readiness orchestration

Add a `prepare-retirement` CLI mode that acquires an exclusive workflow lock
and executes these idempotent stages in order:

1. Resume dataset finalization.
2. Validate dataset completion.
3. Run Shared archive preflight.
4. Resume Shared archive copy.
5. Run quick Shared archive verification.
6. Resume Shared archive checksum verification.
7. Validate Shared archive completion.
8. Build the exhaustive retirement plan and audit the live Slurm queue.
9. Write
   `/mnt/parscratch/users/cop23bi/deletion_job_logs/repeatability_dataset_v1/retirement_ready.json`.

The readiness receipt binds the dataset completion receipt SHA-256, archive
completion receipt SHA-256, retirement-disposition SHA-256, retirement-plan
SHA-256, exact target list, scheduler snapshot hash, and the literal
`retire-submit` command. It records `jobs_submitted: 0` and
`deletion_performed: false`.

`prepare-retirement` stops successfully at this point. The existing explicit
`retire-submit` command remains the only deletion entry point.

### Read-only progress reporting

Add `readiness-status` with:

- finalization item count out of 215;
- dataset completion receipt state;
- archive copy and checksum-root counts out of 4;
- archive completion receipt state;
- retirement-plan/readiness receipt state;
- active lock owner and last recorded failure, if any;
- `deletion_ready`, `jobs_submitted`, and `deletion_performed` booleans.

The command never starts, resumes or mutates a workflow.

## Data flow and integrity boundaries

The existing checksum receipts prove source/destination equality at their
recorded time. The finalization pass independently re-hashes every destination
file once, compares each item with its bound destination inventory hash, and
then assembles the portable per-file catalogue. No source checksum is skipped
when creating the Shared archive: all four Shared source/destination roots are
cryptographically compared before the archive receipt is written.

Retirement planning continues to require:

- a valid dataset completion receipt;
- a valid Shared archive completion receipt for every archive-required root;
- an exhaustive retirement disposition;
- exact allowlist scope and non-overlapping target paths;
- no active Slurm command referencing a retirement target.

## Failure handling and recovery

- Every long item/root produces its receipt only after successful validation.
- Writes use temporary files followed by `os.replace()`.
- A mismatch stops immediately without deleting checkpoints from previously
  matched items.
- A changed item invalidates and recomputes only that item.
- Concurrent preparation is rejected by an exclusive lock containing host,
  PID, start time and command.
- Rerunning `prepare-retirement` after interruption resumes from receipts.
- Existing nonmatching completion or readiness receipts are never overwritten.
- Stage failures are recorded externally with the exception type and message;
  no traceback or failure receipt is interpreted as scientific completion.

## Operator interface

The eventual HPC entry point is one login-node command run in tmux:

```bash
bash repeatability_dataset_consolidation/hpc/submit.sh prepare-retirement
```

Progress is checked independently with:

```bash
bash repeatability_dataset_consolidation/hpc/submit.sh readiness-status
```

When the first command completes, the next operational step is the explicit
receipt-bound `retire-submit` command printed by the workflow. The workflow
does not execute that command.

## Testing and acceptance criteria

Automated miniature-dataset tests must prove:

- recovery from `manifest=complete` with no inventory/completion receipt;
- item checkpoint creation and reuse after simulated interruption;
- changed metadata invalidates one checkpoint;
- changed content with restored mtime fails checksum comparison;
- duplicate/overlapping inventory entries fail closed;
- completion receipt is absent until every item is validated;
- the manifest cannot independently authorize retirement;
- Shared archive copy/checksum resumes per root;
- orchestration reaches deletion-ready state without invoking `sbatch`, `rm`,
  `retire-submit`, or the deletion launcher;
- a source, archive or Slurm-scope mismatch prevents readiness;
- `readiness-status` is read-only;
- existing copy, verification, archive and retirement tests remain green;
- shell syntax, ShellCheck, compilation and whitespace checks pass.

## Operational handoff

After implementation and local verification, the feature is merged and pushed
only on explicit request. The HPC instructions will require pulling the exact
commit, confirming the existing plan SHA-256, running `readiness-status`, and
then launching `prepare-retirement` in tmux. No deletion command will be
included in the automatic launch block.
