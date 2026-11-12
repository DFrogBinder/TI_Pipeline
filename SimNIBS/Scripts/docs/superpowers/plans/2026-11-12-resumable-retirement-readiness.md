# Resumable Repeatability-Dataset Retirement Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one resumable login-node workflow that finishes dataset finalization, completes the Shared archive, produces an exact deletion-ready receipt, and stops before deletion.

**Architecture:** Add bounded-memory streamed inventory checkpoints shared by dataset and archive finalization. Build dataset finalization and retirement readiness as separate idempotent components, then expose them through new CLI modes and the existing shell launcher. Existing matched checksum evidence remains immutable and every destructive entry point remains separate and explicitly confirmed.

**Tech Stack:** Python 3.9-compatible standard library, pytest, Bash, rsync, Slurm read-only `squeue`, JSON/JSONL receipts.

**Spec:** `docs/superpowers/specs/2026-11-12-resumable-retirement-readiness-design.md`

## Global Constraints

- Production plan SHA-256 remains `1ab5c604e4d965ae70a63ca9984c878db2631f89a1f99b5927196e25d696a32c`.
- Scope remains 10 subjects, 80 cells, 3,200 simulations and 215 copy-plan items.
- The workflow runs only on a login node; it submits zero Slurm jobs.
- `prepare-retirement` must never invoke `retire-submit`, `sbatch`, `rm`, or the hardened deletion launcher.
- Existing checksum receipts are immutable inputs; mismatches fail closed.
- Scientific source and consolidated destination files are read-only throughout readiness preparation.
- Shared archive copy remains the only bulk-copy mutation.
- Code must run under Stanage Python 3.9.
- Every long-running item/root is independently resumable.

## Review Focus

- A same-size content change with restored mtime must still invalidate a checkpoint through ctime or checksum comparison; cover in Tasks 1 and 2.
- Overlapping plan destinations must never generate duplicate final inventory paths; cover in Task 2.
- A manifest marked complete without the authoritative completion receipt must remain non-retirable; cover in Task 2.
- A stale Shared archive checkpoint must recompute only its affected root and never be accepted as complete; cover in Task 3.
- The automatic orchestrator must remain incapable of deletion even if a valid readiness receipt exists; cover in Task 4.

---

### Task 1: Add bounded-memory streamed inventory primitives

**Files:**
- Create: `repeatability_dataset_consolidation/streaming_inventory.py`
- Modify: `repeatability_dataset_consolidation/inventory.py`
- Create: `repeatability_dataset_consolidation/tests/test_streaming_inventory.py`

**Interfaces:**
- Produces: `InventoryStreamSummary`, `write_inventory_stream(...)`, `validate_inventory_stream(...)`, `merge_inventory_streams(...)`, and `metadata_fingerprint(...)`.
- Preserves: `FileInventory`, `build_file_inventory(...)`, and `build_materialized_inventory(...)` behavior for compatibility.

- [ ] **Step 1: Write failing tests for streamed destination inventories**

  Add tests asserting lexicographically ordered JSONL entries, the same `inventory_sha256` as `build_file_inventory()`, exact entry count/bytes, and bounded incremental output rather than a returned entries list.

- [ ] **Step 2: Run the focused tests and verify RED**

  Run: `python3 -m pytest repeatability_dataset_consolidation/tests/test_streaming_inventory.py -q`

  Expected: import failure because `streaming_inventory` does not exist.

- [ ] **Step 3: Implement streamed inventory generation**

  Implement:

  ```python
  @dataclass(frozen=True)
  class InventoryStreamSummary:
      root: str
      entry_count: int
      total_bytes: int
      inventory_sha256: str
      entries_sha256: str
      metadata_sha256: str

  def write_inventory_stream(
      root: Path,
      entries_path: Path,
      *,
      materialize_links: bool = False,
      exclude: Iterable[Path] = (),
  ) -> InventoryStreamSummary: ...
  ```

  Traverse directories in sorted order, hash one file at a time, write one canonical JSON object per line, calculate the legacy-compatible list hash incrementally, and atomically replace the stream.

- [ ] **Step 4: Add failing tests for metadata validation and materialized links**

  Cover unchanged reuse, changed contents with restored mtime, changed inode, broken links, internal materialized links, and unsupported special files.

- [ ] **Step 5: Implement metadata fingerprints and stream validation**

  Implement:

  ```python
  def metadata_fingerprint(
      root: Path,
      *,
      materialize_links: bool = False,
      exclude: Iterable[Path] = (),
  ) -> str: ...

  def validate_inventory_stream(
      entries_path: Path,
      summary: InventoryStreamSummary,
  ) -> None: ...
  ```

  Fingerprints bind logical path, type, device, inode, size, mtime_ns and ctime_ns. Stream validation checks canonical line encoding, ordering, count, bytes and hashes without reading scientific file bodies.

- [ ] **Step 6: Add failing merge tests**

  Cover k-way sorted merge, logical path prefixes, control-file entries, duplicate paths, malformed streams and output atomicity.

- [ ] **Step 7: Implement the streaming merge**

  Implement:

  ```python
  def merge_inventory_streams(
      streams: Sequence[tuple[Path, Path]],
      control_entries: Iterable[InventoryEntry],
      output_path: Path,
      *,
      root: Path,
  ) -> InventoryStreamSummary: ...
  ```

  Use `heapq.merge` semantics and stream a valid `FileInventory.as_dict()`-compatible JSON document without retaining all entries.

- [ ] **Step 8: Run focused and existing inventory tests**

  Run: `python3 -m pytest repeatability_dataset_consolidation/tests/test_streaming_inventory.py repeatability_dataset_consolidation/tests/test_completion.py::test_file_inventory_is_sorted_and_rejects_symlinks -q`

- [ ] **Step 9: Commit Task 1**

  Commit message: `feat: add resumable streaming inventories`

### Task 2: Implement checkpointed dataset finalization

**Files:**
- Create: `repeatability_dataset_consolidation/finalization.py`
- Modify: `repeatability_dataset_consolidation/workflow.py`
- Modify: `repeatability_dataset_consolidation/tests/test_completion.py`
- Create: `repeatability_dataset_consolidation/tests/test_finalization.py`

**Interfaces:**
- Consumes: Task 1 streamed inventories and existing checksum/copy receipts.
- Produces: `finalize_dataset(config: DatasetConfig) -> dict[str, object]`, `finalization_status(config: DatasetConfig) -> dict[str, object]`, and lightweight `validate_dataset_completion(...)`.

- [ ] **Step 1: Write a failing recovery test for the production-shaped partial state**

  Simulate matched checksum receipts, `dataset_manifest.status=complete`, a valid disposition, and missing final inventory/completion receipt. Assert finalization changes the manifest to `finalizing`, writes 215-equivalent miniature checkpoints, creates the inventory and only then writes completion.

- [ ] **Step 2: Run the recovery test and verify RED**

  Run: `python3 -m pytest repeatability_dataset_consolidation/tests/test_finalization.py::test_recovers_complete_manifest_without_completion_receipt -q`

- [ ] **Step 3: Implement receipt-only checksum binding validation**

  Add a helper that verifies the top-level checksum receipt, each item receipt path/hash/status/plan/key/destination, and extracts the recorded `destination_inventory_sha256` without rebuilding source or destination inventories.

- [ ] **Step 4: Implement one-item finalization and checkpoint receipts**

  Each item writes `<hash>.jsonl` plus `<hash>.json` containing the plan/key/destination bindings, upstream receipt SHA-256, stream SHA-256, inventory SHA-256, counts, bytes, metadata SHA-256 and status.

- [ ] **Step 5: Verify GREEN for basic recovery**

  Re-run the focused recovery test.

- [ ] **Step 6: Add failing interruption/resume and invalidation tests**

  Inject an interruption after N items; assert a rerun reuses unchanged checkpoints. Mutate one item, including restored mtime, and assert only that checkpoint is invalidated and a checksum mismatch fails closed.

- [ ] **Step 7: Implement resumable item selection and progress status**

  `finalization_status()` reports completed/expected items, current manifest state, inventory/receipt presence and last failure without mutating state.

- [ ] **Step 8: Add failing final-merge and transaction-order tests**

  Cover duplicate dataset-relative paths, unplanned destination files, missing exact scope, control-file inclusion, checkpoint exclusion, disposition binding, and absence of completion receipt before inventory success.

- [ ] **Step 9: Implement final inventory assembly and atomic completion**

  Reuse or regenerate valid source mapping and disposition, validate exact destination coverage/scope, merge item streams and control entries, install the final inventory, then promote the manifest and write the completion receipt.

- [ ] **Step 10: Replace expensive normal status validation**

  Make `validate_dataset_completion()` validate receipt bindings, final inventory JSON integrity and checkpoint metadata fingerprints. Preserve explicit deep checksum behavior under `checksum`; never treat metadata as recovery from a failed checksum.

- [ ] **Step 11: Run dataset completion/finalization tests**

  Run: `python3 -m pytest repeatability_dataset_consolidation/tests/test_finalization.py repeatability_dataset_consolidation/tests/test_completion.py -q`

- [ ] **Step 12: Commit Task 2**

  Commit message: `feat: resume repeatability dataset finalization`

### Task 3: Make Shared archive checksum verification resumable

**Files:**
- Modify: `repeatability_dataset_consolidation/archive.py`
- Modify: `repeatability_dataset_consolidation/tests/test_archive.py`

**Interfaces:**
- Consumes: Task 1 inventory streams.
- Produces: per-root archive checksum streams/receipts and metadata-based `validate_archive_completion(config)`.

- [ ] **Step 1: Write failing per-root interruption/resume tests**

  Interrupt after one miniature archive root and assert a rerun reuses that root while hashing the remaining roots.

- [ ] **Step 2: Run focused archive tests and verify RED**

  Run: `python3 -m pytest repeatability_dataset_consolidation/tests/test_archive.py -q`

- [ ] **Step 3: Implement per-root source/destination streams and receipts**

  Store checkpoints below `_archive/verification/checksum/items/<key>/`, compare source materialized and destination inventory hashes, and write the external completion receipt only after all four roots match.

- [ ] **Step 4: Add failing stale-root and receipt-tampering tests**

  Cover metadata-only source/destination changes, same-size changed content, changed checkpoint JSONL, and an existing nonmatching completion receipt.

- [ ] **Step 5: Implement resumable archive validation**

  Reuse only unchanged, receipt-bound root checkpoints. `validate_archive_completion()` performs receipt and metadata validation without multi-terabyte body re-hashing.

- [ ] **Step 6: Run all archive tests**

  Run: `python3 -m pytest repeatability_dataset_consolidation/tests/test_archive.py -q`

- [ ] **Step 7: Commit Task 3**

  Commit message: `feat: checkpoint Shared archive verification`

### Task 4: Add automatic retirement-readiness orchestration

**Files:**
- Create: `repeatability_dataset_consolidation/readiness.py`
- Modify: `repeatability_dataset_consolidation/retirement.py`
- Modify: `repeatability_dataset_consolidation/cli.py`
- Modify: `repeatability_dataset_consolidation/hpc/submit.sh`
- Create: `repeatability_dataset_consolidation/tests/test_readiness.py`
- Modify: `repeatability_dataset_consolidation/tests/test_cli.py`
- Modify: `repeatability_dataset_consolidation/tests/test_retirement.py`
- Modify: `repeatability_dataset_consolidation/tests/test_hpc_launcher.py`

**Interfaces:**
- Consumes: Tasks 2 and 3 completion/validation APIs and existing `plan_retirement()`.
- Produces: `prepare_retirement(config)`, `readiness_status(config)`, CLI modes `prepare-retirement` and `readiness-status`, and external `retirement_ready.json`.

- [ ] **Step 1: Write a failing end-to-end readiness test**

  Starting from a copied/checksum-matched miniature dataset, assert one call reaches deletion-ready state, writes the exact receipt hashes and command, and leaves every source present.

- [ ] **Step 2: Run the focused readiness test and verify RED**

  Run: `python3 -m pytest repeatability_dataset_consolidation/tests/test_readiness.py::test_prepare_retirement_stops_before_deletion -q`

- [ ] **Step 3: Implement the exclusive lock and stage orchestrator**

  Use `fcntl.flock(LOCK_EX | LOCK_NB)` and an external lock record. Execute finalization, dataset validation, archive preflight/copy/quick/checksum validation and retirement planning in the approved order. Persist stage/failure state outside the dataset.

- [ ] **Step 4: Add failing no-deletion and mismatch tests**

  Monkeypatch `subprocess.run`/launcher boundaries and assert no `sbatch`, `rm`, `retire-submit` or deletion-launcher call. Cover dataset mismatch, archive mismatch, active Slurm reference and concurrent lock.

- [ ] **Step 5: Implement the readiness receipt**

  Write `deletion_log_root/retirement_ready.json` atomically with dataset/archive/disposition/plan hashes, exact targets, scheduler snapshot hash, literal confirmed `retire-submit` command, `jobs_submitted: 0`, and `deletion_performed: false`. Refuse to overwrite a nonmatching receipt.

- [ ] **Step 6: Add failing read-only status tests**

  Snapshot filesystem state, call `readiness_status()`, and assert no path/content/mtime changes. Cover absent, running, failed, and ready states.

- [ ] **Step 7: Implement status and CLI/launcher modes**

  Add `prepare-retirement` and `readiness-status` to the parser and shell allowlist. Print explicit numerical scope and deletion boundary.

- [ ] **Step 8: Refactor retirement planning to use bound checkpoint validation**

  Preserve exact allowlist, archive requirements, disposition equality and live scheduler checks while avoiding redundant full-body hashes already represented by completion checkpoints. Metadata changes fail closed.

- [ ] **Step 9: Run readiness, CLI, retirement and launcher tests**

  Run: `python3 -m pytest repeatability_dataset_consolidation/tests/test_readiness.py repeatability_dataset_consolidation/tests/test_cli.py repeatability_dataset_consolidation/tests/test_retirement.py repeatability_dataset_consolidation/tests/test_hpc_launcher.py -q`

- [ ] **Step 10: Commit Task 4**

  Commit message: `feat: automate retirement readiness preparation`

### Task 5: Document, audit and verify the complete workflow

**Files:**
- Modify: `repeatability_dataset_consolidation/README.md`
- Modify: `docs/superpowers/specs/2026-11-12-resumable-retirement-readiness-design.md` only if implementation-compatible clarifications are required.

**Interfaces:**
- Documents: exact HPC pull/status/tmux/monitoring workflow and explicit deletion boundary.

- [ ] **Step 1: Update the operator runbook**

  Document production scope, current recovery state, `prepare-retirement`, `readiness-status`, progress receipt paths, safe rerun semantics, and the fact that the next step is explicit deletion.

- [ ] **Step 2: Run the complete package test suite**

  Run: `python3 -m pytest repeatability_dataset_consolidation/tests -q`

  Expected: all tests pass without warnings from product code.

- [ ] **Step 3: Run adjacent regression suites**

  Run: `python3 -m pytest simnibs_segmentation_determinism/tests simnibs_326_repeatability/tests CamCan_Experiment/charm_segmentation_batch -q`

- [ ] **Step 4: Run static validation**

  Run:

  ```bash
  bash -n repeatability_dataset_consolidation/hpc/submit.sh
  shellcheck repeatability_dataset_consolidation/hpc/submit.sh
  python3 -m compileall -q repeatability_dataset_consolidation
  git diff --check
  ```

- [ ] **Step 5: Audit the final diff against destructive-operation boundaries**

  Confirm that `prepare-retirement` contains no deletion submission, no source mutation and no scientific destination mutation; verify modules, paths, login-node guard and exact production scope remain unchanged.

- [ ] **Step 6: Commit Task 5**

  Commit message: `docs: add retirement readiness runbook`

- [ ] **Step 7: Request final code review and address findings**

  Review the whole branch for correctness, resumability, memory bounds, Python 3.9 compatibility, receipt integrity and destructive-operation isolation. Re-run affected tests after every correction.
