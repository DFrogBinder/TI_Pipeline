# Final-132 combined repeatability dataset workflow

This login-node workflow creates the analysis-ready physical dataset at:

```text
/mnt/parscratch/users/cop23bi/final_132_repeatability_combined_dataset_v1
```

It consolidates ten subjects (`sub-CC510415` replaces `sub-CC520083`), two
SimNIBS versions, two targets, two conditions, and 40 repeats: 80 scientific
cells and 3,200 simulations. It also preserves the ten selected v3 harmonized
head models, ten selected v4 corrected scaffolds, the paper-analysis tree, and
the nested repeatability dataset.

The workflow does not launch simulations. Copy, verification, and Shared
archival run on the login node. Retirement is a separate, receipt-gated Slurm
maintenance operation and is never triggered automatically by copying.

## Start in tmux

The physical copy and checksum pass may run for a long time. On the Stanage
login node:

```bash
cd ~/Repos/TI_Pipeline/SimNIBS/Scripts
tmux new -s repeatability-dataset

DATASET=repeatability_dataset_consolidation/hpc/submit.sh
OUTPUT_ROOT=/mnt/parscratch/users/cop23bi/final_132_repeatability_combined_dataset_v1
ARCHIVE_ROOT=/shared/boyan/Shared/retired_repeatability_v3_sources_v1
```

Detach with `Ctrl-b d`; reattach with
`tmux attach -t repeatability-dataset`.

## 1. Audit, prepare, and copy

`preflight` is completely read-only. `prepare` creates only the owned skeleton
and immutable plan. `copy` requires the exact destination and never deletes a
source or an extra destination file. Both `preflight` and the first `prepare`
measure the logical bytes that will be materialized and fail before creating
the dataset root when parscratch free space is insufficient. A resume reports
capacity but does not apply the initial whole-copy gate again.

```bash
bash "$DATASET" preflight
bash "$DATASET" prepare
bash "$DATASET" copy --confirm-output-root="$OUTPUT_ROOT"
bash "$DATASET" status
```

Interrupted copies are resumable by running the same `copy` command again.

## 2. Verify and complete the dataset

The quick comparison uses size and modification time. The checksum command
performs full-content comparison, builds the sorted per-file SHA-256 inventory,
promotes `dataset_manifest.json` to `complete`, and writes
`_consolidation/complete.json` only when every gate passes. `status` then
revalidates the live destination against those bindings; altered content is
reported as `invalid`, not `complete`.

```bash
bash "$DATASET" verify
bash "$DATASET" checksum
bash "$DATASET" status
```

Do not proceed to retirement unless `status` reports `complete: true`.

## 3. Archive obsolete v3 references to Shared

Shared is login-node-only and must never replace parscratch in a worker-job
configuration. This is a self-contained whole-tree logical archive. Hard links
are preserved, while every symlink whose target stays within the exact four
archive roots is materialized as a regular file; broken or external links fail
preflight. `archive-preflight` and the first `archive-copy` also fail before
creating the Shared root when free space is below the materialized byte total.

```bash
bash "$DATASET" archive-preflight
bash "$DATASET" archive-copy --confirm-archive-root="$ARCHIVE_ROOT"
bash "$DATASET" archive-verify
bash "$DATASET" archive-checksum
```

The external archive completion receipt is:

```text
/shared/boyan/Shared/.repeatability_retirement_receipts/legacy_v3_archive_v1.complete.json
```

## 4. Plan retirement without deleting

The default is an exhaustive dry-run plan for the complete code-owned
allowlist:

```bash
bash "$DATASET" retire-plan
```

Retirement is intentionally all-at-once. Partial allowlist requests are
rejected because deleting one source subset would make later live source
revalidation ambiguous.

The global `CamCan_Corrected_v4_Scaffolds` root and all primary MRI, atlas,
CHARM, corrected-label, MNI, and `ti_dataset` roots are not retirement targets.

## 5. Explicitly submit guarded deletion

Copy the exact receipt hashes printed by `retire-plan`. Submission requires all
confirmations and still re-runs every gate. The example deliberately uses
placeholders:

```bash
DATASET_RECEIPT_SHA256=<64-lowercase-hex-from-retire-plan>
ARCHIVE_RECEIPT_SHA256=<64-lowercase-hex-from-retire-plan>
RETIREMENT_DISPOSITION_SHA256=<64-lowercase-hex-from-retire-plan>

bash "$DATASET" retire-submit \
    --confirm-output-root="$OUTPUT_ROOT" \
    --dataset-receipt-sha256="$DATASET_RECEIPT_SHA256" \
    --archive-receipt-sha256="$ARCHIVE_RECEIPT_SHA256" \
    --retirement-disposition-sha256="$RETIREMENT_DISPOSITION_SHA256" \
    --confirm-retirement
```

The disposition is an exhaustive per-file record for every allowlisted source
root. Each entry is bound as a canonical dataset copy, a Shared archive copy,
or an explicitly intentional retirement. Any source-file addition, removal,
or content change after dataset completion invalidates retirement planning.

If launcher submission is interrupted after only some deletion jobs are
recorded, run the identical `retire-submit` command again. It resumes only the
missing jobs from the immutable external request receipt; already recorded
jobs are never submitted twice. Resume rechecks the live Slurm queue. A job
with a valid terminal `failed` receipt is submitted as a new attempt and the
failed attempt remains recorded in that job's retry history.

Deletion submission is not deletion completion. Check until every external
worker receipt is terminal and every target is independently absent:

```bash
bash "$DATASET" retire-status
```

Only `retire-submit` can initiate deletion. No copy, verify, checksum, archive,
status, or plan mode calls `rm` or submits simulation work.
