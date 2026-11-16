# Covered-source retirement design

## Goal

Free parscratch space now by retiring only the ten source roots whose files
were dispositioned into the completed combined dataset or explicitly marked
for intentional retirement. Preserve the four legacy-v3 archive-source roots
until their separate Shared archive succeeds.

## Context

The combined dataset has a checksum-bound completion receipt and an exhaustive
per-file retirement disposition. The existing full retirement mode intentionally
requires all fourteen retirement roots and a completed four-root Shared archive.
The Shared archive currently fails because its destination filesystem rejects
rsync permission changes on temporary files. That failure does not invalidate
the completed combined dataset, but it correctly prevents the existing
fourteen-root retirement path.

## Selected approach

Add a separate, opt-in `covered-retire-plan` / `covered-retire-submit` workflow.
Its canonical allowlist is exactly `retirement_allowlist(config)` minus every key
in `config.archive_sources`, preserving insertion order. For the production
configuration this produces ten targets.

The covered plan reuses the completed-dataset gates, live source metadata
binding, exhaustive disposition binding, target identity checks, Slurm queue
inspection, and hardened one-directory-per-job deletion launcher. It does not
require an archive receipt because no selected target may contain a
`shared_archive` disposition. It must reject a selected target unless its
summary record exists, has integer counters, and records zero
`shared_archive_files` and zero `shared_archive_bytes`.

Covered and full requests use distinct immutable request and terminal receipt
paths. The request binds the retirement scope, canonical target order, dataset
completion SHA-256, disposition SHA-256 and summary, source device/inode
identities, allowed root, and deletion plan SHA-256. Resume and status derive
the expected target scope from the request's explicit workflow rather than
accepting arbitrary subsets.

## Operator interface

- `covered-retire-plan` is read-only and prints the ten exact targets and hashes.
- `covered-retire-submit` requires the exact dataset root, dataset receipt hash,
  disposition hash, `--confirm-covered-retirement`, and no archive hash.
- `covered-retire-status` proves completion from terminal worker receipts plus
  independent target absence.
- Existing `retire-plan`, `retire-submit`, and `retire-status` retain their
  current fourteen-root, archive-gated behavior.

The wrapper remains login-node-only. Plan and status submit no Slurm jobs.
Submission delegates one guarded deletion job for each exact root; it never
calls `rm` itself.

## Safety boundaries

- The four keys in `config.archive_sources` can never occur in a covered request.
- Corrected input roots, v4 scaffolds, the combined dataset, and segmentation
  staging are not retirement targets.
- An existing request for one scope cannot be resumed or overwritten by the
  other scope.
- A target addition, removal, order change, metadata change, receipt change,
  disposition change, inode change, or active Slurm reference fails closed.
- Submission remains a separate explicitly confirmed action after inspection
  of the read-only plan.

## Rejected alternatives

Manual `rm` or direct launcher calls were rejected because they would bypass
the checksum, disposition, identity, scheduler, resume, and terminal-receipt
gates. Weakening the existing fourteen-root mode was rejected because its four
archive roots genuinely require the Shared copy. Deleting all fourteen now was
rejected because the archive completion receipt does not exist.
