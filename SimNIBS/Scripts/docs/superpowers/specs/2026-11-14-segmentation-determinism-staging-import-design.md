# Segmentation Determinism Staging and Import Design

## Intent

Allow the approved one-subject segmentation-determinism experiment to run while
the combined repeatability dataset is still finalizing, without reading from or
writing to that active dataset. After both workflows are complete, install the
whole experiment package into the combined dataset as a separately receipted
extension for later backup.

## Fixed scientific scope

- Subject: `sub-CC110174`, selected with seed `20261106` from the code-owned
  final replacement cohort.
- Arms: SimNIBS 3.2.6 headreco segmentation and SimNIBS 4.0.1 CHARM
  segmentation.
- Repeats: 10 per arm; 20 scientific segmentation tasks and 23 scheduler tasks.
- No surface generation, tetrahedral meshing, electrode placement, or FEM.
- Preserve the current modules, Sheffield resources, concurrency, and bounded
  exact-array retry behavior.

## Paths and ownership

Experiment execution reads the authoritative corrected T1w/T2w pair from
`/mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_corrected` only
during initialization. Initialization atomically copies and hashes those inputs
into `/mnt/parscratch/users/cop23bi/segmentation_determinism_staging_v1`; all
jobs subsequently read and write only below that staging root.

The eventual destination is
`/mnt/parscratch/users/cop23bi/final_132_repeatability_combined_dataset_v1/extensions/segmentation_determinism_v1`.
No experiment command may write there. Only the explicit extension installer
may do so, and only after validating both the experiment terminal receipt and
the combined dataset terminal receipt.

## Workflow phases

### Staging experiment

`scope`, `preflight`, `init`, `run`, `status`, and `analyze` operate entirely on
the dedicated staging root. Preflight does not require the combined dataset to
be complete. Initialization records the code-owned cohort, selection seed,
canonical input hashes, and staged input hashes.

### Extension installation

`install-preflight` is read-only. It validates the staging terminal receipt,
the combined dataset completion/manifest binding, the exact destination, and
any existing importer ownership state.

`install-extension` requires the literal combined dataset root as a
confirmation argument. It uses resumable `rsync` with symlink materialization,
then performs a checksum dry-run with deletion detection. It writes a terminal
import receipt only when the comparison is empty. It never deletes or modifies
the staging package.

An interrupted owned destination may be resumed. A foreign or completed
destination is rejected. The importer does not update the original combined
dataset completion receipt or claim the extension was part of its original
inventory; the extension has its own receipt.

## Safety and provenance

- Staging and combined-dataset roots must be disjoint and exact.
- Source images and staging outputs remain protected from collision.
- The running consolidation/finalization workflow is neither queried nor
  modified by staging execution.
- Import remains unavailable until `_consolidation/complete.json` exists and
  binds the finalized dataset manifest by SHA-256.
- No Shared copy and no retirement/deletion action are part of this change.

## Considered approaches

1. **Dedicated staging plus guarded import (selected):** runs now without
   mutating the active dataset and retains a verifiable later migration path.
2. **Write directly into the extension now:** rejected because finalization is
   hashing the dataset and any new files could invalidate its inventory.
3. **Wait for consolidation:** safe but unnecessarily delays the experiment.

## Success criteria

- Existing scientific and scheduler scope is unchanged.
- Staging preflight succeeds without a combined dataset completion receipt.
- Every execution artifact is below the staging root.
- Installation fails closed before either terminal receipt exists.
- A successful installation is checksum-identical apart from its importer
  metadata and leaves the staging package unchanged.
