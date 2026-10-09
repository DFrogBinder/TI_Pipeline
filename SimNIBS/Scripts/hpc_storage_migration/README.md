# Stanage storage migration

This directory contains the guarded, resumable migration from:

- source: `/mnt/parscratch/users/cop23bi`
- destination: `/shared/boyan/Shared`

The reviewed inventory contains 27 top-level entries. Eighteen worker-visible
roots are protected: nine reusable core-data roots plus the nine active
repeatability-paper roots. The reviewed migration scope is therefore nine
completed/inactive entries. The script does not delete source data and does not
update repository paths.

## Safety model

- The live source inventory must exactly match `parscratch_inventory.txt`.
- Every excluded project must be an exact top-level manifest entry.
- Copying requires the destination path to be repeated exactly.
- Each entry is copied independently with resumable `rsync` semantics.
- File contents, directory structure, symlinks, hard links, and timestamps are
  preserved. Unix owner/group/mode metadata is not used for verification because
  `/shared` uses NFSv4/Windows-style ACLs rather than ordinary Unix permissions.
- `verify` performs a read-only size/mtime comparison.
- `checksum` performs a read-only full-content comparison.
- `audit` reports common neuroimaging, surface/mesh, DICOM, and archive counts
  for context. These formats are permitted in completed simulation and
  post-processing outputs; exclusion is determined by workflow dependency, not
  filename extension.
- Source cleanup is deliberately absent. It is a separate decision after
  checksum verification and repository-path cutover.

No writes should occur in the included source directories between the final
copy and checksum verification. Work may continue only in the excluded roots.

The active set is recorded in `active_repeatability_exclusions.txt`. Reusable
source MRI, segmentation, corrected-map, scaffold, atlas, MNI-input, and
defacing roots are recorded in `core_data_exclusions.txt`. Their reviewed union
is `repeatability_migration_exclusions.txt`, which is the exclusion file used by
the commands below. `migration_scope.tsv` records the decision and dependency
rationale for every one of the 27 top-level entries. The active set includes
the three SimNIBS 3.2.6 comparison roots added during the migration.

## Stanage accessibility constraint

Sheffield documents `/shared` research storage as available on Stanage login
nodes only, not worker nodes. The archive migration therefore runs on a login
node, but SimNIBS/Slurm launchers must continue using worker-visible storage such
as `/mnt/parscratch`. Do not replace live launcher defaults with `/shared` paths
unless Research IT confirms worker-node access for this allocation.

## Run sequence on Stanage

```bash
cd /users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/hpc_storage_migration
```

```bash
bash migrate_parscratch_to_shared.sh audit --exclude-file repeatability_migration_exclusions.txt
```

Review the 27-row audit receipt under
`/shared/boyan/Shared/.parscratch_migration_receipts/`, including source sizes,
file counts, imaging/mesh/archive counts, free space, and any top-level symbolic
links. Imaging outputs are expected in migratable completed-study roots.

```bash
bash migrate_parscratch_to_shared.sh copy --exclude-file repeatability_migration_exclusions.txt --confirm-destination /shared/boyan/Shared
```

The copy is resumable: rerun the same command after diagnosing any interruption.

```bash
bash migrate_parscratch_to_shared.sh verify --exclude-file repeatability_migration_exclusions.txt
```

```bash
bash migrate_parscratch_to_shared.sh checksum --exclude-file repeatability_migration_exclusions.txt
```

Do not remove the `parscratch` copies until all nine entries report `match` in the
checksum receipt and the updated repository paths have been validated on the
shared partition.

### Atlas-workspace cleanup condition

`CamCan_Corrected_v4_Atlases` contains the 87 FreeSurfer reconstruction work
directories used to complete the final atlas set. It is migratable because the
132 flattened Destrieux atlases consumed by post-processing are protected under
`/mnt/parscratch/users/cop23bi/ZIPs/atlases`. Before deleting the original atlas
workspace, require all of the following:

1. the shared archive passes full checksum verification;
2. the final-132 atlas preflight reports 132 existing flat atlases, zero missing
   tasks, and zero blocked subjects;
3. no atlas-repair or post-processing job is active;
4. deletion is separately and explicitly approved.
