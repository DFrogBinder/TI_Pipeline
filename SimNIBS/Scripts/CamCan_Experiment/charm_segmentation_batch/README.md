# CHARM-only segmentation array for all available TI-dataset subjects

This workflow creates one fresh, unmodified CHARM tissue map per subject. It
does not retain CHARM's working directory, read or merge the manuscript
segmentation, create cortical surfaces, create a mesh, configure electrodes, or
run a SimNIBS simulation.

## CHARM flags

- `--registerT2`: required for CHARM to use the supplied T2 with the T1.
- `--initatlas`: CHARM's internal initialization of its probabilistic tissue
  atlas. The segmentation stage requires its registered template; no atlas
  deliverable is retained.
- `--segment`: creates `tissue_labeling_upsampled.nii.gz`.
- `--forceqform`: preserves the orientation handling used by the existing
  CamCan CHARM calls by replacing sform with qform inside temporary processing.
- `--surfaces` and `--mesh`: deliberately not passed.

CHARM's segmentation algorithm internally creates registration and normalized
image products. They exist only in task-local temporary storage and are deleted
after the tissue NIfTI has been copied and SHA-256 verified.

## HPC defaults

- Source dataset: `/mnt/parscratch/users/cop23bi/ti_dataset`
- Output root: `/mnt/parscratch/users/cop23bi/charm_segmentations`
- Repository: `/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment`
- Module: `SimNIBS/4.0.1-foss-2023a`
- Array: `0-(N-1)%50`, where `N` is discovered at submission time
- Per task: 8 CPUs, 32 GB RAM, 8-hour Slurm limit
- CHARM stages: the minimum valid T1+T2 segmentation chain; no surfaces or mesh

The settings are a candidate until confirmed on Stanage. The repository path,
module, partition, and per-task resources match the existing CamCan launchers.
The `%50` throttle, dynamic discovery under `ti_dataset`, and segmentation-only
CHARM stage selection are the intentional changes.

## Subject discovery

The submitter scans immediate `/mnt/parscratch/users/cop23bi/ti_dataset/sub-*`
directories in sorted order. A subject is runnable only when both inputs exist
as `.nii` or `.nii.gz`:

```text
<source>/<subject>/anat/<subject>_T1w.nii[.gz]
<source>/<subject>/anat/<subject>_T2w.nii[.gz]
```

Runnable IDs are written to `<output>/submission/subjects.txt`. Incomplete
`sub-*` directories are excluded from the array and recorded as `blocked` in
`<output>/submission/preflight.tsv`. No fixed cohort size is assumed.

## Launch on Stanage

After pushing/pulling this folder on the HPC:

```bash
cd /users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment
bash charm_segmentation_batch/submit_charm_segmentations.sh
```

The submitter discovers all runnable subjects, submits one dynamically sized
array, and submits a small collector job with an `afterany` dependency. The
collector therefore produces an audit report even if an array task fails.

The path defaults can be overridden explicitly:

```bash
export TI_CHARM_SOURCE_ROOT=/mnt/parscratch/users/cop23bi/ti_dataset
export TI_CHARM_OUTPUT_ROOT=/mnt/parscratch/users/cop23bi/charm_segmentations
bash charm_segmentation_batch/submit_charm_segmentations.sh
```

## Outputs

The only retained CHARM-generated scientific files are in:

```text
/mnt/parscratch/users/cop23bi/charm_segmentations/maps/
├── sub-CC110056_CHARM_tissue_labeling_upsampled.nii.gz
└── ... one map for every discovered runnable subject
```

Each file in `maps/` is copied byte-for-byte from CHARM's generated
`m2m_*/label_prep/tissue_labeling_upsampled.nii.gz`. The runner never opens or
resaves the NIfTI. SHA-256 is compared before and after the single copy.

No `m2m_*` working directory is retained. Non-scientific audit files consist of
one provenance JSON per subject under `metadata/`, discovery files under
`submission/`, collection manifests/checksums under `collection/`, and Slurm
logs under `logs/`.

Check or rebuild the collection with:

```bash
python3 charm_segmentation_batch/collect_charm_segmentations.py \
    --out-root /mnt/parscratch/users/cop23bi/charm_segmentations \
    --subjects-file /mnt/parscratch/users/cop23bi/charm_segmentations/submission/subjects.txt
```

A complete run reports `valid=N/N` and exits zero. An incomplete run exits 125
and identifies each missing/invalid runnable subject in
`collection/charm_segmentation_manifest.tsv`.

## Reruns

Completed subjects are hash-validated and skipped automatically. Failed or
partial subjects are regenerated only inside the dedicated CHARM output root;
the source TI dataset is never modified. To deliberately regenerate a single
subject, invoke `run_charm_segmentation.py` with `--force` from a SimNIBS-loaded
HPC shell.

## Install the 175 CHARM maps into all 40 repeat datasets

`install_charm_segmentations.py` installs the collected map for each subject as
the exact filename consumed by CHARM meshing:

```text
<ROI>_Data_<repeat>/<subject>/anat/m2m_<subject>/label_prep/tissue_labeling_upsampled.nii.gz
```

The four default ROI prefixes are `Left_Hippocampus`, `Left_M1`, `Right_DLPC`,
and `Right_Thalamus`; repeats default to 01 through 10. The ROI root is searched
recursively, so both a flat layout and an HPC layout with `*_Runs` parent
directories are supported.

The installer has no target-root default. The CHARM segmentation-generation
profile is known working, but this 7,000-file installation is a candidate until
the audit is run against the live HPC tree. Set the actual uploaded-map and ROI
roots explicitly:

```bash
cd /users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment
MAPS_ROOT=/mnt/parscratch/users/cop23bi/all-seg-maps
ROI_ROOT=/mnt/parscratch/users/cop23bi/ZIPs/Analised-Data
RUN_ID=charm-map-install-$(date +%Y%m%d-%H%M%S)
REPORT_DIR=/mnt/parscratch/users/cop23bi/charm_segmentation_install/$RUN_ID
BACKUP_ROOT=/mnt/parscratch/users/cop23bi/charm_segmentation_backups/$RUN_ID
python3 charm_segmentation_batch/install_charm_segmentations.py --maps-root "$MAPS_ROOT" --roi-root "$ROI_ROOT" --report-dir "$REPORT_DIR"
```

The first invocation is audit-only. It changes no segmentation, mesh, or
simulation file. A ready preflight must report all of the following:

```text
sources=175/175 datasets=40/40 targets=7000/7000 issues=0
```

Inspect the machine-readable evidence before applying:

```bash
sed -n '1,160p' "$REPORT_DIR/preflight_summary.json"
wc -l "$REPORT_DIR/sources.tsv" "$REPORT_DIR/targets.tsv" "$REPORT_DIR/issues.tsv"
```

Expected line counts, including headers, are 176, 7001, and 1. If and only if
the preflight is ready, run the same command with an explicit backup root and
`--apply`:

```bash
python3 charm_segmentation_batch/install_charm_segmentations.py --maps-root "$MAPS_ROOT" --roi-root "$ROI_ROOT" --report-dir "$REPORT_DIR" --backup-root "$BACKUP_ROOT" --apply
```

Each old map is backed up under the same relative path beneath `BACKUP_ROOT`.
On the same filesystem this uses a space-efficient hard link; otherwise it
falls back to a verified copy. Each CHARM map is copied to a temporary file,
SHA-256 checked, atomically moved onto the target, and checked again. The apply
manifest is flushed after every target, and rerunning with a new report and
backup directory skips targets that already have the source hash.

Keep the backup tree and `apply_manifest.tsv` until all new meshes and
simulations have passed validation. The installer deliberately does not remove
or regenerate the existing `.msh` files, so they remain stale immediately after
map replacement.

### Required remesh handoff

Do not use the normal meshing branch of `TI_runner_multi-core.py` after this
installation. That branch calls CHARM with `--forcerun` and, when the existing
manual/ROAST map is present, merges it back into the CHARM map. This would undo
the CHARM-only replacement.

Regenerate each mesh from the installed map with the equivalent of the
following command in the subject's `anat` directory:

```bash
charm <subject> --mesh
```

After remeshing has succeeded, run the simulations through the existing
mesh-reuse path (`--reuse-existing-mesh`). A separate array launcher should be
used for the 7,000 remesh operations so completion can be validated before the
simulation arrays are submitted.
