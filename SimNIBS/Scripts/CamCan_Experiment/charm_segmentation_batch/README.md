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
