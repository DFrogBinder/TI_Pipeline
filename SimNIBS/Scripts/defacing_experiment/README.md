# Defacing Experiment Helpers

This directory contains the local preparation utilities for the `sub-CCMe`
intact-vs-defaced repeatability experiment.

## Experiment intent

The experiment compares two SimNIBS/CHARM input conditions for the same
subject:

- intact `T1w` + intact `T2w`
- defaced `T1w` + defaced `T2w`

No custom segmentation is supplied. Each repeat reruns CHARM from the provided
images so SimNIBS must either use the original face geometry or reconstruct the
missing face internally in the defaced condition.

The staged experiment covers:

- `left-hippocampus`
- `left-m1`
- `10` repeats per target and condition

## Scripts

### `deface_fsl_batch.py`

Standalone helper for face-only `T1w` defacing with `fsl_deface`. This is a
small utility when you only want defaced `T1w` outputs and do not need the full
repeat-batch staging flow.

Example:

```bash
python3 defacing_experiment/deface_fsl_batch.py \
  --t1 /path/to/sub-CCMe_T1w.nii.gz \
  --out-root /tmp/defaced_t1
```

### `prepare_defacing_repeat_batch.py`

Main entrypoint for this experiment. It:

1. defaces the intact `T1w` with `pydeface`
2. saves the `T1w` face-removal mask produced by `pydeface`
3. resamples that keep-mask into `T2w` space
4. applies the resampled keep-mask to the intact `T2w`
5. stages four repeat-batch parent roots:
   - `Left_Hippocampus_Intact`
   - `Left_Hippocampus_Defaced`
   - `Left_M1_Intact`
   - `Left_M1_Defaced`

Run it locally from the repository root:

```bash
python3 defacing_experiment/prepare_defacing_repeat_batch.py \
  --subject sub-CCMe \
  --intact-t1 /home/boyan/sandbox/Jake_Data/SimME/sub-CCMe/anat/sub-CCMe_T1w.nii.gz \
  --intact-t2 /home/boyan/sandbox/Jake_Data/SimME/sub-CCMe/anat/sub-CCMe_T2w.nii \
  --out-root /tmp/defacing_repeat_batch
```

If `pydeface` is not on `PATH`, pass it explicitly:

```bash
python3 defacing_experiment/prepare_defacing_repeat_batch.py \
  --subject sub-CCMe \
  --intact-t1 /home/boyan/sandbox/Jake_Data/SimME/sub-CCMe/anat/sub-CCMe_T1w.nii.gz \
  --intact-t2 /home/boyan/sandbox/Jake_Data/SimME/sub-CCMe/anat/sub-CCMe_T2w.nii \
  --out-root /tmp/defacing_repeat_batch \
  --pydeface-bin /home/boyan/fsl/bin/pydeface
```

## Output layout

The script writes one generated-defaced source tree plus the staged repeat
batches:

```text
<out-root>/
├── _generated_defaced/
│   └── sub-CCMe/anat/
│       ├── sub-CCMe_desc-deface_T1w.nii.gz
│       ├── sub-CCMe_desc-deface_T2w.nii
│       ├── sub-CCMe_desc-deface_mask_T1w.nii.gz
│       └── sub-CCMe_desc-deface_mask_T2w.nii
├── Left_Hippocampus_Intact/
├── Left_Hippocampus_Defaced/
├── Left_M1_Intact/
├── Left_M1_Defaced/
└── experiment_manifest.tsv
```

Each parent root contains:

- `Left_<Target>_Data_01` through `Left_<Target>_Data_10`
- `slurm/manifest.tsv`

Each repeat dataset contains:

```text
<parent-root>/Left_<Target>_Data_01/sub-CCMe/anat/
├── sub-CCMe_T1w.nii.gz
└── sub-CCMe_T2w.nii
```

Important: the defaced condition keeps the canonical input filenames
(`sub-CCMe_T1w...`, `sub-CCMe_T2w...`) inside each repeat directory even though
the voxel data is defaced. This keeps the simulation runner on the standard
subject-input discovery path.

## Simulation assumptions

- both modalities are provided to CHARM for every repeat
- no manual/custom segmentation is copied into the staged datasets
- the simulation runner now accepts either `.nii` or `.nii.gz` for both `T1w`
  and `T2w`
- when no custom segmentation exists, the runner uses the CHARM-produced mesh
  directly instead of trying to merge a manual segmentation
- `pydeface --applyto` is not used because the `T1w` and `T2w` live on different
  grids for `sub-CCMe`; the script resamples the saved `T1w` mask onto `T2w`
  instead

For the end-to-end HPC workflow, see
`CamCan_Experiment/docs/DEFACING_EXPERIMENT_TUTORIAL.md`.
