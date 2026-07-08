# Defacing Experiment Tutorial

This tutorial documents the `sub-CCMe` intact-vs-defaced repeatability
experiment.

## Goal

Compare SimNIBS/CHARM behavior when it receives:

- intact `T1w` and intact `T2w`
- defaced `T1w` and defaced `T2w`

for the same subject, with no custom segmentation supplied.

The scientific question is whether downstream simulation outputs differ when
SimNIBS can use the original face geometry versus when it must reconstruct the
missing face from anonymized inputs.

This workflow stages and runs:

- subject: `sub-CCMe`
- targets: `left-hippocampus`, `left-m1`
- conditions: `intact`, `defaced`
- repeats: `40` per target and condition
- total simulations: `160`

## What changed in the runner

The simulation runner now supports this experiment directly:

- it resolves subject inputs from either `.nii` or `.nii.gz`
- it accepts datasets with no custom segmentation file
- when no custom segmentation exists, it stays on the pure CHARM path and uses
  the CHARM-produced mesh directly

Do not place a manual segmentation into these staged datasets.

## Step 1: Stage the experiment locally

Run the staging script from the repository root:

```bash
python3 defacing_experiment/prepare_defacing_repeat_batch.py \
  --subject sub-CCMe \
  --intact-t1 /home/boyan/sandbox/Jake_Data/SimME/sub-CCMe/anat/sub-CCMe_T1w.nii.gz \
  --intact-t2 /home/boyan/sandbox/Jake_Data/SimME/sub-CCMe/anat/sub-CCMe_T2w.nii \
  --out-root /tmp/defacing_repeat_batch \
  --repeats 40
```

If needed, point the script at a specific `pydeface` binary:

```bash
python3 defacing_experiment/prepare_defacing_repeat_batch.py \
  --subject sub-CCMe \
  --intact-t1 /home/boyan/sandbox/Jake_Data/SimME/sub-CCMe/anat/sub-CCMe_T1w.nii.gz \
  --intact-t2 /home/boyan/sandbox/Jake_Data/SimME/sub-CCMe/anat/sub-CCMe_T2w.nii \
  --out-root /tmp/defacing_repeat_batch \
  --repeats 40 \
  --pydeface-bin /home/boyan/fsl/bin/pydeface
```

This workflow uses `pydeface` for the `T1w`, then resamples the saved defacing
mask into `T2w` space before masking the `T2w`. That is deliberate: for
`sub-CCMe`, the `T1w` and `T2w` are not on the same voxel grid, so a direct
`pydeface --applyto` call would not be reliable.

The script produces:

- generated defaced source files under `<out-root>/_generated_defaced/sub-CCMe/anat/`
- `Left_Hippocampus_Intact`
- `Left_Hippocampus_Defaced`
- `Left_M1_Intact`
- `Left_M1_Defaced`
- `<out-root>/experiment_manifest.tsv`

Each parent root contains:

- `Left_<Target>_Data_01` through `Left_<Target>_Data_40`
- `slurm/manifest.tsv`

Important: the defaced repeats keep the normal filenames
`sub-CCMe_T1w...` and `sub-CCMe_T2w...` inside each repeat directory. Only the
voxel content changes.

## Step 2: Upload to HPC

Upload the four staged parent roots to HPC:

- `Left_Hippocampus_Intact`
- `Left_Hippocampus_Defaced`
- `Left_M1_Intact`
- `Left_M1_Defaced`

The generated defaced source tree and `experiment_manifest.tsv` are useful for
local provenance, but the HPC runs only need the staged parent roots and their
embedded `slurm/manifest.tsv` files.

## Step 3: Submit the simulations on HPC

From the repository root on HPC, launch one parent root at a time with
`submit_defacing_repeat_batch.sh`.

Example setup:

```bash
cd /users/<user>/Repos/TI_Pipeline/SimNIBS/Scripts
BASE=/mnt/parscratch/users/<user>/defacing_repeat_batch
```

Left hippocampus, intact:

```bash
PARENT_ROOT="$BASE/Left_Hippocampus_Intact" \
MONTAGE_PRESET=left-hippocampus \
LOG_DIR="$BASE/logs/left_hippocampus_intact" \
bash CamCan_Experiment/HPC_scripts/submit_defacing_repeat_batch.sh
```

Left hippocampus, defaced:

```bash
PARENT_ROOT="$BASE/Left_Hippocampus_Defaced" \
MONTAGE_PRESET=left-hippocampus \
LOG_DIR="$BASE/logs/left_hippocampus_defaced" \
bash CamCan_Experiment/HPC_scripts/submit_defacing_repeat_batch.sh
```

Left M1, intact:

```bash
PARENT_ROOT="$BASE/Left_M1_Intact" \
MONTAGE_PRESET=left-m1 \
LOG_DIR="$BASE/logs/left_m1_intact" \
bash CamCan_Experiment/HPC_scripts/submit_defacing_repeat_batch.sh
```

Left M1, defaced:

```bash
PARENT_ROOT="$BASE/Left_M1_Defaced" \
MONTAGE_PRESET=left-m1 \
LOG_DIR="$BASE/logs/left_m1_defaced" \
bash CamCan_Experiment/HPC_scripts/submit_defacing_repeat_batch.sh
```

Useful overrides:

- `MAX_CONCURRENT_TASKS=4` to reduce simultaneous repeats
- `CPUS_PER_TASK=8`
- `MEMORY=32G`
- `TIME_LIMIT=08:00:00`
- `MESH_TIMEOUT_HOURS=4`

Example with a smaller concurrency cap:

```bash
PARENT_ROOT="$BASE/Left_Hippocampus_Defaced" \
MONTAGE_PRESET=left-hippocampus \
LOG_DIR="$BASE/logs/left_hippocampus_defaced" \
MAX_CONCURRENT_TASKS=4 \
bash CamCan_Experiment/HPC_scripts/submit_defacing_repeat_batch.sh
```

### Expanding an existing 10-repeat staging tree to the planned 40 repeats

If repeats `01` through `10` have already completed, regenerate or update the
staged tree with `--repeats 40` and upload the updated parent roots/manifests.
Then submit only the missing repeat rows by setting `START_TASK_OFFSET=10`.

The manifest task index is zero-based, so:

- `START_TASK_OFFSET=0` starts at repeat `01`
- `START_TASK_OFFSET=10` starts at repeat `11`

Example for left hippocampus intact:

```bash
PARENT_ROOT="$BASE/Left_Hippocampus_Intact" \
MONTAGE_PRESET=left-hippocampus \
LOG_DIR="$BASE/logs/left_hippocampus_intact_r11_r40" \
START_TASK_OFFSET=10 \
bash CamCan_Experiment/HPC_scripts/submit_defacing_repeat_batch.sh
```

Use the same `START_TASK_OFFSET=10` pattern for the other three arms. Do not
submit the full manifest from offset `0` unless you intentionally want to rerun
repeats `01` through `10`.

## Output layout on HPC

For each parent root, every repeat dataset follows the standard runner layout:

```text
<parent-root>/Left_<Target>_Data_01/
└── sub-CCMe/anat/
    ├── sub-CCMe_T1w.nii.gz
    ├── sub-CCMe_T2w.nii
    └── SimNIBS/
```

The final SimNIBS outputs are written under:

```text
<dataset-root>/sub-CCMe/anat/SimNIBS/
```

The array launcher validates each task after the runner exits and requeues
incomplete tasks until they succeed or the retry limit is reached.

## Step 4: Post-process completed repeats

The existing repeat-batch post-processing pipeline can be reused for each parent
root after the simulations finish.

Use `CamCan_Experiment/HPC_scripts/run_post_processing_batch.slurm`, but note
that it is configured through the variable block near the top of the file.
Before submitting it, update at least:

- `BATCH_ROOT`
- `BATCH_REPEATS`
- `PIPELINE_FASTSURFER_ROOT`
- `PIPELINE_MNI_BASELINE_ROOT`
- `PIPELINE_MNI_FIXED_ATLAS_PATH`

Target-specific baseline roots should match the parent root being analyzed:

- hippocampus runs: `.../MNI152-left-hippocampus`
- left M1 runs: `.../MNI152-left-m1`

Typical pattern:

1. copy `CamCan_Experiment/HPC_scripts/run_post_processing_batch.slurm`
2. set `BATCH_ROOT` to one staged parent root
3. set `PIPELINE_MNI_BASELINE_ROOT` to the matching target baseline
4. submit with `sbatch`

If you prefer to drive the batch post-processing directly from environment
variables inside an interactive allocation, the Python entrypoint is:

```bash
python CamCan_Experiment/post/run_post_processing_batch_env.py
```

That entrypoint reads `BATCH_ROOT`, `PIPELINE_MNI_BASELINE_ROOT`, and the other
`PIPELINE_*` variables from the shell environment.

## Sanity checks before launch

- each parent root has `40` repeat directories
- each parent root has `slurm/manifest.tsv`
- each repeat has both `sub-CCMe_T1w...` and `sub-CCMe_T2w...`
- no manual segmentation file is present in the repeat datasets
- `Left_Hippocampus_*` runs use `MONTAGE_PRESET=left-hippocampus`
- `Left_M1_*` runs use `MONTAGE_PRESET=left-m1`
