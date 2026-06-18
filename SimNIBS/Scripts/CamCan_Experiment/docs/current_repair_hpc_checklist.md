# CamCan Current Repair HPC Checklist

This checklist runs the current-correction workflow for the four CamCan ROIs:

- `left-m1`
- `right-thalamus`
- `right-dlpc`
- `left-hippocampus`

The correction workflow is run independently per ROI. Use separate output roots
for `scaled`, `pair2-rerun`, and `compare` outputs.

Important: `right-thalamus` has equal pair-1 and pair-2 currents in the target
montage table, so it is not affected by the pair-2 current bug. By default the
repair code skips it. Only run the optional bookkeeping copy if you want a
matching repaired-root folder for downstream consistency.

## Current Status

Last updated: 2026-06-13.

### Repeatability

- `scaled`: complete and validated against
  `$BASE/repeatability_scaled`.
- `pair2-rerun`: submitted/completed before CamCan work resumed. Pair2-rerun
  validation and `_analysis` generation are active next steps.
- `compare`: pending until the pair2-rerun root is validated and analyzed.
  - First direct compare attempt found at least one mesh shape mismatch between
    scaled and pair2-rerun pair-2 fields. The comparison code has been updated
    locally to record shape mismatches in the summary instead of aborting.
- Repaired `_analysis` generation: complete for `$BASE/repeatability_scaled`.
- Median-mesh reselection: complete locally from
  `repeatability_scaled/_analysis`.
  - Output CSV:
    `repeatability_scaled/_analysis/median_mesh_selection/median_representative_remesh_repeats.csv`.
  - Selected subjects: `10/10`.
  - Selection metric: `median_roi` for all subjects.

### CamCan Left-Hippocampus

- `scaled`: complete.
  - Expected final count: `1750`.
  - Confirmed `TI.msh` count: `1750`.
- `pair2-rerun`: complete.
  - Use manifest and output counts to confirm final status:
    `current_repair_manifest.json`, `anat/SimNIBS/Output/*/TI.msh`, and
    `anat/SimNIBS/ti_brain_only.nii.gz` should each count to `1750`.
- `compare`: pending.

### CamCan Left-M1

- `scaled`: complete.
  - Repeats complete: `Left_M1_Data_01` through `Left_M1_Data_10`.
- `pair2-rerun`: next stage.
- `compare`: pending.

### CamCan Right-DLPC

- `scaled`: pending.
- `pair2-rerun`: pending.
- `compare`: pending.

### CamCan Right-Thalamus

- Unaffected by the current bug because pair 1 and pair 2 use equal currents in
  the montage table.
- Optional bookkeeping copy: pending/not required.

## 0. Start Here

On Stanage:

```bash
cd /users/cop23bi/Repos/TI_Pipeline

export BASE=/mnt/parscratch/users/cop23bi/current-repair
export CAMCAN_SCRIPT=SimNIBS/Scripts/CamCan_Experiment/HPC_scripts/submit_current_repair_jobArray.sh
```

The submit wrapper automatically splits large task sets into multiple Slurm
array chunks. For example, a 1750-task ROI is submitted as more than one array
job so it does not exceed Stanage's maximum Slurm array index. Use
`MAX_ARRAY_TASKS=500` or another positive integer on the submit command if you
need smaller chunks.

Set the four original CamCan batch roots. Replace these with the actual roots on
Stanage if the names differ:

```bash
export ORIG_left_m1=/mnt/parscratch/users/cop23bi/ZIPs/Analised-Data/Left_M1_Runs
export ORIG_right_thalamus=/mnt/parscratch/users/cop23bi/ZIPs/Analised-Data/Right_Thalamus_Runs
export ORIG_right_dlpc=/mnt/parscratch/users/cop23bi/ZIPs/Analised-Data/Right_DLPC_Runs
export ORIG_left_hippocampus=/mnt/parscratch/users/cop23bi/ZIPs/Analised-Data/Left_Hippocampus_Runs
```

Check each root contains repeat datasets:

```bash
ls "$ORIG_left_m1"
ls "$ORIG_right_thalamus"
ls "$ORIG_right_dlpc"
ls "$ORIG_left_hippocampus"
```

Each root should contain directories like `*_Data_01`, `*_Data_02`, etc.

## 1. Submit Scaled Repairs

Submit these three affected ROIs. They can run independently.

### left-m1

```bash
ORIGINAL_ROOT="$ORIG_left_m1" \
OUTPUT_ROOT="$BASE/camcan_left-m1_scaled" \
MODE=scaled \
MONTAGE_PRESET=left-m1 \
LOG_DIR="$BASE/logs/camcan_left-m1_scaled" \
bash "$CAMCAN_SCRIPT"
```

### right-dlpc

```bash
ORIGINAL_ROOT="$ORIG_right_dlpc" \
OUTPUT_ROOT="$BASE/camcan_right-dlpc_scaled" \
MODE=scaled \
MONTAGE_PRESET=right-dlpc \
LOG_DIR="$BASE/logs/camcan_right-dlpc_scaled" \
bash "$CAMCAN_SCRIPT"
```

### left-hippocampus

```bash
ORIGINAL_ROOT="$ORIG_left_hippocampus" \
OUTPUT_ROOT="$BASE/camcan_left-hippocampus_scaled" \
MODE=scaled \
MONTAGE_PRESET=left-hippocampus \
LOG_DIR="$BASE/logs/camcan_left-hippocampus_scaled" \
bash "$CAMCAN_SCRIPT"
```

## 2. Submit Pair2-Rerun Repairs

Submit these after, or alongside, the scaled jobs. Each ROI is independent.

### left-m1

```bash
ORIGINAL_ROOT="$ORIG_left_m1" \
OUTPUT_ROOT="$BASE/camcan_left-m1_pair2_rerun" \
MODE=pair2-rerun \
MONTAGE_PRESET=left-m1 \
LOG_DIR="$BASE/logs/camcan_left-m1_pair2_rerun" \
bash "$CAMCAN_SCRIPT"
```

### right-dlpc

```bash
ORIGINAL_ROOT="$ORIG_right_dlpc" \
OUTPUT_ROOT="$BASE/camcan_right-dlpc_pair2_rerun" \
MODE=pair2-rerun \
MONTAGE_PRESET=right-dlpc \
LOG_DIR="$BASE/logs/camcan_right-dlpc_pair2_rerun" \
bash "$CAMCAN_SCRIPT"
```

### left-hippocampus

```bash
ORIGINAL_ROOT="$ORIG_left_hippocampus" \
OUTPUT_ROOT="$BASE/camcan_left-hippocampus_pair2_rerun" \
MODE=pair2-rerun \
MONTAGE_PRESET=left-hippocampus \
LOG_DIR="$BASE/logs/camcan_left-hippocampus_pair2_rerun" \
bash "$CAMCAN_SCRIPT"
```

## 3. Optional Right-Thalamus Bookkeeping Copy

Skip this unless you want a matching root under `current-repair` for
right-thalamus. This ROI is not affected by the current bug.

```bash
ORIGINAL_ROOT="$ORIG_right_thalamus" \
OUTPUT_ROOT="$BASE/camcan_right-thalamus_scaled" \
MODE=scaled \
MONTAGE_PRESET=right-thalamus \
INCLUDE_UNAFFECTED=1 \
LOG_DIR="$BASE/logs/camcan_right-thalamus_scaled" \
bash "$CAMCAN_SCRIPT"
```

Do not run `pair2-rerun` for right-thalamus unless you explicitly want an
expensive unaffected-control rerun.

## 4. Monitor Submitted Jobs

After each submission, note the Slurm job id.

Check active jobs:

```bash
squeue -u cop23bi
```

Check completed jobs:

```bash
sacct -j JOBID --format=JobID,JobName%35,State,ExitCode,Elapsed
```

Inspect logs for one ROI:

```bash
ls -lh "$BASE/logs/camcan_left-m1_scaled"
tail -n 80 "$BASE/logs/camcan_left-m1_scaled"/*.log
```

Find failures across all CamCan current-repair logs:

```bash
grep -R "Traceback\\|ERROR\\|FileNotFoundError" "$BASE/logs"/camcan_* -n
```

## 5. Submit Comparisons

Only run comparison after both `scaled` and `pair2-rerun` jobs for that ROI have
completed successfully.

### left-m1

```bash
OUTPUT_ROOT="$BASE/camcan_left-m1_compare" \
MODE=compare \
SCALED_ROOT="$BASE/camcan_left-m1_scaled" \
PAIR2_RERUN_ROOT="$BASE/camcan_left-m1_pair2_rerun" \
MONTAGE_PRESET=left-m1 \
LOG_DIR="$BASE/logs/camcan_left-m1_compare" \
bash "$CAMCAN_SCRIPT"
```

### right-dlpc

```bash
OUTPUT_ROOT="$BASE/camcan_right-dlpc_compare" \
MODE=compare \
SCALED_ROOT="$BASE/camcan_right-dlpc_scaled" \
PAIR2_RERUN_ROOT="$BASE/camcan_right-dlpc_pair2_rerun" \
MONTAGE_PRESET=right-dlpc \
LOG_DIR="$BASE/logs/camcan_right-dlpc_compare" \
bash "$CAMCAN_SCRIPT"
```

### left-hippocampus

```bash
OUTPUT_ROOT="$BASE/camcan_left-hippocampus_compare" \
MODE=compare \
SCALED_ROOT="$BASE/camcan_left-hippocampus_scaled" \
PAIR2_RERUN_ROOT="$BASE/camcan_left-hippocampus_pair2_rerun" \
MONTAGE_PRESET=left-hippocampus \
LOG_DIR="$BASE/logs/camcan_left-hippocampus_compare" \
bash "$CAMCAN_SCRIPT"
```

## 6. Final Analysis Inputs

Use these as the repaired analysis roots for the affected ROIs:

```bash
$BASE/camcan_left-m1_scaled
$BASE/camcan_right-dlpc_scaled
$BASE/camcan_left-hippocampus_scaled
```

For right-thalamus, use the original root because it is unaffected:

```bash
$ORIG_right_thalamus
```

If you created the optional bookkeeping copy, use this instead:

```bash
$BASE/camcan_right-thalamus_scaled
```

## 7. Notes

- Original roots are never modified.
- `scaled` and `pair2-rerun` roots are independent corrected datasets.
- `compare` only compares `scaled` vs `pair2-rerun` within the same ROI.
- Do not compare CamCan ROIs against each other in the current-repair validation.
- If a command returns zero tasks, check that `ORIGINAL_ROOT` points to the
  batch root containing that ROI's `*_Data_##` directories.
