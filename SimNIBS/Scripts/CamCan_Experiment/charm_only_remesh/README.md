# CHARM-only remesh, simulation, and post-processing campaign

This workflow treats the successful 7,000-row installation manifest as the
authority for every dataset copy. It never calls CHARM segmentation. The only
meshing command it permits is:

```text
charm <subject> --mesh
```

The workflow fails closed if a ROAST/custom map remains in a live `anat`
directory, if an installed CHARM label differs from the installation manifest,
or if the original source label is unavailable. Obsolete ROAST maps and meshes
are deleted from the HPC because their recovery copies exist outside the HPC.
Generated simulation/post outputs remain separately archived.

## 1. Set campaign paths

Run these commands from the HPC checkout at
`/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts`:

```bash
ROI_ROOT=/mnt/parscratch/users/cop23bi/ZIPs/Analised-Data
INSTALL_REPORT=/mnt/parscratch/users/cop23bi/charm_segmentation_install/charm-map-install-20260715-203145
INSTALL_MANIFEST="$INSTALL_REPORT/apply_manifest.tsv"
RUN_ID=charm-only-remesh-$(date +%Y%m%d-%H%M%S)
CAMPAIGN_ROOT=/mnt/parscratch/users/cop23bi/charm_only_campaigns/$RUN_ID
REMESH_RESULTS="$CAMPAIGN_ROOT/remesh_results"
mkdir -p "$CAMPAIGN_ROOT" "$REMESH_RESULTS"
```

Keep later simulation-output archives outside `ROI_ROOT`.

## 2. Remove all live ROAST/custom maps

Audit first; this changes nothing:

```bash
python3 CamCan_Experiment/charm_only_remesh/workflow.py remove-roast \
  --install-manifest "$INSTALL_MANIFEST" \
  --roi-root "$ROI_ROOT" \
  --report "$CAMPAIGN_ROOT/roast_removal_audit.tsv"
```

Review the TSV and confirm the off-HPC backup is available. Then delete only
the exact audited ROAST/custom paths:

```bash
python3 CamCan_Experiment/charm_only_remesh/workflow.py remove-roast \
  --install-manifest "$INSTALL_MANIFEST" \
  --roi-root "$ROI_ROOT" \
  --report "$CAMPAIGN_ROOT/roast_removal_apply.tsv" \
  --apply-delete \
  --confirm-external-backup
```

The apply command deletes only
`<subject>_T1w_ras_1mm_T1andT2_masks.nii[.gz]` files discovered through the
successful 7,000-row installation manifest. The report records their former
paths, sizes, and hashes.

### Remove the installer-created ROAST backup tree

The CHARM-map installer also created 7,000 HPC-side copies of the replaced
ROAST labels. Audit that exact manifest-backed tree first:

```bash
INSTALL_BACKUP_ROOT=/mnt/parscratch/users/cop23bi/charm_segmentation_backups/charm-map-install-20260715-203145

python3 CamCan_Experiment/charm_only_remesh/workflow.py remove-install-backups \
  --install-manifest "$INSTALL_MANIFEST" \
  --backup-root "$INSTALL_BACKUP_ROOT" \
  --report "$CAMPAIGN_ROOT/install_backup_removal_audit.tsv"
```

Proceed only if the summary reports `status=audit`, `found=7000`,
`expected=7000`, and `issues=0`. Then delete the exact verified files:

```bash
python3 CamCan_Experiment/charm_only_remesh/workflow.py remove-install-backups \
  --install-manifest "$INSTALL_MANIFEST" \
  --backup-root "$INSTALL_BACKUP_ROOT" \
  --report "$CAMPAIGN_ROOT/install_backup_removal_apply.tsv" \
  --apply-delete \
  --confirm-external-backup
```

This command accepts only backup paths recorded in `apply_manifest.tsv` that
are inside the declared backup root and whose hashes still match
`before_sha256`. It removes the empty backup tree afterward but does not touch
any other campaign backup directory.

## 3. Build the immutable remesh manifest

```bash
python3 CamCan_Experiment/charm_only_remesh/workflow.py preflight \
  --install-manifest "$INSTALL_MANIFEST" \
  --roi-root "$ROI_ROOT" \
  --manifest "$CAMPAIGN_ROOT/remesh_manifest.tsv" \
  --summary "$CAMPAIGN_ROOT/remesh_preflight.json"
```

Do not submit unless the summary reports `status=ready`, `ready=7000`, and
`blocked=0`.

## 4. One-task smoke test

The resource profile is inherited from the successful CHARM segmentation run:
SimNIBS 4.0.1, `sheffield`, 8 CPUs, 32 GB, and 8 hours. Concurrency 8 is a
candidate campaign setting, so start with one task:

```bash
MANIFEST="$CAMPAIGN_ROOT/remesh_manifest.tsv" \
RESULT_DIR="$REMESH_RESULTS" \
LOG_DIR="$CAMPAIGN_ROOT/remesh_logs" \
MAX_CONCURRENT_TASKS=1 \
MAX_ARRAY_TASKS=1 \
MAX_SUBMITTED_CHUNKS=1 \
bash CamCan_Experiment/HPC_scripts/submit_charm_only_remesh.sh
```

After it finishes:

```bash
python3 CamCan_Experiment/charm_only_remesh/workflow.py validate \
  --manifest "$CAMPAIGN_ROOT/remesh_manifest.tsv" \
  --result-dir "$REMESH_RESULTS" \
  --summary "$CAMPAIGN_ROOT/remesh_smoke_validation.tsv" \
  --task-index 0
```

The result JSON records the unchanged CHARM-label hash, the deleted old-mesh
path/hash/size, and the new mesh hash. No obsolete mesh backup is created on
the HPC. A failed task leaves the old mesh absent and is safe to retry.

### End-to-end smoke simulation and post-processing

Before the full 7,000-mesh submission, carry task 0 through simulation and
subject post-processing. Read its identity and select the matching preset:

```bash
SMOKE_DATASET=$(awk -F '\t' 'NR==2 {print $2}' "$CAMPAIGN_ROOT/remesh_manifest.tsv")
SMOKE_ROI=$(awk -F '\t' 'NR==2 {print $3}' "$CAMPAIGN_ROOT/remesh_manifest.tsv")
SMOKE_REPEAT=$(awk -F '\t' 'NR==2 {print $4}' "$CAMPAIGN_ROOT/remesh_manifest.tsv")
SMOKE_SUBJECT=$(awk -F '\t' 'NR==2 {print $6}' "$CAMPAIGN_ROOT/remesh_manifest.tsv")
SMOKE_ROI_PARENT="$ROI_ROOT/${SMOKE_ROI}_Runs"
case "$SMOKE_ROI" in
  Left_M1) SMOKE_PRESET=left-m1 ;;
  Left_Hippocampus) SMOKE_PRESET=left-hippocampus ;;
  Right_DLPC) SMOKE_PRESET=right-dlpfc ;;
  Right_Thalamus) SMOKE_PRESET=right-thalamus ;;
  *) echo "Unsupported smoke ROI: $SMOKE_ROI"; exit 1 ;;
esac
SMOKE_REPORT="$CAMPAIGN_ROOT/smoke/$SMOKE_DATASET/$SMOKE_SUBJECT"
SMOKE_ARCHIVE=/mnt/parscratch/users/cop23bi/pre_charm_simulation_outputs/$RUN_ID/smoke
mkdir -p "$SMOKE_REPORT"
```

Archive any previous outputs and require exactly one simulation task:

```bash
python3 CamCan_Experiment/simulation/prepare_inplace_rerun.py preflight \
  --roi-root "$SMOKE_ROI_PARENT" \
  --dataset-glob "$SMOKE_DATASET" \
  --repeats "$SMOKE_REPEAT" \
  --subjects "$SMOKE_SUBJECT" \
  --expected-tasks 1 \
  --remesh-results-dir "$REMESH_RESULTS" \
  --manifest "$SMOKE_REPORT/tasks.tsv" \
  --cleanup-manifest "$SMOKE_REPORT/output_archive.tsv" \
  --output-archive-root "$SMOKE_ARCHIVE" \
  --apply

ROI_ROOT="$SMOKE_ROI_PARENT" \
MANIFEST="$SMOKE_REPORT/tasks.tsv" \
MONTAGE_PRESET="$SMOKE_PRESET" \
LOG_DIR="$SMOKE_REPORT/logs" \
EXPECTED_TASKS=1 \
MAX_CONCURRENT_TASKS=1 \
MAX_ARRAY_TASKS=1 \
MAX_SUBMITTED_CHUNKS=1 \
TI_TASK_MAX_RETRIES=2 \
bash CamCan_Experiment/HPC_scripts/submit_camcan_inplace_rerun.sh
```

After the simulation validates, set the appropriate ROI-specific MNI baseline
and submit a one-subject/one-repeat post job:

```bash
SMOKE_MNI_BASELINE=/path/to/the/correct/roi-specific/mni-baseline
sbatch --export="ALL,BATCH_ROOT=$SMOKE_ROI_PARENT,BATCH_REPEATS=$SMOKE_REPEAT,PIPELINE_SUBJECTS=$SMOKE_SUBJECT,PIPELINE_MNI_BASELINE_ROOT=$SMOKE_MNI_BASELINE,PIPELINE_FORCE=1,PIPELINE_REPEATABILITY_ENABLED=0,PIPELINE_FIGURE_GENERATION_ENABLED=0" \
  CamCan_Experiment/HPC_scripts/run_post_processing_batch.slurm
```

Inspect the simulation log's `montage_config` event, the post log's
`camcan_post_electrodes` event, and the produced subject metrics before
continuing.

## 5. Full remesh and validation

After the smoke test passes:

```bash
MANIFEST="$CAMPAIGN_ROOT/remesh_manifest.tsv" \
RESULT_DIR="$REMESH_RESULTS" \
LOG_DIR="$CAMPAIGN_ROOT/remesh_logs" \
MAX_CONCURRENT_TASKS=8 \
TI_CHARM_REMESH_MAX_RETRIES=2 \
bash CamCan_Experiment/HPC_scripts/submit_charm_only_remesh.sh
```

Task 0 is idempotent and will be reported as already complete. Validate all
7,000 meshes before starting simulations:

```bash
python3 CamCan_Experiment/charm_only_remesh/workflow.py validate \
  --manifest "$CAMPAIGN_ROOT/remesh_manifest.tsv" \
  --result-dir "$REMESH_RESULTS" \
  --summary "$CAMPAIGN_ROOT/remesh_full_validation.tsv"
```

## 6. Archive old outputs and prepare simulation manifests

Run each ROI separately. The four required mappings are:

| ROI parent | Dataset glob | Montage preset |
|---|---|---|
| `Left_M1_Runs` | `Left_M1_Data_*` | `left-m1` |
| `Left_Hippocampus_Runs` | `Left_Hippocampus_Data_*` | `left-hippocampus` |
| `Right_DLPC_Runs` | `Right_DLPC_Data_*` | `right-dlpfc` |
| `Right_Thalamus_Runs` | `Right_Thalamus_Data_*` | `right-thalamus` |

Example for left M1, first as a dry run:

```bash
ROI_NAME=Left_M1
ROI_PARENT="$ROI_ROOT/${ROI_NAME}_Runs"
SIM_REPORT="$CAMPAIGN_ROOT/simulation/$ROI_NAME"
SIM_ARCHIVE=/mnt/parscratch/users/cop23bi/pre_charm_simulation_outputs/$RUN_ID/$ROI_NAME
mkdir -p "$SIM_REPORT"

python3 CamCan_Experiment/simulation/prepare_inplace_rerun.py preflight \
  --roi-root "$ROI_PARENT" \
  --dataset-glob "${ROI_NAME}_Data_*" \
  --repeats 01 02 03 04 05 06 07 08 09 10 \
  --expected-tasks 1750 \
  --remesh-results-dir "$REMESH_RESULTS" \
  --manifest "$SIM_REPORT/tasks.tsv" \
  --cleanup-manifest "$SIM_REPORT/output_archive_audit.tsv"
```

Review the audit. When previous generated outputs exist, the dry-run task
manifest is deliberately blocked until they are archived. Then archive the
previous outputs atomically and rebuild the manifest:

```bash
python3 CamCan_Experiment/simulation/prepare_inplace_rerun.py preflight \
  --roi-root "$ROI_PARENT" \
  --dataset-glob "${ROI_NAME}_Data_*" \
  --repeats 01 02 03 04 05 06 07 08 09 10 \
  --expected-tasks 1750 \
  --remesh-results-dir "$REMESH_RESULTS" \
  --manifest "$SIM_REPORT/tasks.tsv" \
  --cleanup-manifest "$SIM_REPORT/output_archive_apply.tsv" \
  --output-archive-root "$SIM_ARCHIVE" \
  --apply
```

The archive must be on the same filesystem as the live ROI root; the operation
refuses non-atomic cross-filesystem moves and existing destinations.

## 7. Rerun simulations using the new meshes

```bash
ROI_ROOT="$ROI_PARENT" \
MANIFEST="$SIM_REPORT/tasks.tsv" \
MONTAGE_PRESET=left-m1 \
LOG_DIR="$SIM_REPORT/logs" \
TI_TASK_MAX_RETRIES=2 \
bash CamCan_Experiment/HPC_scripts/submit_camcan_inplace_rerun.sh
```

The submitter and every task verify that the dataset ROI matches the preset and
that `utils/targets.csv` has SHA-256
`97a8c7a72faf88d9af9e4facbdf628fba1a130d327da778bcbd00af66f2916e6`.
The simulation runner is always called with `--reuse-existing-mesh`; that mode
now refuses live ROAST maps and never executes CHARM or segmentation merging.

Validate each ROI after the jobs finish:

```bash
python3 CamCan_Experiment/simulation/prepare_inplace_rerun.py validate \
  --manifest "$SIM_REPORT/tasks.tsv" \
  --summary "$SIM_REPORT/simulation_validation.tsv"
```

## 8. Post-processing

Run post-processing only after simulation validation is complete. In
`HPC_scripts/run_post_processing_batch.slurm`, set `BATCH_ROOT` to one ROI
parent at a time and set the ROI-specific MNI baseline. The launcher is already
restricted to repeats 01-10 and configured with:

```text
PIPELINE_CAMCAN_TARGETS_CSV=/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/utils/targets.csv
PIPELINE_EXPECTED_TARGETS_SHA256=97a8c7a72faf88d9af9e4facbdf628fba1a130d327da778bcbd00af66f2916e6
```

Post-processing derives the four electrode names from the same optimized CSV
and reads their subject-space coordinates from:

```text
{root}/{subject}/anat/m2m_{subject}/eeg_positions/EEG10-10_UI_Jurak_2007.csv
```

It refuses manual/stale electrode datasets when the CamCan targets source is
enabled and fails if any of the four requested cap positions is absent.
