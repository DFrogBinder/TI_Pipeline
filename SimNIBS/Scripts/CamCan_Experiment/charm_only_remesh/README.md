# CHARM-only remesh, simulation, and post-processing campaign

This workflow treats the successful 7,000-row installation manifest as the
authority for every dataset copy and can operate on one validated 1,750-task
ROI scope at a time. It never calls CHARM segmentation. The only meshing
command it permits is:

```text
charm <subject> --mesh
```

The workflow fails closed if a ROAST/custom map remains in a live `anat`
directory or if an installed CHARM label differs from the installation
manifest. By default it also requires the original source label. Campaigns
that deliberately removed the uploaded source directory can explicitly use
`--allow-missing-source-labels`; each task then makes a verified, temporary
rollback snapshot beside the installed label and deletes it after remeshing.
Obsolete ROAST maps and meshes are deleted from the HPC because their recovery
copies exist outside the HPC. Generated simulation/post outputs remain
separately archived.

## 1. Set campaign paths

Run these commands from the HPC checkout at
`/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts`:

```bash
ROI_ROOT=/mnt/parscratch/users/cop23bi/ZIPs/Analised-Data
INSTALL_REPORT=/mnt/parscratch/users/cop23bi/charm_segmentation_install/charm-map-install-20260715-203145
INSTALL_MANIFEST="$INSTALL_REPORT/apply_manifest.tsv"
ROI_PREFIX=Left_Hippocampus
EXPECTED_TARGETS=1750
RUN_ID=charm-only-remesh-${ROI_PREFIX}-$(date +%Y%m%d-%H%M%S)
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
  --report "$CAMPAIGN_ROOT/roast_removal_audit.tsv" \
  --roi-prefix "$ROI_PREFIX" \
  --expected-targets "$EXPECTED_TARGETS"
```

Review the TSV and confirm the off-HPC backup is available. Then delete only
the exact audited ROAST/custom paths:

```bash
python3 CamCan_Experiment/charm_only_remesh/workflow.py remove-roast \
  --install-manifest "$INSTALL_MANIFEST" \
  --roi-root "$ROI_ROOT" \
  --report "$CAMPAIGN_ROOT/roast_removal_apply.tsv" \
  --roi-prefix "$ROI_PREFIX" \
  --expected-targets "$EXPECTED_TARGETS" \
  --apply-delete \
  --confirm-external-backup
```

The apply command deletes only the selected ROI's
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
  --report "$CAMPAIGN_ROOT/install_backup_removal_audit.tsv" \
  --roi-prefix "$ROI_PREFIX" \
  --expected-backups "$EXPECTED_TARGETS"
```

Proceed only if the summary reports `status=audit`, `found=1750`,
`expected=1750`, and `issues=0`. Then delete the exact verified files:

```bash
python3 CamCan_Experiment/charm_only_remesh/workflow.py remove-install-backups \
  --install-manifest "$INSTALL_MANIFEST" \
  --backup-root "$INSTALL_BACKUP_ROOT" \
  --report "$CAMPAIGN_ROOT/install_backup_removal_apply.tsv" \
  --roi-prefix "$ROI_PREFIX" \
  --expected-backups "$EXPECTED_TARGETS" \
  --apply-delete \
  --confirm-external-backup
```

This command accepts only backup paths recorded in `apply_manifest.tsv` that
are inside the declared backup root and whose hashes still match
`before_sha256`. During ROI-by-ROI execution the shared backup root remains in
place until the other ROI backups are removed; no other ROI is touched.

## 3. Build the immutable remesh manifest

```bash
python3 CamCan_Experiment/charm_only_remesh/workflow.py preflight \
  --install-manifest "$INSTALL_MANIFEST" \
  --roi-root "$ROI_ROOT" \
  --manifest "$CAMPAIGN_ROOT/remesh_manifest.tsv" \
  --summary "$CAMPAIGN_ROOT/remesh_preflight.json" \
  --roi-prefix "$ROI_PREFIX" \
  --expected-targets "$EXPECTED_TARGETS"
```

If the uploaded source-map directory was deliberately deleted after all live
copies were hash-verified and installed, add:

```text
--allow-missing-source-labels
```

This opt-in mode still verifies every installed label against
`installed_sha256`. Before deleting an old mesh, the task creates a local
rollback snapshot of that verified label. The snapshot is used to restore a
label if CHARM changes it, survives a hard task interruption for the next
retry, and is removed after a normal success or handled failure.

Do not submit unless the summary reports `status=ready`, `ready=1750`, and
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

Before the full 1,750-mesh ROI submission, carry task 0 through simulation and
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

After the smoke test passes, submit the remaining tasks for this ROI:

```bash
MANIFEST="$CAMPAIGN_ROOT/remesh_manifest.tsv" \
RESULT_DIR="$REMESH_RESULTS" \
LOG_DIR="$CAMPAIGN_ROOT/remesh_logs" \
MAX_CONCURRENT_TASKS=50 \
MAX_ARRAY_TASKS=875 \
MAX_SUBMITTED_CHUNKS=1 \
TI_CHARM_REMESH_MAX_RETRIES=2 \
bash CamCan_Experiment/HPC_scripts/submit_charm_only_remesh.sh
```

This submits global tasks 0-874 as one array capped at 50 concurrent tasks.
Task 0 is idempotent and will be reported as already complete. Stanage's
per-user submitted-job QOS rejects two 875-element arrays in the queue at once,
so submit the second half only after the first array terminates:

```bash
MANIFEST="$CAMPAIGN_ROOT/remesh_manifest.tsv" \
RESULT_DIR="$REMESH_RESULTS" \
LOG_DIR="$CAMPAIGN_ROOT/remesh_logs" \
START_TASK_OFFSET=875 \
MAX_CONCURRENT_TASKS=50 \
MAX_ARRAY_TASKS=875 \
MAX_SUBMITTED_CHUNKS=1 \
TI_CHARM_REMESH_MAX_RETRIES=2 \
bash CamCan_Experiment/HPC_scripts/submit_charm_only_remesh.sh
```

Validate all 1,750 Left Hippocampus meshes before starting simulations:

```bash
python3 CamCan_Experiment/charm_only_remesh/workflow.py validate \
  --manifest "$CAMPAIGN_ROOT/remesh_manifest.tsv" \
  --result-dir "$REMESH_RESULTS" \
  --summary "$CAMPAIGN_ROOT/remesh_full_validation.tsv" \
  --skip-mesh-load
```

Each remesh task already loads and validates its mesh before writing a complete
result. The aggregate validation rechecks installed/source segmentation hashes,
mesh hashes, task identities, and current mesh paths without redundantly loading
all meshes again.

## 6. Delete obsolete outputs and prepare simulation manifests

Run each ROI separately. The four required mappings are:

| ROI parent | Dataset glob | Montage preset |
|---|---|---|
| `Left_M1_Runs` | `Left_M1_Data_*` | `left-m1` |
| `Left_Hippocampus_Runs` | `Left_Hippocampus_Data_*` | `left-hippocampus` |
| `Right_DLPC_Runs` | `Right_DLPC_Data_*` | `right-dlpfc` |
| `Right_Thalamus_Runs` | `Right_Thalamus_Data_*` | `right-thalamus` |

For the current Left Hippocampus campaign, first run a dry audit:

```bash
ROI_NAME=Left_Hippocampus
ROI_PARENT="$ROI_ROOT/${ROI_NAME}_Runs"
SIM_REPORT="$CAMPAIGN_ROOT/simulation/$ROI_NAME"
mkdir -p "$SIM_REPORT"

python3 CamCan_Experiment/simulation/prepare_inplace_rerun.py preflight \
  --roi-root "$ROI_PARENT" \
  --dataset-glob "${ROI_NAME}_Data_*" \
  --repeats 01 02 03 04 05 06 07 08 09 10 \
  --expected-tasks 1750 \
  --manifest "$SIM_REPORT/tasks.tsv" \
  --cleanup-manifest "$SIM_REPORT/output_delete_audit.tsv" \
  --delete-generated-outputs
```

Review the audit. When previous generated outputs exist, the dry-run task
manifest is deliberately blocked until they are removed. Once the obsolete
outputs are confirmed expendable, permanently delete only the allowlisted
generated paths and rebuild the manifest:

```bash
python3 CamCan_Experiment/simulation/prepare_inplace_rerun.py preflight \
  --roi-root "$ROI_PARENT" \
  --dataset-glob "${ROI_NAME}_Data_*" \
  --repeats 01 02 03 04 05 06 07 08 09 10 \
  --expected-tasks 1750 \
  --remesh-results-dir "$REMESH_RESULTS" \
  --manifest "$SIM_REPORT/tasks.tsv" \
  --cleanup-manifest "$SIM_REPORT/output_delete_apply.tsv" \
  --delete-generated-outputs \
  --apply \
  --confirm-obsolete-output-deletion
```

Deletion mode never targets T1/T2 inputs, installed CHARM labels, or head
meshes. It refuses to run without the explicit deletion confirmation.

## 7. Rerun simulations using the new meshes

```bash
ROI_ROOT="$ROI_PARENT" \
MANIFEST="$SIM_REPORT/tasks.tsv" \
MONTAGE_PRESET=left-hippocampus \
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
