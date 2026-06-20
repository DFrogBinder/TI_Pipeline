# Current-Repair and Mesh-Repeat HPC Runbook

Last updated: 2026-06-18

This document records the end-to-end launch procedure for two related workflows:

1. The staged current-repair median-fixed experiment in `SimNIBS/Scripts/ti_current_repair`.
2. The restored original repeatability rerun in `SimNIBS/Scripts/mesh_repeat_analysis`.

The current-repair workflow is the manuscript-specific staged pipeline. It records
stage state under the experiment root and submits simulation and report work as
Slurm arrays. The original mesh-repeat workflow uses the restored mesh-repeat
code path, with corrected pair-2 injection currents.

## Preconditions

Run from the repository root on the HPC:

```bash
cd /users/cop23bi/Repos/TI_Pipeline
git checkout ti_current_repair
git pull --ff-only
git status

export REPO_ROOT=/users/cop23bi/Repos/TI_Pipeline
export CURRENT_REPAIR_DIR="${REPO_ROOT}/SimNIBS/Scripts/ti_current_repair"
export MESH_REPEAT_DIR="${REPO_ROOT}/SimNIBS/Scripts/mesh_repeat_analysis"
```

The preferred state before launching production work is:

```text
nothing to commit, working tree clean
```

If the working tree contains only disposable local edits and generated files,
the destructive cleanup is:

```bash
git reset --hard HEAD
git clean -fd
git status
```

Use the stronger ignored-file cleanup only when caches, logs, and ignored local
outputs can also be deleted:

```bash
git reset --hard HEAD
git clean -fdx
git status
```

## Current Injection Sanity Check

The previous current-overwrite bug was caused by copying pair 1 with
`deepcopy(tdcs1)` and not resetting pair 2 currents afterward. The fixed pattern
is that every runner sets `tdcs2.currents` immediately after creating `tdcs2`.

Optional check from the repository root:

```bash
rg -n "tdcs2 = .*deepcopy\\(tdcs1\\)|tdcs2\\.currents" \
  SimNIBS/Scripts/ti_current_repair/simulation_runners \
  SimNIBS/Scripts/mesh_repeat_analysis/simulation_runners \
  SimNIBS/Scripts/CamCan_Experiment/simulation
```

Focused tests:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=SimNIBS/Scripts/CamCan_Experiment/simulation:SimNIBS/Scripts/mesh_repeat_analysis/simulation_runners \
pytest \
  SimNIBS/Scripts/mesh_repeat_analysis/tests/test_repeatability_montage.py \
  SimNIBS/Scripts/CamCan_Experiment/simulation/test_target_montages.py \
  -q -p no:cacheprovider
```

Expected result:

```text
9 passed
```

## Workflow A: Current-Repair Staged Median-Fixed Pipeline

### What This Pipeline Does

The staged pipeline runs:

1. Initialize stage configs and logging.
2. Submit remesh simulations as a Slurm array.
3. Analyze remesh outputs as a Slurm report array.
4. Select the median representative remesh repeat per subject.
5. Physically copy the selected remesh anatomy and mesh into fixed-mesh inputs.
6. Submit fixed-mesh simulations as a Slurm array.
7. Analyze remesh and fixed-mesh outputs as a paired Slurm report array.
8. Build final figures and summary tables.
9. Report expected versus observed output counts.

Simulation/report stages are Slurm arrays. Local bookkeeping stages run directly
and are recorded in `_pipeline/events.jsonl`.

### Directory and Variables

```bash
cd "$REPO_ROOT"

export RUN_NAME=current_repair_median_fixed_v1
export SOURCE_ROOT=/mnt/parscratch/users/cop23bi/ti_dataset_balanced_10_corrected
export EXPERIMENT_ROOT=/mnt/parscratch/users/cop23bi/current-repair/${RUN_NAME}
export ATLAS_DIR=/mnt/parscratch/users/cop23bi/ZIPs/atlases
export SUBJECTS=sub-CC122620,sub-CC222496,sub-CC120120,sub-CC321506,sub-CC410182,sub-CC420075,sub-CC510534,sub-CC520209,sub-CC711128,sub-CC721418
```

Adjust `RUN_NAME` for each production attempt. Do not reuse an experiment root
unless intentionally resuming or inspecting a previous run.

### Preflight

Run the current-repair unit tests:

```bash
PYTHONDONTWRITEBYTECODE=1 \
pytest "$CURRENT_REPAIR_DIR/tests/test_staged_current_repair_pipeline.py" -q -p no:cacheprovider
```

Dry-run the config generation:

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" init --dry-run \
  --source-root "$SOURCE_ROOT" \
  --experiment-root "$EXPERIMENT_ROOT" \
  --subjects "$SUBJECTS" \
  --repeat-count 40 \
  --atlas-dir "$ATLAS_DIR" \
  --roi-preset left-hippocampus
```

### Stage 1: Initialize

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" init \
  --source-root "$SOURCE_ROOT" \
  --experiment-root "$EXPERIMENT_ROOT" \
  --subjects "$SUBJECTS" \
  --repeat-count 40 \
  --atlas-dir "$ATLAS_DIR" \
  --roi-preset left-hippocampus
```

Expected outputs:

```text
$EXPERIMENT_ROOT/_pipeline/configs/remesh_only.json
$EXPERIMENT_ROOT/_pipeline/configs/fixed_mesh_only.json
$EXPERIMENT_ROOT/_pipeline/configs/paired_analysis.json
$EXPERIMENT_ROOT/_pipeline/experiment_manifest.json
$EXPERIMENT_ROOT/_pipeline/events.jsonl
$EXPERIMENT_ROOT/_pipeline/stage_status.json
```

### Stage 2: Submit Remesh Simulations

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" submit-remesh \
  --experiment-root "$EXPERIMENT_ROOT" \
  --max-concurrent 50
```

This submits a Slurm array using:

```text
ti_current_repair/hpc_scripts/submit_repeatability_experiment.sh
ti_current_repair/hpc_scripts/repeatability_experiment_array.slurm
```

The submission record is written under:

```text
$EXPERIMENT_ROOT/_pipeline/submitted_jobs/
$EXPERIMENT_ROOT/_pipeline/events.jsonl
$EXPERIMENT_ROOT/_pipeline/logs/submit-remesh/
```

Wait for the Slurm array to finish, then check status:

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" status \
  --experiment-root "$EXPERIMENT_ROOT"
```

For 10 subjects and 40 remesh repeats, expected `TI.msh` count is `400`.

### Stage 3: Analyze Remesh Outputs

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" analyze-remesh \
  --experiment-root "$EXPERIMENT_ROOT" \
  --max-concurrent 10
```

This submits a report Slurm array, one task per subject. Wait for the array to
finish, then check:

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" status \
  --experiment-root "$EXPERIMENT_ROOT"
```

Expected analysis summaries:

```text
$EXPERIMENT_ROOT/_analysis/<subject>/remesh/summary.csv
```

### Stage 4: Select Median Remesh Repeats

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" select-medians \
  --experiment-root "$EXPERIMENT_ROOT" \
  --metric median_roi
```

Expected output:

```text
$EXPERIMENT_ROOT/_pipeline/median_mesh_selection/median_representative_remesh_repeats.csv
```

The selection is computed only from this experiment root's newly generated
`_analysis` tree.

### Stage 5: Seed Fixed-Mesh Inputs

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" seed-fixed \
  --experiment-root "$EXPERIMENT_ROOT"
```

This physically copies the selected remesh anatomy and mesh into:

```text
$EXPERIMENT_ROOT/<subject>_repeatability/fixed_mesh/mesh_cache/<subject>/anat
$EXPERIMENT_ROOT/<subject>_repeatability/fixed_mesh/repeats/repeat_*/<subject>/anat
```

The seeder excludes previous `SimNIBS/` outputs, lock files, temp files, and done
markers. It fails if symlinks remain in seeded anatomy workspaces.

Expected output:

```text
$EXPERIMENT_ROOT/_pipeline/fixed_seed_manifest.csv
```

Check before submitting fixed simulations:

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" status \
  --experiment-root "$EXPERIMENT_ROOT"
```

The status should report selected median rows and fixed seed rows for all
subjects, plus passing no-symlink and checksum checks.

### Stage 6: Submit Fixed-Mesh Simulations

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" submit-fixed \
  --experiment-root "$EXPERIMENT_ROOT" \
  --max-concurrent 50
```

Wait for the Slurm array to finish, then check:

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" status \
  --experiment-root "$EXPERIMENT_ROOT"
```

For 10 subjects and 40 fixed repeats, expected fixed `TI.msh` count is `400`.

### Stage 7: Analyze Paired Conditions

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" analyze-paired \
  --experiment-root "$EXPERIMENT_ROOT" \
  --max-concurrent 10
```

This submits a report Slurm array. It analyzes remesh and fixed-mesh conditions
and writes paired comparison outputs.

Expected batch-level outputs:

```text
$EXPERIMENT_ROOT/_analysis/paired_condition_summary.csv
$EXPERIMENT_ROOT/_analysis/paired_condition_summary.json
$EXPERIMENT_ROOT/_analysis/paired_condition_failures.json
```

### Stage 8: Make Figures

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" make-figures \
  --experiment-root "$EXPERIMENT_ROOT"
```

Expected outputs:

```text
$EXPERIMENT_ROOT/_figures/presentation/condition_median_roi_by_repeat.png
$EXPERIMENT_ROOT/_figures/presentation/presentation_condition_summary.csv
$EXPERIMENT_ROOT/_figures/presentation/presentation_manifest.json
```

### Stage 9: Final Status

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" status \
  --experiment-root "$EXPERIMENT_ROOT"
```

The status report should show expected versus observed counts for:

- remesh `TI.msh`
- fixed `TI.msh`
- remesh summaries
- fixed summaries
- selected medians
- fixed seed rows
- no-symlink checks
- mesh checksum matches
- presentation figure outputs

### Current-Repair Logs and Provenance

Primary locations:

```text
$EXPERIMENT_ROOT/_pipeline/events.jsonl
$EXPERIMENT_ROOT/_pipeline/stage_status.json
$EXPERIMENT_ROOT/_pipeline/submitted_jobs/
$EXPERIMENT_ROOT/_pipeline/logs/
$EXPERIMENT_ROOT/_analysis/
$EXPERIMENT_ROOT/_figures/presentation/
```

To inspect submission history:

```bash
ls -lh "$EXPERIMENT_ROOT/_pipeline/submitted_jobs"
tail -n 20 "$EXPERIMENT_ROOT/_pipeline/events.jsonl"
```

## Workflow B: Restored Original Mesh-Repeat Rerun

### What This Workflow Does

This reruns the original repeatability experiment using the restored
`mesh_repeat_analysis` code. The corrected current assignment is present in the
runner: after `tdcs2` is created from `deepcopy(tdcs1)`, pair-2 current is reset
to the intended `T7/P7` values.

This workflow uses:

```text
SimNIBS/Scripts/mesh_repeat_analysis/simulation_runners/repeatability_experiment.py
SimNIBS/Scripts/mesh_repeat_analysis/hpc_scripts/submit_repeatability_experiment.sh
SimNIBS/Scripts/mesh_repeat_analysis/hpc_scripts/repeatability_experiment_array.slurm
SimNIBS/Scripts/mesh_repeat_analysis/post/repeatability_experiment_report.py
SimNIBS/Scripts/mesh_repeat_analysis/hpc_scripts/repeatability_experiment_report.slurm
```

The simulation step is a Slurm array. The restored report wrapper is a Slurm job
that can run all subjects or a subset, depending on its settings.

### Directory and Variables

```bash
cd "$REPO_ROOT"

export ORIGINAL_RUN_NAME=original_repeatability_correct_current_v1
export ORIGINAL_SOURCE_ROOT=/mnt/parscratch/users/cop23bi/ti_dataset_balanced_10_corrected
export ORIGINAL_EXPERIMENT_ROOT=/mnt/parscratch/users/cop23bi/repeatability-rerun-correct-current/${ORIGINAL_RUN_NAME}
export ORIGINAL_CONFIG="${MESH_REPEAT_DIR}/paired_repeatability_experiment.json"
```

### Create Config

For a remesh-only rerun of the original experiment, create:

```bash
cat > "$ORIGINAL_CONFIG" <<'EOF'
{
  "source_root": "/mnt/parscratch/users/cop23bi/ti_dataset_balanced_10_corrected",
  "experiment_root": "/mnt/parscratch/users/cop23bi/repeatability-rerun-correct-current/original_repeatability_correct_current_v1",
  "subjects": [
    "sub-CC122620",
    "sub-CC222496",
    "sub-CC120120",
    "sub-CC321506",
    "sub-CC410182",
    "sub-CC420075",
    "sub-CC510534",
    "sub-CC520209",
    "sub-CC711128",
    "sub-CC721418"
  ],
  "conditions": [
    {
      "name": "remesh",
      "mesh_mode": "remesh",
      "repeat_count": 40,
      "description": "Fresh mesh per repeat with corrected pair-2 current."
    }
  ],
  "analysis": {
    "roi_preset": "left-hippocampus",
    "atlas_dir": "/mnt/parscratch/users/cop23bi/ZIPs/atlases",
    "compare_metric": "median_roi"
  }
}
EOF
```

If the source or atlas paths differ on the HPC, edit the JSON before submitting.

### Preflight

Inspect the planned tasks:

```bash
python "$MESH_REPEAT_DIR/simulation_runners/repeatability_experiment.py" show-plan \
  --config "$ORIGINAL_CONFIG"
```

Count tasks:

```bash
python "$MESH_REPEAT_DIR/simulation_runners/repeatability_experiment.py" show-plan \
  --config "$ORIGINAL_CONFIG" \
  --count-only
```

For 10 subjects and 40 remesh repeats, expected count is:

```text
400
```

Dry-run one task:

```bash
python "$MESH_REPEAT_DIR/simulation_runners/repeatability_experiment.py" run-task \
  --config "$ORIGINAL_CONFIG" \
  --task-index 0 \
  --dry-run
```

### Submit Original Mesh-Repeat Simulations

```bash
PIPELINE_DIR="$MESH_REPEAT_DIR" \
EXPERIMENT_CONFIG="$ORIGINAL_CONFIG" \
MAX_CONCURRENT_TASKS=50 \
JOB_NAME=ti_repeat_correct_current \
LOG_DIR=/mnt/parscratch/users/cop23bi/repeatability-rerun-correct-current/logs \
bash "$MESH_REPEAT_DIR/hpc_scripts/submit_repeatability_experiment.sh"
```

The submitter computes the Slurm array size from the JSON config and submits:

```text
hpc_scripts/repeatability_experiment_array.slurm
```

Normal recovery settings are:

```text
OVERWRITE_OUTPUT=0
FORCE_MESH=0
```

Use `OVERWRITE_OUTPUT=1` only when completed repeat folders should be rerun.
Use `FORCE_MESH=1` only when meshes should be regenerated.

### Validate Outputs

After the array finishes, inspect logs and expected output count:

```bash
find "$ORIGINAL_EXPERIMENT_ROOT" -name TI.msh | wc -l
ls -lh /mnt/parscratch/users/cop23bi/repeatability-rerun-correct-current/logs
```

For the remesh-only 10 subject by 40 repeat config, expected `TI.msh` count is
`400`.

### Run Original Mesh-Repeat Analysis

The restored analysis Slurm wrapper is:

```text
hpc_scripts/repeatability_experiment_report.slurm
```

Submit with explicit exports:

```bash
sbatch \
  --export="ALL,EXPERIMENT_CONFIG=${ORIGINAL_CONFIG},PIPELINE_DIR=${MESH_REPEAT_DIR},ROI_PRESET=left-hippocampus,ATLAS_DIR=/mnt/parscratch/users/cop23bi/ZIPs/atlases" \
  "$MESH_REPEAT_DIR/hpc_scripts/repeatability_experiment_report.slurm"
```

Alternatively, edit the configuration block near the top of
`hpc_scripts/repeatability_experiment_report.slurm` and submit:

```bash
sbatch "$MESH_REPEAT_DIR/hpc_scripts/repeatability_experiment_report.slurm"
```

For a smoke test, set `SUBJECT_ID_CONFIG` or `MAX_SUBJECTS_CONFIG` inside the
Slurm script before submission.

### Original Mesh-Repeat Analysis Outputs

Per subject and condition:

```text
$ORIGINAL_EXPERIMENT_ROOT/_analysis/<subject>/remesh/summary.csv
$ORIGINAL_EXPERIMENT_ROOT/_analysis/<subject>/remesh/repeatability_stats.csv
$ORIGINAL_EXPERIMENT_ROOT/_analysis/<subject>/remesh/repeatability_report.md
```

Batch-level files may include:

```text
$ORIGINAL_EXPERIMENT_ROOT/_analysis/paired_condition_summary.csv
$ORIGINAL_EXPERIMENT_ROOT/_analysis/paired_condition_summary.json
$ORIGINAL_EXPERIMENT_ROOT/_analysis/paired_condition_failures.json
```

For a remesh-only rerun, the most important first check is the per-subject
`summary.csv` output.

## Operational Notes

- Do not run production jobs from a dirty working tree unless the diffs were
  intentional and recorded.
- Keep current-repair outputs under `/mnt/parscratch/users/cop23bi/current-repair/`.
- Keep original repeatability rerun outputs under a separate root, for example
  `/mnt/parscratch/users/cop23bi/repeatability-rerun-correct-current/`.
- Do not reuse old current-correction or current-repair data roots unless the
  run is an intentional resume.
- The current-repair staged pipeline records submission command, environment,
  Slurm job id, and expected output counts in `_pipeline/events.jsonl`.
- The restored mesh-repeat workflow does not have the same `_pipeline` staged
  provenance layer, so preserve the JSON config and Slurm logs for traceability.
