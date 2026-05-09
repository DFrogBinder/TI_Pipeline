#!/bin/bash
set -euo pipefail

REPAIR_PLAN_FILE="${TI_REPAIR_PLAN_FILE:-/users/cop23bi/scripts/simulation_repair_plan/repair_plan.tsv}"
BATCH_SCRIPT="${TI_REPAIR_JOB_ARRAY_SCRIPT:-/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/HPC_scripts/repair_jobArray.slurm}"
MAX_CONCURRENT="${TI_ARRAY_MAX_CONCURRENT:-8}"
COMPLETION_CHECK_PY="${TI_COMPLETION_CHECK_PY:-/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/simulation/validate_simulation_outputs.py}"
SIM_RUNNER_PY="${TI_SIM_RUNNER_PY:-/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/simulation/TI_runner_multi-core.py}"
MESH_TIMEOUT_HOURS="${TI_MESH_TIMEOUT_HOURS:-4}"
# 0 or a negative value means keep requeueing incomplete subjects until the
# simulation exits cleanly and validation passes.
MESH_MAX_RETRIES="${TI_MESH_MAX_RETRIES:-0}"

if ! [[ "$MESH_MAX_RETRIES" =~ ^-?[0-9]+$ ]]; then
    echo "[ERROR] TI_MESH_MAX_RETRIES must be an integer; got '${MESH_MAX_RETRIES}'." >&2
    exit 1
fi

if [ "$MESH_MAX_RETRIES" -gt 0 ]; then
    MESH_RETRY_LIMIT_LABEL="$MESH_MAX_RETRIES"
else
    MESH_RETRY_LIMIT_LABEL="unlimited"
fi

if [ ! -f "$REPAIR_PLAN_FILE" ]; then
    echo "[ERROR] Repair plan file not found: $REPAIR_PLAN_FILE" >&2
    exit 1
fi

if [ ! -f "$BATCH_SCRIPT" ]; then
    echo "[ERROR] Batch script not found: $BATCH_SCRIPT" >&2
    exit 1
fi

awk -F '\t' 'NR == 1 { exit !(($1 == "task_id") && ($4 == "dataset_root") && ($5 == "subject")) }' "$REPAIR_PLAN_FILE" || {
    echo "[ERROR] Repair plan has an unexpected header: $REPAIR_PLAN_FILE" >&2
    exit 1
}

FIRST_BLANK_LINE=$(awk 'NR > 1 && NF == 0 { print NR; exit }' "$REPAIR_PLAN_FILE")
if [ -n "$FIRST_BLANK_LINE" ]; then
    echo "[ERROR] Blank line found in repair plan at line $FIRST_BLANK_LINE: $REPAIR_PLAN_FILE" >&2
    exit 1
fi

REPAIR_TASK_COUNT=$(awk 'NR > 1 && NF > 0 { count++ } END { print count + 0 }' "$REPAIR_PLAN_FILE")
if [ "$REPAIR_TASK_COUNT" -le 0 ]; then
    echo "[INFO] Repair plan contains no runnable tasks: $REPAIR_PLAN_FILE"
    exit 0
fi

ARRAY_END=$((REPAIR_TASK_COUNT - 1))
ARRAY_SPEC="0-${ARRAY_END}%${MAX_CONCURRENT}"

echo "[INFO] Repair plan:      $REPAIR_PLAN_FILE"
echo "[INFO] Repair task count:$REPAIR_TASK_COUNT"
echo "[INFO] Array spec:       $ARRAY_SPEC"
echo "[INFO] Batch script:     $BATCH_SCRIPT"
echo "[INFO] Runner:           $SIM_RUNNER_PY"
echo "[INFO] Completion check: $COMPLETION_CHECK_PY"
echo "[INFO] Mesh timeout:     ${MESH_TIMEOUT_HOURS} hour(s)"
echo "[INFO] Simulation retries: ${MESH_RETRY_LIMIT_LABEL}"

sbatch \
    --array="$ARRAY_SPEC" \
    --export=ALL,TI_REPAIR_PLAN_FILE="$REPAIR_PLAN_FILE",TI_SIM_RUNNER_PY="$SIM_RUNNER_PY",TI_COMPLETION_CHECK_PY="$COMPLETION_CHECK_PY",TI_MESH_TIMEOUT_HOURS="$MESH_TIMEOUT_HOURS",TI_MESH_MAX_RETRIES="$MESH_MAX_RETRIES" \
    "$@" \
    "$BATCH_SCRIPT"
