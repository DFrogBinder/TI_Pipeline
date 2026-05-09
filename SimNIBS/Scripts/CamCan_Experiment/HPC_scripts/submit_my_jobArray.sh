#!/bin/bash
set -euo pipefail

SUBJECTS_FILE="${TI_SUBJECTS_FILE:-/users/cop23bi/scripts/subjects.txt}"
BATCH_SCRIPT="${TI_JOB_ARRAY_SCRIPT:-/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/HPC_scripts/my_jobArray.slurm}"
MAX_CONCURRENT="${TI_ARRAY_MAX_CONCURRENT:-8}"
SIM_ROOT="${TI_SIM_ROOT:-/mnt/parscratch/users/cop23bi/LM1}"
COMPLETION_CHECK_PY="${TI_COMPLETION_CHECK_PY:-/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/simulation/validate_simulation_outputs.py}"
MESH_TIMEOUT_HOURS="${TI_MESH_TIMEOUT_HOURS:-4}"
# 0 or a negative value means keep requeueing timed-out subjects until the
# simulation exits without the mesh-timeout code.
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

if [ ! -f "$SUBJECTS_FILE" ]; then
    echo "[ERROR] Subjects file not found: $SUBJECTS_FILE" >&2
    exit 1
fi

if [ ! -f "$BATCH_SCRIPT" ]; then
    echo "[ERROR] Batch script not found: $BATCH_SCRIPT" >&2
    exit 1
fi

FIRST_BLANK_LINE=$(awk 'NF == 0 { print NR; exit }' "$SUBJECTS_FILE")
if [ -n "$FIRST_BLANK_LINE" ]; then
    echo "[ERROR] Blank line found in subjects file at line $FIRST_BLANK_LINE: $SUBJECTS_FILE" >&2
    exit 1
fi

SUBJECT_COUNT=$(awk 'END { print NR }' "$SUBJECTS_FILE")
if [ "$SUBJECT_COUNT" -le 0 ]; then
    echo "[ERROR] Subjects file is empty: $SUBJECTS_FILE" >&2
    exit 1
fi

ARRAY_END=$((SUBJECT_COUNT - 1))
ARRAY_SPEC="0-${ARRAY_END}%${MAX_CONCURRENT}"

echo "[INFO] Subjects file:   $SUBJECTS_FILE"
echo "[INFO] Subject count:   $SUBJECT_COUNT"
echo "[INFO] Array spec:      $ARRAY_SPEC"
echo "[INFO] Batch script:    $BATCH_SCRIPT"
echo "[INFO] Simulation root: $SIM_ROOT"
echo "[INFO] Completion check:$COMPLETION_CHECK_PY"
echo "[INFO] Mesh timeout:    ${MESH_TIMEOUT_HOURS} hour(s)"
echo "[INFO] Simulation retries: ${MESH_RETRY_LIMIT_LABEL}"

sbatch \
    --array="$ARRAY_SPEC" \
    --export=ALL,TI_SUBJECTS_FILE="$SUBJECTS_FILE",TI_SIM_ROOT="$SIM_ROOT",TI_COMPLETION_CHECK_PY="$COMPLETION_CHECK_PY",TI_MESH_TIMEOUT_HOURS="$MESH_TIMEOUT_HOURS",TI_MESH_MAX_RETRIES="$MESH_MAX_RETRIES" \
    "$@" \
    "$BATCH_SCRIPT"
