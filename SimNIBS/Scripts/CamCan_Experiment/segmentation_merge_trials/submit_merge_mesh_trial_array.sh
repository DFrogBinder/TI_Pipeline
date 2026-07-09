#!/bin/bash
set -euo pipefail

SUBJECTS_FILE="${TI_MERGE_TRIAL_SUBJECTS_FILE:-/users/cop23bi/scripts/merge_mesh_trial_subjects.txt}"
SOURCE_ROOT="${TI_MERGE_TRIAL_SOURCE_ROOT:-/mnt/parscratch/users/cop23bi/ZIPs/Analised-Data/Left_Hippocampus_Runs/Left_Hippocampus_Data_01}"
OUT_ROOT="${TI_MERGE_TRIAL_OUT_ROOT:-/mnt/parscratch/users/cop23bi/merge_mesh_trials/Left_Hippocampus_Data_01}"
BATCH_SCRIPT="${TI_MERGE_TRIAL_BATCH_SCRIPT:-/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/segmentation_merge_trials/merge_mesh_trial_array.slurm}"
RUNNER_PY="${TI_MERGE_TRIAL_RUNNER_PY:-/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/segmentation_merge_trials/run_merge_mesh_trial.py}"
VALIDATOR_PY="${TI_MERGE_TRIAL_VALIDATOR_PY:-/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/segmentation_merge_trials/validate_merge_mesh_trial.py}"
MAX_CONCURRENT="${TI_MERGE_TRIAL_MAX_CONCURRENT:-4}"
TRIAL_TIMEOUT_HOURS="${TI_MERGE_TRIAL_TIMEOUT_HOURS:-7}"
TRIAL_MAX_RETRIES="${TI_MERGE_TRIAL_MAX_RETRIES:-0}"

if ! [[ "$TRIAL_MAX_RETRIES" =~ ^-?[0-9]+$ ]]; then
    echo "[ERROR] TI_MERGE_TRIAL_MAX_RETRIES must be an integer; got '${TRIAL_MAX_RETRIES}'." >&2
    exit 1
fi

if [ "$TRIAL_MAX_RETRIES" -gt 0 ]; then
    TRIAL_RETRY_LIMIT_LABEL="$TRIAL_MAX_RETRIES"
else
    TRIAL_RETRY_LIMIT_LABEL="unlimited"
fi

if [ ! -f "$SUBJECTS_FILE" ]; then
    echo "[ERROR] Subjects file not found: $SUBJECTS_FILE" >&2
    exit 1
fi

if [ ! -f "$BATCH_SCRIPT" ]; then
    echo "[ERROR] Batch script not found: $BATCH_SCRIPT" >&2
    exit 1
fi

if [ ! -f "$RUNNER_PY" ]; then
    echo "[ERROR] Runner not found: $RUNNER_PY" >&2
    exit 1
fi

if [ ! -f "$VALIDATOR_PY" ]; then
    echo "[ERROR] Validator not found: $VALIDATOR_PY" >&2
    exit 1
fi

SUBJECT_COUNT=$(awk 'NF > 0 && $1 !~ /^#/ { count++ } END { print count + 0 }' "$SUBJECTS_FILE")
if [ "$SUBJECT_COUNT" -le 0 ]; then
    echo "[ERROR] Subjects file contains no runnable subjects: $SUBJECTS_FILE" >&2
    exit 1
fi

ARRAY_END=$((SUBJECT_COUNT - 1))
ARRAY_SPEC="0-${ARRAY_END}%${MAX_CONCURRENT}"

echo "[INFO] Subjects file: $SUBJECTS_FILE"
echo "[INFO] Subject count: $SUBJECT_COUNT"
echo "[INFO] Array spec:    $ARRAY_SPEC"
echo "[INFO] Source root:   $SOURCE_ROOT"
echo "[INFO] Output root:   $OUT_ROOT"
echo "[INFO] Batch script:  $BATCH_SCRIPT"
echo "[INFO] Runner:        $RUNNER_PY"
echo "[INFO] Validator:     $VALIDATOR_PY"
echo "[INFO] Timeout:       ${TRIAL_TIMEOUT_HOURS} hour(s)"
echo "[INFO] Retries:       ${TRIAL_RETRY_LIMIT_LABEL}"

sbatch \
    --array="$ARRAY_SPEC" \
    --export=ALL,TI_MERGE_TRIAL_SUBJECTS_FILE="$SUBJECTS_FILE",TI_MERGE_TRIAL_SOURCE_ROOT="$SOURCE_ROOT",TI_MERGE_TRIAL_OUT_ROOT="$OUT_ROOT",TI_MERGE_TRIAL_RUNNER_PY="$RUNNER_PY",TI_MERGE_TRIAL_VALIDATOR_PY="$VALIDATOR_PY",TI_MERGE_TRIAL_TIMEOUT_HOURS="$TRIAL_TIMEOUT_HOURS",TI_MERGE_TRIAL_MAX_RETRIES="$TRIAL_MAX_RETRIES" \
    "$@" \
    "$BATCH_SCRIPT"
