#!/bin/bash
set -euo pipefail

SUBJECTS_FILE="${TI_SUBJECTS_FILE:-/users/cop23bi/scripts/subjects.txt}"
BATCH_SCRIPT="${TI_JOB_ARRAY_SCRIPT:-/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/HPC_scripts/my_jobArray.slurm}"
MAX_CONCURRENT="${TI_ARRAY_MAX_CONCURRENT:-8}"

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

sbatch \
    --array="$ARRAY_SPEC" \
    --export=ALL,TI_SUBJECTS_FILE="$SUBJECTS_FILE" \
    "$@" \
    "$BATCH_SCRIPT"
