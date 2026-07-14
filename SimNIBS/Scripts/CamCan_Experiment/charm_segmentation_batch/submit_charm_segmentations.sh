#!/bin/bash
set -euo pipefail

CAMCAN_DIR="${CAMCAN_DIR:-/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment}"
SOURCE_ROOT="${TI_CHARM_SOURCE_ROOT:-/mnt/parscratch/users/cop23bi/ti_dataset}"
OUT_ROOT="${TI_CHARM_OUTPUT_ROOT:-/mnt/parscratch/users/cop23bi/charm_segmentations}"
SUBJECTS_FILE="${TI_CHARM_SUBJECTS_FILE:-${OUT_ROOT}/submission/subjects.txt}"
RUNNER_PY="${TI_CHARM_RUNNER_PY:-${CAMCAN_DIR}/charm_segmentation_batch/run_charm_segmentation.py}"
COLLECTOR_PY="${TI_CHARM_COLLECTOR_PY:-${CAMCAN_DIR}/charm_segmentation_batch/collect_charm_segmentations.py}"
ARRAY_SCRIPT="${TI_CHARM_ARRAY_SCRIPT:-${CAMCAN_DIR}/charm_segmentation_batch/charm_segmentation_array.slurm}"
COLLECT_SCRIPT="${TI_CHARM_COLLECT_SCRIPT:-${CAMCAN_DIR}/charm_segmentation_batch/collect_charm_segmentations.slurm}"
COLLECTION_DIR="${TI_CHARM_COLLECTION_DIR:-${OUT_ROOT}/collection}"
MAX_CONCURRENT="${TI_CHARM_MAX_CONCURRENT:-50}"
TIMEOUT_HOURS="${TI_CHARM_TIMEOUT_HOURS:-7.5}"
PARTITION="${TI_CHARM_PARTITION:-sheffield}"
CPUS_PER_TASK="${TI_CHARM_CPUS_PER_TASK:-8}"
MEMORY="${TI_CHARM_MEMORY:-32G}"
TIME_LIMIT="${TI_CHARM_TIME_LIMIT:-08:00:00}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"

for required_file in "$RUNNER_PY" "$COLLECTOR_PY" "$ARRAY_SCRIPT" "$COLLECT_SCRIPT"; do
    if [ ! -f "$required_file" ]; then
        echo "[ERROR] Required file not found: $required_file" >&2
        exit 1
    fi
done
if ! [[ "$MAX_CONCURRENT" =~ ^[0-9]+$ ]] || [ "$MAX_CONCURRENT" -lt 1 ]; then
    echo "[ERROR] TI_CHARM_MAX_CONCURRENT must be a positive integer." >&2
    exit 1
fi

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/submission"
PREFLIGHT_REPORT="$OUT_ROOT/submission/preflight.tsv"
python3 "$RUNNER_PY" \
    --source-root "$SOURCE_ROOT" \
    --out-root "$OUT_ROOT" \
    --subjects-file "$SUBJECTS_FILE" \
    --discover-only \
    --preflight-report "$PREFLIGHT_REPORT"

SUBJECT_COUNT=$(awk 'NF > 0 && $1 !~ /^#/ { count++ } END { print count + 0 }' "$SUBJECTS_FILE")
if [ "$SUBJECT_COUNT" -le 0 ]; then
    echo "[ERROR] Discovery produced no runnable subjects: $SUBJECTS_FILE" >&2
    exit 1
fi
BLOCKED_COUNT=$(awk -F '\t' 'NR > 1 && $4 == "blocked" { count++ } END { print count + 0 }' "$PREFLIGHT_REPORT")
ARRAY_END=$((SUBJECT_COUNT - 1))
ARRAY_SPEC="0-${ARRAY_END}%${MAX_CONCURRENT}"
ARRAY_OUTPUT="$OUT_ROOT/logs/slurm-%A_%a.out"
COLLECT_OUTPUT="$OUT_ROOT/logs/collect-%j.out"
EXPORT_VARS="ALL,CAMCAN_DIR=${CAMCAN_DIR},TI_CHARM_SOURCE_ROOT=${SOURCE_ROOT},TI_CHARM_OUTPUT_ROOT=${OUT_ROOT},TI_CHARM_SUBJECTS_FILE=${SUBJECTS_FILE},TI_CHARM_RUNNER_PY=${RUNNER_PY},TI_CHARM_COLLECTOR_PY=${COLLECTOR_PY},TI_CHARM_COLLECTION_DIR=${COLLECTION_DIR},TI_CHARM_TIMEOUT_HOURS=${TIMEOUT_HOURS}"

echo "[INFO] Source root:       $SOURCE_ROOT"
echo "[INFO] Output root:       $OUT_ROOT"
echo "[INFO] Subjects file:     $SUBJECTS_FILE"
echo "[INFO] Runnable subjects: $SUBJECT_COUNT"
echo "[INFO] Blocked subjects:  $BLOCKED_COUNT"
echo "[INFO] Array:             $ARRAY_SPEC"
echo "[INFO] Partition:         $PARTITION"
echo "[INFO] Resources/task:    ${CPUS_PER_TASK} CPUs, ${MEMORY}, ${TIME_LIMIT}"
echo "[INFO] Persistent CHARM output: one tissue_labeling_upsampled.nii.gz per subject"
echo "[INFO] Temporary CHARM products: deleted after each subject; no surfaces, mesh, or simulation"

ARRAY_SUBMISSION=$(
    "$SBATCH_BIN" \
        --parsable \
        --partition="$PARTITION" \
        --cpus-per-task="$CPUS_PER_TASK" \
        --mem="$MEMORY" \
        --time="$TIME_LIMIT" \
        --array="$ARRAY_SPEC" \
        --output="$ARRAY_OUTPUT" \
        --export="$EXPORT_VARS" \
        "$ARRAY_SCRIPT"
)
ARRAY_JOB_ID="${ARRAY_SUBMISSION%%;*}"
if ! [[ "$ARRAY_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Could not parse array job ID from: $ARRAY_SUBMISSION" >&2
    exit 1
fi
echo "[INFO] Submitted CHARM array job: $ARRAY_JOB_ID"

COLLECT_SUBMISSION=$(
    "$SBATCH_BIN" \
        --parsable \
        --dependency="afterany:${ARRAY_JOB_ID}" \
        --partition="$PARTITION" \
        --output="$COLLECT_OUTPUT" \
        --export="$EXPORT_VARS" \
        "$COLLECT_SCRIPT"
)
COLLECT_JOB_ID="${COLLECT_SUBMISSION%%;*}"
if ! [[ "$COLLECT_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Could not parse collector job ID from: $COLLECT_SUBMISSION" >&2
    exit 1
fi
echo "[INFO] Submitted dependent collector job: $COLLECT_JOB_ID"
echo "[INFO] Final flat maps will be in: $COLLECTION_DIR/maps"
echo "[INFO] Audit summary: $COLLECTION_DIR/collection_summary.json"
