#!/bin/bash
set -euo pipefail

PIPELINE_DIR_CONFIG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_CONFIG_CONFIG="${PIPELINE_DIR_CONFIG}/paired_repeatability_experiment.json"
LOG_DIR_CONFIG="${PIPELINE_DIR_CONFIG}/logs"

JOB_NAME_CONFIG="ti_repeat_post"
PARTITION_CONFIG="sheffield"
CPUS_PER_TASK_CONFIG="8"
MEMORY_CONFIG="32G"
TIME_LIMIT_CONFIG="12:00:00"
MAX_CONCURRENT_TASKS_CONFIG="10"
SLURM_OUTPUT_CONFIG=""

PYTHON_BIN_CONFIG="python"
SBATCH_BIN_CONFIG="sbatch"
SLURM_SCRIPT_CONFIG="${PIPELINE_DIR_CONFIG}/hpc_scripts/repeatability_experiment_report_array.slurm"
REPORT_TASK_PY_CONFIG="${PIPELINE_DIR_CONFIG}/hpc_scripts/repeatability_report_array_task.py"

PIPELINE_DIR="${PIPELINE_DIR:-${PIPELINE_DIR_CONFIG}}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${EXPERIMENT_CONFIG_CONFIG}}"
LOG_DIR="${LOG_DIR:-${LOG_DIR_CONFIG}}"
JOB_NAME="${JOB_NAME:-${JOB_NAME_CONFIG}}"
PARTITION="${PARTITION:-${PARTITION_CONFIG}}"
CPUS_PER_TASK="${CPUS_PER_TASK:-${CPUS_PER_TASK_CONFIG}}"
MEMORY="${MEMORY:-${MEMORY_CONFIG}}"
TIME_LIMIT="${TIME_LIMIT:-${TIME_LIMIT_CONFIG}}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-${MAX_CONCURRENT_TASKS_CONFIG}}"
SLURM_OUTPUT="${SLURM_OUTPUT:-${SLURM_OUTPUT_CONFIG}}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON_BIN_CONFIG}}"
SBATCH_BIN="${SBATCH_BIN:-${SBATCH_BIN_CONFIG}}"
SLURM_SCRIPT="${SLURM_SCRIPT:-${SLURM_SCRIPT_CONFIG}}"
REPORT_TASK_PY="${REPORT_TASK_PY:-${REPORT_TASK_PY_CONFIG}}"

CONDITIONS="${CONDITIONS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
ROI_PRESET="${ROI_PRESET:-}"
ROI_NAME="${ROI_NAME:-}"
ROI_LABELS="${ROI_LABELS:-}"
ATLAS_DIR="${ATLAS_DIR:-}"
REFERENCE_REPEAT="${REFERENCE_REPEAT:-}"
SPATIAL_PERCENTILE="${SPATIAL_PERCENTILE:-99.0}"
COMPARE_METRIC="${COMPARE_METRIC:-median_roi}"
COMPARE_COHORT_ROOT="${COMPARE_COHORT_ROOT:-}"
COHORT_REGION_NAME="${COHORT_REGION_NAME:-}"
COHORT_REGION_LABEL="${COHORT_REGION_LABEL:-}"
COHORT_METRIC="${COHORT_METRIC:-}"
SKIP_COHORT="${SKIP_COHORT:-0}"

resolve_path() {
    "${PYTHON_BIN}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1"
}

require_file() {
    local label="$1"
    local path="$2"
    if [ ! -f "${path}" ]; then
        echo "[ERROR] ${label} does not exist: ${path}"
        exit 1
    fi
}

PIPELINE_DIR="$(resolve_path "${PIPELINE_DIR}")"
EXPERIMENT_CONFIG="$(resolve_path "${EXPERIMENT_CONFIG}")"
LOG_DIR="$(resolve_path "${LOG_DIR}")"
SLURM_SCRIPT="$(resolve_path "${SLURM_SCRIPT}")"
REPORT_TASK_PY="$(resolve_path "${REPORT_TASK_PY}")"
mkdir -p "${LOG_DIR}"
if [ -z "${SLURM_OUTPUT}" ]; then
    SLURM_OUTPUT="${LOG_DIR}/slurm-report-%A_%a.out"
fi

if ! [[ "${MAX_CONCURRENT_TASKS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] MAX_CONCURRENT_TASKS must be a positive integer: ${MAX_CONCURRENT_TASKS}"
    exit 1
fi

require_file "Experiment config" "${EXPERIMENT_CONFIG}"
require_file "Report array task wrapper" "${REPORT_TASK_PY}"
require_file "Report Slurm script" "${SLURM_SCRIPT}"

SUBJECT_COUNT="$("${PYTHON_BIN}" -c 'import json, sys; data=json.load(open(sys.argv[1])); print(len(data.get("subjects", [])))' "${EXPERIMENT_CONFIG}")"
if ! [[ "${SUBJECT_COUNT}" =~ ^[0-9]+$ ]] || [ "${SUBJECT_COUNT}" -lt 1 ]; then
    echo "[ERROR] Config produced zero report subjects: ${EXPERIMENT_CONFIG}"
    exit 1
fi

ARRAY_SPEC="0-$((SUBJECT_COUNT - 1))%${MAX_CONCURRENT_TASKS}"
EXPORT_VARS="ALL,EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG},PIPELINE_DIR=${PIPELINE_DIR},LOG_DIR=${LOG_DIR},REPORT_TASK_PY=${REPORT_TASK_PY},CONDITIONS=${CONDITIONS},OUTPUT_DIR=${OUTPUT_DIR},ROI_PRESET=${ROI_PRESET},ROI_NAME=${ROI_NAME},ROI_LABELS=${ROI_LABELS},ATLAS_DIR=${ATLAS_DIR},REFERENCE_REPEAT=${REFERENCE_REPEAT},SPATIAL_PERCENTILE=${SPATIAL_PERCENTILE},COMPARE_METRIC=${COMPARE_METRIC},COMPARE_COHORT_ROOT=${COMPARE_COHORT_ROOT},COHORT_REGION_NAME=${COHORT_REGION_NAME},COHORT_REGION_LABEL=${COHORT_REGION_LABEL},COHORT_METRIC=${COHORT_METRIC},SKIP_COHORT=${SKIP_COHORT}"

echo "[INFO] Pipeline root:     ${PIPELINE_DIR}"
echo "[INFO] Experiment config: ${EXPERIMENT_CONFIG}"
echo "[INFO] Subject count:     ${SUBJECT_COUNT}"
echo "[INFO] Max concurrent:    ${MAX_CONCURRENT_TASKS}"
echo "[INFO] Array spec:        ${ARRAY_SPEC}"
echo "[INFO] Conditions:        ${CONDITIONS:-all configured conditions}"
echo "[INFO] Report task:       ${REPORT_TASK_PY}"
echo "[INFO] Slurm script:      ${SLURM_SCRIPT}"

SBATCH_CMD=(
    "${SBATCH_BIN}"
    --job-name="${JOB_NAME}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${MEMORY}"
    --time="${TIME_LIMIT}"
    --array="${ARRAY_SPEC}"
    --export="${EXPORT_VARS}"
)
if [ -n "${PARTITION}" ]; then
    SBATCH_CMD+=(--partition="${PARTITION}")
fi
if [ -n "${SLURM_OUTPUT}" ]; then
    SBATCH_CMD+=(--output="${SLURM_OUTPUT}")
fi
SBATCH_CMD+=("${SLURM_SCRIPT}")

"${SBATCH_CMD[@]}"
