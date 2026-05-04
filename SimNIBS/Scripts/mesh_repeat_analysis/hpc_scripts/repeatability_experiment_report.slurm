#!/bin/bash
#SBATCH --job-name=ti_repeat_post
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

module purge
module use "$HOME/modules"
module load SimNIBS/4.0.1-foss-2023a

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

set -euo pipefail

PIPELINE_DIR_DEFAULT="/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/mesh_repeat_analysis"
SCRIPT_DIR="${PIPELINE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
if [ -d "${PIPELINE_DIR_DEFAULT}" ]; then
    SCRIPT_DIR="${PIPELINE_DIR:-${PIPELINE_DIR_DEFAULT}}"
fi
cd "$SCRIPT_DIR"

EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-}"
if [ -z "${EXPERIMENT_CONFIG}" ]; then
    echo "[ERROR] EXPERIMENT_CONFIG must point to the paired repeatability JSON config."
    exit 1
fi

if [ -n "${ROI_PRESET:-}" ] && { [ -n "${ROI_NAME:-}" ] || [ -n "${ROI_LABELS:-}" ]; }; then
    echo "[ERROR] Use either ROI_PRESET or ROI_NAME/ROI_LABELS, not both."
    exit 1
fi

LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/repeatability_experiment_report_${SLURM_JOB_ID}.log}"
JSONL_LOG_FILE="${JSONL_LOG_FILE:-${LOG_DIR}/repeatability_experiment_report_${SLURM_JOB_ID}.jsonl}"

echo "[INFO] Pipeline root:     ${SCRIPT_DIR}"
echo "[INFO] Experiment config: ${EXPERIMENT_CONFIG}"
echo "[INFO] CPUs:              ${SLURM_CPUS_PER_TASK:-1}"
echo "[INFO] Log file:          ${LOG_FILE}"
echo "[INFO] JSONL log file:    ${JSONL_LOG_FILE}"

PYTHON_CMD=(
    python -u "${SCRIPT_DIR}/post/repeatability_experiment_report.py"
    --config "${EXPERIMENT_CONFIG}"
    --log-file "${JSONL_LOG_FILE}"
)

if [ -n "${SUBJECT_ID:-}" ]; then
    PYTHON_CMD+=(--subject "${SUBJECT_ID}")
else
    PYTHON_CMD+=(--all-subjects)
fi

if [ -n "${CONDITIONS:-}" ]; then
    PYTHON_CMD+=(--conditions "${CONDITIONS}")
fi
if [ -n "${MAX_SUBJECTS:-}" ]; then
    PYTHON_CMD+=(--max-subjects "${MAX_SUBJECTS}")
fi
if [ -n "${OUTPUT_DIR:-}" ]; then
    PYTHON_CMD+=(--output-dir "${OUTPUT_DIR}")
fi
if [ -n "${ROI_PRESET:-}" ]; then
    PYTHON_CMD+=(--roi-preset "${ROI_PRESET}")
fi
if [ -n "${ROI_NAME:-}" ]; then
    PYTHON_CMD+=(--roi-name "${ROI_NAME}")
fi
if [ -n "${ROI_LABELS:-}" ]; then
    PYTHON_CMD+=(--roi-labels "${ROI_LABELS}")
fi
if [ -n "${ATLAS_DIR:-}" ]; then
    PYTHON_CMD+=(--atlas-dir "${ATLAS_DIR}")
fi
if [ -n "${REFERENCE_REPEAT:-}" ]; then
    PYTHON_CMD+=(--reference-repeat "${REFERENCE_REPEAT}")
fi
if [ -n "${SPATIAL_PERCENTILE:-}" ]; then
    PYTHON_CMD+=(--spatial-percentile "${SPATIAL_PERCENTILE}")
fi
if [ -n "${COMPARE_METRIC:-}" ]; then
    PYTHON_CMD+=(--compare-metric "${COMPARE_METRIC}")
fi
if [ -n "${COMPARE_COHORT_ROOT:-}" ]; then
    PYTHON_CMD+=(--compare-cohort-root "${COMPARE_COHORT_ROOT}")
fi
if [ -n "${COHORT_REGION_NAME:-}" ]; then
    PYTHON_CMD+=(--cohort-region-name "${COHORT_REGION_NAME}")
fi
if [ -n "${COHORT_REGION_LABEL:-}" ]; then
    PYTHON_CMD+=(--cohort-region-label "${COHORT_REGION_LABEL}")
fi
if [ -n "${COHORT_METRIC:-}" ]; then
    PYTHON_CMD+=(--cohort-metric "${COHORT_METRIC}")
fi
if [ "${SKIP_COHORT:-0}" = "1" ]; then
    PYTHON_CMD+=(--skip-cohort)
fi

"${PYTHON_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
