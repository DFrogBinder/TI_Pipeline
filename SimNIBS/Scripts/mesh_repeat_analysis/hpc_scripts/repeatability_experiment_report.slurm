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

# ---------------------------------------------------------------------------
# Post-processing settings
#
# Edit this block once on the HPC, then submit with:
#
#   sbatch hpc_scripts/repeatability_experiment_report.slurm
#
# Blank optional values are ignored. Environment variables with the same names
# still override these defaults, which is useful for one-off smoke tests.
# ---------------------------------------------------------------------------
PIPELINE_DIR_CONFIG="/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/mesh_repeat_analysis"
EXPERIMENT_CONFIG_CONFIG="${PIPELINE_DIR_CONFIG}/paired_repeatability_experiment.json"

# Subject selection. Leave SUBJECT_ID_CONFIG blank to process every subject in
# the JSON config. Set MAX_SUBJECTS_CONFIG for a quick smoke test.
SUBJECT_ID_CONFIG=""
CONDITIONS_CONFIG=""
MAX_SUBJECTS_CONFIG=""

# Output location. Leave blank to write to <experiment_root>/_analysis.
OUTPUT_DIR_CONFIG=""

# ROI selection. Use either ROI_PRESET_CONFIG or ROI_NAME_CONFIG/ROI_LABELS_CONFIG.
ROI_PRESET_CONFIG="left-hippocampus"
ROI_NAME_CONFIG=""
ROI_LABELS_CONFIG=""
ATLAS_DIR_CONFIG="/mnt/parscratch/cop23bi/ZIPs/atlases"

# Repeatability/report parameters.
REFERENCE_REPEAT_CONFIG=""
SPATIAL_PERCENTILE_CONFIG="99.0"
COMPARE_METRIC_CONFIG="median_roi"

# Cohort comparison. Set SKIP_COHORT_CONFIG=1 to disable it even if the config
# has cohort paths.
COMPARE_COHORT_ROOT_CONFIG=""
COHORT_REGION_NAME_CONFIG=""
COHORT_REGION_LABEL_CONFIG=""
COHORT_METRIC_CONFIG=""
SKIP_COHORT_CONFIG="0"

# Logs. Leave LOG_DIR_CONFIG blank to use <pipeline>/logs.
LOG_DIR_CONFIG=""
LOG_FILE_CONFIG=""
JSONL_LOG_FILE_CONFIG=""

PIPELINE_DIR="${PIPELINE_DIR:-${PIPELINE_DIR_CONFIG}}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${EXPERIMENT_CONFIG_CONFIG}}"
SUBJECT_ID="${SUBJECT_ID:-${SUBJECT_ID_CONFIG}}"
CONDITIONS="${CONDITIONS:-${CONDITIONS_CONFIG}}"
MAX_SUBJECTS="${MAX_SUBJECTS:-${MAX_SUBJECTS_CONFIG}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_DIR_CONFIG}}"
ROI_PRESET="${ROI_PRESET:-${ROI_PRESET_CONFIG}}"
ROI_NAME="${ROI_NAME:-${ROI_NAME_CONFIG}}"
ROI_LABELS="${ROI_LABELS:-${ROI_LABELS_CONFIG}}"
ATLAS_DIR="${ATLAS_DIR:-${ATLAS_DIR_CONFIG}}"
REFERENCE_REPEAT="${REFERENCE_REPEAT:-${REFERENCE_REPEAT_CONFIG}}"
SPATIAL_PERCENTILE="${SPATIAL_PERCENTILE:-${SPATIAL_PERCENTILE_CONFIG}}"
COMPARE_METRIC="${COMPARE_METRIC:-${COMPARE_METRIC_CONFIG}}"
COMPARE_COHORT_ROOT="${COMPARE_COHORT_ROOT:-${COMPARE_COHORT_ROOT_CONFIG}}"
COHORT_REGION_NAME="${COHORT_REGION_NAME:-${COHORT_REGION_NAME_CONFIG}}"
COHORT_REGION_LABEL="${COHORT_REGION_LABEL:-${COHORT_REGION_LABEL_CONFIG}}"
COHORT_METRIC="${COHORT_METRIC:-${COHORT_METRIC_CONFIG}}"
SKIP_COHORT="${SKIP_COHORT:-${SKIP_COHORT_CONFIG}}"

SCRIPT_DIR="${PIPELINE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$SCRIPT_DIR"

if [ -z "${EXPERIMENT_CONFIG}" ]; then
    echo "[ERROR] EXPERIMENT_CONFIG must point to the paired repeatability JSON config."
    exit 1
fi
if [ ! -f "${EXPERIMENT_CONFIG}" ]; then
    echo "[ERROR] Experiment config does not exist: ${EXPERIMENT_CONFIG}"
    echo "[ERROR] Edit EXPERIMENT_CONFIG_CONFIG near the top of this script, or create that file from paired_repeatability_experiment.example.json."
    exit 1
fi

if [ -n "${ROI_PRESET}" ] && { [ -n "${ROI_NAME}" ] || [ -n "${ROI_LABELS}" ]; }; then
    echo "[ERROR] Use either ROI_PRESET or ROI_NAME/ROI_LABELS, not both."
    exit 1
fi

LOG_DIR="${LOG_DIR:-${LOG_DIR_CONFIG:-${SCRIPT_DIR}/logs}}"
mkdir -p "$LOG_DIR"
JOB_ID="${SLURM_JOB_ID:-manual}"
LOG_FILE="${LOG_FILE:-${LOG_FILE_CONFIG:-${LOG_DIR}/repeatability_experiment_report_${JOB_ID}.log}}"
JSONL_LOG_FILE="${JSONL_LOG_FILE:-${JSONL_LOG_FILE_CONFIG:-${LOG_DIR}/repeatability_experiment_report_${JOB_ID}.jsonl}}"

echo "[INFO] Pipeline root:     ${SCRIPT_DIR}"
echo "[INFO] Experiment config: ${EXPERIMENT_CONFIG}"
echo "[INFO] CPUs:              ${SLURM_CPUS_PER_TASK:-1}"
echo "[INFO] Log file:          ${LOG_FILE}"
echo "[INFO] JSONL log file:    ${JSONL_LOG_FILE}"
echo "[INFO] Subject mode:      ${SUBJECT_ID:-all subjects}"
echo "[INFO] Conditions:        ${CONDITIONS:-all configured conditions}"
echo "[INFO] ROI preset:        ${ROI_PRESET:-none}"
echo "[INFO] ROI name:          ${ROI_NAME:-none}"
echo "[INFO] ROI labels:        ${ROI_LABELS:-none}"
echo "[INFO] Atlas dir:         ${ATLAS_DIR:-from config or default}"

PYTHON_CMD=(
    python -u "${SCRIPT_DIR}/post/repeatability_experiment_report.py"
    --config "${EXPERIMENT_CONFIG}"
    --log-file "${JSONL_LOG_FILE}"
)

if [ -n "${SUBJECT_ID}" ]; then
    PYTHON_CMD+=(--subject "${SUBJECT_ID}")
else
    PYTHON_CMD+=(--all-subjects)
fi

if [ -n "${CONDITIONS}" ]; then
    PYTHON_CMD+=(--conditions "${CONDITIONS}")
fi
if [ -n "${MAX_SUBJECTS}" ]; then
    PYTHON_CMD+=(--max-subjects "${MAX_SUBJECTS}")
fi
if [ -n "${OUTPUT_DIR}" ]; then
    PYTHON_CMD+=(--output-dir "${OUTPUT_DIR}")
fi
if [ -n "${ROI_PRESET}" ]; then
    PYTHON_CMD+=(--roi-preset "${ROI_PRESET}")
fi
if [ -n "${ROI_NAME}" ]; then
    PYTHON_CMD+=(--roi-name "${ROI_NAME}")
fi
if [ -n "${ROI_LABELS}" ]; then
    PYTHON_CMD+=(--roi-labels "${ROI_LABELS}")
fi
if [ -n "${ATLAS_DIR}" ]; then
    PYTHON_CMD+=(--atlas-dir "${ATLAS_DIR}")
fi
if [ -n "${REFERENCE_REPEAT}" ]; then
    PYTHON_CMD+=(--reference-repeat "${REFERENCE_REPEAT}")
fi
if [ -n "${SPATIAL_PERCENTILE}" ]; then
    PYTHON_CMD+=(--spatial-percentile "${SPATIAL_PERCENTILE}")
fi
if [ -n "${COMPARE_METRIC}" ]; then
    PYTHON_CMD+=(--compare-metric "${COMPARE_METRIC}")
fi
if [ -n "${COMPARE_COHORT_ROOT}" ]; then
    PYTHON_CMD+=(--compare-cohort-root "${COMPARE_COHORT_ROOT}")
fi
if [ -n "${COHORT_REGION_NAME}" ]; then
    PYTHON_CMD+=(--cohort-region-name "${COHORT_REGION_NAME}")
fi
if [ -n "${COHORT_REGION_LABEL}" ]; then
    PYTHON_CMD+=(--cohort-region-label "${COHORT_REGION_LABEL}")
fi
if [ -n "${COHORT_METRIC}" ]; then
    PYTHON_CMD+=(--cohort-metric "${COHORT_METRIC}")
fi
if [ "${SKIP_COHORT}" = "1" ]; then
    PYTHON_CMD+=(--skip-cohort)
fi

"${PYTHON_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
