#!/bin/bash
#SBATCH --job-name=ti_repair_launch
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-repair-launch-%j.out

set -euo pipefail

# =============================================================================
# Repair launcher configuration
#
# Edit this block, then submit this single launcher:
#
#   sbatch HPC_scripts/launch_simulation_repair.slurm
#
# The default stage is "all": discovery first, then one repair array per repeat.
# Set REPAIR_DRY_RUN="1" for a no-submit audit of the sbatch commands.
# This launcher intentionally reads routine configuration from this file only.
# =============================================================================

# Parent directory containing repeat roots such as Left_Hippocampus_Data_01 ... _10.
REPAIR_BATCH_ROOT=""

# discovery: scan/report only
# simulate: submit arrays from a previous discovery
# all: run discovery, then submit arrays
REPAIR_STAGE="all"

# Expected number of repeat directories. Leave empty to disable the warning.
REPAIR_EXPECTED_REPEAT_COUNT="10"

# Optional subset, e.g. "01 02 10". Leave empty to include all matched repeats.
REPAIR_REPEATS=""

# Discovery filters and outputs.
REPAIR_DATASET_GLOB="*_Data_*"
REPAIR_SUBJECT_GLOB="sub-*"
REPAIR_EXPECTED_SUBJECTS_FILE=""
REPAIR_OUT_DIR=""
REPAIR_MIN_BYTES="1"
REPAIR_CHECK_NIFTI="1"
REPAIR_SKIP_INPUT_CHECK="0"

# Submission behavior.
REPAIR_DRY_RUN="0"
REPAIR_ARRAY_MAX_CONCURRENT="8"
REPAIR_MESH_TIMEOUT_HOURS="4"
# 0 or negative means unlimited retry/requeue attempts.
REPAIR_MESH_MAX_RETRIES="0"

# Repository/HPC script paths.
REPAIR_REPO_ROOT="/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment"
REPAIR_PLANNER_PY="${REPAIR_REPO_ROOT}/simulation/plan_simulation_repair.py"
REPAIR_SUBMIT_SCRIPT="${REPAIR_REPO_ROOT}/HPC_scripts/submit_repair_jobArray.sh"
REPAIR_ARRAY_SCRIPT="${REPAIR_REPO_ROOT}/HPC_scripts/repair_jobArray.slurm"
REPAIR_SIM_RUNNER_PY="${REPAIR_REPO_ROOT}/simulation/TI_runner_multi-core.py"
REPAIR_COMPLETION_CHECK_PY="${REPAIR_REPO_ROOT}/simulation/validate_simulation_outputs.py"

# Discovery Python environment. The SimNIBS module is used by default so nibabel
# is available for output validation.
REPAIR_LOAD_SIMNIBS_MODULE="1"
REPAIR_MODULE_DIR="$HOME/modules"
REPAIR_SIMNIBS_MODULE="SimNIBS/4.0.1-foss-2023a"
REPAIR_PYTHON="python"

# Optional repair task state/log locations.
REPAIR_LOG_DIR="/users/cop23bi/scripts/logs/repair"
REPAIR_STATE_DIR="/users/cop23bi/scripts/logs/repair_retry_state"

# =============================================================================
# End user configuration
# =============================================================================

truthy() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

if [ -z "$REPAIR_BATCH_ROOT" ]; then
    echo "[ERROR] REPAIR_BATCH_ROOT is empty." >&2
    echo "[ERROR] Edit the configuration block at the top of HPC_scripts/launch_simulation_repair.slurm," >&2
    echo "        then submit with: sbatch HPC_scripts/launch_simulation_repair.slurm" >&2
    exit 2
fi

case "$REPAIR_STAGE" in
    discovery|simulate|all) ;;
    *)
        echo "[ERROR] REPAIR_STAGE must be one of discovery, simulate, all; got '${REPAIR_STAGE}'." >&2
        exit 2
        ;;
esac

for required_file in "$REPAIR_PLANNER_PY" "$REPAIR_SUBMIT_SCRIPT" "$REPAIR_ARRAY_SCRIPT" "$REPAIR_SIM_RUNNER_PY" "$REPAIR_COMPLETION_CHECK_PY"; do
    if [ ! -f "$required_file" ]; then
        echo "[ERROR] Required file not found: $required_file" >&2
        exit 2
    fi
done

if truthy "$REPAIR_LOAD_SIMNIBS_MODULE"; then
    module purge
    module use "$REPAIR_MODULE_DIR"
    module load "$REPAIR_SIMNIBS_MODULE"
    export PYTHONNOUSERSITE=1
fi

export TI_REPAIR_JOB_ARRAY_SCRIPT="$REPAIR_ARRAY_SCRIPT"
export TI_SIM_RUNNER_PY="$REPAIR_SIM_RUNNER_PY"
export TI_COMPLETION_CHECK_PY="$REPAIR_COMPLETION_CHECK_PY"
export TI_ARRAY_MAX_CONCURRENT="$REPAIR_ARRAY_MAX_CONCURRENT"
export TI_MESH_TIMEOUT_HOURS="$REPAIR_MESH_TIMEOUT_HOURS"
export TI_MESH_MAX_RETRIES="$REPAIR_MESH_MAX_RETRIES"
export TI_REPAIR_LOG_DIR="$REPAIR_LOG_DIR"
export TI_REPAIR_STATE_DIR="$REPAIR_STATE_DIR"

cmd=(
    "$REPAIR_PYTHON"
    "$REPAIR_PLANNER_PY"
    --stage "$REPAIR_STAGE"
    --batch-root "$REPAIR_BATCH_ROOT"
    --dataset-glob "$REPAIR_DATASET_GLOB"
    --subject-glob "$REPAIR_SUBJECT_GLOB"
    --min-bytes "$REPAIR_MIN_BYTES"
    --submit-script "$REPAIR_SUBMIT_SCRIPT"
)

if [ -n "$REPAIR_EXPECTED_REPEAT_COUNT" ]; then
    cmd+=(--expected-repeat-count "$REPAIR_EXPECTED_REPEAT_COUNT")
fi

if [ -n "$REPAIR_EXPECTED_SUBJECTS_FILE" ]; then
    cmd+=(--expected-subjects-file "$REPAIR_EXPECTED_SUBJECTS_FILE")
fi

if [ -n "$REPAIR_OUT_DIR" ]; then
    cmd+=(--out-dir "$REPAIR_OUT_DIR")
fi

if [ -n "$REPAIR_REPEATS" ]; then
    read -r -a repeat_args <<< "$REPAIR_REPEATS"
    cmd+=(--repeats "${repeat_args[@]}")
fi

if ! truthy "$REPAIR_CHECK_NIFTI"; then
    cmd+=(--skip-nifti-load)
fi

if truthy "$REPAIR_SKIP_INPUT_CHECK"; then
    cmd+=(--skip-input-check)
fi

if truthy "$REPAIR_DRY_RUN"; then
    cmd+=(--dry-run)
fi

echo "[INFO] Repair launcher configuration:"
echo "[INFO]   batch root:          $REPAIR_BATCH_ROOT"
echo "[INFO]   stage:               $REPAIR_STAGE"
echo "[INFO]   repeats:             ${REPAIR_REPEATS:-all}"
echo "[INFO]   dataset glob:        $REPAIR_DATASET_GLOB"
echo "[INFO]   expected repeats:    ${REPAIR_EXPECTED_REPEAT_COUNT:-disabled}"
echo "[INFO]   out dir:             ${REPAIR_OUT_DIR:-<batch-root>/simulation_repair_plan}"
echo "[INFO]   dry run:             $REPAIR_DRY_RUN"
echo "[INFO]   array max concurrent:$REPAIR_ARRAY_MAX_CONCURRENT"
echo "[INFO]   mesh timeout hours:  $REPAIR_MESH_TIMEOUT_HOURS"
echo "[INFO]   mesh max retries:    $REPAIR_MESH_MAX_RETRIES"
echo "[INFO]   repair array script: $REPAIR_ARRAY_SCRIPT"
echo "[INFO]   simulation runner:   $REPAIR_SIM_RUNNER_PY"
echo "[INFO]   completion checker:  $REPAIR_COMPLETION_CHECK_PY"
echo "[INFO] Running planner:"
printf '[INFO]   %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
