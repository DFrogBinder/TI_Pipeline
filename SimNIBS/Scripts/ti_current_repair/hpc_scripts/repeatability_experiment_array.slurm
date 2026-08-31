#!/bin/bash
# Fallback defaults only. Normal submissions should edit controls in
# hpc_scripts/submit_repeatability_experiment.sh, which overrides these values
# with sbatch command-line options.
#SBATCH --job-name=ti_repeat_experiment
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-0
#SBATCH --output=slurm-%A_%a.out
#SBATCH --requeue

module purge
module use "$HOME/modules"
module load SimNIBS/4.0.1-foss-2023a

export PYTHONNOUSERSITE=1

SHIM_DIR="${SLURM_TMPDIR:-/tmp}/${USER}_simnibs_shim_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SHIM_DIR"

cat > "${SHIM_DIR}/sitecustomize.py" << 'EOF'
try:
    import numpy as _np
    if not hasattr(_np, "bool"):
        _np.bool = _np.bool_
except Exception:
    pass

import builtins
_real_import = builtins.__import__

def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _real_import(name, globals, locals, fromlist, level)
    try:
        if name == "simnibs" or name.startswith("simnibs."):
            from simnibs import sim_struct as _sim_struct
            if not getattr(_sim_struct.SESSION, "_gmsh_disabled", False):
                _orig_init = _sim_struct.SESSION.__init__
                def _new_init(self, *a, **k):
                    _orig_init(self, *a, **k)
                    try:
                        self.open_in_gmsh = False
                    except Exception:
                        pass
                _sim_struct.SESSION.__init__ = _new_init
                _sim_struct.SESSION._gmsh_disabled = True
    except Exception:
        pass
    return mod

builtins.__import__ = _patched_import
EOF

export PYTHONPATH="${SHIM_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

set -euo pipefail

MESH_TIMEOUT_HOURS="${TI_MESH_TIMEOUT_HOURS:-4}"
MESH_MAX_RETRIES="${TI_MESH_MAX_RETRIES:-0}"
TASK_OFFSET="${TASK_OFFSET:-0}"
MESH_TIMEOUT_EXIT_CODE=124
SIM_INCOMPLETE_EXIT_CODE=125
SIM_INPUT_EXIT_CODE=126

if ! [[ "${TASK_OFFSET}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] TASK_OFFSET must be a non-negative integer; got '${TASK_OFFSET}'." >&2
    exit 2
fi
TASK_INDEX=$((TASK_OFFSET + SLURM_ARRAY_TASK_ID))

if ! [[ "${MESH_MAX_RETRIES}" =~ ^-?[0-9]+$ ]]; then
    echo "[ERROR] TI_MESH_MAX_RETRIES must be an integer; got '${MESH_MAX_RETRIES}'." >&2
    exit 2
fi

if [ "${MESH_MAX_RETRIES}" -gt 0 ]; then
    MESH_RETRY_LIMIT_LABEL="${MESH_MAX_RETRIES}"
else
    MESH_RETRY_LIMIT_LABEL="unlimited"
fi

PIPELINE_DIR_DEFAULT="/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/ti_current_repair"
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

LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/repeatability_experiment_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
STATE_DIR="${LOG_DIR}/retry_state"
mkdir -p "$STATE_DIR"
TASK_KEY="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID}"
RETRY_FILE="${STATE_DIR}/${TASK_KEY}.retry"
SKIPPED_INPUTS_LOG="${LOG_DIR}/tasks_skipped_missing_inputs.tsv"

if [ -f "$RETRY_FILE" ]; then
    MESH_RETRY_COUNT=$(tr -d '[:space:]' < "$RETRY_FILE")
else
    MESH_RETRY_COUNT=0
fi

if ! [[ "${MESH_RETRY_COUNT}" =~ ^[0-9]+$ ]]; then
    echo "[WARN] Invalid retry count in ${RETRY_FILE}; resetting to 0."
    MESH_RETRY_COUNT=0
fi

COMPLETION_CHECK_PY="${COMPLETION_CHECK_PY:-${SCRIPT_DIR}/validate_repeatability_task.py}"

EXTRA_ARGS=()
if [ "${OVERWRITE_OUTPUT:-0}" = "1" ]; then
    EXTRA_ARGS+=(--overwrite)
fi
if [ "${FORCE_MESH:-0}" = "1" ]; then
    EXTRA_ARGS+=(--force-mesh)
fi
if [ "${MESH_RETRY_COUNT}" -gt 0 ] && [ "${OVERWRITE_OUTPUT:-0}" != "1" ]; then
    EXTRA_ARGS+=(--overwrite)
fi

echo "[INFO] Pipeline root:     ${SCRIPT_DIR}"
echo "[INFO] Experiment config: ${EXPERIMENT_CONFIG}"
echo "[INFO] Local array index: ${SLURM_ARRAY_TASK_ID}"
echo "[INFO] Task offset:       ${TASK_OFFSET}"
echo "[INFO] Global task index: ${TASK_INDEX}"
echo "[INFO] CPUs:              ${SLURM_CPUS_PER_TASK}"
echo "[INFO] Log file:          ${LOG_FILE}"
echo "[INFO] Completion check:  ${COMPLETION_CHECK_PY}"
echo "[INFO] Mesh timeout:      ${MESH_TIMEOUT_HOURS} hour(s)"
echo "[INFO] Task retry:        ${MESH_RETRY_COUNT}/${MESH_RETRY_LIMIT_LABEL}"

requeue_or_fail() {
    local reason="$1"
    local final_exit_code="$2"
    local next_retry_count

    next_retry_count=$((MESH_RETRY_COUNT + 1))

    if [ "$MESH_MAX_RETRIES" -le 0 ] || [ "$MESH_RETRY_COUNT" -lt "$MESH_MAX_RETRIES" ]; then
        printf '%s\n' "$next_retry_count" > "$RETRY_FILE"
        echo "[WARN] ${reason}; requeueing global task index ${TASK_INDEX} (retry ${next_retry_count}/${MESH_RETRY_LIMIT_LABEL})."

        set +e
        REQUEUE_OUTPUT=$(scontrol requeue "${SLURM_JOB_ID}" 2>&1)
        REQUEUE_EXIT=$?
        set -e
        echo "$REQUEUE_OUTPUT"

        if [ "$REQUEUE_EXIT" -ne 0 ]; then
            echo "[ERROR] Failed to requeue global task index ${TASK_INDEX}."
            exit "$REQUEUE_EXIT"
        fi

        exit 0
    fi

    echo "[ERROR] Task retries exhausted for global task index ${TASK_INDEX} after ${MESH_RETRY_COUNT} retry/retries."
    echo "[ERROR] Last incomplete reason: ${reason}"
    exit "$final_exit_code"
}

PYTHON_CMD=(
    python -u "${SCRIPT_DIR}/simulation_runners/repeatability_experiment.py"
    run-task
    --config "${EXPERIMENT_CONFIG}"
    --task-index "${TASK_INDEX}"
)
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    PYTHON_CMD+=("${EXTRA_ARGS[@]}")
fi

set +e
timeout --kill-after=30s "${MESH_TIMEOUT_HOURS}h" "${PYTHON_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
PY_EXIT=${PIPESTATUS[0]}
set -e

if [ "$PY_EXIT" -eq "$SIM_INPUT_EXIT_CODE" ]; then
    echo "[WARN] Required simulation inputs are missing for global task index ${TASK_INDEX}; dropping task without requeue."
    printf '%s\t%s\t%s\t%s\n' \
        "$(date -Is)" \
        "${TASK_INDEX}" \
        "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}" \
        "${PY_EXIT}" >> "$SKIPPED_INPUTS_LOG"
    rm -f "$RETRY_FILE"
    exit 0
fi

if [ -f "$COMPLETION_CHECK_PY" ]; then
    set +e
    python -u "$COMPLETION_CHECK_PY" \
        --config "$EXPERIMENT_CONFIG" \
        --task-index "$TASK_INDEX" 2>&1 | tee -a "$LOG_FILE"
    VALIDATION_EXIT=${PIPESTATUS[0]}
    set -e
else
    echo "[ERROR] Completion check script not found: ${COMPLETION_CHECK_PY}" | tee -a "$LOG_FILE"
    VALIDATION_EXIT=127
fi

if [ "$PY_EXIT" -eq 0 ] && [ "$VALIDATION_EXIT" -eq 0 ]; then
    rm -f "$RETRY_FILE"
    exit 0
fi

if [ "$PY_EXIT" -ne 0 ] && [ "$VALIDATION_EXIT" -eq 0 ]; then
    requeue_or_fail "Runner exited ${PY_EXIT}; validation passed but a clean runner exit is required" "$PY_EXIT"
fi

if [ "$PY_EXIT" -eq "$MESH_TIMEOUT_EXIT_CODE" ]; then
    requeue_or_fail "Mesh/simulation timeout detected and final task outputs are incomplete" "$PY_EXIT"
fi

if [ "$PY_EXIT" -ne 0 ]; then
    requeue_or_fail "Runner exited ${PY_EXIT} and final task outputs are incomplete" "$PY_EXIT"
fi

requeue_or_fail "Runner exited 0 but final task output validation failed" "$SIM_INCOMPLETE_EXIT_CODE"
