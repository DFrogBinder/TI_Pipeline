#!/bin/bash
#SBATCH --job-name=ti_repair_array
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-repair-%A_%a.out
#SBATCH --requeue

set -o pipefail

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "[ERROR] This script must be submitted as a Slurm array."
    echo "[ERROR] Use submit_repair_jobArray.sh or pass --array to sbatch."
    exit 1
fi

REPAIR_PLAN_FILE="${TI_REPAIR_PLAN_FILE:-/users/cop23bi/scripts/simulation_repair_plan/repair_plan.tsv}"
SIM_RUNNER_PY="${TI_SIM_RUNNER_PY:-/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/simulation/TI_runner_multi-core.py}"
COMPLETION_CHECK_PY="${TI_COMPLETION_CHECK_PY:-/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/simulation/validate_simulation_outputs.py}"
MONTAGE_PRESET="${TI_MONTAGE_PRESET:-auto}"
MESH_TIMEOUT_HOURS="${TI_MESH_TIMEOUT_HOURS:-4}"
# 0 or a negative value means retry indefinitely. The retry-state file below
# makes Slurm requeues behave like a while loop until final validation passes.
MESH_MAX_RETRIES="${TI_MESH_MAX_RETRIES:-0}"
MESH_TIMEOUT_EXIT_CODE=124
SIM_INCOMPLETE_EXIT_CODE=125
SIM_INPUT_EXIT_CODE=126

if ! [[ "$MESH_MAX_RETRIES" =~ ^-?[0-9]+$ ]]; then
    echo "[ERROR] TI_MESH_MAX_RETRIES must be an integer; got '${MESH_MAX_RETRIES}'." >&2
    exit 2
fi

if [ "$MESH_MAX_RETRIES" -gt 0 ]; then
    MESH_RETRY_LIMIT_LABEL="$MESH_MAX_RETRIES"
else
    MESH_RETRY_LIMIT_LABEL="unlimited"
fi

if [ ! -f "$REPAIR_PLAN_FILE" ]; then
    echo "[ERROR] Repair plan file not found: ${REPAIR_PLAN_FILE}" >&2
    exit 1
fi

module purge
module use "$HOME/modules"
module load SimNIBS/4.0.1-foss-2023a

# Avoid user site-packages overriding the cluster env
export PYTHONNOUSERSITE=1

# --- Runtime shims (NumPy alias + disable SimNIBS Gmsh visualization) ---
SHIM_DIR="${SLURM_TMPDIR:-/tmp}/${USER}_simnibs_shim_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SHIM_DIR"

cat > "${SHIM_DIR}/sitecustomize.py" << 'EOF'
# Auto-imported by Python at startup if on PYTHONPATH.
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
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

STATE_DIR="${TI_REPAIR_STATE_DIR:-/users/cop23bi/scripts/logs/repair_retry_state}"
LOG_DIR="${TI_REPAIR_LOG_DIR:-/users/cop23bi/scripts/logs/repair}"
mkdir -p "$STATE_DIR" "$LOG_DIR"

TASK_KEY="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID}"
RETRY_FILE="${STATE_DIR}/${TASK_KEY}.retry"
TASK_FILE="${STATE_DIR}/${TASK_KEY}.task"

if [ -f "$RETRY_FILE" ]; then
    MESH_RETRY_COUNT=$(tr -d '[:space:]' < "$RETRY_FILE")
else
    MESH_RETRY_COUNT=0
fi

if ! [[ "$MESH_RETRY_COUNT" =~ ^[0-9]+$ ]]; then
    echo "[WARN] Invalid retry count in ${RETRY_FILE}; resetting to 0."
    MESH_RETRY_COUNT=0
fi

if [ -f "$TASK_FILE" ]; then
    TASK_LINE=$(sed -n '1p' "$TASK_FILE")
else
    PLAN_LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 2))
    TASK_LINE=$(sed -n "${PLAN_LINE_NUMBER}p" "$REPAIR_PLAN_FILE")
    if [ -z "$TASK_LINE" ]; then
        echo "[ERROR] No repair task found for array index ${SLURM_ARRAY_TASK_ID} in ${REPAIR_PLAN_FILE}."
        exit 1
    fi
    printf '%s\n' "$TASK_LINE" > "$TASK_FILE"
fi

IFS=$'\t' read -r TASK_ID DATASET_NAME REPEAT_ID TASK_ROOT SUBJECT MISSING_OUTPUTS FAILURE_REASONS <<< "$TASK_LINE"

if [ -z "$TASK_ROOT" ] || [ -z "$SUBJECT" ]; then
    echo "[ERROR] Malformed repair plan row for array index ${SLURM_ARRAY_TASK_ID}: ${TASK_LINE}" >&2
    exit 1
fi

echo "[INFO] Job ID:       ${SLURM_JOB_ID}"
echo "[INFO] Array index:  ${SLURM_ARRAY_TASK_ID}"
echo "[INFO] Task ID:      ${TASK_ID}"
echo "[INFO] Dataset:      ${DATASET_NAME}"
echo "[INFO] Repeat ID:    ${REPEAT_ID}"
echo "[INFO] Subject ID:   ${SUBJECT}"
echo "[INFO] Simulation root: ${TASK_ROOT}"
echo "[INFO] Repair plan:  ${REPAIR_PLAN_FILE}"
echo "[INFO] Missing outputs: ${MISSING_OUTPUTS}"
echo "[INFO] Montage preset: ${MONTAGE_PRESET}"
echo "[INFO] Completion check: ${COMPLETION_CHECK_PY}"
echo "[INFO] Mesh timeout: ${MESH_TIMEOUT_HOURS} hour(s) total across meshing/remeshing"
echo "[INFO] Simulation retry: ${MESH_RETRY_COUNT}/${MESH_RETRY_LIMIT_LABEL}"

SAFE_DATASET_NAME="${DATASET_NAME//[^A-Za-z0-9_.-]/_}"
LOG_FILE="${LOG_DIR}/${SAFE_DATASET_NAME}__${SUBJECT}.log"
SKIPPED_INPUTS_LOG="${LOG_DIR}/repair_tasks_skipped_missing_inputs.tsv"

requeue_or_fail() {
    local reason="$1"
    local final_exit_code="$2"
    local next_retry_count

    next_retry_count=$((MESH_RETRY_COUNT + 1))

    if [ "$MESH_MAX_RETRIES" -le 0 ] || [ "$MESH_RETRY_COUNT" -lt "$MESH_MAX_RETRIES" ]; then
        printf '%s\n' "$next_retry_count" > "$RETRY_FILE"
        echo "[WARN] ${reason}; requeueing ${SUBJECT} in ${DATASET_NAME} at array index ${SLURM_ARRAY_TASK_ID} (retry ${next_retry_count}/${MESH_RETRY_LIMIT_LABEL})."

        REQUEUE_OUTPUT=$(scontrol requeue "${SLURM_JOB_ID}" 2>&1)
        REQUEUE_EXIT=$?
        echo "$REQUEUE_OUTPUT"

        if [ "$REQUEUE_EXIT" -ne 0 ]; then
            echo "[ERROR] Failed to requeue ${SUBJECT} in ${DATASET_NAME}."
            exit "$REQUEUE_EXIT"
        fi

        exit 0
    fi

    echo "[ERROR] Simulation retries exhausted for ${SUBJECT} in ${DATASET_NAME} after ${MESH_RETRY_COUNT} retry/retries."
    echo "[ERROR] Last incomplete reason: ${reason}"
    exit "$final_exit_code"
}

TI_SIM_ROOT="$TASK_ROOT" python -u "$SIM_RUNNER_PY" \
    --subject "$SUBJECT" \
    --montage-preset "$MONTAGE_PRESET" \
    --mesh-timeout-hours "$MESH_TIMEOUT_HOURS" 2>&1 | tee -a "$LOG_FILE"
PY_EXIT=$?

if [ "$PY_EXIT" -eq "$SIM_INPUT_EXIT_CODE" ]; then
    echo "[WARN] Required simulation inputs are missing for ${SUBJECT} in ${DATASET_NAME}; dropping task without requeue."
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -Is)" \
        "${DATASET_NAME}" \
        "${TASK_ROOT}" \
        "${SUBJECT}" \
        "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}" \
        "${SLURM_ARRAY_TASK_ID}" >> "$SKIPPED_INPUTS_LOG"
    rm -f "$RETRY_FILE" "$TASK_FILE"
    exit 0
fi

if [ -f "$COMPLETION_CHECK_PY" ]; then
    python -u "$COMPLETION_CHECK_PY" \
        --root "$TASK_ROOT" \
        --subject "$SUBJECT" 2>&1 | tee -a "$LOG_FILE"
    VALIDATION_EXIT=$?
else
    echo "[ERROR] Completion check script not found: ${COMPLETION_CHECK_PY}" | tee -a "$LOG_FILE"
    VALIDATION_EXIT=127
fi

if [ "$PY_EXIT" -eq 0 ] && [ "$VALIDATION_EXIT" -eq 0 ]; then
    rm -f "$RETRY_FILE" "$TASK_FILE"
    exit 0
fi

if [ "$PY_EXIT" -ne 0 ] && [ "$VALIDATION_EXIT" -eq 0 ]; then
    requeue_or_fail "Runner exited ${PY_EXIT}; validation passed but a clean runner exit is required for ${SUBJECT} in ${DATASET_NAME}" "$PY_EXIT"
fi

if [ "$PY_EXIT" -eq "$MESH_TIMEOUT_EXIT_CODE" ]; then
    requeue_or_fail "Mesh timeout detected and final simulation outputs are incomplete for ${SUBJECT} in ${DATASET_NAME}" "$PY_EXIT"
fi

if [ "$PY_EXIT" -ne 0 ]; then
    requeue_or_fail "Runner exited ${PY_EXIT} and final simulation outputs are incomplete for ${SUBJECT} in ${DATASET_NAME}" "$PY_EXIT"
fi

requeue_or_fail "Runner exited 0 but final simulation output validation failed for ${SUBJECT} in ${DATASET_NAME}" "$SIM_INCOMPLETE_EXIT_CODE"
