#!/bin/bash
#SBATCH --job-name=ti_array
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8             # per-subject threads (tune if needed)
#SBATCH --mem=32G                     # per-subject RAM (tune if needed)
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%A_%a.out
#SBATCH --requeue

set -o pipefail

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "[ERROR] This script must be submitted as a Slurm array."
    echo "[ERROR] Use submit_my_jobArray.sh or pass --array to sbatch."
    exit 1
fi

# -------------------------
# 1) Environment
# -------------------------

MESH_TIMEOUT_HOURS="${TI_MESH_TIMEOUT_HOURS:-4}"
# 0 or a negative value means retry indefinitely. The retry-state file below
# makes Slurm requeues behave like a while loop until final validation passes.
MESH_MAX_RETRIES="${TI_MESH_MAX_RETRIES:-0}"
MESH_TIMEOUT_EXIT_CODE=124
SIM_INCOMPLETE_EXIT_CODE=125
SIM_INPUT_EXIT_CODE=126
SIM_ROOT="${TI_SIM_ROOT:-/mnt/parscratch/users/cop23bi/LM1}"
COMPLETION_CHECK_PY="${TI_COMPLETION_CHECK_PY:-/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/simulation/validate_simulation_outputs.py}"
MONTAGE_PRESET="${TI_MONTAGE_PRESET:-left-m1}"

if ! [[ "$MESH_MAX_RETRIES" =~ ^-?[0-9]+$ ]]; then
    echo "[ERROR] TI_MESH_MAX_RETRIES must be an integer; got '${MESH_MAX_RETRIES}'." >&2
    exit 2
fi

if [ "$MESH_MAX_RETRIES" -gt 0 ]; then
    MESH_RETRY_LIMIT_LABEL="$MESH_MAX_RETRIES"
else
    MESH_RETRY_LIMIT_LABEL="unlimited"
fi

module purge
module use "$HOME/modules"
module load SimNIBS/4.0.1-foss-2023a

# Avoid user site-packages overriding the cluster env
export PYTHONNOUSERSITE=1

# --- FIX: runtime shims (NumPy alias + disable SimNIBS Gmsh visualization) ---
# This avoids:
#   1) SimNIBS 4.0.1 crash with NumPy >= 1.24 (np.bool removal)
#   2) bundled gmsh crash on older glibc nodes (GLIBC_2.23 not found)
SHIM_DIR="${SLURM_TMPDIR:-/tmp}/${USER}_simnibs_shim_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SHIM_DIR"

cat > "${SHIM_DIR}/sitecustomize.py" << 'EOF'
# Auto-imported by Python at startup if on PYTHONPATH.

# (1) NumPy compatibility: restore np.bool if missing (NumPy >= 1.24)
try:
    import numpy as _np
    if not hasattr(_np, "bool"):
        _np.bool = _np.bool_
except Exception:
    pass

# (2) Disable SimNIBS auto-opening/visualizations in Gmsh (avoids bundled gmsh binary)
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
# --- END FIX ---

# Make SimNIBS/OpenMP respect the Slurm allocation
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# -------------------------
# 2) Select subject from list
# -------------------------

cd /users/cop23bi/scripts || exit

SUBJECTS_FILE="${TI_SUBJECTS_FILE:-subjects.txt}"
STATE_DIR="/users/cop23bi/scripts/logs/retry_state"
mkdir -p "$STATE_DIR"

TASK_KEY="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID}"
RETRY_FILE="${STATE_DIR}/${TASK_KEY}.retry"
SUBJECT_FILE="${STATE_DIR}/${TASK_KEY}.subject"

if [ -f "$RETRY_FILE" ]; then
    MESH_RETRY_COUNT=$(tr -d '[:space:]' < "$RETRY_FILE")
else
    MESH_RETRY_COUNT=0
fi

if ! [[ "$MESH_RETRY_COUNT" =~ ^[0-9]+$ ]]; then
    echo "[WARN] Invalid retry count in ${RETRY_FILE}; resetting to 0."
    MESH_RETRY_COUNT=0
fi

# SLURM_ARRAY_TASK_ID is 0-based; sed is 1-based
if [ -f "$SUBJECT_FILE" ]; then
    SUBJECT=$(cat "$SUBJECT_FILE")
else
    SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SUBJECTS_FILE")
    if [ -n "$SUBJECT" ]; then
        printf '%s\n' "$SUBJECT" > "$SUBJECT_FILE"
    fi
fi

if [ -z "$SUBJECT" ]; then
    echo "[ERROR] No subject found for array index ${SLURM_ARRAY_TASK_ID} in ${SUBJECTS_FILE}."
    exit 1
fi

echo "[INFO] Job ID:      ${SLURM_JOB_ID}"
echo "[INFO] Array index: ${SLURM_ARRAY_TASK_ID}"
echo "[INFO] Subject ID:  ${SUBJECT}"
echo "[INFO] Subjects file:${SUBJECTS_FILE}"
echo "[INFO] CPUs:        ${SLURM_CPUS_PER_TASK}"
echo "[INFO] Mem per task:${SLURM_MEM_PER_NODE:-${SLURM_MEM_PER_TASK}} MB"
echo "[INFO] Simulation root: ${SIM_ROOT}"
echo "[INFO] Completion check: ${COMPLETION_CHECK_PY}"
echo "[INFO] Mesh timeout: ${MESH_TIMEOUT_HOURS} hour(s) total across meshing/remeshing"
echo "[INFO] Montage preset: ${MONTAGE_PRESET}"
echo "[INFO] Simulation retry: ${MESH_RETRY_COUNT}/${MESH_RETRY_LIMIT_LABEL}"

# -------------------------
# 3) Run TI pipeline for ONE subject
# -------------------------

LOG_DIR="/users/cop23bi/scripts/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${SUBJECT}.log"
SKIPPED_INPUTS_LOG="${LOG_DIR}/subjects_skipped_missing_inputs.tsv"

requeue_or_fail() {
    local reason="$1"
    local final_exit_code="$2"
    local next_retry_count

    next_retry_count=$((MESH_RETRY_COUNT + 1))

    if [ "$MESH_MAX_RETRIES" -le 0 ] || [ "$MESH_RETRY_COUNT" -lt "$MESH_MAX_RETRIES" ]; then
        printf '%s\n' "$next_retry_count" > "$RETRY_FILE"
        echo "[WARN] ${reason}; requeueing ${SUBJECT} at array index ${SLURM_ARRAY_TASK_ID} (retry ${next_retry_count}/${MESH_RETRY_LIMIT_LABEL})."

        REQUEUE_OUTPUT=$(scontrol requeue "${SLURM_JOB_ID}" 2>&1)
        REQUEUE_EXIT=$?
        echo "$REQUEUE_OUTPUT"

        if [ "$REQUEUE_EXIT" -ne 0 ]; then
            echo "[ERROR] Failed to requeue ${SUBJECT}."
            exit "$REQUEUE_EXIT"
        fi

        exit 0
    fi

    echo "[ERROR] Simulation retries exhausted for ${SUBJECT} after ${MESH_RETRY_COUNT} retry/retries."
    echo "[ERROR] Last incomplete reason: ${reason}"
    exit "$final_exit_code"
}

python -u /users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/simulation/TI_runner_multi-core.py \
    --subject "$SUBJECT" \
    --montage-preset "$MONTAGE_PRESET" \
    --reuse-existing-mesh \
    --mesh-timeout-hours "$MESH_TIMEOUT_HOURS" 2>&1 | tee -a "$LOG_FILE"
PY_EXIT=$?

if [ "$PY_EXIT" -eq "$SIM_INPUT_EXIT_CODE" ]; then
    echo "[WARN] Required simulation inputs are missing for ${SUBJECT}; dropping subject without requeue."
    printf '%s\t%s\t%s\t%s\n' \
        "$(date -Is)" \
        "${SUBJECT}" \
        "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}" \
        "${SLURM_ARRAY_TASK_ID}" >> "$SKIPPED_INPUTS_LOG"
    rm -f "$RETRY_FILE" "$SUBJECT_FILE"
    exit 0
fi

if [ -f "$COMPLETION_CHECK_PY" ]; then
    python -u "$COMPLETION_CHECK_PY" \
        --root "$SIM_ROOT" \
        --subject "$SUBJECT" 2>&1 | tee -a "$LOG_FILE"
    VALIDATION_EXIT=$?
else
    echo "[ERROR] Completion check script not found: ${COMPLETION_CHECK_PY}" | tee -a "$LOG_FILE"
    VALIDATION_EXIT=127
fi

if [ "$PY_EXIT" -eq 0 ] && [ "$VALIDATION_EXIT" -eq 0 ]; then
    rm -f "$RETRY_FILE" "$SUBJECT_FILE"
    exit 0
fi

if [ "$PY_EXIT" -ne 0 ] && [ "$VALIDATION_EXIT" -eq 0 ]; then
    requeue_or_fail "Runner exited ${PY_EXIT}; validation passed but a clean runner exit is required for ${SUBJECT}" "$PY_EXIT"
fi

if [ "$PY_EXIT" -eq "$MESH_TIMEOUT_EXIT_CODE" ]; then
    requeue_or_fail "Mesh timeout detected and final simulation outputs are incomplete for ${SUBJECT}" "$PY_EXIT"
fi

if [ "$PY_EXIT" -ne 0 ]; then
    requeue_or_fail "Runner exited ${PY_EXIT} and final simulation outputs are incomplete for ${SUBJECT}" "$PY_EXIT"
fi

requeue_or_fail "Runner exited 0 but final simulation output validation failed for ${SUBJECT}" "$SIM_INCOMPLETE_EXIT_CODE"
