#!/bin/bash
#SBATCH --job-name=merge_mesh_trial
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-merge-mesh-%A_%a.out
#SBATCH --requeue

set -o pipefail

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "[ERROR] This script must be submitted as a Slurm array."
    echo "[ERROR] Use submit_merge_mesh_trial_array.sh or pass --array to sbatch."
    exit 1
fi

SUBJECTS_FILE="${TI_MERGE_TRIAL_SUBJECTS_FILE:-/users/cop23bi/scripts/merge_mesh_trial_subjects.txt}"
SOURCE_ROOT="${TI_MERGE_TRIAL_SOURCE_ROOT:-/mnt/parscratch/users/cop23bi/ZIPs/Analised-Data/Left_Hippocampus_Runs/Left_Hippocampus_Data_01}"
OUT_ROOT="${TI_MERGE_TRIAL_OUT_ROOT:-/mnt/parscratch/users/cop23bi/merge_mesh_trials/Left_Hippocampus_Data_01}"
RUNNER_PY="${TI_MERGE_TRIAL_RUNNER_PY:-/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/segmentation_merge_trials/run_merge_mesh_trial.py}"
VALIDATOR_PY="${TI_MERGE_TRIAL_VALIDATOR_PY:-/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/segmentation_merge_trials/validate_merge_mesh_trial.py}"
TRIAL_TIMEOUT_HOURS="${TI_MERGE_TRIAL_TIMEOUT_HOURS:-7}"
TRIAL_MAX_RETRIES="${TI_MERGE_TRIAL_MAX_RETRIES:-0}"

TIMEOUT_EXIT_CODE=124
INCOMPLETE_EXIT_CODE=125
INPUT_EXIT_CODE=126

if ! [[ "$TRIAL_MAX_RETRIES" =~ ^-?[0-9]+$ ]]; then
    echo "[ERROR] TI_MERGE_TRIAL_MAX_RETRIES must be an integer; got '${TRIAL_MAX_RETRIES}'." >&2
    exit 2
fi

if [ "$TRIAL_MAX_RETRIES" -gt 0 ]; then
    TRIAL_RETRY_LIMIT_LABEL="$TRIAL_MAX_RETRIES"
else
    TRIAL_RETRY_LIMIT_LABEL="unlimited"
fi

if [ ! -f "$SUBJECTS_FILE" ]; then
    echo "[ERROR] Subjects file not found: ${SUBJECTS_FILE}" >&2
    exit 1
fi

module purge
module use "$HOME/modules"
module load SimNIBS/4.0.1-foss-2023a

export PYTHONNOUSERSITE=1

# Runtime shims copied from the existing CamCan arrays:
#   1. Restore np.bool for SimNIBS 4.0.1 with newer NumPy.
#   2. Disable SimNIBS attempts to auto-open Gmsh visualizations.
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
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

STATE_DIR="${TI_MERGE_TRIAL_STATE_DIR:-/users/cop23bi/scripts/logs/merge_mesh_trial_retry_state}"
LOG_DIR="${TI_MERGE_TRIAL_LOG_DIR:-/users/cop23bi/scripts/logs/merge_mesh_trial}"
mkdir -p "$STATE_DIR" "$LOG_DIR"

TASK_KEY="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID}"
RETRY_FILE="${STATE_DIR}/${TASK_KEY}.retry"
SUBJECT_FILE="${STATE_DIR}/${TASK_KEY}.subject"

if [ -f "$RETRY_FILE" ]; then
    TRIAL_RETRY_COUNT=$(tr -d '[:space:]' < "$RETRY_FILE")
else
    TRIAL_RETRY_COUNT=0
fi

if ! [[ "$TRIAL_RETRY_COUNT" =~ ^[0-9]+$ ]]; then
    echo "[WARN] Invalid retry count in ${RETRY_FILE}; resetting to 0."
    TRIAL_RETRY_COUNT=0
fi

if [ -f "$SUBJECT_FILE" ]; then
    SUBJECT=$(cat "$SUBJECT_FILE")
else
    SUBJECT=$(awk -v target="$((SLURM_ARRAY_TASK_ID + 1))" 'NF > 0 && $1 !~ /^#/ { count++; if (count == target) { print $1; exit } }' "$SUBJECTS_FILE")
    if [ -n "$SUBJECT" ]; then
        printf '%s\n' "$SUBJECT" > "$SUBJECT_FILE"
    fi
fi

if [ -z "$SUBJECT" ]; then
    echo "[ERROR] No subject found for array index ${SLURM_ARRAY_TASK_ID} in ${SUBJECTS_FILE}."
    exit 1
fi

echo "[INFO] Job ID:        ${SLURM_JOB_ID}"
echo "[INFO] Array index:   ${SLURM_ARRAY_TASK_ID}"
echo "[INFO] Subject ID:    ${SUBJECT}"
echo "[INFO] Subjects file: ${SUBJECTS_FILE}"
echo "[INFO] Source root:   ${SOURCE_ROOT}"
echo "[INFO] Output root:   ${OUT_ROOT}"
echo "[INFO] Runner:        ${RUNNER_PY}"
echo "[INFO] Validator:     ${VALIDATOR_PY}"
echo "[INFO] Timeout:       ${TRIAL_TIMEOUT_HOURS} hour(s)"
echo "[INFO] Retry:         ${TRIAL_RETRY_COUNT}/${TRIAL_RETRY_LIMIT_LABEL}"

SAFE_SUBJECT="${SUBJECT//[^A-Za-z0-9_.-]/_}"
LOG_FILE="${LOG_DIR}/${SAFE_SUBJECT}.log"
SKIPPED_INPUTS_LOG="${LOG_DIR}/subjects_skipped_missing_inputs.tsv"

requeue_or_fail() {
    local reason="$1"
    local final_exit_code="$2"
    local next_retry_count

    next_retry_count=$((TRIAL_RETRY_COUNT + 1))

    if [ "$TRIAL_MAX_RETRIES" -le 0 ] || [ "$TRIAL_RETRY_COUNT" -lt "$TRIAL_MAX_RETRIES" ]; then
        printf '%s\n' "$next_retry_count" > "$RETRY_FILE"
        echo "[WARN] ${reason}; requeueing ${SUBJECT} at array index ${SLURM_ARRAY_TASK_ID} (retry ${next_retry_count}/${TRIAL_RETRY_LIMIT_LABEL})."

        REQUEUE_OUTPUT=$(scontrol requeue "${SLURM_JOB_ID}" 2>&1)
        REQUEUE_EXIT=$?
        echo "$REQUEUE_OUTPUT"

        if [ "$REQUEUE_EXIT" -ne 0 ]; then
            echo "[ERROR] Failed to requeue ${SUBJECT}."
            exit "$REQUEUE_EXIT"
        fi

        exit 0
    fi

    echo "[ERROR] Merge-mesh trial retries exhausted for ${SUBJECT} after ${TRIAL_RETRY_COUNT} retry/retries."
    echo "[ERROR] Last incomplete reason: ${reason}"
    exit "$final_exit_code"
}

python -u "$RUNNER_PY" \
    --source-root "$SOURCE_ROOT" \
    --out-root "$OUT_ROOT" \
    --subject "$SUBJECT" \
    --timeout-hours "$TRIAL_TIMEOUT_HOURS" 2>&1 | tee -a "$LOG_FILE"
PY_EXIT=${PIPESTATUS[0]}

if [ "$PY_EXIT" -eq "$INPUT_EXIT_CODE" ]; then
    echo "[WARN] Required merge-mesh inputs are missing for ${SUBJECT}; dropping subject without requeue."
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$(date -Is)" \
        "${SUBJECT}" \
        "${SOURCE_ROOT}" \
        "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}" \
        "${SLURM_ARRAY_TASK_ID}" >> "$SKIPPED_INPUTS_LOG"
    rm -f "$RETRY_FILE" "$SUBJECT_FILE"
    exit 0
fi

if [ -f "$VALIDATOR_PY" ]; then
    python -u "$VALIDATOR_PY" \
        --out-root "$OUT_ROOT" \
        --subject "$SUBJECT" 2>&1 | tee -a "$LOG_FILE"
    VALIDATION_EXIT=${PIPESTATUS[0]}
else
    echo "[ERROR] Validator not found: ${VALIDATOR_PY}" | tee -a "$LOG_FILE"
    VALIDATION_EXIT=127
fi

if [ "$PY_EXIT" -eq 0 ] && [ "$VALIDATION_EXIT" -eq 0 ]; then
    rm -f "$RETRY_FILE" "$SUBJECT_FILE"
    exit 0
fi

if [ "$PY_EXIT" -eq "$TIMEOUT_EXIT_CODE" ]; then
    requeue_or_fail "Runner timed out and trial validation is incomplete for ${SUBJECT}" "$PY_EXIT"
fi

if [ "$PY_EXIT" -ne 0 ]; then
    requeue_or_fail "Runner exited ${PY_EXIT} and trial validation is incomplete for ${SUBJECT}" "$PY_EXIT"
fi

requeue_or_fail "Runner exited 0 but trial validation failed for ${SUBJECT}" "$INCOMPLETE_EXIT_CODE"
