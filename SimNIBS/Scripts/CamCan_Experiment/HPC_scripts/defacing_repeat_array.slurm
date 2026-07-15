#!/bin/bash
#SBATCH --job-name=ti_defacing_repeat
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-0
#SBATCH --output=defacing-repeat-%A_%a.out
#SBATCH --requeue

set -o pipefail

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "[ERROR] Submit this script as a Slurm array."
    exit 1
fi

: "${TI_DEFACING_REPEAT_MANIFEST:?Missing TI_DEFACING_REPEAT_MANIFEST.}"
: "${TI_MONTAGE_PRESET:?Missing TI_MONTAGE_PRESET.}"
: "${TI_MONTAGE_VALIDATOR_PY:?Missing TI_MONTAGE_VALIDATOR_PY.}"
: "${TI_TARGETS_CSV:?Missing TI_TARGETS_CSV.}"
: "${TI_EXPECTED_TARGETS_SHA256:?Missing TI_EXPECTED_TARGETS_SHA256.}"
MONTAGE_PRESET="${TI_MONTAGE_PRESET}"

SIM_RUNNER_PY="${TI_SIM_RUNNER_PY:-/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/simulation/TI_runner_multi-core.py}"
COMPLETION_CHECK_PY="${TI_COMPLETION_CHECK_PY:-/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/simulation/validate_simulation_outputs.py}"
MESH_TIMEOUT_HOURS="${TI_MESH_TIMEOUT_HOURS:-4}"
TASK_MAX_RETRIES="${TI_MESH_MAX_RETRIES:-0}"
SIM_INCOMPLETE_EXIT_CODE=125
SIM_INPUT_EXIT_CODE=126

if ! [[ "${TASK_MAX_RETRIES}" =~ ^-?[0-9]+$ ]]; then
    echo "[ERROR] TI_MESH_MAX_RETRIES must be an integer; got '${TASK_MAX_RETRIES}'." >&2
    exit 2
fi

if [ "${TASK_MAX_RETRIES}" -gt 0 ]; then
    TASK_RETRY_LIMIT_LABEL="${TASK_MAX_RETRIES}"
else
    TASK_RETRY_LIMIT_LABEL="unlimited"
fi

TASK_OFFSET="${TASK_OFFSET:-0}"
if ! [[ "${TASK_OFFSET}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] TASK_OFFSET must be a non-negative integer; got '${TASK_OFFSET}'." >&2
    exit 2
fi
TASK_INDEX=$((TASK_OFFSET + SLURM_ARRAY_TASK_ID))

if [ ! -f "${TI_DEFACING_REPEAT_MANIFEST}" ]; then
    echo "[ERROR] Manifest not found: ${TI_DEFACING_REPEAT_MANIFEST}" >&2
    exit 1
fi

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

STATE_DIR="${TI_DEFACING_REPEAT_STATE_DIR:-/users/cop23bi/scripts/logs/defacing_repeat_retry_state}"
LOG_DIR="${LOG_DIR:-${TI_DEFACING_REPEAT_LOG_DIR:-/users/cop23bi/scripts/logs/defacing_repeat}}"
mkdir -p "$STATE_DIR" "$LOG_DIR"

TASK_KEY="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID}"
RETRY_FILE="${STATE_DIR}/${TASK_KEY}.retry"
TASK_FILE="${STATE_DIR}/${TASK_KEY}.task"

if [ -f "$RETRY_FILE" ]; then
    TASK_RETRY_COUNT=$(tr -d '[:space:]' < "$RETRY_FILE")
else
    TASK_RETRY_COUNT=0
fi

if ! [[ "${TASK_RETRY_COUNT}" =~ ^[0-9]+$ ]]; then
    echo "[WARN] Invalid retry count in ${RETRY_FILE}; resetting to 0."
    TASK_RETRY_COUNT=0
fi

if [ -f "$TASK_FILE" ]; then
    TASK_LINE=$(sed -n '1p' "$TASK_FILE")
else
    MANIFEST_LINE_NUMBER=$((TASK_INDEX + 2))
    TASK_LINE=$(sed -n "${MANIFEST_LINE_NUMBER}p" "$TI_DEFACING_REPEAT_MANIFEST")
    if [ -z "$TASK_LINE" ]; then
        echo "[ERROR] No manifest row found for global task index ${TASK_INDEX}." >&2
        exit 1
    fi
    printf '%s\n' "$TASK_LINE" > "$TASK_FILE"
fi

IFS=$'\t' read -r TASK_ID DATASET_NAME REPEAT_ID DATASET_ROOT SUBJECT ANAT_DIR STATUS MESSAGE <<< "$TASK_LINE"

if [ -z "${DATASET_ROOT:-}" ] || [ -z "${SUBJECT:-}" ]; then
    echo "[ERROR] Malformed manifest row for global task index ${TASK_INDEX}: ${TASK_LINE}" >&2
    exit 1
fi

SAFE_DATASET_NAME="${DATASET_NAME//[^A-Za-z0-9_.-]/_}"
LOG_FILE="${LOG_DIR}/${SAFE_DATASET_NAME}__${SUBJECT}.log"

echo "[INFO] Job ID:          ${SLURM_JOB_ID}"
echo "[INFO] Local index:     ${SLURM_ARRAY_TASK_ID}"
echo "[INFO] Task offset:     ${TASK_OFFSET}"
echo "[INFO] Global index:    ${TASK_INDEX}"
echo "[INFO] Task ID:         ${TASK_ID}"
echo "[INFO] Dataset:         ${DATASET_NAME}"
echo "[INFO] Repeat ID:       ${REPEAT_ID}"
echo "[INFO] Dataset root:    ${DATASET_ROOT}"
echo "[INFO] Subject ID:      ${SUBJECT}"
echo "[INFO] Anat dir:        ${ANAT_DIR}"
echo "[INFO] Status:          ${STATUS}"
echo "[INFO] Message:         ${MESSAGE}"
echo "[INFO] Montage preset:  ${MONTAGE_PRESET}"
echo "[INFO] Mesh timeout:    ${MESH_TIMEOUT_HOURS}"
echo "[INFO] Runner:          ${SIM_RUNNER_PY}"
echo "[INFO] Completion check:${COMPLETION_CHECK_PY}"
echo "[INFO] Log file:        ${LOG_FILE}"
echo "[INFO] Task retry:      ${TASK_RETRY_COUNT}/${TASK_RETRY_LIMIT_LABEL}"

if [ "${STATUS}" != "ready" ]; then
    echo "[WARN] Manifest row is not ready; skipping task ${TASK_ID}: ${MESSAGE}"
    rm -f "$RETRY_FILE" "$TASK_FILE"
    exit 0
fi

if ! python -u "${TI_MONTAGE_VALIDATOR_PY}" \
    --dataset-name "${DATASET_NAME}" \
    --preset "${MONTAGE_PRESET}" \
    --targets-csv "${TI_TARGETS_CSV}" \
    --expected-targets-sha256 "${TI_EXPECTED_TARGETS_SHA256}" 2>&1 | tee -a "${LOG_FILE}"; then
    echo "[ERROR] Dataset/montage/targets.csv validation failed; refusing simulation." >&2
    exit 2
fi

requeue_or_fail() {
    local reason="$1"
    local final_exit_code="$2"
    local next_retry_count

    next_retry_count=$((TASK_RETRY_COUNT + 1))

    if [ "$TASK_MAX_RETRIES" -le 0 ] || [ "$TASK_RETRY_COUNT" -lt "$TASK_MAX_RETRIES" ]; then
        printf '%s\n' "$next_retry_count" > "$RETRY_FILE"
        echo "[WARN] ${reason}; requeueing task ${TASK_ID} at global index ${TASK_INDEX} (retry ${next_retry_count}/${TASK_RETRY_LIMIT_LABEL})."

        REQUEUE_OUTPUT=$(scontrol requeue "${SLURM_JOB_ID}" 2>&1)
        REQUEUE_EXIT=$?
        echo "$REQUEUE_OUTPUT"

        if [ "$REQUEUE_EXIT" -ne 0 ]; then
            echo "[ERROR] Failed to requeue task ${TASK_ID}."
            exit "$REQUEUE_EXIT"
        fi

        exit 0
    fi

    echo "[ERROR] Task retries exhausted for ${TASK_ID} after ${TASK_RETRY_COUNT} retry/retries."
    echo "[ERROR] Last incomplete reason: ${reason}"
    exit "$final_exit_code"
}

set +e
TI_SIM_ROOT="$DATASET_ROOT" python -u "$SIM_RUNNER_PY" \
    --subject "$SUBJECT" \
    --montage-preset "$MONTAGE_PRESET" \
    --generate-charm-mesh \
    --mesh-timeout-hours "$MESH_TIMEOUT_HOURS" 2>&1 | tee -a "$LOG_FILE"
PY_EXIT=${PIPESTATUS[0]}
set -e

if [ "$PY_EXIT" -eq "$SIM_INPUT_EXIT_CODE" ]; then
    echo "[WARN] Required simulation inputs are missing for task ${TASK_ID}; dropping task without requeue."
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -Is)" \
        "${TASK_ID}" \
        "${DATASET_NAME}" \
        "${SUBJECT}" \
        "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}" \
        "${SLURM_ARRAY_TASK_ID}" >> "${LOG_DIR}/tasks_skipped_missing_inputs.tsv"
    rm -f "$RETRY_FILE" "$TASK_FILE"
    exit 0
fi

if [ -f "$COMPLETION_CHECK_PY" ]; then
    set +e
    python -u "$COMPLETION_CHECK_PY" \
        --root "$DATASET_ROOT" \
        --subject "$SUBJECT" 2>&1 | tee -a "$LOG_FILE"
    VALIDATION_EXIT=${PIPESTATUS[0]}
    set -e
else
    echo "[ERROR] Completion check script not found: ${COMPLETION_CHECK_PY}" | tee -a "$LOG_FILE"
    VALIDATION_EXIT=127
fi

if [ "$PY_EXIT" -eq 0 ] && [ "$VALIDATION_EXIT" -eq 0 ]; then
    rm -f "$RETRY_FILE" "$TASK_FILE"
    exit 0
fi

if [ "$PY_EXIT" -ne 0 ] && [ "$VALIDATION_EXIT" -eq 0 ]; then
    requeue_or_fail "Runner exited ${PY_EXIT}; validation passed but a clean runner exit is required" "$PY_EXIT"
fi

if [ "$PY_EXIT" -ne 0 ]; then
    requeue_or_fail "Runner exited ${PY_EXIT} and final task outputs are incomplete" "$PY_EXIT"
fi

requeue_or_fail "Runner exited 0 but final task output validation failed" "$SIM_INCOMPLETE_EXIT_CODE"
