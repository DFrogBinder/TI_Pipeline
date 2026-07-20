#!/bin/bash
#SBATCH --job-name=approved_wave_lhip
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-0
#SBATCH --output=approved-wave-lhip-%A_%a.out
#SBATCH --requeue

set -uo pipefail

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "[ERROR] Submit this script as a Slurm array." >&2
    exit 2
fi

: "${TI_APPROVED_WAVE_MANIFEST:?Missing TI_APPROVED_WAVE_MANIFEST.}"
: "${TI_APPROVED_WAVE_WORKFLOW_PY:?Missing TI_APPROVED_WAVE_WORKFLOW_PY.}"
: "${TI_APPROVED_WAVE_RESULT_DIR:?Missing TI_APPROVED_WAVE_RESULT_DIR.}"
: "${TI_APPROVED_WAVE_LOG_DIR:?Missing TI_APPROVED_WAVE_LOG_DIR.}"
: "${TI_SIM_RUNNER_PY:?Missing TI_SIM_RUNNER_PY.}"
: "${TI_COMPLETION_CHECK_PY:?Missing TI_COMPLETION_CHECK_PY.}"
: "${TI_TARGETS_CSV:?Missing TI_TARGETS_CSV.}"
: "${TI_EXPECTED_TARGETS_SHA256:?Missing TI_EXPECTED_TARGETS_SHA256.}"

TASK_OFFSET="${TASK_OFFSET:-0}"
MAX_RETRIES="${TI_APPROVED_WAVE_MAX_RETRIES:-2}"
MONTAGE_PRESET="${TI_MONTAGE_PRESET:-left-hippocampus}"
STATE_DIR="${TI_APPROVED_WAVE_STATE_DIR:-${TI_APPROVED_WAVE_RESULT_DIR}/retry_state}"

if ! [[ "${TASK_OFFSET}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] TASK_OFFSET must be a non-negative integer." >&2
    exit 2
fi
if ! [[ "${MAX_RETRIES}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] TI_APPROVED_WAVE_MAX_RETRIES must be a non-negative integer." >&2
    exit 2
fi
for required in \
    "${TI_APPROVED_WAVE_MANIFEST}" \
    "${TI_APPROVED_WAVE_WORKFLOW_PY}" \
    "${TI_SIM_RUNNER_PY}" \
    "${TI_COMPLETION_CHECK_PY}" \
    "${TI_TARGETS_CSV}"
do
    if [ ! -f "${required}" ]; then
        echo "[ERROR] Required file is missing: ${required}" >&2
        exit 2
    fi
done

TASK_INDEX=$((TASK_OFFSET + SLURM_ARRAY_TASK_ID))
TASK_LINE=$(sed -n "$((TASK_INDEX + 2))p" "${TI_APPROVED_WAVE_MANIFEST}")
IFS=$'\t' read -r \
    TASK_ID DATASET_NAME REPEAT_ID DATASET_ROOT SUBJECT SOURCE_ANAT \
    SOURCE_T1 SOURCE_T1_HASH SOURCE_T2 SOURCE_T2_HASH APPROVED_LABEL \
    APPROVED_LABEL_HASH ANAT_DIR M2M_DIR MESH_PATH MESH_RESULT_PATH \
    RESULT_PATH STATUS MESSAGE <<< "${TASK_LINE}"

if [ -z "${TASK_LINE}" ] || [ -z "${SUBJECT:-}" ]; then
    echo "[ERROR] No valid manifest row for task index ${TASK_INDEX}." >&2
    exit 2
fi
if [ "${STATUS}" != "ready" ]; then
    echo "[ERROR] Task ${TASK_INDEX} is blocked: ${MESSAGE}" >&2
    exit 2
fi

mkdir -p "${TI_APPROVED_WAVE_LOG_DIR}" "${STATE_DIR}" "${TI_APPROVED_WAVE_RESULT_DIR}"
SAFE_DATASET="${DATASET_NAME//[^A-Za-z0-9_.-]/_}"
TASK_LOG="${TI_APPROVED_WAVE_LOG_DIR}/${SAFE_DATASET}__${SUBJECT}.log"
TASK_KEY="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID}"
RETRY_FILE="${STATE_DIR}/${TASK_KEY}.retry"
if [ -f "${RETRY_FILE}" ]; then
    RETRY_COUNT=$(tr -d '[:space:]' < "${RETRY_FILE}")
else
    RETRY_COUNT=0
fi
if ! [[ "${RETRY_COUNT}" =~ ^[0-9]+$ ]]; then
    RETRY_COUNT=0
fi

module purge
module use "$HOME/modules"
module load SimNIBS/4.0.1-foss-2023a

export PYTHONNOUSERSITE=1
SHIM_DIR="${SLURM_TMPDIR:-/tmp}/${USER}_approved_wave_shim_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${SHIM_DIR}"
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
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "[INFO] Job ID:          ${SLURM_JOB_ID}"
echo "[INFO] Global index:    ${TASK_INDEX}"
echo "[INFO] Task ID:         ${TASK_ID}"
echo "[INFO] Dataset:         ${DATASET_NAME}"
echo "[INFO] Repeat:          ${REPEAT_ID}"
echo "[INFO] Subject:         ${SUBJECT}"
echo "[INFO] Dataset root:    ${DATASET_ROOT}"
echo "[INFO] Source anat:     ${SOURCE_ANAT}"
echo "[INFO] Source T1:       ${SOURCE_T1}"
echo "[INFO] Source T1 hash:  ${SOURCE_T1_HASH}"
echo "[INFO] Source T2:       ${SOURCE_T2}"
echo "[INFO] Source T2 hash:  ${SOURCE_T2_HASH}"
echo "[INFO] Approved label:  ${APPROVED_LABEL}"
echo "[INFO] Label SHA-256:   ${APPROVED_LABEL_HASH}"
echo "[INFO] Anat directory:  ${ANAT_DIR}"
echo "[INFO] m2m directory:   ${M2M_DIR}"
echo "[INFO] Target mesh:     ${MESH_PATH}"
echo "[INFO] Mesh marker:     ${MESH_RESULT_PATH}"
echo "[INFO] Task marker:     ${RESULT_PATH}"
echo "[INFO] Montage:         ${MONTAGE_PRESET}"
echo "[INFO] Retry:           ${RETRY_COUNT}/${MAX_RETRIES}"
echo "[INFO] Operation:       CHARM prerequisites -> exact approved label -> mesh -> simulation -> validation"

set +e
python -u "${TI_APPROVED_WAVE_WORKFLOW_PY}" run-task \
    --manifest "${TI_APPROVED_WAVE_MANIFEST}" \
    --task-index "${TASK_INDEX}" \
    --montage-preset "${MONTAGE_PRESET}" \
    --targets-csv "${TI_TARGETS_CSV}" \
    --expected-targets-sha256 "${TI_EXPECTED_TARGETS_SHA256}" \
    --simulation-runner "${TI_SIM_RUNNER_PY}" \
    --simulation-validator "${TI_COMPLETION_CHECK_PY}" 2>&1 | tee -a "${TASK_LOG}"
RUN_EXIT=${PIPESTATUS[0]}
set -e

if [ "${RUN_EXIT}" -eq 0 ]; then
    rm -f "${RETRY_FILE}"
    exit 0
fi

if [ "${RETRY_COUNT}" -lt "${MAX_RETRIES}" ]; then
    NEXT_RETRY=$((RETRY_COUNT + 1))
    printf '%s\n' "${NEXT_RETRY}" > "${RETRY_FILE}"
    echo "[WARN] Task failed with exit ${RUN_EXIT}; requeueing (${NEXT_RETRY}/${MAX_RETRIES})."
    scontrol requeue "${SLURM_JOB_ID}"
    exit $?
fi

echo "[ERROR] Task failed after ${RETRY_COUNT} retries; see ${TASK_LOG}." >&2
exit "${RUN_EXIT}"
