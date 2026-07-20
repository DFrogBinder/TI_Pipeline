#!/bin/bash
#SBATCH --job-name=approved_wave_lhip
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-0
#SBATCH --output=approved-wave-lhip-%A_%a.out
#SBATCH --requeue

set -euo pipefail

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "[ERROR] Submit this script as a Slurm array." >&2
    exit 2
fi

: "${TI_APPROVED_WAVE_STAGE:?Missing TI_APPROVED_WAVE_STAGE.}"
: "${TI_APPROVED_WAVE_MANIFEST:?Missing TI_APPROVED_WAVE_MANIFEST.}"
: "${TI_APPROVED_WAVE_WORKFLOW_PY:?Missing TI_APPROVED_WAVE_WORKFLOW_PY.}"
: "${TI_APPROVED_WAVE_RESULT_DIR:?Missing TI_APPROVED_WAVE_RESULT_DIR.}"
: "${TI_APPROVED_WAVE_LOG_DIR:?Missing TI_APPROVED_WAVE_LOG_DIR.}"
: "${TI_TARGETS_CSV:?Missing TI_TARGETS_CSV.}"
: "${TI_EXPECTED_TARGETS_SHA256:?Missing TI_EXPECTED_TARGETS_SHA256.}"

if [ "${TI_APPROVED_WAVE_STAGE}" != "prepare" ] && \
   [ "${TI_APPROVED_WAVE_STAGE}" != "simulate" ]
then
    echo "[ERROR] TI_APPROVED_WAVE_STAGE must be prepare or simulate." >&2
    exit 2
fi
if [ "${TI_APPROVED_WAVE_STAGE}" = "simulate" ]; then
    : "${TI_SIM_RUNNER_PY:?Missing TI_SIM_RUNNER_PY.}"
    : "${TI_COMPLETION_CHECK_PY:?Missing TI_COMPLETION_CHECK_PY.}"
fi

TASK_OFFSET="${TASK_OFFSET:-0}"
MAX_RETRIES="${TI_APPROVED_WAVE_MAX_RETRIES:-2}"
MONTAGE_PRESET="${TI_MONTAGE_PRESET:-left-hippocampus}"
STATE_DIR="${TI_APPROVED_WAVE_STATE_DIR:-${TI_APPROVED_WAVE_RESULT_DIR}/retry_state}"

if ! [[ "${TASK_OFFSET}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] TASK_OFFSET must be a non-negative integer." >&2
    exit 2
fi
if ! [[ "${MAX_RETRIES}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] TI_APPROVED_WAVE_MAX_RETRIES must be non-negative." >&2
    exit 2
fi

REQUIRED_FILES=(
    "${TI_APPROVED_WAVE_MANIFEST}"
    "${TI_APPROVED_WAVE_WORKFLOW_PY}"
    "${TI_TARGETS_CSV}"
)
if [ "${TI_APPROVED_WAVE_STAGE}" = "simulate" ]; then
    REQUIRED_FILES+=("${TI_SIM_RUNNER_PY}" "${TI_COMPLETION_CHECK_PY}")
fi
for required in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "${required}" ]; then
        echo "[ERROR] Required file is missing: ${required}" >&2
        exit 2
    fi
done

TASK_INDEX=$((TASK_OFFSET + SLURM_ARRAY_TASK_ID))
TASK_LINE=$(sed -n "$((TASK_INDEX + 2))p" "${TI_APPROVED_WAVE_MANIFEST}")
if [ -z "${TASK_LINE}" ]; then
    echo "[ERROR] No manifest row for global task index ${TASK_INDEX}." >&2
    exit 2
fi

if [ "${TI_APPROVED_WAVE_STAGE}" = "prepare" ]; then
    IFS=$'\t' read -r \
        TASK_ID DATASET_NAME SUBJECT SOURCE_ANAT SOURCE_T1 SOURCE_T1_HASH \
        SOURCE_T2 SOURCE_T2_HASH APPROVED_LABEL APPROVED_LABEL_HASH \
        DATASET_ROOT ANAT_DIR M2M_DIR MESH_PATH RESULT_PATH STATUS MESSAGE \
        <<< "${TASK_LINE}"
    REPEAT_ID=01
else
    IFS=$'\t' read -r \
        TASK_ID DATASET_NAME REPEAT_ID DATASET_ROOT SUBJECT ANAT_DIR \
        CANONICAL_ANAT_DIR CANONICAL_M2M_DIR PREP_RESULT_PATH \
        MESH_RESULT_PATH RESULT_PATH STATUS MESSAGE <<< "${TASK_LINE}"
fi

if [ -z "${SUBJECT:-}" ] || [ "${TASK_ID:-}" != "${TASK_INDEX}" ]; then
    echo "[ERROR] Malformed or misindexed task row: ${TASK_LINE}" >&2
    exit 2
fi
if [ "${STATUS}" != "ready" ]; then
    echo "[ERROR] Task ${TASK_INDEX} is blocked: ${MESSAGE}" >&2
    exit 2
fi

mkdir -p "${TI_APPROVED_WAVE_LOG_DIR}" "${STATE_DIR}" "${TI_APPROVED_WAVE_RESULT_DIR}"
SAFE_DATASET="${DATASET_NAME//[^A-Za-z0-9_.-]/_}"
TASK_LOG="${TI_APPROVED_WAVE_LOG_DIR}/${TI_APPROVED_WAVE_STAGE}__${SAFE_DATASET}__${SUBJECT}.log"
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
echo "[INFO] Stage:           ${TI_APPROVED_WAVE_STAGE}"
echo "[INFO] Global index:    ${TASK_INDEX}"
echo "[INFO] Task ID:         ${TASK_ID}"
echo "[INFO] Dataset:         ${DATASET_NAME}"
echo "[INFO] Repeat:          ${REPEAT_ID}"
echo "[INFO] Subject:         ${SUBJECT}"
echo "[INFO] Task marker:     ${RESULT_PATH}"
echo "[INFO] Montage:         ${MONTAGE_PRESET}"
echo "[INFO] Retry:           ${RETRY_COUNT}/${MAX_RETRIES}"
if [ "${TI_APPROVED_WAVE_STAGE}" = "prepare" ]; then
    echo "[INFO] Operation:       one CHARM support build -> exact approved label -> one mesh"
    echo "[INFO] Source anat:     ${SOURCE_ANAT}"
    echo "[INFO] Source T1:       ${SOURCE_T1}"
    echo "[INFO] Source T1 hash:  ${SOURCE_T1_HASH}"
    echo "[INFO] Source T2:       ${SOURCE_T2}"
    echo "[INFO] Source T2 hash:  ${SOURCE_T2_HASH}"
    echo "[INFO] Approved label:  ${APPROVED_LABEL}"
    echo "[INFO] Label hash:      ${APPROVED_LABEL_HASH}"
    echo "[INFO] Dataset root:    ${DATASET_ROOT}"
    echo "[INFO] Anat directory:  ${ANAT_DIR}"
    echo "[INFO] m2m directory:   ${M2M_DIR}"
    echo "[INFO] Mesh path:       ${MESH_PATH}"
else
    echo "[INFO] Operation:       physical scaffold copy -> independent mesh -> fixed cap -> FEM"
    echo "[INFO] Dataset root:    ${DATASET_ROOT}"
    echo "[INFO] Anat directory:  ${ANAT_DIR}"
    echo "[INFO] Canonical anat:  ${CANONICAL_ANAT_DIR}"
    echo "[INFO] Canonical m2m:   ${CANONICAL_M2M_DIR}"
    echo "[INFO] Prep marker:     ${PREP_RESULT_PATH}"
    echo "[INFO] Mesh marker:     ${MESH_RESULT_PATH}"
fi

COMMAND=(
    python -u "${TI_APPROVED_WAVE_WORKFLOW_PY}"
)
if [ "${TI_APPROVED_WAVE_STAGE}" = "prepare" ]; then
    COMMAND+=(
        prep-task
        --manifest "${TI_APPROVED_WAVE_MANIFEST}"
        --task-index "${TASK_INDEX}"
        --montage-preset "${MONTAGE_PRESET}"
        --targets-csv "${TI_TARGETS_CSV}"
        --expected-targets-sha256 "${TI_EXPECTED_TARGETS_SHA256}"
    )
else
    COMMAND+=(
        simulation-task
        --manifest "${TI_APPROVED_WAVE_MANIFEST}"
        --task-index "${TASK_INDEX}"
        --montage-preset "${MONTAGE_PRESET}"
        --targets-csv "${TI_TARGETS_CSV}"
        --expected-targets-sha256 "${TI_EXPECTED_TARGETS_SHA256}"
        --simulation-runner "${TI_SIM_RUNNER_PY}"
        --simulation-validator "${TI_COMPLETION_CHECK_PY}"
        --python-bin python
    )
fi

set +e
"${COMMAND[@]}" 2>&1 | tee -a "${TASK_LOG}"
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
