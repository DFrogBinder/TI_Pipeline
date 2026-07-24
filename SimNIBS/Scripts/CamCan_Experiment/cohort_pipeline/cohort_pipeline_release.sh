#!/bin/bash
set -euo pipefail

: "${TI_COHORT_RELEASE_PLAN:?Missing TI_COHORT_RELEASE_PLAN.}"
: "${TI_COHORT_RELEASE_STEP:?Missing TI_COHORT_RELEASE_STEP.}"
: "${TI_COHORT_RELEASE_SCRIPT:?Missing TI_COHORT_RELEASE_SCRIPT.}"
: "${TI_COHORT_ARRAY_SCRIPT:?Missing TI_COHORT_ARRAY_SCRIPT.}"
: "${TI_COHORT_WORKFLOW_PY:?Missing TI_COHORT_WORKFLOW_PY.}"
: "${TI_COHORT_LOG_DIR:?Missing TI_COHORT_LOG_DIR.}"
: "${TI_COHORT_JOB_ID_FILE:?Missing TI_COHORT_JOB_ID_FILE.}"
: "${TI_COHORT_RELEASE_STATE_DIR:?Missing TI_COHORT_RELEASE_STATE_DIR.}"
: "${TI_COHORT_MESH_MANIFEST:?Missing TI_COHORT_MESH_MANIFEST.}"
: "${TI_COHORT_SIMULATION_MANIFEST:?Missing TI_COHORT_SIMULATION_MANIFEST.}"
: "${TI_TARGETS_CSV:?Missing TI_TARGETS_CSV.}"
: "${TI_EXPECTED_TARGETS_SHA256:?Missing TI_EXPECTED_TARGETS_SHA256.}"
: "${TI_SIM_RUNNER_PY:?Missing TI_SIM_RUNNER_PY.}"
: "${TI_COMPLETION_CHECK_PY:?Missing TI_COMPLETION_CHECK_PY.}"

SBATCH_BIN="${SBATCH_BIN:-sbatch}"
SCANCEL_BIN="${SCANCEL_BIN:-scancel}"
PARTITION="${PARTITION:-sheffield}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-50}"
MESH_WORKERS="${TI_COHORT_MESH_WORKERS_PER_ELEMENT:-2}"
THREADS_PER_MESH_WORKER="${TI_COHORT_MESH_THREADS_PER_WORKER:-4}"
LOCAL_STAGING="${TI_COHORT_MESH_LOCAL_STAGING:-1}"
MAX_RETRIES="${TI_COHORT_MAX_RETRIES:-unlimited}"
RELEASE_CPUS="${TI_COHORT_RELEASE_CPUS:-1}"
RELEASE_MEMORY="${TI_COHORT_RELEASE_MEMORY:-1G}"
RELEASE_TIME="${TI_COHORT_RELEASE_TIME:-01:00:00}"
RELEASE_SUBMIT_ATTEMPTS="${TI_COHORT_RELEASE_SUBMIT_ATTEMPTS:-60}"
RELEASE_RETRY_DELAY="${TI_COHORT_RELEASE_RETRY_DELAY:-60}"
JOB_PREFIX="${TI_COHORT_JOB_PREFIX:-camcan_cohort}"

for value_name in \
    TI_COHORT_RELEASE_STEP \
    MAX_CONCURRENT_TASKS \
    MESH_WORKERS \
    THREADS_PER_MESH_WORKER \
    RELEASE_CPUS \
    RELEASE_SUBMIT_ATTEMPTS \
    RELEASE_RETRY_DELAY
do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] ${value_name} must be a non-negative integer." >&2
        exit 2
    fi
done
if [ "${MAX_CONCURRENT_TASKS}" -lt 1 ] || \
   [ "${MESH_WORKERS}" -lt 1 ] || \
   [ "${THREADS_PER_MESH_WORKER}" -lt 1 ] || \
   [ "${RELEASE_CPUS}" -lt 1 ] || \
   [ "${RELEASE_SUBMIT_ATTEMPTS}" -lt 1 ]
then
    echo "[ERROR] Release and worker counts must be positive." >&2
    exit 2
fi

for required in \
    "${TI_COHORT_RELEASE_PLAN}" \
    "${TI_COHORT_RELEASE_SCRIPT}" \
    "${TI_COHORT_ARRAY_SCRIPT}" \
    "${TI_COHORT_WORKFLOW_PY}" \
    "${TI_COHORT_MESH_MANIFEST}" \
    "${TI_COHORT_SIMULATION_MANIFEST}" \
    "${TI_TARGETS_CSV}" \
    "${TI_SIM_RUNNER_PY}" \
    "${TI_COMPLETION_CHECK_PY}"
do
    if [ ! -f "${required}" ]; then
        echo "[ERROR] Required release input is missing: ${required}" >&2
        exit 2
    fi
done

mkdir -p \
    "${TI_COHORT_LOG_DIR}" \
    "${TI_COHORT_RELEASE_STATE_DIR}" \
    "$(dirname "${TI_COHORT_JOB_ID_FILE}")"

TOTAL_STEPS=$(awk 'NR > 1 && NF > 0 { count++ } END { print count + 0 }' "${TI_COHORT_RELEASE_PLAN}")
if [ "${TI_COHORT_RELEASE_STEP}" -gt "${TOTAL_STEPS}" ]; then
    echo "[ERROR] Release step ${TI_COHORT_RELEASE_STEP} exceeds plan length ${TOTAL_STEPS}." >&2
    exit 2
fi

append_job_id() {
    local job_id="$1"
    local lock_file="${TI_COHORT_JOB_ID_FILE}.lock"
    (
        flock -x 9
        if ! grep -qx "${job_id}" "${TI_COHORT_JOB_ID_FILE}" 2>/dev/null; then
            printf '%s\n' "${job_id}" >> "${TI_COHORT_JOB_ID_FILE}"
        fi
    ) 9>"${lock_file}"
}

submit_with_qos_retry() {
    local label="$1"
    shift
    local attempt output exit_code
    for ((attempt = 1; attempt <= RELEASE_SUBMIT_ATTEMPTS; attempt++)); do
        set +e
        output=$("$@" 2>&1)
        exit_code=$?
        set -e
        if [ "${exit_code}" -eq 0 ]; then
            printf '%s\n' "${output}"
            return 0
        fi
        echo "[WARN] ${label} submission attempt ${attempt}/${RELEASE_SUBMIT_ATTEMPTS} failed:" >&2
        echo "${output}" >&2
        if ! printf '%s\n' "${output}" | grep -q 'QOSMaxSubmitJobPerUserLimit'; then
            return "${exit_code}"
        fi
        if [ "${attempt}" -lt "${RELEASE_SUBMIT_ATTEMPTS}" ]; then
            echo "[INFO] Waiting ${RELEASE_RETRY_DELAY}s for Stanage QOS capacity." >&2
            sleep "${RELEASE_RETRY_DELAY}"
        fi
    done
    return 1
}

parse_job_id() {
    local submission="$1"
    local job_id="${submission%%;*}"
    if ! [[ "${job_id}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Could not parse Slurm job ID: ${submission}" >&2
        return 2
    fi
    printf '%s\n' "${job_id}"
}

if [ "${TI_COHORT_RELEASE_STEP}" -eq "${TOTAL_STEPS}" ]; then
    COMPLETION_RECEIPT="${TI_COHORT_RELEASE_STATE_DIR}/chain_complete.tsv"
    {
        printf 'status\tcomplete\n'
        printf 'completed_at\t%s\n' "$(date --iso-8601=seconds)"
        printf 'release_steps\t%s\n' "${TOTAL_STEPS}"
        printf 'final_dependency_job\t%s\n' "${SLURM_JOB_ID:-unknown}"
    } > "${COMPLETION_RECEIPT}"
    echo "[INFO] All planned scaffold, mesh, and FEM arrays completed successfully."
    echo "[INFO] Completion receipt: ${COMPLETION_RECEIPT}"
    exit 0
fi

PLAN_LINE=$(awk -F '\t' -v step="${TI_COHORT_RELEASE_STEP}" '
NR > 1 && $1 == step {
    print
    exit
}
' "${TI_COHORT_RELEASE_PLAN}")
if [ -z "${PLAN_LINE}" ]; then
    echo "[ERROR] Release plan has no row for step ${TI_COHORT_RELEASE_STEP}." >&2
    exit 2
fi
IFS=$'\t' read -r PLAN_STEP STAGE CHUNK_INDEX OFFSET COUNT <<< "${PLAN_LINE}"
if [ "${PLAN_STEP}" != "${TI_COHORT_RELEASE_STEP}" ] || \
   ! [[ "${CHUNK_INDEX}" =~ ^[0-9]+$ ]] || \
   ! [[ "${OFFSET}" =~ ^[0-9]+$ ]] || \
   ! [[ "${COUNT}" =~ ^[0-9]+$ ]] || \
   [ "${COUNT}" -lt 1 ]
then
    echo "[ERROR] Malformed release-plan row: ${PLAN_LINE}" >&2
    exit 2
fi
case "${STAGE}" in
    mesh|simulate)
        ;;
    *)
        echo "[ERROR] Unsupported release stage: ${STAGE}" >&2
        exit 2
        ;;
esac

STEP_RECEIPT="${TI_COHORT_RELEASE_STATE_DIR}/release_step_${TI_COHORT_RELEASE_STEP}.tsv"
if [ -s "${STEP_RECEIPT}" ]; then
    echo "[INFO] Release step ${TI_COHORT_RELEASE_STEP} already has a receipt; refusing duplicate submission."
    cat "${STEP_RECEIPT}"
    exit 0
fi

ARRAY_SPEC="0-$((COUNT - 1))%${MAX_CONCURRENT_TASKS}"
COMMON_EXPORTS="TI_COHORT_WORKFLOW_PY=${TI_COHORT_WORKFLOW_PY},TI_COHORT_LOG_DIR=${TI_COHORT_LOG_DIR},TI_COHORT_MAX_RETRIES=${MAX_RETRIES}"
if [ "${STAGE}" = "mesh" ]; then
    OUTPUT_PATTERN="${TI_COHORT_LOG_DIR}/mesh-c${CHUNK_INDEX}-%A_%a.out"
    JOB_NAME="${JOB_PREFIX}_mesh${CHUNK_INDEX}"
    ARRAY_EXPORTS="ALL,${COMMON_EXPORTS},TI_COHORT_STAGE=mesh,TI_COHORT_MANIFEST=${TI_COHORT_MESH_MANIFEST},ELEMENT_OFFSET=${OFFSET},TI_COHORT_MESH_WORKERS_PER_ELEMENT=${MESH_WORKERS},TI_COHORT_MESH_THREADS_PER_WORKER=${THREADS_PER_MESH_WORKER},TI_COHORT_MESH_LOCAL_STAGING=${LOCAL_STAGING}"
else
    OUTPUT_PATTERN="${TI_COHORT_LOG_DIR}/simulate-c${CHUNK_INDEX}-%A_%a.out"
    JOB_NAME="${JOB_PREFIX}_sim${CHUNK_INDEX}"
    ARRAY_EXPORTS="ALL,${COMMON_EXPORTS},TI_COHORT_STAGE=simulate,TI_COHORT_MANIFEST=${TI_COHORT_SIMULATION_MANIFEST},TASK_OFFSET=${OFFSET},TI_TARGETS_CSV=${TI_TARGETS_CSV},TI_EXPECTED_TARGETS_SHA256=${TI_EXPECTED_TARGETS_SHA256},TI_SIM_RUNNER_PY=${TI_SIM_RUNNER_PY},TI_COMPLETION_CHECK_PY=${TI_COMPLETION_CHECK_PY}"
fi

ARRAY_COMMAND=(
    "${SBATCH_BIN}"
    --parsable
    --job-name="${JOB_NAME}"
    --partition="${PARTITION}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${MEMORY}"
    --time="${TIME_LIMIT}"
    --array="${ARRAY_SPEC}"
    --output="${OUTPUT_PATTERN}"
    --export="${ARRAY_EXPORTS}"
    "${TI_COHORT_ARRAY_SCRIPT}"
)
set +e
ARRAY_SUBMISSION=$(submit_with_qos_retry "${STAGE} chunk ${CHUNK_INDEX}" "${ARRAY_COMMAND[@]}")
ARRAY_EXIT=$?
set -e
if [ "${ARRAY_EXIT}" -ne 0 ]; then
    echo "[ERROR] Could not submit ${STAGE} chunk ${CHUNK_INDEX}." >&2
    exit "${ARRAY_EXIT}"
fi
ARRAY_JOB_ID=$(parse_job_id "${ARRAY_SUBMISSION}")
append_job_id "${ARRAY_JOB_ID}"
echo "[INFO] Submitted ${STAGE} chunk ${CHUNK_INDEX}: job=${ARRAY_JOB_ID}, offset=${OFFSET}, tasks=${COUNT}"

NEXT_STEP=$((TI_COHORT_RELEASE_STEP + 1))
RELEASE_EXPORTS="ALL,TI_COHORT_RELEASE_PLAN=${TI_COHORT_RELEASE_PLAN},TI_COHORT_RELEASE_STEP=${NEXT_STEP},TI_COHORT_RELEASE_SCRIPT=${TI_COHORT_RELEASE_SCRIPT},TI_COHORT_ARRAY_SCRIPT=${TI_COHORT_ARRAY_SCRIPT},TI_COHORT_WORKFLOW_PY=${TI_COHORT_WORKFLOW_PY},TI_COHORT_LOG_DIR=${TI_COHORT_LOG_DIR},TI_COHORT_JOB_ID_FILE=${TI_COHORT_JOB_ID_FILE},TI_COHORT_RELEASE_STATE_DIR=${TI_COHORT_RELEASE_STATE_DIR},TI_COHORT_MESH_MANIFEST=${TI_COHORT_MESH_MANIFEST},TI_COHORT_SIMULATION_MANIFEST=${TI_COHORT_SIMULATION_MANIFEST},TI_TARGETS_CSV=${TI_TARGETS_CSV},TI_EXPECTED_TARGETS_SHA256=${TI_EXPECTED_TARGETS_SHA256},TI_SIM_RUNNER_PY=${TI_SIM_RUNNER_PY},TI_COMPLETION_CHECK_PY=${TI_COMPLETION_CHECK_PY},TI_COHORT_MAX_RETRIES=${MAX_RETRIES},TI_COHORT_MESH_WORKERS_PER_ELEMENT=${MESH_WORKERS},TI_COHORT_MESH_THREADS_PER_WORKER=${THREADS_PER_MESH_WORKER},TI_COHORT_MESH_LOCAL_STAGING=${LOCAL_STAGING},TI_COHORT_RELEASE_CPUS=${RELEASE_CPUS},TI_COHORT_RELEASE_MEMORY=${RELEASE_MEMORY},TI_COHORT_RELEASE_TIME=${RELEASE_TIME},TI_COHORT_RELEASE_SUBMIT_ATTEMPTS=${RELEASE_SUBMIT_ATTEMPTS},TI_COHORT_RELEASE_RETRY_DELAY=${RELEASE_RETRY_DELAY},TI_COHORT_JOB_PREFIX=${JOB_PREFIX},PARTITION=${PARTITION},CPUS_PER_TASK=${CPUS_PER_TASK},MEMORY=${MEMORY},TIME_LIMIT=${TIME_LIMIT},MAX_CONCURRENT_TASKS=${MAX_CONCURRENT_TASKS},SBATCH_BIN=${SBATCH_BIN},SCANCEL_BIN=${SCANCEL_BIN}"
RELEASE_COMMAND=(
    "${SBATCH_BIN}"
    --parsable
    --job-name="${JOB_PREFIX}_release${NEXT_STEP}"
    --partition="${PARTITION}"
    --cpus-per-task="${RELEASE_CPUS}"
    --mem="${RELEASE_MEMORY}"
    --time="${RELEASE_TIME}"
    --dependency="afterok:${ARRAY_JOB_ID}"
    --output="${TI_COHORT_LOG_DIR}/release-%j.out"
    --export="${RELEASE_EXPORTS}"
    "${TI_COHORT_RELEASE_SCRIPT}"
)
set +e
NEXT_SUBMISSION=$(submit_with_qos_retry "release step ${NEXT_STEP}" "${RELEASE_COMMAND[@]}")
NEXT_EXIT=$?
set -e
if [ "${NEXT_EXIT}" -ne 0 ]; then
    echo "[ERROR] Could not attach the next release job; cancelling array ${ARRAY_JOB_ID}." >&2
    "${SCANCEL_BIN}" "${ARRAY_JOB_ID}" || true
    exit "${NEXT_EXIT}"
fi
NEXT_RELEASE_JOB_ID=$(parse_job_id "${NEXT_SUBMISSION}")
append_job_id "${NEXT_RELEASE_JOB_ID}"

{
    printf 'step\tstage\tchunk_index\toffset\tcount\tarray_job_id\tnext_release_job_id\n'
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${TI_COHORT_RELEASE_STEP}" \
        "${STAGE}" \
        "${CHUNK_INDEX}" \
        "${OFFSET}" \
        "${COUNT}" \
        "${ARRAY_JOB_ID}" \
        "${NEXT_RELEASE_JOB_ID}"
} > "${STEP_RECEIPT}"

echo "[INFO] Next release step ${NEXT_STEP}: job=${NEXT_RELEASE_JOB_ID}, dependency=afterok:${ARRAY_JOB_ID}"
