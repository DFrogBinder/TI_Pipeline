#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMCAN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PIPELINE_DIR="$(cd "${CAMCAN_DIR}/.." && pwd)"

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: bash $0 COHORT_ID [--preflight]" >&2
    exit 2
fi
COHORT_ID="$1"
PREFLIGHT_ONLY=0
if [ "$#" -eq 2 ]; then
    if [ "$2" != "--preflight" ]; then
        echo "[ERROR] Unknown option: $2" >&2
        exit 2
    fi
    PREFLIGHT_ONLY=1
fi
if ! [[ "${COHORT_ID}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "[ERROR] Invalid cohort ID: ${COHORT_ID}" >&2
    exit 2
fi

STUDY_CONFIG="${STUDY_CONFIG:-${SCRIPT_DIR}/studies/corrected_v4_four_roi.json}"
COHORT_CONFIG="${COHORT_CONFIG:-${SCRIPT_DIR}/cohorts/${COHORT_ID}/cohort.json}"
WORKFLOW_PY="${TI_COHORT_WORKFLOW_PY:-${SCRIPT_DIR}/workflow.py}"
SLURM_SCRIPT="${SLURM_SCRIPT:-${SCRIPT_DIR}/cohort_pipeline_array.slurm}"
TARGETS_CSV="${TI_TARGETS_CSV:-${PIPELINE_DIR}/utils/targets.csv}"
EXPECTED_TARGETS_SHA256="${TI_EXPECTED_TARGETS_SHA256:-97a8c7a72faf88d9af9e4facbdf628fba1a130d327da778bcbd00af66f2916e6}"
SIM_RUNNER_PY="${TI_SIM_RUNNER_PY:-${CAMCAN_DIR}/simulation/TI_runner_multi-core.py}"
COMPLETION_CHECK_PY="${TI_COMPLETION_CHECK_PY:-${CAMCAN_DIR}/simulation/validate_simulation_outputs.py}"

STUDY_ROOT="${STUDY_ROOT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["hpc_study_root"])' "${STUDY_CONFIG}")}"
SCAFFOLD_ROOT="${SCAFFOLD_ROOT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["hpc_scaffold_root"])' "${STUDY_CONFIG}")}"
MAP_MANIFEST="${MAP_MANIFEST:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["hpc_corrected_map_manifest"])' "${STUDY_CONFIG}")}"
LEGACY_SCAFFOLD_MANIFEST="${LEGACY_SCAFFOLD_MANIFEST:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("hpc_legacy_scaffold_manifest",""))' "${STUDY_CONFIG}")}"
SOURCE_ROOT_1="${SOURCE_ROOT_1:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["hpc_source_roots"][0])' "${STUDY_CONFIG}")}"
SOURCE_ROOT_2="${SOURCE_ROOT_2:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["hpc_source_roots"][1])' "${STUDY_CONFIG}")}"

CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${STUDY_ROOT}/campaigns/${COHORT_ID}}"
LOG_DIR="${LOG_DIR:-${CAMPAIGN_ROOT}/logs}"
SCAFFOLD_MANIFEST="${SCAFFOLD_MANIFEST:-${CAMPAIGN_ROOT}/scaffold_tasks.tsv}"
MESH_MANIFEST="${MESH_MANIFEST:-${CAMPAIGN_ROOT}/mesh_tasks.tsv}"
SIMULATION_MANIFEST="${SIMULATION_MANIFEST:-${CAMPAIGN_ROOT}/simulation_tasks.tsv}"
SUMMARY="${SUMMARY:-${CAMPAIGN_ROOT}/preflight.json}"

MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-50}"
MAX_ARRAY_ELEMENTS="${MAX_ARRAY_ELEMENTS:-1000}"
MESH_WORKERS="${TI_COHORT_MESH_WORKERS_PER_ELEMENT:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
MAX_RETRIES="${TI_COHORT_MAX_RETRIES:-unlimited}"
LOCAL_STAGING="${TI_COHORT_MESH_LOCAL_STAGING:-1}"
PARTITION="${PARTITION:-sheffield}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
SCONTROL_BIN="${SCONTROL_BIN:-scontrol}"
SCANCEL_BIN="${SCANCEL_BIN:-scancel}"
SQUEUE_BIN="${SQUEUE_BIN:-squeue}"
JOB_PREFIX="${JOB_PREFIX:-cohort_${COHORT_ID}}"

for required in \
    "${STUDY_CONFIG}" \
    "${COHORT_CONFIG}" \
    "${WORKFLOW_PY}" \
    "${SLURM_SCRIPT}" \
    "${TARGETS_CSV}" \
    "${SIM_RUNNER_PY}" \
    "${COMPLETION_CHECK_PY}" \
    "${MAP_MANIFEST}"
do
    if [ ! -f "${required}" ]; then
        echo "[ERROR] Required file is missing: ${required}" >&2
        exit 2
    fi
done
for required_dir in "${SOURCE_ROOT_1}" "${SOURCE_ROOT_2}"; do
    if [ ! -d "${required_dir}" ]; then
        echo "[ERROR] Required source root is missing: ${required_dir}" >&2
        exit 2
    fi
done
for value_name in \
    MAX_CONCURRENT_TASKS \
    MAX_ARRAY_ELEMENTS \
    MESH_WORKERS \
    CPUS_PER_TASK
do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || [ "${value}" -lt 1 ]; then
        echo "[ERROR] ${value_name} must be a positive integer; got ${value}." >&2
        exit 2
    fi
done
if [ "${MAX_RETRIES}" != "unlimited" ] && ! [[ "${MAX_RETRIES}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] TI_COHORT_MAX_RETRIES must be unlimited or a non-negative integer." >&2
    exit 2
fi
if [ $((CPUS_PER_TASK % MESH_WORKERS)) -ne 0 ]; then
    echo "[ERROR] CPUS_PER_TASK must divide evenly across MESH_WORKERS." >&2
    exit 2
fi
if [ "${LOCAL_STAGING}" != "0" ] && [ "${LOCAL_STAGING}" != "1" ]; then
    echo "[ERROR] TI_COHORT_MESH_LOCAL_STAGING must be 0 or 1." >&2
    exit 2
fi

if command -v "${SCONTROL_BIN}" >/dev/null 2>&1; then
    set +e
    SCONTROL_CONFIG=$("${SCONTROL_BIN}" show config 2>&1)
    SCONTROL_EXIT=$?
    set -e
    if [ "${SCONTROL_EXIT}" -ne 0 ]; then
        echo "[ERROR] Could not inspect the live Slurm MaxArraySize." >&2
        echo "${SCONTROL_CONFIG}" >&2
        exit "${SCONTROL_EXIT}"
    fi
    LIVE_MAX_ARRAY_SIZE=$(printf '%s\n' "${SCONTROL_CONFIG}" | awk -F '=' '$1 ~ /^[[:space:]]*MaxArraySize/ && !found { gsub(/[[:space:]]/, "", $2); print $2; found=1 }')
    if [ -n "${LIVE_MAX_ARRAY_SIZE}" ] && [ "${MAX_ARRAY_ELEMENTS}" -gt "${LIVE_MAX_ARRAY_SIZE}" ]; then
        echo "[INFO] Reducing MAX_ARRAY_ELEMENTS from ${MAX_ARRAY_ELEMENTS} to live MaxArraySize=${LIVE_MAX_ARRAY_SIZE}."
        MAX_ARRAY_ELEMENTS="${LIVE_MAX_ARRAY_SIZE}"
    fi
fi

mkdir -p "${CAMPAIGN_ROOT}" "${LOG_DIR}" "${SCAFFOLD_ROOT}"

PREFLIGHT_COMMAND=(
    python3 "${WORKFLOW_PY}" preflight
    --study-config "${STUDY_CONFIG}"
    --cohort-config "${COHORT_CONFIG}"
    --campaign-root "${CAMPAIGN_ROOT}"
    --targets-csv "${TARGETS_CSV}"
    --map-manifest "${MAP_MANIFEST}"
    --study-root "${STUDY_ROOT}"
    --scaffold-root "${SCAFFOLD_ROOT}"
    --source-root "${SOURCE_ROOT_1}"
    --source-root "${SOURCE_ROOT_2}"
    --max-array-elements "${MAX_ARRAY_ELEMENTS}"
    --mesh-workers "${MESH_WORKERS}"
)
if [ -n "${LEGACY_SCAFFOLD_MANIFEST}" ]; then
    PREFLIGHT_COMMAND+=(--legacy-scaffold-manifest "${LEGACY_SCAFFOLD_MANIFEST}")
fi
"${PREFLIGHT_COMMAND[@]}"

EXPECTED_SUBJECTS=$(awk 'NR > 1 && NF > 0 { count++ } END { print count + 0 }' "${SCAFFOLD_MANIFEST}")
SCAFFOLD_READY=$(awk -F '\t' 'NR > 1 && $18 == "ready" { count++ } END { print count + 0 }' "${SCAFFOLD_MANIFEST}")
SCAFFOLD_REUSE=$(awk -F '\t' 'NR > 1 && $3 == "reuse" { count++ } END { print count + 0 }' "${SCAFFOLD_MANIFEST}")
SCAFFOLD_IMPORT=$(awk -F '\t' 'NR > 1 && $3 == "import" { count++ } END { print count + 0 }' "${SCAFFOLD_MANIFEST}")
SCAFFOLD_BOOTSTRAP=$(awk -F '\t' 'NR > 1 && $3 == "bootstrap" { count++ } END { print count + 0 }' "${SCAFFOLD_MANIFEST}")
MESH_TASKS=$(awk 'NR > 1 && NF > 0 { count++ } END { print count + 0 }' "${MESH_MANIFEST}")
MESH_READY=$(awk -F '\t' 'NR > 1 && $21 == "ready" { count++ } END { print count + 0 }' "${MESH_MANIFEST}")
SIMULATION_TASKS=$(awk 'NR > 1 && NF > 0 { count++ } END { print count + 0 }' "${SIMULATION_MANIFEST}")
SIMULATION_READY=$(awk -F '\t' 'NR > 1 && $21 == "ready" { count++ } END { print count + 0 }' "${SIMULATION_MANIFEST}")
UNIQUE_ROIS=$(awk -F '\t' 'NR > 1 { seen[$2]=1 } END { for (value in seen) count++; print count + 0 }' "${MESH_MANIFEST}")
UNIQUE_REPEATS=$(awk -F '\t' 'NR > 1 { seen[$5]=1 } END { for (value in seen) count++; print count + 0 }' "${MESH_MANIFEST}")
EXPECTED_TASKS=$((EXPECTED_SUBJECTS * 4 * 10))
if [ "${EXPECTED_SUBJECTS}" -lt 1 ] || [ "${EXPECTED_SUBJECTS}" -gt 200 ] || \
   [ "${SCAFFOLD_READY}" -ne "${EXPECTED_SUBJECTS}" ] || \
   [ "${MESH_TASKS}" -ne "${EXPECTED_TASKS}" ] || \
   [ "${MESH_READY}" -ne "${EXPECTED_TASKS}" ] || \
   [ "${SIMULATION_TASKS}" -ne "${EXPECTED_TASKS}" ] || \
   [ "${SIMULATION_READY}" -ne "${EXPECTED_TASKS}" ] || \
   [ "${UNIQUE_ROIS}" -ne 4 ] || \
   [ "${UNIQUE_REPEATS}" -ne 10 ]
then
    echo "[ERROR] Cohort preflight scope gate failed." >&2
    printf 'subjects=%s scaffold_ready=%s mesh=%s/%s simulation=%s/%s rois=%s repeats=%s\n' \
        "${EXPECTED_SUBJECTS}" \
        "${SCAFFOLD_READY}" \
        "${MESH_READY}" \
        "${MESH_TASKS}" \
        "${SIMULATION_READY}" \
        "${SIMULATION_TASKS}" \
        "${UNIQUE_ROIS}" \
        "${UNIQUE_REPEATS}" >&2
    exit 2
fi

TARGETS_SHA256=$(sha256sum "${TARGETS_CSV}" | awk '{print $1}')
if [ "${TARGETS_SHA256}" != "${EXPECTED_TARGETS_SHA256}" ]; then
    echo "[ERROR] targets.csv hash mismatch." >&2
    exit 2
fi

MESH_ELEMENTS=$(( (MESH_TASKS + MESH_WORKERS - 1) / MESH_WORKERS ))
MESH_CHUNKS=$(( (MESH_ELEMENTS + MAX_ARRAY_ELEMENTS - 1) / MAX_ARRAY_ELEMENTS ))
SIMULATION_CHUNKS=$(( (SIMULATION_TASKS + MAX_ARRAY_ELEMENTS - 1) / MAX_ARRAY_ELEMENTS ))
SCHEDULER_TASKS=$((EXPECTED_SUBJECTS + MESH_ELEMENTS + SIMULATION_TASKS))
THREADS_PER_MESH_WORKER=$((CPUS_PER_TASK / MESH_WORKERS))

printf '%s\n' \
    'Scope:' \
    '  study: corrected-v4 CamCan four-ROI campaign' \
    "  cohort: ${COHORT_ID}" \
    "  subjects: ${EXPECTED_SUBJECTS}" \
    '  ROIs: 4 (Left_Hippocampus, Left_M1, Right_DLPC, Right_Thalamus)' \
    '  repeats per ROI: 10 (01-10)' \
    "  scaffold tasks: ${EXPECTED_SUBJECTS}" \
    "  existing scaffold reuse tasks: ${SCAFFOLD_REUSE}" \
    "  legacy scaffold imports: ${SCAFFOLD_IMPORT}" \
    "  one-time CHARM bootstraps: ${SCAFFOLD_BOOTSTRAP}" \
    "  independent mesh tasks: ${MESH_TASKS}" \
    "  packed mesh array elements: ${MESH_ELEMENTS} in ${MESH_CHUNKS} sequential chunk(s)" \
    "  FEM tasks: ${SIMULATION_TASKS} in ${SIMULATION_CHUNKS} sequential chunk(s)" \
    "  scheduler task instances: ${SCHEDULER_TASKS}" \
    "  expected meshes: ${MESH_TASKS}" \
    "  expected validated simulations: ${SIMULATION_TASKS}" \
    '  execution: full requested cohort; not a smoke or subset' \
    '  map source: corrected-v4 collection; original-map results are not reused'

echo "[INFO] Study root:          ${STUDY_ROOT}"
echo "[INFO] Scaffold root:       ${SCAFFOLD_ROOT}"
echo "[INFO] Corrected maps:      ${MAP_MANIFEST}"
echo "[INFO] Campaign root:       ${CAMPAIGN_ROOT}"
echo "[INFO] Scaffold manifest:   ${SCAFFOLD_MANIFEST}"
echo "[INFO] Mesh manifest:       ${MESH_MANIFEST}"
echo "[INFO] Simulation manifest: ${SIMULATION_MANIFEST}"
echo "[INFO] Resource profile:    SimNIBS/4.0.1-foss-2023a, ${PARTITION}, ${CPUS_PER_TASK} CPU, ${MEMORY}, ${TIME_LIMIT}"
echo "[INFO] Mesh architecture:   ${MESH_WORKERS} workers x ${THREADS_PER_MESH_WORKER} cores, node-local staging=${LOCAL_STAGING}"
echo "[INFO] Array concurrency:   ${MAX_CONCURRENT_TASKS}"
echo "[INFO] Array chunk limit:   ${MAX_ARRAY_ELEMENTS}"
echo "[INFO] Retry limit:         ${MAX_RETRIES}"
echo "[INFO] ROAST involvement:   none"

if [ "${PREFLIGHT_ONLY}" -eq 1 ]; then
    echo "[INFO] Preflight passed without submitting jobs."
    echo "[INFO] Submit with: bash CamCan_Experiment/cohort_pipeline/submit_cohort_pipeline.sh ${COHORT_ID}"
    exit 0
fi

SUBMITTED_JOB_FILE="${CAMPAIGN_ROOT}/submitted_job_ids.txt"
if [ -s "${SUBMITTED_JOB_FILE}" ] && command -v "${SQUEUE_BIN}" >/dev/null 2>&1; then
    PREVIOUS_JOB_IDS=$(awk '/^[0-9]+$/ { values = values separator $1; separator = "," } END { print values }' "${SUBMITTED_JOB_FILE}")
    if [ -n "${PREVIOUS_JOB_IDS}" ]; then
        ACTIVE_PREVIOUS_JOBS=$("${SQUEUE_BIN}" -h -j "${PREVIOUS_JOB_IDS}" -o '%i' 2>/dev/null || true)
        if [ -n "${ACTIVE_PREVIOUS_JOBS}" ]; then
            echo "[ERROR] This cohort already has active submitted jobs:" >&2
            printf '%s\n' "${ACTIVE_PREVIOUS_JOBS}" >&2
            echo "[ERROR] Refusing a duplicate submission. Wait, cancel the old chain, or use a new cohort ID." >&2
            exit 2
        fi
    fi
fi

SUBMITTED_JOB_IDS=()
LAST_SUBMITTED_JOB_ID=""
cancel_submitted() {
    if [ "${#SUBMITTED_JOB_IDS[@]}" -gt 0 ]; then
        echo "[WARN] Cancelling already-submitted cohort jobs: ${SUBMITTED_JOB_IDS[*]}" >&2
        "${SCANCEL_BIN}" "${SUBMITTED_JOB_IDS[@]}" || true
    fi
}

submit_array() {
    local stage="$1"
    local array_spec="$2"
    local dependency="$3"
    local output_pattern="$4"
    local exports="$5"
    local job_name="$6"
    local -a command
    local submission exit_code job_id
    command=(
        "${SBATCH_BIN}"
        --parsable
        --job-name="${job_name}"
        --partition="${PARTITION}"
        --cpus-per-task="${CPUS_PER_TASK}"
        --mem="${MEMORY}"
        --time="${TIME_LIMIT}"
        --array="${array_spec}"
        --output="${output_pattern}"
        --export="${exports}"
    )
    if [ -n "${dependency}" ]; then
        command+=(--dependency="afterok:${dependency}")
    fi
    command+=("${SLURM_SCRIPT}")
    set +e
    submission=$("${command[@]}" 2>&1)
    exit_code=$?
    set -e
    echo "${submission}" >&2
    if [ "${exit_code}" -ne 0 ]; then
        echo "[ERROR] ${stage} submission failed." >&2
        cancel_submitted
        exit "${exit_code}"
    fi
    job_id="${submission%%;*}"
    if ! [[ "${job_id}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Could not parse ${stage} job ID: ${submission}" >&2
        cancel_submitted
        exit 2
    fi
    SUBMITTED_JOB_IDS+=("${job_id}")
    LAST_SUBMITTED_JOB_ID="${job_id}"
}

COMMON_EXPORTS="TI_COHORT_WORKFLOW_PY=${WORKFLOW_PY},TI_COHORT_LOG_DIR=${LOG_DIR},TI_COHORT_MAX_RETRIES=${MAX_RETRIES}"
SCAFFOLD_ARRAY="0-$((EXPECTED_SUBJECTS - 1))%${MAX_CONCURRENT_TASKS}"
SCAFFOLD_EXPORTS="ALL,${COMMON_EXPORTS},TI_COHORT_STAGE=scaffold,TI_COHORT_MANIFEST=${SCAFFOLD_MANIFEST}"
submit_array \
    scaffold \
    "${SCAFFOLD_ARRAY}" \
    "" \
    "${LOG_DIR}/scaffold-%A_%a.out" \
    "${SCAFFOLD_EXPORTS}" \
    "${JOB_PREFIX}_scaffold"
PREVIOUS_JOB="${LAST_SUBMITTED_JOB_ID}"
echo "[INFO] Submitted scaffold array: ${PREVIOUS_JOB}"

ELEMENT_OFFSET=0
MESH_CHUNK_INDEX=0
while [ "${ELEMENT_OFFSET}" -lt "${MESH_ELEMENTS}" ]; do
    REMAINING=$((MESH_ELEMENTS - ELEMENT_OFFSET))
    CHUNK_ELEMENTS="${MAX_ARRAY_ELEMENTS}"
    if [ "${REMAINING}" -lt "${CHUNK_ELEMENTS}" ]; then
        CHUNK_ELEMENTS="${REMAINING}"
    fi
    ARRAY_SPEC="0-$((CHUNK_ELEMENTS - 1))%${MAX_CONCURRENT_TASKS}"
    MESH_EXPORTS="ALL,${COMMON_EXPORTS},TI_COHORT_STAGE=mesh,TI_COHORT_MANIFEST=${MESH_MANIFEST},ELEMENT_OFFSET=${ELEMENT_OFFSET},TI_COHORT_MESH_WORKERS_PER_ELEMENT=${MESH_WORKERS},TI_COHORT_MESH_THREADS_PER_WORKER=${THREADS_PER_MESH_WORKER},TI_COHORT_MESH_LOCAL_STAGING=${LOCAL_STAGING}"
    submit_array \
        "mesh chunk ${MESH_CHUNK_INDEX}" \
        "${ARRAY_SPEC}" \
        "${PREVIOUS_JOB}" \
        "${LOG_DIR}/mesh-c${MESH_CHUNK_INDEX}-%A_%a.out" \
        "${MESH_EXPORTS}" \
        "${JOB_PREFIX}_mesh${MESH_CHUNK_INDEX}"
    PREVIOUS_JOB="${LAST_SUBMITTED_JOB_ID}"
    echo "[INFO] Submitted mesh chunk ${MESH_CHUNK_INDEX}: job=${PREVIOUS_JOB}, element_offset=${ELEMENT_OFFSET}, elements=${CHUNK_ELEMENTS}"
    ELEMENT_OFFSET=$((ELEMENT_OFFSET + CHUNK_ELEMENTS))
    MESH_CHUNK_INDEX=$((MESH_CHUNK_INDEX + 1))
done

TASK_OFFSET=0
SIMULATION_CHUNK_INDEX=0
while [ "${TASK_OFFSET}" -lt "${SIMULATION_TASKS}" ]; do
    REMAINING=$((SIMULATION_TASKS - TASK_OFFSET))
    CHUNK_TASKS="${MAX_ARRAY_ELEMENTS}"
    if [ "${REMAINING}" -lt "${CHUNK_TASKS}" ]; then
        CHUNK_TASKS="${REMAINING}"
    fi
    ARRAY_SPEC="0-$((CHUNK_TASKS - 1))%${MAX_CONCURRENT_TASKS}"
    SIMULATION_EXPORTS="ALL,${COMMON_EXPORTS},TI_COHORT_STAGE=simulate,TI_COHORT_MANIFEST=${SIMULATION_MANIFEST},TASK_OFFSET=${TASK_OFFSET},TI_TARGETS_CSV=${TARGETS_CSV},TI_EXPECTED_TARGETS_SHA256=${TARGETS_SHA256},TI_SIM_RUNNER_PY=${SIM_RUNNER_PY},TI_COMPLETION_CHECK_PY=${COMPLETION_CHECK_PY}"
    submit_array \
        "simulation chunk ${SIMULATION_CHUNK_INDEX}" \
        "${ARRAY_SPEC}" \
        "${PREVIOUS_JOB}" \
        "${LOG_DIR}/simulate-c${SIMULATION_CHUNK_INDEX}-%A_%a.out" \
        "${SIMULATION_EXPORTS}" \
        "${JOB_PREFIX}_sim${SIMULATION_CHUNK_INDEX}"
    PREVIOUS_JOB="${LAST_SUBMITTED_JOB_ID}"
    echo "[INFO] Submitted simulation chunk ${SIMULATION_CHUNK_INDEX}: job=${PREVIOUS_JOB}, task_offset=${TASK_OFFSET}, tasks=${CHUNK_TASKS}"
    TASK_OFFSET=$((TASK_OFFSET + CHUNK_TASKS))
    SIMULATION_CHUNK_INDEX=$((SIMULATION_CHUNK_INDEX + 1))
done

printf '%s\n' "${SUBMITTED_JOB_IDS[@]}" > "${SUBMITTED_JOB_FILE}"
echo "[INFO] Submitted ${#SUBMITTED_JOB_IDS[@]} dependent Slurm arrays."
echo "[INFO] Final dependency-chain job: ${PREVIOUS_JOB}"
echo "[INFO] Job IDs: ${SUBMITTED_JOB_IDS[*]}"
