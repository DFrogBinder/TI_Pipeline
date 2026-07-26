#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
    echo "[ERROR] Invalid cohort ID: ${COHORT_ID}." >&2
    exit 2
fi

STUDY_CONFIG="${STUDY_CONFIG:-${SCRIPT_DIR}/studies/corrected_v4_four_roi.json}"
COHORT_CONFIG="${COHORT_CONFIG:-${SCRIPT_DIR}/cohorts/${COHORT_ID}/cohort.json}"
STUDY_ROOT="${STUDY_ROOT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["hpc_study_root"])' "${STUDY_CONFIG}")}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${STUDY_ROOT}/campaigns/${COHORT_ID}}"
ATLAS_CAMPAIGN_ROOT="${ATLAS_CAMPAIGN_ROOT:-${CAMPAIGN_ROOT}/atlas_repair}"
SUBJECTS_FILE="${SUBJECTS_FILE:-${SCRIPT_DIR}/cohorts/${COHORT_ID}/subjects.txt}"
SCAFFOLD_ROOT="${SCAFFOLD_ROOT:-/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Scaffolds}"
FLAT_ATLAS_ROOT="${FLAT_ATLAS_ROOT:-/mnt/parscratch/users/cop23bi/ZIPs/atlases}"
ATLAS_WORK_ROOT="${ATLAS_WORK_ROOT:-/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Atlases}"
FS_LICENSE_FILE="${FS_LICENSE_FILE:-/users/cop23bi/freesurfer_licence.txt}"
ARRAY_SLURM="${ARRAY_SLURM:-${SCRIPT_DIR}/cohort_atlas_repair.slurm}"
COLLECT_SLURM="${COLLECT_SLURM:-${SCRIPT_DIR}/cohort_atlas_collect.slurm}"

PARTITION="${PARTITION:-sheffield}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEMORY="${MEMORY:-64G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-50}"
TI_COHORT_ATLAS_MAX_RETRIES="${TI_COHORT_ATLAS_MAX_RETRIES:-2}"
COLLECTOR_CPUS="${COLLECTOR_CPUS:-1}"
COLLECTOR_MEMORY="${COLLECTOR_MEMORY:-2G}"
COLLECTOR_TIME="${COLLECTOR_TIME:-01:00:00}"
FREESURFER_MODULE="${FREESURFER_MODULE:-FreeSurfer/7.4.1-centos7_x86_64}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
SQUEUE_BIN="${SQUEUE_BIN:-squeue}"

for required_file in \
    "${STUDY_CONFIG}" \
    "${COHORT_CONFIG}" \
    "${SUBJECTS_FILE}" \
    "${ARRAY_SLURM}" \
    "${COLLECT_SLURM}" \
    "${FS_LICENSE_FILE}"
do
    if [ ! -f "${required_file}" ]; then
        echo "[ERROR] Required file is missing: ${required_file}" >&2
        exit 2
    fi
done
if [ ! -d "${SCAFFOLD_ROOT}/subjects" ]; then
    echo "[ERROR] Scaffold subject root is missing: ${SCAFFOLD_ROOT}/subjects" >&2
    exit 2
fi
for value_name in CPUS_PER_TASK MAX_CONCURRENT_TASKS COLLECTOR_CPUS; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] ${value_name} must be a positive integer; received ${value}." >&2
        exit 2
    fi
done
if ! [[ "${TI_COHORT_ATLAS_MAX_RETRIES}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] TI_COHORT_ATLAS_MAX_RETRIES must be a non-negative integer." >&2
    exit 2
fi

EXPECTED_SUBJECTS="$(
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["expected_subjects"])' \
        "${COHORT_CONFIG}"
)"
SUBJECT_COUNT="$(
    awk 'NF > 0 && $1 !~ /^#/ { count++ } END { print count + 0 }' \
        "${SUBJECTS_FILE}"
)"
UNIQUE_SUBJECT_COUNT="$(
    awk 'NF > 0 && $1 !~ /^#/ { seen[$1]=1 } END { for (item in seen) count++; print count + 0 }' \
        "${SUBJECTS_FILE}"
)"
if [ "${SUBJECT_COUNT}" -ne "${EXPECTED_SUBJECTS}" ] || \
   [ "${UNIQUE_SUBJECT_COUNT}" -ne "${EXPECTED_SUBJECTS}" ]
then
    echo "[ERROR] Cohort subject list does not contain exactly ${EXPECTED_SUBJECTS} unique subjects." >&2
    exit 2
fi

mkdir -p \
    "${ATLAS_CAMPAIGN_ROOT}/logs" \
    "${ATLAS_CAMPAIGN_ROOT}/results" \
    "${ATLAS_CAMPAIGN_ROOT}/retry_state" \
    "${ATLAS_WORK_ROOT}/subjects" \
    "${FLAT_ATLAS_ROOT}"

MANIFEST="${ATLAS_CAMPAIGN_ROOT}/atlas_tasks.tsv"
PREFLIGHT_JSON="${ATLAS_CAMPAIGN_ROOT}/preflight.json"
TMP_MANIFEST="${MANIFEST}.tmp.$$"
printf '%s\n' \
    $'task_id\tsubject\tmode\tt1_path\tsource_atlas\tflat_atlas\tfreesurfer_subject_dir\tresult_path' \
    > "${TMP_MANIFEST}"

SOURCE_ROOTS=(
    "${ATLAS_WORK_ROOT}/subjects"
    "/mnt/parscratch/users/cop23bi/ti_dataset/FastSurfer_out"
    "/mnt/parscratch/users/cop23bi/CamCanMRI/FastSurfer_out"
    "/mnt/parscratch/users/cop23bi/ZIPs/FastSurfer_out"
    "/mnt/parscratch/users/cop23bi/destr-atlases/subjects"
)
if [ -n "${ATLAS_SOURCE_ROOTS:-}" ]; then
    IFS=':' read -r -a EXTRA_SOURCE_ROOTS <<< "${ATLAS_SOURCE_ROOTS}"
    SOURCE_ROOTS+=("${EXTRA_SOURCE_ROOTS[@]}")
fi

TASKS=0
FLAT_PRESENT=0
IMPORT_NIFTI=0
CONVERT_MGZ=0
RECONSTRUCT=0
BLOCKED=0

while read -r subject; do
    [ -n "${subject}" ] || continue
    if ! [[ "${subject}" =~ ^sub-[A-Za-z0-9]+$ ]]; then
        echo "[ERROR] Invalid subject ID in cohort list: ${subject}" >&2
        BLOCKED=$((BLOCKED + 1))
        continue
    fi

    flat_atlas="${FLAT_ATLAS_ROOT}/${subject}.nii.gz"
    if [ -s "${flat_atlas}" ] || [ -s "${FLAT_ATLAS_ROOT}/${subject}.nii" ]; then
        FLAT_PRESENT=$((FLAT_PRESENT + 1))
        continue
    fi

    t1_path=""
    for candidate in \
        "${SCAFFOLD_ROOT}/subjects/${subject}/anat/${subject}_T1w.nii.gz" \
        "${SCAFFOLD_ROOT}/subjects/${subject}/anat/${subject}_T1w.nii"
    do
        if [ -s "${candidate}" ]; then
            t1_path="${candidate}"
            break
        fi
    done

    source_atlas=""
    mode=""
    for source_root in "${SOURCE_ROOTS[@]}"; do
        [ -d "${source_root}" ] || continue
        for candidate in \
            "${source_root}/${subject}/mri/aparc.a2009s+aseg.nii.gz" \
            "${source_root}/${subject}/mri/aparc.a2009s+aseg.mgz"
        do
            if [ -s "${candidate}" ]; then
                source_atlas="${candidate}"
                case "${candidate}" in
                    *.nii.gz)
                        mode="import_nifti"
                        IMPORT_NIFTI=$((IMPORT_NIFTI + 1))
                        ;;
                    *.mgz)
                        mode="convert_mgz"
                        CONVERT_MGZ=$((CONVERT_MGZ + 1))
                        ;;
                esac
                break 2
            fi
        done
    done

    if [ -z "${mode}" ]; then
        mode="reconstruct"
        RECONSTRUCT=$((RECONSTRUCT + 1))
        if [ -z "${t1_path}" ]; then
            echo "[ERROR] Missing canonical scaffold T1 for ${subject}." >&2
            BLOCKED=$((BLOCKED + 1))
            continue
        fi
    fi
    if [ -z "${t1_path}" ]; then
        t1_path="-"
    fi
    if [ -z "${source_atlas}" ]; then
        source_atlas="-"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${TASKS}" \
        "${subject}" \
        "${mode}" \
        "${t1_path}" \
        "${source_atlas}" \
        "${flat_atlas}" \
        "${ATLAS_WORK_ROOT}/subjects/${subject}" \
        "${ATLAS_CAMPAIGN_ROOT}/results/${subject}.tsv" \
        >> "${TMP_MANIFEST}"
    TASKS=$((TASKS + 1))
done < "${SUBJECTS_FILE}"

mv "${TMP_MANIFEST}" "${MANIFEST}"

python3 -c \
    'import json,sys; json.dump({"status":sys.argv[1],"subjects":int(sys.argv[2]),"flat_atlases_present":int(sys.argv[3]),"tasks":int(sys.argv[4]),"import_nifti":int(sys.argv[5]),"convert_mgz":int(sys.argv[6]),"reconstruct":int(sys.argv[7]),"blocked":int(sys.argv[8]),"manifest":sys.argv[9],"flat_atlas_root":sys.argv[10],"atlas_work_root":sys.argv[11],"source_atlas":"FreeSurfer Destrieux aparc.a2009s+aseg"},open(sys.argv[12],"w"),indent=2,sort_keys=True); print()' \
    "$([ "${BLOCKED}" -eq 0 ] && echo ready || echo blocked)" \
    "${EXPECTED_SUBJECTS}" \
    "${FLAT_PRESENT}" \
    "${TASKS}" \
    "${IMPORT_NIFTI}" \
    "${CONVERT_MGZ}" \
    "${RECONSTRUCT}" \
    "${BLOCKED}" \
    "${MANIFEST}" \
    "${FLAT_ATLAS_ROOT}" \
    "${ATLAS_WORK_ROOT}" \
    "${PREFLIGHT_JSON}"

cat "${PREFLIGHT_JSON}"
printf '%s\n' \
    'Scope:' \
    '  dataset: final CamCan cohort subject-space Destrieux atlas repair' \
    "  cohort: ${COHORT_ID}" \
    "  subjects: ${EXPECTED_SUBJECTS}" \
    "  existing flat atlases: ${FLAT_PRESENT}" \
    "  missing-atlas tasks: ${TASKS}" \
    "  reusable nested NIfTI imports: ${IMPORT_NIFTI}" \
    "  reusable nested MGZ conversions: ${CONVERT_MGZ}" \
    "  full FreeSurfer reconstructions: ${RECONSTRUCT}" \
    "  expected final flat atlases: ${EXPECTED_SUBJECTS}" \
    '  atlas: FreeSurfer Destrieux aparc.a2009s+aseg' \
    '  source simulations modified: no' \
    '  source corrected segmentations modified: no'

echo "[INFO] Canonical T1 root:  ${SCAFFOLD_ROOT}/subjects"
echo "[INFO] Flat atlas root:    ${FLAT_ATLAS_ROOT}"
echo "[INFO] FreeSurfer work:    ${ATLAS_WORK_ROOT}/subjects"
echo "[INFO] Manifest:           ${MANIFEST}"
echo "[INFO] Logs:               ${ATLAS_CAMPAIGN_ROOT}/logs"
echo "[INFO] Resource profile:   ${PARTITION}, ${CPUS_PER_TASK} CPU, ${MEMORY}, ${TIME_LIMIT}"
echo "[INFO] Array concurrency:  ${MAX_CONCURRENT_TASKS}"
echo "[INFO] Retry limit:        ${TI_COHORT_ATLAS_MAX_RETRIES}"

if [ "${BLOCKED}" -ne 0 ]; then
    echo "[ERROR] Atlas repair preflight found ${BLOCKED} blocking input problem(s)." >&2
    exit 2
fi
if [ "${TASKS}" -eq 0 ]; then
    echo "[INFO] All ${EXPECTED_SUBJECTS} subject-space atlases already exist; no repair job is needed."
    echo "[INFO] You may rerun the cohort post-processing preflight now."
    exit 0
fi
if [ "${PREFLIGHT_ONLY}" -eq 1 ]; then
    echo "[INFO] Preflight passed without submitting jobs."
    echo "[INFO] Submit with: bash CamCan_Experiment/cohort_pipeline/submit_cohort_atlas_repair.sh ${COHORT_ID}"
    exit 0
fi

JOB_ID_FILE="${ATLAS_CAMPAIGN_ROOT}/submitted_job_ids.txt"
if [ -s "${JOB_ID_FILE}" ] && command -v "${SQUEUE_BIN}" >/dev/null 2>&1; then
    PREVIOUS_IDS="$(
        awk '/^[0-9]+$/ { values = values separator $1; separator = "," } END { print values }' \
            "${JOB_ID_FILE}"
    )"
    if [ -n "${PREVIOUS_IDS}" ] && \
       [ -n "$("${SQUEUE_BIN}" -h -j "${PREVIOUS_IDS}" -o '%i' 2>/dev/null || true)" ]
    then
        echo "[ERROR] A previous atlas-repair submission is still active: ${PREVIOUS_IDS}" >&2
        exit 2
    fi
fi

COMMON_EXPORTS="ALL,TI_COHORT_ATLAS_MANIFEST=${MANIFEST},TI_COHORT_ATLAS_LOG_DIR=${ATLAS_CAMPAIGN_ROOT}/logs,TI_COHORT_ATLAS_STATE_DIR=${ATLAS_CAMPAIGN_ROOT}/retry_state,TI_COHORT_ATLAS_MAX_RETRIES=${TI_COHORT_ATLAS_MAX_RETRIES},TI_COHORT_ATLAS_FS_LICENSE=${FS_LICENSE_FILE},TI_COHORT_ATLAS_FREESURFER_MODULE=${FREESURFER_MODULE}"
ARRAY_SUBMISSION="$(
    "${SBATCH_BIN}" \
        --parsable \
        --job-name="cohort_${COHORT_ID}_atlas" \
        --partition="${PARTITION}" \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --mem="${MEMORY}" \
        --time="${TIME_LIMIT}" \
        --array="0-$((TASKS - 1))%${MAX_CONCURRENT_TASKS}" \
        --output="${ATLAS_CAMPAIGN_ROOT}/logs/atlas-%A_%a.out" \
        --export="${COMMON_EXPORTS}" \
        "${ARRAY_SLURM}"
)"
ARRAY_JOB="${ARRAY_SUBMISSION%%;*}"
if ! [[ "${ARRAY_JOB}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Could not parse atlas array job ID from: ${ARRAY_SUBMISSION}" >&2
    exit 2
fi

COLLECT_EXPORTS="ALL,TI_COHORT_ATLAS_SUBJECTS_FILE=${SUBJECTS_FILE},TI_COHORT_ATLAS_FLAT_ROOT=${FLAT_ATLAS_ROOT},TI_COHORT_ATLAS_VALIDATION_TSV=${ATLAS_CAMPAIGN_ROOT}/validation.tsv,TI_COHORT_ATLAS_VALIDATION_JSON=${ATLAS_CAMPAIGN_ROOT}/validation.json"
COLLECT_SUBMISSION="$(
    "${SBATCH_BIN}" \
        --parsable \
        --job-name="cohort_${COHORT_ID}_atlas_collect" \
        --partition="${PARTITION}" \
        --cpus-per-task="${COLLECTOR_CPUS}" \
        --mem="${COLLECTOR_MEMORY}" \
        --time="${COLLECTOR_TIME}" \
        --dependency="afterany:${ARRAY_JOB}" \
        --output="${ATLAS_CAMPAIGN_ROOT}/logs/atlas-collector-%j.out" \
        --export="${COLLECT_EXPORTS}" \
        "${COLLECT_SLURM}"
)"
COLLECT_JOB="${COLLECT_SUBMISSION%%;*}"
if ! [[ "${COLLECT_JOB}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Could not parse atlas collector job ID from: ${COLLECT_SUBMISSION}" >&2
    exit 2
fi

printf '%s\n%s\n' "${ARRAY_JOB}" "${COLLECT_JOB}" > "${JOB_ID_FILE}"
echo "[INFO] Submitted missing-atlas array job: ${ARRAY_JOB}"
echo "[INFO] Submitted afterany atlas collector: ${COLLECT_JOB}"
echo "[INFO] Completion validation: ${ATLAS_CAMPAIGN_ROOT}/validation.json"
