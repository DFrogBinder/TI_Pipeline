#!/bin/bash
#SBATCH --job-name=cohort_atlas_repair
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-0
#SBATCH --requeue

set -uo pipefail

: "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array.}"
: "${TI_COHORT_ATLAS_MANIFEST:?Missing TI_COHORT_ATLAS_MANIFEST.}"
: "${TI_COHORT_ATLAS_LOG_DIR:?Missing TI_COHORT_ATLAS_LOG_DIR.}"
: "${TI_COHORT_ATLAS_FS_LICENSE:?Missing TI_COHORT_ATLAS_FS_LICENSE.}"

MAX_RETRIES="${TI_COHORT_ATLAS_MAX_RETRIES:-2}"
STATE_DIR="${TI_COHORT_ATLAS_STATE_DIR:-${TI_COHORT_ATLAS_LOG_DIR}/retry_state}"
FREESURFER_MODULE="${TI_COHORT_ATLAS_FREESURFER_MODULE:-FreeSurfer/7.4.1-centos7_x86_64}"

if ! [[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] SLURM_ARRAY_TASK_ID must be a non-negative integer." >&2
    exit 2
fi
if ! [[ "${MAX_RETRIES}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] TI_COHORT_ATLAS_MAX_RETRIES must be a non-negative integer." >&2
    exit 2
fi
if [ ! -f "${TI_COHORT_ATLAS_MANIFEST}" ] || [ ! -f "${TI_COHORT_ATLAS_FS_LICENSE}" ]; then
    echo "[ERROR] Atlas manifest or FreeSurfer license is missing." >&2
    exit 2
fi

mkdir -p "${TI_COHORT_ATLAS_LOG_DIR}" "${STATE_DIR}"
ROW="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${TI_COHORT_ATLAS_MANIFEST}")"
if [ -z "${ROW}" ]; then
    echo "[ERROR] Missing manifest row for task ${SLURM_ARRAY_TASK_ID}." >&2
    exit 2
fi
IFS=$'\t' read -r TASK_ID SUBJECT MODE T1_PATH SOURCE_ATLAS FLAT_ATLAS FS_SUBJECT_DIR RESULT_PATH <<< "${ROW}"
if [ "${TASK_ID}" != "${SLURM_ARRAY_TASK_ID}" ] || [ -z "${SUBJECT}" ]; then
    echo "[ERROR] Malformed or misindexed manifest row ${SLURM_ARRAY_TASK_ID}." >&2
    exit 2
fi

TASK_KEY="atlas_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID}"
RETRY_FILE="${STATE_DIR}/${TASK_KEY}.retry"
if [ -f "${RETRY_FILE}" ]; then
    RETRY_COUNT="$(tr -d '[:space:]' < "${RETRY_FILE}")"
else
    RETRY_COUNT=0
fi
if ! [[ "${RETRY_COUNT}" =~ ^[0-9]+$ ]]; then
    RETRY_COUNT=0
fi

TASK_LOG="${TI_COHORT_ATLAS_LOG_DIR}/atlas__${TASK_ID}__${SUBJECT}.log"
{
    echo "[INFO] Job ID:          ${SLURM_JOB_ID}"
    echo "[INFO] Array task:      ${SLURM_ARRAY_TASK_ID}"
    echo "[INFO] Subject:         ${SUBJECT}"
    echo "[INFO] Mode:            ${MODE}"
    echo "[INFO] T1:              $([ "${T1_PATH}" = "-" ] && echo '<not needed>' || echo "${T1_PATH}")"
    echo "[INFO] Source atlas:    $([ "${SOURCE_ATLAS}" = "-" ] && echo '<none>' || echo "${SOURCE_ATLAS}")"
    echo "[INFO] Flat atlas:      ${FLAT_ATLAS}"
    echo "[INFO] FreeSurfer dir:  ${FS_SUBJECT_DIR}"
    echo "[INFO] CPUs:            ${SLURM_CPUS_PER_TASK:-16}"
    echo "[INFO] Retry:           ${RETRY_COUNT}/${MAX_RETRIES}"
} | tee -a "${TASK_LOG}"

run_task() {
    local threads="${SLURM_CPUS_PER_TASK:-16}"
    local subjects_dir
    local atlas_mgz
    local tmp_atlas
    local started_at
    local completed_at
    local atlas_sha256
    local atlas_bytes
    local actual_mode="${MODE}"

    module purge
    module load "${FREESURFER_MODULE}"
    export FS_LICENSE="${TI_COHORT_ATLAS_FS_LICENSE}"
    subjects_dir="$(dirname "${FS_SUBJECT_DIR}")"
    export SUBJECTS_DIR="${subjects_dir}"
    export OMP_NUM_THREADS="${threads}"
    export MKL_NUM_THREADS="${threads}"
    export OPENBLAS_NUM_THREADS="${threads}"
    export NUMEXPR_NUM_THREADS="${threads}"

    command -v recon-all >/dev/null 2>&1 || {
        echo "[ERROR] recon-all is unavailable after loading ${FREESURFER_MODULE}." >&2
        return 2
    }
    command -v mri_convert >/dev/null 2>&1 || {
        echo "[ERROR] mri_convert is unavailable after loading ${FREESURFER_MODULE}." >&2
        return 2
    }
    command -v mri_info >/dev/null 2>&1 || {
        echo "[ERROR] mri_info is unavailable after loading ${FREESURFER_MODULE}." >&2
        return 2
    }

    mkdir -p "$(dirname "${FLAT_ATLAS}")" "$(dirname "${RESULT_PATH}")" "${subjects_dir}"
    started_at="$(date --iso-8601=seconds)"
    atlas_mgz="${FS_SUBJECT_DIR}/mri/aparc.a2009s+aseg.mgz"
    tmp_atlas="${FLAT_ATLAS}.tmp.${SLURM_JOB_ID}.nii.gz"
    rm -f "${tmp_atlas}"

    if [ -s "${FLAT_ATLAS}" ]; then
        actual_mode="reuse_flat"
        echo "[INFO] A non-empty flat atlas already exists; validating and preserving it."
    else
        case "${MODE}" in
            import_nifti)
                [ -s "${SOURCE_ATLAS}" ] || {
                    echo "[ERROR] Reusable NIfTI atlas disappeared: ${SOURCE_ATLAS}" >&2
                    return 1
                }
                cp "${SOURCE_ATLAS}" "${tmp_atlas}"
                ;;
            convert_mgz)
                [ -s "${SOURCE_ATLAS}" ] || {
                    echo "[ERROR] Reusable MGZ atlas disappeared: ${SOURCE_ATLAS}" >&2
                    return 1
                }
                mri_convert "${SOURCE_ATLAS}" "${tmp_atlas}"
                ;;
            reconstruct)
                [ -s "${T1_PATH}" ] || {
                    echo "[ERROR] Canonical T1 disappeared: ${T1_PATH}" >&2
                    return 1
                }
                if [ ! -s "${atlas_mgz}" ]; then
                    if [ -d "${FS_SUBJECT_DIR}" ] && \
                       [ -n "$(find "${FS_SUBJECT_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]
                    then
                        echo "[INFO] Resuming existing FreeSurfer subject state."
                        recon-all \
                            -s "${SUBJECT}" \
                            -sd "${subjects_dir}" \
                            -all \
                            -openmp "${threads}" \
                            -cw256
                    else
                        if [ -d "${FS_SUBJECT_DIR}" ]; then
                            rmdir "${FS_SUBJECT_DIR}"
                        fi
                        echo "[INFO] Starting full native FreeSurfer reconstruction."
                        recon-all \
                            -s "${SUBJECT}" \
                            -i "${T1_PATH}" \
                            -sd "${subjects_dir}" \
                            -all \
                            -openmp "${threads}" \
                            -cw256
                    fi
                else
                    actual_mode="resume_convert"
                    echo "[INFO] Destrieux MGZ already exists; skipping reconstruction."
                fi
                [ -s "${atlas_mgz}" ] || {
                    echo "[ERROR] FreeSurfer did not create ${atlas_mgz}." >&2
                    return 1
                }
                mri_convert "${atlas_mgz}" "${tmp_atlas}"
                ;;
            *)
                echo "[ERROR] Unsupported atlas task mode: ${MODE}" >&2
                return 2
                ;;
        esac

        [ -s "${tmp_atlas}" ] || {
            echo "[ERROR] Atlas conversion did not create a non-empty file." >&2
            return 1
        }
        mv "${tmp_atlas}" "${FLAT_ATLAS}"
    fi

    mri_info "${FLAT_ATLAS}" >/dev/null
    atlas_sha256="$(sha256sum "${FLAT_ATLAS}" | awk '{print $1}')"
    atlas_bytes="$(stat -c '%s' "${FLAT_ATLAS}")"
    completed_at="$(date --iso-8601=seconds)"
    {
        printf '%s\n' \
            $'status\tsubject\tmode\tatlas_path\tatlas_bytes\tatlas_sha256\tstarted_at\tcompleted_at'
        printf 'complete\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${SUBJECT}" \
            "${actual_mode}" \
            "${FLAT_ATLAS}" \
            "${atlas_bytes}" \
            "${atlas_sha256}" \
            "${started_at}" \
            "${completed_at}"
    } > "${RESULT_PATH}.tmp.${SLURM_JOB_ID}"
    mv "${RESULT_PATH}.tmp.${SLURM_JOB_ID}" "${RESULT_PATH}"
    echo "[INFO] Atlas repair complete: ${SUBJECT}"
}

set +e
run_task 2>&1 | tee -a "${TASK_LOG}"
RUN_EXIT=${PIPESTATUS[0]}
set -e

if [ "${RUN_EXIT}" -eq 0 ]; then
    rm -f "${RETRY_FILE}"
    exit 0
fi
if [ "${RETRY_COUNT}" -lt "${MAX_RETRIES}" ]; then
    NEXT_RETRY=$((RETRY_COUNT + 1))
    printf '%s\n' "${NEXT_RETRY}" > "${RETRY_FILE}"
    echo "[WARN] Atlas task failed with exit ${RUN_EXIT}; requeueing (${NEXT_RETRY}/${MAX_RETRIES})." |
        tee -a "${TASK_LOG}"
    scontrol requeue "${SLURM_JOB_ID}"
    exit $?
fi

echo "[ERROR] Atlas task failed after ${RETRY_COUNT} retries." | tee -a "${TASK_LOG}" >&2
exit "${RUN_EXIT}"
