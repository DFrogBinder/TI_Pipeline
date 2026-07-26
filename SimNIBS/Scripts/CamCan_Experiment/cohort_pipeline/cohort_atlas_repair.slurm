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
WORKERS_PER_ELEMENT="${TI_COHORT_ATLAS_WORKERS_PER_ARRAY_TASK:-2}"
THREADS_PER_WORKER="${TI_COHORT_ATLAS_THREADS_PER_WORKER:-8}"
TASKSET_BIN="${TASKSET_BIN:-taskset}"

for value_name in SLURM_ARRAY_TASK_ID MAX_RETRIES WORKERS_PER_ELEMENT THREADS_PER_WORKER; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] ${value_name} must be a non-negative integer." >&2
        exit 2
    fi
done
if [ "${WORKERS_PER_ELEMENT}" -lt 1 ] || [ "${THREADS_PER_WORKER}" -lt 1 ]; then
    echo "[ERROR] Atlas workers and threads per worker must be positive." >&2
    exit 2
fi
REQUIRED_CPUS=$((WORKERS_PER_ELEMENT * THREADS_PER_WORKER))
if [ "${REQUIRED_CPUS}" -gt "${SLURM_CPUS_PER_TASK:-0}" ]; then
    echo "[ERROR] Packed atlas workers need ${REQUIRED_CPUS} CPUs; allocation exposes ${SLURM_CPUS_PER_TASK:-0}." >&2
    exit 2
fi
if [ "${WORKERS_PER_ELEMENT}" -gt 1 ] && ! command -v "${TASKSET_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] Packed atlas execution requires taskset." >&2
    exit 2
fi
if [ ! -f "${TI_COHORT_ATLAS_MANIFEST}" ] || [ ! -f "${TI_COHORT_ATLAS_FS_LICENSE}" ]; then
    echo "[ERROR] Atlas manifest or FreeSurfer license is missing." >&2
    exit 2
fi

TOTAL_TASKS="$(
    awk 'NR > 1 && NF > 0 { count++ } END { print count + 0 }' \
        "${TI_COHORT_ATLAS_MANIFEST}"
)"
FIRST_TASK=$((SLURM_ARRAY_TASK_ID * WORKERS_PER_ELEMENT))
LAST_TASK=$((FIRST_TASK + WORKERS_PER_ELEMENT - 1))
if [ "${FIRST_TASK}" -ge "${TOTAL_TASKS}" ]; then
    echo "[ERROR] Array element starts beyond ${TOTAL_TASKS} manifest tasks." >&2
    exit 2
fi
if [ "${LAST_TASK}" -ge "${TOTAL_TASKS}" ]; then
    LAST_TASK=$((TOTAL_TASKS - 1))
fi

mkdir -p "${TI_COHORT_ATLAS_LOG_DIR}" "${STATE_DIR}"
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

module purge
module load "${FREESURFER_MODULE}"
export FS_LICENSE="${TI_COHORT_ATLAS_FS_LICENSE}"
command -v recon-all >/dev/null 2>&1 || {
    echo "[ERROR] recon-all is unavailable after loading ${FREESURFER_MODULE}." >&2
    exit 2
}
command -v mri_convert >/dev/null 2>&1 || {
    echo "[ERROR] mri_convert is unavailable after loading ${FREESURFER_MODULE}." >&2
    exit 2
}
command -v mri_info >/dev/null 2>&1 || {
    echo "[ERROR] mri_info is unavailable after loading ${FREESURFER_MODULE}." >&2
    exit 2
}

read -r -a CPU_IDS <<< "$(
    python3 -c 'import os; print(" ".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0))))'
)"
if [ "${#CPU_IDS[@]}" -lt "${REQUIRED_CPUS}" ]; then
    echo "[ERROR] CPU affinity exposes ${#CPU_IDS[@]} CPUs; ${REQUIRED_CPUS} are required." >&2
    exit 2
fi

echo "[INFO] Job ID:             ${SLURM_JOB_ID}"
echo "[INFO] Array element:      ${SLURM_ARRAY_TASK_ID}"
echo "[INFO] Manifest tasks:     ${FIRST_TASK}-${LAST_TASK}"
echo "[INFO] Total tasks:        ${TOTAL_TASKS}"
echo "[INFO] Atlas workers:      ${WORKERS_PER_ELEMENT}"
echo "[INFO] Threads per atlas:  ${THREADS_PER_WORKER}"
echo "[INFO] CPUs allocated:     ${SLURM_CPUS_PER_TASK}"
echo "[INFO] Memory allocated:   ${SLURM_MEM_PER_NODE:-unknown}"
echo "[INFO] Retry:              ${RETRY_COUNT}/${MAX_RETRIES}"

run_one_task() {
    local task_index="$1"
    local worker_position="$2"
    local row
    local task_id subject mode t1_path source_atlas flat_atlas fs_subject_dir result_path
    local task_log
    local subjects_dir atlas_mgz tmp_atlas started_at completed_at atlas_sha256 atlas_bytes
    local actual_mode
    local cpu_offset cpu_position cpu_list
    local -a launch_prefix recon_command

    row="$(sed -n "$((task_index + 2))p" "${TI_COHORT_ATLAS_MANIFEST}")"
    if [ -z "${row}" ]; then
        echo "[ERROR] Missing manifest row for task ${task_index}." >&2
        return 2
    fi
    IFS=$'\t' read -r \
        task_id \
        subject \
        mode \
        t1_path \
        source_atlas \
        flat_atlas \
        fs_subject_dir \
        result_path \
        <<< "${row}"
    if [ "${task_id}" != "${task_index}" ] || [ -z "${subject}" ]; then
        echo "[ERROR] Malformed or misindexed manifest row ${task_index}." >&2
        return 2
    fi

    cpu_offset=$((worker_position * THREADS_PER_WORKER))
    cpu_list="${CPU_IDS[$cpu_offset]}"
    for ((cpu_position = 1; cpu_position < THREADS_PER_WORKER; cpu_position++)); do
        cpu_list+=",${CPU_IDS[$((cpu_offset + cpu_position))]}"
    done
    launch_prefix=()
    if [ "${WORKERS_PER_ELEMENT}" -gt 1 ]; then
        launch_prefix=("${TASKSET_BIN}" --cpu-list "${cpu_list}")
    fi

    task_log="${TI_COHORT_ATLAS_LOG_DIR}/atlas__${task_id}__${subject}.log"
    {
        echo "[INFO] Task index:       ${task_id}"
        echo "[INFO] Subject:          ${subject}"
        echo "[INFO] Mode:             ${mode}"
        echo "[INFO] T1:               $([ "${t1_path}" = "-" ] && echo '<not needed>' || echo "${t1_path}")"
        echo "[INFO] Source atlas:     $([ "${source_atlas}" = "-" ] && echo '<none>' || echo "${source_atlas}")"
        echo "[INFO] Flat atlas:       ${flat_atlas}"
        echo "[INFO] FreeSurfer dir:   ${fs_subject_dir}"
        echo "[INFO] Worker CPU set:   ${cpu_list}"

        export OMP_NUM_THREADS="${THREADS_PER_WORKER}"
        export MKL_NUM_THREADS="${THREADS_PER_WORKER}"
        export OPENBLAS_NUM_THREADS="${THREADS_PER_WORKER}"
        export NUMEXPR_NUM_THREADS="${THREADS_PER_WORKER}"

        subjects_dir="$(dirname "${fs_subject_dir}")"
        export SUBJECTS_DIR="${subjects_dir}"
        mkdir -p "$(dirname "${flat_atlas}")" "$(dirname "${result_path}")" "${subjects_dir}"
        started_at="$(date --iso-8601=seconds)"
        atlas_mgz="${fs_subject_dir}/mri/aparc.a2009s+aseg.mgz"
        tmp_atlas="${flat_atlas}.tmp.${SLURM_JOB_ID}.${task_id}.nii.gz"
        rm -f "${tmp_atlas}"
        actual_mode="${mode}"

        if [ -s "${flat_atlas}" ]; then
            actual_mode="reuse_flat"
            echo "[INFO] A non-empty flat atlas already exists; validating and preserving it."
        else
            case "${mode}" in
                import_nifti)
                    [ -s "${source_atlas}" ] || {
                        echo "[ERROR] Reusable NIfTI atlas disappeared: ${source_atlas}" >&2
                        return 1
                    }
                    cp "${source_atlas}" "${tmp_atlas}"
                    ;;
                convert_mgz)
                    [ -s "${source_atlas}" ] || {
                        echo "[ERROR] Reusable MGZ atlas disappeared: ${source_atlas}" >&2
                        return 1
                    }
                    "${launch_prefix[@]}" mri_convert "${source_atlas}" "${tmp_atlas}"
                    ;;
                reconstruct)
                    [ -s "${t1_path}" ] || {
                        echo "[ERROR] Canonical T1 disappeared: ${t1_path}" >&2
                        return 1
                    }
                    if [ ! -s "${atlas_mgz}" ]; then
                        recon_command=(
                            recon-all
                            -s "${subject}"
                            -sd "${subjects_dir}"
                            -all
                            -openmp "${THREADS_PER_WORKER}"
                            -cw256
                        )
                        if [ -d "${fs_subject_dir}" ] && \
                           [ -n "$(find "${fs_subject_dir}" -mindepth 1 -print -quit 2>/dev/null)" ]
                        then
                            echo "[INFO] Resuming existing FreeSurfer subject state."
                        else
                            if [ -d "${fs_subject_dir}" ]; then
                                rmdir "${fs_subject_dir}"
                            fi
                            echo "[INFO] Starting full native FreeSurfer reconstruction."
                            recon_command+=(-i "${t1_path}")
                        fi
                        "${launch_prefix[@]}" "${recon_command[@]}"
                    else
                        actual_mode="resume_convert"
                        echo "[INFO] Destrieux MGZ already exists; skipping reconstruction."
                    fi
                    [ -s "${atlas_mgz}" ] || {
                        echo "[ERROR] FreeSurfer did not create ${atlas_mgz}." >&2
                        return 1
                    }
                    "${launch_prefix[@]}" mri_convert "${atlas_mgz}" "${tmp_atlas}"
                    ;;
                *)
                    echo "[ERROR] Unsupported atlas task mode: ${mode}" >&2
                    return 2
                    ;;
            esac

            [ -s "${tmp_atlas}" ] || {
                echo "[ERROR] Atlas conversion did not create a non-empty file." >&2
                return 1
            }
            mv "${tmp_atlas}" "${flat_atlas}"
        fi

        "${launch_prefix[@]}" mri_info "${flat_atlas}" >/dev/null
        atlas_sha256="$(sha256sum "${flat_atlas}" | awk '{print $1}')"
        atlas_bytes="$(stat -c '%s' "${flat_atlas}")"
        completed_at="$(date --iso-8601=seconds)"
        {
            printf '%s\n' \
                $'status\tsubject\tmode\tatlas_path\tatlas_bytes\tatlas_sha256\tstarted_at\tcompleted_at'
            printf 'complete\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${subject}" \
                "${actual_mode}" \
                "${flat_atlas}" \
                "${atlas_bytes}" \
                "${atlas_sha256}" \
                "${started_at}" \
                "${completed_at}"
        } > "${result_path}.tmp.${SLURM_JOB_ID}"
        mv "${result_path}.tmp.${SLURM_JOB_ID}" "${result_path}"
        echo "[INFO] Atlas repair complete: ${subject}"
    } 2>&1 | tee -a "${task_log}"
    return "${PIPESTATUS[0]}"
}

PIDS=()
TASK_INDICES=()
for ((task_index = FIRST_TASK; task_index <= LAST_TASK; task_index++)); do
    worker_position=$((task_index - FIRST_TASK))
    run_one_task "${task_index}" "${worker_position}" &
    PIDS+=("$!")
    TASK_INDICES+=("${task_index}")
done

RUN_EXIT=0
for position in "${!PIDS[@]}"; do
    if wait "${PIDS[$position]}"; then
        echo "[INFO] Atlas task ${TASK_INDICES[$position]} completed."
    else
        TASK_EXIT=$?
        echo "[ERROR] Atlas task ${TASK_INDICES[$position]} failed with exit ${TASK_EXIT}." >&2
        if [ "${RUN_EXIT}" -eq 0 ]; then
            RUN_EXIT="${TASK_EXIT}"
        fi
    fi
done

if [ "${RUN_EXIT}" -eq 0 ]; then
    rm -f "${RETRY_FILE}"
    exit 0
fi
if [ "${RETRY_COUNT}" -lt "${MAX_RETRIES}" ]; then
    NEXT_RETRY=$((RETRY_COUNT + 1))
    printf '%s\n' "${NEXT_RETRY}" > "${RETRY_FILE}"
    echo "[WARN] Packed atlas element failed with exit ${RUN_EXIT}; requeueing (${NEXT_RETRY}/${MAX_RETRIES})."
    scontrol requeue "${SLURM_JOB_ID}"
    exit $?
fi

echo "[ERROR] Packed atlas element failed after ${RETRY_COUNT} retries." >&2
exit "${RUN_EXIT}"
