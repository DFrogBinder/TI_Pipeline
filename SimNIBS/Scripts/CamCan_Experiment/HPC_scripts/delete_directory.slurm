#!/bin/bash
#SBATCH --job-name=delete_directory
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=24:00:00
#SBATCH --output=delete-directory-%j.out
#SBATCH --error=delete-directory-%j.err

set -euo pipefail

: "${TI_DELETE_TARGET:?Missing TI_DELETE_TARGET. Use submit_delete_directory.sh.}"
: "${TI_DELETE_TARGET_ID:?Missing TI_DELETE_TARGET_ID. Use submit_delete_directory.sh.}"
: "${TI_DELETE_ALLOWED_ROOT:?Missing TI_DELETE_ALLOWED_ROOT. Use submit_delete_directory.sh.}"
: "${TI_DELETE_LOG_DIR:?Missing TI_DELETE_LOG_DIR. Use submit_delete_directory.sh.}"
: "${SLURM_JOB_ID:?This worker must run as a Slurm job.}"

TARGET="${TI_DELETE_TARGET}"
EXPECTED_TARGET_ID="${TI_DELETE_TARGET_ID}"
ALLOWED_ROOT="${TI_DELETE_ALLOWED_ROOT}"
LOG_DIR="${TI_DELETE_LOG_DIR}"
RECEIPT="${LOG_DIR}/delete-directory-${SLURM_JOB_ID}.receipt.tsv"
STARTED_AT="$(date --iso-8601=seconds)"
STATUS="failed"
MESSAGE="job exited before deletion completed"

write_receipt() {
    local receipt_status="$1"
    local receipt_message="$2"
    local completed_at
    local temporary_receipt

    completed_at="$(date --iso-8601=seconds)"
    temporary_receipt="${RECEIPT}.tmp.$$"
    {
        printf 'job_id\tstatus\ttarget\ttarget_id\tallowed_root\tstarted_at\tcompleted_at\tmessage\n'
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${SLURM_JOB_ID}" \
            "${receipt_status}" \
            "${TARGET}" \
            "${EXPECTED_TARGET_ID}" \
            "${ALLOWED_ROOT}" \
            "${STARTED_AT}" \
            "${completed_at}" \
            "${receipt_message}"
    } > "${temporary_receipt}"
    mv -f -- "${temporary_receipt}" "${RECEIPT}"
}

on_exit() {
    local exit_code=$?
    if [ "${STATUS}" != "complete" ]; then
        MESSAGE="${MESSAGE}; exit_code=${exit_code}"
        write_receipt "failed" "${MESSAGE}" || true
    fi
}

on_signal() {
    MESSAGE="job received a termination signal"
    exit 143
}

trap on_exit EXIT
trap on_signal HUP INT TERM

mkdir -p -- "${LOG_DIR}"
write_receipt "started" "identity checks pending"

if [ ! -d "${TARGET}" ] || [ -L "${TARGET}" ]; then
    MESSAGE="target is missing, is not a directory, or is a symbolic link"
    echo "[ERROR] ${MESSAGE}: ${TARGET}" >&2
    exit 2
fi
if [ ! -d "${ALLOWED_ROOT}" ] || [ -L "${ALLOWED_ROOT}" ]; then
    MESSAGE="allowed root is missing, is not a directory, or is a symbolic link"
    echo "[ERROR] ${MESSAGE}: ${ALLOWED_ROOT}" >&2
    exit 2
fi

CURRENT_TARGET="$(realpath -e -- "${TARGET}")"
CURRENT_ALLOWED_ROOT="$(realpath -e -- "${ALLOWED_ROOT}")"
if [ "${CURRENT_TARGET}" != "${TARGET}" ] || [ "${CURRENT_ALLOWED_ROOT}" != "${ALLOWED_ROOT}" ]; then
    MESSAGE="canonical path changed after submission"
    echo "[ERROR] ${MESSAGE}." >&2
    exit 2
fi
if [ "${TARGET}" = "${ALLOWED_ROOT}" ]; then
    MESSAGE="refusing to delete the allowed root itself"
    echo "[ERROR] ${MESSAGE}: ${TARGET}" >&2
    exit 2
fi
case "${TARGET}/" in
    "${ALLOWED_ROOT}/"*) ;;
    *)
        MESSAGE="target is outside the allowed root"
        echo "[ERROR] ${MESSAGE}: ${TARGET}" >&2
        exit 2
        ;;
esac

case "${TARGET}" in
    /|/bin|/boot|/dev|/etc|/home|/lib|/lib64|/mnt|/opt|/proc|/root|/run|/sbin|/srv|/sys|/tmp|/usr|/users|/var|/mnt/parscratch|/mnt/parscratch/users)
        MESSAGE="target is a protected system path"
        echo "[ERROR] ${MESSAGE}: ${TARGET}" >&2
        exit 2
        ;;
esac

if command -v mountpoint >/dev/null 2>&1 && mountpoint -q -- "${TARGET}"; then
    MESSAGE="refusing to delete a mount point"
    echo "[ERROR] ${MESSAGE}: ${TARGET}" >&2
    exit 2
fi

CURRENT_TARGET_ID="$(stat -Lc '%d:%i' -- "${TARGET}")"
if [ "${CURRENT_TARGET_ID}" != "${EXPECTED_TARGET_ID}" ]; then
    MESSAGE="target device/inode changed after submission"
    echo "[ERROR] ${MESSAGE}: expected ${EXPECTED_TARGET_ID}, found ${CURRENT_TARGET_ID}" >&2
    exit 2
fi

echo "[INFO] Host:         $(hostname)"
echo "[INFO] Job ID:       ${SLURM_JOB_ID}"
echo "[INFO] Started:      ${STARTED_AT}"
echo "[INFO] Target:       ${TARGET}"
echo "[INFO] Target ID:    ${CURRENT_TARGET_ID}"
echo "[INFO] Allowed root: ${ALLOWED_ROOT}"
echo "[INFO] Receipt:      ${RECEIPT}"
echo "[INFO] Operation:    permanent recursive directory deletion"

MESSAGE="rm failed before the target was completely removed"
rm -rf --one-file-system -- "${TARGET}"

if [ -e "${TARGET}" ] || [ -L "${TARGET}" ]; then
    MESSAGE="target still exists after rm returned"
    echo "[ERROR] ${MESSAGE}: ${TARGET}" >&2
    exit 2
fi

STATUS="complete"
MESSAGE="target deleted"
write_receipt "complete" "${MESSAGE}"
trap - EXIT HUP INT TERM

echo "[INFO] Completed:    $(date --iso-8601=seconds)"
echo "[INFO] Result:       ${MESSAGE}"

