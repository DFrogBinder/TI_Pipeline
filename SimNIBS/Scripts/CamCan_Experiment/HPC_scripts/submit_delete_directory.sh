#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SLURM_SCRIPT:-${SCRIPT_DIR}/delete_directory.slurm}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
PARTITION="${PARTITION:-sheffield}"
MEMORY="${MEMORY:-1G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
JOB_NAME="${JOB_NAME:-delete_directory}"

TARGET_INPUT=""
CONFIRM_INPUT=""
ALLOWED_ROOT_INPUT="${ALLOWED_ROOT:-/mnt/parscratch/users/${USER}}"
LOG_DIR_INPUT="${LOG_DIR:-/mnt/parscratch/users/${USER}/deletion_job_logs}"

usage() {
    cat <<'EOF'
Usage:
  bash submit_delete_directory.sh \
    --target /absolute/path/to/folder \
    --confirm /absolute/path/to/folder \
    [--allowed-root /mnt/parscratch/users/USER] \
    [--log-dir /mnt/parscratch/users/USER/deletion_job_logs]

The value passed to --confirm must exactly match --target. The submitted
one-task Slurm job permanently deletes that directory and writes logs plus a
TSV receipt outside the target directory.
EOF
}

fail() {
    echo "[ERROR] $*" >&2
    exit 2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --target)
            [ "$#" -ge 2 ] || fail "--target requires a value."
            TARGET_INPUT="$2"
            shift 2
            ;;
        --confirm)
            [ "$#" -ge 2 ] || fail "--confirm requires a value."
            CONFIRM_INPUT="$2"
            shift 2
            ;;
        --allowed-root)
            [ "$#" -ge 2 ] || fail "--allowed-root requires a value."
            ALLOWED_ROOT_INPUT="$2"
            shift 2
            ;;
        --log-dir)
            [ "$#" -ge 2 ] || fail "--log-dir requires a value."
            LOG_DIR_INPUT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "Unknown argument: $1"
            ;;
    esac
done

[ -n "${TARGET_INPUT}" ] || fail "--target is required."
[ -n "${CONFIRM_INPUT}" ] || fail "--confirm is required."
[ "${TARGET_INPUT}" = "${CONFIRM_INPUT}" ] || fail "--confirm must exactly match --target."
[ -f "${SLURM_SCRIPT}" ] || fail "Slurm worker is missing: ${SLURM_SCRIPT}"

case "${TARGET_INPUT}${ALLOWED_ROOT_INPUT}${LOG_DIR_INPUT}" in
    *$'\n'*|*$'\r'*|*$'\t'*|*,*)
        fail "Paths containing commas, tabs, or newlines are not supported."
        ;;
esac
case "${TARGET_INPUT}" in
    /*) ;;
    *) fail "--target must be an absolute path." ;;
esac
case "${ALLOWED_ROOT_INPUT}" in
    /*) ;;
    *) fail "--allowed-root must be an absolute path." ;;
esac
case "${LOG_DIR_INPUT}" in
    /*) ;;
    *) fail "--log-dir must be an absolute path." ;;
esac

[ -d "${TARGET_INPUT}" ] || fail "Target is not an existing directory: ${TARGET_INPUT}"
[ ! -L "${TARGET_INPUT}" ] || fail "Refusing a symbolic-link target: ${TARGET_INPUT}"
[ -d "${ALLOWED_ROOT_INPUT}" ] || fail "Allowed root is not an existing directory: ${ALLOWED_ROOT_INPUT}"
[ ! -L "${ALLOWED_ROOT_INPUT}" ] || fail "Refusing a symbolic-link allowed root: ${ALLOWED_ROOT_INPUT}"

TARGET="$(realpath -e -- "${TARGET_INPUT}")"
ALLOWED_ROOT_PATH="$(realpath -e -- "${ALLOWED_ROOT_INPUT}")"
if [ "${TARGET}" = "${ALLOWED_ROOT_PATH}" ]; then
    fail "Refusing to delete the allowed root itself: ${TARGET}"
fi
case "${TARGET}/" in
    "${ALLOWED_ROOT_PATH}/"*) ;;
    *) fail "Target is outside the allowed root: ${TARGET}" ;;
esac

case "${TARGET}" in
    /|/bin|/boot|/dev|/etc|/home|/lib|/lib64|/mnt|/opt|/proc|/root|/run|/sbin|/srv|/sys|/tmp|/usr|/users|/var|/mnt/parscratch|/mnt/parscratch/users)
        fail "Refusing to delete a protected system path: ${TARGET}"
        ;;
esac

if command -v mountpoint >/dev/null 2>&1 && mountpoint -q -- "${TARGET}"; then
    fail "Refusing to delete a mount point: ${TARGET}"
fi

mkdir -p -- "${LOG_DIR_INPUT}"
LOG_DIR_PATH="$(realpath -e -- "${LOG_DIR_INPUT}")"
case "${LOG_DIR_PATH}/" in
    "${TARGET}/"*) fail "The log directory must be outside the deletion target: ${LOG_DIR_PATH}" ;;
esac

if ! [[ "${JOB_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    fail "JOB_NAME may contain only letters, digits, underscore, dot, and hyphen."
fi

TARGET_ID="$(stat -Lc '%d:%i' -- "${TARGET}")"

printf '%s\n' \
    'Scope:' \
    '  dataset/ROI: filesystem maintenance (no dataset or ROI)' \
    '  subjects: 0' \
    '  tasks: 1' \
    '  array: none' \
    '  expected outputs: 1 deletion receipt' \
    '  execution: full requested deletion; permanent and not recoverable from this job'
echo "[INFO] Target:       ${TARGET}"
echo "[INFO] Target ID:    ${TARGET_ID}"
echo "[INFO] Allowed root: ${ALLOWED_ROOT_PATH}"
echo "[INFO] Log dir:      ${LOG_DIR_PATH}"
echo "[INFO] Resource profile: ${PARTITION}, 1 CPU, ${MEMORY}, ${TIME_LIMIT}"

EXPORT_VARS="ALL,TI_DELETE_TARGET=${TARGET},TI_DELETE_TARGET_ID=${TARGET_ID},TI_DELETE_ALLOWED_ROOT=${ALLOWED_ROOT_PATH},TI_DELETE_LOG_DIR=${LOG_DIR_PATH}"
SUBMISSION=$("${SBATCH_BIN}" \
    --parsable \
    --job-name="${JOB_NAME}" \
    --partition="${PARTITION}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem="${MEMORY}" \
    --time="${TIME_LIMIT}" \
    --output="${LOG_DIR_PATH}/delete-directory-%j.out" \
    --error="${LOG_DIR_PATH}/delete-directory-%j.err" \
    --export="${EXPORT_VARS}" \
    "${SLURM_SCRIPT}")
JOB_ID="${SUBMISSION%%;*}"
if ! [[ "${JOB_ID}" =~ ^[0-9]+$ ]]; then
    fail "Could not parse a Slurm job ID from: ${SUBMISSION}"
fi

REQUEST_RECEIPT="${LOG_DIR_PATH}/delete-directory-${JOB_ID}.request.tsv"
{
    printf 'job_id\tstatus\ttarget\ttarget_id\tallowed_root\tsubmitted_at\n'
    printf '%s\tsubmitted\t%s\t%s\t%s\t%s\n' \
        "${JOB_ID}" \
        "${TARGET}" \
        "${TARGET_ID}" \
        "${ALLOWED_ROOT_PATH}" \
        "$(date --iso-8601=seconds)"
} > "${REQUEST_RECEIPT}"

echo "[INFO] Submitted deletion job: ${JOB_ID}"
echo "[INFO] Request receipt: ${REQUEST_RECEIPT}"
echo "[INFO] Completion receipt: ${LOG_DIR_PATH}/delete-directory-${JOB_ID}.receipt.tsv"

