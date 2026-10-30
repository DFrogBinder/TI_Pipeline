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
POSITIONAL_TARGETS=()
CANONICAL_TARGETS=()
SUBMITTED_JOB_IDS=()
FRIENDLY_MODE=0

usage() {
    cat <<'EOF'
Usage:
  # Friendly interactive mode: one separate Slurm job per directory
  bash submit_delete_directory.sh /absolute/folder/one [/absolute/folder/two ...]

  # Existing non-interactive mode for scripts and automation
  bash submit_delete_directory.sh \
    --target /absolute/path/to/folder \
    --confirm /absolute/path/to/folder \
    [--allowed-root /mnt/parscratch/users/USER] \
    [--log-dir /mnt/parscratch/users/USER/deletion_job_logs]

Friendly mode validates and displays every directory, then asks for a short
confirmation such as DELETE 2. It never asks you to retype a path. Each
directory receives its own one-task Slurm job and external TSV receipt.

The automation form remains available and requires --confirm to exactly match
--target. Every submitted deletion is permanent.
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
        --)
            shift
            while [ "$#" -gt 0 ]; do
                POSITIONAL_TARGETS+=("$1")
                shift
            done
            ;;
        -*)
            fail "Unknown argument: $1"
            ;;
        *)
            POSITIONAL_TARGETS+=("$1")
            shift
            ;;
    esac
done

if [ "${#POSITIONAL_TARGETS[@]}" -gt 0 ]; then
    if [ -n "${TARGET_INPUT}" ] || [ -n "${CONFIRM_INPUT}" ]; then
        fail "Do not combine positional directories with --target or --confirm."
    fi
    FRIENDLY_MODE=1
else
    [ -n "${TARGET_INPUT}" ] || fail "Provide one or more directory paths, or use --target with --confirm."
    [ -n "${CONFIRM_INPUT}" ] || fail "--confirm is required with --target."
    [ "${TARGET_INPUT}" = "${CONFIRM_INPUT}" ] || fail "--confirm must exactly match --target."
    POSITIONAL_TARGETS=("${TARGET_INPUT}")
fi

[ -f "${SLURM_SCRIPT}" ] || fail "Slurm worker is missing: ${SLURM_SCRIPT}"

case "${ALLOWED_ROOT_INPUT}${LOG_DIR_INPUT}" in
    *$'\n'*|*$'\r'*|*$'\t'*|*,*)
        fail "Paths containing commas, tabs, or newlines are not supported."
        ;;
esac
case "${ALLOWED_ROOT_INPUT}" in
    /*) ;;
    *) fail "--allowed-root must be an absolute path." ;;
esac
case "${LOG_DIR_INPUT}" in
    /*) ;;
    *) fail "--log-dir must be an absolute path." ;;
esac

[ -d "${ALLOWED_ROOT_INPUT}" ] || fail "Allowed root is not an existing directory: ${ALLOWED_ROOT_INPUT}"
[ ! -L "${ALLOWED_ROOT_INPUT}" ] || fail "Refusing a symbolic-link allowed root: ${ALLOWED_ROOT_INPUT}"

ALLOWED_ROOT_PATH="$(realpath -e -- "${ALLOWED_ROOT_INPUT}")"
mkdir -p -- "${LOG_DIR_INPUT}"
LOG_DIR_PATH="$(realpath -e -- "${LOG_DIR_INPUT}")"

if ! [[ "${JOB_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    fail "JOB_NAME may contain only letters, digits, underscore, dot, and hyphen."
fi

validate_target() {
    local target_input="$1"
    local target
    local existing_target

    case "${target_input}" in
        *$'\n'*|*$'\r'*|*$'\t'*|*,*)
            fail "Paths containing commas, tabs, or newlines are not supported."
            ;;
    esac
    case "${target_input}" in
        /*) ;;
        *) fail "Target must be an absolute path: ${target_input}" ;;
    esac

    [ -d "${target_input}" ] || fail "Target is not an existing directory: ${target_input}"
    [ ! -L "${target_input}" ] || fail "Refusing a symbolic-link target: ${target_input}"
    target="$(realpath -e -- "${target_input}")"

    if [ "${target}" = "${ALLOWED_ROOT_PATH}" ]; then
        fail "Refusing to delete the allowed root itself: ${target}"
    fi
    case "${target}/" in
        "${ALLOWED_ROOT_PATH}/"*) ;;
        *) fail "Target is outside the allowed root: ${target}" ;;
    esac

    case "${target}" in
        /|/bin|/boot|/dev|/etc|/home|/lib|/lib64|/mnt|/opt|/proc|/root|/run|/sbin|/srv|/sys|/tmp|/usr|/users|/var|/mnt/parscratch|/mnt/parscratch/users)
            fail "Refusing to delete a protected system path: ${target}"
            ;;
    esac

    if command -v mountpoint >/dev/null 2>&1 && mountpoint -q -- "${target}"; then
        fail "Refusing to delete a mount point: ${target}"
    fi

    case "${LOG_DIR_PATH}/" in
        "${target}/"*) fail "The log directory must be outside the deletion target: ${LOG_DIR_PATH}" ;;
    esac

    for existing_target in "${CANONICAL_TARGETS[@]}"; do
        [ "${existing_target}" != "${target}" ] || fail "Duplicate deletion target: ${target}"
    done

    CANONICAL_TARGETS+=("${target}")
}

for TARGET_INPUT in "${POSITIONAL_TARGETS[@]}"; do
    validate_target "${TARGET_INPUT}"
done

TARGET_COUNT="${#CANONICAL_TARGETS[@]}"

printf '%s\n' \
    'Scope:' \
    '  dataset/ROI: filesystem maintenance (no dataset or ROI)' \
    '  subjects: 0' \
    "  directories: ${TARGET_COUNT}" \
    "  tasks: ${TARGET_COUNT}" \
    '  array: none' \
    "  expected outputs: ${TARGET_COUNT} deletion receipt(s)" \
    '  execution: full requested deletions; permanent and not recoverable from these jobs'
echo "[INFO] Targets (${TARGET_COUNT}):"
for TARGET_INDEX in "${!CANONICAL_TARGETS[@]}"; do
    printf '  %d. %s\n' "$((TARGET_INDEX + 1))" "${CANONICAL_TARGETS[${TARGET_INDEX}]}"
done
echo "[INFO] Allowed root: ${ALLOWED_ROOT_PATH}"
echo "[INFO] Log dir:      ${LOG_DIR_PATH}"
echo "[INFO] Resource profile: ${PARTITION}, 1 CPU per job, ${MEMORY}, ${TIME_LIMIT}"

if [ "${FRIENDLY_MODE}" -eq 1 ]; then
    CONFIRMATION_PHRASE="DELETE ${TARGET_COUNT}"
    printf 'Type "%s" to submit these permanent deletion job(s): ' "${CONFIRMATION_PHRASE}"
    if ! IFS= read -r ENTERED_CONFIRMATION; then
        fail "Confirmation input ended; no jobs submitted."
    fi
    if [ "${ENTERED_CONFIRMATION}" != "${CONFIRMATION_PHRASE}" ]; then
        echo "[INFO] Cancelled; no jobs submitted."
        exit 1
    fi
fi

submit_target() {
    local target="$1"
    local target_id
    local export_vars
    local submission
    local job_id
    local request_receipt

    target_id="$(stat -Lc '%d:%i' -- "${target}")"
    export_vars="ALL,TI_DELETE_TARGET=${target},TI_DELETE_TARGET_ID=${target_id},TI_DELETE_ALLOWED_ROOT=${ALLOWED_ROOT_PATH},TI_DELETE_LOG_DIR=${LOG_DIR_PATH}"
    submission=$("${SBATCH_BIN}" \
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
        --export="${export_vars}" \
        "${SLURM_SCRIPT}")
    job_id="${submission%%;*}"
    if ! [[ "${job_id}" =~ ^[0-9]+$ ]]; then
        fail "Could not parse a Slurm job ID from: ${submission}"
    fi

    request_receipt="${LOG_DIR_PATH}/delete-directory-${job_id}.request.tsv"
    {
        printf 'job_id\tstatus\ttarget\ttarget_id\tallowed_root\tsubmitted_at\n'
        printf '%s\tsubmitted\t%s\t%s\t%s\t%s\n' \
            "${job_id}" \
            "${target}" \
            "${target_id}" \
            "${ALLOWED_ROOT_PATH}" \
            "$(date --iso-8601=seconds)"
    } > "${request_receipt}"

    SUBMITTED_JOB_IDS+=("${job_id}")
    echo "[INFO] Submitted deletion job: ${job_id}"
    echo "[INFO] Target: ${target}"
    echo "[INFO] Request receipt: ${request_receipt}"
    echo "[INFO] Completion receipt: ${LOG_DIR_PATH}/delete-directory-${job_id}.receipt.tsv"
}

for TARGET in "${CANONICAL_TARGETS[@]}"; do
    submit_target "${TARGET}"
done

if [ "${FRIENDLY_MODE}" -eq 1 ]; then
    echo "[INFO] Submitted deletion jobs: ${#SUBMITTED_JOB_IDS[@]}"
    echo "[INFO] Job IDs: ${SUBMITTED_JOB_IDS[*]}"
fi
