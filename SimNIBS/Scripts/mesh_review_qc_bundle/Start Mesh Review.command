#!/bin/bash

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_DIR="${1:-${SCRIPT_DIR}/images}"
STATE_DIR="${SCRIPT_DIR}/review_state"
TARGET="${MESH_REVIEW_TARGET:-200}"

pause_before_exit() {
    if [[ -t 0 ]]; then
        printf '\nPress Return to close this window.'
        read -r _
    fi
}

fail() {
    printf '[ERROR] %s\n' "$1" >&2
    pause_before_exit
    exit 1
}

cd "${SCRIPT_DIR}" || fail "Could not open the tool directory."

if ! command -v python3 >/dev/null 2>&1; then
    fail "Python 3.10 or newer is required. Install Python 3 from https://www.python.org/downloads/macos/ and run this launcher again."
fi

PYTHON_BIN="$(command -v python3)"
if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    fail "Python 3.10 or newer is required. The current python3 is $(${PYTHON_BIN} --version 2>&1)."
fi

if [[ ! -d "${IMAGE_DIR}" ]]; then
    fail "Image directory does not exist: ${IMAGE_DIR}"
fi

IMAGE_COUNT="$("${PYTHON_BIN}" -c 'from pathlib import Path; import sys; root=Path(sys.argv[1]); suffixes={".png", ".jpg", ".jpeg", ".webp"}; print(sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes))' "${IMAGE_DIR}")"
if [[ "${IMAGE_COUNT}" == "0" ]]; then
    fail "No PNG, JPG, JPEG, or WebP images were found under ${IMAGE_DIR}. Put the render folders inside the images directory first."
fi

mkdir -p "${STATE_DIR}" || fail "Could not create the review_state directory."

printf '[INFO] Starting Mesh QC Review Tool\n'
printf '[INFO] Images: %s (%s files)\n' "${IMAGE_DIR}" "${IMAGE_COUNT}"
printf '[INFO] State:  %s\n' "${STATE_DIR}"
printf '[INFO] Target: %s accepted subjects\n' "${TARGET}"
printf '[INFO] Keep this Terminal window open while reviewing.\n'
printf '[INFO] To stop safely, return here and press Control-C.\n\n'

"${PYTHON_BIN}" -m mesh_review.server \
    --images "${IMAGE_DIR}" \
    --state-dir "${STATE_DIR}" \
    --target "${TARGET}" \
    --port 0
STATUS=$?

if [[ "${STATUS}" == "0" || "${STATUS}" == "130" ]]; then
    printf '\n[INFO] Review stopped. Decisions and exports are in:\n%s\n' "${STATE_DIR}"
else
    printf '\n[ERROR] The review server stopped with status %s.\n' "${STATUS}" >&2
fi

pause_before_exit
exit "${STATUS}"
