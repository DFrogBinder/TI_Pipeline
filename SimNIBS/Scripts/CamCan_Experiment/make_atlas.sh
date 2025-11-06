#!/usr/bin/env bash
#
# FastSurfer → FreeSurfer segmentation pipeline (Docker)
#
# Usage:
#   ./make_atlas.sh <DATA_DIR> <THREADS> <LICENSE_PATH>
#
# Example:
#   ./make_atlas.sh ~/sandbox/Jake_Data 7 ~/sandbox/utils/freesurfer_licence.txt
#
set -euo pipefail

DOCKER_TERM_FLAGS=("-i")
if [[ -t 1 ]]; then
  DOCKER_TERM_FLAGS=("-it")
fi

SCRIPT_START_TS=$(date +%s)

# ---------- helper functions ----------
log() {
  printf '%s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

format_duration() {
  local total_seconds=$1
  local hours=$(( total_seconds / 3600 ))
  local minutes=$(( (total_seconds % 3600) / 60 ))
  local seconds=$(( total_seconds % 60 ))
  printf '%02d:%02d:%02d' "${hours}" "${minutes}" "${seconds}"
}

report_progress() {
  local completed="$1"
  local total="$2"
  local duration_sum="$3"
  local timed="$4"

  if (( total == 0 )); then
    return
  fi

  local percent=$(( completed * 100 / total ))
  local elapsed=$(( $(date +%s) - SCRIPT_START_TS ))
  local elapsed_formatted
  elapsed_formatted=$(format_duration "${elapsed}")

  local remaining=$(( total - completed ))
  local eta_str="n/a"

  if (( timed > 0 && remaining > 0 )); then
    local avg=$(( duration_sum / timed ))
    if (( avg == 0 && duration_sum > 0 )); then
      avg=1
    fi
    local eta_seconds=$(( avg * remaining ))
    eta_str=$(format_duration "${eta_seconds}")
  elif (( remaining == 0 )); then
    eta_str="00:00:00"
  fi

  log "[progress] ${completed}/${total} (${percent}%) complete | elapsed ${elapsed_formatted} | est. remaining ${eta_str}"
}

fastsurfer_done() {
  local subj_dir="$1"
  [[ -f "${subj_dir}/mri/aparc.DKTatlas+aseg.deep.mgz" ]]
}

recon_done() {
  local subj_dir="$1"
  [[ -f "${subj_dir}/surf/lh.white" && -f "${subj_dir}/surf/rh.white" ]]
}

converted_done() {
  local subj_dir="$1"
  [[ -f "${subj_dir}/mri/T1.nii.gz" && -f "${subj_dir}/mri/aparc.DKTatlas+aseg.deep.nii.gz" ]]
}

process_subject() {
  local sid="$1"
  local subj_dir_host="$2"
  local data_root="$3"
  local threads="$4"
  local license_path="$5"
  local fastsurfer_image="$6"
  local freesurfer_image="$7"

  local t1_container="/data/${sid}/anat/${sid}_T1w.nii.gz"
  local subjects_dir_container="/data/FastSurfer_out"

  if fastsurfer_done "${subj_dir_host}"; then
    log "[ok] ${sid}: FastSurfer output already exists."
  else
    log "[run] ${sid}: FastSurfer segmentation..."
    docker run --rm "${DOCKER_TERM_FLAGS[@]}" \
      -u "$(id -u):$(id -g)" -w /data \
      -v "${data_root}:/data" \
      "${fastsurfer_image}" \
      --t1 "${t1_container}" \
      --sd "${subjects_dir_container}" \
      --sid "${sid}" \
      --seg_only --threads "${threads}" --no_cereb --no_hypothal || return 1
    log "[done] ${sid}: FastSurfer segmentation."
  fi

  if recon_done "${subj_dir_host}"; then
    log "[ok] ${sid}: FreeSurfer recon-all already complete."
  else
    log "[run] ${sid}: FreeSurfer recon-all..."
    docker run --rm "${DOCKER_TERM_FLAGS[@]}" \
      -u "$(id -u):$(id -g)" -w /data \
      -v "${data_root}:/data" \
      -v "${license_path}:/license.txt:ro" \
      -e FS_LICENSE=/license.txt \
      -e SUBJECTS_DIR="${subjects_dir_container}" \
      "${freesurfer_image}" \
      recon-all -s "${sid}" -all -openmp "${threads}" \
      -cw256 || return 1
    log "[done] ${sid}: FreeSurfer recon-all."
  fi

  if converted_done "${subj_dir_host}"; then
    log "[ok] ${sid}: NIfTI conversions already exist."
  else
    log "[run] ${sid}: Converting MGZ → NIfTI..."
    docker run --rm "${DOCKER_TERM_FLAGS[@]}" \
      -v "${data_root}:/data" \
      -v "${license_path}:/license.txt:ro" \
      -e FS_LICENSE=/license.txt \
      "${freesurfer_image}" \
      bash -lc "
        mri_convert /data/FastSurfer_out/${sid}/mri/T1.mgz \
                    /data/FastSurfer_out/${sid}/mri/T1.nii.gz && \
        mri_convert /data/FastSurfer_out/${sid}/mri/aparc.DKTatlas+aseg.deep.mgz \
                    /data/FastSurfer_out/${sid}/mri/aparc.DKTatlas+aseg.deep.nii.gz
      " || return 1
    log "[done] ${sid}: Conversion complete."
  fi
}

# ---------- parse args ----------
DATA="${1:-}"
THREADS="${2:-}"
LICENSE="${3:-}"

# default docker images
FASTSURFER_IMAGE="deepmi/fastsurfer:latest"
FREESURFER_IMAGE="freesurfer/freesurfer:7.4.1"

# ---------- validate ----------
if [[ -z "${DATA}" || -z "${THREADS}" || -z "${LICENSE}" ]]; then
  echo "Usage: $0 <DATA_DIR> <THREADS> <LICENSE_PATH>" >&2
  exit 1
fi

if [[ ! -d "${DATA}" ]]; then
  echo "ERROR: DATA_DIR not found: ${DATA}" >&2
  exit 1
fi

if [[ ! -f "${LICENSE}" ]]; then
  echo "ERROR: LICENSE_PATH not a file: ${LICENSE}" >&2
  exit 1
fi

if [[ ! "${THREADS}" =~ ^[0-9]+$ ]] || (( THREADS < 1 )); then
  echo "ERROR: THREADS must be a positive integer. Received: ${THREADS}" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found in PATH." >&2
  exit 1
fi

# ---------- paths ----------
DATA_ROOT="$(cd "${DATA}" && pwd -P)"
LICENSE_DIR="$(cd "$(dirname "${LICENSE}")" && pwd -P)"
LICENSE_PATH="${LICENSE_DIR}/$(basename "${LICENSE}")"

SUBJECT_ROOT="${DATA_ROOT}"
FASTSURFER_OUT_DIR="${SUBJECT_ROOT}/FastSurfer_out"

mkdir -p "${FASTSURFER_OUT_DIR}"

LOG_DIR="${FASTSURFER_OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/make_atlas_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${LOG_FILE}") 2>&1

log "-----------------------------------------------"
log "Log file      : ${LOG_FILE}"
log "DATA_DIR      : ${DATA_ROOT}"
log "Subjects dir  : ${SUBJECT_ROOT}"
log "THREADS       : ${THREADS}"
log "LICENSE_PATH  : ${LICENSE_PATH}"
log "FastSurfer    : ${FASTSURFER_IMAGE}"
log "FreeSurfer    : ${FREESURFER_IMAGE}"
log "Output dir    : ${FASTSURFER_OUT_DIR}"
log "-----------------------------------------------"

declare -a SUBJECT_IDS=()
while IFS= read -r -d '' dir; do
  SUBJECT_IDS+=("$(basename "${dir}")")
done < <(find "${SUBJECT_ROOT}" -mindepth 1 -maxdepth 1 -type d ! -name "FastSurfer_out" -print0)

if (( ${#SUBJECT_IDS[@]} == 0 )); then
  log "No subject directories found beneath ${SUBJECT_ROOT}. Nothing to process."
  exit 0
fi

mapfile -t SUBJECT_IDS < <(printf '%s\n' "${SUBJECT_IDS[@]}" | LC_ALL=C sort)

log "Found ${#SUBJECT_IDS[@]} subject(s) to process."

total_subjects=${#SUBJECT_IDS[@]}
success_count=0
completed_count=0
timed_count=0
total_subject_duration=0
declare -a failures=()

report_progress "${completed_count}" "${total_subjects}" "${total_subject_duration}" "${timed_count}"

for SID in "${SUBJECT_IDS[@]}"; do
  SUBJ_PATH="${SUBJECT_ROOT}/${SID}"
  T1_HOST="${SUBJ_PATH}/anat/${SID}_T1w.nii.gz"
  SUBJ_DIR_HOST="${FASTSURFER_OUT_DIR}/${SID}"

  if [[ "${SID}" == "FastSurfer_out" ]]; then
    continue
  fi

  if [[ ! -d "${SUBJ_PATH}" ]]; then
    log "[skip] ${SID}: Not a directory at ${SUBJ_PATH}"
    ((completed_count++))
    report_progress "${completed_count}" "${total_subjects}" "${total_subject_duration}" "${timed_count}"
    continue
  fi

  if [[ ! -f "${T1_HOST}" ]]; then
    log "[skip] ${SID}: Missing T1 image at ${T1_HOST}"
    failures+=("${SID}")
    ((completed_count++))
    report_progress "${completed_count}" "${total_subjects}" "${total_subject_duration}" "${timed_count}"
    continue
  fi

  mkdir -p "${SUBJ_DIR_HOST}"

  log "=== ${SID}: starting processing ==="
  subject_start=$(date +%s)

  if process_subject "${SID}" "${SUBJ_DIR_HOST}" "${DATA_ROOT}" "${THREADS}" "${LICENSE_PATH}" "${FASTSURFER_IMAGE}" "${FREESURFER_IMAGE}"; then
    subject_status="success"
  else
    subject_status="failure"
  fi

  subject_end=$(date +%s)
  duration=$(( subject_end - subject_start ))
  duration_formatted=$(format_duration "${duration}")

  if [[ "${subject_status}" == "success" ]]; then
    log "=== ${SID}: completed in ${duration_formatted} ==="
    ((success_count++))
  else
    log "[error] ${SID}: failed after ${duration_formatted}. See details above."
    failures+=("${SID}")
  fi

  ((completed_count++))
  ((timed_count++))
  total_subject_duration=$(( total_subject_duration + duration ))
  report_progress "${completed_count}" "${total_subjects}" "${total_subject_duration}" "${timed_count}"
done

total_duration=$(( $(date +%s) - SCRIPT_START_TS ))
log "Processing finished in $(format_duration "${total_duration}")."
log "Subjects succeeded: ${success_count}"
log "Subjects failed   : ${#failures[@]}"

if (( ${#failures[@]} > 0 )); then
  log "Failed subjects: ${failures[*]}"
  exit 1
fi

log "All subjects completed successfully."
