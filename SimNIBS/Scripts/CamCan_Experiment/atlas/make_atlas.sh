#!/usr/bin/env bash
#
# FastSurfer -> FreeSurfer segmentation pipeline.
#
# Usage:
#   ./make_atlas.sh <DATA_DIR> <THREADS> <LICENSE_PATH> [SUBJECT_ID]
#
# Example:
#   ./make_atlas.sh ~/sandbox/Jake_Data 7 ~/sandbox/utils/freesurfer_licence.txt
#   FASTSURFER_USE_GPU=1 FASTSURFER_DEVICE=cuda ./make_atlas.sh ~/sandbox/Jake_Data 9 ~/sandbox/utils/freesurfer_licence.txt sub-CC110033
#   ATLAS_BACKEND=native ./make_atlas.sh ~/sandbox/Jake_Data 9 ~/sandbox/utils/freesurfer_licence.txt sub-CC110033
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

truthy() {
  case "${1:-}" in
    1|true|TRUE|True|yes|YES|Yes|y|Y|on|ON|On) return 0 ;;
    *) return 1 ;;
  esac
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

destrieux_mgz_done() {
  local subj_dir="$1"
  [[ -f "${subj_dir}/mri/aparc.a2009s+aseg.mgz" ]]
}

converted_done() {
  local subj_dir="$1"
  [[ -f "${subj_dir}/mri/T1.nii.gz" && -f "${subj_dir}/mri/aparc.a2009s+aseg.nii.gz" ]]
}

t1_host_for() {
  local data_root="$1"
  local sid="$2"
  local path=""

  for path in \
    "${data_root}/${sid}/anat/${sid}_T1w.nii.gz" \
    "${data_root}/${sid}/anat/${sid}_T1w.nii"; do
    if [[ -f "${path}" ]]; then
      printf '%s\n' "${path}"
      return 0
    fi
  done

  return 1
}

convert_mgz_outputs_native() {
  local subj_dir_host="$1"
  local src=""
  local dst=""

  for required_name in T1 aparc.a2009s+aseg; do
    src="${subj_dir_host}/mri/${required_name}.mgz"
    if [[ ! -f "${src}" ]]; then
      log "[error] Required MGZ missing: ${src}"
      return 1
    fi
  done

  for name in T1 aparc.DKTatlas+aseg.deep aparc.a2009s+aseg; do
    src="${subj_dir_host}/mri/${name}.mgz"
    dst="${subj_dir_host}/mri/${name}.nii.gz"
    if [[ -f "${src}" ]]; then
      mri_convert "${src}" "${dst}" || return 1
    else
      log "[skip] Optional MGZ missing, not converting: ${src}"
    fi
  done
}

process_subject_docker() {
  local sid="$1"
  local subj_dir_host="$2"
  local data_root="$3"
  local threads="$4"
  local license_path="$5"
  local fastsurfer_image="$6"
  local freesurfer_image="$7"

  local t1_host=""
  t1_host="$(t1_host_for "${data_root}" "${sid}")" || return 1
  local t1_container="/data/${t1_host#"${data_root}/"}"
  local subjects_dir_container="/data/FastSurfer_out"

  if fastsurfer_done "${subj_dir_host}"; then
    log "[ok] ${sid}: FastSurfer output already exists."
  else
    log "[run] ${sid}: FastSurfer segmentation..."
    docker run --rm "${DOCKER_TERM_FLAGS[@]}" \
      "${FASTSURFER_DOCKER_RUN_FLAGS[@]}" \
      -u "$(id -u):$(id -g)" -w /data \
      -v "${data_root}:/data" \
      "${fastsurfer_image}" \
      --t1 "${t1_container}" \
      --sd "${subjects_dir_container}" \
      --sid "${sid}" \
      --seg_only --threads "${threads}" --no_cereb --no_hypothal \
      "${FASTSURFER_RUN_EXTRA_ARGS[@]}" || return 1
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
                    /data/FastSurfer_out/${sid}/mri/aparc.DKTatlas+aseg.deep.nii.gz && \
        mri_convert /data/FastSurfer_out/${sid}/mri/aparc.a2009s+aseg.mgz \
                    /data/FastSurfer_out/${sid}/mri/aparc.a2009s+aseg.nii.gz
      " || return 1
    log "[done] ${sid}: Conversion complete."
  fi
}

process_subject_native() {
  local sid="$1"
  local subj_dir_host="$2"
  local data_root="$3"
  local threads="$4"
  local license_path="$5"

  local t1_host=""
  t1_host="$(t1_host_for "${data_root}" "${sid}")" || return 1
  local subjects_dir_host="${data_root}/FastSurfer_out"

  export SUBJECTS_DIR="${subjects_dir_host}"
  export FS_LICENSE="${license_path}"

  if destrieux_mgz_done "${subj_dir_host}"; then
    log "[ok] ${sid}: Native FreeSurfer Destrieux output already exists."
  else
    log "[run] ${sid}: Native FreeSurfer recon-all..."
    if [[ -f "${subj_dir_host}/mri/orig.mgz" || -f "${subj_dir_host}/mri/T1.mgz" ]]; then
      recon-all -s "${sid}" -all -openmp "${threads}" "${RECON_ALL_EXTRA_ARGS_ARRAY[@]}" || return 1
    else
      recon-all -s "${sid}" -i "${t1_host}" -all -openmp "${threads}" "${RECON_ALL_EXTRA_ARGS_ARRAY[@]}" || return 1
    fi
    log "[done] ${sid}: Native FreeSurfer recon-all."
  fi

  if converted_done "${subj_dir_host}"; then
    log "[ok] ${sid}: NIfTI conversions already exist."
  else
    log "[run] ${sid}: Converting native FreeSurfer MGZ -> NIfTI..."
    convert_mgz_outputs_native "${subj_dir_host}" || return 1
    log "[done] ${sid}: Conversion complete."
  fi
}

process_subject() {
  case "${ATLAS_BACKEND}" in
    docker)
      process_subject_docker "$@"
      ;;
    native)
      process_subject_native "$@"
      ;;
    *)
      log "[error] Unsupported ATLAS_BACKEND: ${ATLAS_BACKEND}"
      return 1
      ;;
  esac
}

# ---------- parse args ----------
DATA="${1:-}"
THREADS="${2:-}"
LICENSE="${3:-}"
SUBJECT_FILTER="${4:-}"

# default docker images
FASTSURFER_IMAGE="${FASTSURFER_IMAGE:-deepmi/fastsurfer:latest}"
FREESURFER_IMAGE="${FREESURFER_IMAGE:-freesurfer/freesurfer:7.4.1}"
ATLAS_BACKEND="${ATLAS_BACKEND:-docker}"
RECON_ALL_EXTRA_ARGS="${RECON_ALL_EXTRA_ARGS:--cw256}"
read -r -a RECON_ALL_EXTRA_ARGS_ARRAY <<< "${RECON_ALL_EXTRA_ARGS}"

FASTSURFER_DOCKER_FLAGS_RAW="${FASTSURFER_DOCKER_FLAGS:-}"
declare -a FASTSURFER_DOCKER_RUN_FLAGS=()
if [[ -n "${FASTSURFER_DOCKER_FLAGS_RAW}" ]]; then
  read -r -a FASTSURFER_DOCKER_RUN_FLAGS <<< "${FASTSURFER_DOCKER_FLAGS_RAW}"
elif truthy "${FASTSURFER_USE_GPU:-0}"; then
  FASTSURFER_DOCKER_RUN_FLAGS=(--gpus "${FASTSURFER_DOCKER_GPUS:-all}")
fi

declare -a FASTSURFER_RUN_EXTRA_ARGS=()
if [[ -n "${FASTSURFER_DEVICE:-}" ]]; then
  FASTSURFER_RUN_EXTRA_ARGS+=(--device "${FASTSURFER_DEVICE}")
fi

# ---------- validate ----------
if [[ -z "${DATA}" || -z "${THREADS}" || -z "${LICENSE}" ]]; then
  echo "Usage: $0 <DATA_DIR> <THREADS> <LICENSE_PATH> [SUBJECT_ID]" >&2
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

case "${ATLAS_BACKEND}" in
  docker)
    if ! command -v docker >/dev/null 2>&1; then
      echo "ERROR: docker not found in PATH." >&2
      exit 1
    fi
    ;;
  native)
    if ! command -v recon-all >/dev/null 2>&1; then
      echo "ERROR: recon-all not found in PATH. Load the FreeSurfer module or set ATLAS_BACKEND=docker." >&2
      exit 1
    fi
    if ! command -v mri_convert >/dev/null 2>&1; then
      echo "ERROR: mri_convert not found in PATH. Load the FreeSurfer module or set ATLAS_BACKEND=docker." >&2
      exit 1
    fi
    ;;
  *)
    echo "ERROR: ATLAS_BACKEND must be 'docker' or 'native'. Received: ${ATLAS_BACKEND}" >&2
    exit 1
    ;;
esac

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
log "Backend       : ${ATLAS_BACKEND}"
log "Recon args    : ${RECON_ALL_EXTRA_ARGS:-<none>}"
log "FastSurfer    : ${FASTSURFER_IMAGE}"
log "FastSurfer GPU: ${FASTSURFER_DOCKER_RUN_FLAGS[*]:-<none>}"
log "FastSurfer args: ${FASTSURFER_RUN_EXTRA_ARGS[*]:-<none>}"
log "FreeSurfer    : ${FREESURFER_IMAGE}"
log "Output dir    : ${FASTSURFER_OUT_DIR}"
if [[ -n "${SUBJECT_FILTER}" ]]; then
  log "Subject filter: ${SUBJECT_FILTER}"
fi
log "-----------------------------------------------"

declare -a SUBJECT_IDS=()
if [[ -n "${SUBJECT_FILTER}" ]]; then
  SUBJECT_IDS=("${SUBJECT_FILTER}")
else
  while IFS= read -r -d '' dir; do
    SUBJECT_IDS+=("$(basename "${dir}")")
  done < <(find "${SUBJECT_ROOT}" -mindepth 1 -maxdepth 1 -type d ! -name "FastSurfer_out" -print0)
fi

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
  SUBJ_DIR_HOST="${FASTSURFER_OUT_DIR}/${SID}"

  if [[ "${SID}" == "FastSurfer_out" ]]; then
    continue
  fi

  if [[ ! -d "${SUBJ_PATH}" ]]; then
    log "[skip] ${SID}: Not a directory at ${SUBJ_PATH}"
    ((completed_count+=1))
    report_progress "${completed_count}" "${total_subjects}" "${total_subject_duration}" "${timed_count}"
    continue
  fi

  if ! T1_HOST="$(t1_host_for "${DATA_ROOT}" "${SID}")"; then
    log "[skip] ${SID}: Missing T1 image at ${SUBJ_PATH}/anat/${SID}_T1w.nii[.gz]"
    failures+=("${SID}")
    ((completed_count+=1))
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
    ((success_count+=1))
  else
    log "[error] ${SID}: failed after ${duration_formatted}. See details above."
    failures+=("${SID}")
  fi

  ((completed_count+=1))
  ((timed_count+=1))
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
