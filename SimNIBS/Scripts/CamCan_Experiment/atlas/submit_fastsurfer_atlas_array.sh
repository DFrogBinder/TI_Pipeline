#!/usr/bin/env bash
#
# Edit the configuration at the top of this file, then submit it directly:
#
#   sbatch submit_fastsurfer_atlas_array.sh
#
# The array range must cover every discovered subject index. For example, if
# local validation reports 420 subjects, keep the start at 0 and set the end to
# at least 419. The value after % controls overnight concurrency.
#
#SBATCH --job-name=freesurfer_atlas
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-999%64
#SBATCH --output=freesurfer_atlas_%A_%a.out
#SBATCH --error=freesurfer_atlas_%A_%a.err
#SBATCH --requeue

# ----------------------------
# User configuration
# ----------------------------

DATA_ROOT="/path/to/CamCan_Data"
OUTPUT_ROOT="/path/to/FreeSurfer_atlases"

FREESURFER_MODULE="FreeSurfer"
FS_LICENSE_FILE=""
FREESURFER_ATLAS_MGZ="mri/aparc+aseg.mgz"

# Set to 1 to print what would run without loading modules or running FreeSurfer.
DRY_RUN="0"

# Set to 1 to rerun conversion even when <OUTPUT_ROOT>/<subject>.nii.gz exists.
REPROCESS_EXISTING="0"

# ----------------------------
# End user configuration
# ----------------------------

set -euo pipefail

declare -a SUBJECT_IDS=()

log() {
  printf '%s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  printf '%s | [ERROR] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
  exit 1
}

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

script_path() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  printf '%s/%s\n' "${script_dir}" "$(basename "${BASH_SOURCE[0]}")"
}

array_directive() {
  awk '$1 == "#SBATCH" && $2 ~ /^--array=/ { sub(/^--array=/, "", $2); print $2; exit }' "$(script_path)"
}

array_max_index() {
  local spec="$1"
  local range="${spec%%%*}"

  if [[ "${range}" =~ ^[0-9]+-([0-9]+)$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  elif [[ "${range}" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "${range}"
  fi
}

validate_no_args() {
  if [[ $# -gt 0 ]]; then
    die "This script is configured only by editing the top-of-file settings; do not pass command-line arguments."
  fi
}

resolve_config_paths() {
  if [[ -z "${DATA_ROOT}" || "${DATA_ROOT}" == "/path/to/CamCan_Data" ]]; then
    die "Edit DATA_ROOT at the top of this file before running."
  fi
  if [[ -z "${OUTPUT_ROOT}" || "${OUTPUT_ROOT}" == "/path/to/FreeSurfer_atlases" ]]; then
    die "Edit OUTPUT_ROOT at the top of this file before running."
  fi
  if [[ -z "${FREESURFER_ATLAS_MGZ}" || "${FREESURFER_ATLAS_MGZ}" == /* ]]; then
    die "FREESURFER_ATLAS_MGZ must be a relative path inside each FreeSurfer subject directory."
  fi
  if [[ ! -d "${DATA_ROOT}" ]]; then
    die "DATA_ROOT does not exist: ${DATA_ROOT}"
  fi

  DATA_ROOT="$(cd "${DATA_ROOT}" && pwd -P)"
  mkdir -p "${OUTPUT_ROOT}"
  OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd -P)"

  if [[ -n "${FS_LICENSE_FILE}" ]]; then
    if [[ ! -f "${FS_LICENSE_FILE}" ]]; then
      die "FS_LICENSE_FILE does not exist: ${FS_LICENSE_FILE}"
    fi
    local license_dir
    license_dir="$(cd "$(dirname "${FS_LICENSE_FILE}")" && pwd -P)"
    FS_LICENSE_FILE="${license_dir}/$(basename "${FS_LICENSE_FILE}")"
  fi

  mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/slurm" "${OUTPUT_ROOT}/subjects"
}

discover_subjects() {
  mapfile -t SUBJECT_IDS < <(
    find "${DATA_ROOT}" -mindepth 1 -maxdepth 1 -type d -print |
      while IFS= read -r subject_dir; do
        subject="$(basename "${subject_dir}")"
        t1_path="${subject_dir}/anat/${subject}_T1w.nii.gz"
        if [[ -f "${t1_path}" ]]; then
          printf '%s\n' "${subject}"
        fi
      done |
      LC_ALL=C sort
  )
}

write_subjects_file() {
  local subjects_file="${OUTPUT_ROOT}/slurm/subjects.txt"
  printf '%s\n' "${SUBJECT_IDS[@]}" > "${subjects_file}"
}

print_local_summary() {
  local subject_count="${#SUBJECT_IDS[@]}"
  local required_end=$((subject_count - 1))
  local subjects_file="${OUTPUT_ROOT}/slurm/subjects.txt"
  local current_array
  local current_end
  current_array="$(array_directive)"
  current_end="$(array_max_index "${current_array}")"

  printf '[INFO] Data root:       %s\n' "${DATA_ROOT}"
  printf '[INFO] Output root:     %s\n' "${OUTPUT_ROOT}"
  printf '[INFO] Subjects file:   %s\n' "${subjects_file}"
  printf '[INFO] Subject count:   %s\n' "${subject_count}"
  printf '[INFO] Required array:  0-%s\n' "${required_end}"
  printf '[INFO] Current array:   %s\n' "${current_array}"
  if [[ -n "${current_end}" ]] && (( current_end < required_end )); then
    printf '[WARN] Current #SBATCH --array=%s only covers through index %s; edit it to at least 0-%s.\n' "${current_array}" "${current_end}" "${required_end}"
  fi
  printf '[INFO] Submit with:     sbatch %s\n' "$(script_path)"

  if is_truthy "${DRY_RUN}"; then
    printf '[DRY-RUN] Local validation only; no Slurm job was submitted.\n'
  else
    printf '[INFO] Local validation only; submit the script with sbatch when ready.\n'
  fi
}

run_local_validation() {
  if (( ${#SUBJECT_IDS[@]} == 0 )); then
    die "No subjects with T1 input found under ${DATA_ROOT}."
  fi

  write_subjects_file
  print_local_summary
}

thread_count() {
  printf '%s\n' "${SLURM_CPUS_PER_TASK:-16}"
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

load_modules() {
  if ! command -v module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/modules.sh ]]; then
      # shellcheck disable=SC1091
      source /etc/profile.d/modules.sh
    fi
  fi

  module purge || true
  module load "${FREESURFER_MODULE}"
}

run_subject_task() {
  if [[ ! "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]]; then
    die "SLURM_ARRAY_TASK_ID must be a non-negative integer. Received: ${SLURM_ARRAY_TASK_ID}"
  fi

  local task_id="${SLURM_ARRAY_TASK_ID}"
  local subject_count="${#SUBJECT_IDS[@]}"

  if (( task_id >= subject_count )); then
    log "[skip] Array index ${task_id} is outside the discovered subject range 0-$((subject_count - 1))."
    exit 0
  fi

  local subject="${SUBJECT_IDS[task_id]}"
  local subjects_dir="${OUTPUT_ROOT}/subjects"
  local t1_input="${DATA_ROOT}/${subject}/anat/${subject}_T1w.nii.gz"
  local subject_log="${OUTPUT_ROOT}/logs/${subject}.log"
  local atlas_mgz="${subjects_dir}/${subject}/${FREESURFER_ATLAS_MGZ}"
  local atlas_flat="${OUTPUT_ROOT}/${subject}.nii.gz"
  local threads
  threads="$(thread_count)"

  mkdir -p "${subjects_dir}" "${OUTPUT_ROOT}/logs"
  exec > >(tee -a "${subject_log}") 2>&1

  log "Host:        $(hostname)"
  log "Job ID:      ${SLURM_JOB_ID:-unknown}"
  log "Array index: ${SLURM_ARRAY_TASK_ID}"
  log "Subject:     ${subject}"
  log "Data root:   ${DATA_ROOT}"
  log "Output root: ${OUTPUT_ROOT}"
  log "Threads:     ${threads}"
  log "Atlas flat:  ${atlas_flat}"

  [[ -f "${t1_input}" ]] || die "Missing T1 input: ${t1_input}"

  if [[ -s "${atlas_flat}" ]] && ! is_truthy "${REPROCESS_EXISTING}"; then
    log "[ok] Flat atlas already exists. Skipping subject."
    exit 0
  fi

  local -a recon_all_cmd=(
    recon-all
    -s "${subject}"
    -i "${t1_input}"
    -sd "${subjects_dir}"
    -all
    -openmp "${threads}"
    -cw256
  )
  local -a convert_cmd=(mri_convert "${atlas_mgz}" "${atlas_flat}")

  if [[ -n "${FS_LICENSE_FILE}" ]]; then
    export FS_LICENSE="${FS_LICENSE_FILE}"
  fi

  if is_truthy "${DRY_RUN}"; then
    log "[DRY-RUN] Would run FreeSurfer recon-all command:"
    print_command "${recon_all_cmd[@]}"
    log "[DRY-RUN] Would convert atlas:"
    print_command "${convert_cmd[@]}"
    exit 0
  fi

  load_modules

  export OMP_NUM_THREADS="${threads}"
  export MKL_NUM_THREADS="${threads}"
  export OPENBLAS_NUM_THREADS="${threads}"
  export NUMEXPR_NUM_THREADS="${threads}"

  if [[ -n "${FS_LICENSE_FILE}" ]]; then
    export FS_LICENSE="${FS_LICENSE_FILE}"
    log "FS_LICENSE:  ${FS_LICENSE}"
  elif [[ -n "${FS_LICENSE:-}" ]]; then
    log "FS_LICENSE:  ${FS_LICENSE}"
  else
    log "FS_LICENSE:  <not set>"
  fi

  command -v recon-all >/dev/null 2>&1 || die "recon-all not found after loading ${FREESURFER_MODULE}."
  command -v mri_convert >/dev/null 2>&1 || die "mri_convert not found after loading ${FREESURFER_MODULE}."

  if [[ -s "${atlas_mgz}" ]] && ! is_truthy "${REPROCESS_EXISTING}"; then
    log "[ok] FreeSurfer atlas MGZ already exists: ${atlas_mgz}"
  else
    log "[run] FreeSurfer recon-all"
    "${recon_all_cmd[@]}"
    log "[done] FreeSurfer recon-all"
  fi

  [[ -s "${atlas_mgz}" ]] || die "Expected atlas MGZ was not created: ${atlas_mgz}"

  log "[run] Converting atlas MGZ to flat NIfTI"
  "${convert_cmd[@]}"
  log "[done] Wrote ${atlas_flat}"
}

main() {
  validate_no_args "$@"
  resolve_config_paths
  discover_subjects

  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    run_local_validation
  else
    run_subject_task
  fi
}

main "$@"
