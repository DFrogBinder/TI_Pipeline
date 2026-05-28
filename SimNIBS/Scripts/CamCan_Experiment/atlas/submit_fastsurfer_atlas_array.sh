#!/usr/bin/env bash
#
# Submit module-based FastSurfer atlas generation as one Slurm array task per
# subject. This intentionally avoids Docker: each task loads FastSurfer and
# FreeSurfer modules on the HPC, runs one subject, and writes a flat atlas NIfTI
# at <OUTPUT_ROOT>/<subject>.nii.gz for downstream post-processing.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  submit_fastsurfer_atlas_array.sh <DATA_ROOT> <OUTPUT_ROOT> [options]

Required:
  DATA_ROOT    Directory containing subject folders:
               <DATA_ROOT>/<subject>/anat/<subject>_T1w.nii.gz
  OUTPUT_ROOT  Directory where FastSurfer outputs and flat atlas NIfTIs go.

Options:
  --partition NAME             Slurm partition (default: sheffield)
  --cpus N                     CPUs per subject job (default: 16)
  --mem SIZE                   Memory per subject job (default: 64G)
  --time HH:MM:SS              Wall time per subject job (default: 08:00:00)
  --max-concurrent N           Max simultaneous array tasks (default: 64)
  --fastsurfer-module NAME     Module to load for FastSurfer (default: FastSurfer)
  --freesurfer-module NAME     Module to load for mri_convert (default: FreeSurfer)
  --license PATH               FreeSurfer license path; defaults to $FS_LICENSE if set
  --dry-run                    Write subject list and job script but do not call sbatch
  -h, --help                   Show this help

Outputs:
  <OUTPUT_ROOT>/subjects/<subject>/...  Full FastSurfer subject directory
  <OUTPUT_ROOT>/<subject>.nii.gz        Flat aparc.DKTatlas+aseg.deep atlas
  <OUTPUT_ROOT>/slurm/...               Generated subject list and Slurm script
  <OUTPUT_ROOT>/logs/...                Slurm and per-subject logs
USAGE
}

die() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "${value}" ]]; then
    die "${option} requires a value."
  fi
}

is_positive_int() {
  [[ "${1:-}" =~ ^[1-9][0-9]*$ ]]
}

DATA_ARG="${1:-}"
OUTPUT_ARG="${2:-}"

if [[ "${DATA_ARG}" == "-h" || "${DATA_ARG}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${DATA_ARG}" || -z "${OUTPUT_ARG}" ]]; then
  usage >&2
  exit 2
fi

shift 2

PARTITION="sheffield"
CPUS="16"
MEM="64G"
WALLTIME="08:00:00"
MAX_CONCURRENT="64"
FASTSURFER_MODULE="FastSurfer"
FREESURFER_MODULE="FreeSurfer"
LICENSE_PATH="${FS_LICENSE:-}"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition)
      require_value "$1" "${2:-}"
      PARTITION="$2"
      shift 2
      ;;
    --cpus)
      require_value "$1" "${2:-}"
      CPUS="$2"
      shift 2
      ;;
    --mem)
      require_value "$1" "${2:-}"
      MEM="$2"
      shift 2
      ;;
    --time)
      require_value "$1" "${2:-}"
      WALLTIME="$2"
      shift 2
      ;;
    --max-concurrent)
      require_value "$1" "${2:-}"
      MAX_CONCURRENT="$2"
      shift 2
      ;;
    --fastsurfer-module)
      require_value "$1" "${2:-}"
      FASTSURFER_MODULE="$2"
      shift 2
      ;;
    --freesurfer-module)
      require_value "$1" "${2:-}"
      FREESURFER_MODULE="$2"
      shift 2
      ;;
    --license)
      require_value "$1" "${2:-}"
      LICENSE_PATH="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

is_positive_int "${CPUS}" || die "--cpus must be a positive integer. Received: ${CPUS}"
is_positive_int "${MAX_CONCURRENT}" || die "--max-concurrent must be a positive integer. Received: ${MAX_CONCURRENT}"

if [[ ! "${WALLTIME}" =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]]; then
  die "--time must look like HH:MM:SS. Received: ${WALLTIME}"
fi

if [[ ! -d "${DATA_ARG}" ]]; then
  die "DATA_ROOT does not exist: ${DATA_ARG}"
fi

DATA_ROOT="$(cd "${DATA_ARG}" && pwd -P)"
mkdir -p "${OUTPUT_ARG}"
OUTPUT_ROOT="$(cd "${OUTPUT_ARG}" && pwd -P)"

if [[ -n "${LICENSE_PATH}" ]]; then
  if [[ ! -f "${LICENSE_PATH}" ]]; then
    die "FreeSurfer license file not found: ${LICENSE_PATH}"
  fi
  LICENSE_DIR="$(cd "$(dirname "${LICENSE_PATH}")" && pwd -P)"
  LICENSE_PATH="${LICENSE_DIR}/$(basename "${LICENSE_PATH}")"
fi

SLURM_DIR="${OUTPUT_ROOT}/slurm"
LOG_DIR="${OUTPUT_ROOT}/logs"
SUBJECTS_FILE="${SLURM_DIR}/subjects.txt"
BATCH_SCRIPT="${SLURM_DIR}/fastsurfer_atlas_subject.slurm"
SUBMIT_COMMAND_FILE="${SLURM_DIR}/submit_command.txt"

mkdir -p "${SLURM_DIR}" "${LOG_DIR}" "${OUTPUT_ROOT}/subjects"

SUBJECTS_TMP="$(mktemp)"
trap 'rm -f "${SUBJECTS_TMP}"' EXIT

while IFS= read -r -d '' subject_dir; do
  subject="$(basename "${subject_dir}")"
  t1_path="${subject_dir}/anat/${subject}_T1w.nii.gz"
  if [[ -f "${t1_path}" ]]; then
    printf '%s\n' "${subject}" >> "${SUBJECTS_TMP}"
  fi
done < <(find "${DATA_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0)

LC_ALL=C sort -u "${SUBJECTS_TMP}" > "${SUBJECTS_FILE}"

SUBJECT_COUNT="$(awk 'END { print NR }' "${SUBJECTS_FILE}")"
if [[ "${SUBJECT_COUNT}" -eq 0 ]]; then
  die "No subjects with T1 input found under ${DATA_ROOT}."
fi

ARRAY_END=$((SUBJECT_COUNT - 1))
ARRAY_SPEC="0-${ARRAY_END}%${MAX_CONCURRENT}"

cat > "${BATCH_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH --job-name=fs_atlas
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${WALLTIME}
#SBATCH --output=${LOG_DIR}/fastsurfer_atlas_%A_%a.out
#SBATCH --error=${LOG_DIR}/fastsurfer_atlas_%A_%a.err
#SBATCH --requeue

set -euo pipefail

log() {
  printf '%s | %s\n' "\$(date '+%Y-%m-%d %H:%M:%S')" "\$*"
}

die() {
  printf '%s | [ERROR] %s\n' "\$(date '+%Y-%m-%d %H:%M:%S')" "\$*" >&2
  exit 1
}

if [[ -z "\${SLURM_ARRAY_TASK_ID:-}" ]]; then
  die "This script must be submitted as a Slurm array task."
fi

: "\${DATA_ROOT:?DATA_ROOT is required}"
: "\${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "\${SUBJECTS_FILE:?SUBJECTS_FILE is required}"

SUBJECT="\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "\${SUBJECTS_FILE}")"
if [[ -z "\${SUBJECT}" ]]; then
  die "No subject found for array index \${SLURM_ARRAY_TASK_ID} in \${SUBJECTS_FILE}."
fi

SUBJECTS_DIR="\${OUTPUT_ROOT}/subjects"
T1_INPUT="\${DATA_ROOT}/\${SUBJECT}/anat/\${SUBJECT}_T1w.nii.gz"
SUBJECT_LOG="\${OUTPUT_ROOT}/logs/\${SUBJECT}.log"
ATLAS_MGZ="\${SUBJECTS_DIR}/\${SUBJECT}/mri/aparc.DKTatlas+aseg.deep.mgz"
ATLAS_FLAT="\${OUTPUT_ROOT}/\${SUBJECT}.nii.gz"

mkdir -p "\${SUBJECTS_DIR}" "\${OUTPUT_ROOT}/logs"
exec > >(tee -a "\${SUBJECT_LOG}") 2>&1

log "Host:        \$(hostname)"
log "Job ID:      \${SLURM_JOB_ID}"
log "Array index: \${SLURM_ARRAY_TASK_ID}"
log "Subject:     \${SUBJECT}"
log "Data root:   \${DATA_ROOT}"
log "Output root: \${OUTPUT_ROOT}"
log "CPUs:        \${SLURM_CPUS_PER_TASK:-${CPUS}}"
log "Atlas flat:  \${ATLAS_FLAT}"

[[ -f "\${T1_INPUT}" ]] || die "Missing T1 input: \${T1_INPUT}"

if [[ -s "\${ATLAS_FLAT}" ]]; then
  log "[ok] Flat atlas already exists. Skipping subject."
  exit 0
fi

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi
fi

module purge || true
module load "\${FASTSURFER_MODULE}"
module load "\${FREESURFER_MODULE}"

export OMP_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-${CPUS}}"
export MKL_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-${CPUS}}"
export OPENBLAS_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-${CPUS}}"
export NUMEXPR_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-${CPUS}}"

if [[ -n "\${FS_LICENSE:-}" ]]; then
  export FS_LICENSE
  log "FS_LICENSE:  \${FS_LICENSE}"
else
  log "FS_LICENSE:  <not set>"
fi

command -v run_fastsurfer.sh >/dev/null 2>&1 || die "run_fastsurfer.sh not found after loading FastSurfer module."
command -v mri_convert >/dev/null 2>&1 || die "mri_convert not found after loading FreeSurfer module."

if [[ -s "\${ATLAS_MGZ}" ]]; then
  log "[ok] FastSurfer atlas MGZ already exists: \${ATLAS_MGZ}"
else
  FASTSURFER_CMD=(
    run_fastsurfer.sh
    --t1 "\${T1_INPUT}"
    --sid "\${SUBJECT}"
    --sd "\${SUBJECTS_DIR}"
    --seg_only
    --threads "\${SLURM_CPUS_PER_TASK:-${CPUS}}"
    --no_cereb
    --no_hypothal
  )

  if [[ -n "\${FS_LICENSE:-}" ]]; then
    FASTSURFER_CMD+=(--fs_license "\${FS_LICENSE}")
  fi

  log "[run] FastSurfer segmentation"
  "\${FASTSURFER_CMD[@]}"
  log "[done] FastSurfer segmentation"
fi

[[ -s "\${ATLAS_MGZ}" ]] || die "Expected atlas MGZ was not created: \${ATLAS_MGZ}"

log "[run] Converting atlas MGZ to flat NIfTI"
mri_convert "\${ATLAS_MGZ}" "\${ATLAS_FLAT}"
log "[done] Wrote \${ATLAS_FLAT}"
EOF

chmod +x "${BATCH_SCRIPT}"

EXPORT_VARS="ALL,DATA_ROOT=${DATA_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},SUBJECTS_FILE=${SUBJECTS_FILE},FASTSURFER_MODULE=${FASTSURFER_MODULE},FREESURFER_MODULE=${FREESURFER_MODULE}"
if [[ -n "${LICENSE_PATH}" ]]; then
  EXPORT_VARS="${EXPORT_VARS},FS_LICENSE=${LICENSE_PATH}"
fi

SBATCH_CMD=(
  sbatch
  "--array=${ARRAY_SPEC}"
  "--export=${EXPORT_VARS}"
  "${BATCH_SCRIPT}"
)

printf '%q ' "${SBATCH_CMD[@]}" > "${SUBMIT_COMMAND_FILE}"
printf '\n' >> "${SUBMIT_COMMAND_FILE}"

printf '[INFO] Data root:       %s\n' "${DATA_ROOT}"
printf '[INFO] Output root:     %s\n' "${OUTPUT_ROOT}"
printf '[INFO] Subjects file:   %s\n' "${SUBJECTS_FILE}"
printf '[INFO] Subject count:   %s\n' "${SUBJECT_COUNT}"
printf '[INFO] Array spec:      %s\n' "${ARRAY_SPEC}"
printf '[INFO] Batch script:    %s\n' "${BATCH_SCRIPT}"
printf '[INFO] Submit command:  %s\n' "${SUBMIT_COMMAND_FILE}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[DRY-RUN] Not submitting. Review %s and run the command in %s when ready.\n' "${BATCH_SCRIPT}" "${SUBMIT_COMMAND_FILE}"
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || die "sbatch not found. Use --dry-run off-cluster, or run this on the HPC login node."

"${SBATCH_CMD[@]}"
