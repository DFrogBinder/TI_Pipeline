#!/bin/bash
#SBATCH --job-name=mni152_atlas
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=mni152_atlas_%j.out
#SBATCH --error=mni152_atlas_%j.err

set -euo pipefail

echo "[INFO] Host: $(hostname)"
echo "[INFO] JobID: ${SLURM_JOB_ID:-<none>}"
echo "[INFO] Start: $(date)"
echo

# Required:
#   MNI_T1=/path/to/MNI152_T1_1mm.nii.gz
#
# Optional overrides:
#   MNI_SUBJECT_ID=sub-mni152
#   SUBJECTS_DIR=/mnt/parscratch/users/cop23bi/ti_dataset/derivatives/freesurfer
#   MNI_ATLAS_DEST=/mnt/parscratch/users/cop23bi/atlases_new
#   FS_LICENSE=/users/cop23bi/freesurfer_licence.txt
#   FREESURFER_MODULE=FreeSurfer/7.4.1-centos7_x86_64
#   SOURCE_ATLAS=aparc.a2009s+aseg.mgz
#   OUTPUT_FILENAME=sub-mni152.nii.gz
#   RECON_ALL_EXTRA_ARGS="-cw256"

MNI_T1="${MNI_T1:-}"
MNI_SUBJECT_ID="${MNI_SUBJECT_ID:-sub-mni152}"
SUBJECTS_DIR="${SUBJECTS_DIR:-/mnt/parscratch/users/cop23bi/ti_dataset/derivatives/freesurfer}"
MNI_ATLAS_DEST="${MNI_ATLAS_DEST:-/mnt/parscratch/users/cop23bi/atlases_new}"
FS_LICENSE="${FS_LICENSE:-/users/cop23bi/freesurfer_licence.txt}"
FREESURFER_MODULE="${FREESURFER_MODULE:-FreeSurfer/7.4.1-centos7_x86_64}"
SOURCE_ATLAS="${SOURCE_ATLAS:-aparc.a2009s+aseg.mgz}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-${MNI_SUBJECT_ID}.nii.gz}"
RECON_ALL_EXTRA_ARGS="${RECON_ALL_EXTRA_ARGS:--cw256}"

# FreeSurfer modules often set SUBJECTS_DIR to the installation's own
# read-only subjects directory. Preserve the requested job paths before
# module load and restore them afterwards.
REQUESTED_SUBJECTS_DIR="${SUBJECTS_DIR}"
REQUESTED_FS_LICENSE="${FS_LICENSE}"

if [[ -z "${MNI_T1}" ]]; then
    echo "[ERROR] MNI_T1 is required." >&2
    echo "[ERROR] Example: sbatch --export=ALL,MNI_T1=/path/to/MNI152_T1_1mm.nii.gz $0" >&2
    exit 2
fi

if [[ ! -f "${MNI_T1}" ]]; then
    echo "[ERROR] MNI_T1 does not exist: ${MNI_T1}" >&2
    exit 2
fi

if [[ ! -f "${FS_LICENSE}" ]]; then
    echo "[ERROR] FS_LICENSE does not exist: ${FS_LICENSE}" >&2
    exit 2
fi

if ! type module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/modules.sh ]]; then
        # Some Slurm batch shells do not initialize the module command by default.
        # shellcheck disable=SC1091
        source /etc/profile.d/modules.sh
    elif [[ -f /usr/share/Modules/init/bash ]]; then
        # shellcheck disable=SC1091
        source /usr/share/Modules/init/bash
    fi
fi

if type module >/dev/null 2>&1; then
    echo "[INFO] Loading FreeSurfer module: ${FREESURFER_MODULE}"
    module load "${FREESURFER_MODULE}"
else
    echo "[ERROR] Environment modules are not available; cannot load ${FREESURFER_MODULE}." >&2
    exit 2
fi

if ! command -v recon-all >/dev/null 2>&1; then
    echo "[ERROR] recon-all was not found in PATH after loading ${FREESURFER_MODULE}." >&2
    exit 2
fi

if ! command -v mri_convert >/dev/null 2>&1; then
    echo "[ERROR] mri_convert was not found in PATH after loading ${FREESURFER_MODULE}." >&2
    exit 2
fi

SUBJECTS_DIR="${REQUESTED_SUBJECTS_DIR}"
FS_LICENSE="${REQUESTED_FS_LICENSE}"
export SUBJECTS_DIR
export FS_LICENSE
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
read -r -a RECON_ALL_EXTRA_ARGS_ARRAY <<< "${RECON_ALL_EXTRA_ARGS}"

SUBJECT_DIR="${SUBJECTS_DIR}/${MNI_SUBJECT_ID}"
SOURCE_ATLAS_PATH="${SUBJECT_DIR}/mri/${SOURCE_ATLAS}"
OUTPUT_ATLAS_PATH="${MNI_ATLAS_DEST}/${OUTPUT_FILENAME}"

echo "[INFO] MNI T1:        ${MNI_T1}"
echo "[INFO] Subject ID:    ${MNI_SUBJECT_ID}"
echo "[INFO] SUBJECTS_DIR:  ${SUBJECTS_DIR}"
echo "[INFO] FS license:    ${FS_LICENSE}"
echo "[INFO] Source atlas:  ${SOURCE_ATLAS_PATH}"
echo "[INFO] Output atlas:  ${OUTPUT_ATLAS_PATH}"
echo "[INFO] Threads:       ${OMP_NUM_THREADS}"
echo "[INFO] Extra recon-all args: ${RECON_ALL_EXTRA_ARGS:-<none>}"
echo

mkdir -p "${SUBJECTS_DIR}"

if [[ -f "${SUBJECT_DIR}/scripts/recon-all.done" && -f "${SOURCE_ATLAS_PATH}" ]]; then
    echo "[INFO] Existing completed recon found; skipping recon-all."
elif [[ -d "${SUBJECT_DIR}" ]]; then
    echo "[INFO] Existing subject directory found; continuing recon-all without -i."
    recon-all -s "${MNI_SUBJECT_ID}" -all -openmp "${OMP_NUM_THREADS}" "${RECON_ALL_EXTRA_ARGS_ARRAY[@]}"
else
    echo "[INFO] Starting new recon-all from MNI_T1."
    recon-all -s "${MNI_SUBJECT_ID}" -i "${MNI_T1}" -all -openmp "${OMP_NUM_THREADS}" "${RECON_ALL_EXTRA_ARGS_ARRAY[@]}"
fi

if [[ ! -f "${SOURCE_ATLAS_PATH}" ]]; then
    echo "[ERROR] Expected atlas was not produced: ${SOURCE_ATLAS_PATH}" >&2
    echo "[ERROR] Available MRI files:" >&2
    find "${SUBJECT_DIR}/mri" -maxdepth 1 -type f -printf "  %f\n" 2>/dev/null | sort >&2 || true
    exit 1
fi

mkdir -p "${MNI_ATLAS_DEST}"
mri_convert "${SOURCE_ATLAS_PATH}" "${OUTPUT_ATLAS_PATH}"

echo
echo "[INFO] Finished: $(date)"
echo "[INFO] Wrote atlas: ${OUTPUT_ATLAS_PATH}"
