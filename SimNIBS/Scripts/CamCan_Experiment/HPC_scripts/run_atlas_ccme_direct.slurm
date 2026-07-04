#!/bin/bash
#SBATCH --job-name=atlas_ccme_direct
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=atlas_ccme_direct_%j.out
#SBATCH --error=atlas_ccme_direct_%j.err

set -euo pipefail

echo "[INFO] Host: $(hostname)"
echo "[INFO] JobID: ${SLURM_JOB_ID:-<none>}"
echo "[INFO] Start: $(date)"
echo

# Direct FreeSurfer atlas generation for the defaced sub-CCMe source tree.
# This intentionally follows HPC_scripts/run_mni152_atlas.slurm rather than
# going through atlas/run_atlasMaker.py or atlas/make_atlas.sh.
#
# Optional overrides:
#   ATLAS_SUBJECT_ID=sub-CCMe
#   ATLAS_T1=/path/to/sub-CCMe_T1w.nii.gz
#   ATLAS_DATA_DIR=/mnt/parscratch/users/cop23bi/defacing_sub_CCMe/source
#   SUBJECTS_DIR=/mnt/parscratch/users/cop23bi/defacing_sub_CCMe/source/FastSurfer_out
#   ATLAS_DEST=/mnt/parscratch/users/cop23bi/ZIPs/atlases
#   FS_LICENSE=/users/cop23bi/freesurfer_licence.txt
#   FREESURFER_MODULE=FreeSurfer/7.4.1-centos7_x86_64
#   SOURCE_ATLAS=aparc.a2009s+aseg.mgz
#   OUTPUT_FILENAME=sub-CCMe.nii.gz
#   RECON_ALL_EXTRA_ARGS="-cw256"

ATLAS_SUBJECT_ID="${ATLAS_SUBJECT_ID:-sub-CCMe}"
ATLAS_DATA_DIR="${ATLAS_DATA_DIR:-/mnt/parscratch/users/cop23bi/defacing_sub_CCMe/source}"
ATLAS_T1="${ATLAS_T1:-}"
SUBJECTS_DIR="${SUBJECTS_DIR:-${ATLAS_DATA_DIR}/FastSurfer_out}"
ATLAS_DEST="${ATLAS_DEST:-/mnt/parscratch/users/cop23bi/ZIPs/atlases}"
FS_LICENSE="${FS_LICENSE:-/users/cop23bi/freesurfer_licence.txt}"
FREESURFER_MODULE="${FREESURFER_MODULE:-FreeSurfer/7.4.1-centos7_x86_64}"
SOURCE_ATLAS="${SOURCE_ATLAS:-aparc.a2009s+aseg.mgz}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-${ATLAS_SUBJECT_ID}.nii.gz}"
RECON_ALL_EXTRA_ARGS="${RECON_ALL_EXTRA_ARGS:--cw256}"

if [[ -z "${ATLAS_T1}" ]]; then
    for candidate in \
        "${ATLAS_DATA_DIR}/${ATLAS_SUBJECT_ID}/anat/${ATLAS_SUBJECT_ID}_T1w.nii.gz" \
        "${ATLAS_DATA_DIR}/${ATLAS_SUBJECT_ID}/anat/${ATLAS_SUBJECT_ID}_T1w.nii"; do
        if [[ -f "${candidate}" ]]; then
            ATLAS_T1="${candidate}"
            break
        fi
    done
fi

# FreeSurfer modules often set SUBJECTS_DIR/FS_LICENSE themselves. Preserve the
# requested job paths before module load and restore them afterwards.
REQUESTED_SUBJECTS_DIR="${SUBJECTS_DIR}"
REQUESTED_FS_LICENSE="${FS_LICENSE}"

if [[ -z "${ATLAS_T1}" ]]; then
    echo "[ERROR] Could not find ATLAS_T1." >&2
    echo "[ERROR] Expected ${ATLAS_DATA_DIR}/${ATLAS_SUBJECT_ID}/anat/${ATLAS_SUBJECT_ID}_T1w.nii.gz or .nii" >&2
    exit 2
fi

if [[ ! -f "${ATLAS_T1}" ]]; then
    echo "[ERROR] ATLAS_T1 does not exist: ${ATLAS_T1}" >&2
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-9}"
read -r -a RECON_ALL_EXTRA_ARGS_ARRAY <<< "${RECON_ALL_EXTRA_ARGS}"

SUBJECT_DIR="${SUBJECTS_DIR}/${ATLAS_SUBJECT_ID}"
SOURCE_ATLAS_PATH="${SUBJECT_DIR}/mri/${SOURCE_ATLAS}"
OUTPUT_ATLAS_PATH="${ATLAS_DEST}/${OUTPUT_FILENAME}"

echo "[INFO] T1 input:      ${ATLAS_T1}"
echo "[INFO] Subject ID:    ${ATLAS_SUBJECT_ID}"
echo "[INFO] SUBJECTS_DIR:  ${SUBJECTS_DIR}"
echo "[INFO] FS license:    ${FS_LICENSE}"
echo "[INFO] Source atlas:  ${SOURCE_ATLAS_PATH}"
echo "[INFO] Output atlas:  ${OUTPUT_ATLAS_PATH}"
echo "[INFO] Threads:       ${OMP_NUM_THREADS}"
echo "[INFO] Extra recon-all args: ${RECON_ALL_EXTRA_ARGS:-<none>}"
echo "[INFO] recon-all:     $(command -v recon-all)"
echo "[INFO] mri_convert:   $(command -v mri_convert)"
echo

mkdir -p "${SUBJECTS_DIR}"

if [[ -f "${SUBJECT_DIR}/scripts/recon-all.done" && -f "${SOURCE_ATLAS_PATH}" ]]; then
    echo "[INFO] Existing completed recon found; skipping recon-all."
elif [[ -d "${SUBJECT_DIR}" ]]; then
    if [[ -z "$(find "${SUBJECT_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
        echo "[INFO] Empty stale subject directory found; removing it before initial recon-all."
        rmdir "${SUBJECT_DIR}"
        echo "[INFO] Starting new recon-all from ATLAS_T1."
        recon-all -s "${ATLAS_SUBJECT_ID}" -i "${ATLAS_T1}" -all -openmp "${OMP_NUM_THREADS}" "${RECON_ALL_EXTRA_ARGS_ARRAY[@]}"
    else
        echo "[INFO] Existing subject directory found; continuing recon-all without -i."
        recon-all -s "${ATLAS_SUBJECT_ID}" -all -openmp "${OMP_NUM_THREADS}" "${RECON_ALL_EXTRA_ARGS_ARRAY[@]}"
    fi
else
    echo "[INFO] Starting new recon-all from ATLAS_T1."
    recon-all -s "${ATLAS_SUBJECT_ID}" -i "${ATLAS_T1}" -all -openmp "${OMP_NUM_THREADS}" "${RECON_ALL_EXTRA_ARGS_ARRAY[@]}"
fi

if [[ ! -f "${SOURCE_ATLAS_PATH}" ]]; then
    echo "[ERROR] Expected atlas was not produced: ${SOURCE_ATLAS_PATH}" >&2
    echo "[ERROR] Available MRI files:" >&2
    find "${SUBJECT_DIR}/mri" -maxdepth 1 -type f -printf "  %f\n" 2>/dev/null | sort >&2 || true
    exit 1
fi

mkdir -p "${ATLAS_DEST}"
mri_convert "${SOURCE_ATLAS_PATH}" "${OUTPUT_ATLAS_PATH}"

echo
echo "[INFO] Finished: $(date)"
echo "[INFO] Wrote atlas: ${OUTPUT_ATLAS_PATH}"
