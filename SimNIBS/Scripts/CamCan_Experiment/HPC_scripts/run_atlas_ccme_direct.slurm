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

set -uo pipefail

REPO_DIR="${REPO_DIR:-/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment}"
ATLAS_DATA_DIR="${ATLAS_DATA_DIR:-/mnt/parscratch/users/cop23bi/defacing_sub_CCMe/source}"
ATLAS_LICENSE_PATH="${ATLAS_LICENSE_PATH:-/users/cop23bi/freesurfer_licence.txt}"
ATLAS_SUBJECT="${ATLAS_SUBJECT:-sub-CCMe}"
ATLAS_THREADS_PER_JOB="${ATLAS_THREADS_PER_JOB:-${SLURM_CPUS_PER_TASK:-9}}"
FREESURFER_MODULE="${FREESURFER_MODULE:-FreeSurfer/7.4.1-centos7_x86_64}"
RECON_ALL_EXTRA_ARGS="${RECON_ALL_EXTRA_ARGS:--cw256}"
ATLAS_TRACE="${ATLAS_TRACE:-0}"

if ! type module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/modules.sh ]]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/modules.sh
    elif [[ -f /usr/share/Modules/init/bash ]]; then
        # shellcheck disable=SC1091
        source /usr/share/Modules/init/bash
    fi
fi

if type module >/dev/null 2>&1; then
    module load "${FREESURFER_MODULE}"
else
    echo "[ERROR] module command is not available in this batch shell." >&2
    exit 2
fi

echo "[INFO] Host:        $(hostname)"
echo "[INFO] JobID:       ${SLURM_JOB_ID:-<none>}"
echo "[INFO] Start:       $(date)"
echo "[INFO] Repository:  ${REPO_DIR}"
echo "[INFO] Data dir:    ${ATLAS_DATA_DIR}"
echo "[INFO] Subject:     ${ATLAS_SUBJECT}"
echo "[INFO] License:     ${ATLAS_LICENSE_PATH}"
echo "[INFO] Threads:     ${ATLAS_THREADS_PER_JOB}"
echo "[INFO] FS module:   ${FREESURFER_MODULE}"
echo "[INFO] Recon args:  ${RECON_ALL_EXTRA_ARGS}"
echo "[INFO] Trace mode:  ${ATLAS_TRACE}"
echo "[INFO] recon-all:   $(command -v recon-all || true)"
echo "[INFO] mri_convert: $(command -v mri_convert || true)"
echo

if [[ ! -d "${REPO_DIR}" ]]; then
    echo "[ERROR] Repository directory does not exist: ${REPO_DIR}" >&2
    exit 2
fi
if [[ ! -f "${REPO_DIR}/atlas/make_atlas.sh" ]]; then
    echo "[ERROR] Atlas script not found: ${REPO_DIR}/atlas/make_atlas.sh" >&2
    exit 2
fi
if [[ ! -d "${ATLAS_DATA_DIR}" ]]; then
    echo "[ERROR] ATLAS_DATA_DIR does not exist: ${ATLAS_DATA_DIR}" >&2
    exit 2
fi
if [[ ! -f "${ATLAS_LICENSE_PATH}" ]]; then
    echo "[ERROR] ATLAS_LICENSE_PATH does not exist: ${ATLAS_LICENSE_PATH}" >&2
    exit 2
fi
if ! command -v recon-all >/dev/null 2>&1; then
    echo "[ERROR] recon-all not found after loading ${FREESURFER_MODULE}" >&2
    exit 2
fi
if ! command -v mri_convert >/dev/null 2>&1; then
    echo "[ERROR] mri_convert not found after loading ${FREESURFER_MODULE}" >&2
    exit 2
fi

echo "[INFO] Input anat directory:"
ls -lh "${ATLAS_DATA_DIR}/${ATLAS_SUBJECT}/anat" || true
echo

cd "${REPO_DIR}" || exit 2

trace_args=()
if [[ "${ATLAS_TRACE}" == "1" ]]; then
    trace_args=(-x)
fi

ATLAS_BACKEND=native \
RECON_ALL_EXTRA_ARGS="${RECON_ALL_EXTRA_ARGS}" \
bash "${trace_args[@]}" atlas/make_atlas.sh \
    "${ATLAS_DATA_DIR}" \
    "${ATLAS_THREADS_PER_JOB}" \
    "${ATLAS_LICENSE_PATH}" \
    "${ATLAS_SUBJECT}"
status=$?

echo
echo "[INFO] make_atlas.sh exit status: ${status}"
echo "[INFO] Output tree:"
find "${ATLAS_DATA_DIR}/FastSurfer_out" -maxdepth 3 -type f -ls 2>/dev/null || true
echo

if compgen -G "${ATLAS_DATA_DIR}/FastSurfer_out/logs/make_atlas_*.log" >/dev/null; then
    echo "[INFO] Tail of make_atlas logs:"
    tail -n 160 "${ATLAS_DATA_DIR}"/FastSurfer_out/logs/make_atlas_*.log
else
    echo "[WARN] No make_atlas log files found under ${ATLAS_DATA_DIR}/FastSurfer_out/logs"
fi

exit "${status}"
