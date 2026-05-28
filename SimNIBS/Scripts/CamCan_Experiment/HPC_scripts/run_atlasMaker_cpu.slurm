#!/bin/bash
#SBATCH --job-name=atlas_cpu
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=27
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=atlas_cpu_%j.out
#SBATCH --error=atlas_cpu_%j.err

# Replace the partition, CPU count, memory, and walltime for your HPC queue.
# Defaults run three subjects concurrently, with nine CPU threads per subject.

set -euo pipefail

echo "[INFO] Host: $(hostname)"
echo "[INFO] JobID: ${SLURM_JOB_ID:-<none>}"
echo "[INFO] Start: $(date)"
echo

# ------------------------------------------------------------------
# User config
#
# Submit with:
#   sbatch HPC_scripts/run_atlasMaker_cpu.slurm
#
# Or override paths without editing the file:
#   sbatch --export=ALL,ATLAS_DATA_DIR=/path/to/CamCan_Data,ATLAS_LICENSE_PATH=/path/to/license.txt HPC_scripts/run_atlasMaker_cpu.slurm
#
# Native FreeSurfer mode is the default for CPU jobs. Use ATLAS_BACKEND=docker
# only on clusters where Docker is available on compute nodes.
#
# For one subject at a time:
#   sbatch --export=ALL,ATLAS_MAX_PARALLEL_JOBS=1,ATLAS_THREADS_PER_JOB=9 HPC_scripts/run_atlasMaker_cpu.slurm
# ------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO_DIR:-}" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/atlas/run_atlasMaker.py" ]]; then
        REPO_DIR="${SLURM_SUBMIT_DIR}"
    else
        REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
    fi
fi

ATLAS_DATA_DIR="${ATLAS_DATA_DIR:-/home/uos/Boyan/CamCan_Data}"
ATLAS_LICENSE_PATH="${ATLAS_LICENSE_PATH:-/home/uos/Boyan/freesurfer_licence.txt}"
ATLAS_THREADS_PER_JOB="${ATLAS_THREADS_PER_JOB:-9}"
ATLAS_MAX_PARALLEL_JOBS="${ATLAS_MAX_PARALLEL_JOBS:-3}"
ATLAS_SUBJECTS="${ATLAS_SUBJECTS:-}"          # Optional comma/space-separated subject IDs
ATLAS_SUBJECTS_FILE="${ATLAS_SUBJECTS_FILE:-}" # Optional text file of subject IDs
ATLAS_BACKEND="${ATLAS_BACKEND:-native}"      # native or docker
RECON_ALL_EXTRA_ARGS="${RECON_ALL_EXTRA_ARGS:--cw256}"

PYTHON="${PYTHON:-python3}"

# Optional module list, for example:
#   ATLAS_MODULES="Docker/24.0.5"
ATLAS_MODULES="${ATLAS_MODULES:-}"
ATLAS_PURGE_MODULES="${ATLAS_PURGE_MODULES:-0}"
FREESURFER_MODULE="${FREESURFER_MODULE:-FreeSurfer/7.4.1-centos7_x86_64}"

# CPU-only FastSurfer settings consumed by atlas/make_atlas.sh.
# Force-clear GPU-related variables in case they were inherited from the shell.
export FASTSURFER_USE_GPU=0
export FASTSURFER_DEVICE=""
export FASTSURFER_DOCKER_GPUS=""
export FASTSURFER_DOCKER_FLAGS=""
export FASTSURFER_IMAGE="${FASTSURFER_IMAGE:-deepmi/fastsurfer:latest}"
export FREESURFER_IMAGE="${FREESURFER_IMAGE:-freesurfer/freesurfer:7.4.1}"

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
    if [[ "${ATLAS_PURGE_MODULES}" == "1" ]]; then
        module purge
    fi
    for module_name in ${ATLAS_MODULES}; do
        module load "${module_name}"
    done
    if [[ "${ATLAS_BACKEND}" == "native" && -n "${FREESURFER_MODULE}" ]]; then
        module load "${FREESURFER_MODULE}"
    fi
fi

export ATLAS_DATA_DIR
export ATLAS_LICENSE_PATH
export ATLAS_THREADS_PER_JOB
export ATLAS_MAX_PARALLEL_JOBS
export ATLAS_SUBJECTS
export ATLAS_BACKEND
export RECON_ALL_EXTRA_ARGS
if [[ -n "${ATLAS_SUBJECTS_FILE}" ]]; then
    export ATLAS_SUBJECTS_FILE
fi

export OMP_NUM_THREADS="${ATLAS_THREADS_PER_JOB}"
export SLURM_EXPORT_ENV=ALL

RUNNER="${REPO_DIR}/atlas/run_atlasMaker.py"

echo "[INFO] Repository:       ${REPO_DIR}"
echo "[INFO] Runner:           ${RUNNER}"
echo "[INFO] Data dir:         ${ATLAS_DATA_DIR}"
echo "[INFO] License:          ${ATLAS_LICENSE_PATH}"
echo "[INFO] Threads/job:      ${ATLAS_THREADS_PER_JOB}"
echo "[INFO] Parallel jobs:    ${ATLAS_MAX_PARALLEL_JOBS}"
echo "[INFO] Slurm CPUs:       ${SLURM_CPUS_PER_TASK:-<not set>}"
echo "[INFO] Subjects:         ${ATLAS_SUBJECTS:-<discover all>}"
echo "[INFO] Subjects file:    ${ATLAS_SUBJECTS_FILE:-<none>}"
echo "[INFO] Atlas backend:    ${ATLAS_BACKEND}"
echo "[INFO] FreeSurfer mod:   ${FREESURFER_MODULE:-<none>}"
echo "[INFO] Recon args:       ${RECON_ALL_EXTRA_ARGS:-<none>}"
echo "[INFO] FastSurfer image: ${FASTSURFER_IMAGE}"
echo "[INFO] FastSurfer mode:  CPU only when backend=docker"
echo

if [[ ! -f "${RUNNER}" ]]; then
    echo "[ERROR] Runner not found: ${RUNNER}" >&2
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

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "[ERROR] Python executable not found: ${PYTHON}" >&2
    exit 2
fi

case "${ATLAS_BACKEND}" in
    docker)
        if ! command -v docker >/dev/null 2>&1; then
            echo "[ERROR] docker was not found in PATH." >&2
            echo "[ERROR] Set ATLAS_MODULES to load your HPC Docker module, or use ATLAS_BACKEND=native with a FreeSurfer module." >&2
            exit 2
        fi
        ;;
    native)
        if ! command -v recon-all >/dev/null 2>&1; then
            echo "[ERROR] recon-all was not found in PATH after loading ${FREESURFER_MODULE:-<none>}." >&2
            exit 2
        fi
        if ! command -v mri_convert >/dev/null 2>&1; then
            echo "[ERROR] mri_convert was not found in PATH after loading ${FREESURFER_MODULE:-<none>}." >&2
            exit 2
        fi
        ;;
    *)
        echo "[ERROR] ATLAS_BACKEND must be 'native' or 'docker'. Received: ${ATLAS_BACKEND}" >&2
        exit 2
        ;;
esac

cd "${REPO_DIR}"
exec "${PYTHON}" "${RUNNER}"
