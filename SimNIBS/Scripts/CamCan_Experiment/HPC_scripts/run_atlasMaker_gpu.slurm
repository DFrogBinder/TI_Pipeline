#!/bin/bash
#SBATCH --job-name=atlas_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=atlas_gpu_%j.out
#SBATCH --error=atlas_gpu_%j.err

# Replace the partition with your HPC GPU partition if different.
# Some clusters use "#SBATCH --gpus-per-node=1" instead of "--gres=gpu:1".

set -euo pipefail

echo "[INFO] Host: $(hostname)"
echo "[INFO] JobID: ${SLURM_JOB_ID:-<none>}"
echo "[INFO] Start: $(date)"
echo

# ------------------------------------------------------------------
# User config
#
# Submit with:
#   sbatch HPC_scripts/run_atlasMaker_gpu.slurm
#
# Or override paths without editing the file:
#   sbatch --export=ALL,ATLAS_DATA_DIR=/path/to/CamCan_Data,ATLAS_LICENSE_PATH=/path/to/license.txt HPC_scripts/run_atlasMaker_gpu.slurm
#
# Keep ATLAS_MAX_PARALLEL_JOBS=1 for one allocated GPU. Increase only if
# requesting enough GPUs and using container flags that pin each process.
# ------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

ATLAS_DATA_DIR="${ATLAS_DATA_DIR:-/home/uos/Boyan/CamCan_Data}"
ATLAS_LICENSE_PATH="${ATLAS_LICENSE_PATH:-/home/uos/Boyan/freesurfer_licence.txt}"
ATLAS_THREADS_PER_JOB="${ATLAS_THREADS_PER_JOB:-${SLURM_CPUS_PER_TASK:-9}}"
ATLAS_MAX_PARALLEL_JOBS="${ATLAS_MAX_PARALLEL_JOBS:-1}"
ATLAS_SUBJECTS="${ATLAS_SUBJECTS:-}"          # Optional comma/space-separated subject IDs
ATLAS_SUBJECTS_FILE="${ATLAS_SUBJECTS_FILE:-}" # Optional text file of subject IDs

PYTHON="${PYTHON:-python3}"

# Optional module list, for example:
#   ATLAS_MODULES="CUDA/12.2 Docker/24.0.5"
ATLAS_MODULES="${ATLAS_MODULES:-}"
ATLAS_PURGE_MODULES="${ATLAS_PURGE_MODULES:-0}"

# Docker/FastSurfer GPU settings consumed by atlas/make_atlas.sh.
export FASTSURFER_USE_GPU="${FASTSURFER_USE_GPU:-1}"
export FASTSURFER_DEVICE="${FASTSURFER_DEVICE:-cuda}"
export FASTSURFER_DOCKER_GPUS="${FASTSURFER_DOCKER_GPUS:-all}"
export FASTSURFER_IMAGE="${FASTSURFER_IMAGE:-deepmi/fastsurfer:latest}"
export FREESURFER_IMAGE="${FREESURFER_IMAGE:-freesurfer/freesurfer:7.4.1}"

if type module >/dev/null 2>&1; then
    if [[ "${ATLAS_PURGE_MODULES}" == "1" ]]; then
        module purge
    fi
    for module_name in ${ATLAS_MODULES}; do
        module load "${module_name}"
    done
fi

export ATLAS_DATA_DIR
export ATLAS_LICENSE_PATH
export ATLAS_THREADS_PER_JOB
export ATLAS_MAX_PARALLEL_JOBS
export ATLAS_SUBJECTS
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
echo "[INFO] Subjects:         ${ATLAS_SUBJECTS:-<discover all>}"
echo "[INFO] Subjects file:    ${ATLAS_SUBJECTS_FILE:-<none>}"
echo "[INFO] FastSurfer image: ${FASTSURFER_IMAGE}"
echo "[INFO] FastSurfer GPU:   use_gpu=${FASTSURFER_USE_GPU} device=${FASTSURFER_DEVICE} docker_gpus=${FASTSURFER_DOCKER_GPUS}"
echo "[INFO] CUDA devices:     ${CUDA_VISIBLE_DEVICES:-<not set>}"
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

if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] docker was not found in PATH." >&2
    echo "[ERROR] Set ATLAS_MODULES to load your HPC Docker module, or adapt atlas/make_atlas.sh for Apptainer/Singularity." >&2
    exit 2
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "[WARN] nvidia-smi not found; continuing because Slurm allocated the GPU resource."
fi
echo

cd "${REPO_DIR}"
exec "${PYTHON}" "${RUNNER}"
