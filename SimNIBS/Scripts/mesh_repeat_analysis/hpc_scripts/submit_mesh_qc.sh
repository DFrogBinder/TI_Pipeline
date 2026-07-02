#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Mesh QC submission controls
#
# Edit this block once on the HPC, then submit with:
#
#   bash hpc_scripts/submit_mesh_qc.sh
#
# Environment variables with the same runtime names still override these values
# for one-off submissions, but normal use should only require this block.
# ---------------------------------------------------------------------------
PIPELINE_DIR_CONFIG="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MESH_QC_ROOT_CONFIG="/mnt/parscratch/users/cop23bi/ZIPs/Analised-Data"
MESH_QC_OUT_CONFIG="/mnt/parscratch/users/cop23bi/mesh-wall"

JOB_NAME_CONFIG="mesh_qc"
PARTITION_CONFIG="sheffield"
CPUS_PER_TASK_CONFIG="8"
MEMORY_CONFIG="32G"
TIME_LIMIT_CONFIG="08:00:00"
SLURM_OUTPUT_CONFIG=""

MESH_GLOB_CONFIG=""
RENDERER_CONFIG="auto"
CHECK_COMPONENTS_CONFIG="0"
ROI_WALLS_CONFIG="0"
PROGRESS_CONFIG="text"
WORKERS_CONFIG="0"
MESH_QC_STAGE_CONFIG="full"
IMAGE_SIZE_CONFIG="1200"
TILE_SIZE_CONFIG="220"
COLS_CONFIG=""
DISCOVERY_PROGRESS_SECONDS_CONFIG="5"
PROGRESS_EVERY_CONFIG="25"

SBATCH_BIN_CONFIG="sbatch"
SLURM_SCRIPT_CONFIG="${PIPELINE_DIR_CONFIG}/mesh_repeat_analysis/hpc_scripts/run_mesh_qc.slurm"
LOG_DIR_CONFIG="${PIPELINE_DIR_CONFIG}/logs"

PIPELINE_DIR="${PIPELINE_DIR:-${PIPELINE_DIR_CONFIG}}"
MESH_QC_ROOT="${MESH_QC_ROOT:-${MESH_QC_ROOT_CONFIG}}"
MESH_QC_OUT="${MESH_QC_OUT:-${MESH_QC_OUT_CONFIG}}"
JOB_NAME="${JOB_NAME:-${JOB_NAME_CONFIG}}"
PARTITION="${PARTITION:-${PARTITION_CONFIG}}"
CPUS_PER_TASK="${CPUS_PER_TASK:-${CPUS_PER_TASK_CONFIG}}"
MEMORY="${MEMORY:-${MEMORY_CONFIG}}"
TIME_LIMIT="${TIME_LIMIT:-${TIME_LIMIT_CONFIG}}"
SLURM_OUTPUT="${SLURM_OUTPUT:-${SLURM_OUTPUT_CONFIG}}"
MESH_GLOB="${MESH_GLOB:-${MESH_GLOB_CONFIG}}"
RENDERER="${RENDERER:-${RENDERER_CONFIG}}"
CHECK_COMPONENTS="${CHECK_COMPONENTS:-${CHECK_COMPONENTS_CONFIG}}"
ROI_WALLS="${ROI_WALLS:-${ROI_WALLS_CONFIG}}"
PROGRESS_MODE="${PROGRESS_MODE:-${PROGRESS_CONFIG}}"
WORKERS="${WORKERS:-${WORKERS_CONFIG}}"
MESH_QC_STAGE="${MESH_QC_STAGE:-${MESH_QC_STAGE_CONFIG}}"
IMAGE_SIZE="${IMAGE_SIZE:-${IMAGE_SIZE_CONFIG}}"
TILE_SIZE="${TILE_SIZE:-${TILE_SIZE_CONFIG}}"
COLS="${COLS:-${COLS_CONFIG}}"
DISCOVERY_PROGRESS_SECONDS="${DISCOVERY_PROGRESS_SECONDS:-${DISCOVERY_PROGRESS_SECONDS_CONFIG}}"
PROGRESS_EVERY="${PROGRESS_EVERY:-${PROGRESS_EVERY_CONFIG}}"
SBATCH_BIN="${SBATCH_BIN:-${SBATCH_BIN_CONFIG}}"
SLURM_SCRIPT="${SLURM_SCRIPT:-${SLURM_SCRIPT_CONFIG}}"
MESH_QC_LOG_DIR="${MESH_QC_LOG_DIR:-${LOG_DIR_CONFIG}}"

if [ "$#" -ne 0 ]; then
    echo "[ERROR] This helper is configured from the controls at the top of the file."
    echo "        Edit the *_CONFIG values there, then run:"
    echo "        bash mesh_repeat_analysis/hpc_scripts/submit_mesh_qc.sh"
    exit 1
fi

resolve_path() {
    python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1"
}

PIPELINE_DIR="$(resolve_path "${PIPELINE_DIR}")"
MESH_QC_ROOT="$(resolve_path "${MESH_QC_ROOT}")"
MESH_QC_OUT="$(resolve_path "${MESH_QC_OUT}")"
SLURM_SCRIPT="$(resolve_path "${SLURM_SCRIPT}")"
MESH_QC_LOG_DIR="$(resolve_path "${MESH_QC_LOG_DIR}")"

if [ ! -d "${PIPELINE_DIR}" ]; then
    echo "[ERROR] Pipeline directory does not exist: ${PIPELINE_DIR}"
    exit 1
fi
if [ ! -d "${MESH_QC_ROOT}" ]; then
    echo "[ERROR] Mesh QC root does not exist: ${MESH_QC_ROOT}"
    exit 1
fi
if [ ! -f "${SLURM_SCRIPT}" ]; then
    echo "[ERROR] Slurm script does not exist: ${SLURM_SCRIPT}"
    exit 1
fi

mkdir -p "${MESH_QC_LOG_DIR}" "${MESH_QC_OUT}"
if [ -z "${SLURM_OUTPUT}" ]; then
    SLURM_OUTPUT="${MESH_QC_LOG_DIR}/slurm-%j.out"
fi

EXPORT_VARS="ALL"
EXPORT_VARS+=",PIPELINE_DIR=${PIPELINE_DIR}"
EXPORT_VARS+=",MESH_QC_ROOT=${MESH_QC_ROOT}"
EXPORT_VARS+=",MESH_QC_OUT=${MESH_QC_OUT}"
EXPORT_VARS+=",MESH_GLOB=${MESH_GLOB}"
EXPORT_VARS+=",RENDERER=${RENDERER}"
EXPORT_VARS+=",CHECK_COMPONENTS=${CHECK_COMPONENTS}"
EXPORT_VARS+=",ROI_WALLS=${ROI_WALLS}"
EXPORT_VARS+=",PROGRESS_MODE=${PROGRESS_MODE}"
EXPORT_VARS+=",WORKERS=${WORKERS}"
EXPORT_VARS+=",MESH_QC_STAGE=${MESH_QC_STAGE}"
EXPORT_VARS+=",IMAGE_SIZE=${IMAGE_SIZE}"
EXPORT_VARS+=",TILE_SIZE=${TILE_SIZE}"
EXPORT_VARS+=",COLS=${COLS}"
EXPORT_VARS+=",DISCOVERY_PROGRESS_SECONDS=${DISCOVERY_PROGRESS_SECONDS}"
EXPORT_VARS+=",PROGRESS_EVERY=${PROGRESS_EVERY}"
EXPORT_VARS+=",MESH_QC_LOG_DIR=${MESH_QC_LOG_DIR}"
EXPORT_VARS+=",LOG_DIR=${MESH_QC_LOG_DIR}"

echo "[INFO] Pipeline root:  ${PIPELINE_DIR}"
echo "[INFO] Mesh root:      ${MESH_QC_ROOT}"
echo "[INFO] Output dir:     ${MESH_QC_OUT}"
echo "[INFO] Slurm script:   ${SLURM_SCRIPT}"
echo "[INFO] Log dir:        ${MESH_QC_LOG_DIR}"
echo "[INFO] Stage:          ${MESH_QC_STAGE}"
echo "[INFO] Renderer:       ${RENDERER}"
echo "[INFO] Workers:        ${WORKERS}"
echo "[INFO] CPUs per task:  ${CPUS_PER_TASK}"
echo "[INFO] Memory:         ${MEMORY}"
echo "[INFO] Time limit:     ${TIME_LIMIT}"
echo "[INFO] Slurm output:   ${SLURM_OUTPUT}"

SBATCH_CMD=(
    "${SBATCH_BIN}"
    --job-name="${JOB_NAME}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${MEMORY}"
    --time="${TIME_LIMIT}"
    --export="${EXPORT_VARS}"
)
if [ -n "${PARTITION}" ]; then
    SBATCH_CMD+=(--partition="${PARTITION}")
fi
if [ -n "${SLURM_OUTPUT}" ]; then
    SBATCH_CMD+=(--output="${SLURM_OUTPUT}")
fi
SBATCH_CMD+=("${SLURM_SCRIPT}")

"${SBATCH_CMD[@]}"
