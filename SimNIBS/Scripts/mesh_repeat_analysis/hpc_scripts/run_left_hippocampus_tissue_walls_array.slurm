#!/bin/bash
#SBATCH --job-name=mesh_lh_tissues
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=1-10%10
#SBATCH --output=mesh_lh_tissues_%A_%a.out
#SBATCH --error=mesh_lh_tissues_%A_%a.err

set -euo pipefail

# Full Left Hippocampus scope: ten repeats, 175 meshes per repeat. Each array
# task writes an independent set of front/back tissue walls and can be resumed
# safely because non-empty tissue tiles are reused.
PIPELINE_DIR_CONFIG="${HOME}/Repos/TI_Pipeline/SimNIBS/Scripts"
MESH_ROOT_CONFIG="/mnt/parscratch/users/cop23bi/ZIPs/Analised-Data/Left_Hippocampus_Runs"
OUTPUT_ROOT_CONFIG="/mnt/parscratch/users/cop23bi/mesh-wall/Left_Hippocampus_tissue_front_back_all_repeats"
LOG_ROOT_CONFIG="/mnt/parscratch/users/cop23bi/mesh-wall/logs/Left_Hippocampus_tissue_front_back_all_repeats"
MESH_QC_PYTHON_CONFIG="${HOME}/.conda/envs/ti-post/bin/python"

EXPECTED_REPEATS_CONFIG="10"
EXPECTED_SUBJECTS_PER_REPEAT_CONFIG="175"
EXPECTED_TOTAL_MESHES_CONFIG="1750"
EXPECTED_TISSUES_CONFIG="9"
EXPECTED_VIEWS_CONFIG="2"
EXPECTED_TILES_PER_REPEAT_CONFIG="3150"
EXPECTED_TOTAL_TILES_CONFIG="31500"
EXPECTED_WALLS_PER_REPEAT_CONFIG="18"
EXPECTED_TOTAL_WALLS_CONFIG="180"

RENDERER_CONFIG="gmsh"
SIMNIBS_MODULE_CONFIG="none"
GMSH_MODULE_CONFIG="gmsh/4.11.1-foss-2022b"
XVFB_MODULE_CONFIG="Xvfb/21.1.6-GCCcore-12.2.0"
MESH_QC_GMSH_TIMEOUT_SECONDS_CONFIG="900"
MESH_QC_STAGE_CONFIG="tissue"
WORKERS_CONFIG="16"
PROGRESS_MODE_CONFIG="text"
PROGRESS_EVERY_CONFIG="1"
DISCOVERY_PROGRESS_SECONDS_CONFIG="5"
IMAGE_SIZE_CONFIG="1200"
TILE_SIZE_CONFIG="220"
COLS_CONFIG="10"

GENERIC_RUNNER="${PIPELINE_DIR_CONFIG}/mesh_repeat_analysis/hpc_scripts/run_mesh_qc.slurm"
REPEAT_INDEX="${SLURM_ARRAY_TASK_ID:-}"

if [[ ! "${REPEAT_INDEX}" =~ ^([1-9]|10)$ ]]; then
    echo "[ERROR] SLURM_ARRAY_TASK_ID must be an integer from 1 to 10; got ${REPEAT_INDEX:-<unset>}." >&2
    exit 2
fi

REPEAT_PADDED="$(printf '%02d' "${REPEAT_INDEX}")"
REPEAT_NAME="Left_Hippocampus_Data_${REPEAT_PADDED}"
MESH_QC_ROOT_CONFIG="${MESH_ROOT_CONFIG}/${REPEAT_NAME}"
MESH_QC_OUT_CONFIG="${OUTPUT_ROOT_CONFIG}/${REPEAT_NAME}"
MESH_QC_LOG_DIR_CONFIG="${LOG_ROOT_CONFIG}/${REPEAT_NAME}"

if [ ! -f "${GENERIC_RUNNER}" ]; then
    echo "[ERROR] Generic mesh-QC Slurm runner does not exist: ${GENERIC_RUNNER}" >&2
    exit 2
fi
if [ ! -d "${PIPELINE_DIR_CONFIG}" ]; then
    echo "[ERROR] Pipeline directory does not exist: ${PIPELINE_DIR_CONFIG}" >&2
    exit 2
fi
if [ ! -d "${MESH_QC_ROOT_CONFIG}" ]; then
    echo "[ERROR] Repeat input directory does not exist: ${MESH_QC_ROOT_CONFIG}" >&2
    exit 2
fi
if [ ! -x "${MESH_QC_PYTHON_CONFIG}" ]; then
    echo "[ERROR] ti-post Python is missing or not executable: ${MESH_QC_PYTHON_CONFIG}" >&2
    exit 2
fi

mkdir -p "${MESH_QC_OUT_CONFIG}" "${MESH_QC_LOG_DIR_CONFIG}"
cd "${PIPELINE_DIR_CONFIG}"
export PYTHONPATH="${PIPELINE_DIR_CONFIG}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1

# Validate the parser before module loading; the generic runner separately
# validates the selected Python again after loading Gmsh and Xvfb.
# shellcheck disable=SC2016
"${MESH_QC_PYTHON_CONFIG}" -c '
import sys
try:
    import meshio
    import numpy
    from PIL import Image
except Exception as exc:
    print(f"[ERROR] ti-post tissue-wall dependency import failed: {exc}", file=sys.stderr)
    print(
        "[ERROR] Install dependencies with: $HOME/.conda/envs/ti-post/bin/python -m pip install "
        "--requirement $HOME/Repos/TI_Pipeline/SimNIBS/Scripts/mesh_repeat_analysis/post/"
        "mesh_qc/requirements-tissue-walls.txt",
        file=sys.stderr,
    )
    raise SystemExit(2)
print(f"[INFO] meshio version: {meshio.__version__}")
'

DISCOVERY_COUNTS="$("${MESH_QC_PYTHON_CONFIG}" -c '
from pathlib import Path
import sys
from mesh_repeat_analysis.post.mesh_qc.discovery import discover_meshes
records = discover_meshes(Path(sys.argv[1]))
print(len(records), len({record.subject for record in records}))
' "${MESH_QC_ROOT_CONFIG}")"
read -r FOUND_MESHES FOUND_SUBJECTS <<< "${DISCOVERY_COUNTS}"

if [ "${FOUND_MESHES}" != "${EXPECTED_SUBJECTS_PER_REPEAT_CONFIG}" ]; then
    echo "[ERROR] Expected ${EXPECTED_SUBJECTS_PER_REPEAT_CONFIG} meshes in ${REPEAT_NAME}, found ${FOUND_MESHES}." >&2
    exit 2
fi
if [ "${FOUND_SUBJECTS}" != "${EXPECTED_SUBJECTS_PER_REPEAT_CONFIG}" ]; then
    echo "[ERROR] Expected ${EXPECTED_SUBJECTS_PER_REPEAT_CONFIG} unique subjects in ${REPEAT_NAME}, found ${FOUND_SUBJECTS}." >&2
    exit 2
fi

echo "[INFO] Scope"
echo "[INFO] Dataset/ROI:          Left Hippocampus"
echo "[INFO] Execution scope:      full"
echo "[INFO] Subjects:             175 unique subjects"
echo "[INFO] Repeats/tasks:        ${EXPECTED_REPEATS_CONFIG}"
echo "[INFO] Slurm array:          1-10%10"
echo "[INFO] Meshes:               ${EXPECTED_TOTAL_MESHES_CONFIG} total; ${EXPECTED_SUBJECTS_PER_REPEAT_CONFIG} in this task"
echo "[INFO] Expected tags/views:  ${EXPECTED_TISSUES_CONFIG} tissues x ${EXPECTED_VIEWS_CONFIG} views per mesh"
echo "[INFO] Expected tissue PNGs: ${EXPECTED_TOTAL_TILES_CONFIG} total; ${EXPECTED_TILES_PER_REPEAT_CONFIG} in this task"
echo "[INFO] Expected walls:       ${EXPECTED_TOTAL_WALLS_CONFIG} total; ${EXPECTED_WALLS_PER_REPEAT_CONFIG} in this task"
echo "[INFO] Current repeat/task:  ${REPEAT_NAME} / ${REPEAT_INDEX}"
echo "[INFO] Whole-mesh renders:   disabled"
echo "[INFO] Workers this task:    ${WORKERS_CONFIG} (memory-aware reduction remains enabled)"
echo "[INFO] Output:               ${MESH_QC_OUT_CONFIG}"
echo "[INFO] Persistent log:       ${MESH_QC_LOG_DIR_CONFIG}"

export PIPELINE_DIR="${PIPELINE_DIR_CONFIG}"
export MESH_QC_ROOT="${MESH_QC_ROOT_CONFIG}"
export MESH_QC_OUT="${MESH_QC_OUT_CONFIG}"
export MESH_QC_LOG_DIR="${MESH_QC_LOG_DIR_CONFIG}"
export MESH_QC_LOG_FILE="${MESH_QC_LOG_DIR_CONFIG}/mesh_qc_${SLURM_ARRAY_JOB_ID:-manual}_${REPEAT_INDEX}.log"
export MESH_QC_PYTHON="${MESH_QC_PYTHON_CONFIG}"
export MESH_QC_STAGE="${MESH_QC_STAGE_CONFIG}"
export RENDERER="${RENDERER_CONFIG}"
export SIMNIBS_MODULE="${SIMNIBS_MODULE_CONFIG}"
export GMSH_MODULE="${GMSH_MODULE_CONFIG}"
export XVFB_MODULE="${XVFB_MODULE_CONFIG}"
export MESH_QC_GMSH_TIMEOUT_SECONDS="${MESH_QC_GMSH_TIMEOUT_SECONDS_CONFIG}"
export CHECK_COMPONENTS="0"
export ROI_WALLS="0"
export TISSUE_WALLS="1"
export WORKERS="${WORKERS_CONFIG}"
export PROGRESS_MODE="${PROGRESS_MODE_CONFIG}"
export PROGRESS_EVERY="${PROGRESS_EVERY_CONFIG}"
export DISCOVERY_PROGRESS_SECONDS="${DISCOVERY_PROGRESS_SECONDS_CONFIG}"
export IMAGE_SIZE="${IMAGE_SIZE_CONFIG}"
export TILE_SIZE="${TILE_SIZE_CONFIG}"
export COLS="${COLS_CONFIG}"

exec bash "${GENERIC_RUNNER}"
