#!/bin/bash
#SBATCH --job-name=mesh_lh_d01_walls
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=mesh_lh_d01_walls_%j.out
#SBATCH --error=mesh_lh_d01_walls_%j.err

set -euo pipefail

# Left Hippocampus Data_01 front/back tissue-wall pilot. These defaults are
# intentionally self-contained so the job needs only one sbatch command.
MESH_QC_ROOT_CONFIG="/mnt/parscratch/users/cop23bi/ZIPs/Analised-Data/Left_Hippocampus_Runs/Left_Hippocampus_Data_01"
MESH_QC_OUT_CONFIG="/mnt/parscratch/users/cop23bi/mesh-wall/Left_Hippocampus_Data_01_front_back_test"
MESH_QC_LOG_DIR_CONFIG="/mnt/parscratch/users/cop23bi/mesh-wall/logs/Left_Hippocampus_Data_01_front_back_test"
MESH_QC_EXPECTED_MESHES_CONFIG="175"
MESH_QC_PYTHON_CONFIG="${HOME}/.conda/envs/ti-post/bin/python"

RENDERER_CONFIG="gmsh"
SIMNIBS_MODULE_CONFIG="none"
GMSH_MODULE_CONFIG="gmsh/4.11.1-foss-2022b"
XVFB_MODULE_CONFIG="Xvfb/21.1.6-GCCcore-12.2.0"
MESH_QC_GMSH_TIMEOUT_SECONDS_CONFIG="900"

MESH_QC_STAGE_CONFIG="full"
CHECK_COMPONENTS_CONFIG="0"
ROI_WALLS_CONFIG="0"
TISSUE_WALLS_CONFIG="1"
WORKERS_CONFIG="1"
PROGRESS_MODE_CONFIG="text"
PROGRESS_EVERY_CONFIG="1"
DISCOVERY_PROGRESS_SECONDS_CONFIG="5"
IMAGE_SIZE_CONFIG="1200"
TILE_SIZE_CONFIG="220"
COLS_CONFIG="10"

# Slurm runs a copied script from /var/spool/slurmd, so BASH_SOURCE cannot be
# used to locate the repository from inside an allocated job.
PIPELINE_DIR_CONFIG="${HOME}/Repos/TI_Pipeline/SimNIBS/Scripts"
GENERIC_RUNNER="${PIPELINE_DIR_CONFIG}/mesh_repeat_analysis/hpc_scripts/run_mesh_qc.slurm"

# Assign the pilot settings unconditionally so stale generic mesh-QC variables
# in the submission shell cannot redirect this job to another dataset.
export PIPELINE_DIR="${PIPELINE_DIR_CONFIG}"
export MESH_QC_ROOT="${MESH_QC_ROOT_CONFIG}"
export MESH_QC_OUT="${MESH_QC_OUT_CONFIG}"
export MESH_QC_LOG_DIR="${MESH_QC_LOG_DIR_CONFIG}"
export MESH_QC_EXPECTED_MESHES="${MESH_QC_EXPECTED_MESHES_CONFIG}"
export MESH_QC_PYTHON="${MESH_QC_PYTHON_CONFIG}"

export RENDERER="${RENDERER_CONFIG}"
export SIMNIBS_MODULE="${SIMNIBS_MODULE_CONFIG}"
export GMSH_MODULE="${GMSH_MODULE_CONFIG}"
export XVFB_MODULE="${XVFB_MODULE_CONFIG}"
export MESH_QC_GMSH_TIMEOUT_SECONDS="${MESH_QC_GMSH_TIMEOUT_SECONDS_CONFIG}"

# A uniquely named override supports an interrupted render-only recovery without
# allowing inherited generic MESH_QC_STAGE values to change a normal pilot run.
export MESH_QC_STAGE="${LEFT_HIPPOCAMPUS_PILOT_STAGE:-${MESH_QC_STAGE_CONFIG}}"
export CHECK_COMPONENTS="${CHECK_COMPONENTS_CONFIG}"
export ROI_WALLS="${ROI_WALLS_CONFIG}"
export TISSUE_WALLS="${TISSUE_WALLS_CONFIG}"
export WORKERS="${WORKERS_CONFIG}"
export PROGRESS_MODE="${PROGRESS_MODE_CONFIG}"
export PROGRESS_EVERY="${PROGRESS_EVERY_CONFIG}"
export DISCOVERY_PROGRESS_SECONDS="${DISCOVERY_PROGRESS_SECONDS_CONFIG}"
export IMAGE_SIZE="${IMAGE_SIZE_CONFIG}"
export TILE_SIZE="${TILE_SIZE_CONFIG}"
export COLS="${COLS_CONFIG}"

if [ ! -f "${GENERIC_RUNNER}" ]; then
    echo "[ERROR] Generic mesh-QC Slurm runner does not exist: ${GENERIC_RUNNER}" >&2
    exit 2
fi
if [ ! -d "${PIPELINE_DIR}" ]; then
    echo "[ERROR] Pipeline directory does not exist: ${PIPELINE_DIR}" >&2
    exit 2
fi
if [ ! -d "${MESH_QC_ROOT}" ]; then
    echo "[ERROR] Left Hippocampus pilot root does not exist: ${MESH_QC_ROOT}" >&2
    exit 2
fi
if [ ! -x "${MESH_QC_PYTHON}" ]; then
    echo "[ERROR] ti-post Python is missing or not executable: ${MESH_QC_PYTHON}" >&2
    exit 2
fi

mkdir -p "${MESH_QC_OUT}" "${MESH_QC_LOG_DIR}"
cd "${PIPELINE_DIR}"
export PYTHONPATH="${PIPELINE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1

"${MESH_QC_PYTHON}" -c '
import importlib.util
import sys

missing = [name for name in ("numpy", "PIL") if importlib.util.find_spec(name) is None]
if importlib.util.find_spec("meshio") is None and importlib.util.find_spec("simnibs") is None:
    missing.append("meshio-or-simnibs")
if missing:
    print("[ERROR] Missing ti-post Python dependencies: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(2)
'

FOUND_MESHES="$("${MESH_QC_PYTHON}" -c '
from pathlib import Path
import sys

from mesh_repeat_analysis.post.mesh_qc.discovery import discover_meshes

print(len(discover_meshes(Path(sys.argv[1]))))
' "${MESH_QC_ROOT}")"

if [ "${FOUND_MESHES}" != "${MESH_QC_EXPECTED_MESHES}" ]; then
    echo "[ERROR] Expected ${MESH_QC_EXPECTED_MESHES} Data_01 meshes, found ${FOUND_MESHES}." >&2
    echo "[ERROR] Correct the pilot root/count before rendering." >&2
    exit 2
fi

echo "[INFO] Left Hippocampus tissue-wall pilot preflight passed"
echo "[INFO] Mesh root:       ${MESH_QC_ROOT}"
echo "[INFO] Meshes found:    ${FOUND_MESHES}"
echo "[INFO] Output root:     ${MESH_QC_OUT}"
echo "[INFO] Persistent logs: ${MESH_QC_LOG_DIR}"
echo "[INFO] Python:          ${MESH_QC_PYTHON}"
echo "[INFO] Stage:           ${MESH_QC_STAGE}"
echo "[INFO] Tissue walls:    ${TISSUE_WALLS}"
echo "[INFO] Workers:         ${WORKERS}"

exec bash "${GENERIC_RUNNER}"
