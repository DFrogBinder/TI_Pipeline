#!/bin/bash
#SBATCH --job-name=mesh_qc
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=mesh_qc_%j.out
#SBATCH --error=mesh_qc_%j.err

set -euo pipefail

SIMNIBS_MODULE_CONFIG="SimNIBS/4.0.1-foss-2023a"
XVFB_MODULE_CONFIG="Xvfb/21.1.6-GCCcore-12.2.0"
GMSH_MODULE_CONFIG=""
MESH_QC_GMSH_BIN_CONFIG=""
IMAGEMAGICK_MODULE_CONFIG=""
MESH_QC_PYTHON_CONFIG="python"

export PYTHONNOUSERSITE=1

# The QC/render code parallelizes across processes. Keep each worker effectively
# single-threaded so BLAS/OpenMP does not multiply CPU usage inside the pool.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [ -n "${LOG_DIR:-}" ] || [ -n "${LOG_FILE:-}" ]; then
    echo "[INFO] Ignoring inherited LOG_DIR/LOG_FILE for mesh QC; use MESH_QC_LOG_DIR/MESH_QC_LOG_FILE instead."
fi

# ---------------------------------------------------------------------------
# Mesh QC batch settings
#
# Edit this block once on the HPC, then submit with:
#
#   sbatch hpc_scripts/run_mesh_qc.slurm
#
# Blank optional values are ignored. Environment variables with the same names
# still override these defaults for one-off runs.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR_CONFIG="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MESH_QC_ROOT_CONFIG="/mnt/parscratch/users/cop23bi/ZIPs/Analised-Data"
MESH_QC_OUT_CONFIG="/mnt/parscratch/users/cop23bi/mesh-wall"
MESH_GLOB_CONFIG=""
ROI_REGEX_CONFIG=""
SUBJECT_REGEX_CONFIG=""
REPEAT_REGEX_CONFIG=""
RENDERER_CONFIG="gmsh"
CHECK_COMPONENTS_CONFIG="0"
ROI_WALLS_CONFIG="0"
TISSUE_WALLS_CONFIG="0"
PROGRESS_CONFIG="tqdm"
WORKERS_CONFIG="0"
MESH_QC_STAGE_CONFIG="full"
IMAGE_SIZE_CONFIG="1200"
TILE_SIZE_CONFIG="220"
COLS_CONFIG=""
DISCOVERY_PROGRESS_SECONDS_CONFIG="5"
PROGRESS_EVERY_CONFIG="25"
MESH_QC_GMSH_TIMEOUT_SECONDS_CONFIG="900"
MESH_QC_EXPECTED_MESHES_CONFIG=""
MESH_QC_EXPECTED_SUBJECTS_CONFIG=""

# Logs. Leave these blank to use <pipeline>/logs.
MESH_QC_LOG_DIR_CONFIG=""
MESH_QC_LOG_FILE_CONFIG=""

PIPELINE_DIR="${PIPELINE_DIR:-${PIPELINE_DIR_CONFIG}}"
MESH_QC_ROOT="${MESH_QC_ROOT:-${MESH_QC_ROOT_CONFIG}}"
MESH_QC_OUT="${MESH_QC_OUT:-${MESH_QC_OUT_CONFIG}}"
MESH_GLOB="${MESH_GLOB:-${MESH_GLOB_CONFIG}}"
ROI_REGEX="${ROI_REGEX:-${ROI_REGEX_CONFIG}}"
SUBJECT_REGEX="${SUBJECT_REGEX:-${SUBJECT_REGEX_CONFIG}}"
REPEAT_REGEX="${REPEAT_REGEX:-${REPEAT_REGEX_CONFIG}}"
RENDERER="${RENDERER:-${RENDERER_CONFIG}}"
CHECK_COMPONENTS="${CHECK_COMPONENTS:-${CHECK_COMPONENTS_CONFIG}}"
ROI_WALLS="${ROI_WALLS:-${ROI_WALLS_CONFIG}}"
TISSUE_WALLS="${TISSUE_WALLS:-${TISSUE_WALLS_CONFIG}}"
PROGRESS_MODE="${PROGRESS_MODE:-${PROGRESS_CONFIG}}"
WORKERS="${WORKERS:-${WORKERS_CONFIG}}"
MESH_QC_STAGE="${MESH_QC_STAGE:-${MESH_QC_STAGE_CONFIG}}"
IMAGE_SIZE="${IMAGE_SIZE:-${IMAGE_SIZE_CONFIG}}"
TILE_SIZE="${TILE_SIZE:-${TILE_SIZE_CONFIG}}"
COLS="${COLS:-${COLS_CONFIG}}"
DISCOVERY_PROGRESS_SECONDS="${DISCOVERY_PROGRESS_SECONDS:-${DISCOVERY_PROGRESS_SECONDS_CONFIG}}"
PROGRESS_EVERY="${PROGRESS_EVERY:-${PROGRESS_EVERY_CONFIG}}"
MESH_QC_GMSH_TIMEOUT_SECONDS="${MESH_QC_GMSH_TIMEOUT_SECONDS:-${MESH_QC_GMSH_TIMEOUT_SECONDS_CONFIG}}"
MESH_QC_EXPECTED_MESHES="${MESH_QC_EXPECTED_MESHES:-${MESH_QC_EXPECTED_MESHES_CONFIG}}"
MESH_QC_EXPECTED_SUBJECTS="${MESH_QC_EXPECTED_SUBJECTS:-${MESH_QC_EXPECTED_SUBJECTS_CONFIG}}"
SIMNIBS_MODULE="${SIMNIBS_MODULE:-${SIMNIBS_MODULE_CONFIG}}"
XVFB_MODULE="${XVFB_MODULE:-${XVFB_MODULE_CONFIG}}"
GMSH_MODULE="${GMSH_MODULE:-${GMSH_MODULE_CONFIG}}"
MESH_QC_GMSH_BIN="${MESH_QC_GMSH_BIN:-${MESH_QC_GMSH_BIN_CONFIG}}"
IMAGEMAGICK_MODULE="${IMAGEMAGICK_MODULE:-${IMAGEMAGICK_MODULE_CONFIG}}"
MESH_QC_PYTHON="${MESH_QC_PYTHON:-${MESH_QC_PYTHON_CONFIG}}"

if ! [[ "${MESH_QC_GMSH_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] MESH_QC_GMSH_TIMEOUT_SECONDS must be a positive integer; got ${MESH_QC_GMSH_TIMEOUT_SECONDS}." >&2
    exit 2
fi
export MESH_QC_GMSH_TIMEOUT_SECONDS

SIMNIBS_MODULE_EFFECTIVE="${SIMNIBS_MODULE}"
GMSH_MODULE_EFFECTIVE="${GMSH_MODULE}"
XVFB_MODULE_EFFECTIVE="${XVFB_MODULE}"
if [ "${MESH_QC_STAGE}" = "render" ] && [ "${RENDERER}" = "gmsh" ] && [ "${MESH_QC_SKIP_SIMNIBS_FOR_RENDER:-0}" = "1" ]; then
    # Preserve the historical default: SimNIBS may be the only provider of
    # gmsh on Stanage. Skip it only when a separate gmsh source is configured.
    SIMNIBS_MODULE_EFFECTIVE=""
fi
if [ "${MESH_QC_STAGE}" = "qc" ]; then
    GMSH_MODULE_EFFECTIVE=""
    XVFB_MODULE_EFFECTIVE=""
fi
case "${SIMNIBS_MODULE}" in
    none|NONE|None|0|false|False|FALSE)
        SIMNIBS_MODULE_EFFECTIVE=""
        ;;
esac
case "${GMSH_MODULE}" in
    none|NONE|None|0|false|False|FALSE)
        GMSH_MODULE_EFFECTIVE=""
        ;;
esac
case "${XVFB_MODULE}" in
    none|NONE|None|0|false|False|FALSE)
        XVFB_MODULE_EFFECTIVE=""
        ;;
esac

module purge
module use "$HOME/modules"
if [ -n "${SIMNIBS_MODULE_EFFECTIVE}" ]; then
    module load "${SIMNIBS_MODULE_EFFECTIVE}"
fi
if [ -n "${GMSH_MODULE_EFFECTIVE}" ]; then
    module load "${GMSH_MODULE_EFFECTIVE}"
fi
if [ -n "${XVFB_MODULE_EFFECTIVE}" ]; then
    module load "${XVFB_MODULE_EFFECTIVE}"
fi
if [ -n "${IMAGEMAGICK_MODULE}" ]; then
    module load "${IMAGEMAGICK_MODULE}"
fi

if [ -z "${PIPELINE_DIR}" ] || [ ! -d "${PIPELINE_DIR}" ]; then
    echo "[ERROR] Pipeline directory does not exist: ${PIPELINE_DIR}" >&2
    exit 1
fi
if [ -z "${MESH_QC_ROOT}" ] || [ ! -d "${MESH_QC_ROOT}" ]; then
    echo "[ERROR] Mesh QC root does not exist: ${MESH_QC_ROOT}" >&2
    exit 1
fi
if [ -z "${MESH_QC_OUT}" ]; then
    echo "[ERROR] Mesh QC output directory is empty." >&2
    exit 1
fi

cd "${PIPELINE_DIR}"

MESH_QC_LOG_DIR="${MESH_QC_LOG_DIR:-${MESH_QC_LOG_DIR_CONFIG:-${PIPELINE_DIR}/logs}}"
mkdir -p "${MESH_QC_LOG_DIR}"
JOB_ID="${SLURM_JOB_ID:-manual}"
MESH_QC_LOG_FILE="${MESH_QC_LOG_FILE:-${MESH_QC_LOG_FILE_CONFIG:-${MESH_QC_LOG_DIR}/mesh_qc_${JOB_ID}.log}}"

echo "[INFO] Mesh QC batch run"
echo "[INFO] Job ID:         ${JOB_ID}"
echo "[INFO] Host:           $(hostname)"
echo "[INFO] Submit dir:     ${SLURM_SUBMIT_DIR:-n/a}"
echo "[INFO] Pipeline root:  ${PIPELINE_DIR}"
echo "[INFO] Mesh root:      ${MESH_QC_ROOT}"
echo "[INFO] Output dir:     ${MESH_QC_OUT}"
echo "[INFO] CPUs alloc:     ${SLURM_CPUS_PER_TASK:-unknown}"
echo "[INFO] Workers:        ${WORKERS}"
echo "[INFO] Stage:          ${MESH_QC_STAGE}"
echo "[INFO] Renderer:       ${RENDERER}"
echo "[INFO] Progress mode:  ${PROGRESS_MODE}"
echo "[INFO] ROI regex:      ${ROI_REGEX:-<none>}"
echo "[INFO] Subject regex:  ${SUBJECT_REGEX:-<none>}"
echo "[INFO] Repeat regex:   ${REPEAT_REGEX:-<none>}"
echo "[INFO] Check comps:    ${CHECK_COMPONENTS}"
echo "[INFO] ROI walls:      ${ROI_WALLS}"
echo "[INFO] Tissue walls:   ${TISSUE_WALLS}"
echo "[INFO] Gmsh timeout:   ${MESH_QC_GMSH_TIMEOUT_SECONDS}s per render"
echo "[INFO] Expected meshes: ${MESH_QC_EXPECTED_MESHES:-<not enforced>}"
echo "[INFO] Expected subjects: ${MESH_QC_EXPECTED_SUBJECTS:-<not enforced>}"
echo "[INFO] Log file:       ${MESH_QC_LOG_FILE}"
echo "[INFO] SimNIBS module: ${SIMNIBS_MODULE_EFFECTIVE:-<none>}"
echo "[INFO] SimNIBS request: ${SIMNIBS_MODULE:-<none>}"
echo "[INFO] Gmsh module:    ${GMSH_MODULE_EFFECTIVE:-<none>}"
echo "[INFO] Gmsh request:   ${GMSH_MODULE:-<none>}"
echo "[INFO] Xvfb module:    ${XVFB_MODULE_EFFECTIVE:-<none>}"
echo "[INFO] Xvfb request:   ${XVFB_MODULE:-<none>}"
echo "[INFO] ImageMagick module: ${IMAGEMAGICK_MODULE:-<none>}"
echo "[INFO] Python request:  ${MESH_QC_PYTHON}"
echo "[INFO] Python path:     $(command -v "${MESH_QC_PYTHON}" 2>/dev/null || echo '<not found>')"
if [ -n "${MESH_QC_GMSH_BIN}" ]; then
    export MESH_QC_GMSH_BIN
fi
NEEDS_GMSH_CHECK=0
if [ "${MESH_QC_STAGE}" != "qc" ] && { [ "${RENDERER}" = "gmsh" ] || [ "${RENDERER}" = "auto" ]; }; then
    NEEDS_GMSH_CHECK=1
fi
GMSH_CHECK_BIN=""
if [ "${NEEDS_GMSH_CHECK}" = "1" ]; then
    GMSH_CHECK_BIN="${MESH_QC_GMSH_BIN:-$(command -v gmsh 2>/dev/null || true)}"
fi
echo "[INFO] Gmsh override:  ${MESH_QC_GMSH_BIN:-<none>}"
echo "[INFO] Gmsh path:      ${GMSH_CHECK_BIN:-<not checked>}"
echo "[INFO] Xvfb path:      $(which Xvfb 2>/dev/null || echo '<not found>')"
echo "[INFO] xvfb-run path:  $(which xvfb-run 2>/dev/null || echo '<not found>')"
echo "[INFO] magick path:    $(which magick 2>/dev/null || echo '<not found>')"
echo "[INFO] montage path:   $(which montage 2>/dev/null || echo '<not found>')"

if [ "${NEEDS_GMSH_CHECK}" = "1" ]; then
    if [ -z "${GMSH_CHECK_BIN}" ]; then
        echo "[ERROR] No gmsh executable found. Load a compatible GMSH_MODULE or set MESH_QC_GMSH_BIN." >&2
        exit 2
    fi
    if ! "${GMSH_CHECK_BIN}" -version >/dev/null 2>&1; then
        echo "[ERROR] gmsh exists but cannot run: ${GMSH_CHECK_BIN}" >&2
        "${GMSH_CHECK_BIN}" -version >&2 || true
        echo "[ERROR] Load a compatible GMSH_MODULE or set MESH_QC_GMSH_BIN to a compatible binary." >&2
        exit 2
    fi
fi
if [ "${NEEDS_GMSH_CHECK}" = "1" ] && [ "${RENDERER}" = "gmsh" ] && [ -z "${DISPLAY:-}" ] && ! command -v Xvfb >/dev/null 2>&1; then
    echo "[ERROR] Forced Gmsh rendering requires Xvfb when DISPLAY is unset." >&2
    echo "[ERROR] Load a working XVFB_MODULE or set DISPLAY and MESH_QC_USE_EXISTING_DISPLAY=1." >&2
    exit 2
fi
if ! command -v "${MESH_QC_PYTHON}" >/dev/null 2>&1; then
    echo "[ERROR] Python executable not found: ${MESH_QC_PYTHON}" >&2
    exit 2
fi
if ! "${MESH_QC_PYTHON}" -E -c 'import numpy; from PIL import Image' >/dev/null 2>&1; then
    echo "[ERROR] The selected Python cannot import its own NumPy/Pillow after module loading." >&2
    echo "[ERROR] Python: ${MESH_QC_PYTHON}" >&2
    echo "[ERROR] Module-provided Python paths must not contaminate the selected environment." >&2
    "${MESH_QC_PYTHON}" -E -c 'import sys; print("sys.path:", *sys.path, sep="\n  "); import numpy; from PIL import Image' >&2 || true
    exit 2
fi
echo "[INFO] Isolated Python imports: OK"

for expected_name in MESH_QC_EXPECTED_MESHES MESH_QC_EXPECTED_SUBJECTS; do
    expected_value="${!expected_name}"
    if [ -n "${expected_value}" ] && ! [[ "${expected_value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] ${expected_name} must be blank or a positive integer; got ${expected_value}." >&2
        exit 2
    fi
done
if [ -n "${MESH_QC_EXPECTED_MESHES}" ] || [ -n "${MESH_QC_EXPECTED_SUBJECTS}" ]; then
    DISCOVERY_COUNTS="$("${MESH_QC_PYTHON}" -E -c '
from pathlib import Path
import sys
from mesh_repeat_analysis.post.mesh_qc.discovery import discover_meshes
records = discover_meshes(
    Path(sys.argv[1]),
    mesh_glob=sys.argv[2] or None,
    roi_regex=sys.argv[3] or None,
    subject_regex=sys.argv[4] or None,
    repeat_regex=sys.argv[5] or None,
)
print(len(records), len({record.subject for record in records}))
' "${MESH_QC_ROOT}" "${MESH_GLOB}" "${ROI_REGEX}" "${SUBJECT_REGEX}" "${REPEAT_REGEX}")"
    read -r FOUND_MESHES FOUND_SUBJECTS <<< "${DISCOVERY_COUNTS}"
    if [ -n "${MESH_QC_EXPECTED_MESHES}" ] && [ "${FOUND_MESHES}" != "${MESH_QC_EXPECTED_MESHES}" ]; then
        echo "[ERROR] Mesh-count gate failed: found ${FOUND_MESHES}, expected ${MESH_QC_EXPECTED_MESHES}." >&2
        exit 2
    fi
    if [ -n "${MESH_QC_EXPECTED_SUBJECTS}" ] && [ "${FOUND_SUBJECTS}" != "${MESH_QC_EXPECTED_SUBJECTS}" ]; then
        echo "[ERROR] Subject-count gate failed: found ${FOUND_SUBJECTS}, expected ${MESH_QC_EXPECTED_SUBJECTS}." >&2
        exit 2
    fi
    echo "[INFO] Discovery count gate passed: meshes=${FOUND_MESHES}, subjects=${FOUND_SUBJECTS}"
fi

PYTHON_CMD=(
    "${MESH_QC_PYTHON}"
    -E
    -u
    "${PIPELINE_DIR}/mesh_repeat_analysis/post/mesh_qc/run_mesh_qc.py"
    --root "${MESH_QC_ROOT}"
    --out "${MESH_QC_OUT}"
    --renderer "${RENDERER}"
    --workers "${WORKERS}"
    --progress "${PROGRESS_MODE}"
    --image-size "${IMAGE_SIZE}"
    --tile-size "${TILE_SIZE}"
    --discovery-progress-seconds "${DISCOVERY_PROGRESS_SECONDS}"
    --progress-every "${PROGRESS_EVERY}"
)

if [ -n "${MESH_GLOB}" ]; then
    PYTHON_CMD+=(--mesh-glob "${MESH_GLOB}")
fi
if [ -n "${ROI_REGEX}" ]; then
    PYTHON_CMD+=(--roi-regex "${ROI_REGEX}")
fi
if [ -n "${SUBJECT_REGEX}" ]; then
    PYTHON_CMD+=(--subject-regex "${SUBJECT_REGEX}")
fi
if [ -n "${REPEAT_REGEX}" ]; then
    PYTHON_CMD+=(--repeat-regex "${REPEAT_REGEX}")
fi
if [ -n "${COLS}" ]; then
    PYTHON_CMD+=(--cols "${COLS}")
fi
if [ "${CHECK_COMPONENTS}" = "1" ]; then
    PYTHON_CMD+=(--check-components)
fi
if [ "${ROI_WALLS}" = "1" ]; then
    PYTHON_CMD+=(--roi-walls)
fi
if [ "${TISSUE_WALLS}" = "1" ]; then
    PYTHON_CMD+=(--tissue-walls)
fi
case "${MESH_QC_STAGE}" in
    full)
        ;;
    qc)
        PYTHON_CMD+=(--qc-only)
        ;;
    render)
        PYTHON_CMD+=(--render-only)
        ;;
    tissue)
        PYTHON_CMD+=(--tissue-only)
        ;;
    *)
        echo "[ERROR] Unsupported MESH_QC_STAGE: ${MESH_QC_STAGE}" >&2
        echo "[ERROR] Use one of: full, qc, render, tissue" >&2
        exit 1
        ;;
esac

echo "[INFO] Command:"
printf '  %q' "${PYTHON_CMD[@]}"
printf '\n'

"${PYTHON_CMD[@]}" 2>&1 | tee "${MESH_QC_LOG_FILE}"
