#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Mesh QC submission controls
#
# Default runs may still be configured by editing this block once on the HPC:
#
#   bash hpc_scripts/submit_mesh_qc.sh
#
# Named profiles below provide repository-tracked settings for established
# campaigns. Environment variables with the same runtime names still override
# either the default block or selected profile for deliberate one-off runs.
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
SLURM_ERROR_CONFIG=""

MESH_GLOB_CONFIG=""
ROI_REGEX_CONFIG=""
SUBJECT_REGEX_CONFIG=""
REPEAT_REGEX_CONFIG=""
RENDERER_CONFIG="gmsh"
SIMNIBS_MODULE_CONFIG="SimNIBS/4.0.1-foss-2023a"
XVFB_MODULE_CONFIG="Xvfb/21.1.6-GCCcore-12.2.0"
GMSH_MODULE_CONFIG=""
MESH_QC_GMSH_BIN_CONFIG=""
MESH_QC_GMSH_TIMEOUT_SECONDS_CONFIG="900"
IMAGEMAGICK_MODULE_CONFIG=""
MESH_QC_PYTHON_CONFIG="python"
CHECK_COMPONENTS_CONFIG="0"
ROI_WALLS_CONFIG="0"
TISSUE_WALLS_CONFIG="0"
PROGRESS_CONFIG="text"
WORKERS_CONFIG="0"
MESH_QC_STAGE_CONFIG="full"
IMAGE_SIZE_CONFIG="1200"
TILE_SIZE_CONFIG="220"
COLS_CONFIG=""
DISCOVERY_PROGRESS_SECONDS_CONFIG="5"
PROGRESS_EVERY_CONFIG="25"
MESH_QC_EXPECTED_MESHES_CONFIG=""
MESH_QC_EXPECTED_SUBJECTS_CONFIG=""

SBATCH_BIN_CONFIG="sbatch"
SLURM_SCRIPT_CONFIG="${PIPELINE_DIR_CONFIG}/mesh_repeat_analysis/hpc_scripts/run_mesh_qc.slurm"
LOG_DIR_CONFIG="${PIPELINE_DIR_CONFIG}/logs"

PROFILE="${MESH_QC_PROFILE:-default}"
PREFLIGHT_ONLY="0"
EXPECTED_MESHES_ARGUMENT=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --profile)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --profile requires a value." >&2
                exit 2
            fi
            PROFILE="$2"
            shift 2
            ;;
        --preflight)
            PREFLIGHT_ONLY="1"
            shift
            ;;
        --expected-meshes)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --expected-meshes requires a positive integer." >&2
                exit 2
            fi
            EXPECTED_MESHES_ARGUMENT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash mesh_repeat_analysis/hpc_scripts/submit_mesh_qc.sh [options]"
            echo ""
            echo "Options:"
            echo "  --profile default|collected-charm-tissues"
            echo "  --preflight                    Discover and report scope without submitting"
            echo "  --expected-meshes N            Require exactly N meshes and unique subjects"
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

case "${PROFILE}" in
    default)
        ;;
    collected-charm-tissues)
        MESH_QC_ROOT_CONFIG="/mnt/parscratch/users/cop23bi/charm_segmentation_meshes_474/subjects"
        MESH_QC_OUT_CONFIG="/mnt/parscratch/users/cop23bi/mesh-wall/charm_segmentation_meshes_474_tissue_views"
        JOB_NAME_CONFIG="mesh_charm_tissues"
        CPUS_PER_TASK_CONFIG="16"
        MEMORY_CONFIG="64G"
        TIME_LIMIT_CONFIG="08:00:00"
        RENDERER_CONFIG="gmsh"
        SIMNIBS_MODULE_CONFIG="none"
        XVFB_MODULE_CONFIG="Xvfb/21.1.6-GCCcore-12.2.0"
        GMSH_MODULE_CONFIG="gmsh/4.11.1-foss-2022b"
        MESH_QC_GMSH_TIMEOUT_SECONDS_CONFIG="900"
        MESH_QC_PYTHON_CONFIG="${HOME}/.conda/envs/ti-post/bin/python"
        CHECK_COMPONENTS_CONFIG="0"
        ROI_WALLS_CONFIG="0"
        TISSUE_WALLS_CONFIG="1"
        PROGRESS_CONFIG="text"
        WORKERS_CONFIG="16"
        MESH_QC_STAGE_CONFIG="tissue"
        IMAGE_SIZE_CONFIG="1200"
        TILE_SIZE_CONFIG="220"
        COLS_CONFIG="10"
        DISCOVERY_PROGRESS_SECONDS_CONFIG="5"
        PROGRESS_EVERY_CONFIG="1"
        LOG_DIR_CONFIG="/mnt/parscratch/users/cop23bi/mesh-wall/logs/charm_segmentation_meshes_474_tissue_views"
        ;;
    *)
        echo "[ERROR] Unsupported mesh-QC profile: ${PROFILE}" >&2
        exit 2
        ;;
esac

if [ -n "${EXPECTED_MESHES_ARGUMENT}" ]; then
    if ! [[ "${EXPECTED_MESHES_ARGUMENT}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] --expected-meshes must be a positive integer." >&2
        exit 2
    fi
    MESH_QC_EXPECTED_MESHES_CONFIG="${EXPECTED_MESHES_ARGUMENT}"
    MESH_QC_EXPECTED_SUBJECTS_CONFIG="${EXPECTED_MESHES_ARGUMENT}"
fi

PIPELINE_DIR="${PIPELINE_DIR:-${PIPELINE_DIR_CONFIG}}"
MESH_QC_ROOT="${MESH_QC_ROOT:-${MESH_QC_ROOT_CONFIG}}"
MESH_QC_OUT="${MESH_QC_OUT:-${MESH_QC_OUT_CONFIG}}"
JOB_NAME="${JOB_NAME:-${JOB_NAME_CONFIG}}"
PARTITION="${PARTITION:-${PARTITION_CONFIG}}"
CPUS_PER_TASK="${CPUS_PER_TASK:-${CPUS_PER_TASK_CONFIG}}"
MEMORY="${MEMORY:-${MEMORY_CONFIG}}"
TIME_LIMIT="${TIME_LIMIT:-${TIME_LIMIT_CONFIG}}"
SLURM_OUTPUT="${SLURM_OUTPUT:-${SLURM_OUTPUT_CONFIG}}"
SLURM_ERROR="${SLURM_ERROR:-${SLURM_ERROR_CONFIG}}"
MESH_GLOB="${MESH_GLOB:-${MESH_GLOB_CONFIG}}"
ROI_REGEX="${ROI_REGEX:-${ROI_REGEX_CONFIG}}"
SUBJECT_REGEX="${SUBJECT_REGEX:-${SUBJECT_REGEX_CONFIG}}"
REPEAT_REGEX="${REPEAT_REGEX:-${REPEAT_REGEX_CONFIG}}"
RENDERER="${RENDERER:-${RENDERER_CONFIG}}"
SIMNIBS_MODULE="${SIMNIBS_MODULE:-${SIMNIBS_MODULE_CONFIG}}"
XVFB_MODULE="${XVFB_MODULE:-${XVFB_MODULE_CONFIG}}"
GMSH_MODULE="${GMSH_MODULE:-${GMSH_MODULE_CONFIG}}"
MESH_QC_GMSH_BIN="${MESH_QC_GMSH_BIN:-${MESH_QC_GMSH_BIN_CONFIG}}"
MESH_QC_GMSH_TIMEOUT_SECONDS="${MESH_QC_GMSH_TIMEOUT_SECONDS:-${MESH_QC_GMSH_TIMEOUT_SECONDS_CONFIG}}"
IMAGEMAGICK_MODULE="${IMAGEMAGICK_MODULE:-${IMAGEMAGICK_MODULE_CONFIG}}"
MESH_QC_PYTHON="${MESH_QC_PYTHON:-${MESH_QC_PYTHON_CONFIG}}"
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
MESH_QC_EXPECTED_MESHES="${MESH_QC_EXPECTED_MESHES:-${MESH_QC_EXPECTED_MESHES_CONFIG}}"
MESH_QC_EXPECTED_SUBJECTS="${MESH_QC_EXPECTED_SUBJECTS:-${MESH_QC_EXPECTED_SUBJECTS_CONFIG}}"
SBATCH_BIN="${SBATCH_BIN:-${SBATCH_BIN_CONFIG}}"
SLURM_SCRIPT="${SLURM_SCRIPT:-${SLURM_SCRIPT_CONFIG}}"
MESH_QC_LOG_DIR="${MESH_QC_LOG_DIR:-${LOG_DIR_CONFIG}}"

if ! [[ "${MESH_QC_GMSH_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] MESH_QC_GMSH_TIMEOUT_SECONDS must be a positive integer; got ${MESH_QC_GMSH_TIMEOUT_SECONDS}." >&2
    exit 2
fi
if [ "${PROFILE}" = "collected-charm-tissues" ] && [ "${MESH_QC_GMSH_TIMEOUT_SECONDS}" != "900" ]; then
    echo "[ERROR] The collected CHARM profile protects MESH_QC_GMSH_TIMEOUT_SECONDS=900; got ${MESH_QC_GMSH_TIMEOUT_SECONDS}." >&2
    exit 2
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
if [ -z "${SLURM_ERROR}" ]; then
    SLURM_ERROR="${MESH_QC_LOG_DIR}/slurm-%j.err"
fi

if [ "${PROFILE}" = "collected-charm-tissues" ]; then
    export PYTHONPATH="${PIPELINE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
    export PYTHONNOUSERSITE=1
    if ! command -v "${MESH_QC_PYTHON}" >/dev/null 2>&1; then
        echo "[ERROR] Mesh-QC Python is missing or not executable: ${MESH_QC_PYTHON}" >&2
        exit 2
    fi
    DISCOVERY_COUNTS="$("${MESH_QC_PYTHON}" -c '
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
    EXPECTED_TILES="$((FOUND_MESHES * 19))"
    OUTPUT_COUNTS="$("${MESH_QC_PYTHON}" -c '
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])

def count_nonempty(path, pattern):
    if not path.is_dir():
        return 0
    count = 0
    for item in path.rglob(pattern):
        try:
            count += int(item.is_file() and item.stat().st_size > 0)
        except OSError:
            pass
    return count

tiles = count_nonempty(root / "renders", "*.png")
walls = count_nonempty(root / "mosaics" / "tissues", "*_wall.png")
marker = root / "tissue_view_convention.json"
version = "none"
if marker.is_file():
    try:
        version = str(json.loads(marker.read_text(encoding="utf-8"))["version"])
    except Exception as exc:
        print(f"[ERROR] Could not read tissue-view marker {marker}: {exc}", file=sys.stderr)
        raise SystemExit(2)
if (tiles or walls) and version != "ras_anatomical_orthographic_compact_top_v3":
    print(
        f"[ERROR] Existing collected outputs have incompatible view marker: {version}",
        file=sys.stderr,
    )
    raise SystemExit(2)
print(tiles, walls, version)
' "${MESH_QC_OUT}")"
    read -r EXISTING_TILES EXISTING_WALLS EXISTING_VIEW_VERSION <<< "${OUTPUT_COUNTS}"
    echo "Scope:"
    echo "  dataset: collected CHARM segmentation meshes"
    echo "  subjects: ${FOUND_SUBJECTS}"
    echo "  meshes: ${FOUND_MESHES}"
    echo "  tasks: 1"
    echo "  expected outputs: ${EXPECTED_TILES} tissue tiles and 19 tissue walls under the complete nine-tag expectation"
    echo "  execution: full collected cohort; tissue-only; no smoke or reduced tasks"
    echo "[INFO] Protected Gmsh timeout: ${MESH_QC_GMSH_TIMEOUT_SECONDS}s per render"
    echo "[INFO] Existing resumable outputs: ${EXISTING_TILES} tissue tiles, ${EXISTING_WALLS} tissue walls"
    echo "[INFO] Existing view marker: ${EXISTING_VIEW_VERSION}"
    if [ "${FOUND_MESHES}" != "${FOUND_SUBJECTS}" ]; then
        echo "[ERROR] Collected root has ${FOUND_MESHES} meshes but ${FOUND_SUBJECTS} unique subjects." >&2
        exit 2
    fi
    if [ -n "${MESH_QC_EXPECTED_MESHES}" ] && [ "${FOUND_MESHES}" != "${MESH_QC_EXPECTED_MESHES}" ]; then
        echo "[ERROR] Found ${FOUND_MESHES} meshes; explicitly expected ${MESH_QC_EXPECTED_MESHES}." >&2
        exit 2
    fi
    if [ -n "${MESH_QC_EXPECTED_SUBJECTS}" ] && [ "${FOUND_SUBJECTS}" != "${MESH_QC_EXPECTED_SUBJECTS}" ]; then
        echo "[ERROR] Found ${FOUND_SUBJECTS} subjects; explicitly expected ${MESH_QC_EXPECTED_SUBJECTS}." >&2
        exit 2
    fi
    if [ "${PREFLIGHT_ONLY}" = "1" ]; then
        echo "[INFO] Preflight passed without submitting a job."
        echo "[INFO] Submit this exact scope with: bash mesh_repeat_analysis/hpc_scripts/submit_mesh_qc.sh --profile collected-charm-tissues --expected-meshes ${FOUND_MESHES}"
        exit 0
    fi
    if [ -z "${MESH_QC_EXPECTED_MESHES}" ]; then
        echo "[ERROR] Refusing collected-cohort submission without --expected-meshes. Run --preflight first." >&2
        exit 2
    fi
elif [ "${PREFLIGHT_ONLY}" = "1" ] || [ -n "${EXPECTED_MESHES_ARGUMENT}" ]; then
    echo "[ERROR] --preflight and --expected-meshes currently require --profile collected-charm-tissues." >&2
    exit 2
fi

EXPORT_VARS="ALL"
EXPORT_VARS+=",PIPELINE_DIR=${PIPELINE_DIR}"
EXPORT_VARS+=",MESH_QC_ROOT=${MESH_QC_ROOT}"
EXPORT_VARS+=",MESH_QC_OUT=${MESH_QC_OUT}"
EXPORT_VARS+=",MESH_GLOB=${MESH_GLOB}"
EXPORT_VARS+=",ROI_REGEX=${ROI_REGEX}"
EXPORT_VARS+=",SUBJECT_REGEX=${SUBJECT_REGEX}"
EXPORT_VARS+=",REPEAT_REGEX=${REPEAT_REGEX}"
EXPORT_VARS+=",RENDERER=${RENDERER}"
EXPORT_VARS+=",SIMNIBS_MODULE=${SIMNIBS_MODULE}"
EXPORT_VARS+=",XVFB_MODULE=${XVFB_MODULE}"
EXPORT_VARS+=",GMSH_MODULE=${GMSH_MODULE}"
EXPORT_VARS+=",MESH_QC_GMSH_BIN=${MESH_QC_GMSH_BIN}"
EXPORT_VARS+=",MESH_QC_GMSH_TIMEOUT_SECONDS=${MESH_QC_GMSH_TIMEOUT_SECONDS}"
EXPORT_VARS+=",IMAGEMAGICK_MODULE=${IMAGEMAGICK_MODULE}"
EXPORT_VARS+=",MESH_QC_PYTHON=${MESH_QC_PYTHON}"
EXPORT_VARS+=",CHECK_COMPONENTS=${CHECK_COMPONENTS}"
EXPORT_VARS+=",ROI_WALLS=${ROI_WALLS}"
EXPORT_VARS+=",TISSUE_WALLS=${TISSUE_WALLS}"
EXPORT_VARS+=",PROGRESS_MODE=${PROGRESS_MODE}"
EXPORT_VARS+=",WORKERS=${WORKERS}"
EXPORT_VARS+=",MESH_QC_STAGE=${MESH_QC_STAGE}"
EXPORT_VARS+=",IMAGE_SIZE=${IMAGE_SIZE}"
EXPORT_VARS+=",TILE_SIZE=${TILE_SIZE}"
EXPORT_VARS+=",COLS=${COLS}"
EXPORT_VARS+=",DISCOVERY_PROGRESS_SECONDS=${DISCOVERY_PROGRESS_SECONDS}"
EXPORT_VARS+=",PROGRESS_EVERY=${PROGRESS_EVERY}"
EXPORT_VARS+=",MESH_QC_EXPECTED_MESHES=${MESH_QC_EXPECTED_MESHES}"
EXPORT_VARS+=",MESH_QC_EXPECTED_SUBJECTS=${MESH_QC_EXPECTED_SUBJECTS}"
EXPORT_VARS+=",MESH_QC_LOG_DIR=${MESH_QC_LOG_DIR}"
EXPORT_VARS+=",LOG_DIR=${MESH_QC_LOG_DIR}"

echo "[INFO] Pipeline root:  ${PIPELINE_DIR}"
echo "[INFO] Profile:        ${PROFILE}"
echo "[INFO] Mesh root:      ${MESH_QC_ROOT}"
echo "[INFO] Output dir:     ${MESH_QC_OUT}"
echo "[INFO] Slurm script:   ${SLURM_SCRIPT}"
echo "[INFO] Log dir:        ${MESH_QC_LOG_DIR}"
echo "[INFO] Stage:          ${MESH_QC_STAGE}"
echo "[INFO] Renderer:       ${RENDERER}"
echo "[INFO] ROI regex:      ${ROI_REGEX:-<none>}"
echo "[INFO] Subject regex:  ${SUBJECT_REGEX:-<none>}"
echo "[INFO] Repeat regex:   ${REPEAT_REGEX:-<none>}"
echo "[INFO] SimNIBS module: ${SIMNIBS_MODULE:-<none>}"
echo "[INFO] Xvfb module:    ${XVFB_MODULE:-<none>}"
echo "[INFO] Gmsh module:    ${GMSH_MODULE:-<none>}"
echo "[INFO] Gmsh override:  ${MESH_QC_GMSH_BIN:-<none>}"
echo "[INFO] Gmsh timeout:   ${MESH_QC_GMSH_TIMEOUT_SECONDS}s per render"
echo "[INFO] ImageMagick module: ${IMAGEMAGICK_MODULE:-<none>}"
echo "[INFO] Python:          ${MESH_QC_PYTHON}"
echo "[INFO] Tissue walls:   ${TISSUE_WALLS}"
echo "[INFO] Workers:        ${WORKERS}"
echo "[INFO] CPUs per task:  ${CPUS_PER_TASK}"
echo "[INFO] Memory:         ${MEMORY}"
echo "[INFO] Time limit:     ${TIME_LIMIT}"
echo "[INFO] Expected meshes: ${MESH_QC_EXPECTED_MESHES:-<not enforced>}"
echo "[INFO] Expected subjects: ${MESH_QC_EXPECTED_SUBJECTS:-<not enforced>}"
echo "[INFO] Slurm output:   ${SLURM_OUTPUT}"
echo "[INFO] Slurm error:    ${SLURM_ERROR}"

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
if [ -n "${SLURM_ERROR}" ]; then
    SBATCH_CMD+=(--error="${SLURM_ERROR}")
fi
SBATCH_CMD+=("${SLURM_SCRIPT}")

"${SBATCH_CMD[@]}"
