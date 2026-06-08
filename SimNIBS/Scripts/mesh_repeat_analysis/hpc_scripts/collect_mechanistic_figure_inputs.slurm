#!/bin/bash
#SBATCH --job-name=collect_mech_fig_inputs
#SBATCH --output=collect_mech_fig_inputs_%j.out
#SBATCH --error=collect_mech_fig_inputs_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

# ---------------------------------------------------------------------------
# Mechanistic figure input collection settings
#
# Edit this block in the version-controlled script, then submit with:
#
#   sbatch hpc_scripts/collect_mechanistic_figure_inputs.slurm
#
# To write manifests without copying files:
#
#   sbatch hpc_scripts/collect_mechanistic_figure_inputs.slurm --dry-run
#
# Environment variables with the same names still override these defaults for
# one-off smoke tests, but normal runs should be tracked by editing this block
# or by passing the explicit --dry-run script argument.
# ---------------------------------------------------------------------------
BASE_OUTPUT_DIR_CONFIG="/mnt/parscratch/users/cop23bi/balaned-repeatability-outputs"
OUT_DIR_CONFIG="/mnt/parscratch/users/cop23bi/mechanistic_figure_bundle"
COPY_MODE_CONFIG="copy"
THRESHOLD_CONFIG="0.2"
MAX_TOTAL_BYTES_CONFIG="5000000000"
COVERAGE_CSV_CONFIG=""
PREFERRED_SUBJECT_CONFIG=""
FIXED_REFERENCE_REPEAT_CONFIG="repeat_001"
PYTHON_BIN_CONFIG="python3"
PIPELINE_DIR_CONFIG=""
COLLECTOR_CONFIG=""

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${BASE_OUTPUT_DIR_CONFIG}}"
OUT_DIR="${OUT_DIR:-${OUT_DIR_CONFIG}}"
COPY_MODE="${COPY_MODE:-${COPY_MODE_CONFIG}}"
THRESHOLD="${THRESHOLD:-${THRESHOLD_CONFIG}}"
MAX_TOTAL_BYTES="${MAX_TOTAL_BYTES:-${MAX_TOTAL_BYTES_CONFIG}}"
COVERAGE_CSV="${COVERAGE_CSV:-${COVERAGE_CSV_CONFIG}}"
PREFERRED_SUBJECT="${PREFERRED_SUBJECT:-${PREFERRED_SUBJECT_CONFIG}}"
FIXED_REFERENCE_REPEAT="${FIXED_REFERENCE_REPEAT:-${FIXED_REFERENCE_REPEAT_CONFIG}}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON_BIN_CONFIG}}"
PIPELINE_DIR="${PIPELINE_DIR:-${PIPELINE_DIR_CONFIG:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}}"
COLLECTOR="${COLLECTOR:-${COLLECTOR_CONFIG:-${PIPELINE_DIR}/post/collect_mechanistic_figure_inputs.py}}"

usage() {
  echo "Usage: sbatch hpc_scripts/collect_mechanistic_figure_inputs.slurm [--dry-run]"
}

for arg in "$@"; do
  case "$arg" in
    --dry-run)
      COPY_MODE="dry-run"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $arg"
      usage
      exit 1
      ;;
  esac
done

echo "[INFO] Mechanistic figure input collection"
echo "[INFO] Job ID:                 ${SLURM_JOB_ID:-manual}"
echo "[INFO] Host:                   $(hostname)"
echo "[INFO] Submit dir:             ${SLURM_SUBMIT_DIR:-n/a}"
echo "[INFO] Pipeline dir:           ${PIPELINE_DIR}"
echo "[INFO] Collector:              ${COLLECTOR}"
echo "[INFO] Python:                 ${PYTHON_BIN}"
echo "[INFO] Base output dir:        ${BASE_OUTPUT_DIR}"
echo "[INFO] Output dir:             ${OUT_DIR}"
echo "[INFO] Copy mode:              ${COPY_MODE}"
echo "[INFO] Threshold:              ${THRESHOLD}"
echo "[INFO] Max total bytes:        ${MAX_TOTAL_BYTES}"
echo "[INFO] Coverage CSV:           ${COVERAGE_CSV:-compute from TI volumes}"
echo "[INFO] Preferred subject:      ${PREFERRED_SUBJECT:-auto-select largest coverage range}"
echo "[INFO] Fixed reference repeat: ${FIXED_REFERENCE_REPEAT}"

if [[ -z "$BASE_OUTPUT_DIR" ]]; then
  echo "[ERROR] BASE_OUTPUT_DIR is empty. Edit BASE_OUTPUT_DIR_CONFIG near the top of this script."
  exit 1
fi
if [[ -z "$OUT_DIR" ]]; then
  echo "[ERROR] OUT_DIR is empty. Edit OUT_DIR_CONFIG near the top of this script."
  exit 1
fi

ARGS=(
  "$COLLECTOR"
  --base-output-dir "$BASE_OUTPUT_DIR"
  --out-dir "$OUT_DIR"
  --threshold "$THRESHOLD"
  --max-total-bytes "$MAX_TOTAL_BYTES"
  --fixed-reference-repeat "$FIXED_REFERENCE_REPEAT"
)

if [[ -n "$COVERAGE_CSV" ]]; then
  ARGS+=(--coverage-csv "$COVERAGE_CSV")
fi

if [[ -n "$PREFERRED_SUBJECT" ]]; then
  ARGS+=(--preferred-subject "$PREFERRED_SUBJECT")
fi

if [[ "$COPY_MODE" == "copy" ]]; then
  ARGS+=(--copy)
else
  ARGS+=(--dry-run)
fi

cd "$PIPELINE_DIR"
echo "[INFO] Command:"
printf '  %q' "$PYTHON_BIN" "${ARGS[@]}"
printf '\n'
"$PYTHON_BIN" "${ARGS[@]}"
