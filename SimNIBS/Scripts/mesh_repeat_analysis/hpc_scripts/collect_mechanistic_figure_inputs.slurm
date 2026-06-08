#!/bin/bash
#SBATCH --job-name=collect_mech_fig_inputs
#SBATCH --output=collect_mech_fig_inputs_%j.out
#SBATCH --error=collect_mech_fig_inputs_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

: "${BASE_OUTPUT_DIR:?Set BASE_OUTPUT_DIR to the full repeatability output root}"
: "${OUT_DIR:?Set OUT_DIR to the compact bundle output directory}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PIPELINE_DIR="${PIPELINE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
COLLECTOR="${COLLECTOR:-${PIPELINE_DIR}/post/collect_mechanistic_figure_inputs.py}"
COPY_MODE="${COPY_MODE:-dry-run}"
THRESHOLD="${THRESHOLD:-0.2}"
MAX_TOTAL_BYTES="${MAX_TOTAL_BYTES:-5000000000}"
COVERAGE_CSV="${COVERAGE_CSV:-}"
PREFERRED_SUBJECT="${PREFERRED_SUBJECT:-}"
FIXED_REFERENCE_REPEAT="${FIXED_REFERENCE_REPEAT:-repeat_001}"

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
"$PYTHON_BIN" "${ARGS[@]}"
