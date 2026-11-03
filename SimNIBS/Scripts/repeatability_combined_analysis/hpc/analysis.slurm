#!/bin/bash
#SBATCH --job-name=ti_repeat_combined
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

module purge
export PYTHONNOUSERSITE=1

: "${PACKAGE_DIR:?Set PACKAGE_DIR.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT.}"
: "${ANALYSIS_PYTHON:?Set ANALYSIS_PYTHON.}"
: "${PREPARED_PLAN_SHA256:?Set PREPARED_PLAN_SHA256.}"
: "${REPOSITORY_COMMIT:?Set REPOSITORY_COMMIT.}"

[ -x "$ANALYSIS_PYTHON" ] || {
    echo "[ERROR] Analysis Python is not executable: $ANALYSIS_PYTHON" >&2
    exit 2
}

SCRIPTS_ROOT="$(dirname "$PACKAGE_DIR")"
export PYTHONPATH="$SCRIPTS_ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="$OUTPUT_ROOT/_workflow/mplconfig-${SLURM_JOB_ID:-manual}"
mkdir -p "$MPLCONFIGDIR"

"$ANALYSIS_PYTHON" -c 'import matplotlib, numpy, pandas, scipy'

CODE_OWNED_ROOT=$("$ANALYSIS_PYTHON" -c \
    'from repeatability_combined_analysis.settings import production_config; print(production_config().output_root)')
[ "$OUTPUT_ROOT" = "$CODE_OWNED_ROOT" ] || {
    echo "[ERROR] Output root differs from the code-owned production root." >&2
    exit 2
}

PLAN_FILE="$OUTPUT_ROOT/_workflow/input_plan.json"
if [ ! -f "$PLAN_FILE" ] || [ -L "$PLAN_FILE" ]; then
    echo "[ERROR] Prepared input plan is missing or unsafe: $PLAN_FILE" >&2
    exit 2
fi
OBSERVED_PLAN_SHA256=$(awk -F'"' '/"plan_sha256"/{print $4; exit}' "$PLAN_FILE")
[ "$OBSERVED_PLAN_SHA256" = "$PREPARED_PLAN_SHA256" ] || {
    echo "[ERROR] Prepared plan hash differs from the submitted hash." >&2
    exit 2
}

CURRENT_COMMIT=$(git -C "$SCRIPTS_ROOT" rev-parse HEAD)
[ "$CURRENT_COMMIT" = "$REPOSITORY_COMMIT" ] || {
    echo "[ERROR] Repository commit differs from the submitted commit." >&2
    exit 2
}

echo "[INFO] Executing source-bound combined repeatability analysis"
echo "[INFO] Output root: $OUTPUT_ROOT"
echo "[INFO] Plan SHA-256: $PREPARED_PLAN_SHA256"
echo "[INFO] Repository commit: $REPOSITORY_COMMIT"

"$ANALYSIS_PYTHON" -m repeatability_combined_analysis execute

MANIFEST="$OUTPUT_ROOT/analysis_manifest.json"
[ -s "$MANIFEST" ] || {
    echo "[ERROR] Validated final manifest is absent: $MANIFEST" >&2
    exit 2
}
cat "$MANIFEST"

