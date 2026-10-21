#!/bin/bash
#SBATCH --job-name=ti326_support_collect
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --open-mode=append

set -euo pipefail

module purge
module use "$HOME/modules"
module load SimNIBS/3.2.6-foss-2023a

export PYTHONNOUSERSITE=1
: "${PIPELINE_DIR:?Set PIPELINE_DIR to simnibs_326_finite_support_audit.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the isolated audit output root.}"
PROJECT_DIR="$(dirname "$PIPELINE_DIR")"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

mkdir -p "$OUTPUT_ROOT"
ACCOUNTING_TMP="$OUTPUT_ROOT/.production_sacct.tsv.tmp-${SLURM_JOB_ID:-manual}"
if sacct -D -X \
    -j 11789553,11789558,11797124,11797126 \
    --format=JobIDRaw,JobName%30,State,ExitCode,Elapsed,Start,End \
    --parsable2 > "$ACCOUNTING_TMP"; then
    mv "$ACCOUNTING_TMP" "$OUTPUT_ROOT/production_sacct.tsv"
else
    printf 'sacct unavailable; immutable task logs remain the run-level evidence\n' \
        > "$OUTPUT_ROOT/production_sacct.tsv"
    rm -f "$ACCOUNTING_TMP"
fi

python -u -m simnibs_326_finite_support_audit collect \
    --output-root "$OUTPUT_ROOT"
