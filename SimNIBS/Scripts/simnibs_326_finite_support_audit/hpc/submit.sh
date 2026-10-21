#!/bin/bash
set -euo pipefail

PIPELINE_DIR="${PIPELINE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROJECT_DIR="$(dirname "$PIPELINE_DIR")"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
OUTPUT_ROOT="${FINITE_SUPPORT_AUDIT_ROOT:-/mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_simnibs326_headreco/_simnibs326/finite_support_forensics_v1}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${1:-preflight}"

ARRAY_SCRIPT="$PIPELINE_DIR/hpc/audit_array.slurm"
COLLECT_SCRIPT="$PIPELINE_DIR/hpc/collect.slurm"
SUBMISSION_RECEIPT="$OUTPUT_ROOT/submission.tsv"

case "$MODE" in
    scope|preflight|run|status) ;;
    *)
        echo "Usage: $0 {scope|preflight|run|status}" >&2
        exit 2
        ;;
esac

for required in "$ARRAY_SCRIPT" "$COLLECT_SCRIPT"; do
    if [ ! -f "$required" ]; then
        echo "[ERROR] Missing workflow file: $required" >&2
        exit 2
    fi
done

print_scope() {
    echo "Finite-support forensic audit scope"
    echo "  study: completed balanced-10 SimNIBS v3.2.6 vs v4 repeatability"
    echo "  subjects: 10"
    echo "  targets: 2 (left hippocampus, right M1)"
    echo "  conditions: 2 (remesh, fixed_mesh)"
    echo "  repeats per condition: 40"
    echo "  v3 retained runs inspected: 1,600"
    echo "  matched v4 retained runs inspected: 1,600"
    echo "  direct v3 TI.msh fields inspected: 1,600"
    echo "  immutable production logs inspected: 1,600 task histories"
    echo "  diagnostic v3/v4 panels: 40"
    echo "  audit array: 0-19%10 (20 subject-target cells)"
    echo "  meshing/FEM tasks: 0"
    echo "  source-output modifications: 0"
    echo "  isolated output: $OUTPUT_ROOT"
}

parse_job_id() {
    local raw="$1"
    local job_id="${raw%%;*}"
    if ! [[ "$job_id" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Could not parse Slurm job ID: $raw" >&2
        exit 2
    fi
    printf '%s\n' "$job_id"
}

run_preflight() {
    "$PYTHON_BIN" -m simnibs_326_finite_support_audit preflight \
        --output-root "$OUTPUT_ROOT"
}

if [ "$MODE" = "scope" ]; then
    print_scope
    exit 0
fi

if [ "$MODE" = "preflight" ]; then
    print_scope
    run_preflight
    exit 0
fi

if [ "$MODE" = "run" ]; then
    print_scope
    run_preflight
    mkdir -p "$OUTPUT_ROOT/logs"
    raw_array="$($SBATCH_BIN --parsable \
        --output="$OUTPUT_ROOT/logs/audit-%A_%a.out" \
        --export="ALL,PIPELINE_DIR=$PIPELINE_DIR,OUTPUT_ROOT=$OUTPUT_ROOT" \
        "$ARRAY_SCRIPT")"
    array_job="$(parse_job_id "$raw_array")"
    raw_collect="$($SBATCH_BIN --parsable \
        --dependency="afterok:$array_job" \
        --output="$OUTPUT_ROOT/logs/collect-%j.out" \
        --export="ALL,PIPELINE_DIR=$PIPELINE_DIR,OUTPUT_ROOT=$OUTPUT_ROOT" \
        "$COLLECT_SCRIPT")"
    collect_job="$(parse_job_id "$raw_collect")"
    printf 'submitted_utc\taudit_array_job\tcollector_job\toutput_root\n' > "$SUBMISSION_RECEIPT"
    printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$array_job" "$collect_job" "$OUTPUT_ROOT" >> "$SUBMISSION_RECEIPT"
    echo "Submitted audit array: $array_job"
    echo "Submitted afterok collector: $collect_job"
    echo "Receipt: $SUBMISSION_RECEIPT"
    exit 0
fi

if [ ! -f "$SUBMISSION_RECEIPT" ]; then
    echo "[ERROR] No submission receipt: $SUBMISSION_RECEIPT" >&2
    exit 2
fi
array_job="$(awk -F '\t' 'NR==2 {print $2}' "$SUBMISSION_RECEIPT")"
collect_job="$(awk -F '\t' 'NR==2 {print $3}' "$SUBMISSION_RECEIPT")"
sacct -D -X -j "$array_job,$collect_job" \
    --format=JobIDRaw,JobName%30,State,ExitCode,Elapsed,Start,End
