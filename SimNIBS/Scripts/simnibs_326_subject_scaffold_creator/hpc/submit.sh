#!/bin/bash
set -euo pipefail

CREATOR_DIR="${CREATOR_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPTS_ROOT="${SCRIPTS_ROOT:-$(cd "$CREATOR_DIR/.." && pwd)}"
REPEATABILITY_DIR="${REPEATABILITY_DIR:-$SCRIPTS_ROOT/simnibs_326_repeatability}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
PARTITION="${PARTITION:-sheffield}"
SIM_CPUS="${SIM_CPUS:-8}"
SIM_MEMORY="${SIM_MEMORY:-32G}"
SIM_TIME="${SIM_TIME:-08:00:00}"
MATLAB_MODULE="${SIMNIBS326_MATLAB_MODULE:-MATLAB/2023b}"
MAX_RETRIES="${SIMNIBS326_MAX_RETRIES:-1}"

MODE="${1:-preflight}"
if [ "$#" -gt 0 ]; then
    shift
fi
SUBJECT=""
CORRECTED_SOURCE_ROOT="/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Scaffolds"
CREATOR_OUTPUT_ROOT="/mnt/parscratch/users/cop23bi/simnibs326_subject_scaffolds_v1"
FINAL132_SUBJECTS="$SCRIPTS_ROOT/CamCan_Experiment/cohort_pipeline/cohorts/final_132/subjects.txt"
CONFIRM_OUTPUT_ROOT=""
for argument in "$@"; do
    case "$argument" in
        --subject=*) SUBJECT="${argument#*=}" ;;
        --corrected-source-root=*) CORRECTED_SOURCE_ROOT="${argument#*=}" ;;
        --output-root=*) CREATOR_OUTPUT_ROOT="${argument#*=}" ;;
        --final132-subjects=*) FINAL132_SUBJECTS="${argument#*=}" ;;
        --confirm-output-root=*) CONFIRM_OUTPUT_ROOT="${argument#*=}" ;;
        *)
            echo "[ERROR] Unknown argument: $argument" >&2
            exit 2
            ;;
    esac
done

case "$MODE" in
    preflight|module-check|submit|status|validate) ;;
    *)
        echo "[ERROR] Unsupported mode: $MODE" >&2
        echo "Usage: $0 {preflight|module-check|submit|status|validate} --subject=sub-CC###### [path options]" >&2
        exit 2
        ;;
esac
if [ -z "$SUBJECT" ]; then
    echo "[ERROR] --subject=sub-CC###### is required." >&2
    exit 2
fi
if ! [[ "$MAX_RETRIES" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] SIMNIBS326_MAX_RETRIES must be a non-negative integer." >&2
    exit 2
fi

WORKER="$CREATOR_DIR/hpc/scaffold_array.slurm"
MODULE_PREFLIGHT="$REPEATABILITY_DIR/hpc/module_preflight.slurm"
COMPAT_SHIM="$REPEATABILITY_DIR/hpc/activate_compat.sh"
CAT12_COMPAT="$REPEATABILITY_DIR/cat12_compat.py"
for required in "$WORKER" "$MODULE_PREFLIGHT" "$COMPAT_SHIM" "$CAT12_COMPAT"; do
    if [ ! -f "$required" ]; then
        echo "[ERROR] Missing workflow file: $required" >&2
        exit 2
    fi
done

export PYTHONPATH="$SCRIPTS_ROOT:${PYTHONPATH:-}"
common_args=(
    "--subject=$SUBJECT"
    "--corrected-source-root=$CORRECTED_SOURCE_ROOT"
    "--output-root=$CREATOR_OUTPUT_ROOT"
    "--final132-subjects=$FINAL132_SUBJECTS"
)

run_creator() {
    local command="$1"
    "$PYTHON_BIN" -m simnibs_326_subject_scaffold_creator \
        "$command" "${common_args[@]}"
}

resolved_path() {
    "$PYTHON_BIN" -c \
        'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve(strict=False))' \
        "$1"
}

require_confirmation() {
    local expected observed
    expected="$(resolved_path "$CREATOR_OUTPUT_ROOT")"
    if [ -z "$CONFIRM_OUTPUT_ROOT" ]; then
        echo "[ERROR] --confirm-output-root=$expected is required for $MODE." >&2
        exit 2
    fi
    observed="$(resolved_path "$CONFIRM_OUTPUT_ROOT")"
    if [ "$observed" != "$expected" ]; then
        echo "[ERROR] --confirm-output-root must exactly match $expected" >&2
        exit 2
    fi
}

parse_job_id() {
    local raw="$1"
    if ! [[ "$raw" =~ ^[0-9]+(\;[^[:space:]\;]+)?$ ]]; then
        echo "[ERROR] Could not parse Slurm job ID from: $raw" >&2
        return 2
    fi
    printf '%s\n' "${raw%%;*}"
}

PREFLIGHT_RECEIPT="$CREATOR_OUTPUT_ROOT/_simnibs326/module_preflight.json"
LOG_DIR="$CREATOR_OUTPUT_ROOT/_simnibs326/logs/$SUBJECT"

submit_module_check() {
    mkdir -p "$(dirname "$PREFLIGHT_RECEIPT")"
    local raw
    raw="$(
        "$SBATCH_BIN" --parsable \
            --partition="$PARTITION" \
            --output="$CREATOR_OUTPUT_ROOT/_simnibs326/module-preflight-%j.out" \
            --export="ALL,PIPELINE_DIR=$REPEATABILITY_DIR,PREFLIGHT_RECEIPT=$PREFLIGHT_RECEIPT,SIMNIBS326_MATLAB_MODULE=$MATLAB_MODULE" \
            "$MODULE_PREFLIGHT"
    )"
    parse_job_id "$raw"
}

submit_scaffold() {
    local module_job="$1"
    mkdir -p "$LOG_DIR"
    local raw
    raw="$(
        "$SBATCH_BIN" --parsable \
            --partition="$PARTITION" \
            --cpus-per-task="$SIM_CPUS" \
            --mem="$SIM_MEMORY" \
            --time="$SIM_TIME" \
            --array="0-0%1" \
            --dependency="afterok:$module_job" \
            --output="$LOG_DIR/slurm-%A_%a.out" \
            --open-mode=append \
            --export="ALL,SCRIPTS_ROOT=$SCRIPTS_ROOT,REPEATABILITY_DIR=$REPEATABILITY_DIR,PREFLIGHT_RECEIPT=$PREFLIGHT_RECEIPT,SUBJECT=$SUBJECT,CORRECTED_SOURCE_ROOT=$CORRECTED_SOURCE_ROOT,CREATOR_OUTPUT_ROOT=$CREATOR_OUTPUT_ROOT,FINAL132_SUBJECTS=$FINAL132_SUBJECTS,LOG_DIR=$LOG_DIR,SIMNIBS326_MATLAB_MODULE=$MATLAB_MODULE,SIMNIBS326_MAX_RETRIES=$MAX_RETRIES" \
            "$WORKER"
    )"
    parse_job_id "$raw"
}

case "$MODE" in
    preflight)
        run_creator preflight
        ;;
    status)
        run_creator status
        ;;
    validate)
        run_creator validate
        ;;
    module-check)
        run_creator preflight >/dev/null
        require_confirmation
        module_job="$(submit_module_check)"
        echo "[INFO] Module-check job: $module_job"
        ;;
    submit)
        run_creator preflight >/dev/null
        require_confirmation
        module_job="$(submit_module_check)"
        scaffold_job="$(submit_scaffold "$module_job")"
        echo "[INFO] Subject scaffold chain: module=$module_job scaffold=$scaffold_job"
        ;;
esac
