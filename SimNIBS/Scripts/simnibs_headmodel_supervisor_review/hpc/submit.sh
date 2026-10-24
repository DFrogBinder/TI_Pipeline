#!/bin/bash
set -euo pipefail

PACKAGE_DIR="${PACKAGE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPTS_ROOT="${SCRIPTS_ROOT:-$(cd "$PACKAGE_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
PARTITION="${PARTITION:-sheffield}"
SIM_CPUS="${SIM_CPUS:-8}"
SIM_MEMORY="${SIM_MEMORY:-32G}"
SIM_TIME="${SIM_TIME:-08:00:00}"

MODE="${1:-preflight}"
if [ "$#" -gt 0 ]; then
    shift
fi

ORIGINAL_CHARM_ROOT="/mnt/parscratch/users/cop23bi/charm_segmentations"
CORRECTED_V4_SCAFFOLD_ROOT="/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Scaffolds"
LEGACY_V3_ROOT="/mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_simnibs326_headreco"
HARMONIZED_V3_ROOT="/mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_simnibs326_harmonized_v2"
ATLAS_ROOT="/mnt/parscratch/users/cop23bi/ZIPs/atlases"
COMPLETED_V4_LEFT_ROOT="/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10"
OUTPUT_ROOT="/mnt/parscratch/users/cop23bi/simnibs_headmodel_supervisor_review_package_v1"
CONFIRM_OUTPUT_ROOT=""

for argument in "$@"; do
    case "$argument" in
        --original-charm-root=*) ORIGINAL_CHARM_ROOT="${argument#*=}" ;;
        --corrected-v4-scaffold-root=*) CORRECTED_V4_SCAFFOLD_ROOT="${argument#*=}" ;;
        --legacy-v3-root=*) LEGACY_V3_ROOT="${argument#*=}" ;;
        --harmonized-v3-root=*) HARMONIZED_V3_ROOT="${argument#*=}" ;;
        --atlas-root=*) ATLAS_ROOT="${argument#*=}" ;;
        --completed-v4-left-root=*) COMPLETED_V4_LEFT_ROOT="${argument#*=}" ;;
        --output-root=*) OUTPUT_ROOT="${argument#*=}" ;;
        --confirm-output-root=*) CONFIRM_OUTPUT_ROOT="${argument#*=}" ;;
        *)
            echo "[ERROR] Unknown argument: $argument" >&2
            exit 2
            ;;
    esac
done

case "$MODE" in
    preflight|submit|status|validate) ;;
    *)
        echo "[ERROR] Unsupported mode: $MODE" >&2
        echo "Usage: $0 {preflight|submit|status|validate} [path options]" >&2
        exit 2
        ;;
esac

WORKER="$PACKAGE_DIR/hpc/build.slurm"
if [ ! -f "$WORKER" ]; then
    echo "[ERROR] Missing workflow file: $WORKER" >&2
    exit 2
fi

export PYTHONPATH="$SCRIPTS_ROOT:${PYTHONPATH:-}"
common_args=(
    "--original-charm-root=$ORIGINAL_CHARM_ROOT"
    "--corrected-v4-scaffold-root=$CORRECTED_V4_SCAFFOLD_ROOT"
    "--legacy-v3-root=$LEGACY_V3_ROOT"
    "--harmonized-v3-root=$HARMONIZED_V3_ROOT"
    "--atlas-root=$ATLAS_ROOT"
    "--completed-v4-left-root=$COMPLETED_V4_LEFT_ROOT"
    "--output-root=$OUTPUT_ROOT"
)

run_review() {
    local command="$1"
    "$PYTHON_BIN" -m simnibs_headmodel_supervisor_review \
        "$command" "${common_args[@]}"
}

resolved_path() {
    "$PYTHON_BIN" -c \
        'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve(strict=False))' \
        "$1"
}

require_confirmation() {
    local expected observed
    expected="$(resolved_path "$OUTPUT_ROOT")"
    if [ -z "$CONFIRM_OUTPUT_ROOT" ]; then
        echo "[ERROR] --confirm-output-root=$expected is required for submit." >&2
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

print_scope() {
    local submitted="$1"
    printf '%s\n' \
        "Scope:" \
        "  workflow: supervisor review package" \
        "  candidates: 9" \
        "  selected subjects: 1" \
        "  scientific artifacts: 8" \
        "  source roots: read-only" \
        "  writable root: $OUTPUT_ROOT" \
        "  resources: $PARTITION; $SIM_CPUS CPU; $SIM_MEMORY; $SIM_TIME" \
        "  module: SimNIBS/3.2.6-foss-2023a" \
        "  jobs submitted: $submitted"
}

case "$MODE" in
    preflight)
        print_scope 0
        run_review preflight
        ;;
    status)
        print_scope 0
        run_review status
        ;;
    validate)
        print_scope 0
        run_review validate
        ;;
    submit)
        run_review preflight >/dev/null
        require_confirmation
        run_review select >/dev/null
        mkdir -p "$OUTPUT_ROOT/logs"
        raw_job_id="$(
            "$SBATCH_BIN" --parsable \
                --partition="$PARTITION" \
                --cpus-per-task="$SIM_CPUS" \
                --mem="$SIM_MEMORY" \
                --time="$SIM_TIME" \
                --output="$OUTPUT_ROOT/logs/slurm-%j.out" \
                --open-mode=append \
                --export="ALL,SCRIPTS_ROOT=$SCRIPTS_ROOT,SIMNIBS_REVIEW_ORIGINAL_CHARM_ROOT=$ORIGINAL_CHARM_ROOT,SIMNIBS_REVIEW_CORRECTED_V4_SCAFFOLD_ROOT=$CORRECTED_V4_SCAFFOLD_ROOT,SIMNIBS_REVIEW_LEGACY_V3_ROOT=$LEGACY_V3_ROOT,SIMNIBS_REVIEW_HARMONIZED_V3_ROOT=$HARMONIZED_V3_ROOT,SIMNIBS_REVIEW_ATLAS_ROOT=$ATLAS_ROOT,SIMNIBS_REVIEW_COMPLETED_V4_LEFT_ROOT=$COMPLETED_V4_LEFT_ROOT,SIMNIBS_REVIEW_OUTPUT_ROOT=$OUTPUT_ROOT" \
                "$WORKER"
        )"
        job_id="$(parse_job_id "$raw_job_id")"
        print_scope 1
        echo "[INFO] Review-package job: $job_id"
        ;;
esac
