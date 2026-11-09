#!/bin/bash
set -euo pipefail

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$HPC_DIR/.." && pwd)"
PIPELINE_DIR="${PIPELINE_DIR:-$(cd "$PACKAGE_DIR/.." && pwd)}"
export PYTHONPATH="$PIPELINE_DIR:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ANALYSIS_PYTHON="${ANALYSIS_PYTHON:-$PYTHON_BIN}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
OUTPUT_ROOT="${SEGDET_OUTPUT_ROOT:-/mnt/parscratch/users/cop23bi/simnibs_segmentation_determinism_v1}"
SOURCE_ROOT="${SEGDET_SOURCE_ROOT:-/mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_corrected}"
PARTITION="${SEGDET_PARTITION:-sheffield}"
CPUS="${SEGDET_CPUS:-8}"
MEMORY="${SEGDET_MEMORY:-32G}"
TIME_LIMIT="${SEGDET_TIME_LIMIT:-08:00:00}"
MAX_CONCURRENT="${SEGDET_MAX_CONCURRENT:-10}"
MAX_RETRIES="${SEGDET_MAX_RETRIES:-1}"
MODE="${1:-scope}"
export SEGDET_OUTPUT_ROOT="$OUTPUT_ROOT"
export SEGDET_SOURCE_ROOT="$SOURCE_ROOT"

case "$MODE" in
    scope|preflight|init|run|status|analyze) ;;
    *)
        echo "Usage: $0 {scope|preflight|init|run|status|analyze}" >&2
        exit 2
        ;;
esac

if [[ "$CPUS" != "8" || "$MEMORY" != "32G" || "$TIME_LIMIT" != "08:00:00" ]]; then
    echo "Segmentation resources are pinned to 8 CPU, 32G, 08:00:00." >&2
    exit 2
fi
if [[ "$MAX_CONCURRENT" != "10" ]]; then
    echo "Array concurrency is pinned to 10." >&2
    exit 2
fi
if ! [[ "$MAX_RETRIES" =~ ^[01]$ ]]; then
    echo "SEGDET_MAX_RETRIES must be 0 or 1." >&2
    exit 2
fi

required=(
    "$HPC_DIR/module_preflight_326.slurm"
    "$HPC_DIR/module_preflight_401.slurm"
    "$HPC_DIR/headreco_array.slurm"
    "$HPC_DIR/charm_array.slurm"
    "$HPC_DIR/collect.slurm"
    "$PIPELINE_DIR/simnibs_326_repeatability/hpc/activate_compat.sh"
    "$PIPELINE_DIR/simnibs_326_repeatability/cat12_compat.py"
)
for path in "${required[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing workflow file: $path" >&2
        exit 2
    fi
done

printf '%s\n' \
    'Scope:' \
    '  workflow: single-subject segmentation determinism' \
    '  subject: sub-CC110174' \
    '  implementations: 2 (SimNIBS 3.2.6 headreco, SimNIBS 4.0.1 CHARM)' \
    '  repeats per implementation: 10' \
    '  scientific segmentation tasks: 20' \
    '  scheduler tasks: 23 (2 module preflights + 20 array elements + 1 collector)' \
    '  surface tasks: 0' \
    '  meshing tasks: 0' \
    '  FEM simulation tasks: 0' \
    '  resources: sheffield; 8 CPU; 32G; 08:00:00; 0-9%10 per arm' \
    '  retries: at most 1 exact array-element requeue' \
    '  submission default: none'
"$PYTHON_BIN" -m simnibs_segmentation_determinism scope

case "$MODE" in
    scope)
        exit 0
        ;;
    preflight)
        exec "$PYTHON_BIN" -m simnibs_segmentation_determinism preflight
        ;;
    init)
        exec "$PYTHON_BIN" -m simnibs_segmentation_determinism init
        ;;
    status)
        exec "$PYTHON_BIN" -m simnibs_segmentation_determinism status
        ;;
    analyze)
        exec "$ANALYSIS_PYTHON" -m simnibs_segmentation_determinism analyze
        ;;
esac

"$PYTHON_BIN" -m simnibs_segmentation_determinism preflight
"$PYTHON_BIN" -m simnibs_segmentation_determinism init

log_root="$OUTPUT_ROOT/_pipeline/logs"
workflow_root="$OUTPUT_ROOT/_pipeline/workflow"
ledger="$OUTPUT_ROOT/_pipeline/job_ids.tsv"
mkdir -p "$log_root/module_preflight" "$log_root/simnibs326_headreco" \
    "$log_root/simnibs401_charm" "$log_root/collector" "$workflow_root"
submission_receipt="$workflow_root/submission.json"
submission_lock="$workflow_root/submission.lock"
if [[ -e "$submission_receipt" || -d "$submission_lock" ]]; then
    echo "A submission receipt or lock already exists; refusing duplicate campaign submission." >&2
    exit 2
fi
if ! mkdir "$submission_lock"; then
    echo "Could not acquire the atomic submission lock: $submission_lock" >&2
    exit 2
fi
if [[ ! -f "$ledger" ]]; then
    printf 'timestamp_utc\tstage\tjob_id\tdependency\n' > "$ledger"
fi

parse_job_id() {
    local raw="$1"
    local job_id="${raw%%;*}"
    if ! [[ "$job_id" =~ ^[0-9]+$ ]]; then
        echo "Could not parse Slurm job ID from: $raw" >&2
        exit 2
    fi
    printf '%s\n' "$job_id"
}

record_job() {
    local stage="$1"
    local job_id="$2"
    local dependency="$3"
    printf '%s\t%s\t%s\t%s\n' \
        "$(date --iso-8601=seconds --utc)" "$stage" "$job_id" "$dependency" >> "$ledger"
}

common_export="ALL,PIPELINE_DIR=$PIPELINE_DIR,SEGDET_OUTPUT_ROOT=$OUTPUT_ROOT,SEGDET_SOURCE_ROOT=$SOURCE_ROOT,SEGDET_MAX_RETRIES=$MAX_RETRIES"

raw="$($SBATCH_BIN --parsable --partition="$PARTITION" --open-mode=append \
    --output="$log_root/module_preflight/simnibs326-%j.out" \
    --export="$common_export" "$HPC_DIR/module_preflight_326.slurm")"
preflight326="$(parse_job_id "$raw")"
record_job module-preflight-326 "$preflight326" ""

raw="$($SBATCH_BIN --parsable --partition="$PARTITION" --open-mode=append \
    --output="$log_root/module_preflight/simnibs401-%j.out" \
    --export="$common_export" "$HPC_DIR/module_preflight_401.slurm")"
preflight401="$(parse_job_id "$raw")"
record_job module-preflight-401 "$preflight401" ""

raw="$($SBATCH_BIN --parsable --partition="$PARTITION" --cpus-per-task="$CPUS" \
    --mem="$MEMORY" --time="$TIME_LIMIT" --array="0-9%$MAX_CONCURRENT" \
    --dependency="afterok:$preflight326" --open-mode=append \
    --output="$log_root/simnibs326_headreco/slurm-%A_%a.out" \
    --export="$common_export" "$HPC_DIR/headreco_array.slurm")"
array326="$(parse_job_id "$raw")"
record_job simnibs326-headreco-array "$array326" "afterok:$preflight326"

raw="$($SBATCH_BIN --parsable --partition="$PARTITION" --cpus-per-task="$CPUS" \
    --mem="$MEMORY" --time="$TIME_LIMIT" --array="0-9%$MAX_CONCURRENT" \
    --dependency="afterok:$preflight401" --open-mode=append \
    --output="$log_root/simnibs401_charm/slurm-%A_%a.out" \
    --export="$common_export" "$HPC_DIR/charm_array.slurm")"
array401="$(parse_job_id "$raw")"
record_job simnibs401-charm-array "$array401" "afterok:$preflight401"

raw="$($SBATCH_BIN --parsable --partition="$PARTITION" --cpus-per-task=4 \
    --mem=16G --time=02:00:00 --dependency="afterany:$array326:$array401" \
    --open-mode=append --output="$log_root/collector/collect-%j.out" \
    --export="$common_export,SEGDET_GIT_COMMIT=$(git -C "$PIPELINE_DIR/../.." rev-parse HEAD)" \
    "$HPC_DIR/collect.slurm")"
collector="$(parse_job_id "$raw")"
record_job collector "$collector" "afterany:$array326:$array401"

export SEGDET_SUBMISSION_PREFLIGHT326="$preflight326"
export SEGDET_SUBMISSION_PREFLIGHT401="$preflight401"
export SEGDET_SUBMISSION_ARRAY326="$array326"
export SEGDET_SUBMISSION_ARRAY401="$array401"
export SEGDET_SUBMISSION_COLLECTOR="$collector"
export SEGDET_SUBMISSION_LEDGER="$ledger"
"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
from simnibs_segmentation_determinism.provenance import write_json_atomic, write_sha256_sidecar

output = Path(os.environ["SEGDET_OUTPUT_ROOT"]) / "_pipeline/workflow/submission.json"
payload = {
    "schema_version": 1,
    "status": "submitted",
    "source_outputs_modified": False,
    "jobs": {
        "module_preflight_326": os.environ["SEGDET_SUBMISSION_PREFLIGHT326"],
        "module_preflight_401": os.environ["SEGDET_SUBMISSION_PREFLIGHT401"],
        "simnibs326_headreco_array": os.environ["SEGDET_SUBMISSION_ARRAY326"],
        "simnibs401_charm_array": os.environ["SEGDET_SUBMISSION_ARRAY401"],
        "collector": os.environ["SEGDET_SUBMISSION_COLLECTOR"],
    },
    "job_ids_tsv": os.environ["SEGDET_SUBMISSION_LEDGER"],
}
write_json_atomic(output, payload)
write_sha256_sidecar(output)
PY
rmdir "$submission_lock"

printf '[INFO] Segmentation determinism jobs: v3-preflight=%s v4-preflight=%s v3-array=%s v4-array=%s collector=%s\n' \
    "$preflight326" "$preflight401" "$array326" "$array401" "$collector"
