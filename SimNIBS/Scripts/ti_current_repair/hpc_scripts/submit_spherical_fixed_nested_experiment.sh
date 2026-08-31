#!/bin/bash
set -euo pipefail

PIPELINE_DIR_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_DIR="${PIPELINE_DIR:-$PIPELINE_DIR_DEFAULT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
PARTITION="${PARTITION:-sheffield}"

LEFT_SOURCE_ROOT="${LEFT_SOURCE_ROOT:-/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10}"
RIGHT_SOURCE_ROOT="${RIGHT_SOURCE_ROOT:-/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_right_m1}"
LEFT_METRICS_CSV="${LEFT_METRICS_CSV:-$LEFT_SOURCE_ROOT/_post_processing/repeatability_optimizer_roi_metrics_v1/optimizer_roi_metrics.csv}"
RIGHT_METRICS_CSV="${RIGHT_METRICS_CSV:-$RIGHT_SOURCE_ROOT/_post_processing/repeatability_optimizer_roi_metrics_v1/optimizer_roi_metrics.csv}"
LEFT_OUTPUT_ROOT="${LEFT_OUTPUT_ROOT:-/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_spherical_fixed_v1}"
RIGHT_OUTPUT_ROOT="${RIGHT_OUTPUT_ROOT:-/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_right_m1_spherical_fixed_v1}"
NESTED_OUTPUT_ROOT="${NESTED_OUTPUT_ROOT:-/mnt/parscratch/users/cop23bi/final_132_repeatability_nested_40x40_v1}"
SELECTION_SEED="${SELECTION_SEED:-20260831}"
NESTED_TARGET="${NESTED_TARGET:-left-hippocampus}"

SIM_CPUS="${SIM_CPUS:-8}"
SIM_MEMORY="${SIM_MEMORY:-32G}"
SIM_TIME="${SIM_TIME:-08:00:00}"
SIM_MAX_CONCURRENT="${SIM_MAX_CONCURRENT:-50}"
# Keep enough Stanage submitted-job QOS headroom for one dependent release
# controller and the small downstream jobs.  The scheduler's hard array cap is
# 1,000, but a 1,000-element array leaves no slot for its continuation job.
MAX_ARRAY_TASKS="${MAX_ARRAY_TASKS:-875}"
ROI_CPUS="${ROI_CPUS:-4}"
ROI_MEMORY="${ROI_MEMORY:-16G}"
ROI_TIME="${ROI_TIME:-02:00:00}"
ROI_MAX_CONCURRENT="${ROI_MAX_CONCURRENT:-10}"
RESUBMIT="${RESUBMIT:-0}"

WORKFLOW="$PIPELINE_DIR/pipeline/spherical_fixed_nested_experiment.py"
SIM_ARRAY="$PIPELINE_DIR/hpc_scripts/repeatability_experiment_array.slurm"
FINALIZER="$PIPELINE_DIR/hpc_scripts/spherical_fixed_nested_finalize.slurm"
ROI_ARRAY="$PIPELINE_DIR/hpc_scripts/repeatability_optimizer_roi_metrics_array.slurm"
ROI_COLLECT="$PIPELINE_DIR/hpc_scripts/repeatability_optimizer_roi_metrics_collect.slurm"
ROI_EXTRACTOR="$PIPELINE_DIR/post/extract_repeatability_optimizer_roi_metrics.py"
NESTED_ANALYSIS="$PIPELINE_DIR/hpc_scripts/nested_repeatability_analysis.slurm"
RELEASE_CONTROLLER="$PIPELINE_DIR/hpc_scripts/spherical_fixed_nested_release.slurm"

MODE="all"
PREFLIGHT_ONLY=0
PREPARE_ONLY=0
ATTACH_CONTINUATION=0
INTERNAL_RELEASE=0
CONTINUE_OFFSET=""
PREVIOUS_JOB=""
for argument in "$@"; do
    case "$argument" in
        all|fixed|nested)
            MODE="$argument"
            ;;
        --preflight)
            PREFLIGHT_ONLY=1
            ;;
        --prepare-only)
            PREPARE_ONLY=1
            ;;
        --attach-continuation)
            ATTACH_CONTINUATION=1
            ;;
        --internal-release)
            INTERNAL_RELEASE=1
            ;;
        --continue-offset=*)
            CONTINUE_OFFSET="${argument#*=}"
            ;;
        --previous-job=*)
            PREVIOUS_JOB="${argument#*=}"
            ;;
        *)
            echo "[ERROR] Unknown argument: $argument" >&2
            echo "Usage: $0 [all|fixed|nested] [--preflight|--prepare-only]" >&2
            exit 2
            ;;
    esac
done

if [ "$PREFLIGHT_ONLY" = "1" ] && [ "$PREPARE_ONLY" = "1" ]; then
    echo "[ERROR] Use only one of --preflight or --prepare-only." >&2
    exit 2
fi
if [ "$ATTACH_CONTINUATION" = "1" ] && [ "$INTERNAL_RELEASE" = "1" ]; then
    echo "[ERROR] --attach-continuation and --internal-release are mutually exclusive." >&2
    exit 2
fi
if [ "$ATTACH_CONTINUATION" = "1" ] || [ "$INTERNAL_RELEASE" = "1" ]; then
    if [ "$MODE" != "nested" ]; then
        echo "[ERROR] Continuation modes are supported only for mode 'nested'." >&2
        exit 2
    fi
    if [ "$PREFLIGHT_ONLY" = "1" ] || [ "$PREPARE_ONLY" = "1" ]; then
        echo "[ERROR] Continuation modes cannot be combined with preflight/prepare-only." >&2
        exit 2
    fi
    if ! [[ "$CONTINUE_OFFSET" =~ ^[1-9][0-9]*$ ]] || [ "$CONTINUE_OFFSET" -ge 1600 ]; then
        echo "[ERROR] --continue-offset must be an integer in 1-1599." >&2
        exit 2
    fi
    if ! [[ "$PREVIOUS_JOB" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] --previous-job must be a numeric Slurm job ID." >&2
        exit 2
    fi
fi
if ! [[ "$SIM_MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] SIM_MAX_CONCURRENT must be a positive integer." >&2
    exit 2
fi
if ! [[ "$MAX_ARRAY_TASKS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] MAX_ARRAY_TASKS must be a positive integer." >&2
    exit 2
fi
if [ "$MAX_ARRAY_TASKS" -gt 1000 ]; then
    echo "[ERROR] MAX_ARRAY_TASKS cannot exceed Stanage's 1,000-task array cap." >&2
    exit 2
fi
if { [ "$MODE" = "all" ] || [ "$MODE" = "nested" ]; } && [ "$MAX_ARRAY_TASKS" -gt 875 ]; then
    echo "[ERROR] Nested MAX_ARRAY_TASKS cannot exceed the QOS-safe 875-task chunk limit." >&2
    echo "[INFO] Stanage counts submitted array elements toward QOSMaxSubmitJobPerUserLimit; headroom is required for the release controller." >&2
    exit 2
fi
if ! [[ "$ROI_MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] ROI_MAX_CONCURRENT must be a positive integer." >&2
    exit 2
fi
if ! [[ "$SELECTION_SEED" =~ ^-?[0-9]+$ ]]; then
    echo "[ERROR] SELECTION_SEED must be an integer." >&2
    exit 2
fi
if [ "$NESTED_TARGET" != "left-hippocampus" ] && [ "$NESTED_TARGET" != "right-m1" ]; then
    echo "[ERROR] NESTED_TARGET must be left-hippocampus or right-m1." >&2
    exit 2
fi
if [ "$MODE" = "all" ] && [ "$PREFLIGHT_ONLY" = "0" ] && [ "$PREPARE_ONLY" = "0" ]; then
    echo "[ERROR] Production submission must be split into separate fixed and nested commands." >&2
    echo "[INFO] Run mode 'fixed' and mode 'nested' separately; mode 'all' is only for --preflight or --prepare-only." >&2
    exit 2
fi

for required in \
    "$WORKFLOW" \
    "$SIM_ARRAY" \
    "$FINALIZER" \
    "$ROI_ARRAY" \
    "$ROI_COLLECT" \
    "$ROI_EXTRACTOR" \
    "$NESTED_ANALYSIS" \
    "$RELEASE_CONTROLLER"; do
    if [ ! -f "$required" ]; then
        echo "[ERROR] Required workflow file is missing: $required" >&2
        exit 2
    fi
done

describe_array_chunks() {
    local label="$1"
    local task_count="$2"
    local global_base="${3:-0}"
    local task_offset=0
    local chunk_index=1
    local chunk_count=$(((task_count + MAX_ARRAY_TASKS - 1) / MAX_ARRAY_TASKS))

    echo "  ${label} array chunks: ${chunk_count} sequential"
    while [ "$task_offset" -lt "$task_count" ]; do
        local remaining=$((task_count - task_offset))
        local chunk_tasks="$MAX_ARRAY_TASKS"
        if [ "$remaining" -lt "$chunk_tasks" ]; then
            chunk_tasks="$remaining"
        fi
        local global_start=$((global_base + task_offset))
        local global_end=$((global_start + chunk_tasks - 1))
        echo "    chunk ${chunk_index}: local 0-$((chunk_tasks - 1))%${SIM_MAX_CONCURRENT}; global ${global_start}-${global_end}"
        task_offset=$((task_offset + chunk_tasks))
        chunk_index=$((chunk_index + 1))
    done
}

echo "Scope:"
echo "  mode: $MODE"
if [ "$MODE" = "all" ] || [ "$MODE" = "fixed" ]; then
    echo "  spherical fixed correction targets: 2"
    echo "  correction subjects: 10 per target (20 participant-target cases)"
    echo "  correction condition: fixed_mesh only"
    echo "  correction repeats: 40 per case"
    echo "  correction simulations: 800"
    echo "  correction targets each use the following plan:"
    describe_array_chunks "correction" 400
    echo "  correction expected TI outputs: 800"
fi
if [ "$MODE" = "all" ] || [ "$MODE" = "nested" ]; then
    echo "  nested cases: 1 persisted random participant"
    echo "  nested prespecified target: $NESTED_TARGET"
    echo "  nested outer meshes: 40"
    echo "  nested repeats per mesh: 40"
    echo "  nested simulations: 1600"
    if [ "$ATTACH_CONTINUATION" = "1" ]; then
        echo "  completed/running chunk ownership: global 0-$((CONTINUE_OFFSET - 1)); job $PREVIOUS_JOB"
        echo "  remaining simulations after controller gate: $((1600 - CONTINUE_OFFSET))"
        echo "  action now: attach one afterok release controller; submit no simulation array"
    elif [ "$INTERNAL_RELEASE" = "1" ]; then
        echo "  validated predecessor: global 0-$((CONTINUE_OFFSET - 1)); job $PREVIOUS_JOB"
        echo "  remaining simulations: $((1600 - CONTINUE_OFFSET))"
        describe_array_chunks "remaining nested" "$((1600 - CONTINUE_OFFSET))" "$CONTINUE_OFFSET"
    else
        describe_array_chunks "nested" 1600
    fi
    echo "  nested expected TI outputs: 1600"
fi
if [ "$MODE" = "all" ]; then
    echo "  total new simulations: 2400"
    echo "  total expected TI outputs: 2400"
fi
echo "  execution: full requested scope; no smoke or reduced subset"
echo "  source remesh outputs: read-only"
echo "  retry policy: same persisted selection, skip complete tasks, unlimited task requeue"
echo "  scheduler hard array task cap: 1000"
echo "  campaign array chunk limit: $MAX_ARRAY_TASKS"
echo "  simulation chunks are sequential to preserve global concurrency: $SIM_MAX_CONCURRENT"
echo "Resources:"
echo "  simulation module: SimNIBS/4.0.1-foss-2023a"
echo "  simulation partition: $PARTITION"
echo "  simulation CPU/memory/time: $SIM_CPUS / $SIM_MEMORY / $SIM_TIME"
echo "  spherical metric CPU/memory/time: $ROI_CPUS / $ROI_MEMORY / $ROI_TIME"

fixed_command() {
    local command="$1"
    local source_root="$2"
    local metrics_csv="$3"
    local output_root="$4"
    "$PYTHON_BIN" "$WORKFLOW" "$command" \
        --source-experiment-root "$source_root" \
        --optimizer-metrics-csv "$metrics_csv" \
        --output-root "$output_root" \
        --repeat-count 40
}

nested_command() {
    local command="$1"
    "$PYTHON_BIN" "$WORKFLOW" "$command" \
        --left-source-root "$LEFT_SOURCE_ROOT" \
        --right-source-root "$RIGHT_SOURCE_ROOT" \
        --left-metrics-csv "$LEFT_METRICS_CSV" \
        --right-metrics-csv "$RIGHT_METRICS_CSV" \
        --output-root "$NESTED_OUTPUT_ROOT" \
        --selection-seed "$SELECTION_SEED" \
        --nested-target "$NESTED_TARGET" \
        --outer-repeat-count 40 \
        --inner-repeat-count 40
}

append_receipt_row() {
    local receipt="$1"
    local stage="$2"
    local job_id="$3"
    local dependency="$4"
    local lock_file="${receipt}.lock"
    (
        flock -x 9
        if awk -F '\t' -v stage="$stage" '$1 == stage { found = 1 } END { exit !found }' "$receipt"; then
            echo "[ERROR] Receipt already contains stage $stage: $receipt" >&2
            exit 2
        fi
        printf "%s\t%s\t%s\n" "$stage" "$job_id" "$dependency" >> "$receipt"
    ) 9>"$lock_file"
}

validate_continuation_receipt() {
    local receipt="$NESTED_OUTPUT_ROOT/_pipeline/submitted_job_ids.tsv"
    if [ ! -f "$receipt" ]; then
        echo "[ERROR] Nested submission receipt is missing: $receipt" >&2
        exit 2
    fi
    if ! awk -F '\t' -v job="$PREVIOUS_JOB" '
        $1 ~ /^simulation_chunk_[0-9]+$/ && $2 == job { found = 1 }
        END { exit !found }
    ' "$receipt"; then
        echo "[ERROR] Receipt does not identify $PREVIOUS_JOB as a submitted simulation chunk." >&2
        exit 2
    fi
    if awk -F '\t' '$1 == "finalize" || $1 == "optimizer_roi_extract" || $1 == "optimizer_roi_collect" || $1 == "nested_analysis" { found = 1 } END { exit !found }' "$receipt"; then
        echo "[ERROR] Downstream nested stages are already recorded; refusing continuation attachment." >&2
        exit 2
    fi
}

submit_release_controller() {
    local label="$1"
    local root="$2"
    local receipt="$3"
    local previous_job="$4"
    local continue_offset="$5"
    local release_number
    local release_stage
    local release_export
    local release_job_raw
    local release_job

    release_number=$(awk -F '\t' '$1 ~ /^release_controller_[0-9]+$/ { count++ } END { print count + 1 }' "$receipt")
    printf -v release_stage "release_controller_%03d" "$release_number"
    release_export="ALL,PIPELINE_DIR=$PIPELINE_DIR,LEFT_SOURCE_ROOT=$LEFT_SOURCE_ROOT,RIGHT_SOURCE_ROOT=$RIGHT_SOURCE_ROOT,LEFT_METRICS_CSV=$LEFT_METRICS_CSV,RIGHT_METRICS_CSV=$RIGHT_METRICS_CSV,NESTED_OUTPUT_ROOT=$NESTED_OUTPUT_ROOT,SELECTION_SEED=$SELECTION_SEED,NESTED_TARGET=$NESTED_TARGET,SIM_CPUS=$SIM_CPUS,SIM_MEMORY=$SIM_MEMORY,SIM_TIME=$SIM_TIME,SIM_MAX_CONCURRENT=$SIM_MAX_CONCURRENT,MAX_ARRAY_TASKS=$MAX_ARRAY_TASKS,ROI_CPUS=$ROI_CPUS,ROI_MEMORY=$ROI_MEMORY,ROI_TIME=$ROI_TIME,ROI_MAX_CONCURRENT=$ROI_MAX_CONCURRENT,SBATCH_BIN=$SBATCH_BIN,CONTINUE_OFFSET=$continue_offset,PREVIOUS_JOB=$previous_job"
    release_job_raw="$(
        "$SBATCH_BIN" \
            --parsable \
            --job-name="ti_${label}_release" \
            --partition="$PARTITION" \
            --cpus-per-task=1 \
            --mem=1G \
            --time=01:00:00 \
            --dependency="afterok:$previous_job" \
            --output="$root/_pipeline/logs/release-%j.out" \
            --error="$root/_pipeline/logs/release-%j.err" \
            --export="$release_export" \
            "$RELEASE_CONTROLLER"
    )"
    release_job="${release_job_raw%%;*}"
    if ! [[ "$release_job" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Could not parse release-controller job ID from: $release_job_raw" >&2
        exit 2
    fi
    append_receipt_row "$receipt" "$release_stage" "$release_job" "afterok:$previous_job"
    echo "[INFO] $label continuation controller: $release_job"
    echo "[INFO] $label continuation dependency: afterok:$previous_job"
    echo "[INFO] $label continuation global offset: $continue_offset"
}

if [ "$ATTACH_CONTINUATION" = "1" ]; then
    validate_continuation_receipt
    receipt="$NESTED_OUTPUT_ROOT/_pipeline/submitted_job_ids.tsv"
    if awk -F '\t' '$1 ~ /^release_controller_[0-9]+$/ { found = 1 } END { exit !found }' "$receipt"; then
        echo "[ERROR] A nested release controller is already recorded: $receipt" >&2
        exit 2
    fi
    submit_release_controller \
        "nested_40x40" \
        "$NESTED_OUTPUT_ROOT" \
        "$receipt" \
        "$PREVIOUS_JOB" \
        "$CONTINUE_OFFSET"
    echo "[INFO] Continuation attached; existing simulation work was not resubmitted."
    exit 0
fi

if [ "$INTERNAL_RELEASE" = "1" ]; then
    validate_continuation_receipt
fi

if [ "$INTERNAL_RELEASE" = "0" ] && { [ "$MODE" = "all" ] || [ "$MODE" = "fixed" ]; }; then
    echo "[INFO] Preflight: left-hippocampus spherical-median fixed correction"
    fixed_command preflight-fixed "$LEFT_SOURCE_ROOT" "$LEFT_METRICS_CSV" "$LEFT_OUTPUT_ROOT"
    echo "[INFO] Preflight: right-M1 spherical-median fixed correction"
    fixed_command preflight-fixed "$RIGHT_SOURCE_ROOT" "$RIGHT_METRICS_CSV" "$RIGHT_OUTPUT_ROOT"
fi
if [ "$INTERNAL_RELEASE" = "0" ] && { [ "$MODE" = "all" ] || [ "$MODE" = "nested" ]; }; then
    echo "[INFO] Preflight: persisted random 40x40 nested case"
    nested_command preflight-nested
fi

if [ "$PREFLIGHT_ONLY" = "1" ]; then
    echo "[INFO] Full-scope preflight passed; no files created and no jobs submitted."
    exit 0
fi

if [ "$INTERNAL_RELEASE" = "0" ] && { [ "$MODE" = "all" ] || [ "$MODE" = "fixed" ]; }; then
    echo "[INFO] Preparing isolated left-hippocampus correction root"
    fixed_command prepare-fixed "$LEFT_SOURCE_ROOT" "$LEFT_METRICS_CSV" "$LEFT_OUTPUT_ROOT"
    echo "[INFO] Preparing isolated right-M1 correction root"
    fixed_command prepare-fixed "$RIGHT_SOURCE_ROOT" "$RIGHT_METRICS_CSV" "$RIGHT_OUTPUT_ROOT"
fi
if [ "$INTERNAL_RELEASE" = "0" ] && { [ "$MODE" = "all" ] || [ "$MODE" = "nested" ]; }; then
    echo "[INFO] Persisting/reusing the nested case and preparing 40 mesh caches"
    nested_command prepare-nested
fi

if [ "$PREPARE_ONLY" = "1" ]; then
    echo "[INFO] Preparation completed; no jobs submitted."
    exit 0
fi

check_submission_receipt() {
    local root="$1"
    local receipt="$root/_pipeline/submitted_job_ids.tsv"
    if [ -f "$receipt" ] && [ "$RESUBMIT" != "1" ]; then
        echo "[ERROR] Submission receipt already exists: $receipt" >&2
        echo "[INFO] Inspect active jobs first. Set RESUBMIT=1 only for a deliberate resumable relaunch." >&2
        exit 2
    fi
}

if [ "$INTERNAL_RELEASE" = "0" ] && { [ "$MODE" = "all" ] || [ "$MODE" = "fixed" ]; }; then
    check_submission_receipt "$LEFT_OUTPUT_ROOT"
    check_submission_receipt "$RIGHT_OUTPUT_ROOT"
fi
if [ "$INTERNAL_RELEASE" = "0" ] && { [ "$MODE" = "all" ] || [ "$MODE" = "nested" ]; }; then
    check_submission_receipt "$NESTED_OUTPUT_ROOT"
fi

submit_study() {
    local label="$1"
    local root="$2"
    local config="$3"
    local simulation_tasks="$4"
    local subjects="$5"
    local is_nested="$6"
    local start_offset="${7:-0}"
    local append_receipt="${8:-0}"
    local receipt="$root/_pipeline/submitted_job_ids.tsv"
    local log_root="$root/_pipeline/logs"
    local optimizer_root="$root/_post_processing/optimizer_roi_metrics_v1"
    local optimizer_archive="$optimizer_root/${label}_optimizer_roi_metrics_v1.tar.gz"
    local sim_export
    local sim_job
    local sim_job_raw
    local sim_job_name
    local sim_dependency=""
    local sim_stage
    local simulation_chunk_count=$(((simulation_tasks + MAX_ARRAY_TASKS - 1) / MAX_ARRAY_TASKS))
    local simulation_chunk_index=1
    local task_offset="$start_offset"
    local finalize_export
    local finalize_job
    local roi_export
    local roi_job
    local collect_job
    local analysis_job=""

    mkdir -p "$log_root" "$optimizer_root/logs"
    if [ "$append_receipt" = "1" ]; then
        if [ ! -f "$receipt" ]; then
            echo "[ERROR] Cannot continue without the existing receipt: $receipt" >&2
            exit 2
        fi
        simulation_chunk_index=$(awk -F '\t' '$1 ~ /^simulation_chunk_[0-9]+$/ || $1 == "simulation" { count++ } END { print count + 1 }' "$receipt")
    else
        if [ -f "$receipt" ] && [ "$RESUBMIT" != "1" ]; then
            echo "[ERROR] Submission receipt already exists: $receipt" >&2
            echo "[INFO] Inspect active jobs first. Set RESUBMIT=1 only for a deliberate resumable relaunch." >&2
            exit 2
        fi
        if [ -f "$receipt" ]; then
            mv "$receipt" "${receipt%.tsv}.previous.$(date -u +%Y%m%dT%H%M%SZ).tsv"
        fi
        printf "stage\tjob_id\tdependency\n" > "$receipt"
    fi

    while [ "$task_offset" -lt "$simulation_tasks" ]; do
        local remaining=$((simulation_tasks - task_offset))
        local chunk_tasks="$MAX_ARRAY_TASKS"
        if [ "$remaining" -lt "$chunk_tasks" ]; then
            chunk_tasks="$remaining"
        fi
        local array_spec="0-$((chunk_tasks - 1))%$SIM_MAX_CONCURRENT"
        local global_end=$((task_offset + chunk_tasks - 1))
        local sim_export="ALL,PIPELINE_DIR=$PIPELINE_DIR,EXPERIMENT_CONFIG=$config,LOG_DIR=$log_root/simulation,TI_MESH_TIMEOUT_HOURS=4,TI_MESH_MAX_RETRIES=0,OVERWRITE_OUTPUT=0,FORCE_MESH=0,TASK_OFFSET=$task_offset"
        local sim_submit=(
            "$SBATCH_BIN"
            --parsable
            --partition="$PARTITION"
            --cpus-per-task="$SIM_CPUS"
            --mem="$SIM_MEMORY"
            --time="$SIM_TIME"
            --array="$array_spec"
            --output="$log_root/simulation-%A_%a.out"
            --error="$log_root/simulation-%A_%a.err"
            --export="$sim_export"
        )

        if [ "$simulation_chunk_count" -eq 1 ]; then
            sim_job_name="ti_${label}"
            sim_stage="simulation"
        else
            printf -v sim_job_name "ti_%s_c%03d" "$label" "$simulation_chunk_index"
            printf -v sim_stage "simulation_chunk_%03d" "$simulation_chunk_index"
        fi
        sim_submit+=(--job-name="$sim_job_name")
        if [ -n "$sim_dependency" ]; then
            sim_submit+=(--dependency="afterok:$sim_dependency")
        fi
        sim_submit+=("$SIM_ARRAY")

        sim_job_raw="$("${sim_submit[@]}")"
        sim_job="${sim_job_raw%%;*}"
        if ! [[ "$sim_job" =~ ^[0-9]+$ ]]; then
            echo "[ERROR] Could not parse simulation job ID from: $sim_job_raw" >&2
            exit 2
        fi
        if [ -n "$sim_dependency" ]; then
            printf "%s\t%s\tafterok:%s\n" "$sim_stage" "$sim_job" "$sim_dependency" >> "$receipt"
        else
            printf "%s\t%s\t\n" "$sim_stage" "$sim_job" >> "$receipt"
        fi
        echo "[INFO] $label simulation chunk ${simulation_chunk_index}/${simulation_chunk_count}: $sim_job (local $array_spec; global $task_offset-$global_end)"

        sim_dependency="$sim_job"
        task_offset=$((task_offset + chunk_tasks))
        simulation_chunk_index=$((simulation_chunk_index + 1))
        if [ "$task_offset" -lt "$simulation_tasks" ]; then
            submit_release_controller \
                "$label" \
                "$root" \
                "$receipt" \
                "$sim_job" \
                "$task_offset"
            echo "[INFO] $label submission log:   $receipt"
            echo "[INFO] Later chunks and downstream stages will be released automatically."
            return 0
        fi
    done

    finalize_export="ALL,PIPELINE_DIR=$PIPELINE_DIR,EXPERIMENT_CONFIG=$config"
    finalize_job="$(
        "$SBATCH_BIN" \
            --parsable \
            --job-name="ti_${label}_finalize" \
            --partition="$PARTITION" \
            --cpus-per-task=1 \
            --mem=8G \
            --time=01:00:00 \
            --dependency="afterok:$sim_job" \
            --output="$log_root/finalize-%j.out" \
            --error="$log_root/finalize-%j.err" \
            --export="$finalize_export" \
            "$FINALIZER"
    )"

    roi_export="ALL,PIPELINE_DIR=$PIPELINE_DIR,EXPERIMENT_CONFIG=$config,OUTPUT_ROOT=$optimizer_root,EXTRACTOR=$ROI_EXTRACTOR,ARCHIVE_PATH=$optimizer_archive"
    roi_job="$(
        "$SBATCH_BIN" \
            --parsable \
            --job-name="ti_${label}_roi" \
            --partition="$PARTITION" \
            --cpus-per-task="$ROI_CPUS" \
            --mem="$ROI_MEMORY" \
            --time="$ROI_TIME" \
            --array="0-$((subjects - 1))%$ROI_MAX_CONCURRENT" \
            --dependency="afterok:$finalize_job" \
            --output="$optimizer_root/logs/extract-%A_%a.out" \
            --error="$optimizer_root/logs/extract-%A_%a.err" \
            --export="$roi_export" \
            "$ROI_ARRAY"
    )"
    collect_job="$(
        "$SBATCH_BIN" \
            --parsable \
            --job-name="ti_${label}_collect" \
            --partition="$PARTITION" \
            --cpus-per-task="$ROI_CPUS" \
            --mem="$ROI_MEMORY" \
            --time=01:00:00 \
            --dependency="afterok:$roi_job" \
            --output="$optimizer_root/logs/collect-%j.out" \
            --error="$optimizer_root/logs/collect-%j.err" \
            --export="$roi_export" \
            "$ROI_COLLECT"
    )"

    if [ "$is_nested" = "1" ]; then
        local nested_output="$root/_analysis/nested_variance"
        local selection="$root/_pipeline/nested_case_selection.json"
        local analysis_export="ALL,PIPELINE_DIR=$PIPELINE_DIR,METRICS_CSV=$optimizer_root/optimizer_roi_metrics.csv,SELECTION_MANIFEST=$selection,OUTPUT_DIR=$nested_output"
        analysis_job="$(
            "$SBATCH_BIN" \
                --parsable \
                --job-name="ti_${label}_analysis" \
                --partition="$PARTITION" \
                --cpus-per-task=1 \
                --mem=8G \
                --time=01:00:00 \
                --dependency="afterok:$collect_job" \
                --output="$log_root/nested-analysis-%j.out" \
                --error="$log_root/nested-analysis-%j.err" \
                --export="$analysis_export" \
                "$NESTED_ANALYSIS"
        )"
    fi

    {
        printf "finalize\t%s\tafterok:%s\n" "$finalize_job" "$sim_job"
        printf "optimizer_roi_extract\t%s\tafterok:%s\n" "$roi_job" "$finalize_job"
        printf "optimizer_roi_collect\t%s\tafterok:%s\n" "$collect_job" "$roi_job"
        if [ -n "$analysis_job" ]; then
            printf "nested_analysis\t%s\tafterok:%s\n" "$analysis_job" "$collect_job"
        fi
    } >> "$receipt"

    echo "[INFO] $label final simulation chunk: $sim_job"
    echo "[INFO] $label finalizer:        $finalize_job"
    echo "[INFO] $label ROI extraction:   $roi_job"
    echo "[INFO] $label ROI collector:    $collect_job"
    if [ -n "$analysis_job" ]; then
        echo "[INFO] $label nested analysis:  $analysis_job"
    fi
    echo "[INFO] $label submission log:   $receipt"
}

if [ "$MODE" = "all" ] || [ "$MODE" = "fixed" ]; then
    submit_study \
        "spherical_fixed_lh" \
        "$LEFT_OUTPUT_ROOT" \
        "$LEFT_OUTPUT_ROOT/_pipeline/configs/fixed_mesh_spherical_median.json" \
        400 \
        10 \
        0 \
        0 \
        0
    submit_study \
        "spherical_fixed_rm1" \
        "$RIGHT_OUTPUT_ROOT" \
        "$RIGHT_OUTPUT_ROOT/_pipeline/configs/fixed_mesh_spherical_median.json" \
        400 \
        10 \
        0 \
        0 \
        0
fi
if [ "$MODE" = "all" ] || [ "$MODE" = "nested" ]; then
    if [ "$INTERNAL_RELEASE" = "1" ]; then
        submit_study \
            "nested_40x40" \
            "$NESTED_OUTPUT_ROOT" \
            "$NESTED_OUTPUT_ROOT/_pipeline/configs/nested_40x40.json" \
            1600 \
            1 \
            1 \
            "$CONTINUE_OFFSET" \
            1
    else
        submit_study \
            "nested_40x40" \
            "$NESTED_OUTPUT_ROOT" \
            "$NESTED_OUTPUT_ROOT/_pipeline/configs/nested_40x40.json" \
            1600 \
            1 \
            1 \
            0 \
            0
    fi
fi
