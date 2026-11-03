#!/bin/bash
set -euo pipefail

PACKAGE_DIR="${PACKAGE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPTS_ROOT="$(dirname "$PACKAGE_DIR")"
export PYTHONPATH="$SCRIPTS_ROOT:${PYTHONPATH:-}"

if [ -n "${PYTHON_BIN:-}" ]; then
    : # Use the explicitly supplied executable.
elif [ -x "$HOME/.conda/envs/ti-post/bin/python" ]; then
    PYTHON_BIN="$HOME/.conda/envs/ti-post/bin/python"
else
    PYTHON_BIN=python3
fi
ANALYSIS_PYTHON="${ANALYSIS_PYTHON:-$PYTHON_BIN}"

MODE="${1:-scope}"
[ "$#" -eq 0 ] || shift
DATASET_ROOT=""
OUTPUT_OVERRIDE=""
for argument in "$@"; do
    case "$argument" in
        --dataset-root=*) DATASET_ROOT="${argument#*=}" ;;
        --output-root=*) OUTPUT_OVERRIDE="${argument#*=}" ;;
        *) echo "[ERROR] Unknown argument: $argument" >&2; exit 2 ;;
    esac
done

case "$MODE" in
    scope|preflight|run|status|rebind-preflight) ;;
    *) echo "[ERROR] Unsupported mode: $MODE" >&2; exit 2 ;;
esac

[ -x "$ANALYSIS_PYTHON" ] || {
    echo "[ERROR] Analysis Python is not executable: $ANALYSIS_PYTHON" >&2
    exit 2
}
"$ANALYSIS_PYTHON" -c 'import matplotlib, numpy, pandas, scipy'

OUTPUT_ROOT=$("$PYTHON_BIN" -c \
    'from repeatability_combined_analysis.settings import production_config; print(production_config().output_root)')
if [ -z "$OUTPUT_ROOT" ] || [[ "$OUTPUT_ROOT" != /* ]]; then
    echo "[ERROR] Could not resolve the code-owned output root." >&2
    exit 2
fi
if [ -n "$OUTPUT_OVERRIDE" ] && [ "$OUTPUT_OVERRIDE" != "$OUTPUT_ROOT" ]; then
    echo "[ERROR] Refusing output root other than the code-owned output root: $OUTPUT_ROOT" >&2
    exit 2
fi

print_scope() {
    printf '%s\n' \
        'Scope:' \
        '  workflow: final-132 combined repeatability analysis' \
        "  output root: $OUTPUT_ROOT" \
        '  subjects: 10' \
        '  versions: 2' \
        '  targets: 2' \
        '  conditions: 2' \
        '  repeats per cell: 40' \
        '  participant cells: 80' \
        '  run-level rows: 3200' \
        '  meshing tasks: 0' \
        '  FEM simulation tasks: 0' \
        '  source campaigns: read-only' \
        '  output: isolated analysis root' \
        '  resources: sheffield; 4 CPU; 16G; 02:00:00' \
        '  jobs submitted by run: 1' \
        '  submission default: none'
}

run_cli() {
    "$PYTHON_BIN" -m repeatability_combined_analysis "$@"
}

print_scope

case "$MODE" in
    scope)
        run_cli scope
        exit 0
        ;;
    preflight)
        run_cli preflight
        exit 0
        ;;
    rebind-preflight)
        [ -n "$DATASET_ROOT" ] || {
            echo "[ERROR] rebind-preflight requires --dataset-root=<completed dataset>." >&2
            exit 2
        }
        run_cli rebind-preflight "--dataset-root=$DATASET_ROOT"
        exit 0
        ;;
    status)
        SUBMISSION="$OUTPUT_ROOT/_workflow/submission.json"
        if [ -f "$SUBMISSION" ] && [ ! -L "$SUBMISSION" ]; then
            JOB_ID=$(awk -F'"' '/"job_id"/{print $4; exit}' "$SUBMISSION")
            if [[ "$JOB_ID" =~ ^[0-9]+$ ]] && command -v sacct >/dev/null 2>&1; then
                printf '\nSLURM ACCOUNTING\n'
                sacct -D -X -j "$JOB_ID" \
                    --format=JobIDRaw,JobName%28,State,ExitCode,Elapsed,Start,End \
                    --parsable2
            fi
        fi
        printf '\nPACKAGE STATUS\n'
        run_cli status
        exit 0
        ;;
esac

run_cli preflight
run_cli prepare "--confirm-output-root=$OUTPUT_ROOT"

PLAN_FILE="$OUTPUT_ROOT/_workflow/input_plan.json"
PLAN_SHA256=$(awk -F'"' '/"plan_sha256"/{print $4; exit}' "$PLAN_FILE")
[[ "$PLAN_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
    echo "[ERROR] Prepared input plan lacks a valid SHA-256." >&2
    exit 2
}
REPOSITORY_COMMIT=$(git -C "$SCRIPTS_ROOT" rev-parse HEAD)
[[ "$REPOSITORY_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
    echo "[ERROR] Could not capture the repository commit." >&2
    exit 2
}

WORKFLOW_DIR="$OUTPUT_ROOT/_workflow"
LOG_DIR="$WORKFLOW_DIR/logs"
mkdir -p "$LOG_DIR"
SBATCH_RESULT=$(sbatch --parsable \
    --output="$LOG_DIR/analysis-%j.out" \
    --export="ALL,PACKAGE_DIR=$PACKAGE_DIR,OUTPUT_ROOT=$OUTPUT_ROOT,ANALYSIS_PYTHON=$ANALYSIS_PYTHON,PREPARED_PLAN_SHA256=$PLAN_SHA256,REPOSITORY_COMMIT=$REPOSITORY_COMMIT" \
    "$PACKAGE_DIR/hpc/analysis.slurm")
JOB_ID="${SBATCH_RESULT%%;*}"
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || {
    echo "[ERROR] Could not parse numeric job ID from: $SBATCH_RESULT" >&2
    exit 2
}

SUBMISSION="$WORKFLOW_DIR/submission.json"
TEMP_SUBMISSION="$WORKFLOW_DIR/.submission.json.tmp-$$"
CREATED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '{\n  "schema_version": 1,\n  "status": "submitted",\n  "job_id": "%s",\n  "created_utc": "%s",\n  "output_root": "%s",\n  "plan_sha256": "%s",\n  "git_commit": "%s",\n  "jobs_submitted": 1\n}\n' \
    "$JOB_ID" "$CREATED_UTC" "$OUTPUT_ROOT" "$PLAN_SHA256" "$REPOSITORY_COMMIT" \
    > "$TEMP_SUBMISSION"
mv "$TEMP_SUBMISSION" "$SUBMISSION"

echo "[INFO] Combined repeatability analysis job: $JOB_ID"
echo "[INFO] Submission receipt: $SUBMISSION"

