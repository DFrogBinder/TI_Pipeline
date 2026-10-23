#!/bin/bash
set -euo pipefail

PACKAGE_DIR="${PACKAGE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPTS_ROOT="$(dirname "$PACKAGE_DIR")"
export PYTHONPATH="$SCRIPTS_ROOT:${PYTHONPATH:-}"
if [ -n "${PYTHON_BIN:-}" ]; then
    PYTHON_BIN="$PYTHON_BIN"
elif [ -x "$HOME/.conda/envs/ti-post/bin/python" ]; then
    PYTHON_BIN="$HOME/.conda/envs/ti-post/bin/python"
else
    PYTHON_BIN=python3
fi
V3_SUBMIT="${V3_SUBMIT:-$SCRIPTS_ROOT/simnibs_326_corrected_label_repeatability/hpc/submit.sh}"

MODE="${1:-preflight}"
[ "$#" -eq 0 ] || shift
SUBJECT=""
REPLACES=""
LEGACY_V3_ROOT=""
FINAL132_SUBJECTS="${FINAL132_SUBJECTS:-$SCRIPTS_ROOT/CamCan_Experiment/cohort_pipeline/cohorts/final_132/subjects.txt}"
CORRECTED_V4_SCAFFOLD="${CORRECTED_V4_SCAFFOLD:-/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Scaffolds}"
ATLAS_ROOT="${ATLAS_ROOT:-/mnt/parscratch/users/cop23bi/ZIPs/atlases}"
TARGETS_CSV="${TARGETS_CSV:-$SCRIPTS_ROOT/utils/targets.csv}"
WORK_ROOT="${REPLACEMENT_WORK_ROOT:-/mnt/parscratch/users/cop23bi/simnibs_subject_replacements}"
MANIFEST_SHA256=""
CONFIRM_REVIEWED=0
DEPENDENCY=""

for argument in "$@"; do
    case "$argument" in
        --subject=*) SUBJECT="${argument#*=}" ;;
        --replaces=*) REPLACES="${argument#*=}" ;;
        --legacy-v3-root=*) LEGACY_V3_ROOT="${argument#*=}" ;;
        --final132-subjects=*) FINAL132_SUBJECTS="${argument#*=}" ;;
        --corrected-v4-scaffold=*) CORRECTED_V4_SCAFFOLD="${argument#*=}" ;;
        --atlas-root=*) ATLAS_ROOT="${argument#*=}" ;;
        --targets-csv=*) TARGETS_CSV="${argument#*=}" ;;
        --work-root=*) WORK_ROOT="${argument#*=}" ;;
        --manifest-sha256=*) MANIFEST_SHA256="${argument#*=}" ;;
        --dependency=*) DEPENDENCY="${argument#*=}" ;;
        --confirm-reviewed) CONFIRM_REVIEWED=1 ;;
        *) echo "[ERROR] Unknown argument: $argument" >&2; exit 2 ;;
    esac
done

case "$MODE" in
    preflight|prepare|v3-module-check|v3-stage-references|v3-headmodels|v3-collect-headmodels|v3-promote|v3-init|v3-left-remesh|v3-left-fixed|v3-right-remesh|v3-right-fixed|v3-analysis|v3-status|v4-init-left|v4-preflight-left|v4-submit-left|v4-init-right|v4-preflight-right|v4-submit-right|status) ;;
    *) echo "[ERROR] Unsupported mode: $MODE" >&2; exit 2 ;;
esac

missing=()
[ -n "$SUBJECT" ] || missing+=(--subject)
[ -n "$REPLACES" ] || missing+=(--replaces)
[ -n "$LEGACY_V3_ROOT" ] || missing+=(--legacy-v3-root)
if [ "${#missing[@]}" -ne 0 ]; then
    echo "[ERROR] Required options: ${missing[*]}" >&2
    exit 2
fi

COMMON=(
    "--subject=$SUBJECT"
    "--replaces=$REPLACES"
    "--legacy-v3-root=$LEGACY_V3_ROOT"
    "--final132-subjects=$FINAL132_SUBJECTS"
    "--corrected-v4-scaffold=$CORRECTED_V4_SCAFFOLD"
    "--atlas-root=$ATLAS_ROOT"
    "--targets-csv=$TARGETS_CSV"
    "--work-root=$WORK_ROOT"
)

run_cli() {
    local command="$1"
    shift
    "$PYTHON_BIN" -m simnibs_subject_replacement "$command" "${COMMON[@]}" "$@"
}

if [ "$MODE" = preflight ]; then
    run_cli preflight
    exit 0
fi
if [ "$MODE" = prepare ]; then
    run_cli prepare
    exit 0
fi

mapfile -t BINDINGS < <(run_cli bindings)
[ "${#BINDINGS[@]}" -eq 10 ] || {
    echo "[ERROR] Prepared manifest did not yield the expected path bindings." >&2
    exit 2
}
MANIFEST="${BINDINGS[0]}"
V4_CORRECTED_ROOT="${BINDINGS[1]}"
V3_HEADMODEL_ROOT="${BINDINGS[2]}"
V3_LEFT_ROOT="${BINDINGS[3]}"
V3_RIGHT_ROOT="${BINDINGS[4]}"
V4_LEFT_ROOT="${BINDINGS[5]}"
V4_RIGHT_ROOT="${BINDINGS[6]}"
LEGACY_V3_ROOT="${BINDINGS[7]}"
ATLAS_ROOT="${BINDINGS[8]}"
TARGETS_CSV="${BINDINGS[9]}"

export SIMNIBS326H_COHORT=external1
export SIMNIBS326H_EXTERNAL_MANIFEST="$MANIFEST"
export SIMNIBS326H_CORRECTED_ROOT="$V4_CORRECTED_ROOT"
export SIMNIBS326H_LEGACY_REFERENCE_ROOT="$LEGACY_V3_ROOT"
export SIMNIBS326H_ATLAS_ROOT="$ATLAS_ROOT"
export SIMNIBS326H_TARGETS_CSV="$TARGETS_CSV"
export SIMNIBS326H_HEADMODEL_ROOT="$V3_HEADMODEL_ROOT"
export SIMNIBS326H_LEFT_ROOT="$V3_LEFT_ROOT"
export SIMNIBS326H_RIGHT_ROOT="$V3_RIGHT_ROOT"

run_v3() {
    [ -x "$V3_SUBMIT" ] || {
        echo "[ERROR] V3 submitter is not executable: $V3_SUBMIT" >&2
        exit 2
    }
    "$V3_SUBMIT" "$@" --cohort=external1
}

dependency_args=()
[ -z "$DEPENDENCY" ] || dependency_args+=("--dependency=$DEPENDENCY")

case "$MODE" in
    v3-module-check) run_v3 module-check ;;
    v3-stage-references) run_v3 stage-references ;;
    v3-headmodels) run_v3 headmodels ;;
    v3-collect-headmodels) run_v3 collect-headmodels ;;
    v3-promote)
        [[ "$MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
            echo "[ERROR] v3-promote requires --manifest-sha256=<64 lowercase hex>." >&2
            exit 2
        }
        [ "$CONFIRM_REVIEWED" -eq 1 ] || {
            echo "[ERROR] v3-promote requires --confirm-reviewed." >&2
            exit 2
        }
        run_v3 promote "--confirm-root=$V3_HEADMODEL_ROOT" \
            "--manifest-sha256=$MANIFEST_SHA256" --confirm-reviewed
        ;;
    v3-init)
        run_cli require-promotion >/dev/null
        run_v3 init "--confirm-roots=$V3_LEFT_ROOT,$V3_RIGHT_ROOT"
        ;;
    v3-left-remesh)
        run_cli require-promotion >/dev/null
        run_v3 target-remesh left-hippocampus "${dependency_args[@]}"
        ;;
    v3-left-fixed)
        run_cli require-promotion >/dev/null
        run_v3 fixed left-hippocampus "${dependency_args[@]}"
        ;;
    v3-right-remesh)
        run_cli require-left --version=v3 >/dev/null
        run_v3 target-remesh right-m1 "${dependency_args[@]}"
        ;;
    v3-right-fixed)
        run_cli require-left --version=v3 >/dev/null
        run_v3 fixed right-m1 "${dependency_args[@]}"
        ;;
    v3-analysis)
        run_cli require-promotion >/dev/null
        run_v3 analysis
        ;;
    v3-status) run_v3 status ;;
    v4-init-left) run_cli v4-init --target=left-hippocampus ;;
    v4-preflight-left) run_cli v4-run --target=left-hippocampus --dry-run ;;
    v4-submit-left) run_cli v4-run --target=left-hippocampus ;;
    v4-init-right) run_cli v4-init --target=right-m1 ;;
    v4-preflight-right) run_cli v4-run --target=right-m1 --dry-run ;;
    v4-submit-right) run_cli v4-run --target=right-m1 ;;
    status) run_cli status ;;
esac
