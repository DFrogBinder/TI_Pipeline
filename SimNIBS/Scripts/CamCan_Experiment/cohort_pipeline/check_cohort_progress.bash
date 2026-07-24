#!/bin/bash

# Fast, read-only status report for a corrected-v4 cohort campaign.
#
# This deliberately avoids mesh hashing and NIfTI loading. Potentially slow
# Slurm queries and parallel-filesystem scans are individually time-bounded so
# the report remains suitable for an interactive login node.

set -uo pipefail

COHORT_ID="${1:-final_132}"
if ! [[ "${COHORT_ID}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "[ERROR] Invalid cohort ID: ${COHORT_ID}" >&2
    exit 2
fi

STUDY_ROOT="${STUDY_ROOT:-/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Four_ROI}"
SCAFFOLD_ROOT="${SCAFFOLD_ROOT:-/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Scaffolds}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${STUDY_ROOT}/campaigns/${COHORT_ID}}"
LOG_DIR="${LOG_DIR:-${CAMPAIGN_ROOT}/logs}"
SCAFFOLD_MANIFEST="${SCAFFOLD_MANIFEST:-${CAMPAIGN_ROOT}/scaffold_tasks.tsv}"
MESH_MANIFEST="${MESH_MANIFEST:-${CAMPAIGN_ROOT}/mesh_tasks.tsv}"
SIMULATION_MANIFEST="${SIMULATION_MANIFEST:-${CAMPAIGN_ROOT}/simulation_tasks.tsv}"
RELEASE_PLAN="${RELEASE_PLAN:-${CAMPAIGN_ROOT}/release_plan.tsv}"
RELEASE_STATE_DIR="${RELEASE_STATE_DIR:-${CAMPAIGN_ROOT}/release_state}"
JOB_ID_FILE="${JOB_ID_FILE:-${CAMPAIGN_ROOT}/submitted_job_ids.txt}"

COMMAND_TIMEOUT_SECONDS="${TI_COHORT_PROGRESS_COMMAND_TIMEOUT_SECONDS:-5}"
SCAN_TIMEOUT_SECONDS="${TI_COHORT_PROGRESS_SCAN_TIMEOUT_SECONDS:-4}"
START_SECONDS="${SECONDS}"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/cohort-progress.XXXXXX")"
trap 'rm -rf -- "${TMP_ROOT}"' EXIT

section() {
    printf '\n== %s ==\n' "$1"
}

manifest_rows() {
    local path="$1"
    if [ ! -f "${path}" ]; then
        printf 'missing'
        return
    fi
    awk 'NR > 1 && NF > 0 { count++ } END { print count + 0 }' "${path}"
}

bounded_find() {
    local root="$1"
    local output="$2"
    shift 2
    : > "${output}"
    if [ ! -d "${root}" ]; then
        printf 'missing'
        return
    fi
    timeout "${SCAN_TIMEOUT_SECONDS}s" \
        find "${root}" "$@" -print > "${output}" 2>"${output}.err"
    local rc=$?
    case "${rc}" in
        0)
            printf 'complete'
            ;;
        124)
            printf 'timeout_partial'
            ;;
        *)
            printf 'error_%s' "${rc}"
            ;;
    esac
}

count_lines() {
    local path="$1"
    if [ -f "${path}" ]; then
        wc -l < "${path}" | tr -d '[:space:]'
    else
        printf '0'
    fi
}

print_roi_counts() {
    local marker="$1"
    local paths_file="$2"
    if [ ! -s "${paths_file}" ]; then
        return
    fi
    awk -F/ -v marker="${marker}" '
    {
        for (index = 1; index <= NF; index++) {
            if ($index == marker && index + 2 <= NF) {
                key = $(index + 1) "/" $(index + 2)
                counts[key]++
                break
            }
        }
    }
    END {
        for (key in counts)
            print key, counts[key]
    }
    ' "${paths_file}" |
        sort |
        awk '{ printf "  %-32s %s\n", $1, $2 }'
}

echo "CAMCAN_COHORT_PROGRESS_V1"
printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
printf 'host=%s\n' "$(hostname -f 2>/dev/null || hostname)"
printf 'user=%s\n' "${USER:-unknown}"
printf 'cohort=%s\n' "${COHORT_ID}"
printf 'campaign_root=%s\n' "${CAMPAIGN_ROOT}"

section "SCOPE"
SCAFFOLD_EXPECTED="$(manifest_rows "${SCAFFOLD_MANIFEST}")"
MESH_EXPECTED="$(manifest_rows "${MESH_MANIFEST}")"
SIMULATION_EXPECTED="$(manifest_rows "${SIMULATION_MANIFEST}")"
RELEASE_EXPECTED="$(manifest_rows "${RELEASE_PLAN}")"
printf 'scaffolds_expected=%s\n' "${SCAFFOLD_EXPECTED}"
printf 'meshes_expected=%s\n' "${MESH_EXPECTED}"
printf 'simulations_expected=%s\n' "${SIMULATION_EXPECTED}"
printf 'release_steps_expected=%s\n' "${RELEASE_EXPECTED}"
if [ -f "${SCAFFOLD_MANIFEST}" ]; then
    awk -F '\t' '
    NR > 1 && NF > 0 { modes[$3]++ }
    END {
        for (mode in modes)
            printf "scaffold_mode_%s=%s\n", mode, modes[mode]
    }
    ' "${SCAFFOLD_MANIFEST}" | sort
fi

section "COMPLETED OUTPUT MARKERS"
SCAFFOLD_PATHS="${TMP_ROOT}/scaffolds.paths"
MESH_PATHS="${TMP_ROOT}/meshes.paths"
SIMULATION_PATHS="${TMP_ROOT}/simulations.paths"

SCAFFOLD_SCAN="$(
    bounded_find \
        "${SCAFFOLD_ROOT}/results" \
        "${SCAFFOLD_PATHS}" \
        -maxdepth 1 -type f -name 'sub-*.json'
)"
MESH_SCAN="$(
    bounded_find \
        "${STUDY_ROOT}/results/meshes" \
        "${MESH_PATHS}" \
        -type f -name 'sub-*.json'
)"
SIMULATION_SCAN="$(
    bounded_find \
        "${STUDY_ROOT}/results/simulations" \
        "${SIMULATION_PATHS}" \
        -type f -name 'sub-*.json'
)"

printf 'scaffolds_complete=%s scan=%s\n' \
    "$(count_lines "${SCAFFOLD_PATHS}")" "${SCAFFOLD_SCAN}"
printf 'meshes_complete=%s scan=%s\n' \
    "$(count_lines "${MESH_PATHS}")" "${MESH_SCAN}"
printf 'simulations_complete=%s scan=%s\n' \
    "$(count_lines "${SIMULATION_PATHS}")" "${SIMULATION_SCAN}"

if [ -s "${MESH_PATHS}" ]; then
    echo "mesh_markers_by_roi_repeat:"
    print_roi_counts "meshes" "${MESH_PATHS}"
fi
if [ -s "${SIMULATION_PATHS}" ]; then
    echo "simulation_markers_by_roi_repeat:"
    print_roi_counts "simulations" "${SIMULATION_PATHS}"
fi

section "RELEASE CHAIN"
if [ -f "${JOB_ID_FILE}" ]; then
    JOB_IDS="$(
        awk '/^[0-9]+$/ {
            values = values separator $1
            separator = ","
        }
        END { print values }' "${JOB_ID_FILE}"
    )"
    printf 'submitted_job_count=%s\n' \
        "$(awk '/^[0-9]+$/ { count++ } END { print count + 0 }' "${JOB_ID_FILE}")"
    printf 'submitted_job_ids=%s\n' "${JOB_IDS:-none}"
else
    JOB_IDS=""
    echo "submitted_job_count=0"
    echo "submitted_job_ids=missing"
fi

RELEASE_RECEIPTS="${TMP_ROOT}/release_receipts.paths"
RELEASE_SCAN="$(
    bounded_find \
        "${RELEASE_STATE_DIR}" \
        "${RELEASE_RECEIPTS}" \
        -maxdepth 1 -type f -name 'release_step_*.tsv'
)"
printf 'release_steps_recorded=%s scan=%s\n' \
    "$(count_lines "${RELEASE_RECEIPTS}")" "${RELEASE_SCAN}"

if [ -s "${RELEASE_STATE_DIR}/chain_complete.tsv" ]; then
    echo "chain_status=complete"
    sed 's/^/  /' "${RELEASE_STATE_DIR}/chain_complete.tsv"
else
    echo "chain_status=in_progress_or_stopped"
fi

section "LIVE SLURM"
SQUEUE_OUTPUT="${TMP_ROOT}/squeue.tsv"
if [ -z "${JOB_IDS}" ]; then
    echo "squeue_status=no_recorded_jobs"
elif ! command -v squeue >/dev/null 2>&1; then
    echo "squeue_status=command_unavailable"
else
    timeout "${COMMAND_TIMEOUT_SECONDS}s" \
        squeue -h -j "${JOB_IDS}" -o '%i|%j|%T|%M|%R|%S' \
        > "${SQUEUE_OUTPUT}" 2>"${SQUEUE_OUTPUT}.err"
    SQUEUE_RC=$?
    case "${SQUEUE_RC}" in
        0)
            echo "squeue_status=complete"
            ;;
        124)
            echo "squeue_status=timeout"
            ;;
        *)
            echo "squeue_status=error_${SQUEUE_RC}"
            sed 's/^/  /' "${SQUEUE_OUTPUT}.err"
            ;;
    esac
fi

if [ -s "${SQUEUE_OUTPUT}" ]; then
    printf 'active_queue_rows=%s\n' "$(count_lines "${SQUEUE_OUTPUT}")"
    echo "active_by_job_and_state:"
    awk -F '|' '
    { counts[$2 "|" $3]++ }
    END {
        for (key in counts)
            print key "|" counts[key]
    }
    ' "${SQUEUE_OUTPUT}" |
        sort |
        awk -F '|' '{ printf "  %-36s %-10s %s\n", $1, $2, $3 }'

    echo "pending_reasons:"
    awk -F '|' '
    $3 == "PENDING" { counts[$5]++ }
    END {
        for (reason in counts)
            print reason "|" counts[reason]
    }
    ' "${SQUEUE_OUTPUT}" |
        sort |
        awk -F '|' '{ printf "  %-36s %s\n", $1, $2 }'

    echo "active_sample_first_12:"
    head -n 12 "${SQUEUE_OUTPUT}" |
        awk -F '|' '{
            printf "  job=%s name=%s state=%s elapsed=%s where=%s\n",
                $1, $2, $3, $4, $5
        }'
else
    echo "active_queue_rows=0"
fi

section "SLURM ACCOUNTING"
SACCT_OUTPUT="${TMP_ROOT}/sacct.tsv"
if [ -z "${JOB_IDS}" ]; then
    echo "sacct_status=no_recorded_jobs"
elif ! command -v sacct >/dev/null 2>&1; then
    echo "sacct_status=command_unavailable"
else
    timeout "${COMMAND_TIMEOUT_SECONDS}s" \
        sacct -j "${JOB_IDS}" \
        --noheader \
        --parsable2 \
        --format=JobIDRaw,JobName%40,State%24,ExitCode,ElapsedRaw \
        > "${SACCT_OUTPUT}" 2>"${SACCT_OUTPUT}.err"
    SACCT_RC=$?
    case "${SACCT_RC}" in
        0)
            echo "sacct_status=complete"
            ;;
        124)
            echo "sacct_status=timeout"
            ;;
        *)
            echo "sacct_status=error_${SACCT_RC}"
            sed 's/^/  /' "${SACCT_OUTPUT}.err"
            ;;
    esac
fi

if [ -s "${SACCT_OUTPUT}" ]; then
    echo "accounting_by_job_and_state:"
    awk -F '|' '
    $1 !~ /\./ {
        state = $3
        sub(/[[:space:]].*$/, "", state)
        sub(/\+$/, "", state)
        counts[$2 "|" state]++
    }
    END {
        for (key in counts)
            print key "|" counts[key]
    }
    ' "${SACCT_OUTPUT}" |
        sort |
        awk -F '|' '{ printf "  %-36s %-14s %s\n", $1, $2, $3 }'

    BAD_ACCOUNTING="${TMP_ROOT}/bad_accounting.tsv"
    awk -F '|' '
    $1 !~ /\./ {
        state = $3
        sub(/[[:space:]].*$/, "", state)
        sub(/\+$/, "", state)
        is_bad = state ~ /^(FAILED|CANCELLED|OUT_OF_MEMORY|TIMEOUT|NODE_FAIL|BOOT_FAIL|DEADLINE|PREEMPTED|REVOKED)$/
        if (is_bad || (state == "COMPLETED" && $4 != "0:0"))
            print $1 "|" $2 "|" state "|" $4 "|" $5
    }
    ' "${SACCT_OUTPUT}" > "${BAD_ACCOUNTING}"
    TERMINAL_PROBLEM_COUNT="$(count_lines "${BAD_ACCOUNTING}")"
    printf 'terminal_problem_records=%s\n' "${TERMINAL_PROBLEM_COUNT}"
    if [ -s "${BAD_ACCOUNTING}" ]; then
        echo "terminal_problem_sample_first_12:"
        head -n 12 "${BAD_ACCOUNTING}" |
            awk -F '|' '{
                printf "  job=%s name=%s state=%s exit=%s elapsed_s=%s\n",
                    $1, $2, $3, $4, $5
            }'
    fi
fi

section "RETRIES"
RETRY_PATHS="${TMP_ROOT}/retries.paths"
RETRY_SCAN="$(
    bounded_find \
        "${LOG_DIR}/retry_state" \
        "${RETRY_PATHS}" \
        -maxdepth 1 -type f -name '*.retry'
)"
printf 'active_retry_files=%s scan=%s\n' \
    "$(count_lines "${RETRY_PATHS}")" "${RETRY_SCAN}"
if [ -s "${RETRY_PATHS}" ]; then
    echo "retry_sample_first_12:"
    head -n 12 "${RETRY_PATHS}" |
        while IFS= read -r retry_file; do
            printf '  %s=%s\n' \
                "$(basename "${retry_file}")" \
                "$(tr -d '[:space:]' < "${retry_file}" 2>/dev/null || echo unreadable)"
        done

    echo "retry_log_excerpts_first_3:"
    head -n 3 "${RETRY_PATHS}" |
        while IFS= read -r retry_file; do
            retry_name="$(basename "${retry_file}" .retry)"
            retry_stage="${retry_name%%_*}"
            retry_rest="${retry_name#*_}"
            retry_array="${retry_rest%%_*}"
            retry_task="${retry_rest##*_}"
            retry_log="$(
                find "${LOG_DIR}" \
                    -maxdepth 1 \
                    -type f \
                    -name "${retry_stage}__${retry_task}__*.log" \
                    -print -quit 2>/dev/null
            )"
            printf '  retry=%s stage=%s array=%s log=%s\n' \
                "${retry_name}" \
                "${retry_stage}" \
                "${retry_array}" \
                "${retry_log:-not_found}"
            if [ -n "${retry_log}" ] && [ -f "${retry_log}" ]; then
                tail -n 120 "${retry_log}" |
                    grep -E '\[(ERROR|WARN)\]|CRITICAL|Traceback|Exception|non-zero|No such file|Killed|OUT_OF_MEMORY' |
                    tail -n 12 |
                    sed 's/^/    /'
            fi
        done
fi

section "LATEST RELEASE MESSAGES"
LATEST_RELEASE_LOG="$(
    find "${LOG_DIR}" \
        -maxdepth 1 \
        -type f \
        -name 'release-*.out' \
        -printf '%T@\t%p\n' 2>/dev/null |
        sort -nr |
        head -n 1 |
        cut -f2-
)"
if [ -n "${LATEST_RELEASE_LOG}" ] && [ -f "${LATEST_RELEASE_LOG}" ]; then
    printf 'latest_release_log=%s\n' "${LATEST_RELEASE_LOG}"
    tail -n 80 "${LATEST_RELEASE_LOG}" |
        grep -E '\[(INFO|WARN|ERROR)\]|QOS|Submitted|complete' |
        tail -n 20 |
        sed 's/^/  /'
else
    echo "latest_release_log=none"
fi

section "REPORT"
printf 'report_elapsed_seconds=%s\n' "$((SECONDS - START_SECONDS))"
echo "report_read_only=true"
RETRY_COUNT="$(count_lines "${RETRY_PATHS}")"
if [ -s "${RELEASE_STATE_DIR}/chain_complete.tsv" ]; then
    echo "report_health=complete"
elif [ "${RETRY_COUNT}" -gt 0 ] || [ "${TERMINAL_PROBLEM_COUNT:-0}" -gt 0 ]; then
    echo "report_health=attention_required"
else
    echo "report_health=healthy_in_progress"
fi
echo "report_complete=true"
