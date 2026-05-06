#!/bin/bash
#SBATCH --job-name=copy_roi_runs
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=copy_roi_runs_%j.out
#SBATCH --error=copy_roi_runs_%j.err

set -euo pipefail
umask 0022

echo "[INFO] Host: $(hostname)"
echo "[INFO] JobID: ${SLURM_JOB_ID:-<none>}"
echo "[INFO] Start: $(date)"
echo

# ------------------------------------------------------------------
# User config
#
# SOURCE_ROOT:
#   Root folder containing the 10 runs for one ROI.
#
# DEST_PARENT:
#   Parent directory under which the full SOURCE_ROOT folder will be copied.
#   Example:
#     SOURCE_ROOT=/path/A/Left_Hippocampus_Post_Data
#     DEST_PARENT=/path/B
#   Result:
#     /path/B/Left_Hippocampus_Post_Data
#
# DEST_NAME:
#   Optional override for the copied top-level folder name.
#   Blank -> reuse basename(SOURCE_ROOT).
#
# LOG_ROOT:
#   Optional directory for the copy logs and verification reports.
#   Blank -> ${DEST_PARENT}/_copy_logs
#
# This script is designed for copies where both SOURCE_ROOT and DEST_PARENT
# already live on the HPC. It submits the copy as a Slurm job so the work
# happens on a worker node instead of the login node.
#
# You can either edit the defaults below or override them at submit time:
#
#   sbatch --export=ALL,\
#SOURCE_ROOT=/path/to/Left_Hippocampus_Post_Data,\
#DEST_PARENT=/path/to/new/location,\
#DEST_NAME=Left_Hippocampus_Post_Data_copy,\
#VERIFY_MODE=checksum,\
#PARALLEL_TOP_LEVEL_JOBS=8 \
#/path/to/copy_roi_runs_with_verification.slurm
# ------------------------------------------------------------------

PLACEHOLDER_SOURCE_ROOT="/path/to/ROI_root_with_all_10_runs"
PLACEHOLDER_DEST_PARENT="/path/to/destination_parent"

# Edit these if you want file-based defaults.
SOURCE_ROOT_DEFAULT="$PLACEHOLDER_SOURCE_ROOT"
DEST_PARENT_DEFAULT="$PLACEHOLDER_DEST_PARENT"
DEST_NAME_DEFAULT=""
LOG_ROOT_DEFAULT=""

# Copy / verification behavior
# VERIFY_MODE:
#   metadata -> fast rsync quick-check verification (size + mtime)
#   checksum -> full-content checksum verification
#   none     -> skip the post-copy rsync verification pass
VERIFY_MODE_DEFAULT="metadata"
GENERATE_SUBJECT_REPORT_DEFAULT="1"  # 1 -> per-run/per-subject presence + file-count report
ALLOW_EXISTING_DEST_DEFAULT="1"    # 1 -> allow merge into an existing destination root, still verify strictly
PARALLEL_TOP_LEVEL_JOBS_DEFAULT="4"  # parallel rsync jobs for immediate child run directories

SOURCE_ROOT="${SOURCE_ROOT:-$SOURCE_ROOT_DEFAULT}"
DEST_PARENT="${DEST_PARENT:-$DEST_PARENT_DEFAULT}"
DEST_NAME="${DEST_NAME:-$DEST_NAME_DEFAULT}"
LOG_ROOT="${LOG_ROOT:-$LOG_ROOT_DEFAULT}"
if [[ -n "${VERIFY_MODE:-}" ]]; then
    VERIFY_MODE="$VERIFY_MODE"
elif [[ -n "${VERIFY_WITH_CHECKSUM:-}" ]]; then
    case "$VERIFY_WITH_CHECKSUM" in
        1) VERIFY_MODE="checksum" ;;
        0) VERIFY_MODE="metadata" ;;
        *)
            echo "[ERROR] Legacy VERIFY_WITH_CHECKSUM must be 0 or 1, got: ${VERIFY_WITH_CHECKSUM}" >&2
            exit 2
            ;;
    esac
else
    VERIFY_MODE="$VERIFY_MODE_DEFAULT"
fi
GENERATE_SUBJECT_REPORT="${GENERATE_SUBJECT_REPORT:-$GENERATE_SUBJECT_REPORT_DEFAULT}"
ALLOW_EXISTING_DEST="${ALLOW_EXISTING_DEST:-$ALLOW_EXISTING_DEST_DEFAULT}"
PARALLEL_TOP_LEVEL_JOBS="${PARALLEL_TOP_LEVEL_JOBS:-$PARALLEL_TOP_LEVEL_JOBS_DEFAULT}"

module purge || true

require_cmd() {
    local name="$1"
    if ! command -v "$name" >/dev/null 2>&1; then
        echo "[ERROR] Required command not found: $name" >&2
        exit 2
    fi
}

require_cmd rsync
require_cmd find
require_cmd awk
require_cmd sort
require_cmd diff
require_cmd xargs

require_positive_int() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value < 1 )); then
        echo "[ERROR] ${name} must be a positive integer, got: ${value}" >&2
        exit 2
    fi
}

require_verify_mode() {
    local value="$1"
    case "$value" in
        metadata|checksum|none) ;;
        *)
            echo "[ERROR] VERIFY_MODE must be one of: metadata, checksum, none. Got: ${value}" >&2
            exit 2
            ;;
    esac
}

abs_existing_dir() {
    local path="$1"
    (
        cd "$path"
        pwd
    )
}

mkdir_and_abs_dir() {
    local path="$1"
    mkdir -p "$path"
    (
        cd "$path"
        pwd
    )
}

count_files() {
    find "$1" -type f | wc -l | awk '{print $1}'
}

count_dirs() {
    find "$1" -type d | wc -l | awk '{print $1}'
}

sum_file_bytes() {
    find "$1" -type f -printf '%s\n' | awk '{s += $1} END {print s + 0}'
}

write_file_manifest() {
    local root="$1"
    local out="$2"
    (
        cd "$root"
        find . -type f -printf '%P\t%s\n' | LC_ALL=C sort
    ) > "$out"
}

write_dir_manifest() {
    local root="$1"
    local out="$2"
    (
        cd "$root"
        find . -type d -printf '%P\n' | LC_ALL=C sort
    ) > "$out"
}

echo "[INFO] rsync: $(rsync --version | head -n 1 || true)"
echo

if [[ "$SOURCE_ROOT" == "$PLACEHOLDER_SOURCE_ROOT" ]]; then
    echo "[ERROR] SOURCE_ROOT is still the placeholder value." >&2
    echo "[ERROR] Edit SOURCE_ROOT_DEFAULT in this script or submit with --export=ALL,SOURCE_ROOT=/real/path,DEST_PARENT=/real/path" >&2
    exit 2
fi

if [[ "$DEST_PARENT" == "$PLACEHOLDER_DEST_PARENT" ]]; then
    echo "[ERROR] DEST_PARENT is still the placeholder value." >&2
    echo "[ERROR] Edit DEST_PARENT_DEFAULT in this script or submit with --export=ALL,SOURCE_ROOT=/real/path,DEST_PARENT=/real/path" >&2
    exit 2
fi

if [[ ! -d "$SOURCE_ROOT" ]]; then
    echo "[ERROR] SOURCE_ROOT does not exist: $SOURCE_ROOT" >&2
    exit 2
fi

require_positive_int "PARALLEL_TOP_LEVEL_JOBS" "$PARALLEL_TOP_LEVEL_JOBS"
require_verify_mode "$VERIFY_MODE"

SOURCE_ROOT="$(abs_existing_dir "$SOURCE_ROOT")"
SOURCE_BASENAME="$(basename "$SOURCE_ROOT")"
FINAL_DEST_NAME="${DEST_NAME:-$SOURCE_BASENAME}"
DEST_PARENT="$(mkdir_and_abs_dir "$DEST_PARENT")"
DEST_ROOT="${DEST_PARENT}/${FINAL_DEST_NAME}"

if [[ -n "$LOG_ROOT" ]]; then
    LOG_ROOT="$(mkdir_and_abs_dir "$LOG_ROOT")"
else
    LOG_ROOT="$(mkdir_and_abs_dir "${DEST_PARENT}/_copy_logs")"
fi

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="${LOG_ROOT}/${FINAL_DEST_NAME}_copy_${SLURM_JOB_ID:-manual}_${RUN_STAMP}"
mkdir -p "$RUN_LOG_DIR"

PRE_DRYRUN_LOG="${RUN_LOG_DIR}/rsync_preflight_dryrun.log"
COPY_LOG="${RUN_LOG_DIR}/rsync_copy.log"
VERIFY_LOG="${RUN_LOG_DIR}/rsync_verify.log"
SOURCE_FILES_MANIFEST="${RUN_LOG_DIR}/source_files_manifest.tsv"
DEST_FILES_MANIFEST="${RUN_LOG_DIR}/dest_files_manifest.tsv"
SOURCE_DIRS_MANIFEST="${RUN_LOG_DIR}/source_dirs_manifest.tsv"
DEST_DIRS_MANIFEST="${RUN_LOG_DIR}/dest_dirs_manifest.tsv"
MANIFEST_DIFF="${RUN_LOG_DIR}/manifest_diff.txt"
VERIFY_MISMATCHES="${RUN_LOG_DIR}/verify_mismatches.txt"
VERIFY_MISMATCH_SUMMARY="${RUN_LOG_DIR}/verify_mismatches_by_subject.tsv"
SUBJECT_REPORT="${RUN_LOG_DIR}/subject_copy_report.tsv"
SUMMARY_TXT="${RUN_LOG_DIR}/copy_summary.txt"

if [[ -e "$DEST_ROOT" && "$ALLOW_EXISTING_DEST" != "1" ]]; then
    echo "[ERROR] Destination already exists and ALLOW_EXISTING_DEST=0: $DEST_ROOT" >&2
    exit 2
fi

echo "[INFO] Source root:        $SOURCE_ROOT"
echo "[INFO] Destination parent: $DEST_PARENT"
echo "[INFO] Destination root:   $DEST_ROOT"
echo "[INFO] Log directory:      $RUN_LOG_DIR"
echo "[INFO] Verify mode:        $VERIFY_MODE"
echo "[INFO] Parallel jobs:      $PARALLEL_TOP_LEVEL_JOBS"
echo

SOURCE_FILE_COUNT="$(count_files "$SOURCE_ROOT")"
SOURCE_DIR_COUNT="$(count_dirs "$SOURCE_ROOT")"
SOURCE_TOTAL_BYTES="$(sum_file_bytes "$SOURCE_ROOT")"

echo "[INFO] Source file count:  $SOURCE_FILE_COUNT"
echo "[INFO] Source dir count:   $SOURCE_DIR_COUNT"
echo "[INFO] Source total bytes: $SOURCE_TOTAL_BYTES"
echo

write_file_manifest "$SOURCE_ROOT" "$SOURCE_FILES_MANIFEST"
write_dir_manifest "$SOURCE_ROOT" "$SOURCE_DIRS_MANIFEST"

echo "[INFO] Preflight rsync dry-run..."
rsync \
    -anH \
    --whole-file \
    --delete \
    --human-readable \
    --itemize-changes \
    --out-format='%i %n%L' \
    "$SOURCE_ROOT/" "$DEST_ROOT/" | tee "$PRE_DRYRUN_LOG"

echo
echo "[INFO] Copying source tree on worker-node storage path..."
mkdir -p "$DEST_ROOT"

TOP_LEVEL_DIRS=()
while IFS= read -r -d '' entry; do
    TOP_LEVEL_DIRS+=("$entry")
done < <(find "$SOURCE_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\0' | LC_ALL=C sort -z)

{
    echo "[INFO] Parallel copy started at $(date)"
    echo "[INFO] Source root: $SOURCE_ROOT"
    echo "[INFO] Destination root: $DEST_ROOT"
    echo "[INFO] Top-level directory count: ${#TOP_LEVEL_DIRS[@]}"
    echo "[INFO] Parallel jobs: $PARALLEL_TOP_LEVEL_JOBS"
} | tee "$COPY_LOG"

# Copy any files that live directly under the source root first.
rsync \
    -aH \
    --whole-file \
    --partial \
    --human-readable \
    --itemize-changes \
    --info=stats2 \
    --exclude='*/' \
    "$SOURCE_ROOT/" "$DEST_ROOT/" | tee -a "$COPY_LOG"

if (( ${#TOP_LEVEL_DIRS[@]} > 0 )); then
    printf '%s\0' "${TOP_LEVEL_DIRS[@]}" | \
        xargs -0 -I{} -P "$PARALLEL_TOP_LEVEL_JOBS" \
        bash -c '
            set -euo pipefail
            entry="$1"
            src_root="$2"
            dest_root="$3"
            log_dir="$4"
            entry_log="${log_dir}/rsync_${entry}.log"

            {
                echo "[INFO] START ${entry} $(date)"
                rsync \
                    -aH \
                    --whole-file \
                    --partial \
                    --human-readable \
                    --itemize-changes \
                    --info=stats2 \
                    "${src_root}/${entry}" "${dest_root}/"
                echo "[INFO] END ${entry} $(date)"
            } > "${entry_log}" 2>&1
        ' _ {} "$SOURCE_ROOT" "$DEST_ROOT" "$RUN_LOG_DIR"
fi

{
    echo "[INFO] Parallel copy finished at $(date)"
    echo "[INFO] Per-entry logs:"
    find "$RUN_LOG_DIR" -maxdepth 1 -type f -name 'rsync_*.log' -printf '  %f\n' | LC_ALL=C sort
} | tee -a "$COPY_LOG"

echo
if [[ ! -d "$DEST_ROOT" ]]; then
    echo "[ERROR] Destination root was not created: $DEST_ROOT" >&2
    exit 2
fi

DEST_FILE_COUNT="$(count_files "$DEST_ROOT")"
DEST_DIR_COUNT="$(count_dirs "$DEST_ROOT")"
DEST_TOTAL_BYTES="$(sum_file_bytes "$DEST_ROOT")"

echo "[INFO] Destination file count:  $DEST_FILE_COUNT"
echo "[INFO] Destination dir count:   $DEST_DIR_COUNT"
echo "[INFO] Destination total bytes: $DEST_TOTAL_BYTES"
echo

write_file_manifest "$DEST_ROOT" "$DEST_FILES_MANIFEST"
write_dir_manifest "$DEST_ROOT" "$DEST_DIRS_MANIFEST"

MANIFEST_OK="1"
if ! diff -u "$SOURCE_DIRS_MANIFEST" "$DEST_DIRS_MANIFEST" > "$MANIFEST_DIFF"; then
    MANIFEST_OK="0"
fi
if ! diff -u "$SOURCE_FILES_MANIFEST" "$DEST_FILES_MANIFEST" >> "$MANIFEST_DIFF"; then
    MANIFEST_OK="0"
fi

VERIFY_OK="1"
if [[ "$VERIFY_MODE" == "checksum" || "$VERIFY_MODE" == "metadata" ]]; then
    if [[ "$VERIFY_MODE" == "checksum" ]]; then
        echo "[INFO] Running checksum-based verification..."
        VERIFY_RSYNC_ARGS=(-anrcH)
    else
        echo "[INFO] Running metadata-based verification (size + mtime)..."
        VERIFY_RSYNC_ARGS=(-anrH)
    fi

    rsync \
        "${VERIFY_RSYNC_ARGS[@]}" \
        --delete \
        --itemize-changes \
        --out-format='%i %n%L' \
        "$SOURCE_ROOT/" "$DEST_ROOT/" | tee "$VERIFY_LOG"

    # Ignore a lone root-directory timestamp delta. It does not indicate a file-content
    # mismatch and can appear after a successful copy when only "./" metadata drifted.
    awk 'NF && $0 != ".d..t...... ./" { print }' "$VERIFY_LOG" > "$VERIFY_MISMATCHES"
    if [[ -s "$VERIFY_MISMATCHES" ]]; then
        VERIFY_OK="0"
        {
            printf "run\tsubject\tmismatch_entries\n"
            awk '
                {
                    path = $2
                    gsub(/^\.\//, "", path)
                    n = split(path, parts, "/")
                    run = (n >= 1 && parts[1] != "" ? parts[1] : "<root>")
                    subject = (n >= 2 && parts[2] != "" ? parts[2] : "<root>")
                    key = run "\t" subject
                    count[key] += 1
                }
                END {
                    for (k in count) {
                        print k "\t" count[k]
                    }
                }
            ' "$VERIFY_MISMATCHES" | LC_ALL=C sort
        } > "$VERIFY_MISMATCH_SUMMARY"
    else
        printf "run\tsubject\tmismatch_entries\n" > "$VERIFY_MISMATCH_SUMMARY"
    fi
else
    echo "[INFO] Post-copy rsync verification skipped (VERIFY_MODE=none)."
    printf "run\tsubject\tmismatch_entries\n" > "$VERIFY_MISMATCH_SUMMARY"
fi

if [[ "$GENERATE_SUBJECT_REPORT" == "1" ]]; then
    declare -A MISMATCH_KEYS=()
    if [[ -s "$VERIFY_MISMATCH_SUMMARY" ]]; then
        while IFS=$'\t' read -r run subject mismatch_entries; do
            if [[ "$run" == "run" ]]; then
                continue
            fi
            MISMATCH_KEYS["${run}|||${subject}"]="$mismatch_entries"
        done < "$VERIFY_MISMATCH_SUMMARY"
    fi

    {
        printf "run\tsubject\tsource_exists\tdest_exists\tsource_files\tdest_files\tstatus\tverify_mismatch_entries\n"
        while IFS= read -r -d '' subject_dir; do
            rel="${subject_dir#"$SOURCE_ROOT"/}"
            run="${rel%%/*}"
            subject="${rel#*/}"
            dest_subject_dir="${DEST_ROOT}/${run}/${subject}"
            source_files="$(count_files "$subject_dir")"
            dest_files="0"
            source_exists="1"
            dest_exists="0"
            status="MISSING_DEST"
            mismatch_entries="0"

            if [[ -d "$dest_subject_dir" ]]; then
                dest_exists="1"
                dest_files="$(count_files "$dest_subject_dir")"
                if [[ "$source_files" == "$dest_files" ]]; then
                    status="FILE_COUNT_MATCH"
                else
                    status="FILE_COUNT_MISMATCH"
                fi
            fi

            key="${run}|||${subject}"
            if [[ -n "${MISMATCH_KEYS[$key]:-}" && "${MISMATCH_KEYS[$key]}" != "0" ]]; then
                mismatch_entries="${MISMATCH_KEYS[$key]}"
                status="VERIFY_MISMATCH"
            fi

            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$run" "$subject" "$source_exists" "$dest_exists" "$source_files" "$dest_files" "$status" "$mismatch_entries"
        done < <(find "$SOURCE_ROOT" -mindepth 2 -maxdepth 2 -type d -print0 | LC_ALL=C sort -z)
    } > "$SUBJECT_REPORT"
fi

{
    echo "copy_completed_at=$(date)"
    echo "source_root=$SOURCE_ROOT"
    echo "destination_root=$DEST_ROOT"
    echo "log_directory=$RUN_LOG_DIR"
    echo "source_file_count=$SOURCE_FILE_COUNT"
    echo "destination_file_count=$DEST_FILE_COUNT"
    echo "source_dir_count=$SOURCE_DIR_COUNT"
    echo "destination_dir_count=$DEST_DIR_COUNT"
    echo "source_total_bytes=$SOURCE_TOTAL_BYTES"
    echo "destination_total_bytes=$DEST_TOTAL_BYTES"
    echo "manifest_ok=$MANIFEST_OK"
    echo "verify_mode=$VERIFY_MODE"
    echo "checksum_verify_enabled=$([[ "$VERIFY_MODE" == "checksum" ]] && echo 1 || echo 0)"
    echo "verify_ok=$VERIFY_OK"
    echo "subject_report=$SUBJECT_REPORT"
    echo "copy_log=$COPY_LOG"
    echo "verify_log=$VERIFY_LOG"
} | tee "$SUMMARY_TXT"

echo
echo "[INFO] Copy summary written to: $SUMMARY_TXT"
echo "[INFO] Subject report:           $SUBJECT_REPORT"
echo "[INFO] Verify mismatch summary:  $VERIFY_MISMATCH_SUMMARY"

if [[ "$MANIFEST_OK" != "1" ]]; then
    echo "[ERROR] Source and destination manifests differ. See: $MANIFEST_DIFF" >&2
    exit 1
fi

if [[ "$VERIFY_OK" != "1" ]]; then
    echo "[ERROR] Checksum verification found mismatches. See: $VERIFY_MISMATCH_SUMMARY" >&2
    exit 1
fi

echo
echo "[INFO] Copy + verification completed successfully."
echo "[INFO] Finished: $(date)"
