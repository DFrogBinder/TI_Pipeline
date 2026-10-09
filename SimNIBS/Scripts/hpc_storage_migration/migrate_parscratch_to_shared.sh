#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${TI_MIGRATION_SOURCE_ROOT:-/mnt/parscratch/users/cop23bi}"
DEST_ROOT="${TI_MIGRATION_DEST_ROOT:-/shared/boyan/Shared}"
MANIFEST="${TI_MIGRATION_MANIFEST:-${SCRIPT_DIR}/parscratch_inventory.txt}"
ACTION=""
EXCLUDED_ENTRIES=()
CONFIRM_DESTINATION=""
MANIFEST_ENTRY_COUNT=0

usage() {
    printf '%s\n' \
        'Usage:' \
        '  bash migrate_parscratch_to_shared.sh ACTION (--exclude TOP_LEVEL_ENTRY | --exclude-file FILE) [options]' \
        '' \
        'Actions:' \
        '  audit       Validate the reviewed inventory and report source sizes/counts.' \
        '  copy        Resumably copy all non-excluded entries with rsync.' \
        '  verify      Read-only rsync comparison using size and modification time.' \
        '  checksum    Read-only full-content rsync checksum comparison.' \
        '' \
        'Required for copy:' \
        '  --confirm-destination /shared/boyan/Shared' \
        '' \
        'The script never deletes source data. Rerunning copy resumes incomplete data.'
}

fail() {
    echo "[ERROR] $*" >&2
    exit 2
}

manifest_entries() {
    awk 'NF && $1 !~ /^#/ {print}' "${MANIFEST}"
}

contains_manifest_entry() {
    local requested_entry="$1"
    manifest_entries | grep -Fqx -- "${requested_entry}"
}

exclusion_file_entries() {
    local exclusion_file="$1"
    awk 'NF && $1 !~ /^#/ {print}' "${exclusion_file}"
}

is_excluded() {
    local requested_entry="$1"
    local excluded_entry
    for excluded_entry in "${EXCLUDED_ENTRIES[@]}"; do
        if [ "${requested_entry}" = "${excluded_entry}" ]; then
            return 0
        fi
    done
    return 1
}

imaging_or_archive_file_count() {
    local source_entry="$1"
    if [ -d "${source_entry}" ] && [ ! -L "${source_entry}" ]; then
        find "${source_entry}" -type f \( \
            -iname '*.nii' -o \
            -iname '*.nii.gz' -o \
            -iname '*.mgz' -o \
            -iname '*.mgh' -o \
            -iname '*.mnc' -o \
            -iname '*.nrrd' -o \
            -iname '*.hdr' -o \
            -iname '*.img' -o \
            -iname '*.dcm' -o \
            -iname '*.ima' -o \
            -iname '*.gii' -o \
            -iname '*.msh' -o \
            -iname '*.stl' -o \
            -iname '*.ply' -o \
            -iname '*.vtk' -o \
            -iname '*.vtp' -o \
            -iname '*.vtu' -o \
            -iname '*.off' -o \
            -iname '*.annot' -o \
            -iname '*.pial' -o \
            -iname '*.white' -o \
            -iname '*.inflated' -o \
            -iname '*.sphere' -o \
            -iname '*.surf' -o \
            -iname '*.zip' -o \
            -iname '*.tar' -o \
            -iname '*.tar.gz' -o \
            -iname '*.tgz' -o \
            -iname '*.7z' \
        \) -printf '.' | wc -c
        return
    fi
    case "${source_entry,,}" in
        *.nii|*.nii.gz|*.mgz|*.mgh|*.mnc|*.nrrd|*.hdr|*.img|*.dcm|*.ima|*.gii|*.msh|*.stl|*.ply|*.vtk|*.vtp|*.vtu|*.off|*.annot|*.pial|*.white|*.inflated|*.sphere|*.surf|*.zip|*.tar|*.tar.gz|*.tgz|*.7z)
            echo 1
            ;;
        *)
            echo 0
            ;;
    esac
}

validate_exclusions() {
    local excluded_entry
    local exclusion_count
    local unique_count

    exclusion_count="${#EXCLUDED_ENTRIES[@]}"
    [ "${exclusion_count}" -gt 0 ] || fail "At least one --exclude or --exclude-file entry is required."
    for excluded_entry in "${EXCLUDED_ENTRIES[@]}"; do
        validate_simple_name "${excluded_entry}"
        contains_manifest_entry "${excluded_entry}" || fail "Excluded entry is not present in the reviewed manifest: ${excluded_entry}"
    done
    unique_count="$(printf '%s\n' "${EXCLUDED_ENTRIES[@]}" | LC_ALL=C sort -u | wc -l)"
    [ "${unique_count}" -eq "${exclusion_count}" ] || fail "Excluded entries must be unique."
}

validate_simple_name() {
    local requested_entry="$1"
    case "${requested_entry}" in
        ""|.|..|*/*|*$'\n'*|*$'\r'*|*$'\t'*)
            fail "Invalid top-level entry name: ${requested_entry}"
            ;;
    esac
}

validate_roots() {
    [ -d "${SOURCE_ROOT}" ] || fail "Source root is not a directory: ${SOURCE_ROOT}"
    [ ! -L "${SOURCE_ROOT}" ] || fail "Source root must not be a symbolic link: ${SOURCE_ROOT}"
    [ -d "${DEST_ROOT}" ] || fail "Destination root is not a directory: ${DEST_ROOT}"
    [ ! -L "${DEST_ROOT}" ] || fail "Destination root must not be a symbolic link: ${DEST_ROOT}"
    [ -r "${SOURCE_ROOT}" ] || fail "Source root is not readable: ${SOURCE_ROOT}"
    [ -w "${DEST_ROOT}" ] || fail "Destination root is not writable: ${DEST_ROOT}"
    [ -f "${MANIFEST}" ] || fail "Inventory manifest is missing: ${MANIFEST}"

    local canonical_source
    local canonical_destination
    canonical_source="$(realpath -e -- "${SOURCE_ROOT}")"
    canonical_destination="$(realpath -e -- "${DEST_ROOT}")"
    [ "${canonical_source}" != "${canonical_destination}" ] || fail "Source and destination resolve to the same directory."
    case "${canonical_destination}/" in
        "${canonical_source}/"*) fail "Destination must not be inside the source root." ;;
    esac
    case "${canonical_source}/" in
        "${canonical_destination}/"*) fail "Source must not be inside the destination root." ;;
    esac

    SOURCE_ROOT="${canonical_source}"
    DEST_ROOT="${canonical_destination}"
}

validate_inventory() {
    local comparison_dir
    local manifest_sorted
    local source_sorted
    local missing_entries
    local unexpected_entries

    comparison_dir="$(mktemp -d)"
    manifest_sorted="${comparison_dir}/manifest.txt"
    source_sorted="${comparison_dir}/source.txt"
    trap 'rm -rf -- "${comparison_dir}"' RETURN

    manifest_entries | LC_ALL=C sort > "${manifest_sorted}"
    find "${SOURCE_ROOT}" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort > "${source_sorted}"

    MANIFEST_ENTRY_COUNT="$(wc -l < "${manifest_sorted}")"
    [ "${MANIFEST_ENTRY_COUNT}" -gt 0 ] || fail "The reviewed inventory manifest is empty."
    [ "$(uniq -d "${manifest_sorted}" | wc -l)" -eq 0 ] || fail "The inventory manifest contains duplicate entries."

    missing_entries="$(comm -23 "${manifest_sorted}" "${source_sorted}")"
    unexpected_entries="$(comm -13 "${manifest_sorted}" "${source_sorted}")"
    if [ -n "${missing_entries}" ] || [ -n "${unexpected_entries}" ]; then
        [ -z "${missing_entries}" ] || printf '[ERROR] Manifest entries missing from source:\n%s\n' "${missing_entries}" >&2
        [ -z "${unexpected_entries}" ] || printf '[ERROR] Unreviewed source entries:\n%s\n' "${unexpected_entries}" >&2
        fail "Live source inventory differs from the reviewed manifest; no copy or verification was attempted."
    fi

    trap - RETURN
    rm -rf -- "${comparison_dir}"
}

receipt_dir() {
    printf '%s\n' "${DEST_ROOT}/.parscratch_migration_receipts"
}

print_scope() {
    local exclusion_count="${#EXCLUDED_ENTRIES[@]}"
    local inclusion_count=$((MANIFEST_ENTRY_COUNT - exclusion_count))
    local excluded_entry
    printf '%s\n' \
        'Scope:' \
        '  dataset/ROI: complete reviewed parscratch top-level inventory' \
        "  top-level entries found: ${MANIFEST_ENTRY_COUNT}" \
        "  top-level entries included: ${inclusion_count}" \
        "  top-level entries excluded: ${exclusion_count}" \
        '  Slurm tasks/array: none; login-node filesystem operation' \
        "  expected destinations: ${inclusion_count} under ${DEST_ROOT}" \
        "  execution: full requested migration scope for action ${ACTION}" \
        '  source deletion: disabled'
    for excluded_entry in "${EXCLUDED_ENTRIES[@]}"; do
        echo "  excluded: ${excluded_entry}"
    done
}

run_audit() {
    local report_dir
    local report
    local entry
    local source_entry
    local entry_type
    local apparent_bytes
    local file_count
    local imaging_or_archive_files
    local link_target

    report_dir="$(receipt_dir)"
    mkdir -p -- "${report_dir}"
    report="${report_dir}/audit-$(date -u +%Y%m%dT%H%M%SZ).tsv"
    printf 'entry\tincluded\ttype\tapparent_bytes\tregular_files\timaging_mesh_or_archive_files\tlink_target\n' > "${report}"

    while IFS= read -r entry; do
        source_entry="${SOURCE_ROOT}/${entry}"
        link_target=""
        if [ -L "${source_entry}" ]; then
            entry_type="symlink"
            link_target="$(readlink -- "${source_entry}")"
        elif [ -d "${source_entry}" ]; then
            entry_type="directory"
        elif [ -f "${source_entry}" ]; then
            entry_type="file"
        else
            entry_type="other"
        fi
        apparent_bytes="$(du -sB1 --apparent-size -- "${source_entry}" | awk '{print $1}')"
        if [ -d "${source_entry}" ] && [ ! -L "${source_entry}" ]; then
            file_count="$(find "${source_entry}" -type f -printf '.' | wc -c)"
        elif [ -f "${source_entry}" ]; then
            file_count="1"
        else
            file_count="0"
        fi
        imaging_or_archive_files="$(imaging_or_archive_file_count "${source_entry}")"
        if is_excluded "${entry}"; then
            printf '%s\tno\t%s\t%s\t%s\t%s\t%s\n' "${entry}" "${entry_type}" "${apparent_bytes}" "${file_count}" "${imaging_or_archive_files}" "${link_target}" | tee -a "${report}"
        else
            printf '%s\tyes\t%s\t%s\t%s\t%s\t%s\n' "${entry}" "${entry_type}" "${apparent_bytes}" "${file_count}" "${imaging_or_archive_files}" "${link_target}" | tee -a "${report}"
        fi
    done < <(manifest_entries)

    echo "[INFO] Audit receipt: ${report}"
    df -h -- "${SOURCE_ROOT}" "${DEST_ROOT}"
}

copy_entry() {
    local entry="$1"
    local source_entry="${SOURCE_ROOT}/${entry}"
    local destination_entry="${DEST_ROOT}/${entry}"

    if [ -d "${source_entry}" ] && [ ! -L "${source_entry}" ]; then
        mkdir -p -- "${destination_entry}"
        rsync -rltH \
            --no-perms \
            --no-owner \
            --no-group \
            --partial \
            --protect-args \
            --human-readable \
            --info=progress2,stats2 \
            -- "${source_entry}/" "${destination_entry}/"
    else
        rsync -rltH \
            --no-perms \
            --no-owner \
            --no-group \
            --partial \
            --protect-args \
            --human-readable \
            --info=progress2,stats2 \
            -- "${source_entry}" "${DEST_ROOT}/"
    fi
}

run_copy() {
    local report_dir
    local status_report
    local entry
    local started_at
    local completed_at

    [ "${CONFIRM_DESTINATION}" = "${DEST_ROOT}" ] || fail "copy requires --confirm-destination ${DEST_ROOT}"
    command -v rsync >/dev/null 2>&1 || fail "rsync is not available."
    report_dir="$(receipt_dir)"
    mkdir -p -- "${report_dir}"
    status_report="${report_dir}/copy-status-$(date -u +%Y%m%dT%H%M%SZ).tsv"
    printf 'entry\tstatus\tstarted_at\tcompleted_at\n' > "${status_report}"

    while IFS= read -r entry; do
        if is_excluded "${entry}"; then
            continue
        fi
        started_at="$(date --iso-8601=seconds)"
        echo "[INFO] Copying ${entry}"
        if copy_entry "${entry}"; then
            completed_at="$(date --iso-8601=seconds)"
            printf '%s\tcomplete\t%s\t%s\n' "${entry}" "${started_at}" "${completed_at}" | tee -a "${status_report}"
        else
            completed_at="$(date --iso-8601=seconds)"
            printf '%s\tfailed\t%s\t%s\n' "${entry}" "${started_at}" "${completed_at}" | tee -a "${status_report}"
            fail "rsync failed for ${entry}; rerun copy to resume after diagnosing the error."
        fi
    done < <(manifest_entries)

    echo "[INFO] Copy status receipt: ${status_report}"
}

verify_entry() {
    local entry="$1"
    local verification_mode="$2"
    local source_entry="${SOURCE_ROOT}/${entry}"
    local destination_entry="${DEST_ROOT}/${entry}"
    local difference_log="$3"
    local checksum_args=()

    if [ "${verification_mode}" = "checksum" ]; then
        checksum_args+=(--checksum)
    fi

    if [ -d "${source_entry}" ] && [ ! -L "${source_entry}" ]; then
        [ -d "${destination_entry}" ] || return 1
        rsync -rltHn \
            --no-perms \
            --no-owner \
            --no-group \
            --delete \
            --itemize-changes \
            --out-format='%i %n%L' \
            --protect-args \
            "${checksum_args[@]}" \
            -- "${source_entry}/" "${destination_entry}/" > "${difference_log}"
        [ ! -s "${difference_log}" ]
    elif [ -L "${source_entry}" ]; then
        [ -L "${destination_entry}" ] || return 1
        [ "$(readlink -- "${source_entry}")" = "$(readlink -- "${destination_entry}")" ]
    else
        [ -e "${destination_entry}" ] || [ -L "${destination_entry}" ] || return 1
        cmp -s -- "${source_entry}" "${destination_entry}"
    fi
}

run_verify() {
    local verification_mode="$1"
    local report_dir
    local status_report
    local difference_dir
    local entry
    local difference_log
    local failures=0

    command -v rsync >/dev/null 2>&1 || fail "rsync is not available."
    report_dir="$(receipt_dir)"
    mkdir -p -- "${report_dir}"
    status_report="${report_dir}/${verification_mode}-status-$(date -u +%Y%m%dT%H%M%SZ).tsv"
    difference_dir="${status_report%.tsv}-differences"
    mkdir -p -- "${difference_dir}"
    printf 'entry\tstatus\tmode\n' > "${status_report}"

    while IFS= read -r entry; do
        if is_excluded "${entry}"; then
            continue
        fi
        difference_log="${difference_dir}/${entry}.txt"
        if verify_entry "${entry}" "${verification_mode}" "${difference_log}"; then
            printf '%s\tmatch\t%s\n' "${entry}" "${verification_mode}" | tee -a "${status_report}"
            if [ -f "${difference_log}" ] && [ ! -s "${difference_log}" ]; then
                rm -f -- "${difference_log}"
            fi
        else
            printf '%s\tdifferent\t%s\n' "${entry}" "${verification_mode}" | tee -a "${status_report}"
            failures=$((failures + 1))
        fi
    done < <(manifest_entries)

    echo "[INFO] Verification status receipt: ${status_report}"
    if [ "${failures}" -ne 0 ]; then
        fail "${failures} entries differ or are missing; inspect ${difference_dir} and rerun copy."
    fi
    rmdir -- "${difference_dir}" 2>/dev/null || true
}

if [ "$#" -eq 0 ]; then
    usage
    exit 2
fi

ACTION="$1"
shift
while [ "$#" -gt 0 ]; do
    case "$1" in
        --exclude)
            [ "$#" -ge 2 ] || fail "--exclude requires a value."
            EXCLUDED_ENTRIES+=("$2")
            shift 2
            ;;
        --exclude-file)
            [ "$#" -ge 2 ] || fail "--exclude-file requires a value."
            [ -f "$2" ] || fail "Exclusion file is missing: $2"
            while IFS= read -r excluded_entry; do
                EXCLUDED_ENTRIES+=("${excluded_entry}")
            done < <(exclusion_file_entries "$2")
            shift 2
            ;;
        --confirm-destination)
            [ "$#" -ge 2 ] || fail "--confirm-destination requires a value."
            CONFIRM_DESTINATION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "Unknown argument: $1"
            ;;
    esac
done

case "${ACTION}" in
    audit|copy|verify|checksum) ;;
    *) fail "Unknown action: ${ACTION}" ;;
esac

validate_roots
validate_exclusions
validate_inventory
print_scope

case "${ACTION}" in
    audit) run_audit ;;
    copy) run_copy ;;
    verify) run_verify "verify" ;;
    checksum) run_verify "checksum" ;;
esac
