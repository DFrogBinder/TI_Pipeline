#!/usr/bin/env bash
# Batch wrapper for register_seg_to_anat_fsl.sh.
# Expected layout: <root-dir>/<subjectID>/anat/
# Expected files in each anat dir:
#   <subjectID>_T1w.nii(.gz)
#   <subjectID>_T1w_ras_1mm_T1andT2_masks.nii(.gz)

set -euo pipefail

usage() {
  echo "Usage: $0 <root-dir> [dof] [-j jobs|--jobs jobs]"
  echo "Example (serial):   $0 /data/study 6"
  echo "Example (parallel): $0 /data/study 6 -j 4"
}

if [[ $# -lt 1 || $# -gt 4 ]]; then
  usage
  exit 1
fi

ROOT_DIR="$1"
shift
DOF="6"          # 6=rigid (default), use 12 for full affine if needed
JOBS="1"         # default keeps previous serial behavior

# Optional positional dof, followed by optional jobs flag.
if [[ $# -gt 0 && "$1" != "-j" && "$1" != "--jobs" ]]; then
  DOF="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -j|--jobs)
      if [[ $# -lt 2 ]]; then
        echo "Error: missing value for $1"
        usage
        exit 1
      fi
      JOBS="$2"
      shift 2
      ;;
    *)
      echo "Error: unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [[ "$JOBS" -lt 1 ]]; then
  echo "Error: jobs must be a positive integer, got: $JOBS"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGISTER_SCRIPT="${SCRIPT_DIR}/register_seg_to_anat_fsl.sh"

if [[ ! -f "$REGISTER_SCRIPT" ]]; then
  echo "Error: cannot find $REGISTER_SCRIPT"
  exit 1
fi

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "Error: root directory does not exist: $ROOT_DIR"
  exit 1
fi

find_nifti() {
  local stem="$1"
  if [[ -f "${stem}.nii" ]]; then
    echo "${stem}.nii"
    return 0
  fi
  if [[ -f "${stem}.nii.gz" ]]; then
    echo "${stem}.nii.gz"
    return 0
  fi
  return 1
}

shopt -s nullglob
anat_dirs=( "$ROOT_DIR"/*/anat )

if [[ ${#anat_dirs[@]} -eq 0 ]]; then
  echo "No subject anat directories found under: $ROOT_DIR"
  exit 1
fi

total=0
success=0
failed=0
skipped=0

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/register_seg_to_anat_fsl_batch.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT

HAVE_WAIT_N=0
if help wait 2>/dev/null | grep -q -- "-n"; then
  HAVE_WAIT_N=1
fi

running_jobs=0
queued=0

active_pids=()
status_files=()
status_subjects=()
log_files=()

wait_for_one_job() {
  if [[ "$running_jobs" -le 0 ]]; then
    return 0
  fi

  if [[ "$HAVE_WAIT_N" -eq 1 ]]; then
    wait -n || true
  else
    local pid="${active_pids[0]:-}"
    if [[ -n "$pid" ]]; then
      wait "$pid" || true
      active_pids=( "${active_pids[@]:1}" )
    fi
  fi

  ((running_jobs -= 1))
}

for anat_dir in "${anat_dirs[@]}"; do
  [[ -d "$anat_dir" ]] || continue
  ((total += 1))

  subject_id="$(basename "$(dirname "$anat_dir")")"
  anat_stem="${anat_dir}/${subject_id}_T1w"
  seg_stem="${anat_dir}/${subject_id}_T1w_ras_1mm_T1andT2_masks"

  anat_file="$(find_nifti "$anat_stem" || true)"
  seg_file="$(find_nifti "$seg_stem" || true)"

  if [[ -z "$anat_file" || -z "$seg_file" ]]; then
    echo "[$subject_id] Skipping: required files not found."
    echo "  Expected: ${subject_id}_T1w.nii(.gz)"
    echo "  Expected: ${subject_id}_T1w_ras_1mm_T1andT2_masks.nii(.gz)"
    ((skipped += 1))
    continue
  fi

  out_prefix="${anat_dir}/${subject_id}"
  status_file="${tmp_dir}/${subject_id}.status"
  log_file="${tmp_dir}/${subject_id}.log"
  status_files+=( "$status_file" )
  status_subjects+=( "$subject_id" )
  log_files+=( "$log_file" )

  echo "[$subject_id] Queueing registration..."
  (
    if bash "$REGISTER_SCRIPT" --replace-input "$seg_file" "$anat_file" "$out_prefix" "$DOF" >"$log_file" 2>&1; then
      echo "success" >"$status_file"
    else
      echo "failed" >"$status_file"
    fi
  ) &
  pid="$!"
  if [[ "$HAVE_WAIT_N" -ne 1 ]]; then
    active_pids+=( "$pid" )
  fi

  ((queued += 1))
  ((running_jobs += 1))

  if [[ "$running_jobs" -ge "$JOBS" ]]; then
    wait_for_one_job
  fi
done

while [[ "$running_jobs" -gt 0 ]]; do
  wait_for_one_job
done

for idx in "${!status_files[@]}"; do
  subject_id="${status_subjects[$idx]}"
  status_file="${status_files[$idx]}"
  log_file="${log_files[$idx]}"
  status="failed"

  if [[ -f "$status_file" ]]; then
    status="$(<"$status_file")"
  fi

  if [[ "$status" == "success" ]]; then
    ((success += 1))
    echo "[$subject_id] Done."
  else
    ((failed += 1))
    echo "[$subject_id] Registration failed."
    if [[ -f "$log_file" ]]; then
      echo "[$subject_id] Error log (last 15 lines):"
      tail -n 15 "$log_file" | sed 's/^/  /'
    fi
  fi
done

echo
echo "Batch complete."
echo "Total subjects found: $total"
echo "Queued:               $queued"
echo "Successful:           $success"
echo "Skipped:              $skipped"
echo "Failed:               $failed"

if [[ $failed -gt 0 ]]; then
  exit 1
fi
