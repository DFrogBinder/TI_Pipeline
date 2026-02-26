#!/usr/bin/env bash
# Batch wrapper for register_seg_to_anat_fsl.sh.
# Expected layout: <root-dir>/<subjectID>/anat/
# Expected files in each anat dir:
#   <subjectID>_T1w.nii(.gz)
#   <subjectID>_T1w_ras_1mm_T1andT2_masks.nii(.gz)

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <root-dir> [dof]"
  echo "Example: $0 /data/study 6"
  exit 1
fi

ROOT_DIR="$1"
DOF="${2:-6}"   # 6=rigid (default), use 12 for full affine if needed

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
  echo "[$subject_id] Registering..."

  if bash "$REGISTER_SCRIPT" "$seg_file" "$anat_file" "$out_prefix" "$DOF"; then
    ((success += 1))
  else
    echo "[$subject_id] Registration failed."
    ((failed += 1))
  fi
done

echo
echo "Batch complete."
echo "Total subjects found: $total"
echo "Successful:           $success"
echo "Skipped:              $skipped"
echo "Failed:               $failed"

if [[ $failed -gt 0 ]]; then
  exit 1
fi
