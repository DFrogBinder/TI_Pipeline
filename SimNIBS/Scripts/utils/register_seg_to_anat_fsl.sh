#!/usr/bin/env bash
# Linear registration (FLIRT) of a segmentation map to an anatomical image.

set -euo pipefail

usage() {
  echo "Usage: $0 [--replace-input] <segmentation.nii(.gz)> <anatomical.nii(.gz)> <out_prefix> [dof]"
  echo "Example (default output):  $0 seg.nii.gz T1.nii.gz sub01 6"
  echo "Example (replace input):   $0 --replace-input seg.nii.gz T1.nii.gz sub01 6"
}

replace_with_registered() {
  local registered_file="$1"
  local target_file="$2"

  if [[ ! -f "$registered_file" ]]; then
    echo "Error: expected registered file not found: $registered_file" >&2
    return 1
  fi

  if [[ "$registered_file" == "$target_file" ]]; then
    return 0
  fi

  # Atomic replacement when target is .nii.gz (move in same directory).
  if [[ "$target_file" == *.nii.gz ]]; then
    mv -f "$registered_file" "$target_file"
    return 0
  fi

  # For .nii targets, convert from gzip stream into a temp .nii then replace.
  if [[ "$target_file" == *.nii ]]; then
    local target_dir target_name tmp_target
    target_dir="$(dirname "$target_file")"
    target_name="$(basename "$target_file")"
    tmp_target="${target_dir}/.${target_name}.tmp.$$"

    rm -f "$tmp_target"
    if ! gzip -dc "$registered_file" >"$tmp_target"; then
      rm -f "$tmp_target"
      echo "Error: failed converting registered output to .nii for: $target_file" >&2
      return 1
    fi

    mv -f "$tmp_target" "$target_file"
    rm -f "$registered_file"
    return 0
  fi

  # Fallback: replace by move.
  mv -f "$registered_file" "$target_file"
}

REPLACE_INPUT=0
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --replace-input|--in-place)
      REPLACE_INPUT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      POSITIONAL+=( "$@" )
      break
      ;;
    -*)
      echo "Error: unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      POSITIONAL+=( "$1" )
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -lt 3 || ${#POSITIONAL[@]} -gt 4 ]]; then
  usage
  exit 1
fi

SEG="${POSITIONAL[0]}"
ANAT="${POSITIONAL[1]}"
OUT="${POSITIONAL[2]}"
DOF="${POSITIONAL[3]:-6}"   # 6=rigid (default), use 12 for full affine if needed
REGISTERED_OUT="${OUT}_seg_in_anat.nii.gz"

for cmd in flirt convert_xfm; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "Error: '$cmd' not found in PATH."; exit 1; }
done
if [[ "$REPLACE_INPUT" -eq 1 && "$SEG" == *.nii ]]; then
  command -v gzip >/dev/null 2>&1 || { echo "Error: 'gzip' not found in PATH."; exit 1; }
fi

# Register segmentation -> anatomical
# nearestneighbour preserves integer labels in the segmentation map
flirt \
  -in "$SEG" \
  -ref "$ANAT" \
  -out "$REGISTERED_OUT" \
  -omat "${OUT}_seg2anat.mat" \
  -dof "$DOF" \
  -cost normmi \
  -searchrx -90 90 \
  -searchry -90 90 \
  -searchrz -90 90 \
  -interp nearestneighbour

# Also save inverse transform (anat -> seg)
convert_xfm -omat "${OUT}_anat2seg.mat" -inverse "${OUT}_seg2anat.mat"

final_seg="$REGISTERED_OUT"
if [[ "$REPLACE_INPUT" -eq 1 ]]; then
  replace_with_registered "$REGISTERED_OUT" "$SEG"
  final_seg="$SEG"
fi

echo "Done."
echo "Registered segmentation: $final_seg"
echo "Transform matrix:        ${OUT}_seg2anat.mat"
echo "Inverse matrix:          ${OUT}_anat2seg.mat"
