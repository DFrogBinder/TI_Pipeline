#!/usr/bin/env bash
# Linear registration (FLIRT) of a segmentation map to an anatomical image.

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 <segmentation.nii.gz> <anatomical.nii.gz> <out_prefix> [dof]"
  echo "Example: $0 seg.nii.gz T1.nii.gz sub01 6"
  exit 1
fi

SEG="$1"
ANAT="$2"
OUT="$3"
DOF="${4:-6}"   # 6=rigid (default), use 12 for full affine if needed

for cmd in flirt convert_xfm; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "Error: '$cmd' not found in PATH."; exit 1; }
done

# Register segmentation -> anatomical
# nearestneighbour preserves integer labels in the segmentation map
flirt \
  -in "$SEG" \
  -ref "$ANAT" \
  -out "${OUT}_seg_in_anat.nii.gz" \
  -omat "${OUT}_seg2anat.mat" \
  -dof "$DOF" \
  -cost normmi \
  -searchrx -90 90 \
  -searchry -90 90 \
  -searchrz -90 90 \
  -interp nearestneighbour

# Also save inverse transform (anat -> seg)
convert_xfm -omat "${OUT}_anat2seg.mat" -inverse "${OUT}_seg2anat.mat"

echo "Done."
echo "Registered segmentation: ${OUT}_seg_in_anat.nii.gz"
echo "Transform matrix:        ${OUT}_seg2anat.mat"
echo "Inverse matrix:          ${OUT}_anat2seg.mat"
