#!/bin/bash
#SBATCH --job-name=mni401_collect
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
umask 0027

: "${MNI401_OUTPUT_PARENT:?Set the isolated SimNIBS 4.0.1 output parent.}"

ARCHIVE="${MNI401_OUTPUT_PARENT}/mni152_simnibs_4p0p1_validated_results.tar.gz"
VALIDATION_RECORDS=(
    MNI152-left-m1
    MNI152-right-dlpc
    MNI152-left-hippocampus
    MNI152-right-thalamus
)

for output_subject in "${VALIDATION_RECORDS[@]}"; do
    summary="${MNI401_OUTPUT_PARENT}/validation_records/${output_subject}.json"
    ti_path="${MNI401_OUTPUT_PARENT}/${output_subject}/anat/SimNIBS/ti_brain_only.nii.gz"
    if [ ! -s "${summary}" ] || \
       ! grep -qF '"status": "complete"' "${summary}" || \
       [ ! -s "${ti_path}" ]
    then
        echo "[ERROR] Incomplete 4.0.1 validation output: ${output_subject}" >&2
        exit 2
    fi
done

tar -C "${MNI401_OUTPUT_PARENT}" -czf "${ARCHIVE}" \
    validation_records \
    MNI152-left-m1/anat/SimNIBS/ti_brain_only.nii.gz \
    MNI152-left-m1/anat/SimNIBS/mni_baseline_provenance.json \
    MNI152-right-dlpc/anat/SimNIBS/ti_brain_only.nii.gz \
    MNI152-right-dlpc/anat/SimNIBS/mni_baseline_provenance.json \
    MNI152-left-hippocampus/anat/SimNIBS/ti_brain_only.nii.gz \
    MNI152-left-hippocampus/anat/SimNIBS/mni_baseline_provenance.json \
    MNI152-right-thalamus/anat/SimNIBS/ti_brain_only.nii.gz \
    MNI152-right-thalamus/anat/SimNIBS/mni_baseline_provenance.json
sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"
sha256sum -c "${ARCHIVE}.sha256"

printf '%s\n' \
    "[INFO] All four SimNIBS 4.0.1 MNI152 baselines validated." \
    "[INFO] Compact archive:    ${ARCHIVE}" \
    "[INFO] Archive checksum:   ${ARCHIVE}.sha256" \
    "[INFO] Compare against the retained 4.5.0 baselines locally after download." \
    "[INFO] Raw FEM meshes remain in the isolated validation tree and are not duplicated in the compact archive."
