#!/bin/bash
#SBATCH --job-name=mni401_compare
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
umask 0027

: "${PIPELINE_DIR:?Set PIPELINE_DIR to the repository Scripts directory.}"
: "${MNI401_OUTPUT_PARENT:?Set the isolated SimNIBS 4.0.1 output parent.}"
: "${MNI45_BASELINE_PARENT:?Set the existing SimNIBS 4.5.0 baseline parent.}"
: "${MNI_FIXED_ATLAS_PATH:?Set the fixed MNI152 atlas path.}"
: "${TI_POST_PYTHON:?Set the explicit ti-post Python interpreter.}"

COMPARISON_PY="${PIPELINE_DIR}/CamCan_Experiment/post/compare_mni152_baseline_versions.py"
COMPARISON_DIR="${MNI401_OUTPUT_PARENT}/comparison"
ARCHIVE="${MNI401_OUTPUT_PARENT}/mni152_simnibs_4p0p1_vs_4p5p0_validation.tar.gz"
export MPLCONFIGDIR="${MNI401_OUTPUT_PARENT}/.matplotlib"
VALIDATION_RECORDS=(
    MNI152-left-m1
    MNI152-right-dlpc
    MNI152-left-hippocampus
    MNI152-right-thalamus
)

for required_file in \
    "${COMPARISON_PY}" \
    "${MNI_FIXED_ATLAS_PATH}" \
    "${TI_POST_PYTHON}"
do
    if [ ! -s "${required_file}" ]; then
        echo "[ERROR] Required collector input is missing: ${required_file}" >&2
        exit 2
    fi
done
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

mkdir -p "${COMPARISON_DIR}" "${MPLCONFIGDIR}"
"${TI_POST_PYTHON}" -c \
    'import nibabel,numpy,pandas; print("mni_version_comparison_dependencies=ready")'
"${TI_POST_PYTHON}" -u "${COMPARISON_PY}" \
    --simnibs-4p5-parent "${MNI45_BASELINE_PARENT}" \
    --simnibs-4p0p1-parent "${MNI401_OUTPUT_PARENT}" \
    --mni-atlas "${MNI_FIXED_ATLAS_PATH}" \
    --out-dir "${COMPARISON_DIR}"

test -s "${COMPARISON_DIR}/comparison_manifest.json"
test -s "${COMPARISON_DIR}/MNI152_SIMNIBS_VERSION_COMPARISON.md"
(
    cd "${COMPARISON_DIR}"
    sha256sum \
        MNI152_SIMNIBS_VERSION_COMPARISON.md \
        comparison_manifest.json \
        mni152_metric_differences_4p0p1_minus_4p5p0.csv \
        mni152_metrics_simnibs_4p0p1.csv \
        mni152_metrics_simnibs_4p5p0.csv \
        mni152_selected_metric_differences.csv \
        mni152_voxelwise_differences.csv \
        > comparison_outputs.sha256
    sha256sum -c comparison_outputs.sha256
)

tar -C "${MNI401_OUTPUT_PARENT}" -czf "${ARCHIVE}" \
    comparison \
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
    "[INFO] Version comparison: ${COMPARISON_DIR}" \
    "[INFO] Compact archive:    ${ARCHIVE}" \
    "[INFO] Archive checksum:   ${ARCHIVE}.sha256" \
    "[INFO] Raw FEM meshes remain in the isolated validation tree and are not duplicated in the compact archive."
