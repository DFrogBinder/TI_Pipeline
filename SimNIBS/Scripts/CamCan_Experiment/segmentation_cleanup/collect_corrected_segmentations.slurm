#!/bin/bash
#SBATCH --job-name=collect_charm_cleanup
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=collect-charm-cleanup-%j.out

set -euo pipefail

: "${TI_CHARM_CLEANUP_MANIFEST:?Missing TI_CHARM_CLEANUP_MANIFEST.}"
: "${TI_CHARM_CLEANUP_WORKFLOW_PY:?Missing TI_CHARM_CLEANUP_WORKFLOW_PY.}"
: "${TI_CHARM_CLEANUP_VALIDATION:?Missing TI_CHARM_CLEANUP_VALIDATION.}"
: "${TI_CHARM_CLEANUP_SUMMARY:?Missing TI_CHARM_CLEANUP_SUMMARY.}"
: "${TI_CHARM_CLEANUP_COLLECTION_MANIFEST:?Missing TI_CHARM_CLEANUP_COLLECTION_MANIFEST.}"
: "${TI_CHARM_CLEANUP_CHECKSUMS:?Missing TI_CHARM_CLEANUP_CHECKSUMS.}"

module purge
module use "$HOME/modules"
module load SimNIBS/4.0.1-foss-2023a
export PYTHONNOUSERSITE=1

COMMAND=(
    python -u "${TI_CHARM_CLEANUP_WORKFLOW_PY}"
    validate
    --manifest "${TI_CHARM_CLEANUP_MANIFEST}"
    --validation "${TI_CHARM_CLEANUP_VALIDATION}"
    --summary "${TI_CHARM_CLEANUP_SUMMARY}"
    --collection-manifest "${TI_CHARM_CLEANUP_COLLECTION_MANIFEST}"
    --checksums "${TI_CHARM_CLEANUP_CHECKSUMS}"
)
if [ "${TI_CHARM_CLEANUP_SKIP_NIFTI_LOAD:-0}" = "1" ]; then
    COMMAND+=(--skip-nifti-load)
fi
"${COMMAND[@]}"
