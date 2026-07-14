#!/bin/bash
#SBATCH --job-name=collect_charm
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/parscratch/users/cop23bi/charm_segmentations/logs/collect-%j.out

set -euo pipefail

CAMCAN_DIR="${CAMCAN_DIR:-/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment}"
OUT_ROOT="${TI_CHARM_OUTPUT_ROOT:-/mnt/parscratch/users/cop23bi/charm_segmentations}"
SUBJECTS_FILE="${TI_CHARM_SUBJECTS_FILE:-${OUT_ROOT}/submission/subjects.txt}"
COLLECTOR_PY="${TI_CHARM_COLLECTOR_PY:-${CAMCAN_DIR}/charm_segmentation_batch/collect_charm_segmentations.py}"
COLLECTION_DIR="${TI_CHARM_COLLECTION_DIR:-${OUT_ROOT}/collection}"

python3 -u "$COLLECTOR_PY" \
    --out-root "$OUT_ROOT" \
    --subjects-file "$SUBJECTS_FILE" \
    --collection-dir "$COLLECTION_DIR"
