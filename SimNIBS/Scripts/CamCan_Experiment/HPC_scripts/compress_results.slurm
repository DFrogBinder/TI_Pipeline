#!/bin/bash
#SBATCH --job-name=compress_export
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=compress_export_%j.out
#SBATCH --error=compress_export_%j.err

set -euo pipefail

echo "[INFO] Host: $(hostname)"
echo "[INFO] JobID: ${SLURM_JOB_ID}"
echo "[INFO] Start time: $(date)"
echo

# ------------------------------------------------------------------
# EDIT THESE PATHS
# ------------------------------------------------------------------
EXPORT_DIR="/mnt/parscratch/users/cop23bi/export_sim_outputs"
OUT_DIR="/mnt/parscratch/users/cop23bi"

ARCHIVE_NAME="export_sim_outputs"
ZSTD_ARCHIVE="${OUT_DIR}/${ARCHIVE_NAME}.tar.zst"
GZIP_ARCHIVE="${OUT_DIR}/${ARCHIVE_NAME}.tar.gz"

# ------------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------------
if [[ ! -d "$EXPORT_DIR" ]]; then
    echo "[ERROR] Export directory not found: $EXPORT_DIR" >&2
    exit 2
fi

echo "[INFO] Export directory size before compression:"
du -sh "$EXPORT_DIR" || true
echo

# ------------------------------------------------------------------
# Compression
# ------------------------------------------------------------------
cd "$OUT_DIR"

if command -v zstd >/dev/null 2>&1; then
    echo "[INFO] Using zstd compression"
    echo "[INFO] Output: $ZSTD_ARCHIVE"

    tar -cf - "$(basename "$EXPORT_DIR")" \
        | zstd -T0 -19 -o "$ZSTD_ARCHIVE"

    echo
    echo "[INFO] zstd archive created:"
    ls -lh "$ZSTD_ARCHIVE"

else
    echo "[WARN] zstd not available; falling back to gzip"
    echo "[INFO] Output: $GZIP_ARCHIVE"

    tar -czf "$GZIP_ARCHIVE" "$(basename "$EXPORT_DIR")"

    echo
    echo "[INFO] gzip archive created:"
    ls -lh "$GZIP_ARCHIVE"
fi

echo
echo "[INFO] Finished time: $(date)"

