#!/bin/bash
#SBATCH --job-name=compress_folders
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=compress_%j.out
#SBATCH --error=compress_%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch compress_folders.sbatch <output.tar.zst|output.tar.gz|output.zip> <dir1> [dir2 ... dirN]

Environment overrides:
  COMPRESS_LEVEL=<int>   Compression level override.
                         tar.zst default: auto (1 if many precompressed files, else 3)
                         tar.gz default: 1
                         zip default: 1
EOF
}

if (( $# < 2 )); then
  usage
  exit 1
fi

OUT="$1"
shift
DIRS=("$@")
THREADS="${SLURM_CPUS_PER_TASK:-1}"

for d in "${DIRS[@]}"; do
  if [[ ! -d "$d" ]]; then
    echo "Error: Not a directory: $d" >&2
    exit 1
  fi
done

# Extensions that are usually already compressed; avoid recompressing heavily.
ZIP_STORE_EXTS="7z:avi:bz2:deb:dmg:flac:gz:iso:jar:jpeg:jpg:m4a:mkv:mov:mp3:mp4:mpeg:mpg:npy:npz:pdf:png:pptx:rar:tgz:txz:wav:webm:webp:xz:zip:zst"

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: Required command not found: $cmd" >&2
    exit 1
  fi
}

count_files() {
  find "${DIRS[@]}" -type f | wc -l
}

count_precompressed_files() {
  local count=0
  local f lc
  while IFS= read -r -d '' f; do
    lc="${f,,}"
    case "$lc" in
      *.7z|*.avi|*.bz2|*.deb|*.dmg|*.flac|*.gz|*.iso|*.jar|*.jpeg|*.jpg|*.m4a|*.mkv|*.mov|*.mp3|*.mp4|*.mpeg|*.mpg|*.npy|*.npz|*.pdf|*.png|*.pptx|*.rar|*.tgz|*.txz|*.wav|*.webm|*.webp|*.xz|*.zip|*.zst)
        ((count++))
        ;;
    esac
  done < <(find "${DIRS[@]}" -type f -print0)
  echo "$count"
}

TOTAL_FILES="$(count_files)"
PRECOMPRESSED_FILES="$(count_precompressed_files)"

if (( TOTAL_FILES > 0 )); then
  PRECOMP_RATIO=$(( PRECOMPRESSED_FILES * 100 / TOTAL_FILES ))
else
  PRECOMP_RATIO=0
fi

echo "Output: $OUT"
echo "Input dirs: ${DIRS[*]}"
echo "Threads: $THREADS"
echo "Total files: $TOTAL_FILES"
echo "Likely precompressed files: $PRECOMPRESSED_FILES (${PRECOMP_RATIO}%)"

case "$OUT" in
  *.tar.zst)
    require_cmd tar
    require_cmd zstd
    if [[ -n "${COMPRESS_LEVEL:-}" ]]; then
      ZSTD_LEVEL="$COMPRESS_LEVEL"
    elif (( PRECOMP_RATIO >= 50 )); then
      # For many precompressed inputs (e.g., lots of .nii.gz), keep compression very light.
      ZSTD_LEVEL=1
    else
      ZSTD_LEVEL=3
    fi
    tar -I "zstd -T${THREADS} -${ZSTD_LEVEL}" -cf "$OUT" "${DIRS[@]}"
    ;;
  *.tar.gz)
    require_cmd tar
    require_cmd pigz
    GZ_LEVEL="${COMPRESS_LEVEL:-1}"
    tar -I "pigz -p ${THREADS} -${GZ_LEVEL}" -cf "$OUT" "${DIRS[@]}"
    ;;
  *.zip)
    require_cmd zip
    ZIP_LEVEL="${COMPRESS_LEVEL:-1}"
    # zip -n stores matching extensions without recompression.
    zip -r "-${ZIP_LEVEL}" -n "$ZIP_STORE_EXTS" "$OUT" "${DIRS[@]}"
    ;;
  *)
    echo "Error: Output must end with .tar.zst, .tar.gz, or .zip" >&2
    exit 1
    ;;
esac

ls -lh "$OUT"
echo "Done."
