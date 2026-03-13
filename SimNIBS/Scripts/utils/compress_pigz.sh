#!/bin/bash

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage:"
  echo "  $0 <file_or_directory> <output_name>"
  exit 1
fi

INPUT="$1"
OUTPUT="$2"

if [ ! -e "$INPUT" ]; then
  echo "Error: '$INPUT' does not exist"
  exit 1
fi

if ! command -v pigz >/dev/null 2>&1; then
  echo "Error: pigz is not installed."
  echo "Install it with:"
  echo "  sudo apt install pigz"
  exit 1
fi

CORES=$(nproc)

echo "Input: $INPUT"
echo "Output: ${OUTPUT}.tar.gz"
echo "Using $CORES CPU cores"

START=$(date +%s)

tar -cf - "$INPUT" | pigz -p "$CORES" >"${OUTPUT}.tar.gz"

END=$(date +%s)

echo "Compression finished in $((END - START)) seconds"
echo "Archive created: ${OUTPUT}.tar.gz"
