#!/bin/bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <file.gz>"
  exit 1
fi

INPUT="$1"

if [ ! -f "$INPUT" ]; then
  echo "Error: '$INPUT' does not exist."
  exit 1
fi

if [[ "$INPUT" != *.gz ]]; then
  echo "Error: input must end in .gz"
  exit 1
fi

if ! command -v pigz >/dev/null 2>&1; then
  echo "Error: pigz is not installed or not in PATH."
  exit 1
fi

CORES=$(nproc)

echo "Input: $INPUT"
echo "Using $CORES CPU cores"

START=$(date +%s)

pigz -d -p "$CORES" -k "$INPUT"

END=$(date +%s)

echo "Decompression finished in $((END - START)) seconds"
echo "Output file: ${INPUT%.gz}"
