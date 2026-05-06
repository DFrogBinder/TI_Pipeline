#!/bin/bash

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <directory> <output_name>"
  exit 1
fi

INPUT_DIR="${1%/}"
OUTPUT_NAME="$2"

if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: directory '$INPUT_DIR' does not exist."
  exit 1
fi

if ! command -v pigz >/dev/null 2>&1; then
  echo "Error: pigz is not available in PATH."
  echo "Try loading it first, e.g. with your HPC module system."
  exit 1
fi

# Size in GiB
SIZE_GIB=$(du -sBG "$INPUT_DIR" | awk '{gsub(/G/,"",$1); print $1}')

# Fallback if du returns 0 for a tiny directory
if [ -z "$SIZE_GIB" ] || [ "$SIZE_GIB" -lt 1 ]; then
  SIZE_GIB=1
fi

# Heuristic resource selection
if [ "$SIZE_GIB" -le 10 ]; then
  CPUS=2
  MEM=4G
  TIME=01:00:00
elif [ "$SIZE_GIB" -le 50 ]; then
  CPUS=4
  MEM=8G
  TIME=02:00:00
elif [ "$SIZE_GIB" -le 200 ]; then
  CPUS=8
  MEM=16G
  TIME=06:00:00
elif [ "$SIZE_GIB" -le 500 ]; then
  CPUS=12
  MEM=24G
  TIME=12:00:00
else
  CPUS=16
  MEM=32G
  TIME=24:00:00
fi

JOB_SCRIPT=$(mktemp /tmp/compress_job.XXXXXX.sbatch)

cat >"$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=compress_$(basename "$INPUT_DIR")
#SBATCH --output=${OUTPUT_NAME}_%j.log
#SBATCH --error=${OUTPUT_NAME}_%j.err
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}

set -euo pipefail

INPUT_DIR="${INPUT_DIR}"
OUTPUT_NAME="${OUTPUT_NAME}"

echo "Job started on: \$(hostname)"
echo "Input directory: \$INPUT_DIR"
echo "Output archive: \${OUTPUT_NAME}.tar.gz"
echo "CPUs: ${CPUS}"
echo "Memory: ${MEM}"
echo "Time limit: ${TIME}"

if ! command -v pigz >/dev/null 2>&1; then
    echo "pigz not found on compute node."
    exit 1
fi

tar -cf - "\$INPUT_DIR" | pigz -p \$SLURM_CPUS_PER_TASK > "\${OUTPUT_NAME}.tar.gz"

echo "Compression complete: \${OUTPUT_NAME}.tar.gz"
EOF

chmod +x "$JOB_SCRIPT"

echo "Detected directory size: ${SIZE_GIB} GiB"
echo "Submitting with: cpus=${CPUS}, mem=${MEM}, time=${TIME}"
sbatch "$JOB_SCRIPT"
