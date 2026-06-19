#!/usr/bin/env bash
#SBATCH --job-name=deface-sim
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

if [[ -z "${TASK_MANIFEST:-}" ]]; then
  echo "TASK_MANIFEST is required" >&2
  exit 2
fi

if [[ -z "${CONFIG:-}" ]]; then
  echo "CONFIG is required" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

python "${SCRIPT_DIR}/pipeline/staged_defacing_experiment.py" run-simulation-task \
  --task-manifest "${TASK_MANIFEST}" \
  --task-index "${TASK_INDEX}"
