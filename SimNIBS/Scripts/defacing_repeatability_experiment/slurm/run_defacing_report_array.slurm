#!/usr/bin/env bash
#SBATCH --job-name=deface-report
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

if [[ -z "${EXPERIMENT_ROOT:-}" ]]; then
  echo "EXPERIMENT_ROOT is required" >&2
  exit 2
fi

if [[ -z "${REPORT_TASK_MANIFEST:-}" ]]; then
  echo "REPORT_TASK_MANIFEST is required" >&2
  exit 2
fi

TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

python "${SCRIPT_DIR}/pipeline/staged_defacing_experiment.py" run-report-task \
  --task-manifest "${REPORT_TASK_MANIFEST}" \
  --task-index "${TASK_INDEX}"
