#!/usr/bin/env bash
#SBATCH --partition=sheffield
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -uo pipefail

: "${V2_STAGE:?V2_STAGE is required}"
: "${V2_PROFILE:?V2_PROFILE is required}"
: "${SIMNIBS_MODULE:?SIMNIBS_MODULE is required}"
: "${EXPECTED_SIMNIBS_VERSION:?EXPECTED_SIMNIBS_VERSION is required}"

if [[ "${EXPECTED_SIMNIBS_VERSION}" != "4.6.0" ]]; then
  echo "Expected SimNIBS 4.6.0, received ${EXPECTED_SIMNIBS_VERSION}" >&2
  exit 2
fi

module purge
module load "${SIMNIBS_MODULE}"

python -m ti_current_repair.repeatability_v2 run-stage \
  --profile "${V2_PROFILE}" \
  --stage "${V2_STAGE}" \
  --task-index "${SLURM_ARRAY_TASK_ID}"
rc=$?

if [[ ${rc} -ne 0 && ${SLURM_RESTART_COUNT:-0} -lt 1 ]]; then
  echo "Task failed with ${rc}; requesting the single automatic retry" >&2
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
fi

if [[ ${rc} -ne 0 ]]; then
  echo "Terminal failure after bounded retry; preserving append-only logs" >&2
fi
exit "${rc}"
