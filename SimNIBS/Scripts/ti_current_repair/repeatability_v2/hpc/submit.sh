#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
SCRIPTS_DIR=$(cd "${PACKAGE_DIR}/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
ACTION=${1:-preflight}
V2_LOG_ROOT=${V2_LOG_ROOT:-${SCRIPTS_DIR}/logs}

: "${V2_PROFILE:?Set V2_PROFILE to the isolated v2 operational profile JSON}"
: "${V2_EVIDENCE:?Set V2_EVIDENCE to the preflight evidence JSON}"
: "${V2_OUTPUT_ROOT:?Set V2_OUTPUT_ROOT to the new isolated production root}"
: "${SIMNIBS_MODULE:?Set SIMNIBS_MODULE explicitly}"
: "${EXPECTED_SIMNIBS_VERSION:?Set EXPECTED_SIMNIBS_VERSION explicitly}"

if [[ "${EXPECTED_SIMNIBS_VERSION}" != "4.6.0" ]]; then
  echo "EXPECTED_SIMNIBS_VERSION must be 4.6.0" >&2
  exit 2
fi

cd "${SCRIPTS_DIR}"

run_preflight() {
  "${PYTHON_BIN}" -m ti_current_repair.repeatability_v2 preflight \
    --profile "${V2_PROFILE}" \
    --evidence "${V2_EVIDENCE}" \
    --output-root "${V2_OUTPUT_ROOT}"
}

if [[ "${ACTION}" == "preflight" ]]; then
  run_preflight
  exit 0
fi

if [[ "${ACTION}" != "run" ]]; then
  echo "Usage: submit.sh [preflight|run]" >&2
  exit 2
fi

: "${V2_LOCAL_VALIDATION:?Set the completed local validation receipt}"
: "${V2_HPC_PILOT:?Set the completed HPC pilot comparison receipt}"
"${PYTHON_BIN}" -m ti_current_repair.repeatability_v2 release-gate \
  --local-validation "${V2_LOCAL_VALIDATION}" \
  --hpc-pilot "${V2_HPC_PILOT}"
run_preflight
mkdir -p "${V2_LOG_ROOT}"

worker="${SCRIPT_DIR}/worker.slurm"
submit_stage() {
  local stage=$1
  local tasks=$2
  local concurrency=$3
  local cpus=$4
  local memory=$5
  local walltime=$6
  local dependency=${7:-}
  local args=(
    --parsable
    "--array=0-$((tasks - 1))%${concurrency}"
    "--cpus-per-task=${cpus}"
    "--mem=${memory}G"
    "--time=${walltime}"
    "--job-name=repv2-${stage}"
    "--output=${V2_LOG_ROOT}/%x-%A_%a.out"
    "--error=${V2_LOG_ROOT}/%x-%A_%a.err"
    "--export=ALL,V2_STAGE=${stage},V2_PROFILE=${V2_PROFILE}"
  )
  if [[ -n "${dependency}" ]]; then
    args+=("--dependency=afterok:${dependency}")
  fi
  sbatch "${args[@]}" "${worker}"
}

left_remesh=$(submit_stage left_hippocampus_remesh 400 50 8 32 08:00:00)
right_remesh=$(submit_stage right_m1_remesh 400 50 8 32 08:00:00)
remesh_extract=$(submit_stage remesh_native_extraction_validation 800 50 1 8 02:00:00 "${left_remesh}:${right_remesh}")
cache=$(submit_stage fixed_cache_preparation 20 20 1 8 02:00:00 "${remesh_extract}")
left_fixed=$(submit_stage left_hippocampus_fixed 400 50 8 32 08:00:00 "${cache}")
right_fixed=$(submit_stage right_m1_fixed 400 50 8 32 08:00:00 "${cache}")
final_extract=$(submit_stage final_native_extraction_geometry_validation 800 50 1 8 02:00:00 "${left_fixed}:${right_fixed}")
qc=$(submit_stage derived_nifti_qc 1600 50 1 8 02:00:00 "${final_extract}")

printf '%s\n' \
  "left_remesh=${left_remesh}" \
  "right_remesh=${right_remesh}" \
  "remesh_extract=${remesh_extract}" \
  "cache=${cache}" \
  "left_fixed=${left_fixed}" \
  "right_fixed=${right_fixed}" \
  "final_extract=${final_extract}" \
  "derived_qc=${qc}"

