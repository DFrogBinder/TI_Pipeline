#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMCAN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PIPELINE_DIR="$(cd "${CAMCAN_DIR}/.." && pwd)"

PREFLIGHT_ONLY=0
if [ "$#" -gt 1 ]; then
    echo "Usage: bash $0 [--preflight]" >&2
    exit 2
fi
if [ "$#" -eq 1 ]; then
    if [ "$1" != "--preflight" ]; then
        echo "[ERROR] Unknown option: $1" >&2
        exit 2
    fi
    PREFLIGHT_ONLY=1
fi

MNI401_OUTPUT_PARENT="${MNI401_OUTPUT_PARENT:-/mnt/parscratch/users/cop23bi/MNI152_SimNIBS401_validation}"
MNI_INPUT_ROOT="${MNI_INPUT_ROOT:-/mnt/parscratch/users/cop23bi/MNI152_SimNIBS401_inputs/m2m_MNI152}"
MNI_MESH_PATH="${MNI_MESH_PATH:-${MNI_INPUT_ROOT}/MNI152.msh}"
MNI_REFERENCE_T1_PATH="${MNI_REFERENCE_T1_PATH:-${MNI_INPUT_ROOT}/T1.nii.gz}"
MNI_EEG_CAP_PATH="${MNI_EEG_CAP_PATH:-${MNI_INPUT_ROOT}/eeg_positions/EEG10-10_UI_Jurak_2007.csv}"
TARGETS_CSV="${PIPELINE_DIR}/utils/targets.csv"
MNI_INPUT_MANIFEST="${CAMCAN_DIR}/simulation/mni152_head_model_manifest.sha256"

MNI_EXPECTED_MESH_SHA256="${MNI_EXPECTED_MESH_SHA256:-0f00843e7ec858b5bdb94904f25c57ffc811bf2a3546e73d577cec09fd5e0f35}"
MNI_EXPECTED_T1_SHA256="${MNI_EXPECTED_T1_SHA256:-5807425aa5ac6ce1f0800cc139d109b864ec117b53d5007a87c373e883f4d033}"
MNI_EXPECTED_EEG_CAP_SHA256="${MNI_EXPECTED_EEG_CAP_SHA256:-3c56ed91f685406919a43f361d581585b2d267fa6fb44f2b79a98c53142ec6a3}"
MNI_EXPECTED_INPUT_MANIFEST_SHA256="${MNI_EXPECTED_INPUT_MANIFEST_SHA256:-b42e9bf8d4cae6f21ed214e48107244c6fad79b83d14e4742fc7bed49e284664}"
MNI_EXPECTED_TARGETS_SHA256="${MNI_EXPECTED_TARGETS_SHA256:-97a8c7a72faf88d9af9e4facbdf628fba1a130d327da778bcbd00af66f2916e6}"

PARTITION="${PARTITION:-sheffield}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
COLLECTOR_CPUS="${COLLECTOR_CPUS:-4}"
COLLECTOR_MEMORY="${COLLECTOR_MEMORY:-16G}"
COLLECTOR_TIME="${COLLECTOR_TIME:-00:30:00}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
SQUEUE_BIN="${SQUEUE_BIN:-squeue}"
SCANCEL_BIN="${SCANCEL_BIN:-scancel}"

ARRAY_SLURM="${SCRIPT_DIR}/mni152_simnibs401_validation_array.slurm"
COLLECT_SLURM="${SCRIPT_DIR}/mni152_simnibs401_validation_collect.slurm"
RUNNER="${CAMCAN_DIR}/simulation/TI_runner_MNI152.py"
VALIDATOR="${CAMCAN_DIR}/simulation/validate_mni152_baseline.py"

if [ ! -d "${MNI_INPUT_ROOT}" ]; then
    cat >&2 <<EOF
[ERROR] The staged MNI152 head-model directory is missing:
        ${MNI_INPUT_ROOT}

The existing SimNIBS 4.5.0 baseline-output tree does not contain this source
head model. Upload the complete local directory before running preflight:
  local: /home/boyan/sandbox/Jake_Data/MNI152-data/m2m_MNI152/
  HPC:   ${MNI_INPUT_ROOT}/

See CamCan_Experiment/docs/README.md for the exact rsync and verification
commands. No jobs were submitted.
EOF
    exit 2
fi

for required_file in \
    "${ARRAY_SLURM}" \
    "${COLLECT_SLURM}" \
    "${RUNNER}" \
    "${VALIDATOR}" \
    "${MNI_INPUT_MANIFEST}" \
    "${MNI_MESH_PATH}" \
    "${MNI_REFERENCE_T1_PATH}" \
    "${MNI_EEG_CAP_PATH}" \
    "${TARGETS_CSV}"
do
    if [ ! -s "${required_file}" ]; then
        echo "[ERROR] Required file is missing or empty: ${required_file}" >&2
        exit 2
    fi
done
if [ "${MNI401_OUTPUT_PARENT}" = "${MNI_INPUT_ROOT}" ]; then
    echo "[ERROR] Validation outputs must not overwrite the staged MNI152 inputs." >&2
    exit 2
fi
for value_name in CPUS_PER_TASK MAX_CONCURRENT COLLECTOR_CPUS; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || [ "${value}" -lt 1 ]; then
        echo "[ERROR] ${value_name} must be a positive integer; got ${value}." >&2
        exit 2
    fi
done
if [ "${MAX_CONCURRENT}" -gt 4 ]; then
    echo "[ERROR] MAX_CONCURRENT cannot exceed the four scientific tasks." >&2
    exit 2
fi

ACTUAL_MESH_SHA256="$(sha256sum "${MNI_MESH_PATH}" | awk '{print $1}')"
ACTUAL_T1_SHA256="$(sha256sum "${MNI_REFERENCE_T1_PATH}" | awk '{print $1}')"
ACTUAL_EEG_CAP_SHA256="$(sha256sum "${MNI_EEG_CAP_PATH}" | awk '{print $1}')"
ACTUAL_TARGETS_SHA256="$(sha256sum "${TARGETS_CSV}" | awk '{print $1}')"
ACTUAL_INPUT_MANIFEST_SHA256="$(sha256sum "${MNI_INPUT_MANIFEST}" | awk '{print $1}')"
if [ "${ACTUAL_INPUT_MANIFEST_SHA256}" != "${MNI_EXPECTED_INPUT_MANIFEST_SHA256}" ]; then
    echo "[ERROR] Versioned MNI152 input manifest hash mismatch: ${ACTUAL_INPUT_MANIFEST_SHA256}" >&2
    exit 2
fi
if ! (
    cd "${MNI_INPUT_ROOT}"
    sha256sum --quiet --strict -c "${MNI_INPUT_MANIFEST}"
); then
    echo "[ERROR] The staged MNI152 head-model tree is incomplete or differs from the local source." >&2
    exit 2
fi
if [ "${ACTUAL_MESH_SHA256}" != "${MNI_EXPECTED_MESH_SHA256}" ]; then
    echo "[ERROR] MNI152 mesh hash mismatch: ${ACTUAL_MESH_SHA256}" >&2
    exit 2
fi
if [ "${ACTUAL_EEG_CAP_SHA256}" != "${MNI_EXPECTED_EEG_CAP_SHA256}" ]; then
    echo "[ERROR] MNI152 EEG cap hash mismatch: ${ACTUAL_EEG_CAP_SHA256}" >&2
    exit 2
fi
if [ "${ACTUAL_T1_SHA256}" != "${MNI_EXPECTED_T1_SHA256}" ]; then
    echo "[ERROR] MNI152 reference T1 hash mismatch: ${ACTUAL_T1_SHA256}" >&2
    exit 2
fi
if [ "${ACTUAL_TARGETS_SHA256}" != "${MNI_EXPECTED_TARGETS_SHA256}" ]; then
    echo "[ERROR] targets.csv hash mismatch: ${ACTUAL_TARGETS_SHA256}" >&2
    exit 2
fi

python3 - "${PIPELINE_DIR}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "CamCan_Experiment" / "simulation"))
from target_montages import resolve_montage_preset

for name in ("left-m1", "right-dlpfc", "left-hippocampus", "right-thalamus"):
    montage = resolve_montage_preset(name)
    print(
        "montage="
        f"{name}|{montage.pair1.anode}-{montage.pair1.cathode}|"
        f"{montage.pair1.current_amp:.16g}A|"
        f"{montage.pair2.anode}-{montage.pair2.cathode}|"
        f"{montage.pair2.current_amp:.16g}A"
    )
PY
export MPLCONFIGDIR="${MPLCONFIGDIR:-${MNI401_OUTPUT_PARENT}/.matplotlib}"
mkdir -p \
    "${MNI401_OUTPUT_PARENT}/logs" \
    "${MNI401_OUTPUT_PARENT}/validation_records" \
    "${MPLCONFIGDIR}"

printf '%s\n' \
    'Scope:' \
    '  purpose: isolate whether the SimNIBS version explains the MNI152 field discrepancy' \
    '  head models: 1 fixed MNI152 mesh (no meshing or remeshing)' \
    '  ROIs: 4 (Left_M1, Right_DLPC, Left_Hippocampus, Right_Thalamus)' \
    '  simulations: 4 (one deterministic fixed-mesh simulation per ROI)' \
    '  tDCS FEM solves: 8 (two electrode pairs per simulation)' \
    '  TImax meshes expected: 4' \
    '  brain-only TI NIfTIs expected: 4' \
    "  array: 0-3%${MAX_CONCURRENT}" \
    '  collector jobs: 1' \
    '  scheduler tasks total: 5' \
    '  execution: full requested four-ROI validation, not a smoke subset' \
    '  retries: none; failures remain visible for targeted manual recovery' \
    'Scientific invariant/change audit:' \
    '  unchanged: MNI152 head-model bundle, mesh, reference T1, EEG coordinates, montage electrodes/currents, conductivities, electrode geometry, element size, and TImax workflow' \
    '  changed: SimNIBS runtime 4.5.0 -> 4.0.1' \
    '  comparison location: local workstation after download; no 4.5.0 inputs are required on Stanage' \
    '  risk: a 4.0.1 API/output incompatibility causes a visible task failure and blocks result packaging'

printf '%s\n' \
    "[INFO] SimNIBS module:        SimNIBS/4.0.1-foss-2023a" \
    "[INFO] Isolated 4.0.1 parent: ${MNI401_OUTPUT_PARENT}" \
    "[INFO] Staged MNI input root: ${MNI_INPUT_ROOT}" \
    "[INFO] Input bundle files:    $(wc -l < "${MNI_INPUT_MANIFEST}")/17 verified" \
    "[INFO] Input manifest SHA:    ${ACTUAL_INPUT_MANIFEST_SHA256}" \
    "[INFO] MNI mesh:              ${MNI_MESH_PATH}" \
    "[INFO] MNI mesh SHA-256:      ${ACTUAL_MESH_SHA256}" \
    "[INFO] Reference T1:          ${MNI_REFERENCE_T1_PATH}" \
    "[INFO] Reference SHA-256:     ${ACTUAL_T1_SHA256}" \
    "[INFO] EEG cap:               ${MNI_EEG_CAP_PATH}" \
    "[INFO] EEG cap SHA-256:       ${ACTUAL_EEG_CAP_SHA256}" \
    "[INFO] targets.csv SHA-256:   ${ACTUAL_TARGETS_SHA256}" \
    "[INFO] Resource profile:      ${PARTITION}, ${CPUS_PER_TASK} CPU, ${MEMORY}, ${TIME_LIMIT}"

if [ "${PREFLIGHT_ONLY}" -eq 1 ]; then
    echo "[INFO] Preflight passed without submitting jobs."
    echo "[INFO] Submit with: bash CamCan_Experiment/HPC_scripts/submit_mni152_simnibs401_validation.sh"
    exit 0
fi

JOB_ID_FILE="${MNI401_OUTPUT_PARENT}/submitted_job_ids.txt"
if [ -s "${JOB_ID_FILE}" ] && command -v "${SQUEUE_BIN}" >/dev/null 2>&1; then
    PREVIOUS_IDS="$(
        awk '/^[0-9]+$/ { values = values separator $1; separator = "," } END { print values }' \
            "${JOB_ID_FILE}"
    )"
    if [ -n "${PREVIOUS_IDS}" ] && \
       [ -n "$("${SQUEUE_BIN}" -h -j "${PREVIOUS_IDS}" -o '%i' 2>/dev/null || true)" ]
    then
        echo "[ERROR] An earlier MNI152 version-validation submission is still active." >&2
        exit 2
    fi
fi

EXPORTS="ALL,PIPELINE_DIR=${PIPELINE_DIR},MNI401_OUTPUT_PARENT=${MNI401_OUTPUT_PARENT},MNI_INPUT_ROOT=${MNI_INPUT_ROOT},MNI_INPUT_MANIFEST=${MNI_INPUT_MANIFEST},MNI_INPUT_MANIFEST_SHA256=${ACTUAL_INPUT_MANIFEST_SHA256},MNI_MESH_PATH=${MNI_MESH_PATH},MNI_REFERENCE_T1_PATH=${MNI_REFERENCE_T1_PATH},MNI_EEG_CAP_PATH=${MNI_EEG_CAP_PATH},MNI_EXPECTED_MESH_SHA256=${MNI_EXPECTED_MESH_SHA256},MNI_EXPECTED_T1_SHA256=${MNI_EXPECTED_T1_SHA256},MNI_EXPECTED_EEG_CAP_SHA256=${MNI_EXPECTED_EEG_CAP_SHA256},MNI_EXPECTED_TARGETS_SHA256=${MNI_EXPECTED_TARGETS_SHA256}"
ARRAY_JOB="$(
    "${SBATCH_BIN}" \
        --parsable \
        --job-name="mni401_validate" \
        --partition="${PARTITION}" \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --mem="${MEMORY}" \
        --time="${TIME_LIMIT}" \
        --array="0-3%${MAX_CONCURRENT}" \
        --output="${MNI401_OUTPUT_PARENT}/logs/simulation-%A_%a.out" \
        --error="${MNI401_OUTPUT_PARENT}/logs/simulation-%A_%a.err" \
        --export="${EXPORTS}" \
        "${ARRAY_SLURM}"
)"
ARRAY_JOB="${ARRAY_JOB%%;*}"
printf '%s\n' "${ARRAY_JOB}" > "${JOB_ID_FILE}"

set +e
COLLECT_JOB="$(
    "${SBATCH_BIN}" \
        --parsable \
        --job-name="mni401_collect" \
        --partition="${PARTITION}" \
        --cpus-per-task="${COLLECTOR_CPUS}" \
        --mem="${COLLECTOR_MEMORY}" \
        --time="${COLLECTOR_TIME}" \
        --dependency="afterok:${ARRAY_JOB}" \
        --output="${MNI401_OUTPUT_PARENT}/logs/collector-%j.out" \
        --error="${MNI401_OUTPUT_PARENT}/logs/collector-%j.err" \
        --export="${EXPORTS}" \
        "${COLLECT_SLURM}"
)"
COLLECT_EXIT=$?
set -e
if [ "${COLLECT_EXIT}" -ne 0 ]; then
    echo "[ERROR] Collector submission failed; cancelling array ${ARRAY_JOB}." >&2
    "${SCANCEL_BIN}" "${ARRAY_JOB}" 2>/dev/null || true
    exit "${COLLECT_EXIT}"
fi
COLLECT_JOB="${COLLECT_JOB%%;*}"
printf '%s\n' "${COLLECT_JOB}" >> "${JOB_ID_FILE}"

printf '%s\n' \
    "[INFO] Submitted four-ROI MNI152 SimNIBS 4.0.1 array: ${ARRAY_JOB}" \
    "[INFO] Submitted dependent validated-result packager: ${COLLECT_JOB}" \
    "[INFO] Collector dependency: afterok:${ARRAY_JOB}" \
    "[INFO] Job IDs: ${JOB_ID_FILE}"
