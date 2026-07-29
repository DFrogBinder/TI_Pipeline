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

MNI45_BASELINE_PARENT="${MNI45_BASELINE_PARENT:-/mnt/parscratch/users/cop23bi/ZIPs/MNI152-data}"
MNI401_OUTPUT_PARENT="${MNI401_OUTPUT_PARENT:-/mnt/parscratch/users/cop23bi/MNI152_SimNIBS401_validation}"
MNI_MESH_PATH="${MNI_MESH_PATH:-${MNI45_BASELINE_PARENT}/m2m_MNI152/MNI152.msh}"
MNI_REFERENCE_T1_PATH="${MNI_REFERENCE_T1_PATH:-${MNI45_BASELINE_PARENT}/m2m_MNI152/T1.nii.gz}"
MNI_FIXED_ATLAS_PATH="${MNI_FIXED_ATLAS_PATH:-/mnt/parscratch/users/cop23bi/ZIPs/atlases/sub-mni152.nii.gz}"
TARGETS_CSV="${PIPELINE_DIR}/utils/targets.csv"
TI_POST_PYTHON="${TI_POST_PYTHON:-/users/cop23bi/.conda/envs/ti-post/bin/python}"

MNI_EXPECTED_MESH_SHA256="${MNI_EXPECTED_MESH_SHA256:-0f00843e7ec858b5bdb94904f25c57ffc811bf2a3546e73d577cec09fd5e0f35}"
MNI_EXPECTED_T1_SHA256="${MNI_EXPECTED_T1_SHA256:-5807425aa5ac6ce1f0800cc139d109b864ec117b53d5007a87c373e883f4d033}"
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
COMPARISON="${CAMCAN_DIR}/post/compare_mni152_baseline_versions.py"
MANUSCRIPT_ANALYSIS="${CAMCAN_DIR}/post/camcan_manuscript_analysis.py"

for required_file in \
    "${ARRAY_SLURM}" \
    "${COLLECT_SLURM}" \
    "${RUNNER}" \
    "${VALIDATOR}" \
    "${COMPARISON}" \
    "${MANUSCRIPT_ANALYSIS}" \
    "${MNI_MESH_PATH}" \
    "${MNI_REFERENCE_T1_PATH}" \
    "${MNI_FIXED_ATLAS_PATH}" \
    "${TARGETS_CSV}" \
    "${TI_POST_PYTHON}"
do
    if [ ! -s "${required_file}" ]; then
        echo "[ERROR] Required file is missing or empty: ${required_file}" >&2
        exit 2
    fi
done
if [ "${MNI401_OUTPUT_PARENT}" = "${MNI45_BASELINE_PARENT}" ]; then
    echo "[ERROR] The 4.0.1 validation root must differ from the 4.5.0 baseline root." >&2
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
ACTUAL_TARGETS_SHA256="$(sha256sum "${TARGETS_CSV}" | awk '{print $1}')"
if [ "${ACTUAL_MESH_SHA256}" != "${MNI_EXPECTED_MESH_SHA256}" ]; then
    echo "[ERROR] MNI152 mesh hash mismatch: ${ACTUAL_MESH_SHA256}" >&2
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

OUTPUT_SUBJECTS=(
    MNI152-left-m1
    MNI152-right-dlpc
    MNI152-left-hippocampus
    MNI152-right-thalamus
)
MISSING_45=0
for output_subject in "${OUTPUT_SUBJECTS[@]}"; do
    baseline_root="${MNI45_BASELINE_PARENT}/${output_subject}/anat/SimNIBS"
    provenance="${baseline_root}/mni_baseline_provenance.json"
    ti_path="${baseline_root}/ti_brain_only.nii.gz"
    if [ ! -s "${provenance}" ] || [ ! -s "${ti_path}" ]; then
        echo "[ERROR] Existing 4.5.0 baseline is incomplete: ${output_subject}" >&2
        MISSING_45=$((MISSING_45 + 1))
        continue
    fi
    if ! grep -qF '"simnibs_version": "4.5.0"' "${provenance}"; then
        echo "[ERROR] Baseline provenance does not confirm SimNIBS 4.5.0: ${provenance}" >&2
        MISSING_45=$((MISSING_45 + 1))
    fi
done
if [ "${MISSING_45}" -ne 0 ]; then
    echo "[ERROR] ${MISSING_45} existing baseline(s) failed provenance validation." >&2
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
    "${MNI401_OUTPUT_PARENT}/comparison" \
    "${MPLCONFIGDIR}"
"${TI_POST_PYTHON}" -c \
    'import nibabel,numpy,pandas; print("mni_validation_post_dependencies=ready")'
"${TI_POST_PYTHON}" "${MANUSCRIPT_ANALYSIS}" \
    validate-atlas \
    --atlas "${MNI_FIXED_ATLAS_PATH}"

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
    '  unchanged: MNI152 mesh, reference T1, montage electrodes/currents, conductivities, electrode geometry, element size, and TImax workflow' \
    '  changed: SimNIBS runtime 4.5.0 -> 4.0.1' \
    '  isolation: existing 4.5.0 baseline tree is read-only; 4.0.1 writes to a separate root' \
    '  risk: a 4.0.1 API/output incompatibility causes a visible task failure and blocks comparison collection'

printf '%s\n' \
    "[INFO] SimNIBS module:        SimNIBS/4.0.1-foss-2023a" \
    "[INFO] Existing 4.5 parent:   ${MNI45_BASELINE_PARENT}" \
    "[INFO] Isolated 4.0.1 parent: ${MNI401_OUTPUT_PARENT}" \
    "[INFO] MNI mesh:              ${MNI_MESH_PATH}" \
    "[INFO] MNI mesh SHA-256:      ${ACTUAL_MESH_SHA256}" \
    "[INFO] Reference T1:          ${MNI_REFERENCE_T1_PATH}" \
    "[INFO] Reference SHA-256:     ${ACTUAL_T1_SHA256}" \
    "[INFO] targets.csv SHA-256:   ${ACTUAL_TARGETS_SHA256}" \
    "[INFO] Resource profile:      ${PARTITION}, ${CPUS_PER_TASK} CPU, ${MEMORY}, ${TIME_LIMIT}" \
    "[INFO] Comparison Python:     ${TI_POST_PYTHON}"

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

EXPORTS="ALL,PIPELINE_DIR=${PIPELINE_DIR},MNI401_OUTPUT_PARENT=${MNI401_OUTPUT_PARENT},MNI45_BASELINE_PARENT=${MNI45_BASELINE_PARENT},MNI_MESH_PATH=${MNI_MESH_PATH},MNI_REFERENCE_T1_PATH=${MNI_REFERENCE_T1_PATH},MNI_FIXED_ATLAS_PATH=${MNI_FIXED_ATLAS_PATH},MNI_EXPECTED_MESH_SHA256=${MNI_EXPECTED_MESH_SHA256},MNI_EXPECTED_T1_SHA256=${MNI_EXPECTED_T1_SHA256},MNI_EXPECTED_TARGETS_SHA256=${MNI_EXPECTED_TARGETS_SHA256},TI_POST_PYTHON=${TI_POST_PYTHON}"
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
        --job-name="mni401_compare" \
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
    "[INFO] Submitted dependent version-comparison collector: ${COLLECT_JOB}" \
    "[INFO] Collector dependency: afterok:${ARRAY_JOB}" \
    "[INFO] Job IDs: ${JOB_ID_FILE}"
