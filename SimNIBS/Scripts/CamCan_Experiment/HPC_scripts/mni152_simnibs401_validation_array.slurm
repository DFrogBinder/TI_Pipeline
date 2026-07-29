#!/bin/bash
#SBATCH --job-name=mni401_validate
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
umask 0027

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "[ERROR] Submit this worker as a four-element Slurm array." >&2
    exit 2
fi

: "${PIPELINE_DIR:?Set PIPELINE_DIR to the repository Scripts directory.}"
: "${MNI401_OUTPUT_PARENT:?Set the isolated SimNIBS 4.0.1 output parent.}"
: "${MNI45_BASELINE_PARENT:?Set the existing SimNIBS 4.5.0 baseline parent.}"
: "${MNI_MESH_PATH:?Set the exact fixed MNI152 mesh path.}"
: "${MNI_REFERENCE_T1_PATH:?Set the exact MNI152 reference T1 path.}"
: "${MNI_EXPECTED_MESH_SHA256:?Set the expected MNI152 mesh SHA-256.}"
: "${MNI_EXPECTED_T1_SHA256:?Set the expected reference T1 SHA-256.}"
: "${MNI_EXPECTED_TARGETS_SHA256:?Set the expected targets.csv SHA-256.}"

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        ROI="Left_M1"
        PRESET="left-m1"
        OUTPUT_SUBJECT="MNI152-left-m1"
        ;;
    1)
        ROI="Right_DLPC"
        PRESET="right-dlpfc"
        OUTPUT_SUBJECT="MNI152-right-dlpc"
        ;;
    2)
        ROI="Left_Hippocampus"
        PRESET="left-hippocampus"
        OUTPUT_SUBJECT="MNI152-left-hippocampus"
        ;;
    3)
        ROI="Right_Thalamus"
        PRESET="right-thalamus"
        OUTPUT_SUBJECT="MNI152-right-thalamus"
        ;;
    *)
        echo "[ERROR] Array index must be 0-3; got ${SLURM_ARRAY_TASK_ID}." >&2
        exit 2
        ;;
esac

RUNNER="${PIPELINE_DIR}/CamCan_Experiment/simulation/TI_runner_MNI152.py"
VALIDATOR="${PIPELINE_DIR}/CamCan_Experiment/simulation/validate_mni152_baseline.py"
SUMMARY="${MNI401_OUTPUT_PARENT}/validation_records/${OUTPUT_SUBJECT}.json"
for required_file in \
    "${RUNNER}" \
    "${VALIDATOR}" \
    "${MNI_MESH_PATH}" \
    "${MNI_REFERENCE_T1_PATH}"
do
    if [ ! -s "${required_file}" ]; then
        echo "[ERROR] Required input is missing or empty: ${required_file}" >&2
        exit 2
    fi
done
if [ "${MNI401_OUTPUT_PARENT}" = "${MNI45_BASELINE_PARENT}" ]; then
    echo "[ERROR] Validation output must not overwrite the 4.5.0 baseline tree." >&2
    exit 2
fi

module purge
module use "$HOME/modules"
module load SimNIBS/4.0.1-foss-2023a
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# SimNIBS 4.0.1 is used successfully by the main cohort with this NumPy
# compatibility shim. Gmsh auto-opening is disabled because this is a
# non-interactive batch run.
SHIM_DIR="${SLURM_TMPDIR:-/tmp}/${USER}_mni401_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${SHIM_DIR}"
cat > "${SHIM_DIR}/sitecustomize.py" <<'EOF'
try:
    import numpy as _np
    if not hasattr(_np, "bool"):
        _np.bool = _np.bool_
except Exception:
    pass

import builtins
_real_import = builtins.__import__

def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _real_import(name, globals, locals, fromlist, level)
    try:
        if name == "simnibs" or name.startswith("simnibs."):
            from simnibs import sim_struct as _sim_struct
            if not getattr(_sim_struct.SESSION, "_gmsh_disabled", False):
                _original_init = _sim_struct.SESSION.__init__
                def _new_init(self, *args, **kwargs):
                    _original_init(self, *args, **kwargs)
                    try:
                        self.open_in_gmsh = False
                    except Exception:
                        pass
                _sim_struct.SESSION.__init__ = _new_init
                _sim_struct.SESSION._gmsh_disabled = True
    except Exception:
        pass
    return mod

builtins.__import__ = _patched_import
EOF
export PYTHONPATH="${SHIM_DIR}:${PYTHONPATH:-}"

mkdir -p \
    "${MNI401_OUTPUT_PARENT}/logs" \
    "${MNI401_OUTPUT_PARENT}/validation_records"

printf '%s\n' \
    "[INFO] Isolated MNI152 SimNIBS-version validation" \
    "[INFO] Array index:           ${SLURM_ARRAY_TASK_ID}" \
    "[INFO] ROI:                   ${ROI}" \
    "[INFO] Montage preset:        ${PRESET}" \
    "[INFO] Output subject:        ${OUTPUT_SUBJECT}" \
    "[INFO] Runtime required:      SimNIBS 4.0.1" \
    "[INFO] Fixed MNI mesh:        ${MNI_MESH_PATH}" \
    "[INFO] Fixed reference T1:    ${MNI_REFERENCE_T1_PATH}" \
    "[INFO] Existing 4.5 baseline: ${MNI45_BASELINE_PARENT}" \
    "[INFO] New 4.0.1 output:      ${MNI401_OUTPUT_PARENT}" \
    "[INFO] Automatic retries:     none; failures remain visible"
module list 2>&1
python -c 'import simnibs; assert simnibs.__version__ == "4.0.1", simnibs.__version__; print("simnibs_version=" + simnibs.__version__)'

VALIDATION_ARGS=(
    --output-parent "${MNI401_OUTPUT_PARENT}"
    --output-subject "${OUTPUT_SUBJECT}"
    --preset "${PRESET}"
    --reference-t1 "${MNI_REFERENCE_T1_PATH}"
    --expected-simnibs-version "4.0.1"
    --expected-mni-mesh-sha256 "${MNI_EXPECTED_MESH_SHA256}"
    --expected-reference-t1-sha256 "${MNI_EXPECTED_T1_SHA256}"
    --expected-targets-sha256 "${MNI_EXPECTED_TARGETS_SHA256}"
    --summary "${SUMMARY}"
)

if [ -s "${SUMMARY}" ] && python -u "${VALIDATOR}" "${VALIDATION_ARGS[@]}"; then
    echo "[INFO] Reusing already validated ${OUTPUT_SUBJECT} output."
    exit 0
fi

python -u "${RUNNER}" \
    --root-dir "${MNI401_OUTPUT_PARENT}" \
    --output-subject "${OUTPUT_SUBJECT}" \
    --mni-mesh-path "${MNI_MESH_PATH}" \
    --reference-t1-path "${MNI_REFERENCE_T1_PATH}" \
    --preset "${PRESET}" \
    --expected-simnibs-version "4.0.1"

python -u "${VALIDATOR}" "${VALIDATION_ARGS[@]}"
echo "[INFO] Completed and validated ${OUTPUT_SUBJECT} under SimNIBS 4.0.1."
