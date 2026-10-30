#!/bin/bash

: "${PIPELINE_DIR:?Set PIPELINE_DIR to ti_current_repair.}"
: "${TDCS_REPLAY_VERSION:?Set TDCS_REPLAY_VERSION to v4 or v3.}"

module purge
module use "$HOME/modules"

case "$TDCS_REPLAY_VERSION" in
    v4)
        module load SimNIBS/4.0.1-foss-2023a
        export PYTHONPATH="$PIPELINE_DIR/hpc_compat_v4:$PIPELINE_DIR:${PYTHONPATH:-}"
        ;;
    v3)
        module load SimNIBS/3.2.6-foss-2023a
        V3_COMPAT="$PIPELINE_DIR/../simnibs_326_repeatability/hpc/activate_compat.sh"
        if [ ! -f "$V3_COMPAT" ]; then
            echo "[ERROR] Missing validated v3 compatibility shim: $V3_COMPAT" >&2
            exit 2
        fi
        # shellcheck source=/dev/null
        source "$V3_COMPAT"
        ;;
    *)
        echo "[ERROR] TDCS_REPLAY_VERSION must be v4 or v3." >&2
        exit 2
        ;;
esac

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
