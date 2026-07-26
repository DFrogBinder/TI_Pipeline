#!/bin/bash

# Shared, source-only configuration helpers for the final-cohort
# post-processing array and collector.

post_roi_for_index() {
    case "$1" in
        0) printf '%s\n' "Left_Hippocampus" ;;
        1) printf '%s\n' "Left_M1" ;;
        2) printf '%s\n' "Right_DLPC" ;;
        3) printf '%s\n' "Right_Thalamus" ;;
        *)
            echo "[ERROR] ROI index must be between 0 and 3; got $1." >&2
            return 2
            ;;
    esac
}

post_mni_baseline_for_roi() {
    local roi="$1"
    local baseline_root="${MNI_BASELINE_PARENT:-/mnt/parscratch/users/cop23bi/ZIPs/MNI152-data}"
    case "${roi}" in
        Left_Hippocampus)
            printf '%s\n' "${baseline_root}/MNI152-left-hippocampus"
            ;;
        Left_M1)
            printf '%s\n' "${baseline_root}/MNI152-left-m1"
            ;;
        Right_DLPC)
            printf '%s\n' "${baseline_root}/MNI152-right-dlpc"
            ;;
        Right_Thalamus)
            printf '%s\n' "${baseline_root}/MNI152-right-thalamus"
            ;;
        *)
            echo "[ERROR] Unsupported post-processing ROI: ${roi}." >&2
            return 2
            ;;
    esac
}

configure_cohort_post_environment() {
    local roi="$1"

    : "${STUDY_ROOT:?Set STUDY_ROOT.}"
    : "${SUBJECTS_FILE:?Set SUBJECTS_FILE.}"
    : "${POST_CAMPAIGN_ROOT:?Set POST_CAMPAIGN_ROOT.}"
    : "${PIPELINE_DIR:?Set PIPELINE_DIR.}"

    export REPO_DIR="${PIPELINE_DIR}/CamCan_Experiment"
    export PYTHON="${PYTHON:-/users/cop23bi/.conda/envs/ti-post/bin/python}"
    export POST_CONDA_ENV=""
    export POST_VENV=""
    export BATCH_ROOT="${STUDY_ROOT}/runs/${roi}_Runs"
    export BATCH_DATASET_GLOB="${roi}_Data_*"
    export PIPELINE_SUBJECTS
    PIPELINE_SUBJECTS="$(
        awk 'NF > 0 && $1 !~ /^#/ { values = values separator $1; separator = " " } END { print values }' \
            "${SUBJECTS_FILE}"
    )"
    export PIPELINE_ATLAS_MODE="fastsurfer"
    export PIPELINE_FASTSURFER_ROOT="${FASTSURFER_ROOT:-/mnt/parscratch/users/cop23bi/ZIPs/atlases}"
    export PIPELINE_FASTSURFER_ATLAS_FILENAME=""
    export PIPELINE_FS_MRI_PATH=""
    export PIPELINE_T1_PATH=""
    export PIPELINE_MAX_WORKERS="${POST_WORKERS:-${SLURM_CPUS_PER_TASK:-12}}"
    export PIPELINE_PLOT_ROI=""
    export PIPELINE_PERCENTILE="${POST_PERCENTILE:-95.0}"
    export PIPELINE_HARD_THRESHOLD="${POST_HARD_THRESHOLD:-0.2}"
    export PIPELINE_OVERLAY_Z_OFFSET_MM="${POST_OVERLAY_Z_OFFSET_MM:-0.0}"
    export PIPELINE_OVERLAY_FULL_FIELD="${POST_OVERLAY_FULL_FIELD:-1}"
    export PIPELINE_WRITE_REGION_TABLE="1"
    export PIPELINE_REGION_PERCENTILE="${POST_REGION_PERCENTILE:-95.0}"
    export PIPELINE_OFFTARGET_THRESHOLD="${POST_OFFTARGET_THRESHOLD:-0.2}"
    export PIPELINE_MNI_BASELINE_ROOT
    PIPELINE_MNI_BASELINE_ROOT="$(post_mni_baseline_for_roi "${roi}")"
    export PIPELINE_MNI_FIXED_ATLAS_PATH="${MNI_FIXED_ATLAS_PATH:-/mnt/parscratch/users/cop23bi/ZIPs/atlases/sub-mni152.nii.gz}"
    export PIPELINE_NEIGHBOR_DILATION_ITER="${POST_NEIGHBOR_DILATION_ITER:-1}"
    export PIPELINE_CSF_LABELS="${POST_CSF_LABELS:-24}"
    export PIPELINE_SKULL_LABELS="${POST_SKULL_LABELS:-}"
    export PIPELINE_ELECTRODE_CSV=""
    export PIPELINE_ELECTRODE_DATASET_DIR=""
    export PIPELINE_ELECTRODE_NAMES=""
    export PIPELINE_EEG_POSITIONS_PATH_TEMPLATE=""
    export PIPELINE_CAMCAN_TARGETS_CSV="${TARGETS_CSV:-${PIPELINE_DIR}/utils/targets.csv}"
    export PIPELINE_EXPECTED_TARGETS_SHA256="${EXPECTED_TARGETS_SHA256:-97a8c7a72faf88d9af9e4facbdf628fba1a130d327da778bcbd00af66f2916e6}"
    export PIPELINE_WRITE_NEIGHBOR_TABLE="1"
    export PIPELINE_WRITE_NEIGHBOR_VISUALIZATION="1"
    export PIPELINE_WRITE_ELECTRODE_TABLE="1"
    export PIPELINE_FORCE="${POST_FORCE:-0}"
    export PIPELINE_VERBOSE="${POST_VERBOSE:-1}"
    export PIPELINE_POPULATION_REGION_FILENAME="region_stats_fastsurfer.csv"
    export PIPELINE_POPULATION_METRICS_FILENAME="subject_metrics.json"
    export PIPELINE_POPULATION_PEAK_THRESHOLD="${POST_POPULATION_PEAK_THRESHOLD:-0.2}"
    export PIPELINE_POPULATION_TARGET_ROI=""
    export PIPELINE_POPULATION_TEMPLATE_REGION_CSV=""
    export PIPELINE_COMPLETE_REPEAT_SUBJECTS_ONLY="1"
}
