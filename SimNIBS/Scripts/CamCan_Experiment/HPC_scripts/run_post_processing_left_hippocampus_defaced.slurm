#!/bin/bash
#SBATCH --job-name=ti_post_lhip_defaced
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=post_left_hippocampus_defaced_%j.out
#SBATCH --error=post_left_hippocampus_defaced_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_DIR="/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment"
POST_CONDA_ENV="ti-post"
POST_ANACONDA_MODULE="Anaconda3/2022.05"

BATCH_ROOT="/mnt/parscratch/users/cop23bi/defacing_sub_CCMe/defacing_prep/Left_Hippocampus_Defaced"
BATCH_DATASET_GLOB="*_Data_*"
BATCH_REPEATS="01 02 03 04 05 06 07 08 09 10"
BATCH_SUMMARY_FILENAME="post_processing_batch_summary.json"
BATCH_STOP_ON_ERROR="0"

PIPELINE_ATLAS_MODE="fastsurfer"
PIPELINE_FASTSURFER_ROOT="/mnt/parscratch/users/cop23bi/ZIPs/atlases"
PIPELINE_FASTSURFER_ATLAS_FILENAME=""
PIPELINE_PLOT_ROI=""
PIPELINE_MNI_BASELINE_ROOT="/mnt/parscratch/users/cop23bi/ZIPs/MNI152-data/MNI152-left-hippocampus"
PIPELINE_MNI_FIXED_ATLAS_PATH="/mnt/parscratch/users/cop23bi/ZIPs/atlases/sub-mni152.nii.gz"

"${SCRIPT_DIR}/run_post_processing_batch_common.sh"
