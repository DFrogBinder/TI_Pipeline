#!/bin/bash
#SBATCH --job-name=ti_post_lhip_intact
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=post_left_hippocampus_intact_%j.out
#SBATCH --error=post_left_hippocampus_intact_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_DIR="/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment"
POST_CONDA_ENV="ti-post"
POST_ANACONDA_MODULE="Anaconda3/2022.05"

BATCH_ROOT="/mnt/parscratch/users/cop23bi/defacing_sub_CCMe/defacing_prep/Left_Hippocampus_Intact"
BATCH_DATASET_GLOB="*_Data_*"
BATCH_REPEATS="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40"
BATCH_SUMMARY_FILENAME="post_processing_batch_summary.json"
BATCH_STOP_ON_ERROR="0"

PIPELINE_ATLAS_MODE="fastsurfer"
PIPELINE_FASTSURFER_ROOT="/mnt/parscratch/users/cop23bi/ZIPs/atlases"
PIPELINE_FASTSURFER_ATLAS_FILENAME=""
PIPELINE_SUBJECTS="sub-CCMe"
PIPELINE_PLOT_ROI=""
PIPELINE_MNI_BASELINE_ROOT=""
PIPELINE_MNI_FIXED_ATLAS_PATH=""
PIPELINE_WRITE_NEIGHBOR_TABLE="0"
PIPELINE_WRITE_NEIGHBOR_VISUALIZATION="0"

source "${REPO_DIR}/HPC_scripts/run_post_processing_batch_common.sh"
