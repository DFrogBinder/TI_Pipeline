from pathlib import Path


SCRIPTS = {
    "run_post_processing_left_hippocampus_intact.slurm": {
        "batch_root": '/mnt/parscratch/users/cop23bi/defacing_sub_CCMe/defacing_prep/Left_Hippocampus_Intact',
        "job_name": 'ti_post_lhip_intact',
    },
    "run_post_processing_left_hippocampus_defaced.slurm": {
        "batch_root": '/mnt/parscratch/users/cop23bi/defacing_sub_CCMe/defacing_prep/Left_Hippocampus_Defaced',
        "job_name": 'ti_post_lhip_defaced',
    },
    "run_post_processing_left_m1_intact.slurm": {
        "batch_root": '/mnt/parscratch/users/cop23bi/defacing_sub_CCMe/defacing_prep/Left_M1_Intact',
        "job_name": 'ti_post_lm1_intact',
    },
    "run_post_processing_left_m1_defaced.slurm": {
        "batch_root": '/mnt/parscratch/users/cop23bi/defacing_sub_CCMe/defacing_prep/Left_M1_Defaced',
        "job_name": 'ti_post_lm1_defaced',
    },
}


def test_arm_specific_postprocess_wrappers_exist_and_pin_expected_roots():
    scripts_dir = Path(__file__).resolve().parents[1] / "HPC_scripts"
    common_runner = scripts_dir / "run_post_processing_batch_common.sh"
    assert common_runner.exists(), f"Missing common runner: {common_runner}"
    common_text = common_runner.read_text(encoding="utf-8")
    assert '"${PYTHON}" -u "${RUN_POST_BATCH_ENV_PY}"' in common_text

    for filename, expected in SCRIPTS.items():
        path = scripts_dir / filename
        assert path.exists(), f"Missing wrapper: {path}"
        text = path.read_text(encoding="utf-8")
        assert f'#SBATCH --job-name={expected["job_name"]}' in text
        assert f'BATCH_ROOT="{expected["batch_root"]}"' in text
        assert f'PIPELINE_PLOT_ROI="{expected.get("plot_roi", "")}"' in text
        assert 'PIPELINE_FASTSURFER_ROOT="/mnt/parscratch/users/cop23bi/ZIPs/atlases"' in text
        assert 'PIPELINE_FASTSURFER_ATLAS_FILENAME=""' in text
        assert 'PIPELINE_SUBJECTS="sub-CCMe"' in text
        assert 'PIPELINE_MNI_BASELINE_ROOT=""' in text
        assert 'PIPELINE_MNI_FIXED_ATLAS_PATH=""' in text
        assert 'PIPELINE_WRITE_NEIGHBOR_TABLE="0"' in text
        assert 'PIPELINE_WRITE_NEIGHBOR_VISUALIZATION="0"' in text
        assert 'BATCH_REPEATS="01 02 03 04 05 06 07 08 09 10"' in text
        assert 'source "${REPO_DIR}/HPC_scripts/run_post_processing_batch_common.sh"' in text
        assert 'source "${SCRIPT_DIR}/run_post_processing_batch_common.sh"' not in text


def test_common_runner_preflights_subject_space_atlas_for_fastsurfer_mode():
    common_runner = (
        Path(__file__).resolve().parents[1]
        / "HPC_scripts"
        / "run_post_processing_batch_common.sh"
    )
    common_text = common_runner.read_text(encoding="utf-8")
    assert 'missing_subject_atlases=()' in common_text
    assert '${PIPELINE_FASTSURFER_ROOT}/${subject}.nii.gz' in common_text
    assert 'Do not point this at the raw MNI atlas' in common_text
