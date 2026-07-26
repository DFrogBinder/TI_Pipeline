import os
import json
import hashlib
import subprocess
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "cohort_pipeline"


def test_post_roi_and_mni_baseline_mappings_are_complete():
    helper = PIPELINE_DIR / "cohort_post_common.sh"
    command = (
        f"source {helper!s}; "
        "for index in 0 1 2 3; do "
        "roi=$(post_roi_for_index \"$index\"); "
        "printf '%s|%s\\n' \"$roi\" \"$(post_mni_baseline_for_roi \"$roi\")\"; "
        "done"
    )
    completed = subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "MNI_BASELINE_PARENT": "/baseline",
        },
    )

    assert completed.stdout.splitlines() == [
        "Left_Hippocampus|/baseline/MNI152-left-hippocampus",
        "Left_M1|/baseline/MNI152-left-m1",
        "Right_DLPC|/baseline/MNI152-right-dlpc",
        "Right_Thalamus|/baseline/MNI152-right-thalamus",
    ]


def test_subject_array_covers_four_rois_and_ten_repeats():
    text = (PIPELINE_DIR / "cohort_post_subjects.slurm").read_text(
        encoding="utf-8"
    )

    assert (
        '${PIPELINE_DIR}/CamCan_Experiment/cohort_pipeline/'
        'cohort_post_common.sh'
    ) in text
    assert 'dirname "${BASH_SOURCE[0]}"' not in text
    assert "ROI_INDEX=$((SLURM_ARRAY_TASK_ID / 10))" in text
    assert "REPEAT_NUMBER=$((SLURM_ARRAY_TASK_ID % 10 + 1))" in text
    assert 'export PIPELINE_POPULATION_ENABLED="0"' in text
    assert 'export PIPELINE_REPEATABILITY_ENABLED="0"' in text
    assert 'export PIPELINE_FIGURE_GENERATION_ENABLED="0"' in text
    assert "run_post_processing_batch_common.sh" in text


def test_collector_runs_complete_case_population_repeatability_and_figures():
    text = (PIPELINE_DIR / "cohort_post_collect.slurm").read_text(
        encoding="utf-8"
    )

    assert (
        '${PIPELINE_DIR}/CamCan_Experiment/cohort_pipeline/'
        'cohort_post_common.sh'
    ) in text
    assert 'dirname "${BASH_SOURCE[0]}"' not in text
    assert 'export BATCH_REPEATS="01 02 03 04 05 06 07 08 09 10"' in text
    assert 'export PIPELINE_POPULATION_ENABLED="1"' in text
    assert 'export PIPELINE_REPEATABILITY_ENABLED="1"' in text
    assert 'export PIPELINE_FIGURE_GENERATION_ENABLED="1"' in text
    assert 'export PIPELINE_COMPLETE_REPEAT_SUBJECTS_ONLY="1"' in (
        PIPELINE_DIR / "cohort_post_common.sh"
    ).read_text(encoding="utf-8")


def test_submitter_pins_full_scope_and_afterok_collector():
    text = (PIPELINE_DIR / "submit_cohort_post_processing.sh").read_text(
        encoding="utf-8"
    )

    assert 'EXPECTED_POST_SUBJECT_TASKS=$((EXPECTED_SUBJECTS * 4 * 10))' in text
    assert 'SUBJECT_ARRAY_ELEMENTS=40' in text
    assert 'COLLECTOR_ARRAY_ELEMENTS=4' in text
    assert '--array="0-39%${MAX_CONCURRENT_DATASETS}"' in text
    assert '--array="0-3%${MAX_CONCURRENT_COLLECTORS}"' in text
    assert '--dependency="afterok:${SUBJECT_JOB}"' in text
    assert "--stage simulations" in text
    assert "--skip-hashes" in text
    assert "execution: full requested cohort; not a smoke or subset" in text
    assert "Missing subject-space atlas" in text
    assert "Missing ROI-specific MNI baseline TI output" in text


def test_submitter_preflight_accepts_complete_synthetic_full_scope(tmp_path):
    subject = "sub-CC000001"
    study_root = tmp_path / "study"
    campaign_root = study_root / "campaigns" / "test"
    post_root = campaign_root / "post_processing"
    subjects_file = tmp_path / "subjects.txt"
    subjects_file.write_text(f"{subject}\n", encoding="utf-8")
    cohort_config = tmp_path / "cohort.json"
    cohort_config.write_text(
        json.dumps({"expected_subjects": 1}),
        encoding="utf-8",
    )
    study_config = tmp_path / "study.json"
    study_config.write_text(
        json.dumps({"hpc_study_root": str(study_root)}),
        encoding="utf-8",
    )
    simulation_manifest = tmp_path / "simulation_tasks.tsv"
    simulation_manifest.write_text("task_id\n", encoding="utf-8")
    chain_receipt = campaign_root / "release_state" / "chain_complete.tsv"
    chain_receipt.parent.mkdir(parents=True)
    chain_receipt.write_text("status\tcomplete\n", encoding="utf-8")

    atlas_root = tmp_path / "atlases"
    atlas_root.mkdir()
    (atlas_root / f"{subject}.nii.gz").write_bytes(b"atlas")
    mni_atlas = atlas_root / "sub-mni152.nii.gz"
    mni_atlas.write_bytes(b"mni-atlas")

    baseline_parent = tmp_path / "MNI152-data"
    roi_baselines = {
        "Left_Hippocampus": "MNI152-left-hippocampus",
        "Left_M1": "MNI152-left-m1",
        "Right_DLPC": "MNI152-right-dlpc",
        "Right_Thalamus": "MNI152-right-thalamus",
    }
    for roi, baseline in roi_baselines.items():
        roi_root = study_root / "runs" / f"{roi}_Runs"
        for repeat in range(1, 11):
            (roi_root / f"{roi}_Data_{repeat:02d}").mkdir(parents=True)
        ti_path = baseline_parent / baseline / "anat" / "SimNIBS" / "ti_brain_only.nii.gz"
        ti_path.parent.mkdir(parents=True)
        ti_path.write_bytes(b"baseline")

    fake_workflow = tmp_path / "workflow.py"
    fake_workflow.write_text(
        "import json, pathlib, sys\n"
        "args = sys.argv\n"
        "summary = pathlib.Path(args[args.index('--summary') + 1])\n"
        "summary.parent.mkdir(parents=True, exist_ok=True)\n"
        "summary.write_text('task_id\\tstatus\\n', encoding='utf-8')\n"
        "payload = {'status': 'complete', 'complete': 40}\n"
        "summary.with_suffix('.json').write_text(json.dumps(payload), encoding='utf-8')\n"
        "print(json.dumps(payload))\n",
        encoding="utf-8",
    )
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/bash\nprintf '%s\\n' 'ti_post_dependencies=ready'\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    targets_csv = Path(__file__).resolve().parents[2] / "utils" / "targets.csv"
    targets_hash = hashlib.sha256(targets_csv.read_bytes()).hexdigest()
    submitter = PIPELINE_DIR / "submit_cohort_post_processing.sh"
    completed = subprocess.run(
        ["bash", str(submitter), "test", "--preflight"],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "STUDY_CONFIG": str(study_config),
            "COHORT_CONFIG": str(cohort_config),
            "STUDY_ROOT": str(study_root),
            "CAMPAIGN_ROOT": str(campaign_root),
            "POST_CAMPAIGN_ROOT": str(post_root),
            "SUBJECTS_FILE": str(subjects_file),
            "SIMULATION_MANIFEST": str(simulation_manifest),
            "CHAIN_RECEIPT": str(chain_receipt),
            "WORKFLOW_PY": str(fake_workflow),
            "FASTSURFER_ROOT": str(atlas_root),
            "MNI_FIXED_ATLAS_PATH": str(mni_atlas),
            "MNI_BASELINE_PARENT": str(baseline_parent),
            "TARGETS_CSV": str(targets_csv),
            "EXPECTED_TARGETS_SHA256": targets_hash,
            "PYTHON": str(fake_python),
            "SQUEUE_BIN": "/bin/true",
        },
    )

    assert "subject-level metric tasks: 40" in completed.stdout
    assert "repeat datasets: 40" in completed.stdout
    assert "Preflight passed without submitting jobs." in completed.stdout
