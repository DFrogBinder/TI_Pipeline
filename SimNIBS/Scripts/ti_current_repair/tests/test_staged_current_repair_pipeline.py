import argparse
import csv
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

CURRENT_REPAIR_ROOT = Path(__file__).resolve().parents[1]
if str(CURRENT_REPAIR_ROOT) not in sys.path:
    sys.path.insert(0, str(CURRENT_REPAIR_ROOT))

from pipeline import provenance
from pipeline import setup_final132_right_m1_repeatability as right_m1_setup
from pipeline import staged_median_fixed_experiment as staged
from post import aggregate_paired_analysis
from post import extract_repeatability_mesh_metrics
from post import make_presentation_figures
from post import mesh_repeat_report
from post import repeatability_experiment_report
from post import seed_fixed_from_median
from post import select_median_remesh_repeats
from simulation_runners import repeatability_experiment
from stimulation_config import (
    CONFIRMED_TARGETS_SHA256,
    TARGETS_CSV_PATH,
    resolve_confirmed_stimulation,
    validate_stimulation_config,
)


def _write_config(path: Path, *, experiment_root: Path, subjects: list[str], repeat_count: int = 3) -> None:
    payload = {
        "source_root": str(experiment_root / "_source"),
        "experiment_root": str(experiment_root),
        "subjects": subjects,
        "conditions": [
            {"name": "remesh", "mesh_mode": "remesh", "repeat_count": repeat_count},
            {"name": "fixed_mesh", "mesh_mode": "fixed_mesh", "repeat_count": repeat_count},
        ],
        "stimulation": resolve_confirmed_stimulation(
            "left-hippocampus"
        ).to_dict(),
        "analysis": {"roi_preset": "left-hippocampus", "compare_metric": "median_roi"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["repeat_tag", "median_roi", "mean_roi", "peak_roi", "mesh_nodes"]
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _seed_remesh_anat(experiment_root: Path, subject: str, repeat_tag: str, *, mesh_text: str = "mesh\n") -> Path:
    anat = experiment_root / f"{subject}_repeatability" / "remesh" / "repeats" / repeat_tag / subject / "anat"
    m2m = anat / f"m2m_{subject}"
    m2m.mkdir(parents=True)
    (m2m / f"{subject}.msh").write_text(mesh_text, encoding="utf-8")
    for suffix in ("T1w.nii", "T2w.nii", "T1w_ras_1mm_T1andT2_masks.nii"):
        (anat / f"{subject}_{suffix}").write_text(f"{suffix}\n", encoding="utf-8")
    (anat / "SimNIBS" / "Output").mkdir(parents=True)
    (anat / "SimNIBS" / "Output" / "old.txt").write_text("old output\n", encoding="utf-8")
    (anat / ".mesh_build.lock").write_text("lock\n", encoding="utf-8")
    return anat


def _init_staged_experiment(
    root: Path,
    *,
    subjects: list[str],
    repeat_count: int,
    targets_csv: Path = TARGETS_CSV_PATH,
    roi_preset: str = "left-hippocampus",
) -> None:
    source = root / "_source"
    atlas_dir = root / "atlases"
    atlas_dir.mkdir(parents=True)
    for subject in subjects:
        anat = source / subject / "anat"
        anat.mkdir(parents=True)
        for suffix in (
            "_T1w.nii",
            "_T2w.nii",
            "_T1w_ras_1mm_T1andT2_masks.nii",
        ):
            (anat / f"{subject}{suffix}").write_text(f"{suffix}\n", encoding="utf-8")
        (atlas_dir / f"{subject}.nii.gz").write_text("atlas\n", encoding="utf-8")
    staged.main(
        [
            "init",
            "--source-root",
            str(source),
            "--experiment-root",
            str(root),
            "--subjects",
            ",".join(subjects),
            "--repeat-count",
            str(repeat_count),
            "--roi-preset",
            roi_preset,
            "--montage-preset",
            roi_preset,
            "--targets-csv",
            str(targets_csv),
            "--atlas-dir",
            str(atlas_dir),
        ]
    )


def test_confirmed_left_hippocampus_stimulation_is_loaded_from_targets_csv():
    stimulation = resolve_confirmed_stimulation("left-hippocampus")
    params = repeatability_experiment._ti_montage_parameters(stimulation)

    assert stimulation.targets_csv == TARGETS_CSV_PATH.resolve()
    assert stimulation.targets_csv_sha256 == CONFIRMED_TARGETS_SHA256
    assert stimulation.target_roi == "Left_Hippocampus"
    assert stimulation.configuration == 2759214
    assert params["montage_pair1"] == ("F8", 0.002, "P8", -0.002)
    assert params["montage_pair2"] == (
        "T7",
        pytest.approx(0.0015886564694485628),
        "P7",
        pytest.approx(-0.0015886564694485628),
    )


def test_confirmed_right_m1_stimulation_is_loaded_from_targets_csv():
    stimulation = resolve_confirmed_stimulation("right-m1")
    params = repeatability_experiment._ti_montage_parameters(stimulation)

    assert stimulation.targets_csv == TARGETS_CSV_PATH.resolve()
    assert stimulation.targets_csv_sha256 == CONFIRMED_TARGETS_SHA256
    assert stimulation.target_roi == "ctx_rh_G_precentral"
    assert stimulation.configuration == 137432
    assert params["montage_pair1"] == ("Fp2", 0.002, "F6", -0.002)
    assert params["montage_pair2"] == (
        "C4",
        pytest.approx(0.0006324555320336759),
        "CP2",
        pytest.approx(-0.0006324555320336759),
    )


def test_right_m1_analysis_preset_uses_destrieux_precentral_label():
    args = argparse.Namespace(
        roi_preset="right-m1",
        roi_name=None,
        roi_labels=None,
        m1_labels=None,
    )

    roi_name, labels = mesh_repeat_report._resolve_roi_selection(args)

    assert roi_name == "ctx_rh_G_precentral"
    assert labels == [12129]


def test_serialized_stimulation_must_exactly_match_confirmed_csv():
    payload = resolve_confirmed_stimulation("left-hippocampus").to_dict()
    payload["pair1"] = {
        "anode": "F10",
        "cathode": "P8",
        "current_a": 0.002,
    }

    with pytest.raises(ValueError, match="do not exactly match"):
        validate_stimulation_config(payload)


def test_workflow_preflight_rejects_targets_csv_drift(tmp_path):
    targets_copy = tmp_path / "targets.csv"
    targets_copy.write_bytes(TARGETS_CSV_PATH.read_bytes())
    root = tmp_path / "experiment"
    _init_staged_experiment(
        root,
        subjects=["sub-01"],
        repeat_count=2,
        targets_csv=targets_copy,
    )

    targets_copy.write_text("changed after initialization\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not the confirmed optimized file"):
        staged._workflow_scope(
            root,
            max_concurrent=50,
            analysis_max_concurrent=10,
        )


def test_show_plan_count_only_does_not_import_scientific_stack(tmp_path):
    root = tmp_path / "experiment"
    config = root / "_pipeline" / "configs" / "paired_analysis.json"
    _write_config(
        config,
        experiment_root=root,
        subjects=["sub-01", "sub-02"],
        repeat_count=1,
    )
    import_blocker = tmp_path / "import_blocker"
    import_blocker.mkdir()
    (import_blocker / "sitecustomize.py").write_text(
        "import builtins\n"
        "_real_import = builtins.__import__\n"
        "def _guarded_import(name, *args, **kwargs):\n"
        "    if name == 'nibabel' or name == 'numpy' or name == 'scipy' "
        "or name.startswith(('nibabel.', 'numpy.', 'scipy.')):\n"
        "        raise ModuleNotFoundError(f'blocked scientific import: {name}')\n"
        "    return _real_import(name, *args, **kwargs)\n"
        "builtins.__import__ = _guarded_import\n",
        encoding="utf-8",
    )
    runner = (
        CURRENT_REPAIR_ROOT
        / "simulation_runners"
        / "repeatability_experiment.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(runner),
            "show-plan",
            "--config",
            str(config),
            "--count-only",
        ],
        env={**os.environ, "PYTHONPATH": str(import_blocker)},
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.strip() == "4"
    assert result.stderr == ""


def _install_fake_scheduler(tmp_path: Path, monkeypatch) -> Path:
    counter = tmp_path / "fake_sbatch_counter"
    fake_sbatch = tmp_path / "fake_sbatch.sh"
    fake_sbatch.write_text(
        "#!/bin/sh\n"
        f"counter={counter}\n"
        "value=7000\n"
        "if [ -f \"$counter\" ]; then value=$(cat \"$counter\"); fi\n"
        "value=$((value + 1))\n"
        "printf '%s\\n' \"$value\" > \"$counter\"\n"
        "printf 'Submitted batch job %s\\n' \"$value\"\n",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)
    fake_scancel = tmp_path / "fake_scancel.sh"
    fake_scancel.write_text("#!/bin/sh\nprintf 'cancelled %s\\n' \"$1\"\n", encoding="utf-8")
    fake_scancel.chmod(0o755)
    monkeypatch.setenv("SBATCH_BIN", str(fake_sbatch))
    monkeypatch.setenv("SCANCEL_BIN", str(fake_scancel))
    monkeypatch.setenv("PYTHON_BIN", sys.executable)
    return counter


def _seed_ti_outputs(
    root: Path,
    *,
    subjects: list[str],
    condition: str,
    repeat_count: int,
) -> None:
    for subject in subjects:
        for index in range(1, repeat_count + 1):
            output = (
                root
                / f"{subject}_repeatability"
                / condition
                / "repeats"
                / f"repeat_{index:03d}"
                / subject
                / "anat"
                / "SimNIBS"
                / "Output"
                / subject
                / "TI.msh"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("TI mesh\n", encoding="utf-8")


def test_provenance_event_log_and_sbatch_job_id_parse(tmp_path):
    log = tmp_path / "events.jsonl"

    provenance.append_event(log, "submit", stage="remesh", job_id="12345")

    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["event"] == "submit"
    assert rows[0]["stage"] == "remesh"
    assert rows[0]["job_id"] == "12345"
    assert "timestamp_utc" in rows[0]
    assert provenance.parse_sbatch_job_id("Submitted batch job 87654\n") == "87654"
    assert provenance.parse_sbatch_job_id("sbatch --array=0-2 fake.slurm\n") is None


def test_select_median_from_just_generated_remesh_analysis(tmp_path):
    root = tmp_path / "experiment"
    subject = "sub-01"
    _write_summary(
        root / "_analysis" / subject / "remesh" / "summary.csv",
        [
            {"repeat_tag": "repeat_001", "median_roi": 2.0, "mean_roi": 2.1, "peak_roi": 5.0, "mesh_nodes": 100},
            {"repeat_tag": "repeat_002", "median_roi": 4.0, "mean_roi": 4.1, "peak_roi": 9.0, "mesh_nodes": 300},
            {"repeat_tag": "repeat_003", "median_roi": 6.0, "mean_roi": 6.1, "peak_roi": 12.0, "mesh_nodes": 200},
        ],
    )
    selected_anat = _seed_remesh_anat(root, subject, "repeat_002")

    out_csv = root / "_pipeline" / "median_mesh_selection" / "median_representative_remesh_repeats.csv"
    selections = select_median_remesh_repeats.select_medians(
        experiment_root=root,
        subjects=[subject],
        metric="median_roi",
        output_csv=out_csv,
    )

    assert selections[0].repeat_tag == "repeat_002"
    rows = list(csv.DictReader(out_csv.open("r", encoding="utf-8", newline="")))
    assert rows[0]["subject"] == subject
    assert rows[0]["selection_status"] == "selected"
    assert rows[0]["selected_repeat_tag"] == "repeat_002"
    assert rows[0]["selected_m2m_dir"] == str(selected_anat / f"m2m_{subject}")
    assert rows[0]["metric_value"] == "4.0"


def test_full_copy_seeder_excludes_outputs_and_rejects_symlinked_destinations(tmp_path):
    root = tmp_path / "experiment"
    subject = "sub-01"
    source_anat = _seed_remesh_anat(root, subject, "repeat_002")
    selection_csv = root / "_pipeline" / "median_mesh_selection" / "median_representative_remesh_repeats.csv"
    selection_csv.parent.mkdir(parents=True)
    with selection_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=select_median_remesh_repeats.output_fields())
        writer.writeheader()
        writer.writerow(
            {
                "subject": subject,
                "selection_status": "selected",
                "selected_repeat_tag": "repeat_002",
                "metric": "median_roi",
                "metric_value": "4.0",
                "median_target": "4.0",
                "selected_m2m_dir": str(source_anat / f"m2m_{subject}"),
                "selected_mesh_path": str(source_anat / f"m2m_{subject}" / f"{subject}.msh"),
                "mesh_nodes": "300",
                "mesh_checksum": provenance.file_sha256(source_anat / f"m2m_{subject}" / f"{subject}.msh"),
                "summary_csv": str(root / "_analysis" / subject / "remesh" / "summary.csv"),
            }
        )

    manifest = seed_fixed_from_median.seed_fixed_meshes(
        experiment_root=root,
        selection_csv=selection_csv,
        repeat_count=2,
        overwrite=True,
    )

    cache_anat = root / f"{subject}_repeatability" / "fixed_mesh" / "mesh_cache" / subject / "anat"
    repeat_anat = root / f"{subject}_repeatability" / "fixed_mesh" / "repeats" / "repeat_001" / subject / "anat"
    assert (cache_anat / f"m2m_{subject}" / f"{subject}.msh").read_text(encoding="utf-8") == "mesh\n"
    assert (repeat_anat / f"m2m_{subject}" / f"{subject}.msh").read_text(encoding="utf-8") == "mesh\n"
    assert not (cache_anat / "SimNIBS").exists()
    assert not (cache_anat / ".mesh_build.lock").exists()
    assert not any(path.is_symlink() for path in cache_anat.rglob("*"))
    assert not any(path.is_symlink() for path in repeat_anat.rglob("*"))
    ready = json.loads((cache_anat / ".mesh_ready.json").read_text(encoding="utf-8"))
    assert ready["status"] == "mesh_ready"
    assert ready["mesh_checksum"] == provenance.file_sha256(cache_anat / f"m2m_{subject}" / f"{subject}.msh")
    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8", newline="")))
    assert rows[0]["copy_mode"] == "physical_copy"
    assert rows[0]["validation_result"] == "ok"

    bad_link = cache_anat / "bad_link"
    bad_link.symlink_to(source_anat / f"{subject}_T1w.nii")
    with pytest.raises(seed_fixed_from_median.SymlinkValidationError):
        seed_fixed_from_median.validate_no_symlinks(cache_anat)


def test_stage_cli_init_configs_submitters_and_status(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    source = root / "_source"
    atlas_dir = root / "atlases"
    subject = "sub-01"
    (source / subject / "anat").mkdir(parents=True)
    atlas_dir.mkdir()
    (atlas_dir / f"{subject}.nii.gz").write_text("atlas\n", encoding="utf-8")

    staged.main(
        [
            "init",
            "--source-root",
            str(source),
            "--experiment-root",
            str(root),
            "--subjects",
            subject,
            "--repeat-count",
            "2",
            "--roi-preset",
            "left-hippocampus",
            "--montage-preset",
            "left-hippocampus",
            "--atlas-dir",
            str(atlas_dir),
        ]
    )

    pipeline_dir = root / "_pipeline"
    remesh_config = json.loads((pipeline_dir / "configs" / "remesh_only.json").read_text(encoding="utf-8"))
    fixed_config = json.loads((pipeline_dir / "configs" / "fixed_mesh_only.json").read_text(encoding="utf-8"))
    paired_config = json.loads((pipeline_dir / "configs" / "paired_analysis.json").read_text(encoding="utf-8"))
    assert [condition["name"] for condition in remesh_config["conditions"]] == ["remesh"]
    assert [condition["name"] for condition in fixed_config["conditions"]] == ["fixed_mesh"]
    assert [condition["name"] for condition in paired_config["conditions"]] == ["remesh", "fixed_mesh"]
    assert paired_config["analysis"]["atlas_dir"] == str(atlas_dir.resolve())
    assert paired_config["stimulation"]["montage_preset"] == "left-hippocampus"
    assert paired_config["stimulation"]["targets_csv_sha256"] == CONFIRMED_TARGETS_SHA256
    assert paired_config["stimulation"]["pair1"] == {
        "anode": "F8",
        "cathode": "P8",
        "current_a": 0.002,
    }
    assert paired_config["stimulation"]["pair2"] == {
        "anode": "T7",
        "cathode": "P7",
        "current_a": pytest.approx(0.0015886564694485628),
    }

    fake_sbatch = tmp_path / "fake_sbatch.sh"
    fake_sbatch.write_text(
        "#!/bin/sh\nprintf 'Submitted batch job 4242\\n'\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)
    monkeypatch.setenv("SBATCH_BIN", str(fake_sbatch))
    monkeypatch.setenv("PYTHON_BIN", sys.executable)

    staged.main(["submit-remesh", "--experiment-root", str(root), "--max-concurrent", "7"])
    staged.main(["analyze-remesh", "--experiment-root", str(root), "--max-concurrent", "3"])
    staged.main(["analyze-paired", "--experiment-root", str(root), "--max-concurrent", "3"])

    events = [json.loads(line) for line in (pipeline_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    submit_events = [row for row in events if row["event"] == "submit"]
    assert submit_events[0]["stage"] == "submit-remesh"
    assert submit_events[0]["job_id"] == "4242"
    assert "--array=0-1%7" in " ".join(submit_events[0]["command"])
    assert "--time=08:00:00" in submit_events[0]["command"]
    assert "--time=08:00:00" in submit_events[0]["stdout_tail"]
    assert submit_events[1]["stage"] == "analyze-remesh"
    assert submit_events[1]["job_id"] == "4242"
    assert "--array=0-0%3" in " ".join(submit_events[1]["command"])
    assert "--time=08:00:00" in submit_events[1]["command"]
    assert "--time=08:00:00" in submit_events[1]["stdout_tail"]
    assert submit_events[2]["stage"] == "analyze-paired"
    assert submit_events[2]["env"]["CONDITIONS"] == ""
    assert "CONDITIONS=" in " ".join(submit_events[2]["command"])
    assert "CONDITIONS=remesh,fixed_mesh" not in " ".join(submit_events[2]["command"])

    status = staged.collect_status(root)
    assert status["remesh_ti_msh"]["expected"] == 2
    assert status["remesh_ti_msh"]["observed"] == 0
    assert status["fixed_seed"]["expected"] == 1
    assert status["figure_outputs"]["expected"] >= 1


def test_submit_all_dry_run_prints_full_scope_without_submission(tmp_path, capsys):
    root = tmp_path / "experiment"
    _init_staged_experiment(
        root,
        subjects=["sub-01", "sub-02"],
        repeat_count=2,
    )

    staged.main(
        [
            "submit-all",
            "--experiment-root",
            str(root),
            "--max-concurrent",
            "50",
            "--analysis-max-concurrent",
            "10",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert "remesh tasks: 4 (0-3%50)" in output
    assert "fixed-mesh tasks: 4 (0-3%50)" in output
    assert "expected TI.msh outputs: 8" in output
    assert "pair 1: F8-P8 2 mA" in output
    assert "pair 2: T7-P7 1.58865646944856 mA" in output
    assert f"targets.csv SHA-256: {CONFIRMED_TARGETS_SHA256}" in output
    assert "not a smoke or subset" in output
    assert not (root / "_pipeline" / "workflow" / "submission.json").exists()


def test_final132_full_scope_is_400_then_400(tmp_path):
    root = tmp_path / "experiment"
    subjects = [
        "sub-CC110174",
        "sub-CC121144",
        "sub-CC310407",
        "sub-CC320616",
        "sub-CC420071",
        "sub-CC410432",
        "sub-CC520083",
        "sub-CC520127",
        "sub-CC610631",
        "sub-CC720941",
    ]
    _init_staged_experiment(root, subjects=subjects, repeat_count=40)

    scope = staged._workflow_scope(
        root,
        max_concurrent=50,
        analysis_max_concurrent=10,
    )

    assert scope["subject_count"] == 10
    assert scope["remesh_tasks"] == 400
    assert scope["fixed_mesh_tasks"] == 400
    assert scope["total_simulation_tasks"] == 800
    assert scope["remesh_array"] == "0-399%50"
    assert scope["fixed_mesh_array"] == "0-399%50"
    assert scope["analysis_array"] == "0-9%10"
    assert scope["expected_ti_msh"] == 800


def test_right_m1_setup_preflights_full_isolated_study(tmp_path, capsys):
    source_root = tmp_path / "staged"
    experiment_root = tmp_path / "right-m1-experiment"
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()
    for subject in right_m1_setup.SUBJECTS:
        anat_dir = source_root / subject / "anat"
        anat_dir.mkdir(parents=True)
        for suffix in staged.SOURCE_SUFFIXES:
            (anat_dir / f"{subject}{suffix}").write_text(
                f"{subject}{suffix}\n",
                encoding="utf-8",
            )
        (atlas_dir / f"{subject}.nii.gz").write_text(
            f"{subject} atlas\n",
            encoding="utf-8",
        )
    (source_root / "subjects.txt").write_text(
        "\n".join(right_m1_setup.SUBJECTS) + "\n",
        encoding="utf-8",
    )

    right_m1_setup.main(
        [
            "--preflight",
            "--source-root",
            str(source_root),
            "--experiment-root",
            str(experiment_root),
            "--atlas-dir",
            str(atlas_dir),
            "--targets-csv",
            str(TARGETS_CSV_PATH),
        ]
    )

    output = capsys.readouterr().out
    manifest = json.loads(
        (
            experiment_root / "_pipeline" / "experiment_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["subjects"] == right_m1_setup.SUBJECTS
    assert manifest["repeat_count"] == 40
    assert manifest["roi_preset"] == "right-m1"
    assert manifest["stimulation"]["montage_preset"] == "right-m1"
    assert manifest["stimulation"]["target_roi"] == "ctx_rh_G_precentral"
    assert "remesh tasks: 400 (0-399%50)" in output
    assert "fixed-mesh tasks: 400 (0-399%50)" in output
    assert "expected TI.msh outputs: 800" in output
    assert "pair 1: Fp2-F6 2 mA" in output
    assert "pair 2: C4-CP2 0.632455532033676 mA" in output
    assert "preflight passed without submitting jobs" in output
    assert not (
        experiment_root / "_pipeline" / "workflow" / "submission.json"
    ).exists()


def test_submit_all_preflight_verifies_staged_dataset_manifest_hashes(tmp_path):
    root = tmp_path / "experiment"
    subject = "sub-01"
    _init_staged_experiment(root, subjects=[subject], repeat_count=2)
    source = root / "_source"
    (source / "subjects.txt").write_text(f"{subject}\n", encoding="utf-8")
    destinations = [
        Path(subject) / "anat" / f"{subject}{suffix}"
        for suffix in (
            "_T1w.nii",
            "_T2w.nii",
            "_T1w_ras_1mm_T1andT2_masks.nii",
        )
    ]
    with (source / "dataset_manifest.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["destination", "sha256"], delimiter="\t")
        writer.writeheader()
        for destination in destinations:
            writer.writerow(
                {
                    "destination": str(destination),
                    "sha256": provenance.file_sha256(source / destination),
                }
            )

    scope = staged._workflow_scope(
        root,
        max_concurrent=50,
        analysis_max_concurrent=10,
    )
    assert scope["input_validation"]["manifest_hashes_verified"] == 3

    (source / destinations[0]).write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        staged._workflow_scope(
            root,
            max_concurrent=50,
            analysis_max_concurrent=10,
        )


def test_submit_all_attaches_afterok_controller_and_refuses_duplicate(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "experiment"
    _init_staged_experiment(root, subjects=["sub-01"], repeat_count=2)
    _install_fake_scheduler(tmp_path, monkeypatch)

    staged.main(
        [
            "submit-all",
            "--experiment-root",
            str(root),
            "--max-concurrent",
            "50",
            "--analysis-max-concurrent",
            "10",
        ]
    )

    submission = json.loads(
        (root / "_pipeline" / "workflow" / "submission.json").read_text(encoding="utf-8")
    )
    assert submission["initial_jobs"] == {
        "remesh": "7001",
        "after_remesh_controller": "7002",
    }
    controller = json.loads(
        (
            root
            / "_pipeline"
            / "submitted_jobs"
            / "workflow-controller-after-remesh.json"
        ).read_text(encoding="utf-8")
    )
    assert "--dependency=afterok:7001" in controller["command"]
    assert "--cpus-per-task=1" in controller["command"]
    assert "--mem=8G" in controller["command"]
    assert "--time=08:00:00" in controller["command"]
    job_rows = list(
        csv.DictReader(
            (root / "_pipeline" / "workflow" / "job_ids.tsv").open(
                encoding="utf-8", newline=""
            ),
            delimiter="\t",
        )
    )
    assert [(row["stage"], row["job_id"]) for row in job_rows] == [
        ("submit-remesh", "7001"),
        ("workflow-controller-after-remesh", "7002"),
    ]

    with pytest.raises(RuntimeError, match="already submitted"):
        staged.main(
            [
                "submit-all",
                "--experiment-root",
                str(root),
            ]
        )


def test_submit_all_retries_after_failed_pre_sbatch_receipt(
    tmp_path,
    monkeypatch,
    capsys,
):
    root = tmp_path / "experiment"
    _init_staged_experiment(root, subjects=["sub-01"], repeat_count=2)
    provenance.write_submitted_job_record(
        root,
        stage="submit-remesh",
        command=["python", "show-plan"],
        env={"PYTHON_BIN": "python"},
        stdout="",
        stderr="ModuleNotFoundError: No module named 'nibabel'",
        returncode=1,
        job_id=None,
        expected_outputs={"ti_msh": 2},
    )
    _install_fake_scheduler(tmp_path, monkeypatch)

    staged.main(
        [
            "submit-all",
            "--experiment-root",
            str(root),
            "--max-concurrent",
            "50",
            "--analysis-max-concurrent",
            "10",
        ]
    )

    output = capsys.readouterr().out
    assert "Retrying after failed pre-sbatch attempt" in output
    current = json.loads(
        (
            root / "_pipeline" / "submitted_jobs" / "submit-remesh.json"
        ).read_text(encoding="utf-8")
    )
    assert current["returncode"] == 0
    assert current["job_id"] == "7001"
    archived = list(
        (
            root
            / "_pipeline"
            / "submitted_jobs"
            / "attempts"
            / "submit-remesh"
        ).glob("*.json")
    )
    assert len(archived) == 1
    failed = json.loads(archived[0].read_text(encoding="utf-8"))
    assert failed["returncode"] == 1
    assert failed["job_id"] is None


def test_after_remesh_controller_validates_outputs_before_releasing_analysis(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "experiment"
    subjects = ["sub-01", "sub-02"]
    _init_staged_experiment(root, subjects=subjects, repeat_count=2)
    _install_fake_scheduler(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="observed 0, expected 4"):
        staged.main(
            [
                "advance-workflow",
                "--experiment-root",
                str(root),
                "--step",
                "after-remesh",
                "--max-concurrent",
                "50",
                "--analysis-max-concurrent",
                "10",
            ]
        )

    _seed_ti_outputs(
        root,
        subjects=subjects,
        condition="remesh",
        repeat_count=2,
    )
    staged.main(
        [
            "advance-workflow",
            "--experiment-root",
            str(root),
            "--step",
            "after-remesh",
            "--max-concurrent",
            "50",
            "--analysis-max-concurrent",
            "10",
        ]
    )

    receipt = json.loads(
        (
            root / "_pipeline" / "workflow" / "receipts" / "after-remesh.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["remesh_ti_msh"] == {"expected": 4, "observed": 4}
    assert receipt["report_job_id"] == "7001"
    assert receipt["next_controller_job_id"] == "7002"
    controller = json.loads(
        (
            root
            / "_pipeline"
            / "submitted_jobs"
            / "workflow-controller-select-seed.json"
        ).read_text(encoding="utf-8")
    )
    assert "--dependency=afterok:7001" in controller["command"]


def test_select_seed_controller_uses_median_mesh_then_releases_fixed(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "experiment"
    subject = "sub-01"
    _init_staged_experiment(root, subjects=[subject], repeat_count=2)
    _install_fake_scheduler(tmp_path, monkeypatch)
    _write_summary(
        root / "_analysis" / subject / "remesh" / "summary.csv",
        [
            {
                "repeat_tag": "repeat_001",
                "median_roi": 2.0,
                "mean_roi": 2.1,
                "peak_roi": 5.0,
                "mesh_nodes": 100,
            },
            {
                "repeat_tag": "repeat_002",
                "median_roi": 4.0,
                "mean_roi": 4.1,
                "peak_roi": 9.0,
                "mesh_nodes": 300,
            },
        ],
    )
    selected_anat = _seed_remesh_anat(root, subject, "repeat_001")

    staged.main(
        [
            "advance-workflow",
            "--experiment-root",
            str(root),
            "--step",
            "select-seed",
            "--max-concurrent",
            "50",
            "--analysis-max-concurrent",
            "10",
        ]
    )

    selection_rows = list(
        csv.DictReader(
            (
                root
                / "_pipeline"
                / "median_mesh_selection"
                / "median_representative_remesh_repeats.csv"
            ).open(encoding="utf-8", newline="")
        )
    )
    assert selection_rows[0]["selected_repeat_tag"] == "repeat_001"
    assert selection_rows[0]["selected_mesh_path"] == str(
        selected_anat / f"m2m_{subject}" / f"{subject}.msh"
    )
    seed_rows = list(
        csv.DictReader(
            (root / "_pipeline" / "fixed_seed_manifest.csv").open(
                encoding="utf-8", newline=""
            )
        )
    )
    assert seed_rows[0]["validation_result"] == "ok"
    assert seed_rows[0]["message"] == "seeded 3 destinations"
    receipt = json.loads(
        (
            root / "_pipeline" / "workflow" / "receipts" / "select-seed.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["fixed_job_id"] == "7001"
    assert receipt["next_controller_job_id"] == "7002"
    controller = json.loads(
        (
            root
            / "_pipeline"
            / "submitted_jobs"
            / "workflow-controller-after-fixed.json"
        ).read_text(encoding="utf-8")
    )
    assert "--dependency=afterok:7001" in controller["command"]


def test_select_seed_controller_rejects_incomplete_or_nonfinite_metric_rows(tmp_path):
    root = tmp_path / "experiment"
    subject = "sub-01"
    _init_staged_experiment(root, subjects=[subject], repeat_count=2)
    summary = root / "_analysis" / subject / "remesh" / "summary.csv"
    _write_summary(
        summary,
        [
            {
                "repeat_tag": "repeat_001",
                "median_roi": 2.0,
                "mean_roi": 2.1,
                "peak_roi": 5.0,
                "mesh_nodes": 100,
            }
        ],
    )

    with pytest.raises(RuntimeError, match="observed 0, expected 1"):
        staged.main(
            [
                "advance-workflow",
                "--experiment-root",
                str(root),
                "--step",
                "select-seed",
                "--max-concurrent",
                "50",
                "--analysis-max-concurrent",
                "10",
            ]
        )

    _write_summary(
        summary,
        [
            {
                "repeat_tag": "repeat_001",
                "median_roi": 2.0,
                "mean_roi": 2.1,
                "peak_roi": 5.0,
                "mesh_nodes": 100,
            },
            {
                "repeat_tag": "repeat_002",
                "median_roi": float("nan"),
                "mean_roi": 4.1,
                "peak_roi": 9.0,
                "mesh_nodes": 300,
            },
        ],
    )
    with pytest.raises(RuntimeError, match="finite median_roi rows"):
        staged.main(
            [
                "advance-workflow",
                "--experiment-root",
                str(root),
                "--step",
                "select-seed",
                "--max-concurrent",
                "50",
                "--analysis-max-concurrent",
                "10",
            ]
        )


def test_after_fixed_controller_releases_paired_analysis(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    subjects = ["sub-01", "sub-02"]
    _init_staged_experiment(root, subjects=subjects, repeat_count=2)
    _install_fake_scheduler(tmp_path, monkeypatch)
    _seed_ti_outputs(
        root,
        subjects=subjects,
        condition="fixed_mesh",
        repeat_count=2,
    )

    staged.main(
        [
            "advance-workflow",
            "--experiment-root",
            str(root),
            "--step",
            "after-fixed",
            "--max-concurrent",
            "50",
            "--analysis-max-concurrent",
            "10",
        ]
    )

    receipt = json.loads(
        (
            root / "_pipeline" / "workflow" / "receipts" / "after-fixed.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["fixed_mesh_ti_msh"] == {"expected": 4, "observed": 4}
    assert receipt["report_job_id"] == "7001"
    assert receipt["next_controller_job_id"] == "7002"
    paired_job = json.loads(
        (
            root / "_pipeline" / "submitted_jobs" / "analyze-paired.json"
        ).read_text(encoding="utf-8")
    )
    assert paired_job["env"]["CONDITIONS"] == ""
    controller = json.loads(
        (
            root
            / "_pipeline"
            / "submitted_jobs"
            / "workflow-controller-finalize.json"
        ).read_text(encoding="utf-8")
    )
    assert "--dependency=afterok:7001" in controller["command"]


def test_finalize_controller_builds_figures_and_completion_receipt(tmp_path):
    root = tmp_path / "experiment"
    subject = "sub-01"
    _init_staged_experiment(root, subjects=[subject], repeat_count=2)
    for condition, offset in (("remesh", 0.0), ("fixed_mesh", 0.5)):
        _write_summary(
            root / "_analysis" / subject / condition / "summary.csv",
            [
                {
                    "repeat_tag": "repeat_001",
                    "median_roi": 2.0 + offset,
                    "mean_roi": 2.1,
                    "peak_roi": 5.0,
                    "mesh_nodes": 100,
                },
                {
                    "repeat_tag": "repeat_002",
                    "median_roi": 4.0 + offset,
                    "mean_roi": 4.1,
                    "peak_roi": 9.0,
                    "mesh_nodes": 110,
                },
            ],
        )
    paired = root / "_analysis" / "paired_condition_summary.csv"
    paired.write_text(
        "subject,status,baseline_condition,comparison_condition,compare_metric,"
        "baseline_std,comparison_std,std_reduction_percent\n"
        "sub-01,complete,remesh,fixed_mesh,median_roi,1.4,0.7,50.0\n",
        encoding="utf-8",
    )

    staged.main(
        [
            "advance-workflow",
            "--experiment-root",
            str(root),
            "--step",
            "finalize",
            "--max-concurrent",
            "50",
            "--analysis-max-concurrent",
            "10",
        ]
    )

    completion = json.loads(
        (
            root / "_pipeline" / "workflow" / "complete.json"
        ).read_text(encoding="utf-8")
    )
    assert completion["status"] == "complete"
    assert completion["scope"]["total_simulation_tasks"] == 4
    assert completion["status_snapshot"]["figure_outputs"]["observed"] == 4
    assert (
        root
        / "_figures"
        / "presentation"
        / "01_primary_median_roi_repeat_distributions.png"
    ).is_file()


def test_controller_attachment_failure_cancels_unmanaged_child(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    cancelled = []
    monkeypatch.setattr(staged, "_submit_controller", lambda **_kwargs: (1, None))
    monkeypatch.setattr(
        staged,
        "_cancel_job",
        lambda _root, job_id, *, reason: cancelled.append((job_id, reason)),
    )

    with pytest.raises(RuntimeError, match="cancelled child job 8123"):
        staged._attach_controller_or_cancel(
            experiment_root=root,
            child_job_id="8123",
            step="after-remesh",
            max_concurrent=50,
            analysis_max_concurrent=10,
        )

    assert cancelled == [
        ("8123", "failed to attach workflow controller for after-remesh")
    ]


def test_init_rejects_missing_exact_subject_atlas(tmp_path):
    root = tmp_path / "experiment"
    source = root / "_source"
    atlas_dir = root / "atlases"
    subject = "sub-01"
    (source / subject / "anat").mkdir(parents=True)
    atlas_dir.mkdir()
    (atlas_dir / f"{subject}_aparc+aseg.nii.gz").write_text("wrong name\n", encoding="utf-8")

    with pytest.raises(SystemExit, match=r"sub-01\.nii\.gz"):
        staged.main(
            [
                "init",
                "--source-root",
                str(source),
                "--experiment-root",
                str(root),
                "--subjects",
                subject,
                "--repeat-count",
                "2",
                "--montage-preset",
                "left-hippocampus",
                "--atlas-dir",
                str(atlas_dir),
            ]
        )


def test_configured_atlas_dir_uses_exact_subject_filename(tmp_path):
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()
    wrong_name = atlas_dir / "sub-01_aparc+aseg.nii.gz"
    wrong_name.write_text("wrong name\n", encoding="utf-8")

    with pytest.raises(SystemExit, match=r"sub-01\.nii\.gz"):
        mesh_repeat_report._resolve_atlas_path("sub-01", None, str(atlas_dir), None)

    exact = atlas_dir / "sub-01.nii.gz"
    exact.write_text("atlas\n", encoding="utf-8")
    assert mesh_repeat_report._resolve_atlas_path("sub-01", None, str(atlas_dir), None) == exact


def test_percentile_summary_uses_finite_values():
    values = [1.0, 2.0, float("nan"), 3.0, 4.0]

    assert mesh_repeat_report._percentile_summary(values, 95.0) == pytest.approx(3.85)
    assert math.isnan(mesh_repeat_report._percentile_summary([float("nan")], 95.0))


def test_p95_metrics_are_in_paired_comparison_order():
    assert "p95_roi" in repeatability_experiment_report.KEY_COMPARISON_METRICS
    assert "p95_head" in repeatability_experiment_report.KEY_COMPARISON_METRICS


def test_report_array_submitter_builds_expected_array(tmp_path):
    root = tmp_path / "experiment"
    config = root / "_pipeline" / "configs" / "remesh_only.json"
    _write_config(config, experiment_root=root, subjects=["sub-01", "sub-02"], repeat_count=2)
    log_dir = root / "_pipeline" / "logs" / "reports"
    script = Path(__file__).resolve().parents[1] / "hpc_scripts" / "submit_repeatability_report_array.sh"

    result = subprocess.run(
        [
            "bash",
            str(script),
        ],
        env={
            **os.environ,
            "PIPELINE_DIR": str(Path(__file__).resolve().parents[1]),
            "EXPERIMENT_CONFIG": str(config),
            "LOG_DIR": str(log_dir),
            "SBATCH_BIN": "echo",
            "PYTHON_BIN": sys.executable,
            "MAX_CONCURRENT_TASKS": "4",
            "CONDITIONS": "remesh",
        },
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--array=0-1%4" in result.stdout
    assert "--time=08:00:00" in result.stdout
    assert "REPORT_TASK_PY=" in result.stdout
    assert "CONDITIONS=remesh" in result.stdout


def test_report_array_submitter_rejects_comma_condition_exports(tmp_path):
    root = tmp_path / "experiment"
    config = root / "_pipeline" / "configs" / "paired_analysis.json"
    _write_config(config, experiment_root=root, subjects=["sub-01"], repeat_count=2)
    script = CURRENT_REPAIR_ROOT / "hpc_scripts" / "submit_repeatability_report_array.sh"

    result = subprocess.run(
        ["bash", str(script)],
        env={
            **os.environ,
            "PIPELINE_DIR": str(CURRENT_REPAIR_ROOT),
            "EXPERIMENT_CONFIG": str(config),
            "LOG_DIR": str(root / "_pipeline" / "logs" / "reports"),
            "SBATCH_BIN": "echo",
            "PYTHON_BIN": sys.executable,
            "CONDITIONS": "remesh,fixed_mesh",
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "must not contain commas" in result.stderr


def test_presentation_figures_from_synthetic_analysis(tmp_path):
    root = tmp_path / "experiment"
    subject = "sub-01"
    manifest = root / "_pipeline" / "experiment_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"roi_preset": "right-m1"}) + "\n",
        encoding="utf-8",
    )
    for condition, offset in (("remesh", 0.0), ("fixed_mesh", 0.5)):
        _write_summary(
            root / "_analysis" / subject / condition / "summary.csv",
            [
                {"repeat_tag": "repeat_001", "median_roi": 2.0 + offset, "mean_roi": 2.1, "peak_roi": 5.0, "mesh_nodes": 100},
                {"repeat_tag": "repeat_002", "median_roi": 4.0 + offset, "mean_roi": 4.1, "peak_roi": 9.0, "mesh_nodes": 110},
            ],
        )
    paired = root / "_analysis" / "paired_condition_summary.csv"
    paired.parent.mkdir(parents=True, exist_ok=True)
    paired.write_text(
        "subject,status,baseline_condition,comparison_condition,compare_metric,baseline_std,comparison_std,std_reduction_percent\n"
        "sub-01,complete,remesh,fixed_mesh,median_roi,1.4,0.7,50.0\n",
        encoding="utf-8",
    )

    outputs = make_presentation_figures.make_figures(experiment_root=root)

    assert (root / "_figures" / "presentation" / "01_primary_median_roi_repeat_distributions.png").is_file()
    assert not (
        root
        / "_figures"
        / "presentation"
        / "condition_median_roi_by_repeat.png"
    ).exists()
    assert (root / "_figures" / "presentation" / "presentation_condition_summary.csv").is_file()
    caption_text = (
        root / "_figures" / "presentation" / "figure_captions.md"
    ).read_text(encoding="utf-8")
    assert "Repeat-level median TI E-field" in caption_text
    assert "All simulations otherwise use" not in caption_text
    assert (
        root / "_figures" / "presentation" / "figure_captions.csv"
    ).is_file()
    summary_rows = list(csv.DictReader((root / "_figures" / "presentation" / "presentation_condition_summary.csv").open("r", encoding="utf-8", newline="")))
    assert "p95_roi" in summary_rows[0]
    assert "p95_head" in summary_rows[0]
    assert outputs["figures_written"] >= 1
    assert outputs["roi_preset"] == "right-m1"
    assert outputs["roi_display_name"] == "right M1"
    assert any(path.endswith("01_primary_median_roi_repeat_distributions.png") for path in outputs["figures"])


def test_presentation_figures_include_element_and_rank_outputs(tmp_path):
    root = tmp_path / "experiment"
    for subject_index in range(10):
        subject = f"sub-{subject_index:02d}"
        for condition, offset in (("remesh", 0.0), ("fixed_mesh", 0.005)):
            _write_summary(
                root / "_analysis" / subject / condition / "summary.csv",
                [
                    {
                        "repeat_tag": f"repeat_{repeat_index:03d}",
                        "median_roi": 0.2 + offset + repeat_index / 10000,
                        "mean_roi": 0.2,
                        "peak_roi": 0.4,
                        "mesh_nodes": 300000 + repeat_index,
                        "mesh_elements": (
                            1_800_000
                            + subject_index * 1_000
                            + repeat_index * 10
                        ),
                        "mesh_elements_by_tissue": json.dumps(
                            {
                                1: 800_000 + repeat_index,
                                2: 700_000 + repeat_index,
                                3: 300_000 + repeat_index,
                            }
                        ),
                        "mesh_volume_mm3_by_tissue": json.dumps(
                            {
                                1: 510_000 + repeat_index,
                                2: 440_000 + 2 * repeat_index,
                                3: 210_000 + 3 * repeat_index,
                            }
                        ),
                    }
                    for repeat_index in range(1, 41)
                ],
            )

    make_presentation_figures.make_figures(experiment_root=root)

    figure = (
        root
        / "_figures"
        / "presentation"
        / "01_primary_median_roi_repeat_distributions.png"
    )
    with figure.open("rb") as handle:
        handle.seek(16)
        width, height = struct.unpack(">II", handle.read(8))
    if (width, height) == (900, 480):
        pytest.skip("Matplotlib unavailable; fallback renderer used")
    assert 1200 <= width <= 4000
    assert 700 <= height <= 3000
    presentation = root / "_figures" / "presentation"
    assert (
        presentation / "02_single_repeat_subject_ranking_uncertainty.png"
    ).is_file()
    assert (
        presentation / "03_primary_mesh_element_repeat_distributions.png"
    ).is_file()
    assert (
        presentation / "04_tissue_element_repeat_distributions.png"
    ).is_file()
    assert (
        presentation / "05_tissue_volume_repeat_distributions.png"
    ).is_file()
    assert (
        presentation / "06_example_subject_tissue_composition.png"
    ).is_file()
    uncertainty = json.loads(
        (
            presentation / "single_repeat_ranking_uncertainty.json"
        ).read_text(encoding="utf-8")
    )
    captions = (presentation / "figure_captions.md").read_text(
        encoding="utf-8"
    )
    assert "20,000 random selections" in captions
    assert "All simulations otherwise use" not in captions
    assert "Tissue-specific tetrahedral element counts" in captions
    assert "Tissue-specific tetrahedral mesh volume" in captions
    assert "sums to 100%" in captions
    assert uncertainty["random_single_repeat_selections"] == 20_000
    assert 0.0 <= uncertainty[
        "probability_of_any_subject_order_reversal"
    ] <= 1.0


def test_binary_gmsh22_mesh_statistics_include_tissue_volume(tmp_path):
    mesh = tmp_path / "small.msh"
    with mesh.open("wb") as handle:
        handle.write(b"$MeshFormat\n2.2 1 8\n")
        handle.write(struct.pack("<i", 1))
        handle.write(b"\n$EndMeshFormat\n$Nodes\n5\n")
        nodes = [
            (1, 0.0, 0.0, 0.0),
            (2, 1.0, 0.0, 0.0),
            (3, 0.0, 1.0, 0.0),
            (4, 0.0, 0.0, 1.0),
            (5, 0.0, 0.0, 2.0),
        ]
        for node in nodes:
            handle.write(struct.pack("<i3d", *node))
        handle.write(b"$EndNodes\n$Elements\n2\n")
        handle.write(struct.pack("<3i", 4, 2, 2))
        handle.write(struct.pack("<7i", 1, 1, 1, 1, 2, 3, 4))
        handle.write(struct.pack("<7i", 2, 2, 2, 1, 2, 3, 5))
        handle.write(b"\n$EndElements\n")

    nodes, elements, tissue_counts, tissue_volumes = (
        mesh_repeat_report._gmsh22_binary_mesh_statistics(mesh)
    )

    assert nodes == 5
    assert elements == 2
    assert tissue_counts == {1: 1, 2: 1}
    assert tissue_volumes[1] == pytest.approx(1.0 / 6.0)
    assert tissue_volumes[2] == pytest.approx(1.0 / 3.0)


def test_mesh_metric_refresh_collects_and_overlays_isolated_results(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "experiment"
    config_path = root / "_pipeline" / "configs" / "paired_analysis.json"
    subjects = ["sub-01", "sub-02"]
    _write_config(
        config_path,
        experiment_root=root,
        subjects=subjects,
        repeat_count=2,
    )
    completion = root / "_pipeline" / "workflow" / "complete.json"
    completion.parent.mkdir(parents=True, exist_ok=True)
    completion.write_text(
        json.dumps(
            {
                "status": "complete",
                "scope": {
                    "subject_count": 2,
                    "repeats_per_condition": 2,
                    "expected_ti_msh": 8,
                    "roi": "left-hippocampus",
                },
            }
        ),
        encoding="utf-8",
    )
    for subject_index, subject in enumerate(subjects):
        for condition in ("remesh", "fixed_mesh"):
            summary_rows = []
            for repeat_index in (1, 2):
                tag = f"repeat_{repeat_index:03d}"
                mesh = (
                    root
                    / f"{subject}_repeatability"
                    / condition
                    / "repeats"
                    / tag
                    / subject
                    / "anat"
                    / "SimNIBS"
                    / "Output"
                    / subject
                    / "TI.msh"
                )
                mesh.parent.mkdir(parents=True, exist_ok=True)
                mesh.write_bytes(b"mesh")
                summary_rows.append(
                    {
                        "repeat_tag": tag,
                        "median_roi": (
                            0.2
                            + subject_index / 100
                            + repeat_index / 1000
                        ),
                        "mean_roi": 0.2,
                        "peak_roi": 0.4,
                        "mesh_nodes": 100,
                    }
                )
            _write_summary(
                root
                / "_analysis"
                / subject
                / condition
                / "summary.csv",
                summary_rows,
            )

    monkeypatch.setattr(
        mesh_repeat_report,
        "_mesh_statistics",
        lambda path: (
            100.0,
            200.0,
            {1: 120, 2: 80},
            {1: 30.0, 2: 20.0},
        ),
    )
    output_root = root / "_post_processing" / "repeatability_mesh_metrics_v1"
    preflight = extract_repeatability_mesh_metrics.preflight(
        config_path=config_path,
        output_root=output_root,
    )
    assert preflight["expected_meshes"] == 8
    assert preflight["source_outputs_modified"] is False

    for subject_index in range(2):
        result = extract_repeatability_mesh_metrics.extract_subject(
            config_path=config_path,
            subject_index=subject_index,
            output_root=output_root,
        )
        assert result["rows"] == 4

    archive = output_root / "left_hippocampus_mesh_metrics.tar.gz"
    collected = extract_repeatability_mesh_metrics.collect(
        config_path=config_path,
        output_root=output_root,
        archive_path=archive,
    )
    assert collected["validation"]["rows"] == 8
    assert collected["source_outputs_modified"] is False
    assert archive.is_file()
    assert archive.with_suffix(archive.suffix + ".sha256").is_file()

    figures = make_presentation_figures.make_figures(
        experiment_root=root,
        output_dir=root / "_figures" / "mesh_overlay",
        mesh_metrics_csv=output_root / "mesh_metrics.csv",
    )
    assert figures["mesh_metrics_csv"] == str(
        output_root / "mesh_metrics.csv"
    )
    assert figures["data_availability"]["mesh_elements"] is True
    assert any(
        path.endswith("03_primary_mesh_element_repeat_distributions.png")
        for path in figures["figures"]
    )


def test_aggregate_paired_summary_from_per_subject_outputs(tmp_path):
    root = tmp_path / "experiment"
    subject_root = root / "_analysis" / "sub-01"
    subject_root.mkdir(parents=True)
    (subject_root / "condition_comparison.json").write_text(
        json.dumps(
            {
                "subject": "sub-01",
                "baseline_condition": "remesh",
                "comparison_condition": "fixed_mesh",
                "compare_metric": "median_roi",
                "metric_rows": [
                    {
                        "metric": "median_roi",
                        "baseline_std": 1.4,
                        "comparison_std": 0.7,
                        "std_ratio_comparison_over_baseline": 0.5,
                        "std_reduction_percent": 50.0,
                        "baseline_cv_percent": 10.0,
                        "comparison_cv_percent": 5.0,
                        "cv_reduction_percent": 50.0,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = aggregate_paired_analysis.aggregate_paired_summary(
        experiment_root=root,
        subjects=["sub-01"],
    )

    assert result["subjects_succeeded"] == 1
    rows = list(csv.DictReader((root / "_analysis" / "paired_condition_summary.csv").open("r", encoding="utf-8", newline="")))
    assert rows[0]["subject"] == "sub-01"
    assert rows[0]["baseline_condition"] == "remesh"
    assert rows[0]["comparison_condition"] == "fixed_mesh"
    assert rows[0]["std_reduction_percent"] == "50.0"
