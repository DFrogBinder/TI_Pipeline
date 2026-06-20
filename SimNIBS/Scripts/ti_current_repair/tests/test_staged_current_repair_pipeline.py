import csv
import json
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
from pipeline import staged_median_fixed_experiment as staged
from post import aggregate_paired_analysis
from post import make_presentation_figures
from post import mesh_repeat_report
from post import seed_fixed_from_median
from post import select_median_remesh_repeats


def _write_config(path: Path, *, experiment_root: Path, subjects: list[str], repeat_count: int = 3) -> None:
    payload = {
        "source_root": str(experiment_root / "_source"),
        "experiment_root": str(experiment_root),
        "subjects": subjects,
        "conditions": [
            {"name": "remesh", "mesh_mode": "remesh", "repeat_count": repeat_count},
            {"name": "fixed_mesh", "mesh_mode": "fixed_mesh", "repeat_count": repeat_count},
        ],
        "analysis": {"roi_preset": "left-hippocampus", "compare_metric": "median_roi"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["repeat_tag", "median_roi", "mean_roi", "peak_roi", "mesh_nodes"])
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

    fake_sbatch = tmp_path / "fake_sbatch.sh"
    fake_sbatch.write_text("#!/bin/sh\nprintf 'Submitted batch job 4242\\n'\n", encoding="utf-8")
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
    assert submit_events[1]["stage"] == "analyze-remesh"
    assert submit_events[1]["job_id"] == "4242"
    assert "--array=0-0%3" in " ".join(submit_events[1]["command"])
    assert submit_events[2]["stage"] == "analyze-paired"
    assert submit_events[2]["env"]["CONDITIONS"] == ""
    assert "CONDITIONS=" in " ".join(submit_events[2]["command"])
    assert "CONDITIONS=remesh,fixed_mesh" not in " ".join(submit_events[2]["command"])

    status = staged.collect_status(root)
    assert status["remesh_ti_msh"]["expected"] == 2
    assert status["remesh_ti_msh"]["observed"] == 0
    assert status["fixed_seed"]["expected"] == 1
    assert status["figure_outputs"]["expected"] >= 1


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
    assert (root / "_figures" / "presentation" / "condition_median_roi_by_repeat.png").is_file()
    assert (root / "_figures" / "presentation" / "presentation_condition_summary.csv").is_file()
    assert outputs["figures_written"] >= 1
    assert any(path.endswith("01_primary_median_roi_repeat_distributions.png") for path in outputs["figures"])


def test_presentation_figures_use_legible_subject_panels(tmp_path):
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
                    }
                    for repeat_index in range(1, 41)
                ],
            )

    make_presentation_figures.make_figures(experiment_root=root)

    figure = root / "_figures" / "presentation" / "condition_median_roi_by_repeat.png"
    with figure.open("rb") as handle:
        handle.seek(16)
        width, height = struct.unpack(">II", handle.read(8))
    if (width, height) == (900, 480):
        pytest.skip("Matplotlib unavailable; fallback renderer used")
    assert 1200 <= width <= 3000
    assert 1500 <= height <= 4000


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
