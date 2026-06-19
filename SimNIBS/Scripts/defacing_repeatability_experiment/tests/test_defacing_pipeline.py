import csv
import json
import math
import os
import stat
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SimNIBS.Scripts.defacing_repeatability_experiment.pipeline import (
    staged_defacing_experiment as pipeline,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_fake_nifti(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_source_root(tmp_path: Path, subject: str = "sub-001") -> Path:
    source_root = tmp_path / "source"
    write_fake_nifti(source_root / subject / "anat" / f"{subject}_T1w.nii.gz", "t1-data")
    write_fake_nifti(source_root / subject / "anat" / f"{subject}_T2w.nii.gz", "t2-data")
    return source_root


def init_experiment(tmp_path: Path, repeat_full: int = 40, repeat_defaced: int = 40) -> Path:
    experiment_root = tmp_path / "experiment"
    pipeline.init_experiment(
        source_root=make_source_root(tmp_path),
        experiment_root=experiment_root,
        subject="sub-001",
        repeat_count_full_face=repeat_full,
        repeat_count_defaced=repeat_defaced,
        roi_preset="left-hippocampus",
    )
    return experiment_root


def test_init_writes_two_condition_config_and_stage_state(tmp_path: Path):
    experiment_root = init_experiment(tmp_path)

    config = read_json(experiment_root / "_pipeline" / "config.json")
    conditions = {item["name"]: item for item in config["conditions"]}
    assert conditions["full_face"] == {
        "name": "full_face",
        "input_mode": "original",
        "repeat_count": 40,
    }
    assert conditions["defaced"] == {
        "name": "defaced",
        "input_mode": "fsl_deface",
        "repeat_count": 40,
    }
    assert config["simulation"]["pair1"]["anode"] == "F10"
    assert config["simulation"]["pair1"]["cathode"] == "P8"
    assert config["simulation"]["pair2"]["anode"] == "T7"
    assert config["simulation"]["pair2"]["cathode"] == "P7"
    assert config["roi"]["preset"] == "left-hippocampus"

    status = read_json(experiment_root / "_pipeline" / "stage_status.json")
    assert status["init"]["status"] == "complete"

    events = (experiment_root / "_pipeline" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(events) == 1
    assert json.loads(events[0])["event"] == "init.complete"


def test_prepare_inputs_records_original_paths_hashes_and_full_face_links(tmp_path: Path):
    experiment_root = init_experiment(tmp_path)

    manifest = pipeline.prepare_inputs(experiment_root)

    assert manifest["subject"] == "sub-001"
    assert manifest["original"]["t1"]["sha256"]
    assert manifest["original"]["t2"]["sha256"]
    assert Path(manifest["conditions"]["full_face"]["t1"]).exists()
    assert Path(manifest["conditions"]["full_face"]["t2"]).exists()
    assert read_json(experiment_root / "_pipeline" / "input_manifest.json") == manifest


def test_deface_inputs_invokes_fsl_deface_for_t1_and_t2_and_writes_manifest(tmp_path: Path):
    experiment_root = init_experiment(tmp_path)
    pipeline.prepare_inputs(experiment_root)
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        Path(cmd[2]).write_text(f"defaced from {cmd[1]}", encoding="utf-8")
        return pipeline.CommandResult(args=cmd, returncode=0, stdout="", stderr="")

    manifest = pipeline.deface_inputs(experiment_root, command_runner=fake_run)

    assert [cmd[0] for cmd in calls] == ["fsl_deface", "fsl_deface"]
    assert calls[0][1].endswith("original/T1w.nii.gz")
    assert calls[1][1].endswith("original/T2w.nii.gz")
    assert Path(manifest["defaced"]["t1"]["path"]).exists()
    assert Path(manifest["defaced"]["t2"]["path"]).exists()
    assert read_json(experiment_root / "_pipeline" / "defacing_manifest.json") == manifest


def test_plan_simulation_tasks_resolves_all_repeats_with_condition_inputs(tmp_path: Path):
    experiment_root = init_experiment(tmp_path, repeat_full=2, repeat_defaced=3)
    pipeline.prepare_inputs(experiment_root)
    pipeline.deface_inputs(
        experiment_root,
        command_runner=lambda cmd, **kwargs: (
            Path(cmd[2]).write_text("defaced", encoding="utf-8"),
            pipeline.CommandResult(args=cmd, returncode=0, stdout="", stderr=""),
        )[1],
    )

    tasks = pipeline.plan_simulation_tasks(experiment_root)

    assert len(tasks) == 5
    assert [task["task_index"] for task in tasks] == [0, 1, 2, 3, 4]
    assert {task["condition"] for task in tasks} == {"full_face", "defaced"}
    assert tasks[0]["repeat"] == 1
    assert tasks[-1]["repeat"] == 3
    assert tasks[-1]["t1_path"].endswith("inputs/defaced/T1w.nii.gz")
    assert tasks[-1]["ti_mesh_path"].endswith("runs/defaced/repeat-003/sub-001/anat/SimNIBS/Output/sub-001/TI.msh")


def test_submit_simulations_parses_sbatch_output_and_uses_repo_root_safe_exports(tmp_path: Path, monkeypatch):
    experiment_root = init_experiment(tmp_path, repeat_full=2, repeat_defaced=2)
    pipeline.prepare_inputs(experiment_root)
    pipeline.deface_inputs(
        experiment_root,
        command_runner=lambda cmd, **kwargs: (
            Path(cmd[2]).write_text("defaced", encoding="utf-8"),
            pipeline.CommandResult(args=cmd, returncode=0, stdout="", stderr=""),
        )[1],
    )
    fake_sbatch = tmp_path / "fake_sbatch"
    fake_sbatch.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$FAKE_SBATCH_ARGS\"\n"
        "echo 'Submitted batch job 98765'\n",
        encoding="utf-8",
    )
    fake_sbatch.chmod(fake_sbatch.stat().st_mode | stat.S_IXUSR)
    args_file = tmp_path / "sbatch_args.txt"
    monkeypatch.setenv("SBATCH_BIN", str(fake_sbatch))
    monkeypatch.setenv("FAKE_SBATCH_ARGS", str(args_file))

    submission = pipeline.submit_simulations(experiment_root, max_concurrent=50)

    assert submission["job_id"] == "98765"
    args_text = args_file.read_text(encoding="utf-8")
    assert "--array=0-3%50" in args_text
    assert "EXPERIMENT_ROOT=" in args_text
    assert "TASK_MANIFEST=" in args_text
    assert "CONFIG=" in args_text
    assert str(experiment_root / "_pipeline" / "simulation_tasks.json") in args_text


def test_status_detects_missing_defaced_inputs_ti_meshes_summaries_and_figures(tmp_path: Path):
    experiment_root = init_experiment(tmp_path, repeat_full=1, repeat_defaced=1)
    pipeline.prepare_inputs(experiment_root)
    status = pipeline.status_experiment(experiment_root)

    missing = "\n".join(status["missing"])
    assert "inputs/defaced/T1w.nii.gz" in missing
    assert "inputs/defaced/T2w.nii.gz" in missing
    assert "TI.msh" in missing
    assert "_analysis/per_repeat_metrics.csv" in missing
    assert "figures/figure_manifest.json" in missing


def test_aggregate_analysis_writes_condition_comparison_roi_whole_head_and_mesh_metrics(tmp_path: Path):
    experiment_root = init_experiment(tmp_path, repeat_full=2, repeat_defaced=2)
    per_repeat = experiment_root / "_analysis" / "per_repeat_metrics.csv"
    per_repeat.parent.mkdir(parents=True, exist_ok=True)
    with per_repeat.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "condition",
                "repeat",
                "median_roi_ti",
                "mean_roi_ti",
                "peak_roi_ti",
                "median_head_ti",
                "mean_head_ti",
                "peak_head_ti",
                "hotspot_x",
                "hotspot_y",
                "hotspot_z",
                "high_field_voxels",
                "mesh_nodes",
                "mesh_cells",
                "tissue_1_count",
                "tissue_2_count",
                "mesh_checksum",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "condition": "full_face",
                    "repeat": 1,
                    "median_roi_ti": 1.0,
                    "mean_roi_ti": 2.0,
                    "peak_roi_ti": 3.0,
                    "median_head_ti": 0.5,
                    "mean_head_ti": 1.5,
                    "peak_head_ti": 2.5,
                    "hotspot_x": 0,
                    "hotspot_y": 0,
                    "hotspot_z": 0,
                    "high_field_voxels": 10,
                    "mesh_nodes": 100,
                    "mesh_cells": 200,
                    "tissue_1_count": 20,
                    "tissue_2_count": 30,
                    "mesh_checksum": "a",
                },
                {
                    "condition": "full_face",
                    "repeat": 2,
                    "median_roi_ti": 3.0,
                    "mean_roi_ti": 4.0,
                    "peak_roi_ti": 5.0,
                    "median_head_ti": 1.5,
                    "mean_head_ti": 2.5,
                    "peak_head_ti": 3.5,
                    "hotspot_x": 3,
                    "hotspot_y": 4,
                    "hotspot_z": 0,
                    "high_field_voxels": 10,
                    "mesh_nodes": 110,
                    "mesh_cells": 210,
                    "tissue_1_count": 22,
                    "tissue_2_count": 31,
                    "mesh_checksum": "b",
                },
                {
                    "condition": "defaced",
                    "repeat": 1,
                    "median_roi_ti": 2.0,
                    "mean_roi_ti": 3.0,
                    "peak_roi_ti": 4.0,
                    "median_head_ti": 1.0,
                    "mean_head_ti": 2.0,
                    "peak_head_ti": 3.0,
                    "hotspot_x": 0,
                    "hotspot_y": 0,
                    "hotspot_z": 0,
                    "high_field_voxels": 5,
                    "mesh_nodes": 90,
                    "mesh_cells": 190,
                    "tissue_1_count": 18,
                    "tissue_2_count": 26,
                    "mesh_checksum": "c",
                },
                {
                    "condition": "defaced",
                    "repeat": 2,
                    "median_roi_ti": 4.0,
                    "mean_roi_ti": 5.0,
                    "peak_roi_ti": 6.0,
                    "median_head_ti": 2.0,
                    "mean_head_ti": 3.0,
                    "peak_head_ti": 4.0,
                    "hotspot_x": 0,
                    "hotspot_y": 6,
                    "hotspot_z": 8,
                    "high_field_voxels": 5,
                    "mesh_nodes": 95,
                    "mesh_cells": 195,
                    "tissue_1_count": 19,
                    "tissue_2_count": 27,
                    "mesh_checksum": "d",
                },
            ]
        )

    outputs = pipeline.aggregate_analysis(experiment_root)

    condition_summary = read_json(outputs["condition_summary_json"])
    comparison = read_json(outputs["comparison_summary_json"])
    mesh = read_json(outputs["mesh_metrics_json"])

    assert condition_summary["full_face"]["median_roi_ti"]["mean"] == 2.0
    assert condition_summary["defaced"]["median_head_ti"]["mean"] == 1.5
    assert comparison["defaced_minus_full_face"]["median_roi_ti"]["mean_delta"] == 1.0
    assert comparison["defaced_vs_full_face"]["high_field_dice_mean"] == pytest.approx(2 / 3)
    assert comparison["defaced_vs_full_face"]["hotspot_distance_mean_mm"] == pytest.approx(math.sqrt(77) / 2)
    assert mesh["full_face"]["mesh_nodes"]["mean"] == 105.0
    assert mesh["defaced"]["tissue_label_counts"]["tissue_1_count"]["mean"] == 18.5


def test_make_figures_writes_manifest_after_analysis(tmp_path: Path):
    experiment_root = init_experiment(tmp_path, repeat_full=1, repeat_defaced=1)
    analysis_dir = experiment_root / "_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "condition_summary.csv").write_text(
        "condition,metric,count,mean,median,min,max,std\n"
        "full_face,median_roi_ti,1,1,1,1,1,0\n"
        "defaced,median_roi_ti,1,2,2,2,2,0\n",
        encoding="utf-8",
    )
    (analysis_dir / "comparison_summary.csv").write_text(
        "comparison,metric,mean_delta\n"
        "defaced_minus_full_face,median_roi_ti,1\n",
        encoding="utf-8",
    )

    manifest = pipeline.make_figures(experiment_root)

    assert Path(manifest["figures"]["condition_metric_means"]).exists()
    assert Path(manifest["figures"]["comparison_metric_deltas"]).exists()
    assert read_json(experiment_root / "figures" / "figure_manifest.json") == manifest


def test_run_report_task_flattens_repeat_summary_json_to_condition_csv(tmp_path: Path):
    experiment_root = init_experiment(tmp_path, repeat_full=1, repeat_defaced=1)
    pipeline.prepare_inputs(experiment_root)
    tasks = pipeline.plan_simulation_tasks(experiment_root)
    for task in tasks:
        summary_path = Path(task["summary_path"])
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "condition": task["condition"],
                    "repeat": task["repeat"],
                    "roi": {"median_ti": 1.0, "mean_ti": 2.0, "peak_ti": 3.0},
                    "whole_head": {
                        "median_ti": 0.5,
                        "mean_ti": 1.5,
                        "peak_ti": 2.5,
                        "hotspot": [1, 2, 3],
                        "high_field_voxels": 10,
                    },
                    "mesh": {
                        "nodes": 100,
                        "cells": 200,
                        "tissue_label_counts": {"1": 20},
                        "checksum": "abc",
                    },
                }
            ),
            encoding="utf-8",
        )
    report_tasks = pipeline.plan_report_tasks(experiment_root)

    output = pipeline.run_report_task(experiment_root / "_pipeline" / "report_tasks.json", report_tasks[0]["task_index"])

    rows = list(csv.DictReader(Path(output["condition_metrics_csv"]).open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["condition"] == "full_face"
    assert rows[0]["median_roi_ti"] == "1.0"
    assert rows[0]["median_head_ti"] == "0.5"
    assert rows[0]["hotspot_z"] == "3"
    assert rows[0]["mesh_nodes"] == "100"
    assert rows[0]["tissue_1_count"] == "20"


def test_analyze_local_only_merges_report_task_outputs_before_aggregation(tmp_path: Path):
    experiment_root = init_experiment(tmp_path, repeat_full=1, repeat_defaced=1)
    pipeline.prepare_inputs(experiment_root)
    pipeline.plan_simulation_tasks(experiment_root)
    report_dir = experiment_root / "_analysis" / "report_tasks"
    report_dir.mkdir(parents=True, exist_ok=True)
    for condition, value in [("full_face", 1.0), ("defaced", 2.0)]:
        (report_dir / f"{condition}_per_repeat_metrics.csv").write_text(
            "condition,repeat,median_roi_ti,mean_roi_ti,peak_roi_ti,median_head_ti,mean_head_ti,peak_head_ti,"
            "hotspot_x,hotspot_y,hotspot_z,high_field_voxels,mesh_nodes,mesh_cells,tissue_1_count,mesh_checksum\n"
            f"{condition},1,{value},{value},{value},{value},{value},{value},0,0,0,10,100,200,20,{condition}\n",
            encoding="utf-8",
        )

    outputs = pipeline.aggregate_report_outputs(experiment_root)

    merged_rows = list(csv.DictReader(Path(outputs["per_repeat_metrics_csv"]).open(encoding="utf-8")))
    comparison = read_json(outputs["comparison_summary_json"])
    assert [row["condition"] for row in merged_rows] == ["defaced", "full_face"]
    assert comparison["defaced_minus_full_face"]["median_roi_ti"]["mean_delta"] == 1.0
