from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "pipeline"
    / "spherical_fixed_nested_experiment.py"
)
SPEC = importlib.util.spec_from_file_location("spherical_fixed_nested_experiment", SCRIPT)
assert SPEC and SPEC.loader
workflow = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = workflow
SPEC.loader.exec_module(workflow)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _source_config(
    *,
    source_dataset: Path,
    experiment_root: Path,
    target: str,
    subjects: list[str],
    repeats: int,
) -> dict[str, object]:
    from stimulation_config import resolve_confirmed_stimulation

    roi = "Left_Hippocampus" if target == "left-hippocampus" else "Right_M1"
    label = 17 if target == "left-hippocampus" else 12129
    return {
        "source_root": str(source_dataset),
        "experiment_root": str(experiment_root),
        "subjects": subjects,
        "conditions": [
            {"name": "remesh", "mesh_mode": "remesh", "repeat_count": repeats},
            {"name": "fixed_mesh", "mesh_mode": "fixed_mesh", "repeat_count": repeats},
        ],
        "stimulation": resolve_confirmed_stimulation(target).to_dict(),
        "analysis": {
            "roi_preset": target,
            "roi_name": roi,
            "roi_labels": [label],
            "atlas_dir": str(experiment_root / "atlases"),
            "compare_metric": "median_roi",
        },
    }


def _populate_source(
    *,
    root: Path,
    target: str,
    subjects: list[str],
    repeats: int,
) -> tuple[Path, Path]:
    source_dataset = root / "source_dataset"
    config = _source_config(
        source_dataset=source_dataset,
        experiment_root=root,
        target=target,
        subjects=subjects,
        repeats=repeats,
    )
    _write_json(root / "_pipeline/configs/paired_analysis.json", config)
    metric_rows: list[dict[str, object]] = []
    roi = workflow.TARGET_SPECS[target]["roi"]
    volume = workflow.TARGET_SPECS[target]["requested_volume_mm3"]
    for subject_index, subject in enumerate(subjects):
        input_anat = source_dataset / subject / "anat"
        input_anat.mkdir(parents=True, exist_ok=True)
        for suffix in ("T1w.nii", "T2w.nii", "T1w_ras_1mm_T1andT2_masks.nii"):
            (input_anat / f"{subject}_{suffix}").write_bytes(b"input")
        for repeat_index in range(1, repeats + 1):
            tag = f"repeat_{repeat_index:03d}"
            anat = (
                root
                / f"{subject}_repeatability/remesh/repeats"
                / tag
                / subject
                / "anat"
            )
            m2m = anat / f"m2m_{subject}"
            m2m.mkdir(parents=True, exist_ok=True)
            for suffix in ("T1w.nii", "T2w.nii", "T1w_ras_1mm_T1andT2_masks.nii"):
                (anat / f"{subject}_{suffix}").write_bytes(b"input")
            (m2m / f"{subject}.msh").write_bytes(
                f"{target}-{subject}-{tag}".encode()
            )
            values = [1.0, 2.0, 3.0, 100.0]
            metric_rows.append(
                {
                    "schema_version": 2,
                    "subject": subject,
                    "condition": "remesh",
                    "repeat_tag": tag,
                    "roi": roi,
                    "requested_roi_volume_mm3": volume,
                    "roi_median_v_per_m": values[repeat_index - 1] + subject_index,
                }
            )
            metric_rows.append(
                {
                    "schema_version": 2,
                    "subject": subject,
                    "condition": "fixed_mesh",
                    "repeat_tag": tag,
                    "roi": roi,
                    "requested_roi_volume_mm3": volume,
                    "roi_median_v_per_m": 50.0,
                }
            )
    metrics = root / "optimizer_roi_metrics.csv"
    _write_csv(metrics, metric_rows)
    return root, metrics


def test_spherical_selector_replaces_anatomical_selection(tmp_path: Path) -> None:
    source, metrics = _populate_source(
        root=tmp_path / "left",
        target="left-hippocampus",
        subjects=["sub-01"],
        repeats=4,
    )
    _write_csv(
        source / "_pipeline/fixed_seed_manifest.csv",
        [{"subject": "sub-01", "selected_repeat_tag": "repeat_001"}],
    )

    _, selections = workflow.select_spherical_medians(
        source_experiment_root=source,
        metrics_csv=metrics,
        repeat_count=4,
    )

    assert len(selections) == 1
    assert selections[0].metric == "roi_median_v_per_m"
    assert selections[0].median_target == 2.5
    assert selections[0].selected_repeat_tag == "repeat_002"
    assert selections[0].previous_selected_repeat_tag == "repeat_001"
    assert selections[0].selection_changed is True


def test_fixed_preparation_is_isolated_and_idempotent(tmp_path: Path) -> None:
    source, metrics = _populate_source(
        root=tmp_path / "left",
        target="left-hippocampus",
        subjects=["sub-01"],
        repeats=4,
    )
    output = tmp_path / "fixed_correction"

    first = workflow.prepare_fixed(
        source_experiment_root=source,
        metrics_csv=metrics,
        output_root=output,
        repeat_count=4,
    )
    second = workflow.prepare_fixed(
        source_experiment_root=source,
        metrics_csv=metrics,
        output_root=output,
        repeat_count=4,
    )

    assert first["simulation_tasks"] == 4
    assert second["simulation_tasks"] == 4
    config = json.loads(Path(first["config"]).read_text(encoding="utf-8"))
    assert config["experiment_root"] == str(output)
    assert [condition["name"] for condition in config["conditions"]] == [
        "fixed_mesh"
    ]
    cache_mesh = (
        output
        / "sub-01_repeatability/fixed_mesh/mesh_cache/sub-01/anat"
        / "m2m_sub-01/sub-01.msh"
    )
    assert cache_mesh.is_file()
    assert cache_mesh.resolve().name == "sub-01.msh"


def test_nested_random_case_is_persisted_across_new_seeds(tmp_path: Path) -> None:
    subjects = ["sub-01", "sub-02"]
    left_root, left_metrics = _populate_source(
        root=tmp_path / "left",
        target="left-hippocampus",
        subjects=subjects,
        repeats=4,
    )
    right_root, right_metrics = _populate_source(
        root=tmp_path / "right",
        target="right-m1",
        subjects=subjects,
        repeats=4,
    )
    output = tmp_path / "nested"
    sources = {
        "left-hippocampus": (left_root, left_metrics),
        "right-m1": (right_root, right_metrics),
    }

    first = workflow.prepare_nested(
        target_sources=sources,
        output_root=output,
        selection_seed=11,
        outer_repeat_count=4,
        inner_repeat_count=4,
    )
    persisted = json.loads(
        (output / "_pipeline/nested_case_selection.json").read_text(encoding="utf-8")
    )
    second = workflow.prepare_nested(
        target_sources=sources,
        output_root=output,
        selection_seed=999999,
        outer_repeat_count=4,
        inner_repeat_count=4,
    )

    assert first["simulation_tasks"] == 16
    assert second["selection_was_already_persisted"] is True
    assert second["selected_subject"] == persisted["selected_subject"]
    assert second["selected_target"] == persisted["selected_target"]
    config = json.loads(
        (output / "_pipeline/configs/nested_40x40.json").read_text(encoding="utf-8")
    )
    assert len(config["conditions"]) == 4
    assert all(condition["repeat_count"] == 4 for condition in config["conditions"])


def test_finalize_writes_optimizer_extractor_compatible_receipt(tmp_path: Path) -> None:
    source, metrics = _populate_source(
        root=tmp_path / "left",
        target="left-hippocampus",
        subjects=["sub-01"],
        repeats=4,
    )
    output = tmp_path / "fixed_correction"
    prepared = workflow.prepare_fixed(
        source_experiment_root=source,
        metrics_csv=metrics,
        output_root=output,
        repeat_count=4,
    )
    config_path = Path(prepared["config"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stimulation = config["stimulation"]
    cache_m2m = (
        output
        / "sub-01_repeatability/fixed_mesh/mesh_cache/sub-01/anat/m2m_sub-01"
    )
    for repeat_index in range(1, 5):
        tag = f"repeat_{repeat_index:03d}"
        repeat_subject = output / f"sub-01_repeatability/fixed_mesh/repeats/{tag}/sub-01"
        anat = repeat_subject / "anat"
        anat.mkdir(parents=True, exist_ok=True)
        (anat / "m2m_sub-01").symlink_to(cache_m2m, target_is_directory=True)
        simnibs = anat / "SimNIBS"
        output_dir = simnibs / "Output/sub-01"
        (output_dir / "Volume_Labels").mkdir(parents=True)
        (output_dir / "Volume_Base").mkdir(parents=True)
        (output_dir / "TI.msh").write_bytes(b"ti")
        (output_dir / "Volume_Labels/TI_Volumetric_Labels.nii.gz").write_bytes(b"labels")
        (output_dir / "Volume_Base/TI_Volumetric_Base.nii.gz").write_bytes(b"field")
        (simnibs / "ti_brain_only.nii.gz").write_bytes(b"field")
        _write_json(repeat_subject / "task_manifest.json", {"stimulation": stimulation})

    status = workflow.collect_simulation_status(config_path)
    receipt = workflow.finalize(config_path)

    assert status["status"] == "complete"
    assert status["observed_complete_tasks"] == 4
    assert receipt["scope"]["subject_count"] == 1
    assert receipt["scope"]["repeats_per_condition"] == 4
    assert receipt["scope"]["expected_ti_nifti"] == 4
    assert (output / "_pipeline/workflow/complete.json").is_file()
