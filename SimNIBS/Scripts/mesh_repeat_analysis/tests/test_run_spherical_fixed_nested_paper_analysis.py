from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import tarfile
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "post"
    / "run_spherical_fixed_nested_paper_analysis.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_spherical_fixed_nested_paper_analysis", SCRIPT
)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def _fixed_rows(roi: str, conditions: tuple[str, ...]) -> list[dict[str, object]]:
    rows = []
    requested_volume = 200.0 if roi == "Left_Hippocampus" else 100.0
    for subject_index in range(10):
        subject = f"sub-CC{subject_index:06d}"
        for condition in conditions:
            for repeat_index in range(1, 41):
                spread = 2e-4 if condition == "remesh" else 2e-6
                rows.append(
                    {
                        "schema_version": 2,
                        "subject": subject,
                        "condition": condition,
                        "repeat_tag": f"repeat_{repeat_index:03d}",
                        "roi": roi,
                        "roi_voxels": int(requested_volume),
                        "requested_roi_volume_mm3": requested_volume,
                        "achieved_roi_volume_mm3": requested_volume,
                        "roi_radius_mm": 3.5 + subject_index * 0.01,
                        "roi_median_v_per_m": (
                            0.18
                            + subject_index * 0.01
                            + (repeat_index - 20.5) * spread
                        ),
                        "finite_roi_voxels": int(requested_volume),
                        "nonfinite_roi_voxels": 0,
                        "finite_roi_fraction": 1.0,
                    }
                )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_nested(root: Path) -> tuple[Path, Path]:
    metrics = root / "nested.csv"
    rows = []
    values = []
    for mesh_index in range(1, 41):
        group = []
        for repeat_index in range(1, 41):
            value = (
                0.24
                + (mesh_index - 20.5) * 5e-4
                + (repeat_index - 20.5) * 1e-6
            )
            group.append(value)
            rows.append(
                {
                    "subject": "sub-CC320616",
                    "condition": f"mesh_{mesh_index:03d}",
                    "repeat_tag": f"repeat_{repeat_index:03d}",
                    "roi": "Left_Hippocampus",
                    "roi_median_v_per_m": value,
                }
            )
        values.append(group)
    _write_csv(metrics, rows)
    components = _load_nested_module()._variance_components(values)
    variance = root / "variance.json"
    variance.write_text(
        json.dumps(
            {
                "status": "complete",
                "selected_subject": "sub-CC320616",
                "roi": "Left_Hippocampus",
                "metric": "roi_median_v_per_m",
                "outer_meshes": 40,
                "inner_repeats_per_mesh": 40,
                "grand_mean": components["grand_mean"],
                "intraclass_correlation_mesh": components["mesh_icc"],
                "variance_components": {
                    "between_mesh": {
                        "variance": components["between_mesh_variance"]
                    },
                    "within_mesh_solver_pipeline": {
                        "variance": components["within_mesh_variance"]
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return metrics, variance


def _load_nested_module():
    path = SCRIPT.parent / "plot_nested_repeatability.py"
    specification = importlib.util.spec_from_file_location("nested_for_test", path)
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_full_analysis_writes_six_figures_and_download_bundle(tmp_path: Path) -> None:
    inputs = {}
    for stem, roi in (
        ("left", "Left_Hippocampus"),
        ("right", "Right_M1"),
    ):
        historical = tmp_path / f"{stem}_historical.csv"
        corrected = tmp_path / f"{stem}_corrected.csv"
        _write_csv(historical, _fixed_rows(roi, ("remesh", "fixed_mesh")))
        _write_csv(corrected, _fixed_rows(roi, ("fixed_mesh",)))
        inputs[f"{stem}_historical"] = historical
        inputs[f"{stem}_corrected"] = corrected
    nested_metrics, nested_variance = _write_nested(tmp_path)
    output = tmp_path / "output"

    result = analysis.run(
        argparse.Namespace(
            left_historical_csv=inputs["left_historical"],
            left_corrected_fixed_csv=inputs["left_corrected"],
            right_historical_csv=inputs["right_historical"],
            right_corrected_fixed_csv=inputs["right_corrected"],
            nested_metrics_csv=nested_metrics,
            nested_variance_json=nested_variance,
            output_root=output,
            rank_seed=17,
            rank_draws=100,
        )
    )

    assert result["status"] == "complete"
    assert result["source_outputs_modified"] is False
    assert result["scope"]["corrected_fixed_analysis_rows"] == 1600
    assert result["scope"]["nested_analysis_rows"] == 1600
    assert len(result["figures"]) == 6
    assert all(Path(path).is_file() for path in result["figures"])
    bundle = Path(result["download_bundle"]["path"])
    assert bundle.is_file()
    assert bundle.with_suffix(bundle.suffix + ".sha256").is_file()
    with tarfile.open(bundle, "r:gz") as archive:
        names = archive.getnames()
    assert any(name.endswith("nested_mesh_by_solver_repeatability.svg") for name in names)
    assert any(name.endswith("analysis_manifest.json") for name in names)
    assert "paper_analysis_summary.json" in names
    manifest = json.loads((output / "paper_analysis_manifest.json").read_text())
    assert manifest["download_bundle"]["sha256"]
