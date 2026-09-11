from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "post"
    / "plot_nested_repeatability.py"
)
SPEC = importlib.util.spec_from_file_location("plot_nested_repeatability", SCRIPT)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def _write_nested_inputs(
    root: Path,
    *,
    outer: int = 4,
    inner: int = 5,
) -> tuple[Path, Path]:
    metrics = root / "metrics.csv"
    rows = []
    values: list[list[float]] = []
    for mesh_index in range(1, outer + 1):
        group = []
        for repeat_index in range(1, inner + 1):
            value = 0.2 + mesh_index * 0.002 + (repeat_index - 3) * 1e-5
            group.append(value)
            rows.append(
                {
                    "subject": "sub-CC000001",
                    "condition": f"mesh_{mesh_index:03d}",
                    "repeat_tag": f"repeat_{repeat_index:03d}",
                    "roi": "Left_Hippocampus",
                    "roi_median_v_per_m": value,
                }
            )
        values.append(group)
    with metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    components = analysis._variance_components(values)
    variance = root / "variance.json"
    variance.write_text(
        json.dumps(
            {
                "status": "complete",
                "selected_subject": "sub-CC000001",
                "roi": "Left_Hippocampus",
                "outer_meshes": outer,
                "inner_repeats_per_mesh": inner,
                "metric": "roi_median_v_per_m",
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


def test_run_writes_hierarchical_figure_and_values(tmp_path: Path) -> None:
    metrics, variance = _write_nested_inputs(tmp_path)
    output_dir = tmp_path / "out"

    result = analysis.run(
        metrics_csv=metrics,
        variance_json=variance,
        output_dir=output_dir,
        expected_meshes=4,
        expected_repeats=5,
    )

    assert result["status"] == "complete"
    assert result["observations"] == 20
    assert result["outer_meshes"] == 4
    assert result["inner_repeats_per_mesh"] == 5
    assert result["sd_ratio_between_over_within"] > 10
    for name in (
        "nested_mesh_by_solver_repeatability.png",
        "nested_mesh_by_solver_repeatability.svg",
        "nested_within_mesh_residual_matrix_supplement.png",
        "nested_within_mesh_residual_matrix_supplement.svg",
        "nested_figure_values.json",
        "nested_figure_caption.md",
        "nested_figure_manifest.json",
    ):
        assert (output_dir / name).is_file()
    caption = (output_dir / "nested_figure_caption.md").read_text()
    assert "one mean for each" in caption
    assert "Supplementary figure caption" in caption


def test_run_rejects_variance_result_that_does_not_match_metrics(
    tmp_path: Path,
) -> None:
    metrics, variance = _write_nested_inputs(tmp_path)
    payload = json.loads(variance.read_text())
    payload["grand_mean"] += 0.1
    variance.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="differs from analysis value"):
        analysis.run(
            metrics_csv=metrics,
            variance_json=variance,
            output_dir=tmp_path / "out",
            expected_meshes=4,
            expected_repeats=5,
        )
