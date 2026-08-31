from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "post"
    / "analyze_nested_repeatability.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_nested_repeatability", SCRIPT)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_balanced_nested_variance_decomposition(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "selected_subject": "sub-01",
                "selected_target": "left-hippocampus",
            }
        ),
        encoding="utf-8",
    )
    metrics = tmp_path / "metrics.csv"
    rows = []
    mesh_offsets = [-1.0, 0.0, 1.0, 2.0]
    repeat_offsets = [-0.3, -0.1, 0.1, 0.3]
    for mesh_index, mesh_offset in enumerate(mesh_offsets, start=1):
        for repeat_index, repeat_offset in enumerate(repeat_offsets, start=1):
            rows.append(
                {
                    "subject": "sub-01",
                    "condition": f"mesh_{mesh_index:03d}",
                    "repeat_tag": f"repeat_{repeat_index:03d}",
                    "roi": "Left_Hippocampus",
                    "roi_median_v_per_m": 10.0 + mesh_offset + repeat_offset,
                }
            )
    with metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = analysis.analyze_nested(
        metrics_csv=metrics,
        selection_manifest=selection,
        output_dir=tmp_path / "out",
        expected_meshes=4,
        expected_repeats_per_mesh=4,
    )

    assert result["status"] == "complete"
    assert result["observations"] == 16
    assert result["grand_mean"] == pytest.approx(10.5)
    components = result["variance_components"]
    assert components["between_mesh"]["variance"] > 0
    assert components["within_mesh_solver_pipeline"]["variance"] > 0
    assert components["between_mesh"]["fraction_of_total_variance"] > 0.9
    assert (tmp_path / "out/nested_mesh_summary.csv").is_file()
    assert (tmp_path / "out/nested_variance_components.json").is_file()
