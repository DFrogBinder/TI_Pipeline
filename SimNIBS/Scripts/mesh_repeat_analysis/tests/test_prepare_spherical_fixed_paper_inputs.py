from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "post"
    / "prepare_spherical_fixed_paper_inputs.py"
)
SPEC = importlib.util.spec_from_file_location(
    "prepare_spherical_fixed_paper_inputs", SCRIPT
)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def _rows(roi: str, conditions: tuple[str, ...]) -> list[dict[str, object]]:
    rows = []
    for subject_index in range(10):
        subject = f"sub-CC{subject_index:06d}"
        for condition in conditions:
            for repeat_index in range(1, 41):
                rows.append(
                    {
                        "schema_version": 2,
                        "subject": subject,
                        "condition": condition,
                        "repeat_tag": f"repeat_{repeat_index:03d}",
                        "roi": roi,
                        "roi_median_v_per_m": (
                            0.2
                            + subject_index * 0.001
                            + repeat_index * 1e-6
                            + (0.01 if condition == "fixed_mesh" else 0.0)
                        ),
                        "roi_voxels": 100,
                        "finite_roi_voxels": 100,
                        "nonfinite_roi_voxels": 0,
                        "finite_roi_fraction": 1.0,
                        "ti_path": f"/{subject}/{condition}/{repeat_index}",
                    }
                )
    return rows


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _target_inputs(root: Path, stem: str, roi: str) -> tuple[Path, Path]:
    historical = root / f"{stem}_historical.csv"
    corrected = root / f"{stem}_corrected.csv"
    _write(historical, _rows(roi, ("remesh", "fixed_mesh")))
    _write(corrected, _rows(roi, ("fixed_mesh",)))
    return historical, corrected


def test_run_replaces_historical_fixed_rows_and_writes_manifest(tmp_path: Path) -> None:
    left_historical, left_corrected = _target_inputs(
        tmp_path, "left", "Left_Hippocampus"
    )
    right_historical, right_corrected = _target_inputs(
        tmp_path, "right", "Right_M1"
    )
    output_dir = tmp_path / "out"

    result = analysis.run(
        argparse.Namespace(
            left_historical_csv=left_historical,
            left_corrected_fixed_csv=left_corrected,
            right_historical_csv=right_historical,
            right_corrected_fixed_csv=right_corrected,
            output_dir=output_dir,
        )
    )

    assert result["status"] == "complete"
    assert result["observed_rows"] == 1600
    assert result["source_outputs_modified"] is False
    left_output = output_dir / "left_hippocampus_optimizer_roi_metrics.csv"
    with left_output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 800
    assert {row["condition"] for row in rows} == {"remesh", "fixed_mesh"}
    fixed = [row for row in rows if row["condition"] == "fixed_mesh"]
    assert all(float(row["roi_median_v_per_m"]) >= 0.21 for row in fixed)
    manifest = json.loads((output_dir / "assembly_manifest.json").read_text())
    assert manifest["targets"]["left_hippocampus"]["rows"] == 800


def test_prepare_target_rejects_incomplete_corrected_repeats(tmp_path: Path) -> None:
    historical, corrected = _target_inputs(
        tmp_path, "left", "Left_Hippocampus"
    )
    fields, rows = analysis._read_csv(corrected)
    analysis._write_csv_atomic(corrected, fieldnames=fields, rows=rows[:-1])

    with pytest.raises(ValueError, match="400 corrected fixed-mesh rows"):
        analysis.prepare_target(
            target="left_hippocampus",
            historical_csv=historical,
            corrected_fixed_csv=corrected,
            output_csv=tmp_path / "combined.csv",
        )
