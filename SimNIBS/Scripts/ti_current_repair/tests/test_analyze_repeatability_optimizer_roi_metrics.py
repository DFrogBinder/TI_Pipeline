import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


CURRENT_REPAIR_ROOT = Path(__file__).resolve().parents[1]
if str(CURRENT_REPAIR_ROOT) not in sys.path:
    sys.path.insert(0, str(CURRENT_REPAIR_ROOT))

from post import analyze_repeatability_optimizer_roi_metrics as analysis  # noqa: E402


def _target_frame(target: str) -> pd.DataFrame:
    meta = analysis.TARGETS[target]
    roi_voxels = int(meta["requested_volume_mm3"])
    rows = []
    for subject_index in range(10):
        subject = f"sub-CC{subject_index:06d}"
        radius = 3.5 + 0.05 * subject_index
        for condition in ("remesh", "fixed_mesh"):
            for run_index in range(1, 41):
                base = 0.15 + 0.01 * subject_index
                offset = (
                    0.0002 * (run_index - 20.5)
                    if condition == "remesh"
                    else 0.000001 * (run_index - 20.5)
                )
                nonfinite = 1 if target == "left_hippocampus" and run_index % 9 == 0 else 0
                finite = roi_voxels - nonfinite
                rows.append(
                    {
                        "schema_version": 2,
                        "subject": subject,
                        "condition": condition,
                        "repeat_tag": f"repeat_{run_index:03d}",
                        "roi": meta["roi"],
                        "roi_voxels": roi_voxels,
                        "requested_roi_volume_mm3": meta["requested_volume_mm3"],
                        "achieved_roi_volume_mm3": float(roi_voxels),
                        "roi_radius_mm": radius,
                        "roi_median_v_per_m": base + offset,
                        "finite_roi_voxels": finite,
                        "nonfinite_roi_voxels": nonfinite,
                        "finite_roi_fraction": finite / roi_voxels,
                    }
                )
    return pd.DataFrame(rows)


def _write_inputs(root: Path) -> tuple[Path, Path]:
    left = root / "left.csv"
    right = root / "right.csv"
    _target_frame("left_hippocampus").to_csv(left, index=False)
    _target_frame("right_m1").to_csv(right, index=False)
    return left, right


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_validates_and_combines_complete_inputs(tmp_path: Path) -> None:
    left, right = _write_inputs(tmp_path)
    frame = analysis._load(left, right)

    assert len(frame) == 1600
    assert set(frame["target"]) == {"left_hippocampus", "right_m1"}
    assert set(frame.groupby(["target", "subject", "condition"]).size()) == {40}

    geometry = analysis._geometry_summary(frame)
    assert geometry["left_hippocampus"]["requested_volume_mm3"] == 200.0
    assert geometry["right_m1"]["requested_volume_mm3"] == 100.0
    assert geometry["left_hippocampus"]["minimum_radius_mm"] == 3.5
    assert geometry["right_m1"]["maximum_radius_mm"] == pytest.approx(3.95)


def test_validation_rejects_incomplete_repeat_tags() -> None:
    frame = _target_frame("left_hippocampus")
    selected = (
        (frame["subject"] == "sub-CC000000")
        & (frame["condition"] == "remesh")
        & (frame["repeat_tag"] == "repeat_040")
    )
    frame.loc[selected, "repeat_tag"] = "repeat_039"

    with pytest.raises(RuntimeError, match="repeat tags"):
        analysis._validate(frame, "left_hippocampus")


def test_rank_analysis_is_reproducible(tmp_path: Path) -> None:
    left, right = _write_inputs(tmp_path)
    frame = analysis._load(left, right)

    pairs_a, draws_a, rank_a = analysis._rank_analysis(frame, seed=17, draws=200)
    pairs_b, draws_b, rank_b = analysis._rank_analysis(frame, seed=17, draws=200)
    pd.testing.assert_frame_equal(pairs_a, pairs_b)
    pd.testing.assert_frame_equal(draws_a, draws_b)
    assert rank_a == rank_b



def test_run_writes_portable_hashed_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left, right = _write_inputs(tmp_path)
    out_dir = tmp_path / "analysis"

    def write_placeholder(*args, **kwargs) -> None:
        path = args[-1]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png-placeholder")

    monkeypatch.setattr(analysis, "_plot_primary", write_placeholder)
    monkeypatch.setattr(analysis, "_plot_rank", write_placeholder)

    result = analysis.run(
        argparse.Namespace(
            left_csv=left,
            right_csv=right,
            out_dir=out_dir,
            seed=101,
            rank_draws=100,
        )
    )

    assert result["status"] == "complete"
    assert result["rows"] == 1600
    assert result["inputs"]["left_csv"]["sha256"] == _sha256(left)
    assert result["inputs"]["right_csv"]["sha256"] == _sha256(right)
    assert result["outputs"]["subject_summary"] == "subject_condition_summary.csv"

    manifest = json.loads((out_dir / "analysis_manifest.json").read_text())
    artifact_paths = {entry["path"] for entry in manifest["artifacts"]}
    assert "analysis_manifest.json" not in artifact_paths
    assert "roi_geometry_summary.json" in artifact_paths
    for entry in manifest["artifacts"]:
        path = out_dir / entry["path"]
        assert entry["size_bytes"] == path.stat().st_size
        assert entry["sha256"] == _sha256(path)
