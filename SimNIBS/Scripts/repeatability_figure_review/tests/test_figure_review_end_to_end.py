from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from repeatability_figure_review.audit import EXPECTED_ANALYSIS_FIGURES
from repeatability_figure_review.build import build_package, tree_manifest


PAPER_FIGURES = (
    "figure_target_field_repeatability.png",
    "figure_single_run_ranking_uncertainty.png",
    "figure_bootstrap_precision.png",
    "figure_mesh_element_count_variability.png",
    "figure_tissue_volume_deviations.png",
)


def _write_png(path: Path, size: tuple[int, int] = (240, 140)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)


def _table(version: str, target: str) -> pd.DataFrame:
    rows = []
    roi = "Left_Hippocampus" if target == "left_hippocampus" else "Right_M1"
    requested = 200.0 if target == "left_hippocampus" else 100.0
    for subject_index in range(10):
        subject = f"sub-CC{110000 + subject_index}"
        base = 0.16 + subject_index * 0.006 + (0.04 if version == "simnibs326" else 0)
        for condition in ("remesh", "fixed_mesh"):
            for repeat in range(1, 41):
                varying = version == "simnibs4" and condition == "remesh"
                value = base + (0.005 * np.sin(repeat + subject_index) if varying else 0.0)
                rows.append(
                    {
                        "schema_version": 2,
                        "subject": subject,
                        "condition": condition,
                        "repeat_tag": f"repeat_{repeat:03d}",
                        "roi": roi,
                        "requested_roi_volume_mm3": requested,
                        "roi_mean_v_per_m": value + 0.002,
                        "roi_median_v_per_m": value,
                        "roi_p95_v_per_m": value + 0.01,
                    }
                )
    return pd.DataFrame(rows)


def _sources(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "analysis"
    manuscript = tmp_path / "manuscript"
    for relative in EXPECTED_ANALYSIS_FIGURES:
        _write_png(source / relative)
    (source / "analysis_manifest.json").write_text("{}\n", encoding="utf-8")
    for version in ("simnibs326", "simnibs4"):
        for target in ("left_hippocampus", "right_m1"):
            path = source / "inputs" / f"{version}_{target}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            _table(version, target).to_csv(path, index=False)
    for name in PAPER_FIGURES:
        _write_png(manuscript / "figures" / "combined" / name)
    (manuscript / "main.tex").write_text("paper\n", encoding="utf-8")
    return source, manuscript


def _verify_sha256s(root: Path) -> None:
    lines = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert lines
    for line in lines:
        expected, relative = line.split("  ", 1)
        digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        assert digest == expected


def test_build_package_is_complete_and_does_not_modify_sources(tmp_path: Path) -> None:
    source, manuscript = _sources(tmp_path)
    source_before = tree_manifest(source)
    manuscript_before = tree_manifest(manuscript)
    output = tmp_path / "companion"

    result = build_package(
        source,
        manuscript,
        output,
        bootstrap_iterations=100,
        bootstrap_max_repeats=5,
    )

    assert result["status"] == "complete"
    assert result["original_figure_count"] == 20
    assert (output / "FIGURE_GUIDE.md").is_file()
    assert (output / "FIGURE_GUIDE.pdf").stat().st_size > 10_000
    assert (output / "figure_inventory.csv").is_file()
    assert (output / "paper_equivalence.csv").is_file()
    assert (output / "review_manifest.json").is_file()
    assert (output / "SHA256SUMS").is_file()
    assert len(list((output / "original_figures").rglob("*.png"))) == 20
    guide = (output / "FIGURE_GUIDE.md").read_text(encoding="utf-8")
    for relative in EXPECTED_ANALYSIS_FIGURES:
        assert relative in guide
    equivalence = pd.read_csv(output / "paper_equivalence.csv")
    assert len(equivalence) == 5
    assert equivalence.loc[
        equivalence.paper_figure == "figure_mesh_element_count_variability.png",
        "status",
    ].item() == "legacy_data_gap"
    _verify_sha256s(output)
    assert tree_manifest(source) == source_before
    assert tree_manifest(manuscript) == manuscript_before


