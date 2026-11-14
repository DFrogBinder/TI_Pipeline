from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

import simnibs_segmentation_determinism.journal_spatial_figures as journal_figures
from simnibs_segmentation_determinism.journal_spatial_data import (
    prepare_journal_spatial_data,
)
from simnibs_segmentation_determinism.journal_spatial_figures import (
    JournalSpatialFigureError,
    _plot_slice_atlas,
    _plot_spatial_overview,
    build_journal_spatial_figures,
)
from simnibs_segmentation_determinism.provenance import sha256_file
from simnibs_segmentation_determinism.tests.test_journal_spatial_data import (
    _non_ras_spatial_fixture,
)


FIGURE_STEMS = (
    "01_charm_spatial_disagreement_overview",
    "02_charm_spatial_disagreement_atlas",
)


def _digest_sources(root: Path, excluded: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and excluded not in path.parents
    }


def test_journal_plots_use_plain_disagreement_labels_and_physical_coordinates(
    configured_experiment,
):
    _non_ras_spatial_fixture(configured_experiment)
    data = prepare_journal_spatial_data(configured_experiment)

    overview = _plot_spatial_overview(data)
    atlas = _plot_slice_atlas(data)

    visible_text = []
    for figure in (overview, atlas):
        visible_text.extend(text.get_text() for text in figure.texts)
        for axis in figure.axes:
            visible_text.extend(
                text.get_text()
                for text in [
                    *axis.texts,
                    axis.title,
                    axis.xaxis.label,
                    axis.yaxis.label,
                ]
            )
    all_text = "\n".join(visible_text)
    assert "Runs not matching the modal tissue label (out of 10)" in all_text
    assert "mm" in all_text
    assert "Repeat-to-repeat CHARM tissue-label disagreement" in all_text
    assert "voxel index" not in all_text.lower()


def test_slice_atlas_uses_compact_panel_titles_without_overlapping_footer(
    configured_experiment,
):
    _non_ras_spatial_fixture(configured_experiment)
    data = prepare_journal_spatial_data(configured_experiment)

    atlas = _plot_slice_atlas(data)

    panel_titles = [axis.get_title() for axis in atlas.axes[:12]]
    assert all(title.endswith("mm") for title in panel_titles)
    assert all(len(title) <= 16 for title in panel_titles)
    assert not any("Slices are selected" in text.get_text() for text in atlas.texts)


def test_build_journal_spatial_figures_creates_bound_three_format_package(
    configured_experiment,
):
    root = _non_ras_spatial_fixture(configured_experiment)
    output = root / "journal_spatial_figures_v1"
    before = _digest_sources(root, output)

    result = build_journal_spatial_figures(configured_experiment)

    assert result == {
        "status": "complete",
        "action": "created",
        "output_root": str(output.resolve()),
        "figure_count": 2,
        "source_outputs_modified": False,
    }
    assert _digest_sources(root, output) == before
    for stem in FIGURE_STEMS:
        for suffix in ("png", "pdf", "svg"):
            assert (output / f"{stem}.{suffix}").is_file()
        with Image.open(output / f"{stem}.png") as image:
            assert image.width >= 4500
            assert image.height >= 3000
            assert image.info["dpi"][0] == pytest.approx(600, abs=1)

    captions = (output / "figure_captions.md").read_text(encoding="utf-8")
    assert "not segmentation error probability" in captions
    assert "one participant" in captions
    assert "SimNIBS 4.0.1" in captions

    summary = json.loads((output / "spatial_summary.json").read_text())
    assert summary["maximum_disagreement_count"] == 4
    assert summary["repeat_count"] == 10

    manifest = json.loads((output / "figure_manifest.json").read_text())
    assert manifest["workflow"] == (
        "simnibs_segmentation_determinism_journal_spatial_figures"
    )
    assert manifest["figure_count"] == 2
    assert manifest["source_outputs_modified"] is False
    assert manifest["source"]["staged_t1"]["sha256"] == sha256_file(
        root / f"inputs/{configured_experiment.subject}/anat/"
        f"{configured_experiment.subject}_T1w.nii.gz"
    )
    for line in (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        assert sha256_file(output / relative) == expected


def test_build_journal_spatial_figures_reuses_only_verified_package(
    configured_experiment,
):
    root = _non_ras_spatial_fixture(configured_experiment)
    assert build_journal_spatial_figures(configured_experiment)["action"] == "created"

    assert build_journal_spatial_figures(configured_experiment)["action"] == "reused"

    (root / "journal_spatial_figures_v1/01_charm_spatial_disagreement_overview.svg").write_text(
        "changed", encoding="utf-8"
    )
    with pytest.raises(JournalSpatialFigureError, match="checksum mismatch"):
        build_journal_spatial_figures(configured_experiment)


def test_build_journal_spatial_figures_refuses_foreign_destination(
    configured_experiment,
):
    root = _non_ras_spatial_fixture(configured_experiment)
    destination = root / "journal_spatial_figures_v1"
    destination.mkdir()
    (destination / "foreign.txt").write_text("foreign\n", encoding="utf-8")

    with pytest.raises(JournalSpatialFigureError, match="foreign or incomplete"):
        build_journal_spatial_figures(configured_experiment)


def test_render_failure_removes_temporary_journal_package(
    configured_experiment, monkeypatch
):
    root = _non_ras_spatial_fixture(configured_experiment)

    def fail_render(data):
        del data
        raise RuntimeError("synthetic journal rendering failure")

    monkeypatch.setitem(
        journal_figures.FIGURE_BUILDERS,
        FIGURE_STEMS[0],
        fail_render,
    )

    with pytest.raises(RuntimeError, match="synthetic journal rendering failure"):
        build_journal_spatial_figures(configured_experiment)

    assert not (root / "journal_spatial_figures_v1").exists()
    assert not tuple(root.glob(".journal_spatial_figures_v1.tmp-*"))
