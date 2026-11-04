from __future__ import annotations

from pathlib import Path

from PIL import Image

from repeatability_figure_review.audit import sha256_file
from repeatability_figure_review.build import tree_manifest
from repeatability_figure_review.paper_figures import (
    PAPER_FIGURE_PATHS,
    build_clarity_directory,
)
from repeatability_figure_review.tests.test_figures import _make_complete_sources


def test_build_clarity_directory_contains_only_the_approved_images(
    tmp_path: Path,
) -> None:
    source, manuscript = _make_complete_sources(tmp_path)
    source_before = tree_manifest(source)
    manuscript_before = tree_manifest(manuscript)
    output = tmp_path / "paper_figures_v2"

    result = build_clarity_directory(
        source,
        manuscript,
        output,
        bootstrap_iterations=100,
        bootstrap_max_repeats=5,
    )

    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert actual == set(PAPER_FIGURE_PATHS)
    assert len(actual) == 12
    assert all(path.endswith(".png") for path in actual)
    assert not any("ranking" in path for path in actual if path.startswith("01_simnibs_v3/"))
    for relative in sorted(actual):
        with Image.open(output / relative) as image:
            image.verify()

    legacy_mapping = {
        "02_simnibs_v4/v4_legacy_mesh_element_count_variability.png":
            "figure_mesh_element_count_variability.png",
        "02_simnibs_v4/v4_legacy_tissue_volume_deviations.png":
            "figure_tissue_volume_deviations.png",
    }
    manuscript_figures = manuscript / "figures" / "combined"
    for relative, source_name in legacy_mapping.items():
        assert sha256_file(output / relative) == sha256_file(
            manuscript_figures / source_name
        )

    assert result == {
        "status": "complete",
        "output_root": str(output.resolve()),
        "figure_count": 12,
        "regenerated_figure_count": 10,
        "legacy_figure_count": 2,
        "source_outputs_modified": False,
    }
    assert tree_manifest(source) == source_before
    assert tree_manifest(manuscript) == manuscript_before
