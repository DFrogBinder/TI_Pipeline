from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import repeatability_figure_review.paper_figures as paper_figures
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


def test_build_clarity_directory_rejects_an_existing_output_root(
    tmp_path: Path,
) -> None:
    source, manuscript = _make_complete_sources(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Output root already exists"):
        build_clarity_directory(source, manuscript, output)

    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_failed_source_immutability_check_removes_only_the_new_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, manuscript = _make_complete_sources(tmp_path)
    output = tmp_path / "paper_figures_v2"
    real_tree_manifest = tree_manifest
    source_calls = 0

    def changed_source_manifest(root: Path):
        nonlocal source_calls
        manifest = real_tree_manifest(root)
        if Path(root).resolve() == source.resolve():
            source_calls += 1
            if source_calls == 2:
                manifest["concurrent-change.txt"] = {
                    "size_bytes": 1,
                    "sha256": "0" * 64,
                }
        return manifest

    monkeypatch.setattr(paper_figures, "tree_manifest", changed_source_manifest)

    with pytest.raises(RuntimeError, match="Downloaded analysis changed"):
        build_clarity_directory(
            source,
            manuscript,
            output,
            bootstrap_iterations=100,
            bootstrap_max_repeats=5,
        )

    assert not output.exists()
