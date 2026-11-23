from __future__ import annotations

from pathlib import Path

from PIL import Image

from repeatability_combined_analysis_v2.statistics import summarize_participant_cells
from repeatability_combined_analysis_v2.tests.test_analysis import final_matrix
from repeatability_figure_review.v2_figures import (
    CONDITION_MARKERS,
    VERSION_COLORS,
    render_v2_figure_package,
)


def test_v2_figures_use_version_colors_condition_markers_and_separate_outputs(
    tmp_path: Path,
) -> None:
    cells = summarize_participant_cells(final_matrix())
    receipt = render_v2_figure_package(cells=cells, output_root=tmp_path / "figures")

    assert set(VERSION_COLORS) == {"simnibs326", "simnibs46"}
    assert len(set(VERSION_COLORS.values())) == 2
    assert CONDITION_MARKERS == {"remesh": "o", "fixed_mesh": "D"}
    assert receipt["status"] == "complete"
    assert receipt["point_unit"] == "participant_cell_summary"
    assert receipt["matched_target_figure"] != receipt["under_anode_figure"]
    assert receipt["participant_order"][6] == "sub-CC510415"
    assert "sub-CC520083" not in receipt["participant_order"]
    for key in (
        "matched_target_figure",
        "under_anode_figure",
        "pipeline_diagram",
        "matched_target_grayscale",
        "under_anode_grayscale",
    ):
        path = Path(receipt[key])
        assert path.is_file()
        with Image.open(path) as image:
            assert image.width >= 800
            assert image.height >= 400
    assert set(Path(receipt["matched_target_grayscale"]).open("rb").read(8))
    assert Path(receipt["manuscript_notes"]).read_text(encoding="utf-8").find(
        "lowering the analysis precision floor"
    ) >= 0
