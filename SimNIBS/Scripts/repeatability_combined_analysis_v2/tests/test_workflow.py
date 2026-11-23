from __future__ import annotations

from pathlib import Path

import pytest

from repeatability_combined_analysis_v2.assembly import MetricSource, assemble_sources
from repeatability_combined_analysis_v2.tests.test_analysis import final_matrix
from repeatability_combined_analysis_v2.workflow import run_final_analysis
from ti_current_repair.repeatability_v2.precision import write_metric_table


def _sources(tmp_path: Path) -> list[MetricSource]:
    frame = final_matrix()
    sources = []
    for (version, target, condition, outcome), group in frame.groupby(
        ["version", "target", "condition", "outcome_mode"], sort=False
    ):
        path = tmp_path / "sources" / version / target / condition / f"{outcome}.csv"
        write_metric_table(path, group.to_dict(orient="records"))
        sources.append(
            MetricSource(
                version=version,
                target=target,
                condition=condition,
                outcome_mode=outcome,
                path=path,
                preliminary=False,
            )
        )
    return sources


def test_assembly_reads_round_trip_tables_and_rejects_preliminary_source(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    frame = assemble_sources(sources, final=True)
    assert len(frame) == 9600
    assert frame["roi_median_v_per_m"].dtype.name == "float64"

    preliminary = [
        MetricSource(
            version="simnibs401",
            target=sources[0].target,
            condition=sources[0].condition,
            outcome_mode=sources[0].outcome_mode,
            path=sources[0].path,
            preliminary=True,
        )
    ]
    with pytest.raises(ValueError, match="preliminary|4.0.1"):
        assemble_sources(preliminary, final=True)


def test_final_workflow_writes_all_outputs_without_modifying_sources(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    before = {source.path: source.path.stat().st_mtime_ns for source in sources}
    output = tmp_path / "analysis-v2"

    receipt = run_final_analysis(sources=sources, output_root=output)

    assert receipt["status"] == "complete"
    assert receipt["run_level_rows"] == 9600
    assert receipt["participant_cells"] == 240
    assert set(receipt["outputs"]) == {
        "run_level_matrix.csv",
        "participant_cell_summary.csv",
        "matched_target_decomposition.csv",
        "under_anode_application_summary.csv",
    }
    assert all((output / name).is_file() for name in receipt["outputs"])
    assert {source.path: source.path.stat().st_mtime_ns for source in sources} == before
