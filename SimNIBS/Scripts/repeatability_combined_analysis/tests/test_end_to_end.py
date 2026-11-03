from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

from repeatability_combined_analysis.assembly import build_assembly_plan
from repeatability_combined_analysis.workflow import (
    execute,
    preflight,
    prepare,
    rebind_preflight,
    status,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_complete_synthetic_analysis_and_dataset_rebind_are_source_read_only(
    analysis_config, tmp_path: Path
) -> None:
    plan = build_assembly_plan(analysis_config)
    source_paths = [selection.path for selection in plan.selections]
    source_paths.extend(analysis_config.sources.receipts)
    before = {path: _sha256(path) for path in source_paths}

    ready = preflight(analysis_config)
    assert ready["scope"]["run_level_rows"] == 3200
    assert ready["scope"]["participant_cells"] == 80
    prepare(analysis_config, str(analysis_config.output_root))
    completed = execute(analysis_config)
    assert completed["status"] == "complete"
    assert completed["scope"]["run_level_rows"] == 3200
    assert completed["scope"]["participant_cells"] == 80
    assert status(analysis_config)["state"] == "complete"

    root = analysis_config.output_root
    manifest = json.loads((root / "analysis_manifest.json").read_text())
    assert manifest["established"]["within_simnibs326"]["rows"] == 1600
    assert manifest["established"]["within_simnibs4"]["rows"] == 1600
    assert (
        manifest["established"]["cross_version_remesh"]["input_rows"]
        == 3200
    )
    assert (
        manifest["established"]["cross_version_remesh"][
            "selected_remesh_rows"
        ]
        == 1600
    )
    report = (root / "comprehensive/analysis_report.md").read_text().lower()
    assert "participant is the inferential unit" in report
    assert "40 technical repeats" in report
    assert "primary" in report and "sensitivity" in report
    assert "mechanistic control" in report
    assert "not identical" in report
    assert "no therapeutic" in report
    figures = sorted((root / "comprehensive/figures").rglob("*.png"))
    assert len(figures) == 8
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in figures)

    dataset = tmp_path / "completed_dataset"
    (dataset / "_consolidation").mkdir(parents=True)
    (dataset / "provenance").mkdir()
    (dataset / "_consolidation/complete.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "workflow": "repeatability_dataset_consolidation",
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for index, selection in enumerate(plan.selections):
        destination = dataset / "analysis_sources" / f"source-{index}.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(selection.path, destination)
        rows.append(
            {
                "source": str(selection.path),
                "destination": str(destination),
                "role": "analysis_metric_table",
                "subject": "",
                "version": "",
                "target": "",
                "condition": "",
                "verification_status": "checksum_matched",
            }
        )
    with (dataset / "provenance/source_mapping.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    binding = rebind_preflight(analysis_config, dataset)
    assert binding["status"] == "matched"
    assert binding["bindings"] == 10
    assert binding["writes_performed"] == 0

    after = {path: _sha256(path) for path in source_paths}
    assert before == after

