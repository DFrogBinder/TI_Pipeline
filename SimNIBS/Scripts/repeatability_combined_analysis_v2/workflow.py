"""Materialize the final participant-level native repeatability package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from ti_current_repair.repeatability_v2.precision import write_metric_table

from .assembly import MetricSource, assemble_sources
from .settings import METRICS, expected_participant_cells
from .statistics import (
    build_matched_decomposition,
    summarize_participant_cells,
    summarize_under_anode,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_final_analysis(
    *, sources: Iterable[MetricSource], output_root: Path
) -> dict[str, object]:
    if output_root.exists() or output_root.is_symlink():
        raise ValueError(f"Final analysis output root must be new: {output_root}")
    source_list = tuple(sources)
    frame = assemble_sources(source_list, final=True)
    cells = summarize_participant_cells(frame)
    decomposition = build_matched_decomposition(cells)
    under = summarize_under_anode(cells)
    output_root.mkdir(parents=True)
    outputs = {
        "run_level_matrix.csv": frame,
        "participant_cell_summary.csv": cells,
        "matched_target_decomposition.csv": decomposition,
        "under_anode_application_summary.csv": under,
    }
    for filename, table in outputs.items():
        write_metric_table(output_root / filename, table.to_dict(orient="records"))
    receipt: dict[str, object] = {
        "schema_version": 2,
        "status": "complete",
        "analysis_dtype": "float64",
        "run_level_rows": len(frame),
        "participant_cells": len(cells) // len(METRICS),
        "participants_are_inferential_units": True,
        "repetitions_are_nested_technical_replicates": True,
        "outputs": {
            filename: {
                "path": str((output_root / filename).resolve()),
                "rows": len(table),
                "sha256": _sha256(output_root / filename),
            }
            for filename, table in outputs.items()
        },
        "sources": [
            {
                "path": str(source.path.resolve()),
                "sha256": _sha256(source.path),
                "version": source.version,
                "target": source.target,
                "condition": source.condition,
                "outcome_mode": source.outcome_mode,
                "preliminary": source.preliminary,
            }
            for source in source_list
        ],
        "interpretation_boundaries": {
            "matched_target": "supports_ti_component_variability_decomposition",
            "under_anode": "separate_application_facing_tdcs",
            "clinical_claims": False,
        },
    }
    if receipt["participant_cells"] != expected_participant_cells():
        raise AssertionError("Participant-cell summary count changed unexpectedly")
    (output_root / "analysis_manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt
