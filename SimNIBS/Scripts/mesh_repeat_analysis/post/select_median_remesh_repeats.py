#!/usr/bin/env python3
"""Select representative remesh repeats from this experiment's analysis tree."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

from pipeline import provenance


REPEAT_RE = re.compile(r"repeat_(\d+)$")


@dataclass(frozen=True)
class MedianSelection:
    subject: str
    selection_status: str
    repeat_tag: str
    metric: str
    metric_value: float | None
    median_target: float | None
    selected_m2m_dir: Path | None
    selected_mesh_path: Path | None
    mesh_nodes: str
    mesh_checksum: str
    summary_csv: Path


def output_fields() -> list[str]:
    return [
        "subject",
        "selection_status",
        "selected_repeat_tag",
        "metric",
        "metric_value",
        "median_target",
        "selected_m2m_dir",
        "selected_mesh_path",
        "mesh_nodes",
        "mesh_checksum",
        "summary_csv",
    ]


def _repeat_index(tag: str) -> int:
    match = REPEAT_RE.search(tag)
    return int(match.group(1)) if match else 10**9


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out


def _median_target(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return 0.5 * (ordered[midpoint - 1] + ordered[midpoint])


def _read_metric_rows(summary_csv: Path, metric: str) -> list[dict[str, str]]:
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    usable = []
    for row in rows:
        value = _safe_float(row.get(metric))
        if math.isfinite(value) and row.get("repeat_tag"):
            row["_metric_value"] = str(value)
            usable.append(row)
    return usable


def _selection_for_subject(experiment_root: Path, subject: str, metric: str) -> MedianSelection:
    summary_csv = experiment_root / "_analysis" / subject / "remesh" / "summary.csv"
    if not summary_csv.is_file():
        return MedianSelection(
            subject=subject,
            selection_status="missing_summary",
            repeat_tag="",
            metric=metric,
            metric_value=None,
            median_target=None,
            selected_m2m_dir=None,
            selected_mesh_path=None,
            mesh_nodes="",
            mesh_checksum="",
            summary_csv=summary_csv,
        )

    rows = _read_metric_rows(summary_csv, metric)
    if not rows:
        return MedianSelection(
            subject=subject,
            selection_status="no_metric_rows",
            repeat_tag="",
            metric=metric,
            metric_value=None,
            median_target=None,
            selected_m2m_dir=None,
            selected_mesh_path=None,
            mesh_nodes="",
            mesh_checksum="",
            summary_csv=summary_csv,
        )

    values = [_safe_float(row[metric]) for row in rows]
    target = _median_target(values)
    selected = min(
        rows,
        key=lambda row: (abs(_safe_float(row[metric]) - target), _repeat_index(str(row["repeat_tag"]))),
    )
    repeat_tag = str(selected["repeat_tag"])
    selected_m2m_dir = (
        experiment_root
        / f"{subject}_repeatability"
        / "remesh"
        / "repeats"
        / repeat_tag
        / subject
        / "anat"
        / f"m2m_{subject}"
    )
    selected_mesh_path = selected_m2m_dir / f"{subject}.msh"
    status = "selected" if selected_mesh_path.is_file() else "missing_mesh"
    checksum = provenance.file_sha256(selected_mesh_path) if selected_mesh_path.is_file() else ""
    return MedianSelection(
        subject=subject,
        selection_status=status,
        repeat_tag=repeat_tag,
        metric=metric,
        metric_value=_safe_float(selected[metric]),
        median_target=target,
        selected_m2m_dir=selected_m2m_dir,
        selected_mesh_path=selected_mesh_path,
        mesh_nodes=str(selected.get("mesh_nodes", "")),
        mesh_checksum=checksum,
        summary_csv=summary_csv,
    )


def _format_optional_float(value: float | None) -> str:
    return "" if value is None else str(value)


def _write_output(path: Path, selections: list[MedianSelection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields())
        writer.writeheader()
        for selection in selections:
            writer.writerow(
                {
                    "subject": selection.subject,
                    "selection_status": selection.selection_status,
                    "selected_repeat_tag": selection.repeat_tag,
                    "metric": selection.metric,
                    "metric_value": _format_optional_float(selection.metric_value),
                    "median_target": _format_optional_float(selection.median_target),
                    "selected_m2m_dir": str(selection.selected_m2m_dir or ""),
                    "selected_mesh_path": str(selection.selected_mesh_path or ""),
                    "mesh_nodes": selection.mesh_nodes,
                    "mesh_checksum": selection.mesh_checksum,
                    "summary_csv": str(selection.summary_csv),
                }
            )


def select_medians(
    *,
    experiment_root: Path,
    subjects: list[str],
    metric: str,
    output_csv: Path,
) -> list[MedianSelection]:
    selections = [_selection_for_subject(experiment_root, subject, metric) for subject in subjects]
    _write_output(output_csv, selections)
    return selections


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--subjects", required=True, help="Comma-separated subject IDs.")
    parser.add_argument("--metric", default="median_roi", choices=("median_roi", "mean_roi", "peak_roi"))
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment_root = args.experiment_root.expanduser().resolve()
    output_csv = args.output_csv or (
        experiment_root / "_pipeline" / "median_mesh_selection" / "median_representative_remesh_repeats.csv"
    )
    subjects = [subject.strip() for subject in args.subjects.split(",") if subject.strip()]
    selections = select_medians(
        experiment_root=experiment_root,
        subjects=subjects,
        metric=args.metric,
        output_csv=output_csv,
    )
    counts: dict[str, int] = {}
    for selection in selections:
        counts[selection.selection_status] = counts.get(selection.selection_status, 0) + 1
    print(f"wrote {output_csv}")
    print(", ".join(f"{key}={counts[key]}" for key in sorted(counts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
