#!/usr/bin/env python3
"""Analyze a balanced mesh-by-within-mesh nested repeatability experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev


DEFAULT_METRIC = "roi_median_v_per_m"
METRIC_CHOICES = (
    "roi_min_v_per_m",
    "roi_mean_v_per_m",
    "roi_median_v_per_m",
    "roi_p95_v_per_m",
    "roi_max_v_per_m",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _write_csv_atomic(
    path: Path,
    rows: list[dict[str, object]],
    fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def _finite(value: object, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected finite {label}; got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Expected finite {label}; got {value!r}")
    return parsed


def _sample_variance(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("At least two observations are required")
    center = mean(values)
    return sum((value - center) ** 2 for value in values) / (len(values) - 1)


def analyze_nested(
    *,
    metrics_csv: Path,
    selection_manifest: Path,
    output_dir: Path,
    metric: str = DEFAULT_METRIC,
    expected_meshes: int = 40,
    expected_repeats_per_mesh: int = 40,
) -> dict[str, object]:
    if metric not in METRIC_CHOICES:
        raise ValueError(f"Unsupported metric {metric!r}")
    selection = _load_json(selection_manifest)
    subject = str(selection.get("selected_subject", ""))
    target = str(selection.get("selected_target", ""))
    if not subject or not target:
        raise ValueError("Nested selection manifest lacks subject/target")

    rows = _read_csv(metrics_csv)
    required = {"subject", "condition", "repeat_tag", "roi", metric}
    if not rows:
        raise ValueError(f"Nested metrics CSV is empty: {metrics_csv}")
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(f"Nested metrics CSV lacks: {', '.join(missing)}")
    if {row["subject"] for row in rows} != {subject}:
        raise ValueError("Nested metrics subject differs from the persisted selection")

    expected_conditions = {
        f"mesh_{index:03d}" for index in range(1, expected_meshes + 1)
    }
    observed_conditions = {row["condition"] for row in rows}
    if observed_conditions != expected_conditions:
        raise ValueError(
            "Nested mesh conditions differ from the expected balanced 40-mesh design"
        )
    expected_tags = {
        f"repeat_{index:03d}"
        for index in range(1, expected_repeats_per_mesh + 1)
    }
    keys = [(row["condition"], row["repeat_tag"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Nested metrics contain duplicate mesh/repeat keys")

    by_mesh: dict[str, list[float]] = {}
    roi_values = {row["roi"] for row in rows}
    if len(roi_values) != 1:
        raise ValueError("Nested metrics contain more than one ROI")
    for condition in sorted(expected_conditions):
        selected = [row for row in rows if row["condition"] == condition]
        if len(selected) != expected_repeats_per_mesh:
            raise ValueError(f"{condition}: unexpected within-mesh repeat count")
        if {row["repeat_tag"] for row in selected} != expected_tags:
            raise ValueError(f"{condition}: within-mesh repeat tags are incomplete")
        selected.sort(key=lambda row: row["repeat_tag"])
        by_mesh[condition] = [
            _finite(row[metric], label=f"{condition}/{row['repeat_tag']}/{metric}")
            for row in selected
        ]

    mesh_means = [mean(values) for values in by_mesh.values()]
    grand_mean = mean(mesh_means)
    within_ss = sum(
        sum((value - mean(values)) ** 2 for value in values)
        for values in by_mesh.values()
    )
    within_df = expected_meshes * (expected_repeats_per_mesh - 1)
    between_ss = expected_repeats_per_mesh * sum(
        (mesh_mean - grand_mean) ** 2 for mesh_mean in mesh_means
    )
    between_df = expected_meshes - 1
    ms_within = within_ss / within_df
    ms_between = between_ss / between_df
    solver_variance = ms_within
    raw_mesh_variance = (ms_between - ms_within) / expected_repeats_per_mesh
    mesh_variance = max(raw_mesh_variance, 0.0)
    total_variance = mesh_variance + solver_variance
    mesh_sd = math.sqrt(mesh_variance)
    solver_sd = math.sqrt(solver_variance)
    total_sd = math.sqrt(total_variance)
    denominator = abs(grand_mean)

    mesh_rows: list[dict[str, object]] = []
    for condition, values in sorted(by_mesh.items()):
        mesh_rows.append(
            {
                "condition": condition,
                "source_repeat_tag": condition.replace("mesh_", "repeat_"),
                "n_inner_repeats": len(values),
                "mean": mean(values),
                "median": median(values),
                "sd": stdev(values),
                "min": min(values),
                "max": max(values),
                "range": max(values) - min(values),
            }
        )

    variance_rows = [
        {
            "component": "between_mesh",
            "variance": mesh_variance,
            "sd": mesh_sd,
            "cv_percent": 100.0 * mesh_sd / denominator if denominator else None,
            "fraction_of_total_variance": (
                mesh_variance / total_variance if total_variance else None
            ),
        },
        {
            "component": "within_mesh_solver_pipeline",
            "variance": solver_variance,
            "sd": solver_sd,
            "cv_percent": 100.0 * solver_sd / denominator if denominator else None,
            "fraction_of_total_variance": (
                solver_variance / total_variance if total_variance else None
            ),
        },
        {
            "component": "total",
            "variance": total_variance,
            "sd": total_sd,
            "cv_percent": 100.0 * total_sd / denominator if denominator else None,
            "fraction_of_total_variance": 1.0 if total_variance else None,
        },
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_summary_path = output_dir / "nested_mesh_summary.csv"
    variance_csv_path = output_dir / "nested_variance_components.csv"
    _write_csv_atomic(
        mesh_summary_path,
        mesh_rows,
        [
            "condition",
            "source_repeat_tag",
            "n_inner_repeats",
            "mean",
            "median",
            "sd",
            "min",
            "max",
            "range",
        ],
    )
    _write_csv_atomic(
        variance_csv_path,
        variance_rows,
        [
            "component",
            "variance",
            "sd",
            "cv_percent",
            "fraction_of_total_variance",
        ],
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "created_utc": _utc_now(),
        "analysis": "balanced one-way random-effects nested variance decomposition",
        "selected_subject": subject,
        "selected_target": target,
        "roi": next(iter(roi_values)),
        "metric": metric,
        "outer_meshes": expected_meshes,
        "inner_repeats_per_mesh": expected_repeats_per_mesh,
        "observations": len(rows),
        "grand_mean": grand_mean,
        "mesh_mean_min": min(mesh_means),
        "mesh_mean_max": max(mesh_means),
        "mesh_mean_range": max(mesh_means) - min(mesh_means),
        "anova": {
            "between_mesh_df": between_df,
            "within_mesh_df": within_df,
            "between_mesh_mean_square": ms_between,
            "within_mesh_mean_square": ms_within,
            "raw_between_mesh_variance_estimate": raw_mesh_variance,
            "negative_between_mesh_estimates_truncated_to_zero": raw_mesh_variance < 0,
        },
        "variance_components": {
            row["component"]: {
                key: value for key, value in row.items() if key != "component"
            }
            for row in variance_rows
        },
        "intraclass_correlation_mesh": (
            mesh_variance / total_variance if total_variance else None
        ),
        "interpretation_boundary": (
            "The between-mesh component measures variation among 40 valid mesh "
            "realizations for one persisted participant-target case. The within-mesh "
            "component measures repeat variation conditional on those meshes. This "
            "single-case nested analysis is a sensitivity demonstration, not a "
            "population estimate."
        ),
        "source_metrics_csv": str(metrics_csv),
        "selection_manifest": str(selection_manifest),
        "mesh_summary_csv": str(mesh_summary_path),
        "variance_components_csv": str(variance_csv_path),
    }
    _write_json_atomic(output_dir / "nested_variance_components.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric", choices=METRIC_CHOICES, default=DEFAULT_METRIC)
    parser.add_argument("--expected-meshes", type=int, default=40)
    parser.add_argument("--expected-repeats-per-mesh", type=int, default=40)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = analyze_nested(
        metrics_csv=args.metrics_csv.expanduser().resolve(),
        selection_manifest=args.selection_manifest.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        metric=args.metric,
        expected_meshes=args.expected_meshes,
        expected_repeats_per_mesh=args.expected_repeats_per_mesh,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
