#!/usr/bin/env python3
"""Create a paper-facing figure for the balanced 40-by-40 nested experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_METRIC = "roi_median_v_per_m"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv_atomic(
    path: Path,
    *,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _finite(value: object, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected finite {label}, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Expected finite {label}, got {value!r}")
    return parsed


def _balanced_values(
    *,
    metrics_csv: Path,
    metric: str,
    expected_meshes: int,
    expected_repeats: int,
) -> tuple[list[str], list[str], list[list[float]], dict[str, str]]:
    rows = _read_csv(metrics_csv)
    required = {"subject", "condition", "repeat_tag", "roi", metric}
    if not rows:
        raise ValueError(f"Nested metrics CSV is empty: {metrics_csv}")
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(f"Nested metrics CSV lacks: {', '.join(missing)}")

    subjects = {row["subject"] for row in rows}
    rois = {row["roi"] for row in rows}
    if len(subjects) != 1 or len(rois) != 1:
        raise ValueError("Nested metrics must contain one participant and one ROI")
    mesh_tags = [f"mesh_{index:03d}" for index in range(1, expected_meshes + 1)]
    repeat_tags = [
        f"repeat_{index:03d}" for index in range(1, expected_repeats + 1)
    ]
    if {row["condition"] for row in rows} != set(mesh_tags):
        raise ValueError("Nested metrics do not contain the expected mesh conditions")
    keys = [(row["condition"], row["repeat_tag"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Nested metrics contain duplicate mesh/repeat keys")

    by_key = {
        (row["condition"], row["repeat_tag"]): _finite(
            row[metric], label=f"{row['condition']}/{row['repeat_tag']}/{metric}"
        )
        for row in rows
    }
    values = []
    for mesh_tag in mesh_tags:
        try:
            values.append([by_key[(mesh_tag, tag)] for tag in repeat_tags])
        except KeyError as exc:
            raise ValueError(f"Nested metrics are incomplete at {exc.args[0]}") from exc
    expected_rows = expected_meshes * expected_repeats
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} nested rows, got {len(rows)}")
    return mesh_tags, repeat_tags, values, {
        "subject": next(iter(subjects)),
        "roi": next(iter(rois)),
    }


def _variance_components(values: list[list[float]]) -> dict[str, float]:
    outer_count = len(values)
    inner_count = len(values[0])
    mesh_means = [mean(group) for group in values]
    grand_mean = mean(mesh_means)
    within_ss = sum(
        sum((value - group_mean) ** 2 for value in group)
        for group, group_mean in zip(values, mesh_means)
    )
    between_ss = inner_count * sum(
        (group_mean - grand_mean) ** 2 for group_mean in mesh_means
    )
    ms_within = within_ss / (outer_count * (inner_count - 1))
    ms_between = between_ss / (outer_count - 1)
    within_variance = ms_within
    between_variance = max((ms_between - ms_within) / inner_count, 0.0)
    total_variance = between_variance + within_variance
    denominator = abs(grand_mean)
    between_sd = math.sqrt(between_variance)
    within_sd = math.sqrt(within_variance)
    return {
        "grand_mean": grand_mean,
        "between_mesh_variance": between_variance,
        "between_mesh_sd": between_sd,
        "between_mesh_cv_percent": 100.0 * between_sd / denominator,
        "within_mesh_variance": within_variance,
        "within_mesh_sd": within_sd,
        "within_mesh_cv_percent": 100.0 * within_sd / denominator,
        "total_variance": total_variance,
        "mesh_variance_fraction": between_variance / total_variance,
        "within_variance_fraction": within_variance / total_variance,
        "mesh_icc": between_variance / total_variance,
        "sd_ratio_between_over_within": between_sd / within_sd,
    }


def _per_mesh_summaries(
    *,
    mesh_tags: list[str],
    values: list[list[float]],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for mesh_tag, group in zip(mesh_tags, values):
        if len(group) < 2:
            raise ValueError("Each mesh requires at least two fixed-mesh repeats")
        mesh_mean = mean(group)
        if mesh_mean == 0.0:
            raise ValueError(f"Cannot calculate a CV for zero-valued {mesh_tag}")
        within_variance = sum((value - mesh_mean) ** 2 for value in group) / (
            len(group) - 1
        )
        within_sd = math.sqrt(within_variance)
        summaries.append(
            {
                "mesh_tag": mesh_tag,
                "repeat_count": len(group),
                "mesh_mean_v_per_m": mesh_mean,
                "within_mesh_sd_v_per_m": within_sd,
                "within_mesh_cv_percent": 100.0 * within_sd / abs(mesh_mean),
            }
        )
    ordered = sorted(
        summaries,
        key=lambda row: (float(row["mesh_mean_v_per_m"]), str(row["mesh_tag"])),
    )
    for position, row in enumerate(ordered, start=1):
        row["ordered_position"] = position
    return ordered


def _assert_close(actual: float, expected: object, *, label: str) -> None:
    expected_float = _finite(expected, label=label)
    if not math.isclose(actual, expected_float, rel_tol=1e-8, abs_tol=1e-12):
        raise ValueError(
            f"Recomputed {label} {actual} differs from analysis value {expected_float}"
        )


def _validate_existing_analysis(
    *,
    result: dict[str, Any],
    components: dict[str, float],
    metric: str,
    expected_meshes: int,
    expected_repeats: int,
    metadata: dict[str, str],
) -> None:
    if result.get("status") != "complete":
        raise ValueError("Existing nested analysis is not complete")
    if int(result.get("outer_meshes", -1)) != expected_meshes:
        raise ValueError("Existing nested analysis has the wrong outer-mesh count")
    if int(result.get("inner_repeats_per_mesh", -1)) != expected_repeats:
        raise ValueError("Existing nested analysis has the wrong inner-repeat count")
    if str(result.get("metric")) != metric:
        raise ValueError("Existing nested analysis uses a different metric")
    if str(result.get("selected_subject")) != metadata["subject"]:
        raise ValueError("Existing analysis participant differs from the metric table")
    if str(result.get("roi")) != metadata["roi"]:
        raise ValueError("Existing analysis ROI differs from the metric table")
    nested = result.get("variance_components")
    if not isinstance(nested, dict):
        raise ValueError("Existing nested analysis lacks variance components")
    between = nested.get("between_mesh")
    within = nested.get("within_mesh_solver_pipeline")
    if not isinstance(between, dict) or not isinstance(within, dict):
        raise ValueError("Existing nested analysis has malformed variance components")
    _assert_close(components["grand_mean"], result.get("grand_mean"), label="grand mean")
    _assert_close(
        components["between_mesh_variance"],
        between.get("variance"),
        label="between-mesh variance",
    )
    _assert_close(
        components["within_mesh_variance"],
        within.get("variance"),
        label="within-mesh variance",
    )
    _assert_close(
        components["mesh_icc"],
        result.get("intraclass_correlation_mesh"),
        label="mesh ICC",
    )


def _render_figure(
    *,
    per_mesh: list[dict[str, object]],
    metadata: dict[str, str],
    components: dict[str, float],
    png_path: Path,
    svg_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    positions = np.arange(1, len(per_mesh) + 1)
    mesh_cvs = np.asarray(
        [float(row["within_mesh_cv_percent"]) for row in per_mesh], dtype=float
    )
    repeat_count = int(per_mesh[0]["repeat_count"])
    pooled_cv = components["within_mesh_cv_percent"]
    upper_limit = max(float(mesh_cvs.max()), pooled_cv) * 1.22
    if upper_limit <= 0.0:
        upper_limit = 1e-6

    figure, axis = plt.subplots(figsize=(11.2, 6.3))
    axis.vlines(
        positions,
        0.0,
        mesh_cvs,
        color="#9ecae1",
        linewidth=1.1,
        zorder=1,
    )
    axis.scatter(
        positions,
        mesh_cvs,
        s=48,
        color="#2171b5",
        edgecolor="white",
        linewidth=0.7,
        label=f"One mesh ({repeat_count} repeats)",
        zorder=3,
    )
    axis.axhline(
        pooled_cv,
        color="#b35806",
        linestyle="--",
        linewidth=1.5,
        label="Pooled within-mesh CV",
        zorder=2,
    )
    ticks = np.unique(
        np.rint(np.linspace(1, len(per_mesh), min(9, len(per_mesh)))).astype(int)
    )
    axis.set_xlim(0.3, len(per_mesh) + 0.7)
    axis.set_ylim(0.0, upper_limit)
    axis.set_xticks(ticks)
    axis.set_xlabel("Meshes ordered by mean spherical-ROI field (lowest → highest)")
    axis.set_ylabel(f"CV across {repeat_count} fixed-mesh repeats (%)")
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper left", ncol=2)

    participant = metadata["subject"].removeprefix("sub-")
    roi = metadata["roi"].replace("_", " ")
    figure.suptitle(
        "Fixed-mesh repeatability across independently generated meshes\n"
        f"{participant}, {roi}",
        fontsize=15.5,
        y=0.98,
    )
    figure.text(
        0.5,
        0.875,
        (
            f"Between-mesh CV: {components['between_mesh_cv_percent']:.3f}%   |   "
            f"Pooled within-mesh CV: {pooled_cv:.4f}%   |   "
            f"SD ratio: {components['sd_ratio_between_over_within']:.0f}×   |   "
            "Mesh variance share: "
            f"{100.0 * components['mesh_variance_fraction']:.4f}%"
        ),
        ha="center",
        va="center",
        fontsize=10.5,
        color="#333333",
    )
    figure.subplots_adjust(top=0.80, bottom=0.15, left=0.10, right=0.98)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)


def run(
    *,
    metrics_csv: Path,
    variance_json: Path,
    output_dir: Path,
    metric: str = DEFAULT_METRIC,
    expected_meshes: int = 40,
    expected_repeats: int = 40,
) -> dict[str, Any]:
    metrics_csv = metrics_csv.expanduser().resolve()
    variance_json = variance_json.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_tags, repeat_tags, values, metadata = _balanced_values(
        metrics_csv=metrics_csv,
        metric=metric,
        expected_meshes=expected_meshes,
        expected_repeats=expected_repeats,
    )
    components = _variance_components(values)
    per_mesh = _per_mesh_summaries(mesh_tags=mesh_tags, values=values)
    existing = _read_json(variance_json)
    _validate_existing_analysis(
        result=existing,
        components=components,
        metric=metric,
        expected_meshes=expected_meshes,
        expected_repeats=expected_repeats,
        metadata=metadata,
    )

    png_path = output_dir / "nested_mesh_by_solver_repeatability.png"
    svg_path = output_dir / "nested_mesh_by_solver_repeatability.svg"
    obsolete_outputs = [
        output_dir / "nested_within_mesh_residual_matrix_supplement.png",
        output_dir / "nested_within_mesh_residual_matrix_supplement.svg",
    ]
    removed_obsolete_outputs = []
    for obsolete_path in obsolete_outputs:
        if obsolete_path.is_file():
            obsolete_path.unlink()
            removed_obsolete_outputs.append(obsolete_path.name)
    _render_figure(
        per_mesh=per_mesh,
        metadata=metadata,
        components=components,
        png_path=png_path,
        svg_path=svg_path,
    )
    per_mesh_path = output_dir / "nested_per_mesh_repeatability.csv"
    _write_csv_atomic(
        per_mesh_path,
        fieldnames=[
            "ordered_position",
            "mesh_tag",
            "repeat_count",
            "mesh_mean_v_per_m",
            "within_mesh_sd_v_per_m",
            "within_mesh_cv_percent",
        ],
        rows=per_mesh,
    )
    variance_component_path = output_dir / "nested_variance_components.csv"
    _write_csv_atomic(
        variance_component_path,
        fieldnames=[
            "variation_source",
            "variance_v_per_m_squared",
            "sd_v_per_m",
            "cv_percent",
            "fraction_of_total_variance",
        ],
        rows=[
            {
                "variation_source": "between_mesh_generation",
                "variance_v_per_m_squared": components["between_mesh_variance"],
                "sd_v_per_m": components["between_mesh_sd"],
                "cv_percent": components["between_mesh_cv_percent"],
                "fraction_of_total_variance": components["mesh_variance_fraction"],
            },
            {
                "variation_source": "within_mesh_solver_pipeline",
                "variance_v_per_m_squared": components["within_mesh_variance"],
                "sd_v_per_m": components["within_mesh_sd"],
                "cv_percent": components["within_mesh_cv_percent"],
                "fraction_of_total_variance": components[
                    "within_variance_fraction"
                ],
            },
        ],
    )
    per_mesh_cvs = [float(row["within_mesh_cv_percent"]) for row in per_mesh]
    figure_values = {
        "schema_version": 1,
        "status": "complete",
        "created_utc": _utc_now(),
        "subject": metadata["subject"],
        "roi": metadata["roi"],
        "metric": metric,
        "outer_meshes": len(mesh_tags),
        "inner_repeats_per_mesh": len(repeat_tags),
        "observations": len(mesh_tags) * len(repeat_tags),
        **components,
        "per_mesh_within_cv_percent": {
            "minimum": min(per_mesh_cvs),
            "median": median(per_mesh_cvs),
            "maximum": max(per_mesh_cvs),
        },
        "interpretation_boundary": (
            "This nested analysis separates variation among mesh realizations "
            "from repeat variation conditional on each mesh for one randomly "
            "selected participant. It is not a population variance estimate."
        ),
    }
    values_path = output_dir / "nested_figure_values.json"
    _write_json_atomic(values_path, figure_values)
    caption_path = output_dir / "nested_figure_caption.md"
    caption_path.write_text(
        (
            "# Figure caption\n\n"
            "Fixed-mesh repeatability across independently generated meshes "
            f"for one randomly selected participant ({metadata['subject']}). "
            f"Each point is the coefficient of variation (CV) among "
            f"{len(repeat_tags)} repeated solutions on one mesh. The "
            f"{len(mesh_tags)} meshes are ordered by their mean spherical-ROI "
            "field solely to aid display. The dashed line is the pooled "
            "within-mesh CV from the balanced one-way random-effects model. "
            "The header reports the between-mesh CV, pooled within-mesh CV, "
            "ratio of component standard deviations, and fraction of total "
            "variance attributed to mesh generation.\n"
        ),
        encoding="utf-8",
    )
    result = {
        **figure_values,
        "inputs": {
            "metrics_csv": {"path": str(metrics_csv), "sha256": _sha256(metrics_csv)},
            "variance_json": {
                "path": str(variance_json),
                "sha256": _sha256(variance_json),
            },
        },
        "outputs": {
            "png": png_path.name,
            "svg": svg_path.name,
            "per_mesh_table": per_mesh_path.name,
            "variance_component_table": variance_component_path.name,
            "values": values_path.name,
            "caption": caption_path.name,
        },
        "removed_obsolete_outputs": removed_obsolete_outputs,
    }
    manifest_path = output_dir / "nested_figure_manifest.json"
    _write_json_atomic(manifest_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--variance-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--expected-meshes", type=int, default=40)
    parser.add_argument("--expected-repeats", type=int, default=40)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(
        metrics_csv=args.metrics_csv,
        variance_json=args.variance_json,
        output_dir=args.output_dir,
        metric=args.metric,
        expected_meshes=args.expected_meshes,
        expected_repeats=args.expected_repeats,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
