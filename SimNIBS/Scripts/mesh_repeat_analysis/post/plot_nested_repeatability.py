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
from statistics import mean
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
    values: list[list[float]],
    metadata: dict[str, str],
    components: dict[str, float],
    png_path: Path,
    svg_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    array = np.asarray(values, dtype=float)
    mesh_means = array.mean(axis=1)
    mesh_min = array.min(axis=1)
    mesh_max = array.max(axis=1)
    residual_micro = (array - mesh_means[:, None]) * 1_000_000.0
    residual_limit = max(float(np.max(np.abs(residual_micro))), 1e-12)
    x = np.arange(1, len(values) + 1)
    field_ticks = np.unique(
        np.rint(np.linspace(1, len(values), min(9, len(values)))).astype(int)
    )
    residual_x_ticks = np.unique(
        np.rint(np.linspace(1, array.shape[1], min(5, array.shape[1]))).astype(int)
    )
    residual_y_ticks = np.unique(
        np.rint(np.linspace(1, array.shape[0], min(5, array.shape[0]))).astype(int)
    )

    figure = plt.figure(figsize=(13.2, 9.2), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(1.05, 1.0))
    field_axis = figure.add_subplot(grid[0, :])
    residual_axis = figure.add_subplot(grid[1, 0])
    component_axis = figure.add_subplot(grid[1, 1])

    rng = np.random.default_rng(20260831)
    for index, group in enumerate(array):
        jitter = rng.uniform(-0.10, 0.10, size=len(group))
        field_axis.scatter(
            np.full(len(group), x[index]) + jitter,
            group,
            s=10,
            color="#9ecae1",
            alpha=0.30,
            edgecolors="none",
            zorder=2,
        )
    field_axis.vlines(x, mesh_min, mesh_max, color="#6baed6", linewidth=0.8, zorder=3)
    field_axis.scatter(
        x,
        mesh_means,
        s=28,
        facecolor="white",
        edgecolor="#08519c",
        linewidth=1.2,
        label="Mesh mean",
        zorder=4,
    )
    field_axis.axhline(
        components["grand_mean"],
        color="#b35806",
        linestyle="--",
        linewidth=1.4,
        label="Grand mean",
        zorder=1,
    )
    field_axis.set_xlim(0.3, len(values) + 0.7)
    field_axis.set_xticks(field_ticks)
    field_axis.set_xlabel("Outer mesh realization")
    field_axis.set_ylabel("Median TIS field in target ROI (V/m)")
    field_axis.set_title("A  Field estimates conditional on each mesh", loc="left")
    field_axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    field_axis.spines[["top", "right"]].set_visible(False)
    field_axis.legend(frameon=False, ncol=2, loc="best")

    image = residual_axis.imshow(
        residual_micro,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-residual_limit,
        vmax=residual_limit,
    )
    residual_axis.set_xticks(residual_x_ticks - 1, residual_x_ticks)
    residual_axis.set_yticks(residual_y_ticks - 1, residual_y_ticks)
    residual_axis.set_xlabel("Within-mesh solver/pipeline repeat")
    residual_axis.set_ylabel("Outer mesh realization")
    residual_axis.set_title("B  Within-mesh residuals", loc="left")
    colorbar = figure.colorbar(image, ax=residual_axis, fraction=0.047, pad=0.03)
    colorbar.set_label("Deviation from mesh mean (µV/m)")

    labels = ["Between meshes", "Within mesh"]
    cvs = np.asarray(
        [
            components["between_mesh_cv_percent"],
            components["within_mesh_cv_percent"],
        ]
    )
    component_axis.scatter(
        cvs,
        [1, 0],
        s=90,
        color=["#08519c", "#b35806"],
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    component_axis.set_xscale("log")
    component_axis.set_yticks([1, 0], labels)
    component_axis.set_ylim(-0.7, 1.7)
    component_axis.set_xlim(cvs.min() / 3.0, cvs.max() * 3.0)
    component_axis.set_xlabel("Coefficient of variation (%) on logarithmic scale")
    component_axis.set_title("C  Estimated variance components", loc="left")
    component_axis.grid(axis="x", which="both", color="#dddddd", linewidth=0.7)
    component_axis.spines[["top", "right", "left"]].set_visible(False)
    component_axis.tick_params(axis="y", length=0)
    for value, position in zip(cvs, [1, 0]):
        component_axis.annotate(
            f"{value:.4g}%",
            (value, position),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
        )
    component_axis.text(
        0.02,
        0.05,
        (
            f"Between-mesh fraction: "
            f"{100.0 * components['mesh_variance_fraction']:.6f}%\n"
            f"Mesh ICC: {components['mesh_icc']:.6f}\n"
            f"SD ratio: {components['sd_ratio_between_over_within']:.1f}×"
        ),
        transform=component_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
    )

    participant = metadata["subject"].removeprefix("sub-")
    roi = metadata["roi"].replace("_", " ")
    figure.suptitle(
        f"Nested mesh-by-solver repeatability for {participant}, {roi}",
        fontsize=16,
    )
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
    _render_figure(
        values=values,
        metadata=metadata,
        components=components,
        png_path=png_path,
        svg_path=svg_path,
    )
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
            "Nested mesh-by-solver repeatability for one randomly selected "
            f"participant ({metadata['subject']}). Panel A shows all "
            f"{len(mesh_tags) * len(repeat_tags):,} spherical-ROI median field "
            f"estimates grouped by the {len(mesh_tags)} independently generated "
            "meshes, with "
            "mesh means and the grand mean. Panel B shows each estimate after "
            "subtracting its mesh-specific mean, which exposes the much smaller "
            "within-mesh solver/pipeline variation. Panel C compares the "
            "between-mesh and within-mesh coefficients of variation on a "
            "logarithmic scale. The variance components were estimated with a "
            "balanced one-way random-effects decomposition.\n"
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
            "values": values_path.name,
            "caption": caption_path.name,
        },
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
