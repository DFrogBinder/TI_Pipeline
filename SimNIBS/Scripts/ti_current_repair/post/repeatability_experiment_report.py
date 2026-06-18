#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired-condition analysis for remesh vs fixed-mesh repeatability experiments."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import traceback
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from experiment_config import (  # noqa: E402
    condition_by_name,
    load_experiment_config,
    subject_analysis_root,
    subject_condition_repeats_root,
)
from post import mesh_repeat_report as base_report  # noqa: E402


KEY_COMPARISON_METRICS = [
    "median_roi",
    "mean_roi",
    "peak_roi",
    "median_head",
    "mean_head",
    "peak_head",
    "high_field_dice_head",
    "hotspot_distance_head_mm",
    "diff_fraction",
    "diff_fraction_roi",
    "mesh_nodes",
]


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return float("nan")
    if denominator == 0.0:
        return float("inf") if numerator > 0 else float("nan")
    return numerator / denominator


def _safe_reduction_percent(reference: float, comparison: float) -> float:
    if not math.isfinite(reference) or not math.isfinite(comparison) or reference == 0.0:
        return float("nan")
    return 100.0 * (1.0 - (comparison / reference))


def _load_summary_rows(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [dict(row) for row in reader]
    numeric_fields = {
        "diff_fraction",
        "diff_fraction_roi",
        "mean_roi",
        "median_roi",
        "peak_roi",
        "mean_head",
        "median_head",
        "peak_head",
        "high_field_dice_head",
        "high_field_centroid_distance_mm",
        "hotspot_distance_head_mm",
        "hotspot_distance_roi_mm",
        "ti_scale_factor",
        "mesh_nodes",
        "label_count",
    }
    for row in rows:
        for field in numeric_fields:
            if field in row:
                row[field] = _safe_float(row[field])
    return rows


def _metric_comparison_rows(
    condition_results: dict[str, dict[str, object]],
    *,
    baseline_condition: str,
    comparison_condition: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    baseline_metrics = condition_results[baseline_condition]["metric_stats_map"]
    comparison_metrics = condition_results[comparison_condition]["metric_stats_map"]
    for metric in KEY_COMPARISON_METRICS:
        base_stats = baseline_metrics.get(metric, {})
        comp_stats = comparison_metrics.get(metric, {})
        base_std = _safe_float(base_stats.get("std"))
        comp_std = _safe_float(comp_stats.get("std"))
        base_cv = _safe_float(base_stats.get("cv_percent"))
        comp_cv = _safe_float(comp_stats.get("cv_percent"))
        rows.append(
            {
                "metric": metric,
                "baseline_condition": baseline_condition,
                "comparison_condition": comparison_condition,
                "baseline_mean": _safe_float(base_stats.get("mean")),
                "comparison_mean": _safe_float(comp_stats.get("mean")),
                "baseline_std": base_std,
                "comparison_std": comp_std,
                "baseline_cv_percent": base_cv,
                "comparison_cv_percent": comp_cv,
                "std_ratio_comparison_over_baseline": _safe_ratio(comp_std, base_std),
                "std_reduction_percent": _safe_reduction_percent(base_std, comp_std),
                "cv_ratio_comparison_over_baseline": _safe_ratio(comp_cv, base_cv),
                "cv_reduction_percent": _safe_reduction_percent(base_cv, comp_cv),
            }
        )
    return rows


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _format_float(value: float, *, pct: bool = False) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.2f}%" if pct else f"{value:.6f}"


def _write_condition_comparison_report(
    *,
    path: Path,
    subject: str,
    condition_results: dict[str, dict[str, object]],
    metric_rows: list[dict[str, object]],
    baseline_condition: str,
    comparison_condition: str,
    compare_metric: str,
) -> None:
    primary_row = next((row for row in metric_rows if row["metric"] == compare_metric), None)
    lines = [
        f"# Paired Repeatability Comparison: {subject}",
        "",
        f"- Baseline condition: `{baseline_condition}`",
        f"- Comparison condition: `{comparison_condition}`",
        f"- Primary comparison metric: `{compare_metric}`",
        "",
        "## Condition Summary",
        "",
        "| Condition | Mean Median ROI | SD Median ROI | CV Median ROI | Mesh Node SD | Mesh Node CV |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for condition_name, result in condition_results.items():
        metrics = result["metric_stats_map"]
        lines.append(
            "| "
            f"{condition_name} | "
            f"{_format_float(_safe_float(metrics['median_roi']['mean']))} | "
            f"{_format_float(_safe_float(metrics['median_roi']['std']))} | "
            f"{_format_float(_safe_float(metrics['median_roi']['cv_percent']), pct=True)} | "
            f"{_format_float(_safe_float(metrics['mesh_nodes']['std']))} | "
            f"{_format_float(_safe_float(metrics['mesh_nodes']['cv_percent']), pct=True)} |"
        )

    if primary_row is not None:
        lines.extend(
            [
                "",
                "## Primary Contrast",
                "",
                f"- `{baseline_condition}` SD: {_format_float(primary_row['baseline_std'])}",
                f"- `{comparison_condition}` SD: {_format_float(primary_row['comparison_std'])}",
                f"- SD ratio (`{comparison_condition}` / `{baseline_condition}`): "
                f"{_format_float(primary_row['std_ratio_comparison_over_baseline'])}",
                f"- SD reduction from `{baseline_condition}` to `{comparison_condition}`: "
                f"{_format_float(primary_row['std_reduction_percent'], pct=True)}",
                f"- CV reduction from `{baseline_condition}` to `{comparison_condition}`: "
                f"{_format_float(primary_row['cv_reduction_percent'], pct=True)}",
            ]
        )

    lines.extend(
        [
            "",
            "## Metric-by-Metric Comparison",
            "",
            "| Metric | Baseline SD | Comparison SD | SD Ratio | SD Reduction | Baseline CV | Comparison CV | CV Reduction |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in metric_rows:
        lines.append(
            "| "
            f"{row['metric']} | "
            f"{_format_float(_safe_float(row['baseline_std']))} | "
            f"{_format_float(_safe_float(row['comparison_std']))} | "
            f"{_format_float(_safe_float(row['std_ratio_comparison_over_baseline']))} | "
            f"{_format_float(_safe_float(row['std_reduction_percent']), pct=True)} | "
            f"{_format_float(_safe_float(row['baseline_cv_percent']), pct=True)} | "
            f"{_format_float(_safe_float(row['comparison_cv_percent']), pct=True)} | "
            f"{_format_float(_safe_float(row['cv_reduction_percent']), pct=True)} |"
        )

    _write_markdown(path, "\n".join(lines) + "\n")


def _plot_condition_metric_bars(
    *,
    out_path: Path,
    metric_rows: list[dict[str, object]],
    ylabel: str,
    value_key_baseline: str,
    value_key_comparison: str,
    baseline_condition: str,
    comparison_condition: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [row["metric"] for row in metric_rows]
    baseline_vals = np.asarray([_safe_float(row[value_key_baseline]) for row in metric_rows], dtype=float)
    comparison_vals = np.asarray([_safe_float(row[value_key_comparison]) for row in metric_rows], dtype=float)

    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(max(10, len(labels) * 0.65), 5))
    plt.bar(x - width / 2, baseline_vals, width=width, label=baseline_condition)
    plt.bar(x + width / 2, comparison_vals, width=width, label=comparison_condition)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by condition")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_condition_lines(
    *,
    out_path: Path,
    condition_rows: dict[str, list[dict[str, object]]],
    metric: str,
    ylabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(max(10, max(len(rows) for rows in condition_rows.values()) * 0.22), 4))
    for condition_name, rows in condition_rows.items():
        tags = [str(row["repeat_tag"]) for row in rows]
        vals = [_safe_float(row[metric]) for row in rows]
        plt.plot(tags, vals, marker="o", label=condition_name)
    plt.xticks(rotation=90, fontsize=7)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by condition")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _resolve_roi_from_config(args: argparse.Namespace, config) -> tuple[str, list[int]]:
    roi_args = copy.deepcopy(args)
    if roi_args.roi_preset is None and roi_args.roi_name is None and roi_args.roi_labels is None:
        roi_args.roi_preset = config.analysis.roi_preset
        roi_args.roi_name = config.analysis.roi_name
        if config.analysis.roi_labels:
            roi_args.roi_labels = ",".join(str(value) for value in config.analysis.roi_labels)
    return base_report._resolve_roi_selection(roi_args)


def _build_single_condition_args(args: argparse.Namespace, config) -> argparse.Namespace:
    single_args = argparse.Namespace()
    single_args.rootdir = str(config.experiment_root)
    single_args.repeats_dir = None
    single_args.max_subjects = None
    single_args.t1_root = None
    single_args.atlas = None
    single_args.atlas_dir = args.atlas_dir or config.analysis.atlas_dir
    single_args.output_dir = None
    single_args.peak_percentile = None
    single_args.reference_repeat = args.reference_repeat
    single_args.spatial_percentile = args.spatial_percentile
    single_args.compare_metric = args.compare_metric or config.analysis.compare_metric or "median_roi"
    single_args.compare_cohort_root = None if args.skip_cohort else (args.compare_cohort_root or config.analysis.compare_cohort_root)
    single_args.cohort_region_name = args.cohort_region_name or config.analysis.cohort_region_name
    single_args.cohort_region_label = args.cohort_region_label or config.analysis.cohort_region_label
    single_args.cohort_metric = args.cohort_metric
    single_args.log_file = None
    return single_args


def _subject_summary_row(
    *,
    subject: str,
    subject_output_root: Path,
    result: dict[str, object] | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "subject": subject,
        "status": "complete" if result is not None else "failed",
        "analysis_root": str(subject_output_root),
        "baseline_condition": None,
        "comparison_condition": None,
        "compare_metric": None,
        "baseline_std": None,
        "comparison_std": None,
        "std_ratio_comparison_over_baseline": None,
        "std_reduction_percent": None,
        "baseline_cv_percent": None,
        "comparison_cv_percent": None,
        "cv_reduction_percent": None,
        "error_type": error_type,
        "error_message": error_message,
    }
    if result is None or "metric_rows" not in result:
        return row

    primary_metric = result["compare_metric"]
    primary_row = next(row_ for row_ in result["metric_rows"] if row_["metric"] == primary_metric)
    row.update(
        {
            "baseline_condition": result["baseline_condition"],
            "comparison_condition": result["comparison_condition"],
            "compare_metric": primary_metric,
            "baseline_std": primary_row["baseline_std"],
            "comparison_std": primary_row["comparison_std"],
            "std_ratio_comparison_over_baseline": primary_row["std_ratio_comparison_over_baseline"],
            "std_reduction_percent": primary_row["std_reduction_percent"],
            "baseline_cv_percent": primary_row["baseline_cv_percent"],
            "comparison_cv_percent": primary_row["comparison_cv_percent"],
            "cv_reduction_percent": primary_row["cv_reduction_percent"],
        }
    )
    return row


def _write_batch_outputs(
    *,
    batch_output_root: Path,
    config_path: Path,
    condition_names: list[str],
    subject_rows: list[dict[str, object]],
    failures: list[dict[str, object]],
) -> None:
    batch_output_root.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "config_path": str(config_path),
        "condition_names": condition_names,
        "subjects_total": len(subject_rows),
        "subjects_succeeded": sum(1 for row in subject_rows if row["status"] == "complete"),
        "subjects_failed": sum(1 for row in subject_rows if row["status"] != "complete"),
        "subjects": subject_rows,
        "failures": failures,
    }
    with (batch_output_root / "paired_condition_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2)
        fh.write("\n")

    if subject_rows:
        fieldnames = [
            "subject",
            "status",
            "analysis_root",
            "baseline_condition",
            "comparison_condition",
            "compare_metric",
            "baseline_std",
            "comparison_std",
            "std_ratio_comparison_over_baseline",
            "std_reduction_percent",
            "baseline_cv_percent",
            "comparison_cv_percent",
            "cv_reduction_percent",
            "error_type",
            "error_message",
        ]
        with (batch_output_root / "paired_condition_summary.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(subject_rows)

    with (batch_output_root / "paired_condition_failures.json").open("w", encoding="utf-8") as fh:
        json.dump(failures, fh, indent=2)
        fh.write("\n")


def analyze_subject(
    *,
    args: argparse.Namespace,
    config,
    subject: str,
    condition_names: list[str],
    roi_name: str,
    roi_labels: list[int],
) -> dict[str, object]:
    per_condition_args = _build_single_condition_args(args, config)
    subject_output_root = (
        Path(args.output_dir).expanduser().resolve() / subject
        if args.output_dir
        else subject_analysis_root(config, subject)
    )
    subject_output_root.mkdir(parents=True, exist_ok=True)

    condition_results: dict[str, dict[str, object]] = {}
    condition_summary_rows: dict[str, list[dict[str, object]]] = {}
    for condition_name in condition_names:
        repeats_root = subject_condition_repeats_root(config, subject, condition_name)
        if not repeats_root.is_dir():
            raise FileNotFoundError(
                f"Condition repeats directory is missing for {subject} / {condition_name}: {repeats_root}"
            )
        output_dir = subject_output_root / condition_name
        result = base_report._run_subject_analysis(
            per_condition_args,
            subject=subject,
            roi_name=roi_name,
            roi_labels=roi_labels,
            rootdir_override=config.experiment_root,
            repeats_dir_override=repeats_root,
            output_dir_override=output_dir,
            condition_name=condition_name,
        )
        condition_results[condition_name] = result
        condition_summary_rows[condition_name] = _load_summary_rows(result["summary_csv"])

    comparison_payload: dict[str, object] = {
        "subject": subject,
        "condition_results": condition_results,
    }

    if len(condition_names) >= 2:
        baseline_condition = "remesh" if "remesh" in condition_names else condition_names[0]
        comparison_condition = (
            "fixed_mesh"
            if "fixed_mesh" in condition_names and "fixed_mesh" != baseline_condition
            else next(name for name in condition_names if name != baseline_condition)
        )
        metric_rows = _metric_comparison_rows(
            condition_results,
            baseline_condition=baseline_condition,
            comparison_condition=comparison_condition,
        )
        comparison_payload.update(
            {
                "baseline_condition": baseline_condition,
                "comparison_condition": comparison_condition,
                "compare_metric": per_condition_args.compare_metric,
                "metric_rows": metric_rows,
            }
        )

        with (subject_output_root / "condition_comparison.json").open("w", encoding="utf-8") as fh:
            json.dump(comparison_payload, fh, indent=2)
            fh.write("\n")

        _write_condition_comparison_report(
            path=subject_output_root / "condition_comparison.md",
            subject=subject,
            condition_results=condition_results,
            metric_rows=metric_rows,
            baseline_condition=baseline_condition,
            comparison_condition=comparison_condition,
            compare_metric=per_condition_args.compare_metric,
        )

        _plot_condition_metric_bars(
            out_path=subject_output_root / "condition_metric_std_comparison.png",
            metric_rows=metric_rows,
            ylabel="Across-repeat SD",
            value_key_baseline="baseline_std",
            value_key_comparison="comparison_std",
            baseline_condition=baseline_condition,
            comparison_condition=comparison_condition,
        )
        _plot_condition_metric_bars(
            out_path=subject_output_root / "condition_metric_cv_comparison.png",
            metric_rows=metric_rows,
            ylabel="Across-repeat CV (%)",
            value_key_baseline="baseline_cv_percent",
            value_key_comparison="comparison_cv_percent",
            baseline_condition=baseline_condition,
            comparison_condition=comparison_condition,
        )
        _plot_condition_lines(
            out_path=subject_output_root / "median_roi_by_condition.png",
            condition_rows=condition_summary_rows,
            metric="median_roi",
            ylabel="Median TI in ROI (V/m)",
        )
        _plot_condition_lines(
            out_path=subject_output_root / "mesh_nodes_by_condition.png",
            condition_rows=condition_summary_rows,
            metric="mesh_nodes",
            ylabel="Mesh nodes",
        )

    return comparison_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired-condition analysis for the remesh vs fixed-mesh repeatability experiment."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--subject", default=None, help="Single subject ID from the experiment config.")
    target_group.add_argument("--all-subjects", action="store_true", help="Analyze every subject listed in the experiment config.")

    parser.add_argument("--config", required=True, help="Path to the paired experiment JSON config.")
    parser.add_argument("--conditions", default=None, help="Optional comma-separated subset of condition names to analyze.")
    parser.add_argument("--max-subjects", type=int, default=None, help="Optional cap in --all-subjects mode.")
    parser.add_argument("--output-dir", default=None, help="Optional override for the batch analysis output root.")

    parser.add_argument("--roi-preset", default=None, choices=sorted(base_report.ROI_PRESET_CHOICES))
    parser.add_argument("--roi-name", default=None)
    parser.add_argument("--roi-labels", default=None)
    parser.add_argument("--m1-labels", default=None)
    parser.add_argument("--atlas-dir", default=None)

    parser.add_argument("--reference-repeat", default=None)
    parser.add_argument("--spatial-percentile", type=float, default=99.0)
    parser.add_argument("--compare-metric", default=None, choices=["median_roi", "mean_roi", "peak_roi"])
    parser.add_argument("--compare-cohort-root", default=None)
    parser.add_argument("--cohort-region-name", default=None)
    parser.add_argument("--cohort-region-label", type=int, default=None)
    parser.add_argument("--cohort-metric", default=None, choices=["mean", "median", "max", "p95", "std", "cv"])
    parser.add_argument("--skip-cohort", action="store_true", help="Disable cohort comparison even if configured in the JSON config.")
    parser.add_argument("--log-file", default=None, help="Optional JSONL log file.")
    parser.add_argument(
        "--skip-batch-summary",
        action="store_true",
        help="Do not write shared paired_condition_summary outputs. Use for Slurm array subject tasks.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    base_report.LOG_FILE = Path(args.log_file) if args.log_file else None

    roi_name, roi_labels = _resolve_roi_from_config(args, config)
    if args.conditions:
        condition_names = [name.strip() for name in args.conditions.split(",") if name.strip()]
    else:
        condition_names = [condition.name for condition in config.conditions]
    if not condition_names:
        raise SystemExit("No conditions were selected for analysis.")
    if len(condition_names) > 2:
        raise SystemExit(
            "The paired repeatability report supports at most two conditions at a time. "
            "Select a two-condition subset with --conditions."
        )

    if args.subject:
        subjects = [args.subject.strip()]
    else:
        subjects = config.subjects[: args.max_subjects] if args.max_subjects is not None else config.subjects

    batch_output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else config.experiment_root / "_analysis"
    )
    batch_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for subject in subjects:
        subject_output_root = (
            batch_output_root / subject
            if args.output_dir
            else subject_analysis_root(config, subject)
        )
        try:
            result = analyze_subject(
                args=args,
                config=config,
                subject=subject,
                condition_names=condition_names,
                roi_name=roi_name,
                roi_labels=roi_labels,
            )
            batch_rows.append(
                _subject_summary_row(
                    subject=subject,
                    subject_output_root=subject_output_root,
                    result=result,
                )
            )
            base_report.log_event("subject_done", subject=subject, output_dir=str(subject_output_root))
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            error_type = type(exc).__name__
            error_message = str(exc)
            failure = {
                "subject": subject,
                "analysis_root": str(subject_output_root),
                "condition_names": condition_names,
                "error_type": error_type,
                "error_message": error_message,
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            batch_rows.append(
                _subject_summary_row(
                    subject=subject,
                    subject_output_root=subject_output_root,
                    error_type=error_type,
                    error_message=error_message,
                )
            )
            base_report.log_event(
                "subject_error",
                subject=subject,
                error_type=error_type,
                error_message=error_message,
                output_dir=str(subject_output_root),
            )

    if not args.skip_batch_summary:
        _write_batch_outputs(
            batch_output_root=batch_output_root,
            config_path=config.config_path,
            condition_names=condition_names,
            subject_rows=batch_rows,
            failures=failures,
        )

    base_report.log_event(
        "batch_done",
        output_dir=str(batch_output_root),
        subjects_total=len(batch_rows),
        subjects_succeeded=sum(1 for row in batch_rows if row["status"] == "complete"),
        subjects_failed=len(failures),
    )

    if not any(row["status"] == "complete" for row in batch_rows):
        raise SystemExit("Post-processing did not complete successfully for any subject.")

    if failures:
        print(
            json.dumps(
                {
                    "status": "partial_success",
                    "subjects_total": len(batch_rows),
                    "subjects_succeeded": sum(1 for row in batch_rows if row["status"] == "complete"),
                    "subjects_failed": len(failures),
                    "summary_json": str(batch_output_root / "paired_condition_summary.json"),
                    "failures_json": str(batch_output_root / "paired_condition_failures.json"),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
