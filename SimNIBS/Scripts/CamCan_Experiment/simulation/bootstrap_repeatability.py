#!/usr/bin/env python3
"""
Bootstrap repeatability curves to estimate how many repeats per subject are needed.

The script expects a repeatability batch laid out like:

    <root>/_analysis/sub-*/summary.csv

or directly:

    <root>/sub-*/summary.csv

Each `summary.csv` must contain one row per repeat and a metric column such as
`peak_roi`. The default workflow bootstraps the subject-level mean of `peak_roi`
and recommends the smallest repeat count `n` that meets a relative precision
target for every subject.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd


EstimatorFn = Callable[[np.ndarray], float]


@dataclass
class SubjectData:
    subject: str
    values: np.ndarray
    reference_value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap repeat-level E-field metrics to estimate the number of "
            "repeats needed per subject."
        )
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help=(
            "Repeatability root. Can point either at the batch root containing "
            "'_analysis/' or directly at the analysis directory itself."
        ),
    )
    parser.add_argument(
        "--metric",
        default="peak_roi",
        help="Column in each summary.csv to bootstrap. Default: peak_roi",
    )
    parser.add_argument(
        "--estimator",
        choices=("mean", "median"),
        default="mean",
        help="Subject-level estimator applied to the repeats. Default: mean",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help=(
            "Relative tolerance against the all-repeat reference estimate. "
            "Default: 0.05 (5%%)"
        ),
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.95,
        help="Coverage used for the bootstrap interval/error quantile. Default: 0.95",
    )
    parser.add_argument(
        "--criterion",
        choices=("ci_half_width", "abs_error_quantile"),
        default="ci_half_width",
        help=(
            "Recommendation rule. 'ci_half_width' uses the central bootstrap "
            "interval half-width. 'abs_error_quantile' uses the coverage-level "
            "quantile of absolute relative error. Default: ci_half_width"
        ),
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=20000,
        help="Bootstrap resamples per subject per repeat count. Default: 20000",
    )
    parser.add_argument(
        "--min-repeats",
        type=int,
        default=2,
        help="Smallest repeat count to evaluate. Default: 2",
    )
    parser.add_argument(
        "--max-repeats",
        type=int,
        default=None,
        help=(
            "Largest repeat count to evaluate. By default uses the smallest "
            "available repeat count across subjects."
        ),
    )
    parser.add_argument(
        "--subject-glob",
        default="sub-*",
        help="Glob for subject folders under the analysis directory. Default: sub-*",
    )
    parser.add_argument(
        "--out-dir",
        default="bootstrap_repeatability_outputs",
        help="Directory for CSV/JSON outputs. Default: bootstrap_repeatability_outputs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible bootstraps. Default: 0",
    )
    return parser.parse_args()


def resolve_analysis_dir(input_root: Path) -> Path:
    input_root = input_root.expanduser().resolve()
    if not input_root.exists():
        raise SystemExit(f"Input path does not exist: {input_root}")

    if (input_root / "_analysis").is_dir():
        return input_root / "_analysis"

    if any((p / "summary.csv").is_file() for p in input_root.glob("sub-*")):
        return input_root

    raise SystemExit(
        "Could not find subject summary files. Expected either "
        "<input>/_analysis/sub-*/summary.csv or <input>/sub-*/summary.csv."
    )


def get_estimator(name: str) -> EstimatorFn:
    if name == "mean":
        return lambda arr: float(np.mean(arr))
    if name == "median":
        return lambda arr: float(np.median(arr))
    raise ValueError(f"Unsupported estimator: {name}")


def load_subject_data(
    analysis_dir: Path,
    subject_glob: str,
    metric: str,
    estimator_name: str,
) -> List[SubjectData]:
    estimator = get_estimator(estimator_name)
    subjects: List[SubjectData] = []

    for subject_dir in sorted(p for p in analysis_dir.glob(subject_glob) if p.is_dir()):
        summary_path = subject_dir / "summary.csv"
        if not summary_path.is_file():
            continue

        df = pd.read_csv(summary_path)
        if metric not in df.columns:
            raise SystemExit(f"Metric '{metric}' not found in {summary_path}")

        values = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue

        reference_value = estimator(values)
        if reference_value == 0.0:
            raise SystemExit(
                f"Reference value is zero for {subject_dir.name}; relative "
                "precision is undefined."
            )

        subjects.append(
            SubjectData(
                subject=subject_dir.name,
                values=values,
                reference_value=reference_value,
            )
        )

    if not subjects:
        raise SystemExit(f"No valid subject summaries found in: {analysis_dir}")

    return subjects


def bootstrap_estimates(
    values: np.ndarray,
    n_repeats: int,
    iterations: int,
    estimator_name: str,
    rng: np.random.Generator,
) -> np.ndarray:
    sample_idx = rng.integers(0, values.size, size=(iterations, n_repeats))
    sampled = values[sample_idx]
    if estimator_name == "mean":
        return sampled.mean(axis=1)
    if estimator_name == "median":
        return np.median(sampled, axis=1)
    raise ValueError(f"Unsupported estimator: {estimator_name}")


def first_passing_repeat_for_threshold(
    df: pd.DataFrame,
    metric_col: str,
    tolerance: float,
) -> int | None:
    passed = df[df[metric_col] <= tolerance]
    if passed.empty:
        return None
    return int(passed["n_repeats"].iloc[0])


def first_true_repeat(df: pd.DataFrame, pass_col: str) -> int | None:
    passed = df[df[pass_col].astype(bool)]
    if passed.empty:
        return None
    return int(passed["n_repeats"].iloc[0])


def write_precision_plot(
    *,
    subject_curve: pd.DataFrame,
    overall: pd.DataFrame,
    metric: str,
    coverage: float,
    tolerance: float,
    ci_recommendation: int | None,
    abs_error_recommendation: int | None,
    out_path: Path,
) -> bool:
    try:
        mpl_config_dir = out_path.parent / ".mplconfig"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is not installed; skipping PNG plot output.")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    panels = [
        (
            axes[0],
            "ci_half_width_rel",
            "worst_ci_half_width_rel",
            ci_recommendation,
            f"Central {coverage:.0%} interval half-width",
        ),
        (
            axes[1],
            "abs_error_quantile_rel",
            "worst_abs_error_quantile_rel",
            abs_error_recommendation,
            f"{coverage:.0%} quantile of absolute relative error",
        ),
    ]

    for axis, subject_col, overall_col, recommendation, title in panels:
        for subject_name, subject_df in subject_curve.groupby("subject", sort=True):
            axis.plot(
                subject_df["n_repeats"],
                subject_df[subject_col] * 100.0,
                linewidth=1.3,
                alpha=0.75,
                label=subject_name,
            )
        axis.plot(
            overall["n_repeats"],
            overall[overall_col] * 100.0,
            color="black",
            linewidth=2.5,
            label="Worst subject",
        )
        axis.axhline(
            tolerance * 100.0,
            color="crimson",
            linestyle="--",
            linewidth=1.4,
            label=f"Tolerance ({tolerance * 100.0:.1f}%)",
        )
        if recommendation is not None:
            axis.axvline(
                recommendation,
                color="dimgray",
                linestyle=":",
                linewidth=1.4,
                label=f"Recommendation ({recommendation})",
            )
        axis.set_title(title)
        axis.set_xlabel("Repeats per subject")
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Relative precision/error (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    seen = {}
    uniq_handles = []
    uniq_labels = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen[label] = True
        uniq_handles.append(handle)
        uniq_labels.append(label)
    fig.legend(uniq_handles, uniq_labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"Bootstrap repeatability curves for {metric}", y=1.02)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    if not (0.0 < args.coverage < 1.0):
        raise SystemExit("--coverage must be between 0 and 1.")
    if args.tolerance <= 0.0:
        raise SystemExit("--tolerance must be > 0.")
    if args.bootstrap_iterations < 100:
        raise SystemExit("--bootstrap-iterations must be at least 100.")

    analysis_dir = resolve_analysis_dir(Path(args.input_root))
    subjects = load_subject_data(
        analysis_dir=analysis_dir,
        subject_glob=args.subject_glob,
        metric=args.metric,
        estimator_name=args.estimator,
    )

    min_available = min(subject.values.size for subject in subjects)
    max_repeats = min_available if args.max_repeats is None else min(args.max_repeats, min_available)
    if args.min_repeats > max_repeats:
        raise SystemExit(
            f"Requested minimum repeats ({args.min_repeats}) exceeds the common "
            f"maximum available repeats ({max_repeats})."
        )

    alpha = (1.0 - args.coverage) / 2.0
    rng = np.random.default_rng(args.seed)
    subject_rows: List[Dict[str, float | int | str | bool]] = []

    for subject in subjects:
        for n_repeats in range(args.min_repeats, max_repeats + 1):
            estimates = bootstrap_estimates(
                values=subject.values,
                n_repeats=n_repeats,
                iterations=args.bootstrap_iterations,
                estimator_name=args.estimator,
                rng=rng,
            )

            ci_lower, ci_upper = np.quantile(estimates, [alpha, 1.0 - alpha])
            abs_error_rel = np.abs(estimates - subject.reference_value) / abs(subject.reference_value)
            ci_half_width_rel = (ci_upper - ci_lower) / (2.0 * abs(subject.reference_value))
            abs_error_quantile_rel = float(np.quantile(abs_error_rel, args.coverage))

            subject_rows.append(
                {
                    "subject": subject.subject,
                    "available_repeats": int(subject.values.size),
                    "n_repeats": n_repeats,
                    "reference_value": subject.reference_value,
                    "bootstrap_mean": float(np.mean(estimates)),
                    "bootstrap_std": float(np.std(estimates, ddof=1)),
                    "ci_lower": float(ci_lower),
                    "ci_upper": float(ci_upper),
                    "ci_half_width_rel": float(ci_half_width_rel),
                    "abs_error_quantile_rel": abs_error_quantile_rel,
                    "pass_ci_half_width": bool(ci_half_width_rel <= args.tolerance),
                    "pass_abs_error_quantile": bool(abs_error_quantile_rel <= args.tolerance),
                }
            )

    subject_curve = pd.DataFrame(subject_rows)
    subject_curve.sort_values(["subject", "n_repeats"], inplace=True)

    overall = (
        subject_curve.groupby("n_repeats", as_index=False)
        .agg(
            subjects=("subject", "nunique"),
            subjects_passing_ci=("pass_ci_half_width", "sum"),
            subjects_passing_abs_error=("pass_abs_error_quantile", "sum"),
            worst_ci_half_width_rel=("ci_half_width_rel", "max"),
            median_ci_half_width_rel=("ci_half_width_rel", "median"),
            worst_abs_error_quantile_rel=("abs_error_quantile_rel", "max"),
            median_abs_error_quantile_rel=("abs_error_quantile_rel", "median"),
        )
    )
    overall["pass_all_ci_half_width"] = overall["subjects_passing_ci"] == overall["subjects"]
    overall["pass_all_abs_error_quantile"] = (
        overall["subjects_passing_abs_error"] == overall["subjects"]
    )

    criterion_to_column = {
        "ci_half_width": "pass_all_ci_half_width",
        "abs_error_quantile": "pass_all_abs_error_quantile",
    }
    configured_recommendation = first_true_repeat(
        overall,
        pass_col=criterion_to_column[args.criterion],
    )
    ci_recommendation = first_true_repeat(
        overall,
        pass_col="pass_all_ci_half_width",
    )
    abs_error_recommendation = first_true_repeat(
        overall,
        pass_col="pass_all_abs_error_quantile",
    )

    per_subject_summary = []
    for subject_name, subject_df in subject_curve.groupby("subject", sort=True):
        per_subject_summary.append(
            {
                "subject": subject_name,
                "available_repeats": int(subject_df["available_repeats"].iloc[0]),
                "reference_value": float(subject_df["reference_value"].iloc[0]),
                "recommendation_ci_half_width": first_passing_repeat_for_threshold(
                    subject_df,
                    metric_col="ci_half_width_rel",
                    tolerance=args.tolerance,
                ),
                "recommendation_abs_error_quantile": first_passing_repeat_for_threshold(
                    subject_df,
                    metric_col="abs_error_quantile_rel",
                    tolerance=args.tolerance,
                ),
            }
        )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    subject_curve_path = out_dir / "per_subject_curve.csv"
    overall_path = out_dir / "overall_curve.csv"
    recommendation_path = out_dir / "recommendation.json"
    plot_path = out_dir / "precision_curves.png"

    subject_curve.to_csv(subject_curve_path, index=False)
    overall.to_csv(overall_path, index=False)

    recommendation = {
        "input_root": str(Path(args.input_root).expanduser().resolve()),
        "analysis_dir": str(analysis_dir),
        "metric": args.metric,
        "estimator": args.estimator,
        "tolerance": args.tolerance,
        "coverage": args.coverage,
        "criterion": args.criterion,
        "bootstrap_iterations": args.bootstrap_iterations,
        "common_max_repeats": int(max_repeats),
        "subjects": len(subjects),
        "configured_recommendation": configured_recommendation,
        "recommendation_ci_half_width": ci_recommendation,
        "recommendation_abs_error_quantile": abs_error_recommendation,
        "per_subject": per_subject_summary,
    }
    recommendation_path.write_text(json.dumps(recommendation, indent=2), encoding="utf-8")
    plot_written = write_precision_plot(
        subject_curve=subject_curve,
        overall=overall,
        metric=args.metric,
        coverage=args.coverage,
        tolerance=args.tolerance,
        ci_recommendation=ci_recommendation,
        abs_error_recommendation=abs_error_recommendation,
        out_path=plot_path,
    )

    print(f"[INFO] Loaded {len(subjects)} subject(s) from {analysis_dir}")
    print(f"[INFO] Metric: {args.metric}")
    print(f"[INFO] Estimator: {args.estimator}")
    print(f"[INFO] Tolerance: {args.tolerance:.2%}")
    print(f"[INFO] Coverage: {args.coverage:.2%}")
    print(f"[INFO] Common max repeats: {max_repeats}")
    print()
    print(
        f"[INFO] Recommended repeats (configured criterion: {args.criterion}): "
        f"{configured_recommendation if configured_recommendation is not None else 'not reached'}"
    )
    print(
        f"[INFO] Recommended repeats (ci_half_width): "
        f"{ci_recommendation if ci_recommendation is not None else 'not reached'}"
    )
    print(
        f"[INFO] Recommended repeats (abs_error_quantile): "
        f"{abs_error_recommendation if abs_error_recommendation is not None else 'not reached'}"
    )
    print()
    for item in per_subject_summary:
        print(
            "[INFO] "
            f"{item['subject']}: "
            f"ci_half_width={item['recommendation_ci_half_width']}, "
            f"abs_error_quantile={item['recommendation_abs_error_quantile']}"
        )
    print()
    print(f"[INFO] Wrote {subject_curve_path}")
    print(f"[INFO] Wrote {overall_path}")
    print(f"[INFO] Wrote {recommendation_path}")
    if plot_written:
        print(f"[INFO] Wrote {plot_path}")


if __name__ == "__main__":
    main()
