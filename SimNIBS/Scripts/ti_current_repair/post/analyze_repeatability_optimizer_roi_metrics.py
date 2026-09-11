#!/usr/bin/env python3
"""Analyse optimizer-matched ROI metrics for the repeatability manuscript.

The script validates the two read-only extraction tables, audits finite ROI
support, computes subject and target summaries, quantifies single-run ranking
uncertainty, and writes the target-field and ranking figures whose values depend
on the optimizer-matched ROI definition. Repeat-count convergence is generated
separately with the original manuscript analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


TARGETS = {
    "left_hippocampus": {
        "label": "Left hippocampus",
        "plot_label": "left hippocampus",
        "short_label": "Hippocampus",
        "roi": "Left_Hippocampus",
        "requested_volume_mm3": 200.0,
    },
    "right_m1": {
        "label": "Right M1",
        "plot_label": "right M1",
        "short_label": "M1",
        "roi": "Right_M1",
        "requested_volume_mm3": 100.0,
    },
}

REQUIRED_COLUMNS = {
    "schema_version",
    "subject",
    "condition",
    "repeat_tag",
    "roi",
    "roi_voxels",
    "requested_roi_volume_mm3",
    "achieved_roi_volume_mm3",
    "roi_radius_mm",
    "roi_median_v_per_m",
    "finite_roi_voxels",
    "nonfinite_roi_voxels",
    "finite_roi_fraction",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate(frame: pd.DataFrame, target: str) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{target}: missing columns {sorted(missing)}")
    meta = TARGETS[target]
    if len(frame) != 800:
        raise RuntimeError(f"{target}: expected 800 rows, observed {len(frame)}")
    if set(frame["schema_version"].astype(int)) != {2}:
        raise RuntimeError(f"{target}: expected schema version 2")
    if set(frame["roi"]) != {meta["roi"]}:
        raise RuntimeError(f"{target}: unexpected ROI values")
    if set(frame["condition"]) != {"remesh", "fixed_mesh"}:
        raise RuntimeError(f"{target}: unexpected conditions")
    if frame["subject"].nunique() != 10:
        raise RuntimeError(f"{target}: expected ten subjects")
    requested = frame["requested_roi_volume_mm3"].to_numpy(dtype=float)
    if not np.allclose(requested, meta["requested_volume_mm3"]):
        raise RuntimeError(f"{target}: unexpected requested ROI volume")
    if (frame["achieved_roi_volume_mm3"] + 1e-9 < requested).any():
        raise RuntimeError(f"{target}: achieved ROI volume is below its target")
    per_subject_support = frame.groupby("subject").agg(
        roi_voxels=("roi_voxels", "nunique"),
        achieved_volumes=("achieved_roi_volume_mm3", "nunique"),
    )
    if not (per_subject_support == 1).all().all():
        raise RuntimeError(f"{target}: ROI support changes within a subject")
    counts = frame.groupby(["subject", "condition"]).size()
    if set(counts.astype(int)) != {40}:
        raise RuntimeError(f"{target}: each subject-condition must contain 40 runs")
    expected_tags = {f"repeat_{index:03d}" for index in range(1, 41)}
    observed_tags = frame.groupby(["subject", "condition"])["repeat_tag"].agg(set)
    if not all(tags == expected_tags for tags in observed_tags):
        raise RuntimeError(f"{target}: repeat tags do not cover repeat_001..repeat_040")
    keys = frame[["subject", "condition", "repeat_tag"]]
    if keys.duplicated().any():
        raise RuntimeError(f"{target}: duplicate subject-condition-run keys")
    values = frame["roi_median_v_per_m"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{target}: non-finite scalar ROI summaries")
    if (frame["finite_roi_voxels"].astype(int) <= 0).any():
        raise RuntimeError(f"{target}: a run has no finite ROI support")
    support_sum = (
        frame["finite_roi_voxels"].astype(int)
        + frame["nonfinite_roi_voxels"].astype(int)
    )
    if not (support_sum == frame["roi_voxels"].astype(int)).all():
        raise RuntimeError(f"{target}: inconsistent finite-support counts")
    expected_fraction = (
        frame["finite_roi_voxels"].to_numpy(dtype=float)
        / frame["roi_voxels"].to_numpy(dtype=float)
    )
    if not np.allclose(
        frame["finite_roi_fraction"].to_numpy(dtype=float),
        expected_fraction,
        atol=5e-4,
    ):
        raise RuntimeError(f"{target}: inconsistent finite-support fractions")


def _load(left_csv: Path, right_csv: Path) -> pd.DataFrame:
    frames = []
    for target, path in (
        ("left_hippocampus", left_csv),
        ("right_m1", right_csv),
    ):
        frame = pd.read_csv(path)
        _validate(frame, target)
        frame.insert(0, "target", target)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan, math.nan
    return float(pearsonr(x, y)[0]), float(spearmanr(x, y)[0])


def _geometry_summary(frame: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {}
    for target, group in frame.groupby("target", sort=False):
        summary[target] = {
            "requested_volume_mm3": float(
                group["requested_roi_volume_mm3"].iloc[0]
            ),
            "minimum_achieved_volume_mm3": float(
                group["achieved_roi_volume_mm3"].min()
            ),
            "maximum_achieved_volume_mm3": float(
                group["achieved_roi_volume_mm3"].max()
            ),
            "minimum_radius_mm": float(group["roi_radius_mm"].min()),
            "maximum_radius_mm": float(group["roi_radius_mm"].max()),
            "minimum_voxels": int(group["roi_voxels"].min()),
            "maximum_voxels": int(group["roi_voxels"].max()),
        }
    return summary


def _support_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    group_rows: list[dict[str, object]] = []
    target_summary: dict[str, object] = {}
    for target, target_frame in frame.groupby("target", sort=False):
        centered_missing: list[float] = []
        centered_field: list[float] = []
        for (subject, condition), group in target_frame.groupby(
            ["subject", "condition"], sort=False
        ):
            missing = group["nonfinite_roi_voxels"].to_numpy(dtype=float)
            field = group["roi_median_v_per_m"].to_numpy(dtype=float)
            pearson, spearman = _safe_correlation(missing, field)
            centered_missing.extend((missing - missing.mean()).tolist())
            centered_field.extend((field - field.mean()).tolist())
            zero = field[missing == 0]
            affected = field[missing > 0]
            delta = (
                float(affected.mean() - zero.mean())
                if len(zero) and len(affected)
                else math.nan
            )
            delta_percent = (
                100.0 * delta / float(field.mean())
                if math.isfinite(delta) and field.mean() != 0
                else math.nan
            )
            group_rows.append(
                {
                    "target": target,
                    "subject": subject,
                    "condition": condition,
                    "runs": len(group),
                    "runs_with_nonfinite_support": int((missing > 0).sum()),
                    "minimum_finite_fraction": float(
                        group["finite_roi_fraction"].min()
                    ),
                    "maximum_nonfinite_voxels": int(missing.max()),
                    "unique_nonfinite_counts": int(len(np.unique(missing))),
                    "pearson_missing_vs_median": pearson,
                    "spearman_missing_vs_median": spearman,
                    "affected_minus_complete_mean_v_per_m": delta,
                    "affected_minus_complete_percent": delta_percent,
                }
            )
        centered_missing_array = np.asarray(centered_missing, dtype=float)
        centered_field_array = np.asarray(centered_field, dtype=float)
        within_pearson, within_spearman = _safe_correlation(
            centered_missing_array, centered_field_array
        )
        condition_values = {}
        for condition, condition_frame in target_frame.groupby("condition"):
            condition_values[condition] = {
                "runs": int(len(condition_frame)),
                "runs_with_nonfinite_support": int(
                    (condition_frame["nonfinite_roi_voxels"] > 0).sum()
                ),
                "maximum_nonfinite_voxels": int(
                    condition_frame["nonfinite_roi_voxels"].max()
                ),
                "minimum_finite_fraction": float(
                    condition_frame["finite_roi_fraction"].min()
                ),
            }
        target_summary[target] = {
            "minimum_roi_voxels": int(target_frame["roi_voxels"].min()),
            "maximum_roi_voxels": int(target_frame["roi_voxels"].max()),
            "runs": int(len(target_frame)),
            "runs_with_nonfinite_support": int(
                (target_frame["nonfinite_roi_voxels"] > 0).sum()
            ),
            "maximum_nonfinite_voxels": int(
                target_frame["nonfinite_roi_voxels"].max()
            ),
            "minimum_finite_fraction": float(
                target_frame["finite_roi_fraction"].min()
            ),
            "within_subject_condition_pearson": within_pearson,
            "within_subject_condition_spearman": within_spearman,
            "by_condition": condition_values,
        }
    group_frame = pd.DataFrame(group_rows)
    finite_deltas = group_frame["affected_minus_complete_percent"].dropna()
    summary = {
        "targets": target_summary,
        "groups_with_both_complete_and_affected_runs": int(len(finite_deltas)),
        "largest_absolute_affected_complete_difference_percent": (
            float(finite_deltas.abs().max()) if len(finite_deltas) else None
        ),
        "interpretation": (
            "Finite-support counts were audited within each subject and condition. "
            "Correlations are descriptive sensitivity checks and do not establish "
            "that missing voxels cause field changes."
        ),
    }
    return group_frame, summary


def _subject_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (target, subject, condition), group in frame.groupby(
        ["target", "subject", "condition"], sort=False
    ):
        values = group["roi_median_v_per_m"].to_numpy(dtype=float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        rows.append(
            {
                "target": target,
                "subject": subject,
                "condition": condition,
                "runs": len(values),
                "mean_v_per_m": mean,
                "sd_v_per_m": sd,
                "cv_percent": 100.0 * sd / mean,
                "minimum_v_per_m": float(values.min()),
                "maximum_v_per_m": float(values.max()),
                "range_v_per_m": float(values.max() - values.min()),
                "runs_below_0p2_v_per_m": int((values < 0.2).sum()),
                "runs_at_or_above_0p2_v_per_m": int((values >= 0.2).sum()),
            }
        )
    result = pd.DataFrame(rows)
    wide = result.pivot(index=["target", "subject"], columns="condition")
    reductions = 100.0 * (
        1.0
        - wide["sd_v_per_m"]["fixed_mesh"]
        / wide["sd_v_per_m"]["remesh"]
    )
    reduction_map = reductions.to_dict()
    result["sd_reduction_percent"] = [
        reduction_map[(row.target, row.subject)]
        if row.condition == "fixed_mesh"
        else math.nan
        for row in result.itertuples()
    ]
    return result.sort_values(["target", "subject", "condition"])


def _rank_analysis(
    frame: pd.DataFrame,
    *,
    seed: int,
    draws: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    pair_rows: list[dict[str, object]] = []
    draw_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    for target, target_frame in frame[frame["condition"] == "remesh"].groupby(
        "target", sort=False
    ):
        groups = {
            subject: group.sort_values("repeat_tag")[
                "roi_median_v_per_m"
            ].to_numpy(dtype=float)
            for subject, group in target_frame.groupby("subject")
        }
        subjects = sorted(groups, key=lambda key: groups[key].mean(), reverse=True)
        count = len(subjects)
        pair_count = count * (count - 1) // 2
        for i in range(count):
            for j in range(i + 1, count):
                higher = groups[subjects[i]]
                lower = groups[subjects[j]]
                pair_rows.append(
                    {
                        "target": target,
                        "higher_mean_subject": subjects[i],
                        "lower_mean_subject": subjects[j],
                        "higher_mean_v_per_m": float(higher.mean()),
                        "lower_mean_v_per_m": float(lower.mean()),
                        "mean_difference_v_per_m": float(
                            higher.mean() - lower.mean()
                        ),
                        "single_run_reversal_probability": float(
                            np.mean(higher[:, None] <= lower[None, :])
                        ),
                    }
                )
        rng = np.random.default_rng(seed + list(TARGETS).index(target))
        inversion_counts = np.empty(draws, dtype=int)
        agreements = np.empty(draws, dtype=float)
        for draw_index in range(draws):
            selected = np.asarray(
                [rng.choice(groups[subject]) for subject in subjects], dtype=float
            )
            inversions = sum(
                selected[i] <= selected[j]
                for i in range(count)
                for j in range(i + 1, count)
            )
            inversion_counts[draw_index] = inversions
            agreements[draw_index] = 1.0 - 2.0 * inversions / pair_count
        for agreement, inversions in zip(agreements, inversion_counts):
            draw_rows.append(
                {
                    "target": target,
                    "kendall_rank_agreement": float(agreement),
                    "pairwise_reversals": int(inversions),
                }
            )
        target_pairs = [row for row in pair_rows if row["target"] == target]
        summary[target] = {
            "subjects": subjects,
            "draws": draws,
            "median_kendall_rank_agreement": float(np.median(agreements)),
            "kendall_q1": float(np.percentile(agreements, 25)),
            "kendall_q3": float(np.percentile(agreements, 75)),
            "probability_any_reversal": float(np.mean(inversion_counts > 0)),
            "mean_pairwise_reversals": float(inversion_counts.mean()),
            "maximum_pairwise_reversal_probability": float(
                max(row["single_run_reversal_probability"] for row in target_pairs)
            ),
        }
    return pd.DataFrame(pair_rows), pd.DataFrame(draw_rows), summary


def _plot_primary(frame: pd.DataFrame, target: str, path: Path) -> None:
    subset = frame[frame["target"] == target]
    remesh_means = (
        subset[subset["condition"] == "remesh"]
        .groupby("subject")["roi_median_v_per_m"]
        .mean()
        .sort_values(ascending=False)
    )
    subjects = remesh_means.index.tolist()
    colors = {"remesh": "#8ecae6", "fixed_mesh": "#f4a261"}
    edges = {"remesh": "#125d84", "fixed_mesh": "#a94700"}
    offsets = {"remesh": -0.12, "fixed_mesh": 0.12}
    markers = {"remesh": "o", "fixed_mesh": "D"}
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(11.8, 5.7))
    for index, subject in enumerate(subjects):
        for condition in ("remesh", "fixed_mesh"):
            values = subset[
                (subset["subject"] == subject)
                & (subset["condition"] == condition)
            ]["roi_median_v_per_m"].to_numpy(dtype=float)
            center = index + offsets[condition]
            jitter = rng.uniform(-0.045, 0.045, len(values))
            ax.scatter(
                center + jitter,
                values,
                s=20,
                alpha=0.40,
                color=colors[condition],
                edgecolors="none",
                label=(
                    "Remesh runs" if index == 0 and condition == "remesh" else
                    "Fixed-mesh runs" if index == 0 else None
                ),
                zorder=2,
            )
            ax.errorbar(
                center,
                values.mean(),
                yerr=values.std(ddof=1),
                fmt=markers[condition],
                color=edges[condition],
                markerfacecolor="white",
                markeredgewidth=2.0,
                markersize=8 if condition == "remesh" else 6,
                capsize=4,
                linewidth=2,
                label=(
                    "Remesh mean ± SD"
                    if index == 0 and condition == "remesh"
                    else "Fixed-mesh mean ± SD"
                    if index == 0
                    else None
                ),
                zorder=4,
            )
    ax.set_title(
        f"Median TI field in the {TARGETS[target]['plot_label']} ROI",
        pad=16,
    )
    ax.set_ylabel("Median TI field (V/m)")
    ax.set_xlabel("Subject IDs (ordered by decreasing field mean)")
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(
        [subject.removeprefix("sub-CC") for subject in subjects],
        rotation=35,
        ha="right",
    )
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="best", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_rank(
    frame: pd.DataFrame,
    pair_frame: pd.DataFrame,
    draw_frame: pd.DataFrame,
    target: str,
    path: Path,
) -> None:
    remesh = frame[(frame["target"] == target) & (frame["condition"] == "remesh")]
    means = remesh.groupby("subject")["roi_median_v_per_m"].mean()
    subjects = means.sort_values(ascending=False).index.tolist()
    matrix = np.full((len(subjects), len(subjects)), np.nan)
    target_pairs = pair_frame[pair_frame["target"] == target]
    for row in target_pairs.itertuples():
        i = subjects.index(row.higher_mean_subject)
        j = subjects.index(row.lower_mean_subject)
        matrix[i, j] = row.single_run_reversal_probability
    agreements = draw_frame[draw_frame["target"] == target][
        "kendall_rank_agreement"
    ].to_numpy(dtype=float)
    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1.1, 1.0]}
    )
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")
    image = axes[0].imshow(matrix, cmap=cmap, vmin=0, vmax=0.6)
    labels = [subject.removeprefix("sub-CC") for subject in subjects]
    axes[0].set_xticks(range(len(subjects)), labels, rotation=42, ha="right")
    axes[0].set_yticks(range(len(subjects)), labels)
    axes[0].set_xlabel("Lower mean-ranked subject")
    axes[0].set_ylabel("Higher mean-ranked subject")
    axes[0].set_title("A  Pairwise rank-reversal probability", loc="left")
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.047, pad=0.04)
    colorbar.set_label("Probability")
    lower = max(-1.0, float(agreements.min()) - 0.04)
    axes[1].hist(
        agreements,
        bins=np.linspace(lower, 1.005, 18),
        weights=np.full(len(agreements), 100.0 / len(agreements)),
        color="#1f77b4",
        alpha=0.78,
        edgecolor="white",
    )
    median = float(np.median(agreements))
    axes[1].axvline(median, color="#14527a", lw=2.2, label=f"Median = {median:.2f}")
    axes[1].set_xlim(lower, 1.005)
    axes[1].set_xlabel("Kendall's tau relative to 40-run mean ordering")
    axes[1].set_ylabel("Random single-run selections (%)")
    axes[1].set_title("B  Whole-cohort ordering uncertainty", loc="left")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False)
    fig.suptitle(
        f"{TARGETS[target]['label']}: uncertainty from selecting one remesh run",
        fontsize=16,
        y=1.01,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _artifact_manifest(out_dir: Path) -> list[dict[str, object]]:
    artifacts = []
    for path in sorted(item for item in out_dir.rglob("*") if item.is_file()):
        if path.name == "analysis_manifest.json":
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(out_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return artifacts


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    left_csv = args.left_csv.resolve()
    right_csv = args.right_csv.resolve()
    frame = _load(left_csv, right_csv)
    frame.to_csv(out_dir / "combined_optimizer_roi_metrics.csv", index=False)

    geometry_summary = _geometry_summary(frame)
    _write_json(out_dir / "roi_geometry_summary.json", geometry_summary)

    support_groups, support_summary = _support_audit(frame)
    support_groups.to_csv(out_dir / "finite_support_group_audit.csv", index=False)
    _write_json(out_dir / "finite_support_audit.json", support_summary)

    subject_summary = _subject_summary(frame)
    subject_summary.to_csv(out_dir / "subject_condition_summary.csv", index=False)

    pairs, draws, rank_summary = _rank_analysis(
        frame, seed=args.seed, draws=args.rank_draws
    )
    pairs.to_csv(out_dir / "pairwise_rank_reversal_probabilities.csv", index=False)
    draws.to_csv(out_dir / "rank_selection_draws.csv", index=False)
    _write_json(out_dir / "rank_uncertainty_summary.json", rank_summary)

    figure_dir = out_dir / "figures"
    for target in TARGETS:
        target_dir = figure_dir / target
        _plot_primary(
            frame,
            target,
            target_dir / "01_primary_median_roi_repeat_distributions.png",
        )
        _plot_rank(
            frame,
            pairs,
            draws,
            target,
            target_dir / "02_single_repeat_subject_ranking_uncertainty.png",
        )
    result = {
        "status": "complete",
        "schema_version": 2,
        "rows": int(len(frame)),
        "targets": list(TARGETS),
        "inputs": {
            "left_csv": {
                "path": str(left_csv),
                "sha256": _sha256_file(left_csv),
            },
            "right_csv": {
                "path": str(right_csv),
                "sha256": _sha256_file(right_csv),
            },
        },
        "analysis_parameters": {
            "seed": int(args.seed),
            "rank_draws": int(args.rank_draws),
        },
        "roi_geometry": geometry_summary,
        "support_audit": support_summary,
        "rank_uncertainty": rank_summary,
        "outputs": {
            "subject_summary": "subject_condition_summary.csv",
            "geometry_summary": "roi_geometry_summary.json",
            "support_audit": "finite_support_audit.json",
            "rank_summary": "rank_uncertainty_summary.json",
            "figures": "figures",
        },
    }
    result["artifacts"] = _artifact_manifest(out_dir)
    _write_json(out_dir / "analysis_manifest.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-csv", type=Path, required=True)
    parser.add_argument("--right-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--rank-draws", type=int, default=20_000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
