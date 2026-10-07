#!/usr/bin/env python3
"""Two-stage Bayesian stability analysis for the repeatability cohort.

Stage 1 performs exact leave-one-participant-out posterior-predictive checks.
Stage 2 fits all participants and simulates sequential cohort expansion.

The two target measurements are modelled jointly.  Each participant contributes
one paired observation, irrespective of the 40 technical repeats used to
estimate each participant-target CV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import invwishart


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = Path(
    "/home/boyan/sandbox/repeatability_paper_analysis_v1/"
    "corrected_fixed_analysis/subject_condition_summary.csv"
)
DEFAULT_OUTPUT = HERE / "outputs"
TARGETS = ("left_hippocampus", "right_m1")
TARGET_LABELS = {
    "left_hippocampus": "Left hippocampus",
    "right_m1": "Right M1",
}
REFERENCE_PRIOR = "reference_weak"


@dataclass(frozen=True)
class PriorSpec:
    """Proper Normal-Inverse-Wishart prior on paired log-CV values."""

    name: str
    description: str
    center_cv_percent: tuple[float, float]
    kappa0: float
    nu0: float
    expected_log_sd: float
    correlation: float = 0.0

    def parameters(self) -> tuple[np.ndarray, float, float, np.ndarray]:
        dimension = len(self.center_cv_percent)
        if self.nu0 <= dimension + 1:
            raise ValueError("nu0 must exceed dimension + 1 for E[Sigma] to exist")
        center = np.log(np.asarray(self.center_cv_percent, dtype=float))
        covariance = np.full((dimension, dimension), self.correlation)
        np.fill_diagonal(covariance, 1.0)
        covariance *= self.expected_log_sd**2
        # E[Sigma] = scale / (nu - dimension - 1).
        scale = covariance * (self.nu0 - dimension - 1)
        return center, self.kappa0, self.nu0, scale


@dataclass(frozen=True)
class NIWPosterior:
    mean: np.ndarray
    kappa: float
    nu: float
    scale: np.ndarray


PRIORS = (
    PriorSpec(
        name=REFERENCE_PRIOR,
        description=(
            "Weak proper prior: 0.05 participant-equivalents on the population "
            "mean, centred at 2.5% CV; broad log-SD 0.75."
        ),
        center_cv_percent=(2.5, 2.5),
        kappa0=0.05,
        nu0=5.0,
        expected_log_sd=0.75,
    ),
    PriorSpec(
        name="diffuse_weak",
        description=(
            "More diffuse weak prior: 0.005 participant-equivalents on the "
            "population mean and log-SD 1.50."
        ),
        center_cv_percent=(2.5, 2.5),
        kappa0=0.005,
        nu0=5.0,
        expected_log_sd=1.50,
    ),
    PriorSpec(
        name="low_variability_challenge",
        description=(
            "Deliberately more informative sensitivity challenge: 0.5 "
            "participant-equivalents centred at 1% CV; log-SD 0.50."
        ),
        center_cv_percent=(1.0, 1.0),
        kappa0=0.5,
        nu0=5.0,
        expected_log_sd=0.50,
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def labelled_seed(base_seed: int, *labels: object) -> int:
    text = "|".join([str(base_seed), *(str(label) for label in labels)])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def load_paired_remesh_cv(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path)
    required = {"subject", "target", "condition", "runs", "cv_percent"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    expected_conditions = {"remesh", "fixed_mesh"}
    if set(raw["condition"]) != expected_conditions:
        raise ValueError(
            f"Expected conditions {sorted(expected_conditions)}, got "
            f"{sorted(set(raw['condition']))}"
        )
    if set(raw["target"]) != set(TARGETS):
        raise ValueError(f"Expected targets {TARGETS}, got {sorted(set(raw['target']))}")
    if not (raw["runs"] == 40).all():
        raise ValueError("Every participant-target-condition row must contain 40 runs")
    if raw.duplicated(["subject", "target", "condition"]).any():
        raise ValueError("Duplicate participant-target-condition rows detected")

    remesh = raw.loc[raw["condition"] == "remesh"].copy()
    paired = remesh.pivot(index="subject", columns="target", values="cv_percent")
    paired = paired.reindex(columns=TARGETS).sort_index()
    if paired.shape != (10, 2) or paired.isna().any().any():
        raise ValueError(
            "The analysis expects 10 participants with both target measurements; "
            f"got shape {paired.shape} and {int(paired.isna().sum().sum())} missing values"
        )
    if not np.isfinite(paired.to_numpy()).all() or (paired.to_numpy() <= 0).any():
        raise ValueError("Remesh CV values must all be finite and strictly positive")
    return paired, raw


def fit_niw(log_values: np.ndarray, prior: PriorSpec) -> NIWPosterior:
    values = np.asarray(log_values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(TARGETS):
        raise ValueError(f"Expected an n x {len(TARGETS)} matrix")
    if values.shape[0] < 2:
        raise ValueError("At least two participants are required")

    m0, kappa0, nu0, scale0 = prior.parameters()
    n = values.shape[0]
    sample_mean = values.mean(axis=0)
    centered = values - sample_mean
    scatter = centered.T @ centered
    kappa_n = kappa0 + n
    nu_n = nu0 + n
    mean_n = (kappa0 * m0 + n * sample_mean) / kappa_n
    mean_delta = (sample_mean - m0).reshape(-1, 1)
    scale_n = (
        scale0
        + scatter
        + (kappa0 * n / kappa_n) * (mean_delta @ mean_delta.T)
    )
    return NIWPosterior(mean=mean_n, kappa=kappa_n, nu=nu_n, scale=scale_n)


def sample_population_parameters(
    posterior: NIWPosterior,
    draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    sigma = invwishart.rvs(
        df=posterior.nu,
        scale=posterior.scale,
        size=draws,
        random_state=rng,
    )
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 2:
        sigma = sigma[np.newaxis, :, :]
    chol = np.linalg.cholesky(sigma)
    noise = rng.standard_normal((draws, len(TARGETS)))
    mu = posterior.mean + np.einsum(
        "nij,nj->ni", chol / np.sqrt(posterior.kappa), noise
    )
    return mu, sigma


def sample_new_participants(
    posterior: NIWPosterior,
    draws: int,
    n_new: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_new < 1:
        raise ValueError("n_new must be positive")
    mu, sigma = sample_population_parameters(posterior, draws, rng)
    chol = np.linalg.cholesky(sigma)
    noise = rng.standard_normal((draws, n_new, len(TARGETS)))
    log_cv = mu[:, np.newaxis, :] + np.einsum("nij,nkj->nki", chol, noise)
    cv = np.exp(log_cv)
    if not np.isfinite(cv).all():
        raise FloatingPointError("Non-finite posterior-predictive CV draw")
    return cv


def quantile_summary(values: np.ndarray) -> dict[str, float]:
    q = np.quantile(values, [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975])
    return {
        "q025": float(q[0]),
        "q05": float(q[1]),
        "q10": float(q[2]),
        "q25": float(q[3]),
        "median": float(q[4]),
        "q75": float(q[5]),
        "q90": float(q[6]),
        "q95": float(q[7]),
        "q975": float(q[8]),
        "mean": float(np.mean(values)),
    }


def run_loo(
    paired: pd.DataFrame,
    priors: Iterable[PriorSpec],
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix = paired.to_numpy(dtype=float)
    log_matrix = np.log(matrix)
    observation_rows: list[dict[str, object]] = []
    median_rows: list[dict[str, object]] = []

    for prior in priors:
        for held_index, subject in enumerate(paired.index):
            keep = np.arange(len(paired)) != held_index
            posterior = fit_niw(log_matrix[keep], prior)
            rng = np.random.default_rng(labelled_seed(seed, "loo", prior.name, subject))
            predicted = sample_new_participants(posterior, draws, 1, rng)[:, 0, :]

            pred_log = np.log(predicted)
            pred_log_mean = pred_log.mean(axis=0)
            pred_log_cov = np.cov(pred_log, rowvar=False)
            inv_cov = np.linalg.inv(pred_log_cov)
            centered_draws = pred_log - pred_log_mean
            draw_distance = np.einsum(
                "ni,ij,nj->n", centered_draws, inv_cov, centered_draws
            )
            held_delta = log_matrix[held_index] - pred_log_mean
            held_distance = float(held_delta @ inv_cov @ held_delta)
            joint_percentile = float(np.mean(draw_distance <= held_distance))

            for target_index, target in enumerate(TARGETS):
                observed = matrix[held_index, target_index]
                predicted_target = predicted[:, target_index]
                summary = quantile_summary(predicted_target)
                observation_rows.append(
                    {
                        "prior": prior.name,
                        "held_out_subject": subject,
                        "target": target,
                        "observed_cv_percent": observed,
                        "predictive_percentile": float(np.mean(predicted_target <= observed)),
                        "joint_predictive_percentile": joint_percentile,
                        "covered_50": summary["q25"] <= observed <= summary["q75"],
                        "covered_80": summary["q10"] <= observed <= summary["q90"],
                        "covered_90": summary["q05"] <= observed <= summary["q95"],
                        "covered_95": summary["q025"] <= observed <= summary["q975"],
                        "absolute_error_from_predictive_median": abs(
                            observed - summary["median"]
                        ),
                        "width_80": summary["q90"] - summary["q10"],
                        "width_95": summary["q975"] - summary["q025"],
                        **{f"predictive_{key}": value for key, value in summary.items()},
                    }
                )

                reconstructed = np.median(
                    np.column_stack(
                        [
                            np.broadcast_to(matrix[keep, target_index], (draws, 9)),
                            predicted_target,
                        ]
                    ),
                    axis=1,
                )
                median_summary = quantile_summary(reconstructed)
                actual_full_median = float(np.median(matrix[:, target_index]))
                median_rows.append(
                    {
                        "prior": prior.name,
                        "held_out_subject": subject,
                        "target": target,
                        "actual_full_cohort_median": actual_full_median,
                        "covered_80": median_summary["q10"]
                        <= actual_full_median
                        <= median_summary["q90"],
                        "covered_95": median_summary["q025"]
                        <= actual_full_median
                        <= median_summary["q975"],
                        "absolute_error_from_predictive_median": abs(
                            actual_full_median - median_summary["median"]
                        ),
                        **{f"predictive_{key}": value for key, value in median_summary.items()},
                    }
                )

    observations = pd.DataFrame(observation_rows)
    medians = pd.DataFrame(median_rows)
    summary_rows: list[dict[str, object]] = []
    for (prior_name, target), group in observations.groupby(["prior", "target"]):
        summary_rows.append(
            {
                "prior": prior_name,
                "target": target,
                "n_folds": len(group),
                "coverage_50": group["covered_50"].mean(),
                "coverage_80": group["covered_80"].mean(),
                "coverage_90": group["covered_90"].mean(),
                "coverage_95": group["covered_95"].mean(),
                "median_absolute_error_cv_percentage_points": group[
                    "absolute_error_from_predictive_median"
                ].median(),
                "median_width_80_cv_percentage_points": group["width_80"].median(),
                "median_width_95_cv_percentage_points": group["width_95"].median(),
                "joint_coverage_95": (
                    group["joint_predictive_percentile"] <= 0.95
                ).mean(),
            }
        )
    return observations, pd.DataFrame(summary_rows), medians


def run_expansion(
    paired: pd.DataFrame,
    priors: Iterable[PriorSpec],
    draws: int,
    seed: int,
    max_total_n: int,
    deltas: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = paired.to_numpy(dtype=float)
    n_observed = observed.shape[0]
    if max_total_n <= n_observed:
        raise ValueError("max_total_n must exceed the observed cohort size")

    summary_rows: list[dict[str, object]] = []
    probability_rows: list[dict[str, object]] = []
    for prior in priors:
        posterior = fit_niw(np.log(observed), prior)
        rng = np.random.default_rng(labelled_seed(seed, "expansion", prior.name))
        predicted = sample_new_participants(
            posterior, draws, max_total_n - n_observed, rng
        )
        observed_draws = np.broadcast_to(
            observed, (draws, n_observed, len(TARGETS))
        )
        expanded = np.concatenate([observed_draws, predicted], axis=1)

        for total_n in range(n_observed, max_total_n + 1):
            if total_n == n_observed:
                cohort_medians = np.broadcast_to(
                    np.median(observed, axis=0), (draws, len(TARGETS))
                )
            else:
                cohort_medians = np.median(expanded[:, :total_n, :], axis=1)
            for target_index, target in enumerate(TARGETS):
                current_median = float(np.median(observed[:, target_index]))
                median_draws = cohort_medians[:, target_index]
                summary = quantile_summary(median_draws)
                summary_rows.append(
                    {
                        "prior": prior.name,
                        "total_cohort_n": total_n,
                        "additional_participants": total_n - n_observed,
                        "target": target,
                        "current_n10_median_cv_percent": current_median,
                        **summary,
                    }
                )
                absolute_shift = np.abs(median_draws - current_median)
                for delta in deltas:
                    probability_rows.append(
                        {
                            "prior": prior.name,
                            "total_cohort_n": total_n,
                            "additional_participants": total_n - n_observed,
                            "target": target,
                            "delta_cv_percentage_points": float(delta),
                            "probability_absolute_shift_exceeds_delta": float(
                                np.mean(absolute_shift > delta)
                            ),
                        }
                    )
    return pd.DataFrame(summary_rows), pd.DataFrame(probability_rows)


def run_descriptive_stability(
    paired: pd.DataFrame, draws: int, seed: int
) -> pd.DataFrame:
    matrix = paired.to_numpy(dtype=float)
    rng = np.random.default_rng(labelled_seed(seed, "participant_bootstrap"))
    indices = rng.integers(0, len(paired), size=(draws, len(paired)))
    boot = np.median(matrix[indices], axis=1)
    rows = []
    for target_index, target in enumerate(TARGETS):
        loo = [
            np.median(np.delete(matrix[:, target_index], index))
            for index in range(len(paired))
        ]
        rows.append(
            {
                "target": target,
                "n_participants": len(paired),
                "minimum_observed_cv_percent": matrix[:, target_index].min(),
                "maximum_observed_cv_percent": matrix[:, target_index].max(),
                "mean_observed_cv_percent": matrix[:, target_index].mean(),
                "median_observed_cv_percent": np.median(matrix[:, target_index]),
                "leave_one_out_median_min": np.min(loo),
                "leave_one_out_median_max": np.max(loo),
                "participant_bootstrap_median_q025": np.quantile(
                    boot[:, target_index], 0.025
                ),
                "participant_bootstrap_median_q975": np.quantile(
                    boot[:, target_index], 0.975
                ),
            }
        )
    return pd.DataFrame(rows)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_loo(observations: pd.DataFrame, output: Path) -> None:
    configure_plotting()
    frame = observations.loc[observations["prior"] == REFERENCE_PRIOR].copy()
    subjects = sorted(frame["held_out_subject"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.5), sharey=True)
    for axis, target in zip(axes, TARGETS):
        target_frame = (
            frame.loc[frame["target"] == target]
            .set_index("held_out_subject")
            .loc[subjects]
            .reset_index()
        )
        y = np.arange(len(subjects))
        axis.hlines(
            y,
            target_frame["predictive_q025"],
            target_frame["predictive_q975"],
            color="#9db7ca",
            linewidth=2.2,
            label="95% predictive interval",
        )
        axis.hlines(
            y,
            target_frame["predictive_q10"],
            target_frame["predictive_q90"],
            color="#216e9e",
            linewidth=6.0,
            label="80% predictive interval",
        )
        axis.scatter(
            target_frame["predictive_median"],
            y,
            marker="|",
            s=90,
            linewidth=2,
            color="#12344d",
            label="Predictive median",
            zorder=3,
        )
        axis.scatter(
            target_frame["observed_cv_percent"],
            y,
            s=34,
            color="#c65b24",
            edgecolor="white",
            linewidth=0.6,
            label="Held-out observed CV",
            zorder=4,
        )
        axis.set_title(TARGET_LABELS[target], fontweight="bold")
        axis.set_xlabel("Remesh CV (%)")
        axis.grid(axis="x", color="#dbe3e9", linewidth=0.7)
        axis.set_yticks(y, subjects)
        axis.invert_yaxis()
    axes[0].set_ylabel("Participant omitted from model fit")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Stage 1 — Leave-one-participant-out posterior-predictive validation",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.018,
        "Each fit uses the other 9 paired participants. Intervals predict a new participant, not a new technical repeat.",
        ha="center",
        color="#4f6473",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.02, 0.16, 0.98, 0.93])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_expansion(summary: pd.DataFrame, output: Path) -> None:
    configure_plotting()
    frame = summary.loc[summary["prior"] == REFERENCE_PRIOR].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), sharey=False)
    for axis, target in zip(axes, TARGETS):
        group = frame.loc[
            (frame["target"] == target) & (frame["total_cohort_n"] % 2 == 0)
        ].sort_values("total_cohort_n")
        x = group["total_cohort_n"].to_numpy()
        axis.fill_between(
            x,
            group["q025"].to_numpy(),
            group["q975"].to_numpy(),
            color="#dbeaf3",
            label="95% predictive interval",
        )
        axis.fill_between(
            x,
            group["q10"].to_numpy(),
            group["q90"].to_numpy(),
            color="#8fc0d9",
            label="80% predictive interval",
        )
        axis.plot(x, group["median"], color="#175f8c", linewidth=2.2, label="Predictive median")
        current = float(group["current_n10_median_cv_percent"].iloc[0])
        axis.axhline(
            current,
            color="#c65b24",
            linewidth=1.8,
            linestyle="--",
            label=f"Observed n=10 median ({current:.2f}%)",
        )
        axis.set_title(TARGET_LABELS[target], fontweight="bold")
        axis.set_xlabel("Total cohort size")
        axis.set_ylabel("Expanded-cohort median remesh CV (%)")
        axis.grid(color="#dbe3e9", linewidth=0.7)
        axis.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        "Stage 2 — Posterior-predictive stability under cohort expansion",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        -0.01,
        "Observed participants are retained; additional paired anatomies are simulated. Even total n shown to avoid the standard median's odd/even plotting artefact.",
        ha="center",
        color="#4f6473",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.93])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_material_change(probabilities: pd.DataFrame, output: Path) -> None:
    configure_plotting()
    frame = probabilities.loc[probabilities["prior"] == REFERENCE_PRIOR].copy()
    selected_deltas = (0.10, 0.25, 0.50)
    colors = ("#2b8c7f", "#d88b2d", "#a63d40")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), sharey=True)
    for axis, target in zip(axes, TARGETS):
        target_frame = frame.loc[
            (frame["target"] == target) & (frame["total_cohort_n"] % 2 == 0)
        ]
        for delta, color in zip(selected_deltas, colors):
            group = target_frame.loc[
                np.isclose(target_frame["delta_cv_percentage_points"], delta)
            ].sort_values("total_cohort_n")
            axis.plot(
                group["total_cohort_n"],
                100 * group["probability_absolute_shift_exceeds_delta"],
                linewidth=2.2,
                color=color,
                label=f"Δ = {delta:.2f} percentage points",
            )
        axis.set_title(TARGET_LABELS[target], fontweight="bold")
        axis.set_xlabel("Total cohort size")
        axis.set_ylim(-1, 101)
        axis.set_ylabel("Posterior probability shift exceeds Δ (%)")
        axis.grid(color="#dbe3e9", linewidth=0.7)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Probability that adding participants materially changes the cohort median",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        -0.01,
        "Δ must be chosen scientifically. Even total n shown; every integer n is retained in the CSV output.",
        ha="center",
        color="#4f6473",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.93])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def format_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(
    paired: pd.DataFrame,
    raw: pd.DataFrame,
    loo_summary: pd.DataFrame,
    expansion_summary: pd.DataFrame,
    probabilities: pd.DataFrame,
    descriptive: pd.DataFrame,
    input_path: Path,
    draws: int,
    seed: int,
    max_total_n: int,
) -> str:
    ref_loo = loo_summary.loc[loo_summary["prior"] == REFERENCE_PRIOR]
    ref_expansion = expansion_summary.loc[
        expansion_summary["prior"] == REFERENCE_PRIOR
    ]
    ref_prob = probabilities.loc[probabilities["prior"] == REFERENCE_PRIOR]

    descriptive_rows = []
    loo_rows = []
    expansion_rows = []
    probability_rows = []
    for target in TARGETS:
        desc = descriptive.loc[descriptive["target"] == target].iloc[0]
        descriptive_rows.append(
            [
                TARGET_LABELS[target],
                f"{desc['median_observed_cv_percent']:.2f}%",
                f"{desc['leave_one_out_median_min']:.2f}–{desc['leave_one_out_median_max']:.2f}%",
                (
                    f"{desc['participant_bootstrap_median_q025']:.2f}–"
                    f"{desc['participant_bootstrap_median_q975']:.2f}%"
                ),
            ]
        )
        loo = ref_loo.loc[ref_loo["target"] == target].iloc[0]
        loo_rows.append(
            [
                TARGET_LABELS[target],
                f"{int(round(10 * loo['coverage_80']))}/10",
                f"{int(round(10 * loo['coverage_95']))}/10",
                f"{loo['median_absolute_error_cv_percentage_points']:.2f} pp",
                f"{loo['median_width_80_cv_percentage_points']:.2f} pp",
            ]
        )
        for total_n in dict.fromkeys((15, 20, 30, 40, max_total_n)):
            if total_n > max_total_n:
                continue
            exp = ref_expansion.loc[
                (ref_expansion["target"] == target)
                & (ref_expansion["total_cohort_n"] == total_n)
            ].iloc[0]
            expansion_rows.append(
                [
                    TARGET_LABELS[target],
                    str(total_n),
                    f"{exp['median']:.2f}%",
                    f"{exp['q10']:.2f}–{exp['q90']:.2f}%",
                    f"{exp['q025']:.2f}–{exp['q975']:.2f}%",
                ]
            )
        for total_n in dict.fromkeys((20, 30, 40, max_total_n)):
            if total_n > max_total_n:
                continue
            values = []
            for delta in (0.10, 0.25, 0.50):
                row = ref_prob.loc[
                    (ref_prob["target"] == target)
                    & (ref_prob["total_cohort_n"] == total_n)
                    & np.isclose(ref_prob["delta_cv_percentage_points"], delta)
                ].iloc[0]
                values.append(
                    f"{100 * row['probability_absolute_shift_exceeds_delta']:.1f}%"
                )
            probability_rows.append([TARGET_LABELS[target], str(total_n), *values])

    fixed = raw.loc[raw["condition"] == "fixed_mesh"]
    fixed_max = float(fixed["cv_percent"].max())
    fixed_zeroish = int((fixed["cv_percent"] < 1e-6).sum())
    prior_text = "\n".join(
        f"- `{prior.name}`: {prior.description}" for prior in PRIORS
    )

    return f"""# Two-stage Bayesian stability analysis

## Question addressed

Would the participant-level remeshing-variability summaries plausibly change if
new anatomies were added to the present 10-participant cohort?

This analysis does **not** prove that 10 participants are universally
representative. It tests internal predictive calibration and then quantifies
model-conditional stability under explicit cohort expansion.

## Inferential unit and outcome

- Inferential unit: participant, not technical repeat.
- Outcome: paired participant-level remesh CV (%) for left hippocampus and right
  M1. Each CV was estimated from 40 remeshing repeats.
- The pairing is preserved because both targets come from the same participant.
- Input: `{input_path}`
- Participants: {len(paired)}; remesh rows: {len(paired) * 2}; technical repeats
  per participant-target: 40.
- Fixed mesh is retained as a descriptive numerical-control arm. Of its 20
  participant-target CVs, {fixed_zeroish} are below 0.000001%, and the maximum is
  {fixed_max:.5f}%. A positive continuous log-CV population model is therefore
  inappropriate for that near-degenerate arm.

## Model

For participant *i*, let
`z_i = [log(CV_hippocampus), log(CV_M1)]`. The sampling model is

`z_i ~ MultivariateNormal(mu, Sigma)`.

`mu` describes the population-average paired log-CVs and `Sigma` describes
between-participant variation and cross-target correlation. A proper
Normal-Inverse-Wishart prior gives an exact posterior, so the analysis does not
depend on MCMC convergence. Posterior-predictive draws are exponentiated back to
CV percentage units.

Priors assessed:

{prior_text}

The reference results below use `{REFERENCE_PRIOR}`. Full CSV outputs contain
all three priors.

## Descriptive, assumption-light stability

{format_markdown_table(
    ["Target", "Observed median", "Leave-one-out median range", "Participant bootstrap 95% interval"],
    descriptive_rows,
)}

The bootstrap resamples whole paired participants. It describes uncertainty in
the current cohort median without assuming the Bayesian population model.

## Stage 1: leave-one-participant-out posterior prediction

The model was refitted 10 times. Each time, both targets and all technical
repeats belonging to one participant were withheld. A full predictive
distribution for that participant was generated from the remaining nine.

{format_markdown_table(
    ["Target", "Inside 80% interval", "Inside 95% interval", "Median absolute error", "Median 80% width"],
    loo_rows,
)}

Coverage is necessarily coarse: one participant changes the rate by 10
percentage points. The observed coverage is compatible with internal
calibration, but the predictive intervals are broad and can over-cover. It does
not demonstrate coverage of rare anatomies absent from this cohort.

## Stage 2: posterior-predictive cohort expansion

The model was fitted to all 10 participants. For every posterior draw, up to
{max_total_n - 10} additional paired participants were sampled and appended to
the observed cohort. The statistic recomputed after expansion was the cohort
median remesh CV.

{format_markdown_table(
    ["Target", "Total n", "Predictive median", "80% interval", "95% interval"],
    expansion_rows,
)}

The intervals widen away from n=10 because the observed n=10 median is fixed,
whereas the expanded median depends increasingly on unseen participants and on
uncertainty about the population distribution. This is **not** evidence that a
larger completed cohort would be less precise. It is uncertainty about how far
the present n=10 result could move while that cohort is being expanded.

The next table shows the posterior probability that the expanded-cohort median
differs from the observed n=10 median by more than a candidate tolerance
`Delta`. Units are CV percentage points, not relative percent.

{format_markdown_table(
    ["Target", "Total n", "P(shift > 0.10 pp)", "P(shift > 0.25 pp)", "P(shift > 0.50 pp)"],
    probability_rows,
)}

No single tolerance is declared acceptable here. That threshold should be set
from the scientific or practical consequence of a change before using this
analysis to make an adequacy claim. The complete probability surface is in
`material_change_probability.csv` for `Delta = 0.05` to `1.00` percentage
points.

The prior-sensitivity output is important with only 10 participants. The
diffuse prior produces wider predictive ranges than the reference prior; the
more informative low-variability challenge can shift the predicted population
centre. Agreement of a substantive conclusion across these columns is more
defensible than reliance on one prior alone.

## Defensible interpretation

If held-out observations are reasonably calibrated and the probability of
exceeding a prespecified meaningful `Delta` is low at relevant expanded cohort
sizes, the results support the narrower claim that the **central cohort median
is unlikely to change materially under this fitted population model**.

They do not establish the population tails, rare anatomies, subgroup effects,
or universal representativeness. An independent extension cohort remains the
strongest validation. This analysis also addresses participant-level remesh CV;
it does not re-estimate the 501x nested between-mesh/residual SD ratio, which was
obtained from one fully nested participant-target case.

## Reproducibility

- Posterior draws per fit: {draws:,}
- Random seed: {seed}
- Maximum simulated total n: {max_total_n}
- Exact leave-one-out refits: 10 per prior
- Input SHA-256: `{file_sha256(input_path)}`

Run:

```bash
/home/boyan/anaconda3/envs/simnibs_post/bin/python \\
  {Path(__file__).resolve()}
```
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--draws", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--max-total-n", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.draws < 2_000:
        raise ValueError("Use at least 2,000 draws for stable interval/probability estimates")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paired, raw = load_paired_remesh_cv(args.input)
    deltas = np.round(np.arange(0.05, 1.0001, 0.05), 2)
    loo_observations, loo_summary, loo_medians = run_loo(
        paired, PRIORS, args.draws, args.seed
    )
    expansion_summary, probabilities = run_expansion(
        paired,
        PRIORS,
        args.draws,
        args.seed,
        args.max_total_n,
        deltas,
    )
    descriptive = run_descriptive_stability(paired, args.draws, args.seed)

    paired.reset_index().to_csv(args.output_dir / "paired_subject_remesh_cv.csv", index=False)
    descriptive.to_csv(args.output_dir / "descriptive_stability.csv", index=False)
    loo_observations.to_csv(
        args.output_dir / "loo_posterior_predictive_validation.csv", index=False
    )
    loo_summary.to_csv(args.output_dir / "loo_validation_summary.csv", index=False)
    loo_medians.to_csv(
        args.output_dir / "loo_cohort_median_reconstruction.csv", index=False
    )
    expansion_summary.to_csv(
        args.output_dir / "cohort_expansion_summary.csv", index=False
    )
    probabilities.to_csv(
        args.output_dir / "material_change_probability.csv", index=False
    )
    sensitivity_rows: list[dict[str, object]] = []
    for prior in (prior.name for prior in PRIORS):
        for target in TARGETS:
            for total_n in dict.fromkeys((20, args.max_total_n)):
                if total_n > args.max_total_n:
                    continue
                expansion_row = expansion_summary.loc[
                    (expansion_summary["prior"] == prior)
                    & (expansion_summary["target"] == target)
                    & (expansion_summary["total_cohort_n"] == total_n)
                ].iloc[0]
                row: dict[str, object] = {
                    "prior": prior,
                    "target": target,
                    "total_cohort_n": total_n,
                    "predictive_median_cv_percent": expansion_row["median"],
                    "predictive_q025_cv_percent": expansion_row["q025"],
                    "predictive_q975_cv_percent": expansion_row["q975"],
                }
                for delta in (0.10, 0.25, 0.50):
                    probability_row = probabilities.loc[
                        (probabilities["prior"] == prior)
                        & (probabilities["target"] == target)
                        & (probabilities["total_cohort_n"] == total_n)
                        & np.isclose(
                            probabilities["delta_cv_percentage_points"], delta
                        )
                    ].iloc[0]
                    row[f"probability_shift_gt_{delta:.2f}_pp"] = probability_row[
                        "probability_absolute_shift_exceeds_delta"
                    ]
                sensitivity_rows.append(row)
    pd.DataFrame(sensitivity_rows).to_csv(
        args.output_dir / "prior_sensitivity_key_results.csv", index=False
    )

    plot_loo(
        loo_observations,
        args.output_dir / "01_loo_posterior_predictive_validation.png",
    )
    plot_expansion(
        expansion_summary,
        args.output_dir / "02_cohort_expansion_median_stability.png",
    )
    plot_material_change(
        probabilities,
        args.output_dir / "03_probability_of_material_change.png",
    )

    report = build_report(
        paired,
        raw,
        loo_summary,
        expansion_summary,
        probabilities,
        descriptive,
        args.input,
        args.draws,
        args.seed,
        args.max_total_n,
    )
    (args.output_dir / "bayesian_stability_report.md").write_text(report)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "input": str(args.input.resolve()),
        "input_sha256": file_sha256(args.input),
        "inferential_unit": "participant",
        "outcome": "paired participant-level remesh CV percent",
        "n_participants": len(paired),
        "targets": list(TARGETS),
        "technical_repeats_per_participant_target": 40,
        "draws_per_fit": args.draws,
        "seed": args.seed,
        "max_total_n": args.max_total_n,
        "delta_grid_cv_percentage_points": deltas.tolist(),
        "model": "bivariate lognormal population model with Normal-Inverse-Wishart prior",
        "priors": [asdict(prior) for prior in PRIORS],
        "limitations": [
            "Internal validation cannot establish representation of anatomies absent from the cohort.",
            "Expansion probabilities are conditional on the lognormal population model and prior.",
            "Technical-repeat uncertainty in each estimated CV is conditioned on, not explicitly propagated.",
            "The fixed-mesh arm is descriptive because its CV distribution is near-degenerate at zero.",
        ],
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print(f"Wrote two-stage Bayesian analysis to {args.output_dir}")
    print(f"Participants: {len(paired)}; draws per fit: {args.draws:,}")


if __name__ == "__main__":
    main()
