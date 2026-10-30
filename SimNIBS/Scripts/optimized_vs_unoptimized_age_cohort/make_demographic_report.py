#!/usr/bin/env python3
"""Create demographic plots and a selection report beside downloaded meshes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
COHORT_ROOT = (
    REPO_ROOT
    / "CamCan_Experiment"
    / "cohort_pipeline"
    / "cohorts"
    / "optimized_vs_unoptimized_age_balanced_14"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "optimized_vs_unoptimized_age_cohort"
    / "downloads"
    / "optimized_vs_unoptimized_age_balanced_14"
)
SELECTION_MANIFEST = COHORT_ROOT / "selection_manifest.csv"
COHORT_JSON = COHORT_ROOT / "cohort.json"
FEMALE_COLOR = "#7A5195"
MALE_COLOR = "#2F6B9A"
GRID_COLOR = "#D7DEE8"
TEXT_COLOR = "#172033"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate(output_root: Path) -> tuple[pd.DataFrame, dict, dict]:
    selection = pd.read_csv(SELECTION_MANIFEST)
    cohort = json.loads(COHORT_JSON.read_text(encoding="utf-8"))
    receipt_path = output_root / "download_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    expected_columns = {
        "subject",
        "age_years_precise",
        "recorded_sex",
        "age_band_index",
        "age_band_years",
    }
    missing_columns = expected_columns - set(selection.columns)
    if missing_columns:
        raise ValueError(f"Selection manifest is missing columns: {sorted(missing_columns)}")
    if len(selection) != cohort["expected_subjects"]:
        raise ValueError(
            f"Selection contains {len(selection)} rows; expected {cohort['expected_subjects']}."
        )
    if selection["subject"].duplicated().any():
        raise ValueError("Selection manifest contains duplicate subjects.")
    if set(selection["recorded_sex"]) != {"FEMALE", "MALE"}:
        raise ValueError("Expected exactly the recorded-sex categories FEMALE and MALE.")
    if receipt.get("status") != "complete":
        raise ValueError("Download receipt is not complete.")

    selected_subjects = set(selection["subject"])
    receipt_meshes = {item["subject"]: item for item in receipt.get("meshes", [])}
    if set(receipt_meshes) != selected_subjects:
        raise ValueError("Downloaded subjects do not match the selected demographic cohort.")

    for subject, item in receipt_meshes.items():
        mesh_path = output_root / "meshes" / f"{subject}.msh"
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Downloaded mesh is missing: {mesh_path}")
        if mesh_path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"Downloaded mesh byte count differs for {subject}.")
        if sha256_file(mesh_path) != item["sha256"]:
            raise ValueError(f"Downloaded mesh SHA-256 differs for {subject}.")

    selection = selection.copy()
    selection["age_years_precise"] = pd.to_numeric(
        selection["age_years_precise"], errors="raise"
    )
    selection["age_band_index"] = pd.to_numeric(
        selection["age_band_index"], errors="raise"
    ).astype(int)
    return selection, cohort, receipt


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "axes.edgecolor": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, output_root: Path, stem: str) -> None:
    fig.savefig(output_root / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_root / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_age_distribution(
    selection: pd.DataFrame,
    cohort: dict,
    output_root: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.4))
    bands = cohort["binning"]["bands"]
    for position, band in enumerate(bands):
        ax.axvspan(
            float(band["lower"]),
            float(band["upper"]),
            color="#F2F5F9" if position % 2 == 0 else "#FFFFFF",
            zorder=0,
        )
        ax.axvline(float(band["lower"]), color=GRID_COLOR, linewidth=0.8, zorder=1)
        ax.text(
            float(band["center"]),
            1.52,
            f"Band {band['index']}\n{float(band['lower']):.0f}–{float(band['upper']):.0f}",
            ha="center",
            va="center",
            fontsize=8.5,
            color="#536176",
        )
    ax.axvline(float(bands[-1]["upper"]), color=GRID_COLOR, linewidth=0.8, zorder=1)

    specifications = [
        ("FEMALE", 1.0, FEMALE_COLOR, "o"),
        ("MALE", 0.0, MALE_COLOR, "s"),
    ]
    for sex, y_value, color, marker in specifications:
        rows = selection.loc[selection["recorded_sex"] == sex].sort_values(
            "age_years_precise"
        )
        ax.scatter(
            rows["age_years_precise"],
            [y_value] * len(rows),
            s=90,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        for row in rows.itertuples(index=False):
            label = str(row.subject).removeprefix("sub-")
            vertical_offset = 0.15 if sex == "FEMALE" else -0.15
            vertical_alignment = "bottom" if sex == "FEMALE" else "top"
            ax.text(
                float(row.age_years_precise),
                y_value + vertical_offset,
                f"{label}\n{float(row.age_years_precise):.2f}",
                ha="center",
                va=vertical_alignment,
                fontsize=7.4,
                color=TEXT_COLOR,
                zorder=4,
            )

    ax.set_title("Age distribution of the selected mesh cohort", loc="left", pad=18)
    ax.text(
        0,
        1.02,
        "One recorded-female and one recorded-male participant per prespecified age band",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#536176",
    )
    ax.set_xlabel("Age (years)")
    ax.set_yticks([0, 1], labels=["Recorded male", "Recorded female"])
    ax.set_xlim(float(bands[0]["lower"]) - 1, float(bands[-1]["upper"]) + 1)
    ax.set_ylim(-0.48, 1.72)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.12, right=0.98)
    save_figure(fig, output_root, "age_distribution_by_recorded_sex")


def plot_sex_distribution(selection: pd.DataFrame, output_root: Path) -> None:
    order = ["FEMALE", "MALE"]
    counts = selection["recorded_sex"].value_counts().reindex(order)
    labels = ["Recorded female", "Recorded male"]
    colors = [FEMALE_COLOR, MALE_COLOR]

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    bars = ax.bar(labels, counts.values, color=colors, width=0.58)
    total = int(counts.sum())
    for bar, count in zip(bars, counts.values, strict=True):
        percentage = 100 * int(count) / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.18,
            f"{int(count)} ({percentage:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_title("Recorded-sex distribution", loc="left", pad=18)
    ax.text(
        0,
        1.01,
        f"Selected cohort, n={total}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#536176",
    )
    ax.set_ylabel("Participants (count)")
    ax.set_ylim(0, max(counts.values) + 1.5)
    ax.set_yticks(range(0, max(counts.values) + 2))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.15, right=0.96)
    save_figure(fig, output_root, "recorded_sex_distribution")


def demographic_table(selection: pd.DataFrame) -> str:
    rows = [
        "| Age band (years) | Recorded female | Age | Recorded male | Age |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for _, group in selection.groupby("age_band_index", sort=True):
        female = group.loc[group["recorded_sex"] == "FEMALE"].iloc[0]
        male = group.loc[group["recorded_sex"] == "MALE"].iloc[0]
        rows.append(
            f"| {female['age_band_years']} | `{female['subject']}` | "
            f"{float(female['age_years_precise']):.2f} | `{male['subject']}` | "
            f"{float(male['age_years_precise']):.2f} |"
        )
    return "\n".join(rows)


def write_report(
    selection: pd.DataFrame,
    cohort: dict,
    receipt: dict,
    output_root: Path,
) -> Path:
    validation = cohort["validation"]
    binning = cohort["binning"]
    female_ages = selection.loc[
        selection["recorded_sex"] == "FEMALE", "age_years_precise"
    ]
    male_ages = selection.loc[
        selection["recorded_sex"] == "MALE", "age_years_precise"
    ]
    report_path = output_root / "COHORT_SELECTION_REPORT.md"
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = f"""# Age- and recorded-sex distribution report

Generated: {generated}

## Result

The downloaded mesh cohort contains all {len(selection)} selected participants:
7 recorded-female and 7 recorded-male participants. Ages range from
{selection['age_years_precise'].min():.2f} to
{selection['age_years_precise'].max():.2f} years (mean
{selection['age_years_precise'].mean():.2f}, median
{selection['age_years_precise'].median():.2f}). The mesh receipt reports
{receipt['scope']['local_meshes']} local meshes, and every local byte count and
SHA-256 checksum was revalidated before these outputs were generated.

![Age distribution by recorded sex](age_distribution_by_recorded_sex.png)

![Recorded-sex distribution](recorded_sex_distribution.png)

## How the cohort was selected

1. The sampling frame was the final-132 CamCAN cohort, with precise ages from
   {binning['age_min_years']:.2f} to {binning['age_max_years']:.2f} years and no
   missing age or recorded-sex values.
2. Seven participants previously selected as best or worst simulated TI cases
   were held out. This prevented an outcome-informed primary sample. Those
   seven remain available as a separate extreme-case sensitivity cohort.
3. The Freedman-Diaconis rule was applied to all 132 ages. With an interquartile
   range of {binning['iqr_years']:.2f} years, it proposed a width of
   {binning['fd_target_width_years']:.2f} years and therefore
   {binning['fd_bin_count']} equal-width age bands. Sturges' rule proposed
   {binning['sturges_bin_count']} bands and the square-root rule proposed
   {binning['sqrt_bin_count']}, but those designs could not supply an eligible
   recorded-female participant in every oldest-age stratum after the holdout.
   The bin-width reference is Freedman and Diaconis (1981):
   https://doi.org/10.1007/BF01025868.
4. One recorded-female and one recorded-male participant were selected from
   each of the seven bands. Within each recorded-sex-by-band stratum, selection
   was based on minimum distance to the band midpoint. Ties were resolved by
   lower age and then lexical subject ID. No simulated TI outcome was used.
5. The age bands enforce sampling coverage. They are not seven powered
   inferential groups. The planned primary analysis is the within-participant
   individualized-minus-generic montage contrast with age represented
   continuously. Recorded sex may be included as a prespecified covariate,
   while interactions should remain exploratory unless separately powered.

## Selected participants

{demographic_table(selection)}

Recorded-female ages: range {female_ages.min():.2f}–{female_ages.max():.2f}
years, mean {female_ages.mean():.2f}, median {female_ages.median():.2f}.
Recorded-male ages: range {male_ages.min():.2f}–{male_ages.max():.2f} years,
mean {male_ages.mean():.2f}, median {male_ages.median():.2f}. The maximum
within-band female/male age difference is
{validation['maximum_within_band_pair_age_gap_years']:.2f} years.

## Provenance

- Cohort: `optimized_vs_unoptimized_age_balanced_14`
- Demographic source: `CamCan_Experiment/cohort_pipeline/cohorts/optimized_vs_unoptimized_age_balanced_14/selection_manifest.csv`
- Machine-readable selection method: `CamCan_Experiment/cohort_pipeline/cohorts/optimized_vs_unoptimized_age_balanced_14/cohort.json`
- Mesh receipt: `download_receipt.json`
- Mesh checksums: `SHA256SUMS`
- Downloaded subjects matching demographic selection: {len(selection)}/{len(selection)}
- Overlap with the held-out optimized seven: {validation['optimized_7_overlap']}/{len(selection)}
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Downloaded cohort root containing meshes/ and download_receipt.json.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selection, cohort, receipt = load_and_validate(output_root)
    apply_style()
    plot_age_distribution(selection, cohort, output_root)
    plot_sex_distribution(selection, output_root)
    report_path = write_report(selection, cohort, receipt, output_root)
    print(f"Created: {output_root / 'age_distribution_by_recorded_sex.png'}")
    print(f"Created: {output_root / 'age_distribution_by_recorded_sex.pdf'}")
    print(f"Created: {output_root / 'recorded_sex_distribution.png'}")
    print(f"Created: {output_root / 'recorded_sex_distribution.pdf'}")
    print(f"Created: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
