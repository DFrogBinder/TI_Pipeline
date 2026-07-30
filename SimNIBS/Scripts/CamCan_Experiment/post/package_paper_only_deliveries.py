#!/usr/bin/env python3
"""Build minimal CamCan and repeatability manuscript-material packages.

Each output contains only:

* publication PNG figures;
* figure and table captions in Markdown/CSV form;
* compact manuscript tables as CSV;
* a matching LaTeX representation for every included manuscript table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROI_ORDER = [
    "Left_M1",
    "Right_DLPC",
    "Left_Hippocampus",
    "Right_Thalamus",
]
ROI_LABELS = {
    "Left_M1": "Left M1",
    "Right_DLPC": "Right DLPFC",
    "Left_Hippocampus": "Left hippocampus",
    "Right_Thalamus": "Right thalamus",
}
REPEATABILITY_ROIS = {
    "left_hippocampus": "Left hippocampus",
    "right_m1": "Right M1",
}
CAMCAN_FIGURE_STEMS = {
    "figure_personalization_subject_changes_all_rois_at_mni_roi_threshold",
    "figure_population_mean_field_and_target_offtarget_ratio_absolute",
    "figure_population_mean_field_offtarget_relationship_at_mni_roi_threshold",
    "figure_population_target_offtarget_relationship_at_mni_roi_threshold",
}
REPEATABILITY_FIGURE_STEMS = {
    "01_primary_median_roi_repeat_distributions",
    "02_single_repeat_subject_ranking_uncertainty",
    "03_primary_mesh_element_repeat_distributions",
}


@dataclass(frozen=True)
class TableSpec:
    stem: str
    frame: pd.DataFrame
    caption: str
    note: str
    label: str
    alignment: str
    wide: bool = True


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _latex_escape(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return r"\textemdash{}"
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "–": "--",
        "—": "---",
        "−": "--",
        "∞": r"$\infty$",
        "±": r"$\pm$",
        "ρ": r"$\rho$",
        "²": r"$^2$",
        "³": r"$^3$",
    }
    return "".join(replacements.get(character, character) for character in text)


def _write_latex_table(path: Path, spec: TableSpec) -> None:
    environment = "table*" if spec.wide else "table"
    width = r"\textwidth" if spec.wide else r"\linewidth"
    lines = [
        r"% Requires \usepackage{booktabs} and \usepackage{graphicx}.",
        rf"\begin{{{environment}}}[!htbp]",
        r"\centering",
        rf"\caption{{{_latex_escape(spec.caption)}}}",
        rf"\label{{{spec.label}}}",
        r"\small",
        rf"\resizebox{{{width}}}{{!}}{{%",
        rf"\begin{{tabular}}{{{spec.alignment}}}",
        r"\toprule",
        " & ".join(_latex_escape(column) for column in spec.frame.columns)
        + r" \\",
        r"\midrule",
    ]
    for row in spec.frame.itertuples(index=False, name=None):
        lines.append(
            " & ".join(_latex_escape(value) for value in row) + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\vspace{2pt}",
            rf"\begin{{minipage}}{{{width}}}",
            r"\footnotesize",
            rf"\textit{{Note.}} {_latex_escape(spec.note)}",
            r"\end{minipage}",
            rf"\end{{{environment}}}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_tables(
    root: Path,
    specs: list[TableSpec],
) -> None:
    csv_dir = root / "tables" / "csv"
    latex_dir = root / "tables" / "latex"
    csv_dir.mkdir(parents=True)
    latex_dir.mkdir(parents=True)
    caption_rows = []
    caption_lines = ["# Self-contained table captions", ""]
    for spec in specs:
        csv_path = csv_dir / f"{spec.stem}.csv"
        tex_path = latex_dir / f"{spec.stem}.tex"
        spec.frame.to_csv(csv_path, index=False)
        _write_latex_table(tex_path, spec)
        caption_rows.append(
            {
                "table": spec.stem,
                "caption": spec.caption,
                "note": spec.note,
            }
        )
        caption_lines.extend(
            [
                f"## {spec.stem}",
                "",
                spec.caption,
                "",
                f"**Note.** {spec.note}",
                "",
            ]
        )
    captions_dir = root / "captions"
    captions_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(caption_rows).to_csv(
        captions_dir / "table_captions.csv",
        index=False,
    )
    (captions_dir / "table_captions.md").write_text(
        "\n".join(caption_lines),
        encoding="utf-8",
    )


def _write_figure_caption_files(
    rows: list[dict[str, str]],
    output: Path,
    *,
    title: str,
) -> None:
    captions_dir = output / "captions"
    captions_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        captions_dir / "figure_captions.csv",
        index=False,
    )
    lines = [f"# {title}", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['figure']}",
                "",
                row["caption"],
                "",
            ]
        )
    (captions_dir / "figure_captions.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _copy_camcan_figure_captions(source: Path, output: Path) -> None:
    frame = pd.read_csv(_require(source / "figure_captions.csv"))
    frame = frame.loc[frame["figure"].isin(CAMCAN_FIGURE_STEMS)].copy()
    missing = CAMCAN_FIGURE_STEMS - set(frame["figure"])
    if missing:
        raise RuntimeError(f"Missing CamCan figure captions: {sorted(missing)}")
    rows = frame[["figure", "caption"]].to_dict("records")
    _write_figure_caption_files(
        rows,
        output,
        title="CamCan figure captions",
    )


def _copy_repeatability_figure_captions(
    source: Path,
    output: Path,
) -> None:
    rows: list[dict[str, str]] = []
    for roi_slug, roi_label in REPEATABILITY_ROIS.items():
        caption_csv = _require(source / roi_slug / "figure_captions.csv")
        with caption_csv.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            for row in csv.DictReader(handle):
                if row["figure"] not in REPEATABILITY_FIGURE_STEMS:
                    continue
                figure = f"{roi_slug}/{row['figure']}"
                rows.append(
                    {
                        "roi": roi_label,
                        "figure": figure,
                        "caption": row["caption"],
                    }
                )
    _write_figure_caption_files(
        rows,
        output,
        title="Repeatability figure captions",
    )


def _camcan_threshold_table(source: Path) -> TableSpec:
    frame = pd.read_csv(
        _require(source / "tables" / "table_mni152_roi_thresholds.csv")
    )
    frame = frame.set_index("roi").loc[ROI_ORDER].reset_index()
    result = pd.DataFrame(
        {
            "ROI": frame["roi"].map(ROI_LABELS),
            "Depth": frame["roi_group"].str.title(),
            "Minimum (V/m)": frame["mni_min_v_per_m"].map(
                lambda value: f"{value:.3f}"
            ),
            "Mean/threshold (V/m)": frame["mni_mean_v_per_m"].map(
                lambda value: f"{value:.3f}"
            ),
            "Maximum P99.9 (V/m)": frame[
                "mni_max_p99_9_v_per_m"
            ].map(lambda value: f"{value:.3f}"),
            "SimNIBS": frame["simnibs_version"].astype(str),
        }
    )
    return TableSpec(
        stem="table_1_mni152_roi_field_summary",
        frame=result,
        caption=(
            "MNI152 target-region temporal-interference E-field summaries "
            "and ROI-specific evaluation thresholds."
        ),
        note=(
            "Values were calculated on the fixed MNI152 head model with "
            "SimNIBS 4.0.1. The mean E-field inside each parcel-clipped "
            "target ROI was used as that ROI's threshold for all subsequent "
            "coverage calculations. Maximum denotes the 99.9th percentile "
            "rather than the single-voxel absolute maximum. M1, primary motor "
            "cortex; DLPFC, dorsolateral prefrontal cortex."
        ),
        label="tab:camcan_mni152_roi_fields",
        alignment="llrrrr",
        wide=False,
    )


def _camcan_population_summary(source: Path) -> TableSpec:
    frame = pd.read_csv(
        _require(
            source
            / "tables"
            / "table_population_mean_field_and_ratio.csv"
        )
    )
    rows = []
    outcomes = [
        ("mean_target_field", "Mean target E-field", "V/m", 3),
        (
            "target_to_off_target_coverage_ratio",
            "Target/off-target ratio",
            "Dimensionless",
            2,
        ),
    ]
    for roi in ROI_ORDER:
        for outcome, outcome_label, unit, precision in outcomes:
            group = frame.loc[
                (frame["roi"] == roi) & (frame["outcome"] == outcome)
            ]
            finite = group.loc[
                group["status"] == "finite", "absolute_value"
            ].astype(float)
            if finite.empty:
                raise RuntimeError(f"No finite {roi} {outcome} values")
            mni = group["mni_reference"].dropna().astype(float).unique()
            if len(mni) != 1:
                raise RuntimeError(f"Non-unique MNI reference for {roi}")
            rows.append(
                {
                    "ROI": ROI_LABELS[roi],
                    "Outcome": outcome_label,
                    "Unit": unit,
                    "n": len(group),
                    "CamCan median": f"{finite.median():.{precision}f}",
                    "Q1": f"{finite.quantile(0.25):.{precision}f}",
                    "Q3": f"{finite.quantile(0.75):.{precision}f}",
                    "MNI152": f"{mni[0]:.{precision}f}",
                    "Infinite n": int((group["status"] == "infinite").sum()),
                    "Undefined n": int(
                        (group["status"] == "undefined").sum()
                    ),
                }
            )
    return TableSpec(
        stem="table_2_camcan_population_outcomes",
        frame=pd.DataFrame(rows),
        caption=(
            "CamCan population summaries for mean target E-field and "
            "thresholded target selectivity, with the MNI152 reference."
        ),
        note=(
            "The cohort contains 132 adults. Each subject value is the "
            "arithmetic mean of a metric calculated separately in ten "
            "independently remeshed simulations. Q1 and Q3 are the 25th and "
            "75th percentiles. Selectivity is target coverage divided by "
            "off-target coverage at the ROI-specific MNI152 mean threshold. "
            "Infinite denotes positive target coverage with zero off-target "
            "coverage; undefined denotes 0/0. M1, primary motor cortex; "
            "DLPFC, dorsolateral prefrontal cortex."
        ),
        label="tab:camcan_population_outcomes",
        alignment="lllrrrrrrr",
    )


def _camcan_relationship_statistics(source: Path) -> TableSpec:
    frame = pd.read_csv(
        _require(
            source / "tables" / "table_population_linear_fit_statistics.csv"
        )
    )
    relationship_labels = {
        "mean_field": "Mean field vs off-target coverage",
        "target_coverage": "Target vs off-target coverage",
    }
    frame["_roi_order"] = frame["roi"].map(
        {roi: index for index, roi in enumerate(ROI_ORDER)}
    )
    frame["_relationship_order"] = frame["x_outcome"].map(
        {"mean_field": 0, "target_coverage": 1}
    )
    frame = frame.sort_values(
        ["_relationship_order", "_roi_order"]
    )
    result = pd.DataFrame(
        {
            "Relationship": frame["x_outcome"].map(relationship_labels),
            "ROI": frame["roi"].map(ROI_LABELS),
            "Threshold (V/m)": frame["threshold_v_per_m"].map(
                lambda value: f"{value:.3f}"
            ),
            "n": frame["n"].astype(int),
            "Slope": frame["slope"].map(lambda value: f"{value:.3f}"),
            "Intercept": frame["intercept"].map(
                lambda value: f"{value:.3f}"
            ),
            "R²": frame["r_squared"].map(lambda value: f"{value:.3f}"),
            "Spearman ρ": frame["spearman_rho"].map(
                lambda value: f"{value:.3f}"
            ),
        }
    )
    return TableSpec(
        stem="table_3_population_relationship_statistics",
        frame=result,
        caption=(
            "Descriptive associations between target-field outcomes and "
            "off-target coverage in the CamCan cohort."
        ),
        note=(
            "Each model contains 132 subject-level repeat means. For the "
            "mean-field relationship, slope is the change in off-target "
            "coverage percentage points per 1 V/m increase in mean target "
            "field. For the coverage relationship, slope is the change in "
            "off-target coverage percentage points per one percentage-point "
            "increase in target coverage. R² is the coefficient of "
            "determination from an ordinary least-squares linear fit; "
            "Spearman rho is the rank correlation. These are descriptive "
            "associations and do not imply causality."
        ),
        label="tab:camcan_relationship_statistics",
        alignment="llrrrrrr",
    )


def _paired_direction_counts(
    generic: pd.Series,
    personalized: pd.Series,
) -> tuple[int, int, int]:
    delta = personalized.to_numpy(dtype=float) - generic.to_numpy(dtype=float)
    tolerance = 1e-12
    return (
        int(np.sum(delta > tolerance)),
        int(np.sum(delta < -tolerance)),
        int(np.sum(np.abs(delta) <= tolerance)),
    )


def _camcan_personalization_summary(source: Path) -> TableSpec:
    frame = pd.read_csv(
        _require(
            source / "tables" / "table_personalization_subject_changes.csv"
        )
    )
    panel_labels = {
        "target_coverage": ("Target coverage", "%"),
        "off_target_coverage": ("Off-target coverage", "%"),
        "target_to_off_target_ratio": (
            "Target/off-target ratio",
            "Dimensionless",
        ),
    }
    rows = []
    for roi in ROI_ORDER:
        for panel, (label, unit) in panel_labels.items():
            group = frame.loc[
                (frame["roi"] == roi) & (frame["panel"] == panel)
            ]
            pivot = group.pivot(
                index="subject",
                columns="condition",
                values="absolute_value",
            ).sort_index()
            generic = pivot["generic"].astype(float)
            personalized = pivot["personalized"].astype(float)
            higher, lower, unchanged = _paired_direction_counts(
                generic,
                personalized,
            )
            threshold = group["threshold_v_per_m"].astype(float).unique()
            if len(threshold) != 1:
                raise RuntimeError(f"Non-unique threshold for {roi} {panel}")
            rows.append(
                {
                    "ROI": ROI_LABELS[roi],
                    "Outcome": label,
                    "Unit": unit,
                    "Threshold (V/m)": f"{threshold[0]:.3f}",
                    "n": len(pivot),
                    "Generic median": f"{generic.median():.2f}",
                    "Personalized median": f"{personalized.median():.2f}",
                    "Median paired change": (
                        f"{(personalized - generic).median():+.2f}"
                    ),
                    "Higher n": higher,
                    "Lower n": lower,
                    "Unchanged n": unchanged,
                }
            )
    return TableSpec(
        stem="table_4_personalization_summary",
        frame=pd.DataFrame(rows),
        caption=(
            "Generic-to-personalized changes in target coverage, off-target "
            "coverage, and thresholded selectivity."
        ),
        note=(
            "Seven CamCan subjects were evaluated for each ROI. The generic "
            "condition applies the MNI152-derived montage to the individual "
            "head, whereas the personalized condition applies that subject's "
            "Pareto-optimized montage. Each subject-condition value is the "
            "arithmetic mean of ten independently remeshed simulations. "
            "Median paired change is personalized minus generic. Higher, "
            "lower, and unchanged counts describe the direction of that "
            "within-subject change; whether a higher value is desirable "
            "depends on the outcome (higher target coverage and selectivity "
            "are favourable, whereas lower off-target coverage is "
            "favourable)."
        ),
        label="tab:camcan_personalization",
        alignment="lllrrrrrrrr",
    )


def build_camcan_package(source: Path, output: Path) -> None:
    _clean_dir(output)
    figures_dir = output / "figures"
    figures_dir.mkdir()
    for figure in sorted((source / "figures").glob("*.png")):
        if figure.stem in CAMCAN_FIGURE_STEMS:
            shutil.copy2(figure, figures_dir / figure.name)
    copied = {path.stem for path in figures_dir.glob("*.png")}
    if copied != CAMCAN_FIGURE_STEMS:
        raise RuntimeError(
            "CamCan package figure mismatch: "
            f"missing={sorted(CAMCAN_FIGURE_STEMS - copied)}, "
            f"unexpected={sorted(copied - CAMCAN_FIGURE_STEMS)}"
        )
    _copy_camcan_figure_captions(source, output)
    _write_tables(
        output,
        [
            _camcan_threshold_table(source),
            _camcan_population_summary(source),
            _camcan_relationship_statistics(source),
            _camcan_personalization_summary(source),
        ],
    )


def _repeatability_subject_summary(source: Path) -> TableSpec:
    rows = []
    for roi_slug, roi_label in REPEATABILITY_ROIS.items():
        roi_root = source / roi_slug
        repeats = pd.read_csv(
            _require(roi_root / "presentation_condition_summary.csv")
        )
        repeats["median_roi"] = pd.to_numeric(
            repeats["median_roi"], errors="raise"
        )
        grouped = (
            repeats.groupby(["subject", "condition"])["median_roi"]
            .agg(["mean", "std"])
            .reset_index()
        )
        mean_pivot = grouped.pivot(
            index="subject", columns="condition", values="mean"
        )
        std_pivot = grouped.pivot(
            index="subject", columns="condition", values="std"
        )
        paired = pd.read_csv(
            _require(roi_root / "presentation_paired_condition_summary.csv")
        ).set_index("subject")
        for subject in sorted(mean_pivot.index):
            rows.append(
                {
                    "ROI": roi_label,
                    "Subject": subject.removeprefix("sub-"),
                    "Remesh mean (V/m)": (
                        f"{mean_pivot.loc[subject, 'remesh']:.5f}"
                    ),
                    "Remesh SD (V/m)": (
                        f"{std_pivot.loc[subject, 'remesh']:.5f}"
                    ),
                    "Fixed mean (V/m)": (
                        f"{mean_pivot.loc[subject, 'fixed_mesh']:.5f}"
                    ),
                    "Fixed SD (V/m)": (
                        f"{std_pivot.loc[subject, 'fixed_mesh']:.5f}"
                    ),
                    "SD reduction (%)": (
                        f"{paired.loc[subject, 'std_reduction_percent']:.1f}"
                    ),
                    "Remesh CV (%)": (
                        f"{paired.loc[subject, 'baseline_cv_percent']:.2f}"
                    ),
                    "Fixed CV (%)": (
                        f"{paired.loc[subject, 'comparison_cv_percent']:.2f}"
                    ),
                    "CV reduction (%)": (
                        f"{paired.loc[subject, 'cv_reduction_percent']:.1f}"
                    ),
                }
            )
    return TableSpec(
        stem="table_1_subject_repeatability_summary",
        frame=pd.DataFrame(rows),
        caption=(
            "Subject-level repeatability of the median target-ROI "
            "temporal-interference E-field under remeshed and fixed-mesh "
            "conditions."
        ),
        note=(
            "Each row summarizes 40 remesh and 40 fixed-mesh simulations. In "
            "the remesh condition a new head mesh is generated for every "
            "repeat; in the fixed-mesh condition all repeats reuse one "
            "representative subject-specific mesh. Mean and sample standard "
            "deviation (SD) are calculated across the 40 repeat-level median "
            "target-ROI E-fields. CV is 100 multiplied by SD divided by the "
            "mean. Reductions are calculated as 100 multiplied by "
            "(remesh minus fixed) divided by remesh."
        ),
        label="tab:repeatability_subject_summary",
        alignment="llrrrrrrrr",
    )


def _repeatability_rank_summary(source: Path) -> TableSpec:
    rows = []
    for roi_slug, roi_label in REPEATABILITY_ROIS.items():
        payload = json.loads(
            _require(
                source
                / roi_slug
                / "single_repeat_ranking_uncertainty.json"
            ).read_text(encoding="utf-8")
        )
        repeat_counts = set(payload["repeats_per_subject"].values())
        if len(repeat_counts) != 1:
            raise RuntimeError(f"Unequal repeat counts for {roi_slug}")
        rows.append(
            {
                "ROI": roi_label,
                "Subjects (n)": len(payload["subjects"]),
                "Repeats/subject": repeat_counts.pop(),
                "Median Kendall agreement": (
                    f"{payload['median_kendall_rank_agreement']:.3f}"
                ),
                "Kendall Q1": (
                    f"{payload['kendall_rank_agreement_iqr'][0]:.3f}"
                ),
                "Kendall Q3": (
                    f"{payload['kendall_rank_agreement_iqr'][1]:.3f}"
                ),
                "Any reversal (%)": (
                    f"{100 * payload['probability_of_any_subject_order_reversal']:.1f}"
                ),
                "Mean reversals/draw": (
                    f"{payload['mean_number_of_pairwise_reversals']:.2f}"
                ),
                "Maximum pair probability (%)": (
                    f"{100 * payload['maximum_pairwise_reversal_probability']:.1f}"
                ),
            }
        )
    return TableSpec(
        stem="table_2_single_repeat_ranking_uncertainty",
        frame=pd.DataFrame(rows),
        caption=(
            "Uncertainty in the between-subject ranking produced by selecting "
            "one remesh repeat per subject."
        ),
        note=(
            "For each of 20,000 Monte Carlo draws, one of 40 remesh repeats "
            "was selected independently for every subject and the resulting "
            "ordering was compared with the ordering based on the 40-repeat "
            "means. Kendall agreement is Kendall's rank correlation (+1, "
            "identical ordering; 0, no net concordance; -1, complete "
            "reversal). Any reversal is the percentage of draws containing "
            "at least one reversed subject pair. Maximum pair probability is "
            "the largest exact rank-reversal probability across all subject "
            "pairs and all 40 by 40 repeat combinations. These quantities are "
            "uncertainty measures, not p-values."
        ),
        label="tab:repeatability_rank_uncertainty",
        alignment="lrrrrrrrr",
    )


def _repeatability_top_pairs(source: Path) -> TableSpec:
    rows = []
    for roi_slug, roi_label in REPEATABILITY_ROIS.items():
        frame = pd.read_csv(
            _require(
                source
                / roi_slug
                / "single_repeat_pairwise_rank_reversal_probabilities.csv"
            )
        )
        frame = frame.sort_values(
            "single_repeat_order_reversal_probability",
            ascending=False,
        ).head(5)
        for rank, row in enumerate(frame.itertuples(index=False), start=1):
            rows.append(
                {
                    "ROI": roi_label,
                    "Rank": rank,
                    "Higher-mean subject": str(
                        row.higher_mean_subject
                    ).removeprefix("sub-"),
                    "Higher mean (V/m)": f"{row.higher_mean_v_per_m:.5f}",
                    "Lower-mean subject": str(
                        row.lower_mean_subject
                    ).removeprefix("sub-"),
                    "Lower mean (V/m)": f"{row.lower_mean_v_per_m:.5f}",
                    "Reversal probability (%)": (
                        f"{100 * row.single_repeat_order_reversal_probability:.1f}"
                    ),
                }
            )
    return TableSpec(
        stem="table_3_highest_pairwise_rank_reversal_probabilities",
        frame=pd.DataFrame(rows),
        caption=(
            "Subject pairs with the highest probability of reversing their "
            "mean-based E-field ordering when one remesh repeat is selected."
        ),
        note=(
            "Subjects are ranked by their arithmetic mean median target-ROI "
            "E-field across 40 remesh repeats. For each pair, reversal "
            "probability is the fraction of all 40 by 40 = 1,600 independent "
            "repeat combinations in which the lower-mean subject equals or "
            "exceeds the higher-mean subject. The five most unstable pairs "
            "are reported for each ROI. Probabilities quantify repeat-choice "
            "uncertainty and are not p-values."
        ),
        label="tab:repeatability_highest_pair_reversals",
        alignment="lrllrlr",
    )


def build_repeatability_package(source: Path, output: Path) -> None:
    _clean_dir(output)
    figures_dir = output / "figures"
    for roi_slug in REPEATABILITY_ROIS:
        roi_figures = figures_dir / roi_slug
        roi_figures.mkdir(parents=True)
        for figure in sorted((source / roi_slug).glob("*.png")):
            if figure.stem in REPEATABILITY_FIGURE_STEMS:
                shutil.copy2(figure, roi_figures / figure.name)
        copied = {path.stem for path in roi_figures.glob("*.png")}
        if copied != REPEATABILITY_FIGURE_STEMS:
            raise RuntimeError(
                f"{roi_slug} repeatability figure mismatch: "
                f"missing={sorted(REPEATABILITY_FIGURE_STEMS - copied)}, "
                f"unexpected={sorted(copied - REPEATABILITY_FIGURE_STEMS)}"
            )
    _copy_repeatability_figure_captions(source, output)
    _write_tables(
        output,
        [
            _repeatability_subject_summary(source),
            _repeatability_rank_summary(source),
            _repeatability_top_pairs(source),
        ],
    )


def _validate_package(root: Path) -> None:
    allowed = {".png", ".md", ".csv", ".tex"}
    files = [path for path in root.rglob("*") if path.is_file()]
    unexpected = [path for path in files if path.suffix.lower() not in allowed]
    if unexpected:
        raise RuntimeError(f"Unexpected package files: {unexpected}")
    figures = sorted((root / "figures").rglob("*.png"))
    if not figures:
        raise RuntimeError(f"No PNG figures in {root}")
    caption_csv = pd.read_csv(root / "captions" / "figure_captions.csv")
    caption_names = set(caption_csv["figure"])
    relative_stems = {
        str(path.relative_to(root / "figures").with_suffix(""))
        for path in figures
    }
    if caption_names != relative_stems:
        raise RuntimeError(
            f"Figure/caption mismatch in {root}: "
            f"{sorted(relative_stems - caption_names)} "
            f"{sorted(caption_names - relative_stems)}"
        )
    csv_tables = sorted((root / "tables" / "csv").glob("*.csv"))
    tex_tables = sorted((root / "tables" / "latex").glob("*.tex"))
    if {path.stem for path in csv_tables} != {
        path.stem for path in tex_tables
    }:
        raise RuntimeError(f"CSV/LaTeX table mismatch in {root}")
    table_captions = pd.read_csv(root / "captions" / "table_captions.csv")
    if set(table_captions["table"]) != {path.stem for path in csv_tables}:
        raise RuntimeError(f"Table/caption mismatch in {root}")
    for csv_path in csv_tables:
        path = root / "tables" / "latex" / f"{csv_path.stem}.tex"
        text = path.read_text(encoding="utf-8")
        if (
            text.count("{") != text.count("}")
            or text.count(r"\begin{") != text.count(r"\end{")
            or r"\caption{" not in text
            or r"\label{" not in text
            or r"\toprule" not in text
            or r"\bottomrule" not in text
        ):
            raise RuntimeError(f"Invalid LaTeX structure in {path}")
        frame = pd.read_csv(
            csv_path,
            dtype=str,
            keep_default_na=False,
        )
        expected_header = (
            " & ".join(_latex_escape(column) for column in frame.columns)
            + r" \\"
        )
        if expected_header not in text:
            raise RuntimeError(f"CSV header missing from LaTeX table {path}")
        for row in frame.itertuples(index=False, name=None):
            expected_row = (
                " & ".join(_latex_escape(value) for value in row) + r" \\"
            )
            if expected_row not in text:
                raise RuntimeError(
                    f"CSV row missing from LaTeX table {path}: {row}"
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camcan-source", type=Path, required=True)
    parser.add_argument("--repeatability-source", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    camcan_output = out_root / "camcan_paper_package_v3"
    repeatability_output = out_root / "repeatability_paper_package_v3"
    build_camcan_package(
        args.camcan_source.expanduser().resolve(),
        camcan_output,
    )
    build_repeatability_package(
        args.repeatability_source.expanduser().resolve(),
        repeatability_output,
    )
    _validate_package(camcan_output)
    _validate_package(repeatability_output)
    payload = {
        "camcan": {
            "path": str(camcan_output),
            "figures": len(list((camcan_output / "figures").glob("*.png"))),
            "tables": len(
                list((camcan_output / "tables" / "csv").glob("*.csv"))
            ),
        },
        "repeatability": {
            "path": str(repeatability_output),
            "figures": len(
                list((repeatability_output / "figures").rglob("*.png"))
            ),
            "tables": len(
                list(
                    (repeatability_output / "tables" / "csv").glob("*.csv")
                )
            ),
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
