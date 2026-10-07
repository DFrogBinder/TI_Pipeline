#!/usr/bin/env python3
"""Build combined manuscript figures for the two repeatability targets.

The script preserves the approved run-level, ranking, and bootstrap panels by
combining their existing PNG outputs. It rebuilds the mesh-element and tissue-
volume figures directly from the repeat-level mesh-metric tables so the two
targets share one consistent figure layout.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


TISSUE_NAMES = {
    1: "White matter",
    2: "Grey matter",
    3: "CSF",
    4: "Bone",
    5: "Scalp",
    6: "Eyes",
    7: "Compact bone",
    8: "Spongy bone",
    9: "Blood",
    10: "Muscle",
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _repeat_number(repeat_tag: str) -> int:
    return int(repeat_tag.rsplit("_", 1)[-1])


def _mapping(value: str) -> dict[int, float]:
    return {int(key): float(number) for key, number in json.loads(value).items()}


def _participant_label(subject: str) -> str:
    return subject.removeprefix("sub-").removeprefix("CC")


def _combine_vertical(
    top_path: Path,
    bottom_path: Path,
    output_path: Path,
    *,
    panel_letters: bool,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    images = [Image.open(path).convert("RGB") for path in (top_path, bottom_path)]
    width = max(image.width for image in images)
    resized = []
    for image in images:
        if image.width != width:
            height = round(image.height * width / image.width)
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        resized.append(image)

    separator = max(12, width // 180)
    canvas = Image.new(
        "RGB",
        (width, sum(image.height for image in resized) + separator),
        "white",
    )
    y = 0
    panel_positions = []
    for index, image in enumerate(resized):
        panel_positions.append(y)
        canvas.paste(image, (0, y))
        y += image.height
        if index == 0:
            draw = ImageDraw.Draw(canvas)
            draw.line(
                (width * 0.04, y + separator / 2, width * 0.96, y + separator / 2),
                fill="#d0d0d0",
                width=max(2, width // 1400),
            )
            y += separator

    if panel_letters:
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(34, width // 72))
        except OSError:
            font = ImageFont.load_default()
        for letter, panel_y in zip(("A", "B"), panel_positions):
            draw.text(
                (max(18, width // 175), panel_y + max(14, width // 220)),
                letter,
                fill="black",
                font=font,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, dpi=(220, 220), optimize=True)


def _condition_values(
    rows: list[dict[str, str]],
    condition: str,
) -> dict[str, list[tuple[int, float]]]:
    values: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if row["condition"] != condition:
            continue
        values[row["subject"]].append(
            (_repeat_number(row["repeat_tag"]), float(row["mesh_elements"]))
        )
    for subject in values:
        values[subject].sort()
    return dict(values)


def _write_element_figure(
    output_path: Path,
    target_rows: list[tuple[str, list[dict[str, str]]]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(2, 1, figsize=(13.2, 10.0), sharey=True)
    remesh_color = "#0b5d88"
    point_color = "#a9d5ea"
    fixed_color = "#b35400"

    all_deviations: list[float] = []
    prepared = []
    for target_name, rows in target_rows:
        remesh = _condition_values(rows, "remesh")
        fixed = _condition_values(rows, "fixed_mesh")
        subjects = sorted(
            remesh,
            key=lambda subject: statistics.mean(value for _run, value in remesh[subject]),
            reverse=True,
        )
        target_data = []
        for subject in subjects:
            remesh_values = [value for _run, value in remesh[subject]]
            mean_value = statistics.mean(remesh_values)
            deviations = np.asarray(remesh_values, dtype=float) - mean_value
            fixed_values = [value for _run, value in fixed[subject]]
            fixed_deviation = statistics.mean(fixed_values) - mean_value
            target_data.append((subject, deviations, fixed_deviation))
            all_deviations.extend(deviations.tolist())
            all_deviations.append(fixed_deviation)
        prepared.append((target_name, target_data))

    absolute_limit = max(abs(value) for value in all_deviations) * 1.12
    rng = np.random.default_rng(20260804)
    for panel_index, (axis, (target_name, target_data)) in enumerate(
        zip(axes, prepared)
    ):
        for subject_index, (_subject, deviations, fixed_deviation) in enumerate(
            target_data
        ):
            jitter = rng.uniform(-0.055, 0.055, size=len(deviations))
            axis.scatter(
                subject_index + jitter,
                deviations,
                s=23,
                color=point_color,
                alpha=0.74,
                edgecolors="none",
                zorder=2,
            )
            sd = float(np.std(deviations, ddof=1))
            axis.errorbar(
                subject_index,
                0.0,
                yerr=sd,
                fmt="o",
                markersize=9,
                markerfacecolor="white",
                markeredgecolor=remesh_color,
                markeredgewidth=2,
                ecolor=remesh_color,
                elinewidth=2,
                capsize=4,
                capthick=2,
                zorder=4,
            )
            axis.scatter(
                subject_index + 0.16,
                fixed_deviation,
                marker="D",
                s=53,
                color=fixed_color,
                edgecolors="white",
                linewidths=0.7,
                zorder=5,
            )
        axis.axhline(0, color="#8c8c8c", linestyle="--", linewidth=1)
        axis.set_ylim(-absolute_limit, absolute_limit)
        axis.set_xticks(
            range(len(target_data)),
            [_participant_label(subject) for subject, _values, _fixed in target_data],
            rotation=38,
            ha="right",
        )
        axis.set_ylabel("Deviation from 40-run mean\n(elements)")
        axis.set_title(f"{chr(65 + panel_index)}  {target_name}", loc="left")
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.65)
        axis.set_axisbelow(True)
    axes[-1].set_xlabel("Participant (ordered by decreasing remesh mean)")

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=point_color,
            markeredgecolor="none",
            label="Remesh runs",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor=fixed_color,
            markeredgecolor="white",
            label="Fixed-mesh element count",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=remesh_color,
            markerfacecolor="white",
            markeredgewidth=2,
            label="Remesh mean and SD",
        ),
    ]
    figure.suptitle("Within-participant variation in mesh element count", y=0.995)
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.925))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _select_tissue_example(
    rows: list[dict[str, str]],
) -> tuple[str, list[tuple[int, dict[int, float]]]]:
    import numpy as np

    by_subject: dict[str, list[tuple[int, dict[int, float]]]] = defaultdict(list)
    for row in rows:
        if row["condition"] != "remesh":
            continue
        mapping = _mapping(row["mesh_volume_mm3_by_tissue"])
        by_subject[row["subject"]].append(
            (_repeat_number(row["repeat_tag"]), mapping)
        )
    for subject in by_subject:
        by_subject[subject].sort()

    def score(subject: str) -> float:
        subject_rows = by_subject[subject]
        return float(
            sum(
                np.std(
                    [
                        mapping.get(tissue, 0.0) / sum(mapping.values())
                        for _run, mapping in subject_rows
                    ]
                )
                for tissue in TISSUE_NAMES
            )
        )

    subject = max(by_subject, key=score)
    return subject, by_subject[subject]


def _write_tissue_figure(
    output_path: Path,
    target_rows: list[tuple[str, list[dict[str, str]]]],
    summary_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    prepared = []
    maximum_absolute_deviation = 0.0
    summary: dict[str, object] = {}
    for target_name, rows in target_rows:
        subject, subject_rows = _select_tissue_example(rows)
        tissue_ids = [
            tissue
            for tissue in TISSUE_NAMES
            if any(mapping.get(tissue, 0.0) > 0 for _run, mapping in subject_rows)
        ]
        volumes = {
            tissue: np.asarray(
                [mapping.get(tissue, 0.0) for _run, mapping in subject_rows],
                dtype=float,
            )
            for tissue in tissue_ids
        }
        deviations = np.vstack(
            [volumes[tissue] - np.mean(volumes[tissue]) for tissue in tissue_ids]
        )
        maximum_absolute_deviation = max(
            maximum_absolute_deviation,
            float(np.max(np.abs(deviations))),
        )
        target_summary = {}
        for tissue in tissue_ids:
            values = volumes[tissue]
            target_summary[TISSUE_NAMES[tissue]] = {
                "range_mm3": float(np.max(values) - np.min(values)),
                "minimum_deviation_mm3": float(np.min(values) - np.mean(values)),
                "maximum_deviation_mm3": float(np.max(values) - np.mean(values)),
            }
        summary[target_name] = {
            "participant": subject,
            "tissues": target_summary,
        }
        prepared.append((target_name, subject, subject_rows, tissue_ids, deviations))

    figure, axes = plt.subplots(2, 1, figsize=(13.2, 8.2))
    image = None
    for panel_index, (
        axis,
        (target_name, subject, subject_rows, tissue_ids, deviations),
    ) in enumerate(zip(axes, prepared)):
        image = axis.imshow(
            deviations,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-maximum_absolute_deviation,
            vmax=maximum_absolute_deviation,
        )
        axis.set_yticks(
            range(len(tissue_ids)),
            [TISSUE_NAMES[tissue] for tissue in tissue_ids],
        )
        tick_indices = np.arange(0, len(subject_rows), 4)
        axis.set_xticks(
            tick_indices,
            [str(subject_rows[index][0]) for index in tick_indices],
        )
        axis.set_title(
            f"{chr(65 + panel_index)}  {target_name}: participant "
            f"{_participant_label(subject)}",
            loc="left",
        )
        axis.set_xlabel("Remesh run")
    assert image is not None
    colorbar = figure.colorbar(
        image,
        ax=axes,
        fraction=0.022,
        pad=0.025,
    )
    colorbar.set_label("Deviation from tissue-specific 40-run mean (mm³)")
    figure.suptitle("Absolute tissue-volume deviations across remesh runs", y=0.995)
    figure.subplots_adjust(left=0.14, right=0.88, top=0.91, bottom=0.08, hspace=0.40)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-figure-root", type=Path, required=True)
    parser.add_argument("--left-mesh-metrics", type=Path, required=True)
    parser.add_argument("--m1-mesh-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    source_root = args.source_figure_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_specs = [
        (
            "01_primary_median_roi_repeat_distributions.png",
            "figure_target_field_repeatability.png",
            True,
        ),
        (
            "02_single_repeat_subject_ranking_uncertainty.png",
            "figure_single_run_ranking_uncertainty.png",
            False,
        ),
    ]
    for source_name, output_name, panel_letters in combined_specs:
        _combine_vertical(
            source_root / "left_hippocampus" / source_name,
            source_root / "right_m1" / source_name,
            output_dir / output_name,
            panel_letters=panel_letters,
        )
    _combine_vertical(
        source_root / "precision_curves_left_hippocampus.png",
        source_root / "precision_curves_right_m1.png",
        output_dir / "figure_bootstrap_precision.png",
        panel_letters=False,
    )

    left_rows = _read_rows(args.left_mesh_metrics.resolve())
    m1_rows = _read_rows(args.m1_mesh_metrics.resolve())
    target_rows = [("Left hippocampus", left_rows), ("Right M1", m1_rows)]
    _write_element_figure(
        output_dir / "figure_mesh_element_count_variability.png",
        target_rows,
    )
    _write_tissue_figure(
        output_dir / "figure_tissue_volume_deviations.png",
        target_rows,
        output_dir / "figure_tissue_volume_deviations_summary.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
