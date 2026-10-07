#!/usr/bin/env python3
"""Build full-cohort precision curves from optimizer-matched spherical ROIs.

The established bootstrap implementation remains unchanged. This adapter
validates the current repeatability cohort, converts each target's remesh
measurements into the directory structure expected by the canonical script,
runs that script once per target, and renders manuscript figures without
threshold, worst-subject, or recommendation overlays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_COLUMN = "roi_mean_v_per_m"
CANONICAL_METRIC_NAME = "mean_target_sphere_v_per_m"
EXPECTED_SUBJECTS = 10
EXPECTED_REMESH_RUNS = 40
BOOTSTRAP_ITERATIONS = 20_000
BOOTSTRAP_SEED = 0
BOOTSTRAP_COVERAGE = 0.95

TARGETS = {
    "left_hippocampus": {
        "roi": "Left_Hippocampus",
        "volume_mm3": 200.0,
        "title": "200 mm³ left hippocampal spherical target ROI",
    },
    "right_m1": {
        "roi": "Right_M1",
        "volume_mm3": 100.0,
        "title": "100 mm³ right M1 spherical target ROI",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full-cohort bootstrap precision figures for both targets."
    )
    parser.add_argument("--left-csv", required=True, type=Path)
    parser.add_argument("--right-csv", required=True, type=Path)
    parser.add_argument("--canonical-script", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_target_frame(path: Path, target_key: str) -> tuple[pd.DataFrame, list[str]]:
    spec = TARGETS[target_key]
    frame = pd.read_csv(path)
    required = {
        "schema_version",
        "subject",
        "condition",
        "repeat_tag",
        "roi",
        "requested_roi_volume_mm3",
        METRIC_COLUMN,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")

    if set(frame["schema_version"].astype(int)) != {2}:
        raise ValueError(f"{path} does not contain only schema-version 2 records")
    if set(frame["roi"].astype(str)) != {spec["roi"]}:
        raise ValueError(f"{path} does not contain only {spec['roi']} records")
    volumes = pd.to_numeric(frame["requested_roi_volume_mm3"], errors="coerce")
    if not np.allclose(volumes, spec["volume_mm3"], rtol=0.0, atol=1e-9):
        raise ValueError(f"{path} does not use the expected {spec['volume_mm3']} mm3 ROI")

    remesh = frame.loc[frame["condition"].astype(str) == "remesh"].copy()
    remesh[METRIC_COLUMN] = pd.to_numeric(remesh[METRIC_COLUMN], errors="coerce")
    if not np.isfinite(remesh[METRIC_COLUMN].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains non-finite {METRIC_COLUMN} values")

    subjects = sorted(remesh["subject"].astype(str).unique().tolist())
    if len(subjects) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"{path} contains {len(subjects)} remesh subjects, expected {EXPECTED_SUBJECTS}"
        )

    counts = remesh.groupby("subject")["repeat_tag"].nunique()
    bad_counts = counts[counts != EXPECTED_REMESH_RUNS]
    if not bad_counts.empty:
        raise ValueError(f"Unexpected remesh run counts in {path}: {bad_counts.to_dict()}")
    if remesh.duplicated(["subject", "repeat_tag"]).any():
        raise ValueError(f"{path} contains duplicate subject/run records")

    return remesh.sort_values(["subject", "repeat_tag"]), subjects


def write_canonical_inputs(frame: pd.DataFrame, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for subject, subject_frame in frame.groupby("subject", sort=True):
        subject_dir = destination / str(subject)
        subject_dir.mkdir()
        adapted = pd.DataFrame(
            {
                "repeat_tag": subject_frame["repeat_tag"].astype(str).to_numpy(),
                CANONICAL_METRIC_NAME: subject_frame[METRIC_COLUMN].to_numpy(dtype=float),
            }
        )
        adapted.to_csv(subject_dir / "summary.csv", index=False)


def run_canonical_bootstrap(
    canonical_script: Path,
    input_dir: Path,
    output_dir: Path,
) -> None:
    command = [
        sys.executable,
        str(canonical_script),
        "--input-root",
        str(input_dir),
        "--metric",
        CANONICAL_METRIC_NAME,
        "--estimator",
        "mean",
        "--tolerance",
        "0.05",
        "--coverage",
        str(BOOTSTRAP_COVERAGE),
        "--criterion",
        "ci_half_width",
        "--bootstrap-iterations",
        str(BOOTSTRAP_ITERATIONS),
        "--min-repeats",
        "2",
        "--max-repeats",
        str(EXPECTED_REMESH_RUNS),
        "--out-dir",
        str(output_dir),
        "--seed",
        str(BOOTSTRAP_SEED),
    ]
    subprocess.run(command, check=True)


def render_subject_curves(curve_path: Path, target_key: str, output_path: Path) -> None:
    spec = TARGETS[target_key]
    curve = pd.read_csv(curve_path)
    subjects = sorted(curve["subject"].astype(str).unique().tolist())
    if len(subjects) != EXPECTED_SUBJECTS:
        raise ValueError(f"Canonical output contains {len(subjects)} subjects")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2), sharex=True, sharey=True)
    panels = [
        (axes[0], "ci_half_width_rel", "Central 95% interval half-width"),
        (
            axes[1],
            "abs_error_quantile_rel",
            "95th percentile of absolute relative error",
        ),
    ]
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(subjects)))

    for axis, value_column, panel_title in panels:
        for color, subject in zip(colors, subjects):
            subject_curve = curve.loc[curve["subject"].astype(str) == subject]
            axis.plot(
                subject_curve["n_repeats"],
                subject_curve[value_column] * 100.0,
                color=color,
                linewidth=1.45,
                alpha=0.9,
                label=subject,
            )
        axis.set_title(panel_title, fontsize=13)
        axis.set_xlabel("Number of remesh runs averaged")
        axis.set_xlim(2, EXPECTED_REMESH_RUNS)
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Relative uncertainty (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9.5,
        handlelength=2.8,
        columnspacing=1.6,
    )
    fig.suptitle(
        f"Bootstrap precision of mean field in the {spec['title']}",
        fontsize=16,
        y=0.985,
    )
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 0.93))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_curve(curve_path: Path, target_key: str) -> list[dict[str, float | int | str]]:
    curve = pd.read_csv(curve_path)
    rows: list[dict[str, float | int | str]] = []
    for n_repeats in (2, EXPECTED_REMESH_RUNS):
        subset = curve.loc[curve["n_repeats"] == n_repeats]
        rows.append(
            {
                "target": target_key,
                "n_repeats": n_repeats,
                "subjects": int(subset["subject"].nunique()),
                "ci_half_width_percent_min": float(subset["ci_half_width_rel"].min() * 100.0),
                "ci_half_width_percent_max": float(subset["ci_half_width_rel"].max() * 100.0),
                "abs_error_percent_min": float(
                    subset["abs_error_quantile_rel"].min() * 100.0
                ),
                "abs_error_percent_max": float(
                    subset["abs_error_quantile_rel"].max() * 100.0
                ),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    left_csv = args.left_csv.expanduser().resolve()
    right_csv = args.right_csv.expanduser().resolve()
    canonical_script = args.canonical_script.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))

    left, left_subjects = validate_target_frame(left_csv, "left_hippocampus")
    right, right_subjects = validate_target_frame(right_csv, "right_m1")
    if left_subjects != right_subjects:
        raise ValueError("The two target datasets do not contain the same subjects")

    input_root = output_dir / "canonical_inputs"
    canonical_root = output_dir / "canonical_outputs"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    summary_rows: list[dict[str, float | int | str]] = []

    for target_key, frame in (
        ("left_hippocampus", left),
        ("right_m1", right),
    ):
        target_input = input_root / target_key
        target_output = canonical_root / target_key
        write_canonical_inputs(frame, target_input)
        run_canonical_bootstrap(canonical_script, target_input, target_output)
        curve_path = target_output / "per_subject_curve.csv"
        figure_path = figures_dir / f"precision_curves_{target_key}.png"
        render_subject_curves(curve_path, target_key, figure_path)
        summary_rows.extend(summarize_curve(curve_path, target_key))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "precision_curve_summary.csv", index=False)
    manifest = {
        "status": "complete",
        "analysis": "full-cohort bootstrap precision of mean target-sphere field",
        "subject_count": EXPECTED_SUBJECTS,
        "subjects": left_subjects,
        "targets": {
            key: {
                "roi": value["roi"],
                "requested_volume_mm3": value["volume_mm3"],
            }
            for key, value in TARGETS.items()
        },
        "condition": "remesh",
        "runs_per_subject": EXPECTED_REMESH_RUNS,
        "run_counts_evaluated": [2, EXPECTED_REMESH_RUNS],
        "run_count_range_inclusive": [2, EXPECTED_REMESH_RUNS],
        "run_level_metric": "mean maximum-TI envelope within optimizer-matched target sphere",
        "source_column": METRIC_COLUMN,
        "across_run_estimator": "arithmetic mean",
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_sampling": "with replacement",
        "coverage": BOOTSTRAP_COVERAGE,
        "seed": BOOTSTRAP_SEED,
        "display_policy": {
            "subject_ids": True,
            "worst_subject_curve": False,
            "tolerance": False,
            "recommendation": False,
        },
        "canonical_script": str(canonical_script),
        "canonical_script_sha256": sha256_file(canonical_script),
        "source_csvs": {
            "left_hippocampus": {
                "path": str(left_csv),
                "sha256": sha256_file(left_csv),
            },
            "right_m1": {
                "path": str(right_csv),
                "sha256": sha256_file(right_csv),
            },
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    for figure in sorted(figures_dir.glob("*.png")):
        print(f"[INFO] Wrote {figure}")
    print(f"[INFO] Wrote {output_dir / 'precision_curve_summary.csv'}")
    print(f"[INFO] Wrote {output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
