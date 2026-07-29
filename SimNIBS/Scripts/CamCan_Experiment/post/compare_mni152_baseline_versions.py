#!/usr/bin/env python3
"""Compare isolated SimNIBS-version MNI152 baselines on identical inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post.camcan_manuscript_analysis import (  # noqa: E402
    DEFAULT_ROBUST_MAX_PERCENTILE,
    DEFAULT_THRESHOLDS_V_PER_M,
    DEFAULT_TOP_PERCENTILE,
    DEFAULT_UPPER_TAIL_FRACTION,
    MNI_BASELINE_NAMES,
    _compute_mni_record,
    _find_baseline_ti,
)


ROI_ORDER = (
    "Left_M1",
    "Right_DLPC",
    "Left_Hippocampus",
    "Right_Thalamus",
)
SELECTED_METRICS = (
    "roi_min_v_per_m",
    "roi_mean_v_per_m",
    "roi_median_v_per_m",
    "roi_robust_max_p99_9_v_per_m",
    "target_coverage_percent_ge_0p2",
    "off_target_coverage_percent_ge_0p2",
    "anatomical_roi_mean_v_per_m",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_provenance(parent: Path, roi: str) -> dict[str, Any]:
    path = (
        parent
        / MNI_BASELINE_NAMES[roi]
        / "anat"
        / "SimNIBS"
        / "mni_baseline_provenance.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing MNI152 provenance: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"Incomplete MNI152 provenance: {path}")
    return payload


def provenance_version(payload: dict[str, Any]) -> str:
    software = payload.get("software")
    if isinstance(software, dict):
        return str(software.get("simnibs_version", "unknown"))
    log_validation = payload.get("log_validation")
    if isinstance(log_validation, dict):
        return str(log_validation.get("simnibs_version", "unknown"))
    return "unknown"


def provenance_input_hash(payload: dict[str, Any], name: str) -> str:
    inputs = payload.get("inputs")
    if isinstance(inputs, dict):
        return str(inputs.get(name, ""))
    return str(payload.get(name, ""))


def compare_volumes(old_path: Path, new_path: Path) -> dict[str, Any]:
    old_img = nib.load(str(old_path))
    new_img = nib.load(str(new_path))
    if old_img.shape != new_img.shape:
        raise ValueError(
            f"MNI baseline shape mismatch: {old_img.shape} != {new_img.shape}."
        )
    if not np.allclose(old_img.affine, new_img.affine, atol=1e-3):
        raise ValueError("MNI baseline affine mismatch.")
    old = old_img.get_fdata(dtype=np.float32)
    new = new_img.get_fdata(dtype=np.float32)
    finite = np.isfinite(old) & np.isfinite(new)
    if not finite.any():
        raise ValueError("No shared finite MNI baseline voxels.")
    old_values = old[finite].astype(np.float64, copy=False)
    new_values = new[finite].astype(np.float64, copy=False)
    difference = new_values - old_values
    absolute = np.abs(difference)
    denominator = float(np.linalg.norm(old_values))
    correlation = (
        float(np.corrcoef(old_values, new_values)[0, 1])
        if old_values.size > 1
        else math.nan
    )
    return {
        "shared_finite_voxels": int(finite.sum()),
        "old_mean_v_per_m": float(np.mean(old_values)),
        "new_mean_v_per_m": float(np.mean(new_values)),
        "signed_mean_difference_v_per_m": float(np.mean(difference)),
        "mean_absolute_difference_v_per_m": float(np.mean(absolute)),
        "root_mean_square_difference_v_per_m": float(
            np.sqrt(np.mean(np.square(difference)))
        ),
        "p95_absolute_difference_v_per_m": float(np.percentile(absolute, 95.0)),
        "p99_absolute_difference_v_per_m": float(np.percentile(absolute, 99.0)),
        "maximum_absolute_difference_v_per_m": float(np.max(absolute)),
        "relative_l2_difference_percent": (
            float(np.linalg.norm(difference) / denominator * 100.0)
            if denominator > 0
            else math.nan
        ),
        "pearson_r": correlation,
    }


def numeric_metric_deltas(
    roi: str,
    old_record: dict[str, Any],
    new_record: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for metric in old_record:
        if (
            metric in {"subject", "roi"}
            or metric.startswith("optimizer_roi_")
            or metric not in new_record
        ):
            continue
        try:
            old_value = float(old_record[metric])
            new_value = float(new_record[metric])
        except (TypeError, ValueError):
            continue
        delta = new_value - old_value
        rows.append(
            {
                "roi": roi,
                "metric": metric,
                "simnibs_4p5p0": old_value,
                "simnibs_4p0p1": new_value,
                "delta_4p0p1_minus_4p5p0": delta,
                "percent_delta_relative_to_4p5p0": (
                    delta / old_value * 100.0
                    if old_value != 0.0
                    else math.nan
                ),
            }
        )
    return rows


def build_report(
    *,
    out_path: Path,
    selected_deltas: pd.DataFrame,
    voxelwise: pd.DataFrame,
    mesh_sha256: str,
    t1_sha256: str,
    targets_sha256: str,
) -> None:
    lines = [
        "# MNI152 SimNIBS 4.0.1 versus 4.5.0 validation",
        "",
        "The same fixed MNI152 mesh, reference T1, ROI-specific generic "
        "montages, currents, conductivities, electrode geometry, and "
        "`TI_utils.get_maxTI` workflow were used. Only the SimNIBS runtime "
        "version changed.",
        "",
        f"- MNI152 mesh SHA-256: `{mesh_sha256}`",
        f"- Reference T1 SHA-256: `{t1_sha256}`",
        f"- `targets.csv` SHA-256: `{targets_sha256}`",
        "",
        "## Target-ROI metrics",
        "",
        "| ROI | Metric | 4.5.0 | 4.0.1 | Δ (4.0.1 − 4.5.0) | Δ % |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in selected_deltas.to_dict(orient="records"):
        percent = row["percent_delta_relative_to_4p5p0"]
        percent_text = "NA" if not np.isfinite(percent) else f"{percent:.4g}"
        lines.append(
            f"| {row['roi']} | `{row['metric']}` | "
            f"{row['simnibs_4p5p0']:.8g} | "
            f"{row['simnibs_4p0p1']:.8g} | "
            f"{row['delta_4p0p1_minus_4p5p0']:.8g} | "
            f"{percent_text} |"
        )
    lines.extend(
        [
            "",
            "## Whole-brain voxelwise agreement",
            "",
            "| ROI | MAE (V/m) | RMSE (V/m) | Max | Relative L2 % | Pearson r |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in voxelwise.to_dict(orient="records"):
        lines.append(
            f"| {row['roi']} | "
            f"{row['mean_absolute_difference_v_per_m']:.8g} | "
            f"{row['root_mean_square_difference_v_per_m']:.8g} | "
            f"{row['maximum_absolute_difference_v_per_m']:.8g} | "
            f"{row['relative_l2_difference_percent']:.8g} | "
            f"{row['pearson_r']:.8g} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    old_parent = args.simnibs_4p5_parent.expanduser().resolve(strict=True)
    new_parent = args.simnibs_4p0p1_parent.expanduser().resolve(strict=True)
    atlas = args.mni_atlas.expanduser().resolve(strict=True)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    old_records = []
    new_records = []
    voxel_rows = []
    delta_rows = []
    versions: dict[str, set[str]] = {"4.5.0": set(), "4.0.1": set()}
    invariant_hashes: dict[str, set[str]] = {
        "mni_mesh_sha256": set(),
        "reference_t1_sha256": set(),
        "targets_csv_sha256": set(),
    }

    for roi in ROI_ORDER:
        old_provenance = load_provenance(old_parent, roi)
        new_provenance = load_provenance(new_parent, roi)
        versions["4.5.0"].add(provenance_version(old_provenance))
        versions["4.0.1"].add(provenance_version(new_provenance))
        for name in invariant_hashes:
            invariant_hashes[name].add(
                provenance_input_hash(old_provenance, name)
            )
            invariant_hashes[name].add(
                provenance_input_hash(new_provenance, name)
            )

        old_record = _compute_mni_record(
            roi=roi,
            baseline_parent=old_parent,
            mni_atlas_path=atlas,
            thresholds=DEFAULT_THRESHOLDS_V_PER_M,
            top_percentile=DEFAULT_TOP_PERCENTILE,
            robust_max_percentile=DEFAULT_ROBUST_MAX_PERCENTILE,
            upper_tail_fraction=DEFAULT_UPPER_TAIL_FRACTION,
        )
        new_record = _compute_mni_record(
            roi=roi,
            baseline_parent=new_parent,
            mni_atlas_path=atlas,
            thresholds=DEFAULT_THRESHOLDS_V_PER_M,
            top_percentile=DEFAULT_TOP_PERCENTILE,
            robust_max_percentile=DEFAULT_ROBUST_MAX_PERCENTILE,
            upper_tail_fraction=DEFAULT_UPPER_TAIL_FRACTION,
        )
        old_records.append(old_record)
        new_records.append(new_record)
        delta_rows.extend(numeric_metric_deltas(roi, old_record, new_record))

        old_path = _find_baseline_ti(old_parent / MNI_BASELINE_NAMES[roi])
        new_path = _find_baseline_ti(new_parent / MNI_BASELINE_NAMES[roi])
        voxel_rows.append(
            {
                "roi": roi,
                "simnibs_4p5p0_ti_sha256": sha256_file(old_path),
                "simnibs_4p0p1_ti_sha256": sha256_file(new_path),
                **compare_volumes(old_path, new_path),
            }
        )

    if versions["4.5.0"] != {"4.5.0"}:
        raise ValueError(f"Unexpected source SimNIBS versions: {versions['4.5.0']}.")
    if versions["4.0.1"] != {"4.0.1"}:
        raise ValueError(f"Unexpected validation SimNIBS versions: {versions['4.0.1']}.")
    for label, values in invariant_hashes.items():
        if len(values) != 1 or "" in values:
            raise ValueError(f"{label} is not invariant across versions: {values}.")

    old_csv = out_dir / "mni152_metrics_simnibs_4p5p0.csv"
    new_csv = out_dir / "mni152_metrics_simnibs_4p0p1.csv"
    delta_csv = out_dir / "mni152_metric_differences_4p0p1_minus_4p5p0.csv"
    selected_csv = out_dir / "mni152_selected_metric_differences.csv"
    voxel_csv = out_dir / "mni152_voxelwise_differences.csv"
    write_csv(old_csv, old_records)
    write_csv(new_csv, new_records)
    write_csv(delta_csv, delta_rows)
    selected_rows = [
        row for row in delta_rows if row["metric"] in SELECTED_METRICS
    ]
    write_csv(selected_csv, selected_rows)
    write_csv(voxel_csv, voxel_rows)

    selected_df = pd.DataFrame(selected_rows)
    selected_df["roi"] = pd.Categorical(
        selected_df["roi"],
        categories=ROI_ORDER,
        ordered=True,
    )
    selected_df["metric"] = pd.Categorical(
        selected_df["metric"],
        categories=SELECTED_METRICS,
        ordered=True,
    )
    selected_df = selected_df.sort_values(["roi", "metric"])
    voxel_df = pd.DataFrame(voxel_rows)
    voxel_df["roi"] = pd.Categorical(
        voxel_df["roi"],
        categories=ROI_ORDER,
        ordered=True,
    )
    voxel_df = voxel_df.sort_values("roi")
    report_path = out_dir / "MNI152_SIMNIBS_VERSION_COMPARISON.md"
    build_report(
        out_path=report_path,
        selected_deltas=selected_df,
        voxelwise=voxel_df,
        mesh_sha256=next(iter(invariant_hashes["mni_mesh_sha256"])),
        t1_sha256=next(iter(invariant_hashes["reference_t1_sha256"])),
        targets_sha256=next(iter(invariant_hashes["targets_csv_sha256"])),
    )

    outputs = [
        old_csv,
        new_csv,
        delta_csv,
        selected_csv,
        voxel_csv,
        report_path,
    ]
    manifest = {
        "comparison_schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": "SimNIBS 4.0.1 minus SimNIBS 4.5.0",
        "scientific_invariants": {
            key: next(iter(values))
            for key, values in invariant_hashes.items()
        },
        "rois": list(ROI_ORDER),
        "simulations_per_version": 4,
        "tdcs_fem_solves_per_version": 8,
        "simnibs_versions": {
            "source": "4.5.0",
            "validation": "4.0.1",
        },
        "outputs": [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in outputs
        ],
    }
    manifest_path = out_dir / "comparison_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simnibs-4p5-parent", type=Path, required=True)
    parser.add_argument("--simnibs-4p0p1-parent", type=Path, required=True)
    parser.add_argument("--mni-atlas", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    manifest = run(parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
