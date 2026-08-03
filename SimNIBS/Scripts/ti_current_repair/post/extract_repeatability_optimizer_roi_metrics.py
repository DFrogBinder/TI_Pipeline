#!/usr/bin/env python3
"""Extract optimizer-matched ROI metrics from completed repeatability fields.

This is a read-only metric refresh. It reconstructs the parcel-clipped target
sphere used by the montage optimization on each subject's TI image grid, then
summarizes the already-computed ``ti_brain_only.nii.gz`` fields. It never
changes the simulations, meshes, or existing analysis directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PIPELINE_ROOT.parent
CAMCAN_ROOT = SCRIPTS_ROOT / "CamCan_Experiment"
for candidate in (SCRIPTS_ROOT, PIPELINE_ROOT, CAMCAN_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiment_config import (  # noqa: E402
    load_experiment_config,
    repeat_tag,
    subject_condition_repeats_root,
)


SCHEMA_VERSION = 2
ROI_SPECS = {
    "left-hippocampus": {
        "roi": "Left_Hippocampus",
        "roi_class": "subcortical",
        "label_ids": (17,),
        "target_volume_mm3": 200.0,
    },
    "right-m1": {
        "roi": "Right_M1",
        "roi_class": "cortical",
        "label_ids": (12129,),
        "target_volume_mm3": 100.0,
    },
}
ROW_FIELDS = [
    "schema_version",
    "subject",
    "condition",
    "repeat_tag",
    "roi",
    "roi_class",
    "anatomical_label_ids",
    "requested_roi_volume_mm3",
    "achieved_roi_volume_mm3",
    "roi_radius_mm",
    "roi_voxels",
    "voxel_volume_mm3",
    "roi_min_v_per_m",
    "roi_mean_v_per_m",
    "roi_median_v_per_m",
    "roi_p95_v_per_m",
    "roi_max_v_per_m",
    "finite_roi_voxels",
    "nonfinite_roi_voxels",
    "finite_roi_fraction",
    "ti_path",
    "ti_size_bytes",
    "atlas_path",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv_atomic(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _completion_scope(config) -> dict[str, object]:
    receipt_path = config.experiment_root / "_pipeline" / "workflow" / "complete.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"Missing completion receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "complete":
        raise RuntimeError(f"Experiment is not complete: {receipt_path}")
    scope = receipt.get("scope")
    if not isinstance(scope, dict):
        raise RuntimeError(f"Completion receipt has no scope: {receipt_path}")

    expected_subjects = len(config.subjects)
    condition_counts = {
        condition.name: int(condition.repeat_count)
        for condition in config.conditions
    }
    expected_fields = expected_subjects * sum(condition_counts.values())
    if int(scope.get("subject_count", -1)) != expected_subjects:
        raise RuntimeError("Completion receipt subject count differs from the config")
    if len(set(condition_counts.values())) != 1:
        raise RuntimeError("Conditions do not share one run count")
    run_count = next(iter(condition_counts.values()))
    if int(scope.get("repeats_per_condition", -1)) != run_count:
        raise RuntimeError("Completion receipt run count differs from the config")
    expected_receipt_fields = scope.get("expected_ti_nifti")
    if expected_receipt_fields is not None and int(expected_receipt_fields) != expected_fields:
        raise RuntimeError("Completion receipt TI NIfTI count differs from the config")
    return {
        "receipt": str(receipt_path),
        "receipt_sha256": _sha256_file(receipt_path),
        "subject_count": expected_subjects,
        "conditions": condition_counts,
        "expected_fields": expected_fields,
        "receipt_roi": scope.get("roi"),
    }


def _roi_spec(config) -> dict[str, object]:
    preset = str(config.analysis.roi_preset or "").strip().lower()
    try:
        return {"preset": preset, **ROI_SPECS[preset]}
    except KeyError as exc:
        raise ValueError(
            f"Unsupported repeatability ROI preset {preset!r}. "
            f"Expected one of {sorted(ROI_SPECS)}."
        ) from exc


def _atlas_path(config, subject: str) -> Path:
    if not config.analysis.atlas_dir:
        raise ValueError("analysis.atlas_dir is required")
    atlas_root = Path(config.analysis.atlas_dir).expanduser().resolve()
    for suffix in (".nii.gz", ".nii"):
        candidate = atlas_root / f"{subject}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing subject atlas: {atlas_root}/{subject}.nii[.gz]")


def _ti_path(config, subject: str, condition: str, run_number: int) -> Path:
    tag = repeat_tag(run_number)
    return (
        subject_condition_repeats_root(config, subject, condition)
        / tag
        / subject
        / "anat"
        / "SimNIBS"
        / "ti_brain_only.nii.gz"
    )


def preflight(*, config_path: Path, output_root: Path) -> dict[str, object]:
    config = load_experiment_config(config_path, validate_paths=False)
    scope = _completion_scope(config)
    spec = _roi_spec(config)
    if not config.experiment_root.is_dir():
        raise FileNotFoundError(config.experiment_root)
    atlas_paths = [_atlas_path(config, subject) for subject in config.subjects]
    return {
        "status": "ready",
        "mode": "read_only_optimizer_roi_metric_extraction",
        "config": str(config.config_path),
        "config_sha256": _sha256_file(config.config_path),
        "experiment_root": str(config.experiment_root),
        "output_root": str(output_root),
        "source_outputs_modified": False,
        "meshing_tasks": 0,
        "fem_tasks": 0,
        "atlas_files": len(atlas_paths),
        "optimizer_roi": spec,
        **scope,
    }


def _grid_key(image) -> tuple[object, ...]:
    import numpy as np

    return (
        tuple(int(value) for value in image.shape[:3]),
        tuple(np.round(np.asarray(image.affine, dtype=float).ravel(), 7)),
    )


def extract_subject(
    *,
    config_path: Path,
    subject_index: int,
    output_root: Path,
) -> dict[str, object]:
    import nibabel as nib
    import numpy as np

    from nibabel.processing import resample_from_to
    from CamCan_Experiment.post.optimizer_target_roi import (
        build_optimizer_target_roi,
    )

    config = load_experiment_config(config_path, validate_paths=False)
    scope = _completion_scope(config)
    spec = _roi_spec(config)
    if subject_index < 0 or subject_index >= len(config.subjects):
        raise IndexError(
            f"subject index {subject_index} outside 0..{len(config.subjects) - 1}"
        )
    subject = config.subjects[subject_index]
    atlas_path = _atlas_path(config, subject)
    atlas_img = nib.load(str(atlas_path))
    atlas_native = np.asarray(atlas_img.dataobj).astype(np.int32, copy=False)
    label_ids = tuple(int(value) for value in spec["label_ids"])
    missing_labels = [
        value for value in label_ids if not np.any(atlas_native == value)
    ]
    if missing_labels:
        raise ValueError(f"{atlas_path} lacks required labels {missing_labels}")

    roi_cache: dict[tuple[object, ...], tuple[np.ndarray, dict[str, Any]]] = {}
    rows: list[dict[str, object]] = []
    for condition in config.conditions:
        for run_number in range(1, int(condition.repeat_count) + 1):
            tag = repeat_tag(run_number)
            ti_path = _ti_path(config, subject, condition.name, run_number)
            if not ti_path.is_file():
                raise FileNotFoundError(ti_path)
            ti_img = nib.load(str(ti_path))
            ti_data = np.asarray(ti_img.dataobj)
            if ti_data.ndim == 4 and ti_data.shape[3] == 3:
                ti_data = np.linalg.norm(ti_data, axis=3)
            elif ti_data.ndim != 3:
                raise ValueError(f"Unexpected TI data shape: {ti_data.shape}")
            ti_data = np.asarray(ti_data, dtype=np.float64)
            key = _grid_key(ti_img)
            if key not in roi_cache:
                if (
                    atlas_img.shape[:3] == ti_img.shape[:3]
                    and np.allclose(atlas_img.affine, ti_img.affine, atol=1e-5)
                ):
                    mapped_atlas = atlas_img
                else:
                    mapped_atlas = resample_from_to(atlas_img, ti_img, order=0)
                mapped_labels = np.asarray(mapped_atlas.dataobj).astype(
                    np.int32,
                    copy=False,
                )
                anatomical_mask = np.isin(mapped_labels, label_ids)
                optimizer_roi = build_optimizer_target_roi(
                    anatomical_mask=anatomical_mask,
                    reference_img=ti_img,
                    roi=str(spec["roi"]),
                    target_volume_mm3=float(spec["target_volume_mm3"]),
                    roi_class=str(spec["roi_class"]),
                )
                roi_cache[key] = (optimizer_roi.mask, optimizer_roi.metadata)
            roi_mask, roi_metadata = roi_cache[key]
            roi_voxels = int(roi_metadata["target_voxels"])
            finite_roi_mask = roi_mask & np.isfinite(ti_data)
            values = ti_data[finite_roi_mask]
            finite_roi_voxels = int(values.size)
            nonfinite_roi_voxels = roi_voxels - finite_roi_voxels
            if finite_roi_voxels == 0:
                raise RuntimeError(
                    f"{subject}/{condition.name}/{tag}: optimizer ROI has no "
                    "finite field values"
                )
            summaries = {
                "roi_min_v_per_m": float(np.min(values)),
                "roi_mean_v_per_m": float(np.mean(values)),
                "roi_median_v_per_m": float(np.median(values)),
                "roi_p95_v_per_m": float(np.percentile(values, 95.0)),
                "roi_max_v_per_m": float(np.max(values)),
            }
            if not all(math.isfinite(value) for value in summaries.values()):
                raise RuntimeError(
                    f"{subject}/{condition.name}/{tag}: invalid ROI summaries"
                )
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "subject": subject,
                    "condition": condition.name,
                    "repeat_tag": tag,
                    "roi": spec["roi"],
                    "roi_class": spec["roi_class"],
                    "anatomical_label_ids": json.dumps(label_ids),
                    "requested_roi_volume_mm3": spec["target_volume_mm3"],
                    "achieved_roi_volume_mm3": roi_metadata["achieved_volume_mm3"],
                    "roi_radius_mm": roi_metadata["radius_mm"],
                    "roi_voxels": roi_voxels,
                    "voxel_volume_mm3": roi_metadata["voxel_volume_mm3"],
                    **summaries,
                    "finite_roi_voxels": finite_roi_voxels,
                    "nonfinite_roi_voxels": nonfinite_roi_voxels,
                    "finite_roi_fraction": finite_roi_voxels / roi_voxels,
                    "ti_path": str(ti_path),
                    "ti_size_bytes": ti_path.stat().st_size,
                    "atlas_path": str(atlas_path),
                }
            )

    expected_rows = sum(int(item.repeat_count) for item in config.conditions)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"{subject}: expected {expected_rows} rows, extracted {len(rows)}"
        )
    subject_csv = output_root / "subjects" / f"{subject}.csv"
    _write_csv_atomic(subject_csv, rows, ROW_FIELDS)
    receipt = {
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "config": str(config.config_path),
        "config_sha256": _sha256_file(config.config_path),
        "source_completion_receipt": scope["receipt"],
        "source_completion_receipt_sha256": scope["receipt_sha256"],
        "subject_index": subject_index,
        "subject": subject,
        "rows": len(rows),
        "optimizer_roi": spec,
        "grid_realizations": len(roi_cache),
        "runs_with_nonfinite_roi_values": sum(
            int(row["nonfinite_roi_voxels"]) > 0 for row in rows
        ),
        "minimum_finite_roi_fraction": min(
            float(row["finite_roi_fraction"]) for row in rows
        ),
        "maximum_nonfinite_roi_voxels": max(
            int(row["nonfinite_roi_voxels"]) for row in rows
        ),
        "subject_csv": str(subject_csv),
        "subject_csv_sha256": _sha256_file(subject_csv),
        "source_outputs_modified": False,
    }
    receipt_path = output_root / "subjects" / f"{subject}.json"
    _write_json_atomic(receipt_path, receipt)
    return receipt


def _validate_rows(*, rows: list[dict[str, str]], config) -> dict[str, object]:
    expected_by_condition = {
        condition.name: int(condition.repeat_count)
        for condition in config.conditions
    }
    expected_total = len(config.subjects) * sum(expected_by_condition.values())
    if len(rows) != expected_total:
        raise RuntimeError(f"Expected {expected_total} rows, collected {len(rows)}")
    keys = [
        (row["subject"], row["condition"], row["repeat_tag"])
        for row in rows
    ]
    if len(set(keys)) != len(keys):
        raise RuntimeError("Duplicate subject/condition/run records")
    finite_fractions: list[float] = []
    nonfinite_counts: list[int] = []
    for subject in config.subjects:
        for condition, expected in expected_by_condition.items():
            selected = [
                row for row in rows
                if row["subject"] == subject and row["condition"] == condition
            ]
            if len(selected) != expected:
                raise RuntimeError(
                    f"{subject}/{condition}: expected {expected}, got {len(selected)}"
                )
            expected_tags = {
                repeat_tag(index) for index in range(1, expected + 1)
            }
            if {row["repeat_tag"] for row in selected} != expected_tags:
                raise RuntimeError(f"{subject}/{condition}: run tag mismatch")
            for row in selected:
                if int(row["schema_version"]) != SCHEMA_VERSION:
                    raise RuntimeError("Optimizer ROI metric schema mismatch")
                for field in (
                    "roi_mean_v_per_m",
                    "roi_median_v_per_m",
                    "achieved_roi_volume_mm3",
                    "roi_radius_mm",
                ):
                    if not math.isfinite(float(row[field])):
                        raise RuntimeError(f"Non-finite {field}")
                roi_voxels = int(row["roi_voxels"])
                finite_roi_voxels = int(row["finite_roi_voxels"])
                nonfinite_roi_voxels = int(row["nonfinite_roi_voxels"])
                finite_roi_fraction = float(row["finite_roi_fraction"])
                if roi_voxels <= 0:
                    raise RuntimeError("Optimizer ROI has no voxels")
                if not 0 < finite_roi_voxels <= roi_voxels:
                    raise RuntimeError("Invalid finite optimizer-ROI voxel count")
                if nonfinite_roi_voxels != roi_voxels - finite_roi_voxels:
                    raise RuntimeError("Invalid non-finite optimizer-ROI voxel count")
                expected_fraction = finite_roi_voxels / roi_voxels
                if not math.isclose(
                    finite_roi_fraction,
                    expected_fraction,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise RuntimeError("Invalid finite optimizer-ROI fraction")
                finite_fractions.append(finite_roi_fraction)
                nonfinite_counts.append(nonfinite_roi_voxels)
    runs_with_nonfinite = sum(value > 0 for value in nonfinite_counts)
    return {
        "rows": len(rows),
        "expected_rows": expected_total,
        "unique_keys": len(set(keys)),
        "all_runs_have_finite_roi_values": True,
        "all_roi_values_finite": runs_with_nonfinite == 0,
        "runs_with_nonfinite_roi_values": runs_with_nonfinite,
        "minimum_finite_roi_fraction": min(finite_fractions),
        "maximum_nonfinite_roi_voxels": max(nonfinite_counts),
        "roi_support_policy": (
            "Scalar summaries use finite optimizer-ROI voxels. A run is invalid "
            "only when the ROI contains no finite field values."
        ),
    }


def collect(
    *,
    config_path: Path,
    output_root: Path,
    archive_path: Path | None,
) -> dict[str, object]:
    config = load_experiment_config(config_path, validate_paths=False)
    scope = _completion_scope(config)
    spec = _roi_spec(config)
    rows: list[dict[str, str]] = []
    subject_files: list[Path] = []
    receipt_files: list[Path] = []
    for subject in config.subjects:
        subject_csv = output_root / "subjects" / f"{subject}.csv"
        receipt_path = output_root / "subjects" / f"{subject}.json"
        if not subject_csv.is_file() or not receipt_path.is_file():
            raise FileNotFoundError(
                f"Missing subject ROI metrics for {subject}: "
                f"{subject_csv} / {receipt_path}"
            )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "complete":
            raise RuntimeError(f"Incomplete subject receipt: {receipt_path}")
        if int(receipt.get("schema_version", -1)) != SCHEMA_VERSION:
            raise RuntimeError(f"Subject receipt schema mismatch: {receipt_path}")
        if receipt.get("subject_csv_sha256") != _sha256_file(subject_csv):
            raise RuntimeError(f"Subject CSV checksum mismatch: {subject_csv}")
        rows.extend(_read_csv(subject_csv))
        subject_files.append(subject_csv)
        receipt_files.append(receipt_path)

    condition_order = [item.name for item in config.conditions]
    rows.sort(
        key=lambda row: (
            config.subjects.index(row["subject"]),
            condition_order.index(row["condition"]),
            int(row["repeat_tag"].rsplit("_", 1)[-1]),
        )
    )
    validation = _validate_rows(rows=rows, config=config)
    combined_csv = output_root / "optimizer_roi_metrics.csv"
    _write_csv_atomic(combined_csv, rows, ROW_FIELDS)
    manifest = {
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "mode": "read_only_optimizer_roi_metric_extraction",
        "config": str(config.config_path),
        "config_sha256": _sha256_file(config.config_path),
        "experiment_root": str(config.experiment_root),
        "output_root": str(output_root),
        "optimizer_roi": spec,
        "subjects": list(config.subjects),
        "subject_count": len(config.subjects),
        "conditions": scope["conditions"],
        "expected_fields": scope["expected_fields"],
        "source_completion_receipt": scope["receipt"],
        "source_completion_receipt_sha256": scope["receipt_sha256"],
        "source_outputs_modified": False,
        "meshing_tasks": 0,
        "fem_tasks": 0,
        "combined_csv": str(combined_csv),
        "combined_csv_sha256": _sha256_file(combined_csv),
        "validation": validation,
    }
    manifest_path = output_root / "manifest.json"
    _write_json_atomic(manifest_path, manifest)
    checksum_paths = [combined_csv, manifest_path, *subject_files, *receipt_files]
    checksum_path = output_root / "checksums.sha256"
    checksum_path.write_text(
        "\n".join(
            f"{_sha256_file(path)}  {path.relative_to(output_root)}"
            for path in checksum_paths
        ) + "\n",
        encoding="utf-8",
    )

    if archive_path is not None:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive_path.with_name(f".{archive_path.name}.tmp-{os.getpid()}")
        with tarfile.open(temporary, "w:gz") as archive:
            for path in [combined_csv, manifest_path, checksum_path]:
                archive.add(path, arcname=path.name)
            for path in [*subject_files, *receipt_files]:
                archive.add(path, arcname=str(path.relative_to(output_root)))
        temporary.replace(archive_path)
        archive_sha256 = _sha256_file(archive_path)
        archive_path.with_suffix(archive_path.suffix + ".sha256").write_text(
            f"{archive_sha256}  {archive_path}\n",
            encoding="utf-8",
        )
        manifest["archive"] = str(archive_path)
        manifest["archive_sha256"] = archive_sha256
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "extract-subject", "collect"):
        child = subparsers.add_parser(name)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--output-root", type=Path, required=True)
        if name == "extract-subject":
            child.add_argument("--subject-index", type=int, required=True)
        if name == "collect":
            child.add_argument("--archive", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if args.command == "preflight":
        payload = preflight(config_path=config_path, output_root=output_root)
    elif args.command == "extract-subject":
        payload = extract_subject(
            config_path=config_path,
            subject_index=args.subject_index,
            output_root=output_root,
        )
    else:
        payload = collect(
            config_path=config_path,
            output_root=output_root,
            archive_path=(
                args.archive.expanduser().resolve() if args.archive else None
            ),
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
