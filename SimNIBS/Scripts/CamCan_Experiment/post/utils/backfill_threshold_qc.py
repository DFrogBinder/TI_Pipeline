#!/usr/bin/env python3
"""Backfill threshold/QC metadata in existing subject_metrics.json files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _support_payload(
    *,
    voxels: int,
    volume_mm3: float | None,
    denominator_voxels: int,
    threshold: float,
    comparator: str,
) -> dict[str, Any]:
    return {
        "threshold": float(threshold),
        "comparator": comparator,
        "voxels": int(voxels),
        "volume_mm3": float(volume_mm3 if volume_mm3 is not None else voxels),
        "percent_of_denominator": (
            float((voxels / denominator_voxels) * 100.0) if denominator_voxels else None
        ),
        "has_voxels": bool(voxels > 0),
        "reason": None if voxels > 0 else "no_voxels_above_threshold",
    }


def _only_overlay_qc_is_blocking(payload: dict[str, Any]) -> bool:
    qc_meta = payload.get("qc_meta")
    if not isinstance(qc_meta, dict):
        return False
    checks = qc_meta.get("error_checks")
    if not isinstance(checks, list):
        return False
    normalized_checks = {str(check) for check in checks}
    return bool(normalized_checks) and normalized_checks <= {"overlays"}


def _extended_complete(payload: dict[str, Any]) -> bool:
    subject_meta = payload.get("subject_metrics_meta")
    if isinstance(subject_meta, dict) and subject_meta.get("extended_metrics_status") == "complete":
        return True
    extended_meta = payload.get("extended_metrics_meta")
    return isinstance(extended_meta, dict) and extended_meta.get("status") == "complete"


def backfill_payload(payload: dict[str, Any], *, default_threshold: float) -> bool:
    changed = False
    whole_brain_voxels = int(payload.get("whole_brain_voxels") or 0)
    voxel_volume_mm3 = float(payload.get("voxel_volume_mm3") or 1.0)
    extended_metrics = payload.get("extended_metrics") if isinstance(payload.get("extended_metrics"), dict) else {}
    threshold = float(extended_metrics.get("focality_threshold_v_per_m") or default_threshold)
    overlay_threshold = float(
        payload.get("threshold_qc", {}).get("overlay_threshold")
        if isinstance(payload.get("threshold_qc"), dict)
        and payload.get("threshold_qc", {}).get("overlay_threshold") is not None
        else default_threshold
    )
    whole_voxels = int(extended_metrics.get("focality_voxels_gt_threshold") or 0)
    whole_volume = extended_metrics.get("focality_volume_mm3_gt_threshold")

    threshold_qc = payload.get("threshold_qc")
    if not isinstance(threshold_qc, dict):
        threshold_qc = {
            "schema_version": 1,
            "metric_threshold": threshold,
            "whole_brain": {},
            "rois": {},
            "backfilled": True,
        }
        payload["threshold_qc"] = threshold_qc
        changed = True
    if threshold_qc.get("metric_threshold") is None:
        threshold_qc["metric_threshold"] = threshold
        changed = True
    if "whole_brain" not in threshold_qc:
        threshold_qc["whole_brain"] = {}
    if threshold_qc.get("overlay_threshold") is None:
        threshold_qc["overlay_threshold"] = overlay_threshold
        changed = True
    if "metric_threshold" not in threshold_qc["whole_brain"]:
        threshold_qc["whole_brain"]["metric_threshold"] = _support_payload(
            voxels=whole_voxels,
            volume_mm3=whole_volume,
            denominator_voxels=whole_brain_voxels,
            threshold=threshold,
            comparator=">=",
        )
        changed = True
    if "overlay_threshold" not in threshold_qc["whole_brain"]:
        threshold_qc["whole_brain"]["overlay_threshold"] = _support_payload(
            voxels=whole_voxels,
            volume_mm3=whole_volume,
            denominator_voxels=whole_brain_voxels,
            threshold=overlay_threshold,
            comparator=">=",
        )
        changed = True

    roi_qc = threshold_qc.setdefault("rois", {})
    rois = payload.get("rois") if isinstance(payload.get("rois"), dict) else {}
    for roi_name, roi_metrics in rois.items():
        if not isinstance(roi_metrics, dict):
            continue
        roi_voxels = int(roi_metrics.get("roi_voxels") or 0)
        roi_threshold_voxels = int(roi_metrics.get("focality_in_roi_voxels_gt_threshold") or 0)
        roi_threshold_volume = roi_metrics.get("focality_in_roi_volume_mm3_gt_threshold")
        roi_payload = roi_qc.setdefault(str(roi_name), {"roi_voxels": roi_voxels})
        if "metric_threshold" not in roi_payload:
            roi_payload["metric_threshold"] = _support_payload(
                voxels=roi_threshold_voxels,
                volume_mm3=roi_threshold_volume,
                denominator_voxels=whole_brain_voxels,
                threshold=threshold,
                comparator=">=",
            )
            if roi_threshold_voxels == 0:
                roi_payload["metric_threshold"]["reason"] = (
                    "no_roi_voxels_above_metric_threshold"
                )
            changed = True
        if "overlay_threshold" not in roi_payload:
            roi_payload["overlay_threshold"] = _support_payload(
                voxels=roi_threshold_voxels,
                volume_mm3=roi_threshold_volume,
                denominator_voxels=whole_brain_voxels,
                threshold=overlay_threshold,
                comparator=">=",
            )
            if roi_threshold_voxels == 0:
                roi_payload["overlay_threshold"]["reason"] = (
                    "no_roi_voxels_at_or_above_overlay_threshold"
                )
            changed = True
        if roi_metrics.get("threshold_qc") != roi_payload:
            roi_metrics["threshold_qc"] = roi_payload
            changed = True

    subject_meta = payload.setdefault("subject_metrics_meta", {})
    overlay_qc_only = _only_overlay_qc_is_blocking(payload)
    if _extended_complete(payload) and overlay_qc_only:
        if subject_meta.get("status") != "complete":
            subject_meta["status"] = "complete"
            changed = True
        if subject_meta.get("blocking_qc_checks") != []:
            subject_meta["blocking_qc_checks"] = []
            changed = True
        if subject_meta.get("nonblocking_qc_checks") != ["overlays"]:
            subject_meta["nonblocking_qc_checks"] = ["overlays"]
            changed = True
    elif subject_meta.get("nonblocking_qc_checks") == ["overlays"]:
        subject_meta["nonblocking_qc_checks"] = []
        changed = True
        if subject_meta.get("blocking_qc_checks") is None:
            subject_meta["blocking_qc_checks"] = []

    return changed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill threshold_qc and overlay-only completion status in subject_metrics.json files."
    )
    parser.add_argument("root", help="Dataset or batch root to scan.")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).expanduser()
    paths = sorted(root.rglob("subject_metrics.json"))
    changed_paths: list[Path] = []

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        if backfill_payload(payload, default_threshold=args.threshold):
            changed_paths.append(path)
            if not args.dry_run:
                path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    action = "Would update" if args.dry_run else "Updated"
    print(f"{action} {len(changed_paths)} / {len(paths)} subject_metrics.json file(s).")
    for path in changed_paths[:20]:
        print(path)
    if len(changed_paths) > 20:
        print(f"... {len(changed_paths) - 20} more")


if __name__ == "__main__":
    main()
