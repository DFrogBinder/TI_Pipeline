from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from repeatability_combined_analysis.settings import (
    AnalysisConfig,
    EXCLUDED_SUBJECT,
    REPLACEMENT_SUBJECT,
    SUBJECTS,
    SourcePaths,
)


BASE9 = tuple(subject for subject in SUBJECTS if subject != REPLACEMENT_SUBJECT)
ORIGINAL10 = (*BASE9, EXCLUDED_SUBJECT)


def metric_frame(
    subjects: tuple[str, ...],
    conditions: tuple[str, ...],
    target: str,
    origin: str,
) -> pd.DataFrame:
    roi = "Left_Hippocampus" if target == "left_hippocampus" else "Right_M1"
    requested = 200.0 if target == "left_hippocampus" else 100.0
    roi_voxels = int(requested) + 1
    rows: list[dict[str, object]] = []
    for subject_index, subject in enumerate(subjects):
        for condition_index, condition in enumerate(conditions):
            for repeat in range(1, 41):
                base = 0.10 + 0.005 * subject_index + 0.01 * condition_index
                value = base + 0.0001 * repeat
                nonfinite = int(
                    target == "left_hippocampus" and repeat % 17 == 0
                )
                finite = roi_voxels - nonfinite
                rows.append(
                    {
                        "schema_version": 2,
                        "subject": subject,
                        "condition": condition,
                        "repeat_tag": f"repeat_{repeat:03d}",
                        "roi": roi,
                        "roi_class": (
                            "subcortical"
                            if target == "left_hippocampus"
                            else "cortical"
                        ),
                        "anatomical_label_ids": "17" if target == "left_hippocampus" else "12129",
                        "requested_roi_volume_mm3": requested,
                        "achieved_roi_volume_mm3": float(roi_voxels),
                        "roi_radius_mm": 3.75 if target == "left_hippocampus" else 4.88,
                        "roi_voxels": roi_voxels,
                        "voxel_volume_mm3": 1.0,
                        "roi_min_v_per_m": value - 0.02,
                        "roi_mean_v_per_m": value + 0.01,
                        "roi_median_v_per_m": value,
                        "roi_p95_v_per_m": value + 0.03,
                        "roi_max_v_per_m": value + 0.04,
                        "finite_roi_voxels": finite,
                        "nonfinite_roi_voxels": nonfinite,
                        "finite_roi_fraction": finite / roi_voxels,
                        "ti_path": f"/{origin}/{subject}/{condition}/repeat_{repeat:03d}.nii.gz",
                        "ti_size_bytes": 1234,
                        "atlas_path": f"/atlases/{subject}.nii.gz",
                        "source_origin": origin,
                    }
                )
    return pd.DataFrame(rows)


def write_receipt(
    path: Path,
    *,
    target: str,
    subjects: tuple[str, ...],
    conditions: tuple[str, ...],
    style: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if style == "v3":
        payload = {
            "status": "complete",
            "scope": {
                "target": target,
                "subjects": list(subjects),
                "subject_count": len(subjects),
                "conditions": {condition: 40 for condition in conditions},
                "repeats_per_condition": 40,
                "stage": "final",
            },
        }
    else:
        payload = {
            "status": "complete",
            "scope": {
                "roi": target.replace("_", "-"),
                "subjects": list(subjects),
                "subject_count": len(subjects),
                "repeats_per_condition": 40,
                "remesh_tasks": 40 * len(subjects) if "remesh" in conditions else 0,
                "fixed_mesh_tasks": 40 * len(subjects) if "fixed_mesh" in conditions else 0,
            },
            "validation": {"status": "complete"},
        }
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def analysis_config(tmp_path: Path) -> AnalysisConfig:
    source_root = tmp_path / "sources"
    tables: dict[str, Path] = {}
    specs = {
        "v3_base9_left": (BASE9, ("remesh", "fixed_mesh"), "left_hippocampus"),
        "v3_base9_right": (BASE9, ("remesh", "fixed_mesh"), "right_m1"),
        "v3_replacement_left": ((REPLACEMENT_SUBJECT,), ("remesh", "fixed_mesh"), "left_hippocampus"),
        "v3_replacement_right": ((REPLACEMENT_SUBJECT,), ("remesh", "fixed_mesh"), "right_m1"),
        "v4_historical_left": (ORIGINAL10, ("remesh", "fixed_mesh"), "left_hippocampus"),
        "v4_historical_right": (ORIGINAL10, ("remesh", "fixed_mesh"), "right_m1"),
        "v4_corrected_fixed_left": (ORIGINAL10, ("fixed_mesh",), "left_hippocampus"),
        "v4_corrected_fixed_right": (ORIGINAL10, ("fixed_mesh",), "right_m1"),
        "v4_replacement_left": ((REPLACEMENT_SUBJECT,), ("remesh", "fixed_mesh"), "left_hippocampus"),
        "v4_replacement_right": ((REPLACEMENT_SUBJECT,), ("remesh", "fixed_mesh"), "right_m1"),
    }
    for name, (subjects, conditions, target) in specs.items():
        path = source_root / name / "optimizer_roi_metrics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        metric_frame(subjects, conditions, target, name).to_csv(path, index=False)
        tables[name] = path

    receipt_specs = (
        ("v3_base9_left", "left_hippocampus", BASE9, ("remesh", "fixed_mesh"), "v3"),
        ("v3_base9_right", "right_m1", BASE9, ("remesh", "fixed_mesh"), "v3"),
        ("v3_replacement_left", "left_hippocampus", (REPLACEMENT_SUBJECT,), ("remesh", "fixed_mesh"), "v3"),
        ("v3_replacement_right", "right_m1", (REPLACEMENT_SUBJECT,), ("remesh", "fixed_mesh"), "v3"),
        ("v4_historical_left", "left_hippocampus", ORIGINAL10, ("remesh",), "v4"),
        ("v4_historical_right", "right_m1", ORIGINAL10, ("remesh",), "v4"),
        ("v4_corrected_fixed_left", "left_hippocampus", ORIGINAL10, ("fixed_mesh",), "v4"),
        ("v4_corrected_fixed_right", "right_m1", ORIGINAL10, ("fixed_mesh",), "v4"),
        ("v4_replacement_left", "left_hippocampus", (REPLACEMENT_SUBJECT,), ("remesh", "fixed_mesh"), "v4"),
        ("v4_replacement_right", "right_m1", (REPLACEMENT_SUBJECT,), ("remesh", "fixed_mesh"), "v4"),
    )
    receipts: list[Path] = []
    for name, target, subjects, conditions, style in receipt_specs:
        path = source_root / name / "_pipeline" / "workflow" / "complete.json"
        write_receipt(
            path,
            target=target,
            subjects=subjects,
            conditions=conditions,
            style=style,
        )
        receipts.append(path)

    return AnalysisConfig(
        output_root=tmp_path / "analysis",
        sources=SourcePaths(
            v3_base9_left_csv=tables["v3_base9_left"],
            v3_base9_right_csv=tables["v3_base9_right"],
            v3_replacement_left_csv=tables["v3_replacement_left"],
            v3_replacement_right_csv=tables["v3_replacement_right"],
            v4_historical_left_csv=tables["v4_historical_left"],
            v4_historical_right_csv=tables["v4_historical_right"],
            v4_corrected_fixed_left_csv=tables["v4_corrected_fixed_left"],
            v4_corrected_fixed_right_csv=tables["v4_corrected_fixed_right"],
            v4_replacement_left_csv=tables["v4_replacement_left"],
            v4_replacement_right_csv=tables["v4_replacement_right"],
            receipts=tuple(receipts),
        ),
    )

