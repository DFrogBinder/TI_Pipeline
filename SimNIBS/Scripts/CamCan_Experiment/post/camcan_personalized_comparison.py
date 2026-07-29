"""All-configuration comparison of generic and personalized CamCan TI montages.

The completed individualized campaign contains a 7-subject x 4-ROI simulation
grid, and the optimization table contains the correct subject-specific montage
for every one of those 28 subject/ROI configurations. This module compares
each personalized Pareto montage with the generic MNI152-derived montage on
the same corrected-v4 subject head across ten independent remeshing repeats.

Metrics are calculated independently for every repeat and only then
arithmetic-mean aggregated within condition.  Repeat 01 in one condition is
not treated as paired with repeat 01 in the other condition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post.camcan_manuscript_analysis import (  # noqa: E402
    ANALYSIS_SCHEMA_VERSION,
    DEFAULT_ROBUST_MAX_PERCENTILE,
    DEFAULT_TOP_PERCENTILE,
    DEFAULT_UPPER_TAIL_FRACTION,
    ROI_ORDER,
    _resolve_atlas,
    _threshold_slug,
    compute_optimizer_matched_metric_bundle,
    manuscript_metric_names,
)
from post.optimizer_target_roi import (  # noqa: E402
    ROI_DEFINITION_SCHEMA_VERSION,
    flatten_roi_metadata,
)
from post.post_functions import roi_masks_on_ti_grid  # noqa: E402
from utils.roi_registry import match_fastsurfer_roi_from_directory  # noqa: E402
from utils.ti_utils import load_ti_as_scalar  # noqa: E402


COMPARISON_SCHEMA_VERSION = 3
CONDITIONS = ("generic", "personalized")
REPEATS = tuple(f"{number:02d}" for number in range(1, 11))
EXPECTED_SUBJECTS = 7
EXPECTED_CONFIGURATIONS = EXPECTED_SUBJECTS * len(ROI_ORDER)
DEFAULT_THRESHOLDS = (0.20, 0.18, 0.15)
GENERIC_TARGET_BY_ROI = {
    "Left_Hippocampus": "Left_Hippocampus",
    "Left_M1": "ctx_lh_G_precentral",
    "Right_DLPC": "ctx_rh_G_front_middle",
    "Right_Thalamus": "Right_Thalamus",
}
MAIN_METRICS = (
    "roi_mean_v_per_m",
    "roi_median_v_per_m",
    "roi_robust_max_p99_9_v_per_m",
    "roi_upper_1_percent_median_v_per_m",
    "top_5_percent_target_coverage_percent",
    "top_5_percent_localization_percent_in_roi",
    "target_coverage_percent_ge_0p20",
    "off_target_coverage_percent_ge_0p20",
    "threshold_localization_percent_in_roi_ge_0p20",
    "target_coverage_percent_ge_0p18",
    "off_target_coverage_percent_ge_0p18",
    "threshold_localization_percent_in_roi_ge_0p18",
    "target_coverage_percent_ge_0p15",
    "off_target_coverage_percent_ge_0p15",
    "threshold_localization_percent_in_roi_ge_0p15",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_ready(dict(payload)), indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _row_by_key(
    rows: Sequence[Mapping[str, str]],
    *,
    key: str,
    value: str,
    description: str,
) -> Mapping[str, str]:
    matches = [row for row in rows if row.get(key, "").strip() == value]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {description} row with {key}={value!r}; "
            f"found {len(matches)}."
        )
    return matches[0]


def _float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def build_allowlist(
    *,
    cohort_config: Path,
    individualized_targets_csv: Path,
    generic_targets_csv: Path,
) -> pd.DataFrame:
    cohort = json.loads(cohort_config.read_text(encoding="utf-8"))
    selection = cohort.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("Cohort config has no selection mapping.")
    if tuple(selection) != ROI_ORDER:
        raise ValueError(
            f"Selection ROI order/set does not match the planned ROIs: {ROI_ORDER}."
        )

    individualized_rows = _read_csv(individualized_targets_csv)
    generic_rows = _read_csv(generic_targets_csv)
    records: list[dict[str, Any]] = []
    selected_subjects = {
        str(subject).strip()
        for roles in selection.values()
        for subject in roles.values()
    }
    if len(selected_subjects) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_SUBJECTS} unique optimized subjects; "
            f"found {len(selected_subjects)}."
        )

    pair_index = 0
    for roi in ROI_ORDER:
        roles = selection.get(roi)
        if not isinstance(roles, Mapping) or set(roles) != {"best", "worst"}:
            raise ValueError(f"{roi} must define exactly best and worst subjects.")
        generic_target = GENERIC_TARGET_BY_ROI[roi]
        generic = _row_by_key(
            generic_rows,
            key="roi",
            value=generic_target,
            description="generic target",
        )
        roi_rows = [
            row
            for row in individualized_rows
            if row.get("dataset_roi", "").strip() == roi
            and row.get("subject", "").strip() in selected_subjects
        ]
        if len(roi_rows) != EXPECTED_SUBJECTS:
            raise ValueError(
                f"Expected {EXPECTED_SUBJECTS} individualized rows for {roi}; "
                f"found {len(roi_rows)}."
            )
        if len({row["subject"].strip() for row in roi_rows}) != EXPECTED_SUBJECTS:
            raise ValueError(f"Individualized rows for {roi} contain duplicate subjects.")
        for personalized in sorted(
            roi_rows, key=lambda row: row.get("subject", "").strip()
        ):
            subject = personalized["subject"].strip()
            if subject == str(roles["best"]).strip():
                role = "best"
            elif subject == str(roles["worst"]).strip():
                role = "worst"
            else:
                role = "cross_target"
            recorded_roles = {
                value.strip()
                for value in personalized.get("cohort_selection_role", "").split(";")
                if value.strip()
            }
            if role in {"best", "worst"} and f"{role}_{roi}" not in recorded_roles:
                raise ValueError(
                    f"{subject}/{roi} is not tagged {role}_{roi} in the "
                    "individualized target table."
                )
            records.append(
                {
                    "pair_index": pair_index,
                    "roi": roi,
                    "selection_role": role,
                    "subject": subject,
                    "generic_target_roi": generic_target,
                    "generic_pair1": generic["pair1"].strip(),
                    "generic_pair2": generic["pair2"].strip(),
                    "generic_current1_ma": _float(
                        generic["current1"], "generic current1"
                    ),
                    "generic_current2_ma": _float(
                        generic["current2"], "generic current2"
                    ),
                    "generic_configuration": int(generic["configuration"]),
                    "generic_optimization_e_target_v_per_m": _float(
                        generic["E_target"], "generic E_target"
                    ),
                    "generic_optimization_stimulated_volume": _float(
                        generic["stimulated_volume"], "generic stimulated_volume"
                    ),
                    "personalized_target_roi": personalized["roi"].strip(),
                    "personalized_pair1": personalized["pair1"].strip(),
                    "personalized_pair2": personalized["pair2"].strip(),
                    "personalized_current1_ma": _float(
                        personalized["current1"], "personalized current1"
                    ),
                    "personalized_current2_ma": _float(
                        personalized["current2"], "personalized current2"
                    ),
                    "personalized_configuration": int(personalized["configuration"]),
                    "personalized_optimization_e_target_v_per_m": _float(
                        personalized["E_target"], "personalized E_target"
                    ),
                    "personalized_optimization_stimulated_volume": _float(
                        personalized["stimulated_volume"],
                        "personalized stimulated_volume",
                    ),
                    "personalized_pareto_selection": personalized[
                        "pareto_selection"
                    ].strip(),
                    "personalized_source_target_id": personalized[
                        "source_target_id"
                    ].strip(),
                    "personalized_source_mat": personalized["source_mat"].strip(),
                    "personalized_source_mat_sha256": personalized[
                        "source_mat_sha256"
                    ].strip(),
                }
            )
            pair_index += 1
    frame = pd.DataFrame(records)
    if (
        len(frame) != EXPECTED_CONFIGURATIONS
        or frame.duplicated(["subject", "roi"]).any()
    ):
        raise ValueError(
            f"Analysis allowlist must contain {EXPECTED_CONFIGURATIONS} unique "
            "subject/ROI configurations."
        )
    if frame.groupby("roi").size().to_dict() != {
        roi: EXPECTED_SUBJECTS for roi in ROI_ORDER
    }:
        raise ValueError(
            f"Analysis allowlist must contain {EXPECTED_SUBJECTS} subjects per ROI."
        )
    return frame


def _ti_path(study_root: Path, roi: str, repeat: str, subject: str) -> Path:
    return (
        study_root
        / "runs"
        / f"{roi}_Runs"
        / f"{roi}_Data_{repeat}"
        / subject
        / "anat"
        / "SimNIBS"
        / "ti_brain_only.nii.gz"
    )


def _marker_path(study_root: Path, roi: str, repeat: str, subject: str) -> Path:
    return study_root / "results" / "simulations" / roi / repeat / f"{subject}.json"


def _load_marker(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable simulation marker: {path}") from exc
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        raise ValueError(f"Simulation marker is not complete: {path}")
    return payload


def _assert_close(actual: Any, expected: Any, label: str) -> None:
    if not math.isclose(
        _float(actual, label),
        _float(expected, label),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def validate_source_record(
    *,
    condition: str,
    pair: Mapping[str, Any],
    repeat: str,
    study_root: Path,
    targets_sha256: str,
    individualized_targets_sha256: str,
) -> dict[str, Any]:
    subject = str(pair["subject"])
    roi = str(pair["roi"])
    ti_path = _ti_path(study_root, roi, repeat, subject)
    marker_path = _marker_path(study_root, roi, repeat, subject)
    if not ti_path.is_file() or ti_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing whole-brain TI field: {ti_path}")
    marker = _load_marker(marker_path)
    expected_identity = {
        "subject": subject,
        "roi": roi,
        "repeat_id": repeat,
        "targets_csv_sha256": targets_sha256,
    }
    for key, expected in expected_identity.items():
        if marker.get(key) != expected:
            raise ValueError(
                f"{condition} marker {key} mismatch for {subject}/{roi}/{repeat}: "
                f"{marker.get(key)!r} != {expected!r}"
            )
    if not marker.get("mesh_sha256") or not marker.get("corrected_label_sha256"):
        raise ValueError(f"Incomplete mesh/label provenance in {marker_path}")

    if condition == "generic":
        if marker.get("individualized_targets_csv_sha256") is not None:
            raise ValueError(
                f"Generic marker unexpectedly uses individualized targets: {marker_path}"
            )
    elif condition == "personalized":
        if (
            marker.get("individualized_targets_csv_sha256")
            != individualized_targets_sha256
        ):
            raise ValueError(
                f"Personalized target-table hash mismatch in {marker_path}"
            )
        if marker.get("pareto_selection") != "TI_free.Emin":
            raise ValueError(f"Unexpected Pareto selection in {marker_path}")
        expected_fields = {
            "optimized_configuration": pair["personalized_configuration"],
            "optimized_pair1": pair["personalized_pair1"],
            "optimized_pair2": pair["personalized_pair2"],
        }
        for key, expected in expected_fields.items():
            if marker.get(key) != expected:
                raise ValueError(
                    f"Personalized marker {key} mismatch in {marker_path}: "
                    f"{marker.get(key)!r} != {expected!r}"
                )
        numeric_fields = {
            "optimized_e_target_v_per_m": pair[
                "personalized_optimization_e_target_v_per_m"
            ],
            "optimized_stimulated_volume": pair[
                "personalized_optimization_stimulated_volume"
            ],
            "optimized_current1_ma": pair["personalized_current1_ma"],
            "optimized_current2_ma": pair["personalized_current2_ma"],
        }
        for key, expected in numeric_fields.items():
            _assert_close(marker.get(key), expected, f"{key} in {marker_path}")
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return {
        "ti_path": str(ti_path.resolve()),
        "marker_path": str(marker_path.resolve()),
        "mesh_sha256": marker["mesh_sha256"],
        "corrected_label_sha256": marker["corrected_label_sha256"],
    }


def prepare_analysis(
    *,
    cohort_config: Path,
    individualized_targets_csv: Path,
    generic_targets_csv: Path,
    generic_study_root: Path,
    personalized_study_root: Path,
    atlas_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    allowlist = build_allowlist(
        cohort_config=cohort_config,
        individualized_targets_csv=individualized_targets_csv,
        generic_targets_csv=generic_targets_csv,
    )
    cohort = json.loads(cohort_config.read_text(encoding="utf-8"))
    expected_individualized_hash = str(cohort["individualized_targets_csv_sha256"])
    actual_individualized_hash = sha256_file(individualized_targets_csv)
    if actual_individualized_hash != expected_individualized_hash:
        raise ValueError(
            "Individualized target-table hash does not match cohort config."
        )
    expected_generic_hash = (
        "97a8c7a72faf88d9af9e4facbdf628fba1a130d327da778bcbd00af66f2916e6"
    )
    actual_generic_hash = sha256_file(generic_targets_csv)
    if actual_generic_hash != expected_generic_hash:
        raise ValueError(
            "Generic targets.csv hash is not the recorded manuscript hash."
        )

    atlas_subjects = sorted(set(allowlist["subject"]))
    for subject in atlas_subjects:
        _resolve_atlas(atlas_root, subject)

    source_rows: list[dict[str, Any]] = []
    for pair in allowlist.to_dict(orient="records"):
        for repeat in REPEATS:
            generic = validate_source_record(
                condition="generic",
                pair=pair,
                repeat=repeat,
                study_root=generic_study_root,
                targets_sha256=actual_generic_hash,
                individualized_targets_sha256=actual_individualized_hash,
            )
            personalized = validate_source_record(
                condition="personalized",
                pair=pair,
                repeat=repeat,
                study_root=personalized_study_root,
                targets_sha256=actual_generic_hash,
                individualized_targets_sha256=actual_individualized_hash,
            )
            if (
                generic["corrected_label_sha256"]
                != personalized["corrected_label_sha256"]
            ):
                raise ValueError(
                    "Paired conditions do not use the same corrected label for "
                    f"{pair['subject']}/{pair['roi']}/{repeat}."
                )
            source_rows.extend(
                [
                    {
                        "pair_index": pair["pair_index"],
                        "subject": pair["subject"],
                        "roi": pair["roi"],
                        "selection_role": pair["selection_role"],
                        "condition": "generic",
                        "repeat": repeat,
                        **generic,
                    },
                    {
                        "pair_index": pair["pair_index"],
                        "subject": pair["subject"],
                        "roi": pair["roi"],
                        "selection_role": pair["selection_role"],
                        "condition": "personalized",
                        "repeat": repeat,
                        **personalized,
                    },
                ]
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    allowlist_path = out_dir / "selection_allowlist.csv"
    sources_path = out_dir / "validated_source_records.csv"
    allowlist.to_csv(allowlist_path, index=False)
    pd.DataFrame(source_rows).to_csv(sources_path, index=False)
    payload = {
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "status": "ready",
        "comparison": (
            "MNI152-derived generic montage versus subject-personalized Pareto "
            "montage on the same corrected-v4 subject head."
        ),
        "subject_roi_configurations": len(allowlist),
        "originally_selected_extreme_configurations": int(
            allowlist["selection_role"].isin(["best", "worst"]).sum()
        ),
        "cross_target_configurations": int(
            allowlist["selection_role"].eq("cross_target").sum()
        ),
        "unique_subjects": len(atlas_subjects),
        "conditions": list(CONDITIONS),
        "repeats_per_condition": len(REPEATS),
        "required_repeat_level_inputs": len(source_rows),
        "required_generic_inputs": sum(
            row["condition"] == "generic" for row in source_rows
        ),
        "required_personalized_inputs": sum(
            row["condition"] == "personalized" for row in source_rows
        ),
        "personalized_simulations_in_scope": len(allowlist) * len(REPEATS),
        "out_of_scope_personalized_simulations": 0,
        "roi_definition": (
            "MakeROIs.m-equivalent parcel-clipped sphere centred on the "
            "anatomical parcel volume centroid"
        ),
        "generic_targets_csv_sha256": actual_generic_hash,
        "individualized_targets_csv_sha256": actual_individualized_hash,
        "selection_allowlist": str(allowlist_path.resolve()),
        "selection_allowlist_sha256": sha256_file(allowlist_path),
        "validated_source_records": str(sources_path.resolve()),
        "validated_source_records_sha256": sha256_file(sources_path),
    }
    _atomic_json(out_dir / "preflight.json", payload)
    return payload


@dataclass(frozen=True)
class RepeatTask:
    condition: str
    pair: dict[str, Any]
    repeat: str
    study_root: str
    atlas_root: str
    output_path: str
    thresholds: tuple[float, ...]
    top_percentile: float
    robust_max_percentile: float
    upper_tail_fraction: float
    targets_sha256: str
    individualized_targets_sha256: str
    force: bool


def _identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _repeat_fingerprint(
    task: RepeatTask, source: Mapping[str, Any], atlas: Path
) -> str:
    payload = {
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "manuscript_analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "roi_definition_schema_version": ROI_DEFINITION_SCHEMA_VERSION,
        "condition": task.condition,
        "pair_index": int(task.pair["pair_index"]),
        "subject": task.pair["subject"],
        "roi": task.pair["roi"],
        "selection_role": task.pair["selection_role"],
        "repeat": task.repeat,
        "ti": _identity(Path(source["ti_path"])),
        "atlas": _identity(atlas),
        "marker_path": source["marker_path"],
        "mesh_sha256": source["mesh_sha256"],
        "corrected_label_sha256": source["corrected_label_sha256"],
        "thresholds": list(task.thresholds),
        "top_percentile": task.top_percentile,
        "robust_max_percentile": task.robust_max_percentile,
        "upper_tail_fraction": task.upper_tail_fraction,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _extract_repeat(task: RepeatTask) -> dict[str, Any]:
    source = validate_source_record(
        condition=task.condition,
        pair=task.pair,
        repeat=task.repeat,
        study_root=Path(task.study_root),
        targets_sha256=task.targets_sha256,
        individualized_targets_sha256=task.individualized_targets_sha256,
    )
    atlas = _resolve_atlas(Path(task.atlas_root), str(task.pair["subject"]))
    fingerprint = _repeat_fingerprint(task, source, atlas)
    output_path = Path(task.output_path)
    if not task.force and output_path.is_file():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if (
            isinstance(existing, Mapping)
            and existing.get("status") == "complete"
            and existing.get("config_fingerprint") == fingerprint
        ):
            return {"status": "skipped", "output": str(output_path)}

    ti_path = Path(source["ti_path"])
    ti_img = nib.load(str(ti_path))
    ti_data = load_ti_as_scalar(ti_img)
    canonical_roi = match_fastsurfer_roi_from_directory(
        f"{task.pair['roi']}_Data_{task.repeat}"
    ).canonical_name
    roi_masks, _ = roi_masks_on_ti_grid(
        ti_img,
        atlas_mode="fastsurfer",
        subject=str(task.pair["subject"]),
        fastsurfer_atlas_path=str(atlas),
        roi_names=[canonical_roi],
    )
    roi_mask = roi_masks.get(canonical_roi)
    if roi_mask is None and len(roi_masks) == 1:
        roi_mask = next(iter(roi_masks.values()))
    if roi_mask is None:
        raise ValueError(
            f"ROI {canonical_roi!r} was not returned for {task.pair['subject']}."
        )
    metrics, roi_definition = compute_optimizer_matched_metric_bundle(
        ti_img=ti_img,
        ti_data=ti_data,
        anatomical_roi_mask=roi_mask,
        roi=str(task.pair["roi"]),
        thresholds=task.thresholds,
        top_percentile=task.top_percentile,
        robust_max_percentile=task.robust_max_percentile,
        upper_tail_fraction=task.upper_tail_fraction,
    )
    payload = {
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "manuscript_analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "complete",
        "config_fingerprint": fingerprint,
        "pair_index": int(task.pair["pair_index"]),
        "subject": task.pair["subject"],
        "roi": task.pair["roi"],
        "canonical_roi": canonical_roi,
        "selection_role": task.pair["selection_role"],
        "condition": task.condition,
        "repeat": task.repeat,
        "source": source,
        "roi_definition": roi_definition,
        "metrics": metrics,
    }
    _atomic_json(output_path, payload)
    return {"status": "complete", "output": str(output_path)}


def extract_pair(
    *,
    pair_index: int,
    allowlist_path: Path,
    generic_study_root: Path,
    personalized_study_root: Path,
    atlas_root: Path,
    output_root: Path,
    thresholds: Sequence[float],
    top_percentile: float,
    robust_max_percentile: float,
    upper_tail_fraction: float,
    targets_sha256: str,
    individualized_targets_sha256: str,
    workers: int,
    force: bool,
) -> dict[str, Any]:
    allowlist = pd.read_csv(allowlist_path)
    selected = allowlist.loc[allowlist["pair_index"] == pair_index]
    if len(selected) != 1:
        raise ValueError(
            f"Allowlist does not contain exactly one pair index {pair_index}."
        )
    pair = selected.iloc[0].to_dict()
    tasks: list[RepeatTask] = []
    for condition, study_root in (
        ("generic", generic_study_root),
        ("personalized", personalized_study_root),
    ):
        for repeat in REPEATS:
            output_path = (
                output_root
                / "repeat_records"
                / condition
                / str(pair["roi"])
                / str(pair["subject"])
                / f"repeat_{repeat}.json"
            )
            tasks.append(
                RepeatTask(
                    condition=condition,
                    pair=pair,
                    repeat=repeat,
                    study_root=str(study_root),
                    atlas_root=str(atlas_root),
                    output_path=str(output_path),
                    thresholds=tuple(float(value) for value in thresholds),
                    top_percentile=float(top_percentile),
                    robust_max_percentile=float(robust_max_percentile),
                    upper_tail_fraction=float(upper_tail_fraction),
                    targets_sha256=targets_sha256,
                    individualized_targets_sha256=individualized_targets_sha256,
                    force=force,
                )
            )
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with ProcessPoolExecutor(
        max_workers=min(max(1, workers), len(tasks)),
        mp_context=get_context("spawn"),
    ) as pool:
        future_map = {pool.submit(_extract_repeat, task): task for task in tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append(
                    {
                        "condition": task.condition,
                        "repeat": task.repeat,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    payload = {
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "status": "complete" if not errors and len(results) == 20 else "incomplete",
        "pair_index": pair_index,
        "subject": pair["subject"],
        "roi": pair["roi"],
        "selection_role": pair["selection_role"],
        "records": len(results),
        "computed": sum(row["status"] == "complete" for row in results),
        "skipped": sum(row["status"] == "skipped" for row in results),
        "errors": errors,
    }
    summary = output_root / "pair_summaries" / f"pair_{pair_index:02d}.json"
    _atomic_json(summary, payload)
    if payload["status"] != "complete":
        raise RuntimeError(f"Pair {pair_index} extraction failed; see {summary}.")
    return payload


def _flatten_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Repeat record has no metrics mapping.")
    roi_definition = payload.get("roi_definition")
    if not isinstance(roi_definition, Mapping):
        raise ValueError("Repeat record has no optimizer ROI-definition mapping.")
    return {
        "pair_index": int(payload["pair_index"]),
        "subject": payload["subject"],
        "roi": payload["roi"],
        "canonical_roi": payload["canonical_roi"],
        "selection_role": payload["selection_role"],
        "condition": payload["condition"],
        "repeat": str(payload["repeat"]).zfill(2),
        "config_fingerprint": payload["config_fingerprint"],
        "source_ti_path": payload["source"]["ti_path"],
        "source_marker_path": payload["source"]["marker_path"],
        "source_mesh_sha256": payload["source"]["mesh_sha256"],
        "corrected_label_sha256": payload["source"]["corrected_label_sha256"],
        **flatten_roi_metadata(dict(roi_definition)),
        **metrics,
    }


def _metric_label(metric: str) -> tuple[str, str]:
    if metric.startswith("anatomical_"):
        label, unit = _metric_label(metric.removeprefix("anatomical_"))
        return (f"Full anatomical parcel: {label}", unit)
    fixed = {
        "roi_min_v_per_m": ("Minimum target-ROI TI field", "V/m"),
        "roi_mean_v_per_m": ("Mean target-ROI TI field", "V/m"),
        "roi_median_v_per_m": ("Median target-ROI TI field", "V/m"),
        "roi_robust_max_p99_9_v_per_m": (
            "Target-ROI robust maximum (P99.9)",
            "V/m",
        ),
        "roi_upper_1_percent_median_v_per_m": (
            "Median of upper 1% target-ROI TI field",
            "V/m",
        ),
        "top_5_percent_target_coverage_percent": (
            "Target coverage by whole-brain top 5% field",
            "%",
        ),
        "top_5_percent_localization_percent_in_roi": (
            "Localization of whole-brain top 5% field in target",
            "%",
        ),
    }
    if metric in fixed:
        return fixed[metric]
    for threshold in DEFAULT_THRESHOLDS:
        slug = _threshold_slug(threshold)
        if metric == f"target_coverage_percent_ge_{slug}":
            return (f"Target coverage ≥ {threshold:.2f} V/m", "%")
        if metric == f"off_target_coverage_percent_ge_{slug}":
            return (f"Off-target coverage ≥ {threshold:.2f} V/m", "%")
        if metric == f"whole_brain_coverage_percent_ge_{slug}":
            return (f"Whole-brain coverage ≥ {threshold:.2f} V/m", "%")
        if metric == f"threshold_localization_percent_in_roi_ge_{slug}":
            return (f"Suprathreshold localization in target ≥ {threshold:.2f} V/m", "%")
    return (metric, "")


def _write_paired_dumbbell(condition_frame: pd.DataFrame, output_base: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plot_metrics = (
        "roi_mean_v_per_m",
        "roi_median_v_per_m",
        "target_coverage_percent_ge_0p2",
        "off_target_coverage_percent_ge_0p2",
    )
    pair_count = int(condition_frame["pair_index"].nunique())
    fig_height = max(8.0, 4.5 + 0.27 * pair_count)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, fig_height))
    labels = [
        f"{row.roi.replace('_', ' ')} – {row.subject.replace('sub-', '')}"
        for row in condition_frame.drop_duplicates(
            ["pair_index", "roi", "selection_role"]
        )
        .sort_values("pair_index")
        .itertuples()
    ]
    y = np.arange(len(labels))
    for axis_index, (axis, metric) in enumerate(zip(axes.flat, plot_metrics)):
        generic = (
            condition_frame.loc[condition_frame["condition"] == "generic"]
            .sort_values("pair_index")[metric]
            .to_numpy()
        )
        personalized = (
            condition_frame.loc[condition_frame["condition"] == "personalized"]
            .sort_values("pair_index")[metric]
            .to_numpy()
        )
        for index, (left, right) in enumerate(zip(generic, personalized)):
            axis.plot([left, right], [index, index], color="#A6A6A6", linewidth=1.2)
        axis.scatter(generic, y, color="#2878B5", s=38, zorder=3)
        axis.scatter(personalized, y, color="#D55E00", s=38, zorder=3)
        label, unit = _metric_label(metric)
        axis.set_title(label)
        axis.set_xlabel(unit)
        axis.set_yticks(y, labels if axis_index % 2 == 0 else [], fontsize=7)
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.6)
        axis.invert_yaxis()
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color="#2878B5",
                label="Generic MNI152-derived montage",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color="#D55E00",
                label="Subject-personalized montage",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.947),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Generic versus personalized TI stimulation on the same subject heads",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_effectiveness_trajectories(
    condition_frame: pd.DataFrame,
    *,
    threshold: float,
    output_base: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    slug = _threshold_slug(threshold)
    x_metric = f"target_coverage_percent_ge_{slug}"
    y_metric = f"off_target_coverage_percent_ge_{slug}"
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2), sharex=True, sharey=True)
    subjects = sorted(condition_frame["subject"].unique())
    palette = plt.get_cmap("tab10")
    subject_colors = {
        subject: palette(index % 10) for index, subject in enumerate(subjects)
    }
    for axis, roi in zip(axes, ROI_ORDER):
        rows = condition_frame.loc[condition_frame["roi"] == roi]
        for pair_index in sorted(rows["pair_index"].unique()):
            pair_rows = rows.loc[rows["pair_index"] == pair_index].set_index(
                "condition"
            )
            generic = pair_rows.loc["generic"]
            personalized = pair_rows.loc["personalized"]
            subject = str(generic["subject"])
            color = subject_colors[subject]
            axis.plot(
                [generic[x_metric], personalized[x_metric]],
                [generic[y_metric], personalized[y_metric]],
                color=color,
                lw=1.2,
                alpha=0.72,
                zorder=2,
            )
            axis.scatter(
                generic[x_metric],
                generic[y_metric],
                marker="o",
                facecolor="white",
                edgecolor=color,
                s=55,
                linewidth=1.5,
                zorder=3,
            )
            axis.scatter(
                personalized[x_metric],
                personalized[y_metric],
                marker="o",
                facecolor=color,
                edgecolor=color,
                s=55,
                zorder=4,
            )
        axis.set_title(roi.replace("_", " "))
        axis.set_xlabel(
            f"Target coverage ≥ {threshold:.2f} V/m (%)"
        )
        axis.grid(True, color="#D9D9D9", linewidth=0.6)
    axes[0].set_ylabel(f"Off-target coverage ≥ {threshold:.2f} V/m (%)")
    fig.legend(
        handles=[
            *[
                Line2D([0], [0], color=subject_colors[subject], lw=2, label=subject)
                for subject in subjects
            ],
            Line2D(
                [0],
                [0],
                marker="o",
                markerfacecolor="white",
                markeredgecolor="#555555",
                linestyle="none",
                label="Generic montage",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                markerfacecolor="#555555",
                markeredgecolor="#555555",
                linestyle="none",
                label="Personalized montage",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=5,
        frameon=False,
        fontsize=7.5,
    )
    fig.suptitle(
        "Change in effectiveness–spread balance after personalization", y=0.995
    )
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_repeat_distribution(repeat_frame: pd.DataFrame, output_base: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subjects = sorted(repeat_frame["subject"].unique())
    fig, axes = plt.subplots(
        len(subjects),
        len(ROI_ORDER),
        figsize=(15.5, 3.0 * len(subjects)),
        sharey=False,
        squeeze=False,
    )
    rng = np.random.default_rng(20260728)
    for row_index, subject in enumerate(subjects):
        for column_index, roi in enumerate(ROI_ORDER):
            axis = axes[row_index, column_index]
            rows = repeat_frame.loc[
                (repeat_frame["subject"] == subject)
                & (repeat_frame["roi"] == roi)
            ]
            if rows.empty:
                axis.set_visible(False)
                continue
            pair = rows.iloc[0]
            values = [
                rows.loc[
                    rows["condition"] == condition, "roi_mean_v_per_m"
                ].to_numpy()
                for condition in CONDITIONS
            ]
            box = axis.boxplot(
                values,
                tick_labels=["Generic", "Personalized"],
                widths=0.55,
                patch_artist=True,
                showfliers=False,
            )
            for patch, color in zip(box["boxes"], ("#A6CEE3", "#FDBF6F")):
                patch.set_facecolor(color)
            for x, condition_values in enumerate(values, start=1):
                jitter = rng.uniform(-0.055, 0.055, len(condition_values))
                axis.scatter(
                    x + jitter,
                    condition_values,
                    s=14,
                    color="#333333",
                    alpha=0.65,
                    zorder=3,
                )
            axis.set_title(
                f"{roi.replace('_', ' ')}\n{subject.replace('sub-', '')}",
                fontsize=8.2,
            )
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
            if column_index == 0:
                axis.set_ylabel("Mean target-ROI field (V/m)")
    fig.suptitle(
        "Technical-repeat distributions for all 28 optimized subject–ROI configurations"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def collect_analysis(
    *,
    allowlist_path: Path,
    output_root: Path,
    thresholds: Sequence[float],
    top_percentile: float,
    robust_max_percentile: float,
    upper_tail_fraction: float,
) -> dict[str, Any]:
    allowlist = pd.read_csv(allowlist_path)
    pair_count = len(allowlist)
    if pair_count != EXPECTED_CONFIGURATIONS:
        raise RuntimeError(
            f"Collector requires exactly {EXPECTED_CONFIGURATIONS} subject/ROI "
            f"configurations; found {pair_count}."
        )
    pair_indices = sorted(int(value) for value in allowlist["pair_index"].unique())
    if pair_indices != list(range(pair_count)):
        raise RuntimeError("Allowlist pair indices must be contiguous from zero.")
    summary_rows: list[dict[str, Any]] = []
    for pair_index in pair_indices:
        summary_path = output_root / "pair_summaries" / f"pair_{pair_index:02d}.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Missing pair summary: {summary_path}") from exc
        if (
            summary.get("status") != "complete"
            or summary.get("pair_index") != pair_index
            or summary.get("records") != 20
        ):
            raise RuntimeError(f"Invalid pair summary: {summary_path}")
        summary_rows.append(summary)
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for pair in allowlist.to_dict(orient="records"):
        for condition in CONDITIONS:
            for repeat in REPEATS:
                path = (
                    output_root
                    / "repeat_records"
                    / condition
                    / str(pair["roi"])
                    / str(pair["subject"])
                    / f"repeat_{repeat}.json"
                )
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    missing.append(str(path))
                    continue
                if payload.get("status") != "complete":
                    missing.append(str(path))
                    continue
                expected = {
                    "pair_index": int(pair["pair_index"]),
                    "subject": pair["subject"],
                    "roi": pair["roi"],
                    "selection_role": pair["selection_role"],
                    "condition": condition,
                    "repeat": repeat,
                }
                if any(payload.get(key) != value for key, value in expected.items()):
                    raise RuntimeError(f"Repeat record identity mismatch: {path}")
                records.append(_flatten_record(payload))
    expected_records = pair_count * len(CONDITIONS) * len(REPEATS)
    if missing or len(records) != expected_records:
        raise RuntimeError(
            f"Expected {expected_records} repeat records; found {len(records)} with "
            f"{len(missing)} missing/incomplete."
        )
    repeat_frame = pd.DataFrame(records).sort_values(
        ["pair_index", "condition", "repeat"]
    )
    if repeat_frame.duplicated(["pair_index", "condition", "repeat"]).any():
        raise RuntimeError("Duplicate repeat records were collected.")
    expected_product = {
        (pair, condition, repeat)
        for pair in pair_indices
        for condition in CONDITIONS
        for repeat in REPEATS
    }
    actual_product = set(
        zip(
            repeat_frame["pair_index"],
            repeat_frame["condition"],
            repeat_frame["repeat"],
        )
    )
    if actual_product != expected_product:
        raise RuntimeError(
            "Collected records do not match the exact allowlisted product."
        )

    metric_columns = manuscript_metric_names(thresholds)
    missing_metrics = set(metric_columns).difference(repeat_frame.columns)
    if missing_metrics:
        raise RuntimeError(f"Missing metrics: {sorted(missing_metrics)}")
    finite = np.isfinite(repeat_frame[metric_columns].to_numpy(dtype=float, copy=False))
    if not finite.all():
        bad = repeat_frame[metric_columns].columns[~finite.all(axis=0)].tolist()
        raise RuntimeError(f"Non-finite repeat metrics: {bad}")

    group = [
        "pair_index",
        "subject",
        "roi",
        "canonical_roi",
        "selection_role",
        "condition",
    ]
    roi_definition_columns = [
        column
        for column in repeat_frame.columns
        if column.startswith("optimizer_roi_")
        and pd.api.types.is_numeric_dtype(repeat_frame[column])
    ]
    means = (
        repeat_frame.groupby(group, sort=False)[
            metric_columns + roi_definition_columns
        ]
        .mean()
        .reset_index()
    )
    sds = (
        repeat_frame.groupby(group, sort=False)[metric_columns]
        .std(ddof=1)
        .add_suffix("__repeat_sd")
        .reset_index()
    )
    counts = (
        repeat_frame.groupby(group, sort=False)
        .size()
        .rename("repeat_count")
        .reset_index()
    )
    condition_frame = means.merge(sds, on=group).merge(counts, on=group)
    expected_condition_means = pair_count * len(CONDITIONS)
    if (
        len(condition_frame) != expected_condition_means
        or not condition_frame["repeat_count"].eq(len(REPEATS)).all()
    ):
        raise RuntimeError(
            f"Expected {expected_condition_means} condition means with "
            f"{len(REPEATS)} repeats each."
        )

    comparison_rows: list[dict[str, Any]] = []
    for pair in allowlist.to_dict(orient="records"):
        pair_rows = condition_frame.loc[
            condition_frame["pair_index"] == pair["pair_index"]
        ].set_index("condition")
        if set(pair_rows.index) != set(CONDITIONS):
            raise RuntimeError(
                f"Pair {pair['pair_index']} lacks one comparison condition."
            )
        for metric in metric_columns:
            generic = float(pair_rows.loc["generic", metric])
            personalized = float(pair_rows.loc["personalized", metric])
            generic_sd = float(pair_rows.loc["generic", f"{metric}__repeat_sd"])
            personalized_sd = float(
                pair_rows.loc["personalized", f"{metric}__repeat_sd"]
            )
            label, unit = _metric_label(metric)
            comparison_rows.append(
                {
                    "pair_index": int(pair["pair_index"]),
                    "subject": pair["subject"],
                    "roi": pair["roi"],
                    "selection_role": pair["selection_role"],
                    "metric": metric,
                    "metric_label": label,
                    "unit": unit,
                    "generic_repeat_mean": generic,
                    "generic_repeat_sd": generic_sd,
                    "personalized_repeat_mean": personalized,
                    "personalized_repeat_sd": personalized_sd,
                    "absolute_change_personalized_minus_generic": personalized
                    - generic,
                    "percent_change_from_generic": (
                        (personalized - generic) / generic * 100.0
                        if generic != 0.0
                        else math.nan
                    ),
                }
            )
    comparison_long = pd.DataFrame(comparison_rows)
    comparison_wide_rows: list[dict[str, Any]] = []
    for pair in allowlist.to_dict(orient="records"):
        row: dict[str, Any] = {
            "pair_index": int(pair["pair_index"]),
            "subject": pair["subject"],
            "roi": pair["roi"],
            "selection_role": pair["selection_role"],
        }
        metric_rows = comparison_long.loc[
            comparison_long["pair_index"] == pair["pair_index"]
        ]
        for metric_row in metric_rows.itertuples(index=False):
            prefix = metric_row.metric
            row[f"{prefix}__generic_repeat_mean"] = metric_row.generic_repeat_mean
            row[f"{prefix}__generic_repeat_sd"] = metric_row.generic_repeat_sd
            row[f"{prefix}__personalized_repeat_mean"] = (
                metric_row.personalized_repeat_mean
            )
            row[f"{prefix}__personalized_repeat_sd"] = metric_row.personalized_repeat_sd
            row[f"{prefix}__absolute_change"] = (
                metric_row.absolute_change_personalized_minus_generic
            )
            row[f"{prefix}__percent_change"] = metric_row.percent_change_from_generic
        comparison_wide_rows.append(row)
    comparison_wide = pd.DataFrame(comparison_wide_rows)
    selected_long = comparison_long.loc[
        comparison_long["metric"].isin(MAIN_METRICS)
    ].copy()
    selected_long["generic_mean_sd"] = selected_long.apply(
        lambda row: f"{row['generic_repeat_mean']:.3f} ± {row['generic_repeat_sd']:.3f}",
        axis=1,
    )
    selected_long["personalized_mean_sd"] = selected_long.apply(
        lambda row: (
            f"{row['personalized_repeat_mean']:.3f} ± "
            f"{row['personalized_repeat_sd']:.3f}"
        ),
        axis=1,
    )
    parameters = allowlist.copy()
    parameters.insert(
        len(parameters.columns),
        "comparison_definition",
        "same subject head: generic MNI152-derived montage vs personalized montage",
    )
    dictionary = pd.DataFrame(
        [
            {
                "metric": metric,
                "metric_label": _metric_label(metric)[0],
                "unit": _metric_label(metric)[1],
                "main_table": metric in MAIN_METRICS,
                "repeat_aggregation": (
                    "calculated per repeat, then arithmetic mean and sample SD "
                    "within condition"
                ),
            }
            for metric in metric_columns
        ]
    )

    results_dir = output_root / "results"
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    repeat_frame.to_csv(results_dir / "repeat_level_metrics.csv", index=False)
    condition_frame.to_csv(
        results_dir / "condition_repeat_mean_metrics.csv", index=False
    )
    condition_frame[
        group + ["repeat_count"] + roi_definition_columns
    ].to_csv(results_dir / "table_optimizer_roi_definitions.csv", index=False)
    comparison_long.to_csv(
        results_dir / "paired_personalized_vs_generic_long.csv", index=False
    )
    comparison_wide.to_csv(
        results_dir / "paired_personalized_vs_generic.csv", index=False
    )
    selected_long.to_csv(results_dir / "table_main_selected_metrics.csv", index=False)
    parameters.to_csv(results_dir / "table_stimulation_parameters.csv", index=False)
    dictionary.to_csv(results_dir / "metric_dictionary.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(
        results_dir / "pair_extraction_summaries.csv", index=False
    )
    allowlist.to_csv(results_dir / "selection_allowlist.csv", index=False)
    validated_sources = output_root / "validated_source_records.csv"
    if validated_sources.is_file():
        pd.read_csv(validated_sources).to_csv(
            results_dir / "validated_source_records.csv", index=False
        )

    _write_paired_dumbbell(
        condition_frame,
        figures_dir / "paired_generic_vs_personalized_dumbbell",
    )
    for threshold in thresholds:
        _write_effectiveness_trajectories(
            condition_frame,
            threshold=float(threshold),
            output_base=figures_dir
            / f"effectiveness_off_target_trajectories_ge_{_threshold_slug(float(threshold))}",
        )
    _write_repeat_distribution(
        repeat_frame,
        figures_dir / "technical_repeat_roi_mean_distributions",
    )

    manifest = {
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "manuscript_analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "complete",
        "comparison": (
            "MNI152-derived generic montage versus subject-personalized Pareto "
            "montage, both simulated on the same corrected-v4 subject head."
        ),
        "subject_roi_configurations": pair_count,
        "originally_selected_extreme_configurations": int(
            allowlist["selection_role"].isin(["best", "worst"]).sum()
        ),
        "cross_target_configurations": int(
            allowlist["selection_role"].eq("cross_target").sum()
        ),
        "unique_subjects": int(allowlist["subject"].nunique()),
        "conditions": list(CONDITIONS),
        "repeats_per_pair_condition": 10,
        "repeat_level_records": len(repeat_frame),
        "condition_repeat_mean_records": len(condition_frame),
        "paired_metric_records": len(comparison_long),
        "thresholds_v_per_m": [float(value) for value in thresholds],
        "top_percentile": float(top_percentile),
        "robust_max_percentile": float(robust_max_percentile),
        "upper_tail_fraction": float(upper_tail_fraction),
        "aggregation": (
            "Metrics calculated per repeat, then arithmetic-mean aggregated "
            "within each condition. Repeat numbers are not paired between conditions."
        ),
        "inference": (
            "Descriptive analysis of seven subjects originally selected as "
            "outcome extremes; no population inference."
        ),
        "personalized_simulations_included": pair_count * len(REPEATS),
        "excluded_personalized_simulations": 0,
        "primary_roi_definition": (
            "MakeROIs.m-equivalent parcel-clipped sphere centred on the "
            "anatomical parcel volume centroid"
        ),
        "secondary_roi_definition": (
            "full anatomical atlas parcel; metrics carry the anatomical_ prefix"
        ),
        "historical_extreme_selection_metric": (
            "full-anatomical-parcel roi_median_v_per_m averaged after "
            "per-repeat calculation in the generic final-132 cohort analysis"
        ),
        "current_optimizer_objective_metric": "roi_mean_v_per_m",
        "outputs": [],
    }
    preflight_path = output_root / "preflight.json"
    if preflight_path.is_file():
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        manifest["source_provenance"] = {
            key: preflight[key]
            for key in (
                "generic_targets_csv_sha256",
                "individualized_targets_csv_sha256",
                "selection_allowlist_sha256",
                "validated_source_records_sha256",
            )
        }
    manifest["outputs"] = sorted(
        str(path.relative_to(results_dir))
        for path in results_dir.rglob("*")
        if path.is_file() and path.name != "analysis_manifest.json"
    )
    _atomic_json(results_dir / "analysis_manifest.json", manifest)
    return manifest


def _parse_thresholds(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(not math.isfinite(item) or item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Thresholds must be positive finite numbers.")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("Thresholds must not contain duplicates.")
    return values


def _common_metric_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--thresholds", type=_parse_thresholds, default=DEFAULT_THRESHOLDS
    )
    parser.add_argument("--top-percentile", type=float, default=DEFAULT_TOP_PERCENTILE)
    parser.add_argument(
        "--robust-max-percentile",
        type=float,
        default=DEFAULT_ROBUST_MAX_PERCENTILE,
    )
    parser.add_argument(
        "--upper-tail-fraction",
        type=float,
        default=DEFAULT_UPPER_TAIL_FRACTION,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--cohort-config", type=Path, required=True)
    prepare.add_argument("--individualized-targets", type=Path, required=True)
    prepare.add_argument("--generic-targets", type=Path, required=True)
    prepare.add_argument("--generic-study-root", type=Path, required=True)
    prepare.add_argument("--personalized-study-root", type=Path, required=True)
    prepare.add_argument("--atlas-root", type=Path, required=True)
    prepare.add_argument("--out-dir", type=Path, required=True)

    extract = commands.add_parser("extract-pair")
    extract.add_argument("--pair-index", type=int, required=True)
    extract.add_argument("--allowlist", type=Path, required=True)
    extract.add_argument("--generic-study-root", type=Path, required=True)
    extract.add_argument("--personalized-study-root", type=Path, required=True)
    extract.add_argument("--atlas-root", type=Path, required=True)
    extract.add_argument("--output-root", type=Path, required=True)
    extract.add_argument("--targets-sha256", required=True)
    extract.add_argument("--individualized-targets-sha256", required=True)
    extract.add_argument("--workers", type=int, default=4)
    extract.add_argument("--force", action="store_true")
    _common_metric_arguments(extract)

    collect = commands.add_parser("collect")
    collect.add_argument("--allowlist", type=Path, required=True)
    collect.add_argument("--output-root", type=Path, required=True)
    _common_metric_arguments(collect)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_analysis(
            cohort_config=args.cohort_config,
            individualized_targets_csv=args.individualized_targets,
            generic_targets_csv=args.generic_targets,
            generic_study_root=args.generic_study_root,
            personalized_study_root=args.personalized_study_root,
            atlas_root=args.atlas_root,
            out_dir=args.out_dir,
        )
    elif args.command == "extract-pair":
        allowlist = pd.read_csv(args.allowlist)
        valid_pair_indices = {
            int(value) for value in allowlist["pair_index"].tolist()
        }
        if args.pair_index not in valid_pair_indices:
            raise ValueError(
                f"pair-index {args.pair_index} is not present in {args.allowlist}."
            )
        result = extract_pair(
            pair_index=args.pair_index,
            allowlist_path=args.allowlist,
            generic_study_root=args.generic_study_root,
            personalized_study_root=args.personalized_study_root,
            atlas_root=args.atlas_root,
            output_root=args.output_root,
            thresholds=args.thresholds,
            top_percentile=args.top_percentile,
            robust_max_percentile=args.robust_max_percentile,
            upper_tail_fraction=args.upper_tail_fraction,
            targets_sha256=args.targets_sha256,
            individualized_targets_sha256=args.individualized_targets_sha256,
            workers=args.workers,
            force=args.force,
        )
    else:
        result = collect_analysis(
            allowlist_path=args.allowlist,
            output_root=args.output_root,
            thresholds=args.thresholds,
            top_percentile=args.top_percentile,
            robust_max_percentile=args.robust_max_percentile,
            upper_tail_fraction=args.upper_tail_fraction,
        )
    print(json.dumps(_json_ready(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
