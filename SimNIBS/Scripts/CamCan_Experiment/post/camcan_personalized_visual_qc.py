"""Full-post and visual QC for the eight selected personalized CamCan cases.

The individualized FEM campaign deliberately contains a 7-subject x 4-ROI
Cartesian grid.  Only eight subject/ROI pairs are scientifically in scope:
the best and worst generic-montage case selected for each of four ROIs.

This module runs the existing full subject-level post-processing implementation
for exactly those eight pairs under both conditions (generic and personalized)
and ten independently remeshed repeats.  Outputs are written to an isolated QC
tree; source simulations and their existing post-processing directories are
read-only.

In addition to the legacy single-field products, each repeat receives
side-by-side generic-versus-personalized figures with identical cut coordinates
and a common colour scale.  Repeat identifiers are aligned for convenient
visual inspection only and are not treated as statistically paired.
"""

from __future__ import annotations

import argparse
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

from post.camcan_personalized_comparison import (  # noqa: E402
    CONDITIONS,
    REPEATS,
    _identity,
    _json_ready,
    _resolve_atlas,
    _threshold_slug,
    validate_source_record,
)
from post.post_functions import roi_masks_on_ti_grid  # noqa: E402
from post.post_process import PostProcessConfig, run_post_process  # noqa: E402
from utils.roi_registry import match_fastsurfer_roi_from_directory  # noqa: E402
from utils.ti_utils import load_ti_as_scalar, normalize_roi_name  # noqa: E402


VISUAL_QC_SCHEMA_VERSION = 1
EXPECTED_PAIR_COUNT = 8
EXPECTED_POST_RECORDS = 160
EXPECTED_PAIRED_PNGS = 160
EXPECTED_PAIR_REPORTS = 16
SIMNIBS_CAP_TEMPLATE = (
    "{root}/{subject}/anat/m2m_{subject}/eeg_positions/"
    "EEG10-10_UI_Jurak_2007.csv"
)
BASELINE_DIR_BY_ROI = {
    "Left_Hippocampus": "MNI152-left-hippocampus",
    "Left_M1": "MNI152-left-m1",
    "Right_DLPC": "MNI152-right-dlpc",
    "Right_Thalamus": "MNI152-right-thalamus",
}
PAIR_MODES = (
    ("full_field", None),
    ("above_0p20", 0.20),
)
EXPECTED_EMPTY_THRESHOLD_OVERLAY_TYPES = frozenset(
    {"context_threshold", "roi_focus_threshold"}
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_ready(dict(payload)), indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_single_pair(allowlist_path: Path, pair_index: int) -> dict[str, Any]:
    allowlist = pd.read_csv(allowlist_path)
    if len(allowlist) != EXPECTED_PAIR_COUNT:
        raise ValueError(
            f"Visual QC requires exactly {EXPECTED_PAIR_COUNT} allowlisted pairs."
        )
    selected = allowlist.loc[allowlist["pair_index"] == pair_index]
    if len(selected) != 1:
        raise ValueError(
            f"Allowlist does not contain exactly one pair index {pair_index}."
        )
    return selected.iloc[0].to_dict()


def _electrode_names(pair: Mapping[str, Any], condition: str) -> tuple[str, ...]:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    prefix = "generic" if condition == "generic" else "personalized"
    names: list[str] = []
    for pair_number in (1, 2):
        value = str(pair[f"{prefix}_pair{pair_number}"]).strip()
        components = tuple(item.strip() for item in value.split("-") if item.strip())
        if len(components) != 2:
            raise ValueError(
                f"{prefix}_pair{pair_number} is not an electrode pair: {value!r}"
            )
        names.extend(components)
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError(
            f"{condition} montage must contain four unique electrodes; got {names}."
        )
    return tuple(names)


def _source_dataset_root(ti_path: Path, subject: str) -> Path:
    expected_tail = Path(subject) / "anat" / "SimNIBS" / "ti_brain_only.nii.gz"
    if Path(*ti_path.parts[-4:]) != expected_tail:
        raise ValueError(f"Unexpected TI source layout: {ti_path}")
    return ti_path.parents[3]


def _source_t1(dataset_root: Path, subject: str) -> Path:
    base = dataset_root / subject / "anat" / f"{subject}_T1w.nii"
    if base.is_file() and base.stat().st_size > 0:
        return base
    compressed = base.with_name(f"{base.name}.gz")
    if compressed.is_file() and compressed.stat().st_size > 0:
        return compressed
    raise FileNotFoundError(f"Missing subject T1 for visual QC: {base}[.gz]")


def _canonical_roi(pair: Mapping[str, Any]) -> str:
    return match_fastsurfer_roi_from_directory(
        f"{pair['roi']}_Data_01"
    ).canonical_name


def _post_record_path(
    output_root: Path,
    pair_index: int,
    condition: str,
    repeat: str,
) -> Path:
    return (
        output_root
        / "post_records"
        / f"pair_{pair_index:02d}"
        / condition
        / f"repeat_{repeat}.json"
    )


def _post_output_dir(
    output_root: Path,
    pair_index: int,
    condition: str,
    repeat: str,
) -> Path:
    return (
        output_root
        / "post_products"
        / f"pair_{pair_index:02d}"
        / condition
        / f"repeat_{repeat}"
    )


@dataclass(frozen=True)
class FullPostTask:
    condition: str
    pair: dict[str, Any]
    repeat: str
    study_root: str
    atlas_root: str
    mni_fixed_atlas: str
    mni_baseline_parent: str
    output_root: str
    targets_sha256: str
    individualized_targets_sha256: str
    force: bool


def _full_post_fingerprint(
    *,
    task: FullPostTask,
    source: Mapping[str, Any],
    atlas: Path,
    t1_path: Path,
) -> str:
    return _sha256_json(
        {
            "visual_qc_schema_version": VISUAL_QC_SCHEMA_VERSION,
            "condition": task.condition,
            "pair_index": int(task.pair["pair_index"]),
            "subject": task.pair["subject"],
            "roi": task.pair["roi"],
            "selection_role": task.pair["selection_role"],
            "repeat": task.repeat,
            "ti": _identity(Path(source["ti_path"])),
            "atlas": _identity(atlas),
            "t1": _identity(t1_path),
            "source_marker_path": source["marker_path"],
            "source_mesh_sha256": source["mesh_sha256"],
            "corrected_label_sha256": source["corrected_label_sha256"],
            "mni_fixed_atlas": _identity(Path(task.mni_fixed_atlas)),
            "mni_baseline_parent": str(
                Path(task.mni_baseline_parent).expanduser().resolve()
            ),
            "electrode_names": list(_electrode_names(task.pair, task.condition)),
            "percentile": 95.0,
            "hard_threshold_v_per_m": 0.20,
            "off_target_threshold_v_per_m": 0.20,
            "overlay_full_field": True,
        }
    )


def _complete_existing_record(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, Mapping)
        or payload.get("status") != "complete"
        or payload.get("config_fingerprint") != fingerprint
    ):
        return None
    required_paths = payload.get("required_output_paths")
    if not isinstance(required_paths, list) or not required_paths:
        return None
    if any(not Path(str(item)).is_file() for item in required_paths):
        return None
    return dict(payload)


def _run_full_post_task(task: FullPostTask) -> dict[str, Any]:
    pair_index = int(task.pair["pair_index"])
    subject = str(task.pair["subject"])
    roi = str(task.pair["roi"])
    study_root = Path(task.study_root)
    source = validate_source_record(
        condition=task.condition,
        pair=task.pair,
        repeat=task.repeat,
        study_root=study_root,
        targets_sha256=task.targets_sha256,
        individualized_targets_sha256=task.individualized_targets_sha256,
    )
    ti_path = Path(source["ti_path"])
    dataset_root = _source_dataset_root(ti_path, subject)
    t1_path = _source_t1(dataset_root, subject)
    atlas = _resolve_atlas(Path(task.atlas_root), subject)
    record_path = _post_record_path(
        Path(task.output_root),
        pair_index,
        task.condition,
        task.repeat,
    )
    output_dir = _post_output_dir(
        Path(task.output_root),
        pair_index,
        task.condition,
        task.repeat,
    )
    fingerprint = _full_post_fingerprint(
        task=task,
        source=source,
        atlas=atlas,
        t1_path=t1_path,
    )
    if not task.force:
        existing = _complete_existing_record(record_path, fingerprint)
        if existing is not None:
            return {
                "status": "skipped",
                "record": str(record_path),
                "condition": task.condition,
                "repeat": task.repeat,
            }

    baseline_dir = BASELINE_DIR_BY_ROI.get(roi)
    if baseline_dir is None:
        raise ValueError(f"No MNI152 baseline mapping for ROI: {roi}")
    baseline_root = Path(task.mni_baseline_parent) / baseline_dir
    if not baseline_root.is_dir():
        raise FileNotFoundError(f"Missing MNI152 baseline root: {baseline_root}")

    canonical_roi = _canonical_roi(task.pair)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = PostProcessConfig(
        root_dir=str(dataset_root),
        subject=subject,
        ti_path=str(ti_path),
        t1_path=str(t1_path),
        atlas_mode="fastsurfer",
        fastsurfer_root=str(Path(task.atlas_root)),
        fs_mri_path=str(atlas),
        out_dir=str(output_dir),
        plot_roi=canonical_roi,
        percentile=95.0,
        hard_threshold=0.20,
        overlay_z_offset_mm=0.0,
        overlay_full_field=True,
        write_region_table=True,
        region_percentile=95.0,
        offtarget_threshold=0.20,
        mni_baseline_root=str(baseline_root),
        mni_fixed_atlas_path=task.mni_fixed_atlas,
        neighbor_dilation_iter=1,
        csf_labels=[24],
        skull_labels=None,
        electrode_names=_electrode_names(task.pair, task.condition),
        eeg_positions_path_template=SIMNIBS_CAP_TEMPLATE,
        write_neighbor_table=True,
        write_neighbor_visualization=True,
        write_electrode_table=True,
        verbose=False,
    )
    result = run_post_process(cfg)
    metrics_path = Path(str(result["metrics_path"]))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    subject_meta = metrics.get("subject_metrics_meta", {})
    overlay_paths = [
        Path(str(path))
        for paths in result.get("overlays", {}).values()
        for path in paths
    ]
    if subject_meta.get("status") != "complete":
        raise RuntimeError(
            f"Full post-processing was not complete for "
            f"{subject}/{roi}/{task.condition}/{task.repeat}: {subject_meta}"
        )
    empty_threshold_placeholders: list[Path] = []
    if len(overlay_paths) != 7:
        empty_threshold_placeholders = _write_expected_empty_threshold_overlays(
            metrics=metrics,
            output_dir=output_dir,
            subject=subject,
            canonical_roi=canonical_roi,
            ti_path=ti_path,
            t1_path=t1_path,
            atlas_path=atlas,
            existing_overlay_paths=overlay_paths,
        )
        overlay_paths.extend(empty_threshold_placeholders)
    required_paths = [metrics_path, *overlay_paths]
    if len(overlay_paths) != 7 or any(not path.is_file() for path in overlay_paths):
        raise RuntimeError(
            f"Expected seven legacy E-field overlays for "
            f"{subject}/{roi}/{task.condition}/{task.repeat}; "
            f"found {len(overlay_paths)}."
        )
    payload = {
        "visual_qc_schema_version": VISUAL_QC_SCHEMA_VERSION,
        "status": "complete",
        "config_fingerprint": fingerprint,
        "pair_index": pair_index,
        "subject": subject,
        "roi": roi,
        "canonical_roi": canonical_roi,
        "selection_role": task.pair["selection_role"],
        "condition": task.condition,
        "repeat": task.repeat,
        "source": source,
        "source_dataset_root": str(dataset_root.resolve()),
        "source_t1_path": str(t1_path.resolve()),
        "atlas_path": str(atlas.resolve()),
        "electrode_names": list(_electrode_names(task.pair, task.condition)),
        "post_output_dir": str(output_dir.resolve()),
        "subject_metrics_path": str(metrics_path.resolve()),
        "subject_metrics_status": subject_meta.get("status"),
        "legacy_overlay_paths": [str(path.resolve()) for path in overlay_paths],
        "empty_threshold_placeholder_paths": [
            str(path.resolve()) for path in empty_threshold_placeholders
        ],
        "empty_threshold_placeholder_policy": (
            "When the whole brain contains zero finite voxels at or above "
            "0.20 V/m, the two mathematically empty threshold overlays are "
            "represented by explicit annotated anatomy/ROI panels. Zero "
            "support is verified from threshold QC metadata when available "
            "and otherwise by a direct recount of the source TI image."
        ),
        "required_output_paths": [str(path.resolve()) for path in required_paths],
    }
    _atomic_json(record_path, payload)
    return {
        "status": "complete",
        "record": str(record_path),
        "condition": task.condition,
        "repeat": task.repeat,
    }


def _load_post_record(
    output_root: Path,
    pair_index: int,
    condition: str,
    repeat: str,
) -> dict[str, Any]:
    path = _post_record_path(output_root, pair_index, condition, repeat)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Missing full-post QC record: {path}") from exc
    expected = {
        "status": "complete",
        "pair_index": pair_index,
        "condition": condition,
        "repeat": repeat,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Invalid full-post QC record: {path}")
    return payload


def _pair_common_vmax(
    records: Sequence[Mapping[str, Any]],
    percentile: float = 99.9,
) -> float:
    robust_maxima: list[float] = []
    for record in records:
        image = nib.load(str(record["source"]["ti_path"]))
        data = np.asarray(load_ti_as_scalar(image), dtype=float)
        values = data[np.isfinite(data) & (data > 0)]
        if values.size == 0:
            raise ValueError(f"TI field has no finite positive values: {record}")
        robust_maxima.append(float(np.percentile(values, percentile)))
    vmax = max(robust_maxima)
    if not math.isfinite(vmax) or vmax <= 0:
        raise ValueError(f"Invalid pair-wide robust maximum: {vmax}")
    return vmax


def _roi_mask_image(
    ti_img: nib.Nifti1Image,
    *,
    atlas_path: Path,
    subject: str,
    canonical_roi: str,
) -> nib.Nifti1Image:
    masks, _ = roi_masks_on_ti_grid(
        ti_img,
        atlas_mode="fastsurfer",
        subject=subject,
        fastsurfer_atlas_path=str(atlas_path),
        roi_names=[canonical_roi],
    )
    mask = masks.get(canonical_roi)
    if mask is None and len(masks) == 1:
        mask = next(iter(masks.values()))
    if mask is None or not np.any(mask):
        raise ValueError(f"Target ROI mask is empty for {subject}/{canonical_roi}.")
    return nib.Nifti1Image(
        np.asarray(mask, dtype=np.uint8),
        ti_img.affine,
        ti_img.header,
    )


def _roi_center_world(roi_img: nib.Nifti1Image) -> tuple[float, float, float]:
    mask = np.asarray(roi_img.dataobj) > 0
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("Target ROI mask is empty.")
    center = nib.affines.apply_affine(roi_img.affine, coordinates.mean(axis=0))
    return tuple(float(value) for value in center)


def _write_expected_empty_threshold_overlays(
    *,
    metrics: Mapping[str, Any],
    output_dir: Path,
    subject: str,
    canonical_roi: str,
    ti_path: Path,
    t1_path: Path,
    atlas_path: Path,
    existing_overlay_paths: Sequence[Path] = (),
) -> list[Path]:
    """Represent a valid zero-support threshold result without inventing data.

    Nilearn cannot render an overlay/colorbar when the selected field contains
    no voxels.  The standard full-post pipeline therefore omits the context and
    ROI-focus threshold PNGs.  This is a valid scientific result, not a missing
    simulation.  Only that exact, provenance-backed case is repaired here.
    """

    overlay_qc = (
        metrics.get("qc_meta", {})
        .get("checks", {})
        .get("overlays", {})
    )
    missing_types = frozenset(overlay_qc.get("missing_overlay_types", []))
    threshold_support = (
        metrics.get("threshold_qc", {})
        .get("whole_brain", {})
        .get("overlay_threshold", {})
    )
    reported_threshold = float(
        threshold_support.get("threshold", float("nan"))
    )
    threshold = reported_threshold if math.isfinite(reported_threshold) else 0.20
    metadata_proves_expected_empty = (
        missing_types == EXPECTED_EMPTY_THRESHOLD_OVERLAY_TYPES
        and threshold_support.get("has_voxels") is False
        and int(threshold_support.get("voxels", -1)) == 0
        and math.isclose(threshold, 0.20, rel_tol=0.0, abs_tol=1e-12)
    )

    roi_stub = normalize_roi_name(canonical_roi)
    paths = [
        output_dir
        / f"{roi_stub}_TI_overlay_context_{subject}_above{threshold:.2f}.png",
        output_dir
        / f"{roi_stub}_TI_overlay_roi_focus_{subject}_above{threshold:.2f}.png",
    ]
    existing_resolved = {
        str(Path(path).expanduser().resolve()) for path in existing_overlay_paths
    }
    expected_threshold_paths_are_missing = all(
        str(path.expanduser().resolve()) not in existing_resolved for path in paths
    )
    direct_recount_proves_expected_empty = False
    if (
        len(existing_overlay_paths) == 5
        and expected_threshold_paths_are_missing
        and math.isclose(threshold, 0.20, rel_tol=0.0, abs_tol=1e-12)
    ):
        ti_img = nib.load(str(ti_path))
        ti_data = np.asarray(load_ti_as_scalar(ti_img), dtype=float)
        direct_support_voxels = int(
            np.count_nonzero(np.isfinite(ti_data) & (ti_data >= threshold))
        )
        direct_recount_proves_expected_empty = direct_support_voxels == 0

    if not (
        metadata_proves_expected_empty
        or direct_recount_proves_expected_empty
    ):
        return []

    panel_labels = ("whole-brain context", "target-ROI focus")
    for path, panel_label in zip(paths, panel_labels):
        if path.is_file() and path.stat().st_size > 0:
            continue
        _render_empty_threshold_overlay(
            ti_path=ti_path,
            t1_path=t1_path,
            atlas_path=atlas_path,
            subject=subject,
            canonical_roi=canonical_roi,
            threshold=threshold,
            panel_label=panel_label,
            output_path=path,
        )
    return paths


def _render_empty_threshold_overlay(
    *,
    ti_path: Path,
    t1_path: Path,
    atlas_path: Path,
    subject: str,
    canonical_roi: str,
    threshold: float,
    panel_label: str,
    output_path: Path,
) -> None:
    """Write an anatomy/ROI panel explicitly documenting zero field support."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn.image import resample_to_img
    from nilearn.plotting import plot_anat

    ti_img = nib.load(str(ti_path))
    roi_img = _roi_mask_image(
        ti_img,
        atlas_path=atlas_path,
        subject=subject,
        canonical_roi=canonical_roi,
    )
    t1_on_ti = resample_to_img(
        nib.load(str(t1_path)),
        ti_img,
        interpolation="continuous",
        force_resample=True,
        copy_header=True,
    )
    center = _roi_center_world(roi_img)
    directions = (("x", center[0]), ("y", center[1]), ("z", center[2]))
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4), facecolor="white")
    for axis, (display_mode, coordinate) in zip(axes, directions):
        display = plot_anat(
            t1_on_ti,
            display_mode=display_mode,
            cut_coords=[coordinate],
            figure=fig,
            axes=axis,
            annotate=True,
            draw_cross=False,
            colorbar=False,
            black_bg=True,
        )
        display.add_contours(
            roi_img,
            levels=[0.5],
            colors=["#FF3B30"],
            linewidths=1.2,
        )
    fig.suptitle(
        f"{subject} | {canonical_roi.replace('-', ' ')} | {panel_label}\n"
        f"No finite whole-brain TI field voxels reached ≥ {threshold:.2f} V/m",
        fontsize=12,
        y=0.98,
    )
    fig.text(
        0.5,
        0.025,
        (
            "Red contour = subject-space target ROI. The absent colour overlay "
            "is a measured zero-coverage result, not missing simulation data."
        ),
        ha="center",
        fontsize=8.5,
        color="#333333",
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.80,
        bottom=0.13,
        wspace=0.03,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _montage_label(pair: Mapping[str, Any], condition: str) -> str:
    prefix = "generic" if condition == "generic" else "personalized"
    pair1 = str(pair[f"{prefix}_pair1"])
    pair2 = str(pair[f"{prefix}_pair2"])
    current1 = float(pair[f"{prefix}_current1_ma"])
    current2 = float(pair[f"{prefix}_current2_ma"])
    return (
        f"{condition.capitalize()}: {pair1} at {current1:.3g} mA; "
        f"{pair2} at {current2:.3g} mA"
    )


def _render_paired_repeat(
    *,
    pair: Mapping[str, Any],
    repeat: str,
    generic_record: Mapping[str, Any],
    personalized_record: Mapping[str, Any],
    vmax: float,
    threshold: float | None,
    output_path: Path,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from nilearn.image import resample_to_img
    from nilearn.plotting import plot_anat

    records = {
        "generic": generic_record,
        "personalized": personalized_record,
    }
    images: dict[
        str,
        tuple[nib.Nifti1Image, nib.Nifti1Image, nib.Nifti1Image, bool],
    ] = {}
    medians: dict[str, float] = {}
    canonical_roi = str(generic_record["canonical_roi"])
    for condition, record in records.items():
        ti_img = nib.load(str(record["source"]["ti_path"]))
        ti_data = np.asarray(load_ti_as_scalar(ti_img), dtype=float)
        atlas_path = Path(str(record["atlas_path"]))
        roi_img = _roi_mask_image(
            ti_img,
            atlas_path=atlas_path,
            subject=str(pair["subject"]),
            canonical_roi=canonical_roi,
        )
        t1_img = nib.load(str(record["source_t1_path"]))
        t1_on_ti = resample_to_img(
            t1_img,
            ti_img,
            interpolation="continuous",
            force_resample=True,
            copy_header=True,
        )
        finite = np.isfinite(ti_data)
        roi = np.asarray(roi_img.dataobj) > 0
        medians[condition] = float(np.median(ti_data[finite & roi]))
        visible = np.where(
            finite & ((ti_data >= threshold) if threshold is not None else (ti_data > 0)),
            ti_data,
            0.0,
        )
        overlay = nib.Nifti1Image(visible, ti_img.affine, ti_img.header)
        images[condition] = (
            t1_on_ti,
            overlay,
            roi_img,
            bool(np.any(visible > 0)),
        )

    center = _roi_center_world(images["generic"][2])
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.2), facecolor="white")
    directions = (("x", center[0]), ("y", center[1]), ("z", center[2]))
    for row, condition in enumerate(CONDITIONS):
        t1_img, overlay_img, roi_img, has_visible_field = images[condition]
        for column, (display_mode, coordinate) in enumerate(directions):
            axis = axes[row, column]
            display = plot_anat(
                t1_img,
                display_mode=display_mode,
                cut_coords=[coordinate],
                figure=fig,
                axes=axis,
                annotate=True,
                draw_cross=False,
                colorbar=False,
                black_bg=True,
            )
            if has_visible_field:
                display.add_overlay(
                    overlay_img,
                    threshold=1e-12,
                    colorbar=False,
                    vmin=0.0,
                    vmax=vmax,
                    cmap="viridis",
                )
            elif column == 1:
                axis.text(
                    0.5,
                    0.08,
                    "No whole-brain voxels meet this threshold",
                    color="white",
                    fontsize=8,
                    ha="center",
                    transform=axis.transAxes,
                    bbox={"facecolor": "black", "alpha": 0.7, "edgecolor": "none"},
                )
            display.add_contours(
                roi_img,
                levels=[0.5],
                colors=["#FF3B30"],
                linewidths=1.0,
            )
            if column == 1:
                axis.set_title(
                    f"{_montage_label(pair, condition)}\n"
                    f"target-ROI median = {medians[condition]:.4f} V/m",
                    fontsize=9.4,
                    pad=8,
                )

    threshold_text = (
        "full positive field"
        if threshold is None
        else f"field at or above {threshold:.2f} V/m"
    )
    fig.suptitle(
        f"{pair['roi'].replace('_', ' ')} | {pair['selection_role']} selected case | "
        f"{pair['subject']} | technical repeat {repeat}\n"
        f"Generic MNI152-derived versus personalized Pareto montage: {threshold_text}",
        fontsize=13,
        y=0.985,
    )
    scalar = ScalarMappable(norm=Normalize(vmin=0.0, vmax=vmax), cmap="viridis")
    scalar.set_array([])
    colorbar = fig.colorbar(
        scalar,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.035,
        pad=0.055,
        aspect=45,
    )
    colorbar.set_label(
        "TI electric-field magnitude (V/m); common scale for all 20 fields in this subject–ROI pair"
    )
    fig.text(
        0.5,
        0.012,
        (
            "Red contour = subject-space target ROI. Rows are independently "
            "remeshed simulations. Matching repeat numbers are aligned only for QC "
            "and are not treated as statistically paired."
        ),
        ha="center",
        fontsize=8.5,
        color="#333333",
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.88,
        bottom=0.12,
        hspace=0.22,
        wspace=0.03,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "path": str(output_path.resolve()),
        "repeat": repeat,
        "mode": "full_field" if threshold is None else f"above_{threshold:.2f}",
        "threshold_v_per_m": threshold,
        "common_vmax_v_per_m": vmax,
        "generic_roi_median_v_per_m": medians["generic"],
        "personalized_roi_median_v_per_m": medians["personalized"],
    }


def _write_multipage_report(paths: Sequence[Path], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if len(paths) != len(REPEATS):
        raise ValueError(
            f"A pair report requires {len(REPEATS)} repeat images; got {len(paths)}."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as document:
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Missing paired visual: {path}")
            image = plt.imread(path)
            fig = plt.figure(figsize=(15.6, 8.2), facecolor="white")
            axis = fig.add_axes((0, 0, 1, 1))
            axis.imshow(image)
            axis.axis("off")
            document.savefig(fig, dpi=150, bbox_inches="tight", pad_inches=0)
            plt.close(fig)


def run_pair_qc(
    *,
    pair_index: int,
    allowlist_path: Path,
    generic_study_root: Path,
    personalized_study_root: Path,
    atlas_root: Path,
    mni_fixed_atlas: Path,
    mni_baseline_parent: Path,
    output_root: Path,
    targets_sha256: str,
    individualized_targets_sha256: str,
    workers: int,
    force: bool,
) -> dict[str, Any]:
    pair = _read_single_pair(allowlist_path, pair_index)
    tasks: list[FullPostTask] = []
    for condition, study_root in (
        ("generic", generic_study_root),
        ("personalized", personalized_study_root),
    ):
        for repeat in REPEATS:
            tasks.append(
                FullPostTask(
                    condition=condition,
                    pair=pair,
                    repeat=repeat,
                    study_root=str(study_root),
                    atlas_root=str(atlas_root),
                    mni_fixed_atlas=str(mni_fixed_atlas),
                    mni_baseline_parent=str(mni_baseline_parent),
                    output_root=str(output_root),
                    targets_sha256=targets_sha256,
                    individualized_targets_sha256=individualized_targets_sha256,
                    force=force,
                )
            )

    task_results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with ProcessPoolExecutor(
        max_workers=min(max(1, workers), len(tasks)),
        mp_context=get_context("spawn"),
    ) as pool:
        future_map = {pool.submit(_run_full_post_task, task): task for task in tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                task_results.append(future.result())
            except Exception as exc:
                errors.append(
                    {
                        "condition": task.condition,
                        "repeat": task.repeat,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    if errors or len(task_results) != 20:
        summary = {
            "visual_qc_schema_version": VISUAL_QC_SCHEMA_VERSION,
            "status": "incomplete",
            "pair_index": pair_index,
            "subject": pair["subject"],
            "roi": pair["roi"],
            "selection_role": pair["selection_role"],
            "post_records": len(task_results),
            "errors": errors,
        }
        summary_path = output_root / "pair_summaries" / f"pair_{pair_index:02d}.json"
        _atomic_json(summary_path, summary)
        raise RuntimeError(f"Full-post pair QC failed; see {summary_path}.")

    records = [
        _load_post_record(output_root, pair_index, condition, repeat)
        for condition in CONDITIONS
        for repeat in REPEATS
    ]
    common_vmax = _pair_common_vmax(records)
    visual_rows: list[dict[str, Any]] = []
    report_paths: list[str] = []
    for mode, threshold in PAIR_MODES:
        mode_paths: list[Path] = []
        for repeat in REPEATS:
            generic = _load_post_record(
                output_root, pair_index, "generic", repeat
            )
            personalized = _load_post_record(
                output_root, pair_index, "personalized", repeat
            )
            output_path = (
                output_root
                / "paired_fields"
                / f"pair_{pair_index:02d}"
                / f"repeat_{repeat}_{mode}.png"
            )
            visual_rows.append(
                _render_paired_repeat(
                    pair=pair,
                    repeat=repeat,
                    generic_record=generic,
                    personalized_record=personalized,
                    vmax=common_vmax,
                    threshold=threshold,
                    output_path=output_path,
                )
            )
            mode_paths.append(output_path)
        report_path = (
            output_root
            / "pair_reports"
            / f"pair_{pair_index:02d}_{mode}_10_repeats.pdf"
        )
        _write_multipage_report(mode_paths, report_path)
        report_paths.append(str(report_path.resolve()))

    visual_index_path = (
        output_root / "pair_summaries" / f"pair_{pair_index:02d}_visual_index.csv"
    )
    visual_index_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(visual_rows).to_csv(visual_index_path, index=False)
    summary = {
        "visual_qc_schema_version": VISUAL_QC_SCHEMA_VERSION,
        "status": "complete",
        "scope": "one allowlisted subject-ROI pair only",
        "pair_index": pair_index,
        "subject": pair["subject"],
        "roi": pair["roi"],
        "selection_role": pair["selection_role"],
        "conditions": list(CONDITIONS),
        "repeats_per_condition": len(REPEATS),
        "post_records": len(records),
        "legacy_overlays": sum(
            len(record["legacy_overlay_paths"]) for record in records
        ),
        "paired_visualizations": len(visual_rows),
        "pair_reports": len(report_paths),
        "common_scale_definition": (
            "maximum of the per-field 99.9th percentiles across the 20 "
            "generic/personalized source fields in this subject-ROI pair"
        ),
        "common_vmax_v_per_m": common_vmax,
        "visual_index": str(visual_index_path.resolve()),
        "report_paths": report_paths,
        "excluded_out_of_scope_personalized_simulations": 200,
        "errors": [],
    }
    summary_path = output_root / "pair_summaries" / f"pair_{pair_index:02d}.json"
    _atomic_json(summary_path, summary)
    return summary


def _qc_readme() -> str:
    return """# Personalized-versus-generic full-post visual QC

This package contains quality-control outputs for exactly eight selected
subject–ROI cases: the best and worst generic-montage case for each of four
target ROIs.  For each case, ten independently remeshed simulations from the
generic MNI152-derived montage are compared with ten simulations from the
subject-personalized Pareto montage.  The other 200 personalized simulations
from the deliberately over-broad 7-subject × 4-ROI campaign are excluded.

## Recommended files for visual review

- `pair_reports/*full_field_10_repeats.pdf`: one page per technical repeat,
  showing generic and personalized full fields on the same anatomical planes.
- `pair_reports/*above_0p20_10_repeats.pdf`: the same comparisons after hiding
  field values below 0.20 V/m.
- `paired_fields/`: the individual PNG pages used to build those reports.

Each subject–ROI pair uses one fixed colour scale across all 20 source fields.
The upper bound is the largest per-field P99.9 value in that pair.  The target
ROI is outlined in red.  Matching repeat numbers are aligned only to make QC
inspection convenient; the independently remeshed conditions are not treated
as statistically paired.

## Full subject-level post-processing

`post_products/` contains the standard subject-level post-processing outputs
for all 160 in-scope fields.  The compact download archive excludes derived
NIfTI files to control archive size, while preserving PNG, PDF, CSV, and JSON
outputs.  Source FEM fields are read-only and are not copied into this package.

The legacy single-field overlays select display scales independently and
therefore should not be used to compare colour intensity between conditions.
Use the common-scale paired figures for that purpose.
"""


def collect_qc(
    *,
    allowlist_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    allowlist = pd.read_csv(allowlist_path)
    if len(allowlist) != EXPECTED_PAIR_COUNT:
        raise RuntimeError(
            f"Collector requires exactly {EXPECTED_PAIR_COUNT} allowlisted pairs."
        )
    summaries: list[dict[str, Any]] = []
    post_rows: list[dict[str, Any]] = []
    visual_frames: list[pd.DataFrame] = []
    expected_product = {
        (pair, condition, repeat)
        for pair in range(EXPECTED_PAIR_COUNT)
        for condition in CONDITIONS
        for repeat in REPEATS
    }
    for pair_index in range(EXPECTED_PAIR_COUNT):
        path = output_root / "pair_summaries" / f"pair_{pair_index:02d}.json"
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Missing pair QC summary: {path}") from exc
        expected = {
            "status": "complete",
            "pair_index": pair_index,
            "post_records": 20,
            "paired_visualizations": 20,
            "pair_reports": 2,
        }
        if any(summary.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Invalid pair QC summary: {path}")
        summaries.append(summary)
        visual_index = Path(str(summary["visual_index"]))
        visual_frame = pd.read_csv(visual_index)
        if len(visual_frame) != 20:
            raise RuntimeError(f"Invalid paired visual index: {visual_index}")
        if any(not Path(str(item)).is_file() for item in visual_frame["path"]):
            raise RuntimeError(f"Missing paired PNG listed in: {visual_index}")
        if any(not Path(str(item)).is_file() for item in summary["report_paths"]):
            raise RuntimeError(f"Missing pair report listed in: {path}")
        visual_frame.insert(0, "pair_index", pair_index)
        visual_frame.insert(1, "subject", summary["subject"])
        visual_frame.insert(2, "roi", summary["roi"])
        visual_frame.insert(3, "selection_role", summary["selection_role"])
        visual_frames.append(visual_frame)
        for condition in CONDITIONS:
            for repeat in REPEATS:
                post_rows.append(
                    _load_post_record(
                        output_root,
                        pair_index,
                        condition,
                        repeat,
                    )
                )
    actual_product = {
        (int(row["pair_index"]), str(row["condition"]), str(row["repeat"]))
        for row in post_rows
    }
    if actual_product != expected_product or len(post_rows) != EXPECTED_POST_RECORDS:
        raise RuntimeError(
            "Collected post records do not match the exact "
            "8-pair x 2-condition x 10-repeat scope."
        )
    visual_frame = pd.concat(visual_frames, ignore_index=True)
    if len(visual_frame) != EXPECTED_PAIRED_PNGS:
        raise RuntimeError(
            f"Expected {EXPECTED_PAIRED_PNGS} paired PNGs; "
            f"found {len(visual_frame)}."
        )
    report_count = sum(int(summary["pair_reports"]) for summary in summaries)
    if report_count != EXPECTED_PAIR_REPORTS:
        raise RuntimeError(
            f"Expected {EXPECTED_PAIR_REPORTS} pair reports; found {report_count}."
        )

    results_dir = output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    visual_frame.insert(
        visual_frame.columns.get_loc("path") + 1,
        "relative_path",
        [
            str(Path(str(path)).resolve().relative_to(output_root.resolve()))
            for path in visual_frame["path"]
        ],
    )
    post_frame = pd.DataFrame(post_rows)
    for source_column, relative_column in (
        ("post_output_dir", "post_output_relative_path"),
        ("subject_metrics_path", "subject_metrics_relative_path"),
    ):
        post_frame[relative_column] = [
            str(Path(str(path)).resolve().relative_to(output_root.resolve()))
            for path in post_frame[source_column]
        ]
    pd.DataFrame(summaries).to_csv(
        results_dir / "pair_qc_summaries.csv",
        index=False,
    )
    post_frame.to_csv(
        results_dir / "full_post_processing_index.csv",
        index=False,
    )
    visual_frame.to_csv(
        results_dir / "paired_visualization_index.csv",
        index=False,
    )
    allowlist.to_csv(results_dir / "selection_allowlist.csv", index=False)
    (results_dir / "README.md").write_text(_qc_readme(), encoding="utf-8")
    manifest = {
        "visual_qc_schema_version": VISUAL_QC_SCHEMA_VERSION,
        "status": "complete",
        "scope": (
            "Exactly eight allowlisted subject-ROI pairs; generic and "
            "personalized conditions; ten repeats per condition."
        ),
        "selected_subject_roi_pairs": EXPECTED_PAIR_COUNT,
        "unique_subjects": int(allowlist["subject"].nunique()),
        "conditions": list(CONDITIONS),
        "repeats_per_pair_condition": len(REPEATS),
        "full_post_records": len(post_rows),
        "paired_visualization_pngs": len(visual_frame),
        "multipage_pair_reports": report_count,
        "legacy_overlay_pngs": sum(
            int(summary["legacy_overlays"]) for summary in summaries
        ),
        "excluded_out_of_scope_personalized_simulations": 200,
        "source_simulations_modified": False,
        "source_post_directories_modified": False,
        "compact_archive_policy": (
            "Include PNG, PDF, CSV, JSON, and Markdown; exclude derived NIfTI "
            "files and source FEM fields."
        ),
        "outputs": sorted(
            str(path.relative_to(results_dir))
            for path in results_dir.rglob("*")
            if path.is_file() and path.name != "analysis_manifest.json"
        ),
    }
    _atomic_json(results_dir / "analysis_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    pair = commands.add_parser("run-pair")
    pair.add_argument("--pair-index", type=int, required=True)
    pair.add_argument("--allowlist", type=Path, required=True)
    pair.add_argument("--generic-study-root", type=Path, required=True)
    pair.add_argument("--personalized-study-root", type=Path, required=True)
    pair.add_argument("--atlas-root", type=Path, required=True)
    pair.add_argument("--mni-fixed-atlas", type=Path, required=True)
    pair.add_argument("--mni-baseline-parent", type=Path, required=True)
    pair.add_argument("--output-root", type=Path, required=True)
    pair.add_argument("--targets-sha256", required=True)
    pair.add_argument("--individualized-targets-sha256", required=True)
    pair.add_argument("--workers", type=int, default=4)
    pair.add_argument("--force", action="store_true")

    collect = commands.add_parser("collect")
    collect.add_argument("--allowlist", type=Path, required=True)
    collect.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run-pair":
        if args.pair_index not in range(EXPECTED_PAIR_COUNT):
            raise ValueError("pair-index must be in 0-7.")
        if args.workers < 1:
            raise ValueError("workers must be at least one.")
        result = run_pair_qc(
            pair_index=args.pair_index,
            allowlist_path=args.allowlist,
            generic_study_root=args.generic_study_root,
            personalized_study_root=args.personalized_study_root,
            atlas_root=args.atlas_root,
            mni_fixed_atlas=args.mni_fixed_atlas,
            mni_baseline_parent=args.mni_baseline_parent,
            output_root=args.output_root,
            targets_sha256=args.targets_sha256,
            individualized_targets_sha256=args.individualized_targets_sha256,
            workers=args.workers,
            force=args.force,
        )
    else:
        result = collect_qc(
            allowlist_path=args.allowlist,
            output_root=args.output_root,
        )
    print(json.dumps(_json_ready(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
