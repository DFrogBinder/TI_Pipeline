# post_process.py
import gzip
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional, Dict, Iterable, Tuple, Sequence

import numpy as np
import json
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post.post_functions import (
    _resolve_fastsurfer_atlas,
    build_context_scale_mask_from_fastsurfer,
    fastsurfer_dkt_labels,
    make_outline,
    neighbor_overlay_color_map,
    overlay_neighbor_regions_and_efield_on_t1_with_roi,
    overlay_neighbor_regions_on_t1_with_roi,
    overlay_ti_full_field_true_vmax_reference_on_t1_with_roi,
    overlay_ti_thresholds_on_t1_with_roi,
    overlay_ti_thresholds_on_t1_with_roi_individual_scale,
    overlay_ti_thresholds_on_t1_with_roi_whole_brain_scale,
    roi_masks_on_ti_grid,
    write_csv,
)
from post.metric_extensions import (
    ANATOMY_DISTANCE_METRIC_KEYS,
    BASELINE_METRIC_KEYS,
    CENTROID_METRIC_KEYS,
    ELECTRODE_METRIC_KEYS,
    EXTENDED_METRIC_FIELDS,
    EXTENDED_METRIC_SCHEMA_VERSION,
    FOCALITY_METRIC_KEYS,
    NEIGHBOR_METRIC_KEYS,
    ROI_INTENSITY_METRIC_KEYS,
    WHOLE_BRAIN_COVERAGE_METRIC_KEYS,
    build_extended_metric_message_scaffold,
    build_extended_metric_status_scaffold,
    build_extended_metrics_scaffold,
    build_fixed_neighbor_masks,
    compute_anatomy_distance_metrics,
    compute_baseline_delta_metrics,
    compute_centroid_metrics,
    compute_electrode_distance_metrics,
    compute_focality_metrics,
    compute_neighbor_metrics,
    compute_roi_intensity_metrics,
    compute_whole_brain_coverage_metrics,
    extended_metrics_config_fingerprint,
    json_ready_metric_value,
    load_subject_fastsurfer_atlas_data,
)
from utils.roi_registry import resolve_fastsurfer_roi_label_ids
from utils.paths import post_root, ti_brain_path, t1_path
from utils.ti_utils import (
    ensure_dir,
    extract_table,
    load_ti_as_scalar,
    normalize_roi_name,
    resample_atlas_to_ti_grid,
    save_masked_nii,
    summarize_atlas_regions,
    vol_mm3,
)

@dataclass
class PostProcessConfig:
    # Required-ish
    root_dir: str
    subject: str = "MNI152"
    ti_path: Optional[str] = None
    t1_path: Optional[str] = None

    # Atlas selection
    atlas_mode: str = "mni"          # "auto" | "mni" | "fastsurfer"
    fastsurfer_root: Optional[str] = None
    fs_mri_path: Optional[str] = None # explicit path to the subject atlas NIfTI

    # Behavior
    out_dir: Optional[str] = None
    plot_roi: str = "ctx-lh-precentral"     # which ROI to highlight in overlays
    percentile: float = 95.0
    hard_threshold: float = 200.0
    overlay_z_offset_mm: float = 0.0
    overlay_full_field: bool = True
    write_region_table: bool = True
    region_percentile: float = 95.0
    offtarget_threshold: float = 0.2  # V/m threshold for focality checks
    mni_baseline_root: Optional[str] = None
    mni_fixed_atlas_path: Optional[str] = None
    neighbor_dilation_iter: int = 1
    csf_labels: Optional[Sequence[int]] = None
    skull_labels: Optional[Sequence[int]] = None
    electrode_csv: Optional[str] = None
    electrode_dataset_dir: Optional[str] = None
    electrode_names: Optional[Sequence[str]] = None
    eeg_positions_path_template: Optional[str] = None
    write_neighbor_table: bool = True
    write_neighbor_visualization: bool = True
    write_electrode_table: bool = True

    # Debug/logging
    verbose: bool = True


BASE_EXPECTED_OVERLAY_TYPES = (
    "context_top95",
    "context_threshold",
    "roi_focus_top95",
    "roi_focus_threshold",
    "whole_brain_reference_full",
)
FULL_FIELD_EXPECTED_OVERLAY_TYPES = (
    "context_full",
    "roi_focus_full",
)


def _expected_overlay_types(cfg: PostProcessConfig) -> Tuple[str, ...]:
    expected = list(BASE_EXPECTED_OVERLAY_TYPES)
    if cfg.overlay_full_field:
        expected.extend(FULL_FIELD_EXPECTED_OVERLAY_TYPES)
    return tuple(expected)


def _overlay_type_from_path(path: str) -> Optional[str]:
    name = Path(path).name
    if "_TI_overlay_context_" in name and name.endswith("_full.png"):
        return "context_full"
    if "_TI_overlay_context_" in name and "_top" in name:
        return "context_top95"
    if "_TI_overlay_context_" in name and "_above" in name:
        return "context_threshold"
    if "_TI_overlay_roi_focus_" in name and name.endswith("_full.png"):
        return "roi_focus_full"
    if "_TI_overlay_roi_focus_" in name and "_top" in name:
        return "roi_focus_top95"
    if "_TI_overlay_roi_focus_" in name and "_above" in name:
        return "roi_focus_threshold"
    if "_TI_overlay_whole_brain_reference_" in name and name.endswith("_full.png"):
        return "whole_brain_reference_full"
    return None


def _overlay_qc(
    *,
    cfg: PostProcessConfig,
    overlay_paths: Sequence[str],
    attempted: bool,
    error: Optional[str],
) -> Dict[str, Any]:
    expected = _expected_overlay_types(cfg)
    present = sorted(
        overlay_type
        for overlay_type in (_overlay_type_from_path(path) for path in overlay_paths)
        if overlay_type is not None
    )
    missing = [overlay_type for overlay_type in expected if overlay_type not in present]
    if not attempted:
        status = "skipped"
        message = "Overlay generation was skipped because no T1 background image was available."
    elif error is not None:
        status = "error"
        message = error
    elif missing:
        status = "error"
        message = (
            f"Expected {len(expected)} overlay PNG(s), but wrote {len(present)}. "
            f"Missing overlay type(s): {', '.join(missing)}."
        )
    else:
        status = "ok"
        message = None
    return {
        "status": status,
        "message": message,
        "expected_overlay_count": len(expected),
        "written_overlay_count": len(present),
        "expected_overlay_types": list(expected),
        "written_overlay_types": present,
        "missing_overlay_types": missing,
        "overlay_paths": list(overlay_paths),
    }


def _threshold_support_payload(
    *,
    voxels: int,
    denominator_voxels: int,
    voxel_volume_mm3: float,
    threshold: float,
    comparator: str,
) -> Dict[str, Any]:
    percent = (
        float((voxels / denominator_voxels) * 100.0)
        if denominator_voxels
        else float("nan")
    )
    return {
        "threshold": float(threshold),
        "comparator": comparator,
        "voxels": int(voxels),
        "volume_mm3": float(voxels * voxel_volume_mm3),
        "percent_of_denominator": percent,
        "has_voxels": bool(voxels > 0),
        "reason": None if voxels > 0 else "no_voxels_above_threshold",
    }


def _build_threshold_qc(
    *,
    cfg: PostProcessConfig,
    ti_data: np.ndarray,
    finite_mask: np.ndarray,
    roi_masks: Dict[str, np.ndarray],
    whole_brain_voxels: int,
    voxel_volume_mm3: float,
) -> Dict[str, Any]:
    metric_mask = finite_mask & (ti_data >= cfg.offtarget_threshold)
    overlay_mask = finite_mask & (ti_data >= cfg.hard_threshold)
    whole_brain_metric_voxels = int(np.count_nonzero(metric_mask))
    whole_brain_overlay_voxels = int(np.count_nonzero(overlay_mask))

    roi_payload: Dict[str, Any] = {}
    for roi_name, roi_mask in roi_masks.items():
        if roi_mask is None:
            continue
        roi_voxels = int(np.count_nonzero(roi_mask))
        metric_voxels = int(np.count_nonzero(metric_mask & roi_mask))
        overlay_voxels = int(np.count_nonzero(overlay_mask & roi_mask))
        roi_payload[roi_name] = {
            "roi_voxels": roi_voxels,
            "metric_threshold": _threshold_support_payload(
                voxels=metric_voxels,
                denominator_voxels=whole_brain_voxels,
                voxel_volume_mm3=voxel_volume_mm3,
                threshold=cfg.offtarget_threshold,
                comparator=">=",
            ),
            "overlay_threshold": _threshold_support_payload(
                voxels=overlay_voxels,
                denominator_voxels=whole_brain_voxels,
                voxel_volume_mm3=voxel_volume_mm3,
                threshold=cfg.hard_threshold,
                comparator=">=",
            ),
        }
        if metric_voxels == 0:
            roi_payload[roi_name]["metric_threshold"]["reason"] = (
                "no_roi_voxels_above_metric_threshold"
            )
        if overlay_voxels == 0:
            roi_payload[roi_name]["overlay_threshold"]["reason"] = (
                "no_roi_voxels_at_or_above_overlay_threshold"
            )

    return {
        "schema_version": 1,
        "metric_threshold": float(cfg.offtarget_threshold),
        "overlay_threshold": float(cfg.hard_threshold),
        "whole_brain": {
            "metric_threshold": _threshold_support_payload(
                voxels=whole_brain_metric_voxels,
                denominator_voxels=whole_brain_voxels,
                voxel_volume_mm3=voxel_volume_mm3,
                threshold=cfg.offtarget_threshold,
                comparator=">=",
            ),
            "overlay_threshold": _threshold_support_payload(
                voxels=whole_brain_overlay_voxels,
                denominator_voxels=whole_brain_voxels,
                voxel_volume_mm3=voxel_volume_mm3,
                threshold=cfg.hard_threshold,
                comparator=">=",
            ),
        },
        "rois": roi_payload,
    }


def _fastsurfer_roi_label_ids(roi_name: Optional[str]) -> list[int]:
    if not roi_name:
        return []
    try:
        return [int(label_id) for label_id in resolve_fastsurfer_roi_label_ids(roi_name)]
    except ValueError:
        return []


def _configured_path_status(path_value: Optional[str], *, must_be_file: bool) -> Dict[str, Any]:
    if not path_value:
        return {"configured": False, "exists": False, "is_file": False, "path": None}
    path = Path(path_value).expanduser()
    return {
        "configured": True,
        "exists": path.exists(),
        "is_file": path.is_file(),
        "path": str(path),
        "valid": path.is_file() if must_be_file else path.exists(),
    }


def _build_subject_qc_meta(
    *,
    cfg: PostProcessConfig,
    selected_roi: Optional[str],
    roi_mask: Optional[np.ndarray],
    overlay_qc: Dict[str, Any],
) -> Dict[str, Any]:
    roi_voxels = int(np.count_nonzero(roi_mask)) if roi_mask is not None else 0
    roi_status = "ok" if roi_voxels > 0 else "error"
    checks = {
        "target_roi_mask": {
            "status": roi_status,
            "message": None
            if roi_status == "ok"
            else f"Target ROI '{selected_roi}' is missing or empty on the TI grid.",
            "target_roi": selected_roi,
            "label_ids": _fastsurfer_roi_label_ids(selected_roi),
            "roi_voxels": roi_voxels,
        },
        "overlays": overlay_qc,
        "mni_baseline_root": _configured_path_status(cfg.mni_baseline_root, must_be_file=False),
        "mni_fixed_atlas_path": _configured_path_status(cfg.mni_fixed_atlas_path, must_be_file=True),
        "electrode_csv": _configured_path_status(cfg.electrode_csv, must_be_file=True),
        "electrode_dataset_dir": _configured_path_status(cfg.electrode_dataset_dir, must_be_file=False),
    }
    error_checks = [
        name
        for name, payload in checks.items()
        if isinstance(payload, dict) and payload.get("status") in {"error", "skipped"}
    ]
    status = "partial" if error_checks else "complete"
    return {
        "schema_version": 1,
        "status": status,
        "error_checks": error_checks,
        "checks": checks,
    }


def _infer_paths(cfg: PostProcessConfig) -> Tuple[str, str, Optional[str], str]:
    subj = cfg.subject
    root = os.path.abspath(cfg.root_dir)
    out_root = cfg.out_dir or str(post_root(root, subj))

    ti_path = cfg.ti_path or str(ti_brain_path(root, subj))
    if cfg.t1_path:
        t1_file = str(Path(cfg.t1_path).expanduser())
        t1_source = "cfg.t1_path"
    else:
        if subj.upper() == "MNI152":
            t1_file = "/home/boyan/sandbox/simnibs4_exmaples/m2m_MNI152/T1.nii.gz"
            t1_source = "built-in MNI152 template fallback"
        else:
            t1_file = str(t1_path(root, subj))
            t1_source = f"derived from utils.paths.t1_path(root={root!r}, subject={subj!r})"

    return out_root, ti_path, t1_file, t1_source


def _nearby_t1_candidates(t1_candidate: Path) -> Tuple[Path, ...]:
    candidates = []
    name = t1_candidate.name

    if name.endswith(".nii.gz"):
        candidates.append(t1_candidate.with_name(name[:-3]))
    elif t1_candidate.suffix == ".nii":
        candidates.append(t1_candidate.with_name(f"{name}.gz"))

    return tuple(candidates)


def _looks_like_nifti_bytes(payload: bytes) -> bool:
    if len(payload) < 348:
        return False

    sizeof_hdr = int.from_bytes(payload[:4], byteorder="little", signed=False)
    magic = payload[344:348]
    return sizeof_hdr == 348 and magic in {b"n+1\x00", b"ni1\x00"}


def _load_t1_image(t1_path: Path, cfg: PostProcessConfig) -> nib.spatialimages.SpatialImage:
    try:
        return nib.load(str(t1_path))
    except Exception:
        if not t1_path.name.endswith(".nii.gz"):
            raise

        outer_payload = gzip.open(t1_path, "rb").read()
        if not outer_payload.startswith(b"\x1f\x8b"):
            raise

        inner_payload = gzip.decompress(outer_payload)
        if not _looks_like_nifti_bytes(inner_payload):
            raise

        if cfg.verbose:
            print(
                f"[WARN] Detected double-gzipped T1 at {t1_path}; "
                "loading after one extra decompression."
            )

        return nib.Nifti1Image.from_bytes(inner_payload)


def _log_t1_lookup_details(
    cfg: PostProcessConfig,
    t1_candidate: Optional[str],
    t1_source: str,
    *,
    error: Optional[Exception] = None,
) -> None:
    if not cfg.verbose:
        return

    print(f"[INFO] Overlay T1 source: {t1_source}")
    if not t1_candidate:
        print("[WARN] Overlay T1 path is empty; overlays that need T1 will be skipped.")
        return

    expanded = Path(t1_candidate).expanduser()
    resolved = expanded.resolve()
    print(f"[INFO] Overlay T1 candidate: {t1_candidate}")
    print(f"[INFO] Overlay T1 resolved path: {resolved}")
    print(f"[INFO] Overlay T1 exists={resolved.exists()} is_file={resolved.is_file()}")

    nearby_existing = [str(path) for path in _nearby_t1_candidates(resolved) if path.is_file()]
    if nearby_existing:
        print(f"[INFO] Nearby existing T1 candidate(s): {', '.join(nearby_existing)}")

    if error is not None:
        print(f"[WARN] Failed loading T1 for overlays: {type(error).__name__}: {error}")
    elif not resolved.is_file():
        print("[WARN] T1 file for overlays was not found at the resolved path above.")


def _generate_selected_roi_overlays(
    *,
    cfg: PostProcessConfig,
    ti_img: nib.spatialimages.SpatialImage,
    ti_data: np.ndarray,
    t1_img_full: nib.spatialimages.SpatialImage,
    roi_mask: np.ndarray,
    fs_atlas_img: Optional[nib.Nifti1Image],
    out_dir: str,
    roi_name: str,
) -> tuple[list[str], str]:
    roi_mask_img = nib.Nifti1Image(roi_mask.astype(np.uint8), ti_img.affine, ti_img.header)
    context_scale_mask_img = (
        build_context_scale_mask_from_fastsurfer(fs_atlas_img)
        if fs_atlas_img is not None
        else None
    )
    sel_norm = normalize_roi_name(roi_name)
    out_base = os.path.join(out_dir, f"{sel_norm}_TI_overlay")

    def expected_triplet_present(
        top_path: Optional[str],
        threshold_path: Optional[str],
        full_path: Optional[str],
    ) -> bool:
        return top_path is not None and threshold_path is not None and (
            full_path is not None or not cfg.overlay_full_field
        )

    png_95 = png_02 = png_full = None
    try:
        png_95, png_02, png_full = overlay_ti_thresholds_on_t1_with_roi(
            ti_img=nib.Nifti1Image(ti_data, ti_img.affine, ti_img.header),
            t1_img=t1_img_full,
            roi_mask_img=roi_mask_img,
            out_prefix=f"{out_base}_context",
            subject=cfg.subject,
            z_offset_mm=cfg.overlay_z_offset_mm,
            include_full_field=cfg.overlay_full_field,
            percentile=cfg.percentile,
            hard_threshold=cfg.hard_threshold,
            scale_mask_img=context_scale_mask_img,
        )
    except Exception as exc:
        if cfg.verbose:
            print(
                f"[WARN] Context overlays failed for {cfg.subject}: "
                f"{type(exc).__name__}: {exc}"
            )

    roi_base = f"{out_base}_roi_focus"
    roi_overlay_mode = "roi_focus"
    roi_95 = roi_02 = roi_full = None
    try:
        roi_95, roi_02, roi_full = overlay_ti_thresholds_on_t1_with_roi_individual_scale(
            ti_img=nib.Nifti1Image(ti_data, ti_img.affine, ti_img.header),
            t1_img=t1_img_full,
            roi_mask_img=roi_mask_img,
            out_prefix=roi_base,
            subject=cfg.subject,
            z_offset_mm=cfg.overlay_z_offset_mm,
            include_full_field=cfg.overlay_full_field,
            percentile=cfg.percentile,
            hard_threshold=cfg.hard_threshold,
        )
        if not expected_triplet_present(roi_95, roi_02, roi_full):
            raise ValueError("ROI-focused overlay generation did not write all expected PNGs.")
    except Exception as exc:
        roi_overlay_mode = "whole_brain_fallback"
        if cfg.verbose:
            print(
                f"[WARN] ROI-focused overlays failed for {cfg.subject}: "
                f"{type(exc).__name__}: {exc}"
            )
            print(
                f"[INFO] Retrying ROI-focused overlays for {cfg.subject} "
                "with whole-brain display scaling."
            )
        try:
            roi_95, roi_02, roi_full = overlay_ti_thresholds_on_t1_with_roi_whole_brain_scale(
                ti_img=nib.Nifti1Image(ti_data, ti_img.affine, ti_img.header),
                t1_img=t1_img_full,
                roi_mask_img=roi_mask_img,
                out_prefix=roi_base,
                subject=cfg.subject,
                z_offset_mm=cfg.overlay_z_offset_mm,
                include_full_field=cfg.overlay_full_field,
                percentile=cfg.percentile,
                hard_threshold=cfg.hard_threshold,
            )
        except Exception as fallback_exc:
            roi_overlay_mode = "failed"
            if cfg.verbose:
                print(
                    f"[WARN] ROI-focused fallback overlays failed for {cfg.subject}: "
                    f"{type(fallback_exc).__name__}: {fallback_exc}"
                )

    reference_full = None
    try:
        reference_full = overlay_ti_full_field_true_vmax_reference_on_t1_with_roi(
            ti_img=nib.Nifti1Image(ti_data, ti_img.affine, ti_img.header),
            t1_img=t1_img_full,
            roi_mask_img=roi_mask_img,
            out_prefix=f"{out_base}_whole_brain_reference",
            subject=cfg.subject,
            z_offset_mm=cfg.overlay_z_offset_mm,
        )
    except Exception as exc:
        if cfg.verbose:
            print(
                f"[WARN] Whole-brain reference overlay failed for {cfg.subject}: "
                f"{type(exc).__name__}: {exc}"
            )

    overlay_paths = [
        path
        for path in (png_full, png_95, png_02, roi_full, roi_95, roi_02, reference_full)
        if path is not None
    ]
    return overlay_paths, roi_overlay_mode


def _write_neighbor_visualization_outputs(
    *,
    cfg: PostProcessConfig,
    ti_img: nib.Nifti1Image,
    t1_img_full: Optional[nib.Nifti1Image],
    roi_mask: np.ndarray,
    subject_atlas_data: Optional[np.ndarray],
    out_dir: str,
    roi_name: str,
) -> Dict[str, Any]:
    if not cfg.write_neighbor_visualization:
        return {
            "status": "disabled",
            "message": "Neighbor visualization export is disabled.",
        }
    if not cfg.mni_fixed_atlas_path:
        return {
            "status": "not_configured",
            "message": "Neighbor visualization requires mni_fixed_atlas_path.",
        }
    if subject_atlas_data is None:
        return {
            "status": "error",
            "message": "Neighbor visualization requires a subject FastSurfer atlas on the TI grid.",
        }

    roi_stub = normalize_roi_name(roi_name)
    result: Dict[str, Any] = {
        "status": "pending",
        "message": None,
        "neighbor_dilation_iter": int(cfg.neighbor_dilation_iter),
        "neighbor_label_ids": [],
        "neighbor_label_names": [],
        "union_mask_path": None,
        "categorical_mask_path": None,
        "metadata_path": None,
        "overlay_path": None,
        "efield_overlay_path": None,
        "neighbor_overlay_colors": [],
    }

    try:
        masks = build_fixed_neighbor_masks(
            mni_fixed_atlas_path=cfg.mni_fixed_atlas_path,
            roi_name=roi_name,
            dilation_iter=cfg.neighbor_dilation_iter,
            subject_atlas_data=subject_atlas_data,
        )
        neighbor_template = masks["neighbor_template"]
        neighbor_union_mask = masks["neighbor_union_mask"]
        neighbor_categorical_mask = masks["neighbor_categorical_mask"]
        neighbor_label_ids = [int(row["label_id"]) for row in neighbor_template]
        neighbor_color_map = neighbor_overlay_color_map(neighbor_label_ids)
        neighbor_overlay_colors = [
            {
                "label_id": int(row["label_id"]),
                "label_name": str(row["label_name"]),
                "color": neighbor_color_map[int(row["label_id"])],
            }
            for row in neighbor_template
        ]

        union_path = os.path.join(out_dir, f"{roi_stub}_fixed_neighbor_union_mask.nii.gz")
        categorical_path = os.path.join(out_dir, f"{roi_stub}_fixed_neighbor_categorical_mask.nii.gz")
        metadata_path = os.path.join(out_dir, f"{roi_stub}_fixed_neighbor_visualization.json")
        nib.save(nib.Nifti1Image(neighbor_union_mask.astype(np.uint8), ti_img.affine), union_path)
        nib.save(nib.Nifti1Image(neighbor_categorical_mask.astype(np.int32), ti_img.affine), categorical_path)

        metadata = {
            "roi_name": roi_name,
            "roi_label_ids": _fastsurfer_roi_label_ids(roi_name),
            "mni_fixed_atlas_path": cfg.mni_fixed_atlas_path,
            "neighbor_dilation_iter": int(cfg.neighbor_dilation_iter),
            "neighbor_template": neighbor_template,
            "neighbor_overlay_colors": neighbor_overlay_colors,
            "neighbor_union_voxels": int(np.count_nonzero(neighbor_union_mask)),
        }
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(json_ready_metric_value(metadata), handle, indent=2)

        result.update(
            {
                "neighbor_label_ids": neighbor_label_ids,
                "neighbor_label_names": [str(row["label_name"]) for row in neighbor_template],
                "union_mask_path": union_path,
                "categorical_mask_path": categorical_path,
                "metadata_path": metadata_path,
                "neighbor_overlay_colors": neighbor_overlay_colors,
            }
        )

        if t1_img_full is not None and np.any(neighbor_union_mask):
            overlay_path = os.path.join(out_dir, f"{roi_stub}_fixed_neighbor_categorical_overlay.png")
            overlay_neighbor_regions_on_t1_with_roi(
                ti_img=ti_img,
                t1_img=t1_img_full,
                roi_mask_img=nib.Nifti1Image(roi_mask.astype(np.uint8), ti_img.affine),
                neighbor_mask_img=nib.Nifti1Image(neighbor_categorical_mask.astype(np.int32), ti_img.affine),
                out_png=overlay_path,
                subject=cfg.subject,
                z_offset_mm=cfg.overlay_z_offset_mm,
                neighbor_label_ids=neighbor_label_ids,
            )
            result["overlay_path"] = overlay_path
            efield_overlay_path = os.path.join(out_dir, f"{roi_stub}_fixed_neighbor_efield_overlay.png")
            try:
                overlay_neighbor_regions_and_efield_on_t1_with_roi(
                    ti_img=ti_img,
                    t1_img=t1_img_full,
                    roi_mask_img=nib.Nifti1Image(roi_mask.astype(np.uint8), ti_img.affine),
                    neighbor_mask_img=nib.Nifti1Image(neighbor_categorical_mask.astype(np.int32), ti_img.affine),
                    out_png=efield_overlay_path,
                    subject=cfg.subject,
                    z_offset_mm=cfg.overlay_z_offset_mm,
                    neighbor_label_ids=neighbor_label_ids,
                )
                result["efield_overlay_path"] = efield_overlay_path
                result["status"] = "complete"
            except Exception as exc:
                result["status"] = "partial"
                result["message"] = (
                    "Categorical neighbor overlay was written, but the cropped e-field "
                    f"overlay failed: {type(exc).__name__}: {exc}"
                )
                if cfg.verbose:
                    print(f"[WARN] {result['message']}")
        elif np.any(neighbor_union_mask):
            result["status"] = "mask_only"
            result["message"] = "T1 background was unavailable, so only neighbor mask NIfTI files were written."
        else:
            result["status"] = "empty"
            result["message"] = "Neighbor template resolved, but the subject atlas produced an empty neighbor union mask."
    except Exception as exc:
        result["status"] = "error"
        result["message"] = f"{type(exc).__name__}: {exc}"
        if cfg.verbose:
            print(f"[WARN] Neighbor visualization export failed for {cfg.subject}: {result['message']}")
    return result


def extended_metrics_fingerprint_for_cfg(cfg: PostProcessConfig) -> str:
    _, ti_path, _, _ = _infer_paths(cfg)
    return extended_metrics_config_fingerprint(
        root_dir=cfg.root_dir,
        subject=cfg.subject,
        ti_path=ti_path,
        atlas_mode=cfg.atlas_mode,
        fastsurfer_root=cfg.fastsurfer_root,
        subject_fastsurfer_atlas_path=cfg.fs_mri_path,
        roi_name=cfg.plot_roi or "",
        percentile=cfg.percentile,
        region_percentile=cfg.region_percentile,
        focality_threshold=cfg.offtarget_threshold,
        mni_baseline_root=cfg.mni_baseline_root,
        mni_fixed_atlas_path=cfg.mni_fixed_atlas_path,
        neighbor_dilation_iter=cfg.neighbor_dilation_iter,
        write_neighbor_visualization=cfg.write_neighbor_visualization,
        csf_labels=cfg.csf_labels,
        skull_labels=cfg.skull_labels,
        electrode_csv=cfg.electrode_csv,
        electrode_dataset_dir=cfg.electrode_dataset_dir,
        electrode_names=cfg.electrode_names,
        eeg_positions_path_template=cfg.eeg_positions_path_template,
    )


def _set_metric_group_state(
    *,
    metric_status: Dict[str, str],
    metric_messages: Dict[str, Optional[str]],
    metric_keys: Iterable[str],
    status: str,
    message: Optional[str] = None,
) -> None:
    for key in metric_keys:
        metric_status[key] = status
        metric_messages[key] = message


def _apply_metric_values(
    metric_values: Dict[str, Any],
    values: Dict[str, Any],
) -> None:
    for key, value in values.items():
        metric_values[key] = json_ready_metric_value(value)


def run_post_process(cfg: PostProcessConfig) -> Dict[str, dict]:
    """
    Library entrypoint. Returns a dict with useful results and file paths.
    Set breakpoints inside to step through.
    """
    # ---- Paths & IO ----
    out_dir, ti_path, t1_path, t1_path_source = _infer_paths(cfg)
    if cfg.verbose:
        print(f"[cfg] subject={cfg.subject} | atlas_mode={cfg.atlas_mode}")
        print(f"[cfg] ti_path={ti_path}")
        print(f"[cfg] t1_path={t1_path}")
        print(f"[cfg] t1_path_source={t1_path_source}")
        print(f"[cfg] out_dir={out_dir}")

    ensure_dir(out_dir)
    ti_img = nib.load(ti_path)
    ti_data = load_ti_as_scalar(ti_img)

    # ---- Atlas / ROI masks on TI grid ----
    roi_masks, atlas_imgs = roi_masks_on_ti_grid(
        ti_img,
        atlas_mode=cfg.atlas_mode,
        subject=cfg.subject,
        fastsurfer_root=cfg.fastsurfer_root,
        fastsurfer_atlas_path=cfg.fs_mri_path,
        roi_names=[cfg.plot_roi] if cfg.plot_roi else None,
    )

    selected_plot_roi = cfg.plot_roi
    if selected_plot_roi not in roi_masks and len(roi_masks) == 1:
        selected_plot_roi = next(iter(roi_masks))

    # Resolve full FastSurfer atlas if available (for full-region summary)
    fs_atlas_path = None
    if cfg.atlas_mode in ("auto", "fastsurfer"):
        # try:
        fs_atlas_path = _resolve_fastsurfer_atlas(cfg.subject, cfg.fastsurfer_root, cfg.fs_mri_path)
        # except Exception:
        #     fs_atlas_path = None
    
    mask = roi_masks.get(selected_plot_roi)
    if cfg.verbose:
        print("[INFO]:MODE CHECK")
        print("[INFO]:subject:", cfg.subject, "| atlas_mode:", cfg.atlas_mode)
        print("[INFO]:fs path:", cfg.fs_mri_path or (cfg.fastsurfer_root and os.path.join(cfg.fastsurfer_root, f"{cfg.subject}.nii.gz")))
        print("[INFO]:mask dtype/shape:", mask.dtype if mask is not None else None, mask.shape if mask is not None else None)
        print("[INFO]:mask voxels >0:", int(mask.sum()) if mask is not None else 0)


    # ---- Thresholds ----
    finite = np.isfinite(ti_data)
    if not np.any(finite):
        raise RuntimeError("[ERROR] No finite values in ti_data; check your masking.")

    whole_brain_voxels = int(np.sum(finite))
    thr = float(np.nanpercentile(ti_data[finite], cfg.percentile))
    topP_mask = finite & (ti_data >= thr)
    top_percentile_voxels = int(np.sum(topP_mask))
    top_percentile_percent_of_whole_brain = (
        float((top_percentile_voxels / whole_brain_voxels) * 100.0)
        if whole_brain_voxels
        else float("nan")
    )
    focality_mask = finite & (ti_data >= cfg.offtarget_threshold)

    # Background for overlays (resampled to TI grid if available)
    t1_img_full = None
    if t1_path:
        try:
            t1_resolved = Path(t1_path).expanduser()
            if t1_resolved.is_file():
                t1_img_full = _load_t1_image(t1_resolved, cfg)
            else:
                _log_t1_lookup_details(cfg, t1_path, t1_path_source)
        except Exception as e:
            _log_t1_lookup_details(cfg, t1_path, t1_path_source, error=e)

    # Save global masks
    vox_vol = vol_mm3(ti_img)
    whole_brain_volume_mm3 = float(whole_brain_voxels * vox_vol)
    threshold_qc = _build_threshold_qc(
        cfg=cfg,
        ti_data=ti_data,
        finite_mask=finite,
        roi_masks=roi_masks,
        whole_brain_voxels=whole_brain_voxels,
        voxel_volume_mm3=vox_vol,
    )
    topP_mask_path = os.path.join(out_dir, f"efield_top{int(cfg.percentile)}pct_mask.nii.gz")
    nib.save(nib.Nifti1Image(topP_mask.astype(np.uint8), ti_img.affine), topP_mask_path)

    topP_ti_path = os.path.join(out_dir, f"TI_in_Top{int(cfg.percentile)}.nii.gz")
    save_masked_nii(ti_data, topP_mask, ti_img, topP_ti_path)

    # ---- Per-ROI products ----
    per_roi = {}
    per_roi_metrics = {}
    for roi_name, mask in roi_masks.items():
        if mask is None:
            continue

        overlap_mask = topP_mask & mask
        focality_in_roi_mask = focality_mask & mask
        roi_voxels = int(np.sum(mask))
        overlap_top_voxels = int(np.sum(overlap_mask))
        focality_in_roi_voxels = int(np.sum(focality_in_roi_mask))
        roi_mask_img = nib.Nifti1Image(mask.astype(np.uint8), ti_img.affine, ti_img.header)
        overlap_mask_img = nib.Nifti1Image(overlap_mask.astype(np.uint8), ti_img.affine, ti_img.header)

        if cfg.verbose:
            print(f"[ROI:{roi_name}] thr@{cfg.percentile}th = {thr:.6g}")
            print(f"[ROI:{roi_name}] ROI voxels: {roi_voxels:,} ({roi_voxels*vox_vol/1e3:.3f} mL)")
            print(f"[ROI:{roi_name}] Overlap voxels: {overlap_top_voxels:,} "
                  f"({overlap_top_voxels*vox_vol/1e3:.3f} mL)")

        # Tables
        ijk_r, xyz_r, vals_r = extract_table(mask,         ti_img, ti_data)
        ijk_t, xyz_t, vals_t = extract_table(topP_mask,    ti_img, ti_data)
        ijk_o, xyz_o, vals_o = extract_table(overlap_mask, ti_img, ti_data)

        roi_csv         = os.path.join(out_dir, f"{roi_name}_values.csv")
        topp_csv        = os.path.join(out_dir, f"Top{int(cfg.percentile)}_values.csv")
        overlap_csv     = os.path.join(out_dir, f"{roi_name}_Top{int(cfg.percentile)}_overlap_values.csv")
        write_csv(roi_csv,     ijk_r, xyz_r, vals_r)
        write_csv(topp_csv,    ijk_t, xyz_t, vals_t)
        write_csv(overlap_csv, ijk_o, xyz_o, vals_o)

        # Save masks
        atlas_mask_path = os.path.join(out_dir, f"atlas_{roi_name}_mask.nii.gz")
        overlap_mask_path = os.path.join(out_dir, f"{roi_name}_overlap_top{int(cfg.percentile)}pct_mask.nii.gz")
        nib.save(roi_mask_img, atlas_mask_path)
        nib.save(overlap_mask_img, overlap_mask_path)

        # Save masked TI
        ti_in_roi_path        = os.path.join(out_dir, f"TI_in_{roi_name}.nii.gz")
        ti_in_roi_topP_path   = os.path.join(out_dir, f"TI_in_{roi_name}_Top{int(cfg.percentile)}.nii.gz")
        save_masked_nii(ti_data, mask,         ti_img, ti_in_roi_path)
        save_masked_nii(ti_data, overlap_mask, ti_img, ti_in_roi_topP_path)

        per_roi[roi_name] = dict(
            roi_csv=roi_csv,
            topp_csv=topp_csv,
            overlap_csv=overlap_csv,
            atlas_mask=atlas_mask_path,
            overlap_mask=overlap_mask_path,
            ti_in_roi=ti_in_roi_path,
            ti_in_roi_topP=ti_in_roi_topP_path,
        )

        roi_finite_mask = mask & finite
        roi_percentile_value = (
            float(np.nanpercentile(ti_data[roi_finite_mask], cfg.region_percentile))
            if np.any(roi_finite_mask)
            else float("nan")
        )

        per_roi_metrics[roi_name] = dict(
            roi_voxels=roi_voxels,
            overlap_top_voxels=overlap_top_voxels,
            roi_volume_mm3=float(roi_voxels * vox_vol),
            overlap_volume_mm3=float(overlap_top_voxels * vox_vol),
            overlap_fraction=float(overlap_top_voxels / roi_voxels) if roi_voxels else 0.0,
            roi_percent_of_whole_brain=(
                float((roi_voxels / whole_brain_voxels) * 100.0)
                if whole_brain_voxels
                else float("nan")
            ),
            overlap_top_percent_of_whole_brain=(
                float((overlap_top_voxels / whole_brain_voxels) * 100.0)
                if whole_brain_voxels
                else float("nan")
            ),
            focality_in_roi_voxels_gt_threshold=focality_in_roi_voxels,
            focality_in_roi_volume_mm3_gt_threshold=float(focality_in_roi_voxels * vox_vol),
            focality_in_roi_percent_of_whole_brain_gt_threshold=(
                float((focality_in_roi_voxels / whole_brain_voxels) * 100.0)
                if whole_brain_voxels
                else float("nan")
            ),
            threshold_qc=threshold_qc["rois"].get(roi_name, {}),
            roi_percentile=cfg.region_percentile,
            roi_percentile_value=roi_percentile_value,
        )
        
        hippo_outline_path = None
        hippo_label_path   = None
        hippo_boldmask_path = None

        if "hippocampus" in roi_name.lower():
            roi_stub = normalize_roi_name(roi_name)
            outline = make_outline(mask, iterations=1)  # increase to 2-3 if you want thicker edge
            hippo_outline_path = os.path.join(out_dir, f"atlas_{roi_stub}_outline_mask.nii.gz")
            nib.save(nib.Nifti1Image(outline, ti_img.affine), hippo_outline_path)

            # 2b) Make a "bold" (high intensity) filled mask for viewers that fade 0/1 overlays
            bold = (mask.astype(np.uint8) * 255)
            hippo_boldmask_path = os.path.join(out_dir, f"atlas_{roi_stub}_mask_255.nii.gz")
            nib.save(nib.Nifti1Image(bold, ti_img.affine), hippo_boldmask_path)

            # 2c) Export a label-ID volume on the TI grid (17 for L, 53 for R; 0 elsewhere)
            #     This is often the easiest to color distinctly in Freeview/FSLEyes.
            #     We reconstruct it from the original atlas image on the TI grid:
            label_vol = np.zeros(mask.shape, dtype=np.uint16)
            if roi_name.lower().startswith("left-hippocampus"):
                label_vol[mask] = 17
            elif roi_name.lower().startswith("right-hippocampus"):
                label_vol[mask] = 53
            else:
                ijk = np.argwhere(mask)
                xyz = nib.affines.apply_affine(ti_img.affine, ijk)
                left_idx = xyz[:, 0] < 0
                right_idx = ~left_idx
                if ijk.size > 0:
                    if left_idx.any():
                        label_vol[tuple(ijk[left_idx].T)] = 17
                    if right_idx.any():
                        label_vol[tuple(ijk[right_idx].T)] = 53

            hippo_label_path = os.path.join(out_dir, f"atlas_{roi_stub}_labels_on_TI.nii.gz")
            nib.save(nib.Nifti1Image(label_vol, ti_img.affine), hippo_label_path)

            # Track in the return dict
            per_roi[roi_name].update(dict(
                hippo_outline=hippo_outline_path,
                hippo_mask_255=hippo_boldmask_path,
                hippo_label_ids=hippo_label_path,
            ))

    # ---- Full atlas region summaries (FastSurfer only) ----
    region_table_path = None
    region_df = None
    fs_atlas_img = None
    if cfg.write_region_table and fs_atlas_path:
        try:
            fs_atlas_img = resample_atlas_to_ti_grid(nib.load(fs_atlas_path), ti_img)
            region_df = summarize_atlas_regions(
                nib.Nifti1Image(ti_data, ti_img.affine, ti_img.header),
                fs_atlas_img,
                fastsurfer_dkt_labels,
                percentile=cfg.region_percentile,
            )
            region_table_path = os.path.join(out_dir, "region_stats_fastsurfer.csv")
            region_df.to_csv(region_table_path, index=False)
            if cfg.verbose:
                print(f"[INFO] Saved region stats to {region_table_path}")
        except Exception as e:
            if cfg.verbose:
                print(f"[WARN] Skipped region table: {e}")
    elif fs_atlas_path:
        try:
            fs_atlas_img = resample_atlas_to_ti_grid(nib.load(fs_atlas_path), ti_img)
        except Exception as e:
            if cfg.verbose:
                print(f"[WARN] Failed loading FastSurfer atlas for overlay scaling: {e}")


    sel = selected_plot_roi

    # ---- Extended per-repeat metrics for downstream repeatability analysis ----
    extended_metrics = build_extended_metrics_scaffold()
    extended_metric_status = build_extended_metric_status_scaffold()
    extended_metric_messages = build_extended_metric_message_scaffold()
    extended_group_status: Dict[str, str] = {}
    extended_group_messages: Dict[str, Optional[str]] = {}
    extended_config_fingerprint = extended_metrics_fingerprint_for_cfg(cfg)
    neighbor_table_path = None
    neighbor_visualization: Optional[Dict[str, Any]] = None
    electrode_table_path = None
    if sel in roi_masks:
        roi_mask = roi_masks[sel]
        subject_atlas_data = None
        subject_atlas_error: Optional[Exception] = None
        if fs_atlas_path:
            try:
                subject_atlas_data = load_subject_fastsurfer_atlas_data(ti_img, fs_atlas_path)
            except Exception as exc:
                subject_atlas_error = exc
                if cfg.verbose:
                    print(
                        f"[WARN] Failed loading subject FastSurfer atlas for extended metrics "
                        f"for {cfg.subject}: {type(exc).__name__}: {exc}"
                    )

        def execute_metric_group(
            group_name: str,
            metric_keys: Sequence[str],
            compute_fn,
        ) -> None:
            try:
                values = compute_fn()
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                extended_group_status[group_name] = "error"
                extended_group_messages[group_name] = message
                _set_metric_group_state(
                    metric_status=extended_metric_status,
                    metric_messages=extended_metric_messages,
                    metric_keys=metric_keys,
                    status="error",
                    message=message,
                )
                if cfg.verbose:
                    print(
                        f"[WARN] Extended metric group '{group_name}' failed for {cfg.subject}: "
                        f"{message}"
                    )
                return
            _apply_metric_values(extended_metrics, values)
            extended_group_status[group_name] = "ok"
            extended_group_messages[group_name] = None
            _set_metric_group_state(
                metric_status=extended_metric_status,
                metric_messages=extended_metric_messages,
                metric_keys=metric_keys,
                status="ok",
            )

        def mark_metric_group_not_configured(
            group_name: str,
            metric_keys: Sequence[str],
            message: str,
        ) -> None:
            extended_group_status[group_name] = "not_configured"
            extended_group_messages[group_name] = message
            _set_metric_group_state(
                metric_status=extended_metric_status,
                metric_messages=extended_metric_messages,
                metric_keys=metric_keys,
                status="not_configured",
                message=message,
            )

        execute_metric_group(
            "roi_intensity",
            ROI_INTENSITY_METRIC_KEYS,
            lambda: compute_roi_intensity_metrics(
                ti_img=ti_img,
                ti_data=ti_data,
                roi_mask=roi_mask,
                finite_mask=finite,
            ),
        )
        execute_metric_group(
            "focality",
            FOCALITY_METRIC_KEYS,
            lambda: compute_focality_metrics(
                ti_img=ti_img,
                ti_data=ti_data,
                finite_mask=finite,
                focality_threshold=cfg.offtarget_threshold,
            ),
        )
        execute_metric_group(
            "whole_brain_coverage",
            WHOLE_BRAIN_COVERAGE_METRIC_KEYS,
            lambda: compute_whole_brain_coverage_metrics(
                ti_img=ti_img,
                ti_data=ti_data,
                finite_mask=finite,
            ),
        )

        baseline_prereq_failed = (
            extended_group_status.get("roi_intensity") == "error"
            or extended_group_status.get("focality") == "error"
        )
        if baseline_prereq_failed:
            message = (
                "Baseline metrics depend on successful ROI intensity and focality metrics."
            )
            extended_group_status["baseline"] = "error"
            extended_group_messages["baseline"] = message
            _set_metric_group_state(
                metric_status=extended_metric_status,
                metric_messages=extended_metric_messages,
                metric_keys=BASELINE_METRIC_KEYS,
                status="error",
                message=message,
            )
            if cfg.verbose:
                print(
                    f"[WARN] Extended metric group 'baseline' failed for {cfg.subject}: "
                    f"{message}"
                )
        elif cfg.mni_baseline_root and cfg.mni_fixed_atlas_path:
            execute_metric_group(
                "baseline",
                BASELINE_METRIC_KEYS,
                lambda: compute_baseline_delta_metrics(
                    roi_name=sel,
                    mni_baseline_root=cfg.mni_baseline_root,
                    mni_fixed_atlas_path=cfg.mni_fixed_atlas_path,
                    focality_threshold=cfg.offtarget_threshold,
                    roi_peak=extended_metrics.get("roi_peak"),
                    roi_mean=extended_metrics.get("roi_mean"),
                    focality_voxels_gt_threshold=extended_metrics.get("focality_voxels_gt_threshold"),
                    focality_volume_mm3_gt_threshold=extended_metrics.get("focality_volume_mm3_gt_threshold"),
                ),
            )
        else:
            mark_metric_group_not_configured(
                "baseline",
                BASELINE_METRIC_KEYS,
                "Baseline comparison requires both mni_baseline_root and mni_fixed_atlas_path.",
            )

        if cfg.mni_fixed_atlas_path:
            try:
                neighbor_values = compute_neighbor_metrics(
                    mni_fixed_atlas_path=cfg.mni_fixed_atlas_path,
                    roi_name=sel,
                    dilation_iter=cfg.neighbor_dilation_iter,
                    subject_atlas_data=subject_atlas_data,
                    ti_data=ti_data,
                    finite_mask=finite,
                    voxel_volume_mm3=vox_vol,
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                extended_group_status["neighbors"] = "error"
                extended_group_messages["neighbors"] = message
                _set_metric_group_state(
                    metric_status=extended_metric_status,
                    metric_messages=extended_metric_messages,
                    metric_keys=NEIGHBOR_METRIC_KEYS,
                    status="error",
                    message=message,
                )
                if cfg.verbose:
                    print(
                        f"[WARN] Extended metric group 'neighbors' failed for {cfg.subject}: "
                        f"{message}"
                    )
            else:
                _apply_metric_values(extended_metrics, neighbor_values)
                if subject_atlas_error is not None and fs_atlas_path:
                    message = (
                        "Neighbor template metrics were computed, but subject atlas-dependent "
                        f"neighbor summaries failed because the subject atlas could not be loaded: "
                        f"{type(subject_atlas_error).__name__}: {subject_atlas_error}"
                    )
                    extended_group_status["neighbors"] = "error"
                    extended_group_messages["neighbors"] = message
                    _set_metric_group_state(
                        metric_status=extended_metric_status,
                        metric_messages=extended_metric_messages,
                        metric_keys=NEIGHBOR_METRIC_KEYS,
                        status="error",
                        message=message,
                    )
                    if cfg.verbose:
                        print(f"[WARN] Extended metric group 'neighbors' partial failure for {cfg.subject}: {message}")
                else:
                    extended_group_status["neighbors"] = "ok"
                    extended_group_messages["neighbors"] = None
                    _set_metric_group_state(
                        metric_status=extended_metric_status,
                        metric_messages=extended_metric_messages,
                        metric_keys=NEIGHBOR_METRIC_KEYS,
                        status="ok",
                    )
        else:
            mark_metric_group_not_configured(
                "neighbors",
                NEIGHBOR_METRIC_KEYS,
                "Neighbor metrics require mni_fixed_atlas_path.",
            )

        roi_centroid_ijk = None
        roi_centroid_xyz = None
        try:
            centroid_metrics, roi_centroid_ijk, roi_centroid_xyz = compute_centroid_metrics(
                roi_mask,
                ti_img.affine,
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            extended_group_status["centroid"] = "error"
            extended_group_messages["centroid"] = message
            _set_metric_group_state(
                metric_status=extended_metric_status,
                metric_messages=extended_metric_messages,
                metric_keys=CENTROID_METRIC_KEYS,
                status="error",
                message=message,
            )
            if cfg.verbose:
                print(
                    f"[WARN] Extended metric group 'centroid' failed for {cfg.subject}: "
                    f"{message}"
                )
        else:
            _apply_metric_values(extended_metrics, centroid_metrics)
            extended_group_status["centroid"] = "ok"
            extended_group_messages["centroid"] = None
            _set_metric_group_state(
                metric_status=extended_metric_status,
                metric_messages=extended_metric_messages,
                metric_keys=CENTROID_METRIC_KEYS,
                status="ok",
            )

        anatomy_labels_present = bool((cfg.csf_labels or [24])) or bool(cfg.skull_labels)
        centroid_failed = extended_group_status.get("centroid") == "error"
        centroid_failure_message = extended_group_messages.get("centroid")
        if centroid_failed:
            message = f"Centroid metrics failed, so anatomy distances could not be computed: {centroid_failure_message}"
            extended_group_status["anatomy_distances"] = "error"
            extended_group_messages["anatomy_distances"] = message
            _set_metric_group_state(
                metric_status=extended_metric_status,
                metric_messages=extended_metric_messages,
                metric_keys=ANATOMY_DISTANCE_METRIC_KEYS,
                status="error",
                message=message,
            )
            if cfg.verbose:
                print(
                    f"[WARN] Extended metric group 'anatomy_distances' failed for {cfg.subject}: "
                    f"{message}"
                )
        elif not anatomy_labels_present:
            mark_metric_group_not_configured(
                "anatomy_distances",
                ANATOMY_DISTANCE_METRIC_KEYS,
                "Anatomy distance metrics require csf_labels and/or skull_labels.",
            )
        elif not fs_atlas_path:
            mark_metric_group_not_configured(
                "anatomy_distances",
                ANATOMY_DISTANCE_METRIC_KEYS,
                "Anatomy distance metrics require a subject FastSurfer atlas.",
            )
        elif subject_atlas_error is not None:
            message = f"{type(subject_atlas_error).__name__}: {subject_atlas_error}"
            extended_group_status["anatomy_distances"] = "error"
            extended_group_messages["anatomy_distances"] = message
            _set_metric_group_state(
                metric_status=extended_metric_status,
                metric_messages=extended_metric_messages,
                metric_keys=ANATOMY_DISTANCE_METRIC_KEYS,
                status="error",
                message=message,
            )
            if cfg.verbose:
                print(
                    f"[WARN] Extended metric group 'anatomy_distances' failed for {cfg.subject}: "
                    f"{message}"
                )
        else:
            execute_metric_group(
                "anatomy_distances",
                ANATOMY_DISTANCE_METRIC_KEYS,
                lambda: compute_anatomy_distance_metrics(
                    atlas_data=subject_atlas_data,
                    roi_centroid_ijk=roi_centroid_ijk,
                    zooms=ti_img.header.get_zooms()[:3],
                    csf_labels=cfg.csf_labels or [24],
                    skull_labels=cfg.skull_labels,
                ),
            )

        electrodes_configured = bool(
            cfg.electrode_csv or cfg.electrode_dataset_dir or cfg.electrode_names
        )
        if centroid_failed and electrodes_configured:
            message = f"Centroid metrics failed, so electrode distances could not be computed: {centroid_failure_message}"
            extended_group_status["electrodes"] = "error"
            extended_group_messages["electrodes"] = message
            _set_metric_group_state(
                metric_status=extended_metric_status,
                metric_messages=extended_metric_messages,
                metric_keys=ELECTRODE_METRIC_KEYS,
                status="error",
                message=message,
            )
            if cfg.verbose:
                print(
                    f"[WARN] Extended metric group 'electrodes' failed for {cfg.subject}: "
                    f"{message}"
                )
        elif electrodes_configured:
            execute_metric_group(
                "electrodes",
                ELECTRODE_METRIC_KEYS,
                lambda: compute_electrode_distance_metrics(
                    root_dir=cfg.root_dir,
                    subject=cfg.subject,
                    roi_name=sel,
                    roi_centroid_xyz=roi_centroid_xyz,
                    electrode_csv=cfg.electrode_csv,
                    electrode_dataset_dir=cfg.electrode_dataset_dir,
                    electrode_names=cfg.electrode_names,
                    eeg_positions_path_template=cfg.eeg_positions_path_template,
                ),
            )
        else:
            mark_metric_group_not_configured(
                "electrodes",
                ELECTRODE_METRIC_KEYS,
                "Electrode distance metrics require electrode_csv, electrode_dataset_dir, or electrode_names.",
            )

        if cfg.write_neighbor_table and extended_metrics.get("neighbors"):
            roi_stub = normalize_roi_name(sel)
            neighbor_table_path = os.path.join(out_dir, f"{roi_stub}_fixed_neighbors.json")
            with open(neighbor_table_path, "w", encoding="utf-8") as handle:
                json.dump(extended_metrics["neighbors"], handle, indent=2)
        if cfg.write_neighbor_visualization:
            if extended_group_status.get("neighbors") == "ok":
                neighbor_visualization = _write_neighbor_visualization_outputs(
                    cfg=cfg,
                    ti_img=ti_img,
                    t1_img_full=t1_img_full,
                    roi_mask=roi_mask,
                    subject_atlas_data=subject_atlas_data,
                    out_dir=out_dir,
                    roi_name=sel,
                )
            else:
                neighbor_visualization = {
                    "status": "skipped",
                    "message": "Neighbor visualization skipped because neighbor metrics were not complete.",
                }
        if cfg.write_electrode_table and extended_metrics.get("electrode_distances"):
            roi_stub = normalize_roi_name(sel)
            electrode_table_path = os.path.join(out_dir, f"{roi_stub}_electrode_distances.json")
            with open(electrode_table_path, "w", encoding="utf-8") as handle:
                json.dump(extended_metrics["electrode_distances"], handle, indent=2)
    else:
        missing_roi_message = f"Selected target ROI '{sel}' was not present on the TI grid."
        for group_name, metric_keys in (
            ("roi_intensity", ROI_INTENSITY_METRIC_KEYS),
            ("focality", FOCALITY_METRIC_KEYS),
            ("whole_brain_coverage", WHOLE_BRAIN_COVERAGE_METRIC_KEYS),
            ("baseline", BASELINE_METRIC_KEYS),
            ("neighbors", NEIGHBOR_METRIC_KEYS),
            ("centroid", CENTROID_METRIC_KEYS),
            ("anatomy_distances", ANATOMY_DISTANCE_METRIC_KEYS),
            ("electrodes", ELECTRODE_METRIC_KEYS),
        ):
            extended_group_status[group_name] = "error"
            extended_group_messages[group_name] = missing_roi_message
            _set_metric_group_state(
                metric_status=extended_metric_status,
                metric_messages=extended_metric_messages,
                metric_keys=metric_keys,
                status="error",
                message=missing_roi_message,
            )
        if cfg.verbose:
            print(f"[WARN] Extended metrics failed for {cfg.subject}: {missing_roi_message}")

    pending_fields = [key for key in EXTENDED_METRIC_FIELDS if extended_metric_status.get(key) == "pending"]
    if pending_fields:
        _set_metric_group_state(
            metric_status=extended_metric_status,
            metric_messages=extended_metric_messages,
            metric_keys=pending_fields,
            status="error",
            message="Metric status was never finalized due to an internal pipeline bug.",
        )
    has_errors = any(status == "error" for status in extended_metric_status.values())
    overall_extended_status = "partial" if has_errors else "complete"
    extended_metrics_meta = {
        "schema_version": EXTENDED_METRIC_SCHEMA_VERSION,
        "status": overall_extended_status,
        "config_fingerprint": extended_config_fingerprint,
        "group_statuses": extended_group_status,
        "group_messages": extended_group_messages,
    }

    # ---- Pretty overlays for selected ROI (optional) ----
    overlay_paths = {}
    overlay_strategy = None
    overlay_attempted = False
    overlay_error = None
    if t1_img_full is not None and sel in roi_masks:
        overlay_attempted = True
        try:
            overlay_paths[sel], overlay_strategy = _generate_selected_roi_overlays(
                cfg=cfg,
                ti_img=ti_img,
                ti_data=ti_data,
                t1_img_full=t1_img_full,
                roi_mask=roi_masks[sel],
                fs_atlas_img=fs_atlas_img,
                out_dir=out_dir,
                roi_name=sel,
            )
        except Exception as exc:
            overlay_paths[sel] = []
            overlay_error = f"{type(exc).__name__}: {exc}"
            if cfg.verbose:
                print(f"[WARN] Overlay generation failed for {cfg.subject}: {overlay_error}")
    elif t1_img_full is None:
        if cfg.verbose:
            print("[INFO] Skipping overlays because no T1 background image could be loaded.")

    selected_overlay_paths = overlay_paths.get(sel, [])
    qc_meta = _build_subject_qc_meta(
        cfg=cfg,
        selected_roi=sel,
        roi_mask=roi_masks.get(sel),
        overlay_qc=_overlay_qc(
            cfg=cfg,
            overlay_paths=selected_overlay_paths,
            attempted=overlay_attempted,
            error=overlay_error,
        ),
    )
    qc_error_checks = [
        str(check)
        for check in qc_meta.get("error_checks", [])
        if isinstance(check, str)
    ]
    blocking_qc_checks = [check for check in qc_error_checks if check != "overlays"]
    nonblocking_qc_checks = [check for check in qc_error_checks if check == "overlays"]
    subject_status = (
        "complete"
        if extended_metrics_meta["status"] == "complete" and not blocking_qc_checks
        else "partial"
    )

    # ---- Subject-level robustness metrics ----
    subject_metrics = dict(
        schema_version=EXTENDED_METRIC_SCHEMA_VERSION,
        subject=cfg.subject,
        subject_metrics_meta={
            "status": subject_status,
            "extended_metrics_status": extended_metrics_meta["status"],
            "qc_status": qc_meta["status"],
            "blocking_qc_checks": blocking_qc_checks,
            "nonblocking_qc_checks": nonblocking_qc_checks,
        },
        target_roi=sel,
        target_roi_label_ids=_fastsurfer_roi_label_ids(sel),
        percentile=cfg.percentile,
        percentile_value=float(thr),
        region_percentile=cfg.region_percentile,
        voxel_volume_mm3=vox_vol,
        whole_brain_voxels=whole_brain_voxels,
        whole_brain_volume_mm3=whole_brain_volume_mm3,
        top_percentile_voxels=top_percentile_voxels,
        top_percentile_percent_of_whole_brain=top_percentile_percent_of_whole_brain,
        rois=per_roi_metrics,
        extended_metrics=extended_metrics,
        extended_metric_status=extended_metric_status,
        extended_metric_messages=extended_metric_messages,
        extended_metrics_meta=extended_metrics_meta,
        neighbor_visualization=neighbor_visualization,
        qc_meta=qc_meta,
        threshold_qc=threshold_qc,
    )
    subject_metrics = json_ready_metric_value(subject_metrics)

    metrics_path = os.path.join(out_dir, "subject_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(subject_metrics, f, indent=2)

    # ---- Return everything useful for tests / notebooks ----
    return dict(
        config=cfg,
        ti_path=ti_path,
        t1_path=t1_path,
        out_dir=out_dir,
        percentile_threshold=thr,
        topP_mask_path=topP_mask_path,
        topP_ti_path=topP_ti_path,
        per_roi=per_roi,
        overlays=overlay_paths,
        region_table=region_table_path,
        region_df=region_df,
        metrics_path=metrics_path,
        extended_metrics=extended_metrics,
        extended_metric_status=extended_metric_status,
        extended_metric_messages=extended_metric_messages,
        extended_metrics_meta=extended_metrics_meta,
        qc_meta=qc_meta,
        subject_status=subject_status,
        blocking_qc_checks=blocking_qc_checks,
        nonblocking_qc_checks=nonblocking_qc_checks,
        threshold_qc=threshold_qc,
        neighbor_table_path=neighbor_table_path,
        neighbor_visualization=neighbor_visualization,
        electrode_table_path=electrode_table_path,
        overlay_strategy=overlay_strategy,
    )

if __name__ == "__main__":
    raise SystemExit(
        "Use post/run_post_processing.py to run batch post-processing. "
        "This module is intended to be imported and called as a library."
    )
