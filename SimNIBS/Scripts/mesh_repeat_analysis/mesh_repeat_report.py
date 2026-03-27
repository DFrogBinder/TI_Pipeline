#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repeatability analysis for SimNIBS mesh outputs.

Compares per-repeat volumetric label maps derived from TI.msh, and links
mesh differences to TI summary metrics (mean/median) within an ROI
(e.g., FreeSurfer M1 label) and across the whole head model.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
from configparser import ConfigParser
from pathlib import Path
from datetime import datetime

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to


REPEAT_PREFIX = "repeat_"
LOG_FILE: Path | None = None
TISSUE_LABELS = list(range(1, 11))
DEFAULT_REPEATABILITY_ROOTS = [
    Path("/media/boyan/main/PhD/CamCan-SimNIBS_Repeatability/new_params"),
    Path("/media/boyan/main/PhD/CamCan-SimNIBS_Repeatability/simulation-data"),
    Path("/media/boyan/main/PhD/CamCan-SimNIBS_Repeatability"),
]
ROI_LABEL_PRESETS = {
    "ctxlhprecentral": [1022],
    "m1": [1022],
    "leftm1": [1022],
    "lefthippocampus": [17],
    "righthippocampus": [53],
}


def _short_path(value: str, max_len: int = 120) -> str:
    if len(value) <= max_len:
        return value
    return "…" + value[-(max_len - 1):]


def _fmt_value(value: object) -> str:
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str) and ("/" in value or "\\" in value):
        return _short_path(value)
    return str(value)


def _human_lines(event: str, fields: dict) -> list[str]:
    timestamp = datetime.now().strftime("%H:%M:%S")
    header = f"[{timestamp}] {event.replace('_', ' ').capitalize()}"
    items = [(k, v) for k, v in fields.items() if v is not None]
    if not items:
        return [header]
    if len(items) <= 2 and all(len(_fmt_value(v)) <= 60 for _, v in items):
        tail = " | ".join(f"{k}: {_fmt_value(v)}" for k, v in items)
        return [f"{header} | {tail}"]
    lines = [header]
    for key, value in items:
        lines.append(f"  - {key}: {_fmt_value(value)}")
    return lines


def log_event(event: str, **fields) -> None:
    # Human-readable stdout
    for line in _human_lines(event, fields):
        print(line)
    # JSONL file logging (optional)
    if LOG_FILE:
        payload = {"event": event, **fields}
        line = json.dumps(payload, default=str)
        try:
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with LOG_FILE.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass


def _find_repeat_dirs(repeats_root: Path, subject: str) -> list[Path]:
    repeat_dirs = []
    if not repeats_root.exists():
        return repeat_dirs
    for child in sorted(repeats_root.iterdir()):
        if child.is_dir() and child.name.startswith(REPEAT_PREFIX):
            anat_dir = child / subject / "anat"
            if anat_dir.is_dir():
                repeat_dirs.append(anat_dir)
    return repeat_dirs


def _find_volume_candidate(parent: Path, prefix: str) -> Path | None:
    if not parent.is_dir():
        return None
    candidates = sorted([p for p in parent.iterdir() if p.name.startswith(prefix)])
    return candidates[0] if candidates else None


def _run_msh2nii(ti_msh: Path, t1_path: Path, out_base: Path, *, mode: str) -> None:
    cmd = ["msh2nii", str(ti_msh), str(t1_path), str(out_base), mode]
    log_event("run_cmd", label="msh2nii", cmd=cmd, cwd=str(out_base.parent))
    result = subprocess.run(cmd, cwd=str(out_base.parent), capture_output=True, text=True)
    log_event(
        "cmd_result",
        label="msh2nii",
        returncode=result.returncode,
        stdout_tail=result.stdout[-2000:] if result.stdout else "",
        stderr_tail=result.stderr[-2000:] if result.stderr else "",
    )
    result.check_returncode()


def _load_or_create_volumes(anat_dir: Path, subject: str, t1_path: Path) -> tuple[Path, Path, Path]:
    out_dir = anat_dir / "SimNIBS" / "Output" / subject
    labels_dir = out_dir / "Volume_Labels"
    base_dir = out_dir / "Volume_Base"
    ti_brain_only = anat_dir / "SimNIBS" / "ti_brain_only.nii.gz"

    label_file = _find_volume_candidate(labels_dir, "TI_Volumetric_")
    base_file = _find_volume_candidate(base_dir, "TI_Volumetric_")

    if label_file and base_file and ti_brain_only.exists():
        log_event(
            "volume_found",
            subject=subject,
            anat_dir=str(anat_dir),
            label=str(label_file),
            base=str(base_file),
            ti_brain_only=str(ti_brain_only),
        )
        return label_file, base_file, ti_brain_only

    ti_msh = out_dir / "TI.msh"
    if not ti_msh.exists():
        log_event("missing", kind="ti_msh", path=str(ti_msh))
        raise FileNotFoundError(f"Missing TI.msh at {ti_msh}")
    if not t1_path.exists():
        log_event("missing", kind="t1", path=str(t1_path))
        raise FileNotFoundError(f"Missing T1 at {t1_path}")

    labels_dir.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)

    label_base = labels_dir / "TI_Volumetric_Labels"
    base_base = base_dir / "TI_Volumetric_Base"

    if not label_file:
        _run_msh2nii(ti_msh, t1_path, label_base, mode="--create_label")
        label_file = _find_volume_candidate(labels_dir, "TI_Volumetric_")
    if not base_file:
        _run_msh2nii(ti_msh, t1_path, base_base, mode="--create_base")
        base_file = _find_volume_candidate(base_dir, "TI_Volumetric_")

    if not label_file or not base_file:
        log_event(
            "missing",
            kind="ti_volumes",
            label=str(label_file) if label_file else None,
            base=str(base_file) if base_file else None,
        )
        raise FileNotFoundError("Failed to generate label/base volumes with msh2nii.")

    if not ti_brain_only.exists():
        log_event("missing", kind="ti_brain_only", path=str(ti_brain_only))
        raise FileNotFoundError(f"Missing ti_brain_only at {ti_brain_only}")

    return label_file, base_file, ti_brain_only


def _ensure_label_grid(label_img: nib.Nifti1Image, ref_img: nib.Nifti1Image, *, label: str) -> nib.Nifti1Image:
    same_shape = label_img.shape == ref_img.shape
    same_affine = np.allclose(label_img.affine, ref_img.affine, atol=1e-4)
    log_event(
        "grid_check",
        label=label,
        same_shape=same_shape,
        same_affine=same_affine,
        src_shape=label_img.shape,
        ref_shape=ref_img.shape,
    )
    if same_shape and same_affine:
        return label_img
    log_event("resample", label=label, mode="nearest")
    return resample_from_to(label_img, ref_img, order=0)


def _ensure_scalar_grid(scalar_img: nib.Nifti1Image, ref_img: nib.Nifti1Image, *, label: str) -> nib.Nifti1Image:
    same_shape = scalar_img.shape == ref_img.shape
    same_affine = np.allclose(scalar_img.affine, ref_img.affine, atol=1e-4)
    log_event(
        "grid_check",
        label=label,
        same_shape=same_shape,
        same_affine=same_affine,
        src_shape=scalar_img.shape,
        ref_shape=ref_img.shape,
    )
    if same_shape and same_affine:
        return scalar_img
    log_event("resample", label=label, mode="linear")
    return resample_from_to(scalar_img, ref_img, order=1)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return (2.0 * inter / denom) if denom > 0 else np.nan


def _count_msh_nodes_fallback(ti_msh: Path) -> float:
    try:
        with ti_msh.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.strip() == "$Nodes":
                    header = next(fh, "").strip().split()
                    if len(header) == 1:
                        return float(int(header[0]))
                    if len(header) >= 4:
                        # Gmsh v4.x: numEntityBlocks numNodes minNodeTag maxNodeTag
                        return float(int(header[1]))
                    break
    except Exception as exc:
        log_event("mesh_fallback_error", path=str(ti_msh), error=str(exc))
    return float("nan")


def _count_msh_nodes(ti_msh: Path) -> float:
    try:
        import meshio
    except Exception:
        log_event(
            "missing_dep",
            kind="meshio",
            note="meshio not available; using fallback .msh parser",
        )
        return _count_msh_nodes_fallback(ti_msh)
    try:
        mesh = meshio.read(str(ti_msh))
        return float(mesh.points.shape[0])
    except Exception as exc:
        log_event("mesh_read_error", path=str(ti_msh), error=str(exc))
        fallback = _count_msh_nodes_fallback(ti_msh)
        if not np.isnan(fallback):
            log_event("mesh_fallback_used", path=str(ti_msh), nodes=fallback)
        return fallback


def _label_name(label_id: int) -> str:
    # Common SimNIBS tissue labels. Unknowns fallback to numeric.
    label_map = {
        0: "Background",
        1: "WM",
        2: "GM",
        3: "CSF",
        4: "Bone",
        5: "Scalp",
        6: "Eyes",
        7: "Compact Bone",
        8: "Spongy Bone",
        9: "Blood",
        10: "Muscle",
    }
    return label_map.get(int(label_id), f"Label {label_id}")


def _plot_diff_overlay(
    t1_img: nib.Nifti1Image,
    diff_freq: np.ndarray,
    out_path: Path,
    *,
    title: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    t1_data = np.asarray(t1_img.dataobj, dtype=np.float32)
    mid = tuple(s // 2 for s in t1_data.shape)
    slices = [
        (t1_data[mid[0], :, :], diff_freq[mid[0], :, :], "sagittal"),
        (t1_data[:, mid[1], :], diff_freq[:, mid[1], :], "coronal"),
        (t1_data[:, :, mid[2]], diff_freq[:, :, mid[2]], "axial"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (bg, fg, slice_name) in zip(axes, slices, strict=False):
        ax.imshow(bg.T, cmap="gray", origin="lower")
        overlay = np.ma.masked_where(fg <= 0, fg)
        ax.imshow(overlay.T, cmap="hot", alpha=0.7, origin="lower", vmin=0, vmax=1)
        ax.set_title(slice_name)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _roi_center_ijk(mask: np.ndarray) -> tuple[int, int, int]:
    idx = np.argwhere(mask)
    if idx.size == 0:
        return tuple(int(s // 2) for s in mask.shape)
    center = np.rint(idx.mean(axis=0)).astype(int)
    return tuple(
        int(np.clip(center[i], 0, mask.shape[i] - 1))
        for i in range(mask.ndim)
    )


def _plot_roi_qc(
    t1_img: nib.Nifti1Image,
    roi_mask: np.ndarray,
    out_path: Path,
    *,
    title: str,
    ti_data: np.ndarray | None = None,
    ti_label: str = "TI (V/m)",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    t1_data = np.asarray(t1_img.dataobj, dtype=np.float32)
    center = _roi_center_ijk(roi_mask)
    slices = [
        ("sagittal", t1_data[center[0], :, :], roi_mask[center[0], :, :], ti_data[center[0], :, :] if ti_data is not None else None),
        ("coronal", t1_data[:, center[1], :], roi_mask[:, center[1], :], ti_data[:, center[1], :] if ti_data is not None else None),
        ("axial", t1_data[:, :, center[2]], roi_mask[:, :, center[2]], ti_data[:, :, center[2]] if ti_data is not None else None),
    ]

    if ti_data is not None:
        valid_ti = _finite_values(ti_data[np.isfinite(ti_data) & (ti_data > 0)])
        if valid_ti.size:
            vmin = float(np.nanpercentile(valid_ti, 5))
            vmax = float(np.nanpercentile(valid_ti, 99))
            if vmax <= vmin:
                vmax = float(np.nanmax(valid_ti))
                vmin = float(np.nanmin(valid_ti))
        else:
            vmin, vmax = 0.0, 1.0
    else:
        vmin = vmax = None

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    overlay_artist = None
    for ax, (slice_name, bg, mask2d, ti2d) in zip(axes, slices, strict=False):
        ax.imshow(bg.T, cmap="gray", origin="lower")
        if ti2d is not None:
            overlay = np.ma.masked_where(~np.isfinite(ti2d.T) | (ti2d.T <= 0), ti2d.T)
            overlay_artist = ax.imshow(
                overlay,
                cmap="viridis",
                alpha=0.65,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
        if np.any(mask2d):
            ax.contour(
                mask2d.T.astype(np.float32),
                levels=[0.5],
                colors=["cyan"],
                linewidths=1.2,
                origin="lower",
            )
        ax.set_title(slice_name)
        ax.axis("off")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    if overlay_artist is not None:
        cbar = fig.colorbar(overlay_artist, ax=axes, fraction=0.03, pad=0.02)
        cbar.set_label(ti_label)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _resolve_t1_path(subject: str, anat_dir: Path, t1_root: Path | None) -> Path:
    local_t1 = anat_dir / f"{subject}_T1w.nii"
    if local_t1.exists():
        log_event("t1_resolve", subject=subject, source="repeat_anat", path=str(local_t1))
        return local_t1
    if t1_root:
        alt = t1_root / f"{subject}_repeatability" / subject / "anat" / f"{subject}_T1w.nii"
        if alt.exists():
            log_event("t1_resolve", subject=subject, source="t1_root", path=str(alt))
            return alt
    log_event("t1_resolve", subject=subject, source="missing", path=str(local_t1))
    return local_t1


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _normalize_roi_key(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def _infer_roi_labels(roi_name: str) -> list[int] | None:
    return ROI_LABEL_PRESETS.get(_normalize_roi_key(roi_name))


def _candidate_atlas_dirs(rootdir: Path | None) -> list[Path]:
    script_path = Path(__file__).resolve()
    candidates = [
        Path.cwd() / "atlases",
        script_path.parent / "atlases",
        Path.home() / "sandbox" / "Jake_Data" / "atlases",
    ]
    if len(script_path.parents) > 4:
        candidates.append(script_path.parents[4] / "Jake_Data" / "atlases")
    if rootdir is not None:
        candidates.extend(
            [
                rootdir / "atlases",
                rootdir.parent / "atlases",
            ]
        )

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for path in candidates:
        resolved = path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(resolved)
    return unique_candidates


def _resolve_atlas_path(subject: str, atlas: str | None, atlas_dir: str | None, rootdir: Path | None) -> Path:
    if atlas:
        atlas_path = Path(atlas).expanduser()
        if not atlas_path.exists():
            raise SystemExit(f"Atlas not found: {atlas_path}")
        log_event("atlas_resolve", source="explicit", path=str(atlas_path))
        return atlas_path

    candidates: list[Path] = []
    if atlas_dir:
        candidates.append(Path(atlas_dir).expanduser() / f"{subject}.nii.gz")
    else:
        for directory in _candidate_atlas_dirs(rootdir):
            candidates.append(directory / f"{subject}.nii.gz")

    for candidate in candidates:
        if candidate.exists():
            log_event("atlas_resolve", source="auto", path=str(candidate))
            return candidate

    attempted = ", ".join(str(path) for path in candidates[:6])
    raise SystemExit(
        "Atlas not found. Provide --atlas or place the subject atlas at one of the "
        f"expected locations (first candidates: {attempted})."
    )


def _candidate_repeats_roots(base: Path, subject: str) -> list[Path]:
    return [
        base / f"{subject}_repeatability" / "repeats",
        base / "repeats",
    ]


def _canonical_repeatability_base(path: Path, subject: str) -> Path:
    if path.name == "repeats":
        if path.parent.name == f"{subject}_repeatability":
            return path.parent.parent
        return path.parent
    if path.name == f"{subject}_repeatability":
        return path.parent
    return path


def _resolve_rootdir_and_repeats(
    subject: str,
    rootdir: str | None,
    repeats_dir: str | None,
) -> tuple[Path | None, Path]:
    if repeats_dir:
        repeats_root = Path(repeats_dir).expanduser()
        if rootdir:
            base_root = _canonical_repeatability_base(Path(rootdir).expanduser(), subject)
        else:
            base_root = _canonical_repeatability_base(repeats_root, subject)
        log_event("repeats_resolve", source="explicit", path=str(repeats_root))
        return base_root, repeats_root

    candidate_roots: list[Path] = []
    if rootdir:
        candidate_roots.append(_canonical_repeatability_base(Path(rootdir).expanduser(), subject))
    else:
        candidate_roots.extend(DEFAULT_REPEATABILITY_ROOTS)

    seen: set[Path] = set()
    for base in candidate_roots:
        if base in seen:
            continue
        seen.add(base)
        for repeats_root in _candidate_repeats_roots(base, subject):
            if repeats_root.exists():
                log_event("repeats_resolve", source="auto", rootdir=str(base), path=str(repeats_root))
                return base, repeats_root

    attempted = [str(path) for base in candidate_roots for path in _candidate_repeats_roots(base, subject)]
    raise SystemExit(
        "No repeat directory found. Provide --rootdir/--repeats-dir or place the dataset at one "
        f"of the expected locations (first candidates: {', '.join(attempted[:6])})."
    )


def _canonical_batch_root(path: Path) -> Path:
    if path.name == "repeats" and path.parent.name.endswith("_repeatability"):
        return path.parent.parent
    if path.name.endswith("_repeatability"):
        return path.parent
    return path


def _discover_repeatability_subjects(rootdir: Path) -> list[str]:
    suffix = "_repeatability"
    if rootdir.name == "repeats" and rootdir.parent.name.endswith(suffix):
        subject = rootdir.parent.name[: -len(suffix)]
        return [subject] if subject else []
    if rootdir.name.endswith(suffix):
        subject = rootdir.name[: -len(suffix)]
        return [subject] if subject else []
    if not rootdir.exists():
        return []

    subjects: list[str] = []
    for child in sorted(rootdir.iterdir()):
        if not child.is_dir() or not child.name.endswith(suffix):
            continue
        subject = child.name[: -len(suffix)]
        if subject and (child / "repeats").is_dir():
            subjects.append(subject)
    return subjects


def _resolve_batch_root(rootdir: str | None) -> Path:
    candidate_roots: list[Path] = []
    if rootdir:
        candidate_roots.append(_canonical_batch_root(Path(rootdir).expanduser()))
    else:
        candidate_roots.extend(DEFAULT_REPEATABILITY_ROOTS)

    seen: set[Path] = set()
    for base in candidate_roots:
        if base in seen:
            continue
        seen.add(base)
        subjects = _discover_repeatability_subjects(base)
        if subjects:
            log_event("batch_root_resolve", source="auto" if rootdir is None else "explicit", path=str(base), subjects=len(subjects))
            return base

    attempted = ", ".join(str(path) for path in candidate_roots[:6])
    raise SystemExit(
        "No repeatability studies found for batch mode. Provide --rootdir pointing at the parent "
        f"folder that contains <subject>_repeatability directories (first candidates: {attempted})."
    )


def _finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _metric_stats(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = _finite_values(np.asarray(values, dtype=np.float64))
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "cv": float("nan"),
            "cv_percent": float("nan"),
        }

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    cv = float(std / mean) if mean != 0 else float("nan")
    return {
        "n": int(arr.size),
        "mean": mean,
        "std": std,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "cv": cv,
        "cv_percent": float(cv * 100.0) if np.isfinite(cv) else float("nan"),
    }


def _normalize_ti_units(
    data: np.ndarray,
    mask: np.ndarray,
    *,
    label: str,
    threshold_v_per_m: float = 10.0,
) -> tuple[np.ndarray, float]:
    valid = _finite_values(data[mask])
    if valid.size == 0:
        return data, 1.0
    p999 = float(np.nanpercentile(valid, 99.9))
    if p999 > threshold_v_per_m:
        log_event(
            "unit_rescale",
            label=label,
            percentile_99_9=p999,
            applied_scale=0.001,
            note="Assuming mV/m-like values and converting to V/m.",
        )
        return data * 0.001, 0.001
    return data, 1.0


def _json_string(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def _save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (dict, list, tuple)):
                    serializable[key] = _json_string(value)
                else:
                    serializable[key] = value
            writer.writerow(serializable)


def _peak_info(data: np.ndarray, mask: np.ndarray, affine: np.ndarray) -> dict[str, object]:
    valid = np.isfinite(data) & mask
    if not np.any(valid):
        return {
            "value": float("nan"),
            "ijk": None,
            "xyz_mm": None,
        }
    ijk_all = np.argwhere(valid)
    vals = data[valid]
    peak_idx = int(np.nanargmax(vals))
    ijk = ijk_all[peak_idx]
    xyz = nib.affines.apply_affine(affine, ijk)
    return {
        "value": float(vals[peak_idx]),
        "ijk": [int(x) for x in ijk.tolist()],
        "xyz_mm": [float(x) for x in xyz.tolist()],
    }


def _top_percentile_mask(data: np.ndarray, mask: np.ndarray, percentile: float) -> np.ndarray:
    valid = np.isfinite(data) & mask
    if not np.any(valid):
        return np.zeros(mask.shape, dtype=bool)
    threshold = float(np.nanpercentile(data[valid], percentile))
    return valid & (data >= threshold)


def _centroid_xyz(mask: np.ndarray, affine: np.ndarray) -> list[float] | None:
    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    xyz = nib.affines.apply_affine(affine, idx)
    center = np.mean(xyz, axis=0)
    return [float(x) for x in center.tolist()]


def _distance_mm(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None:
        return float("nan")
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a_arr - b_arr))


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_scalar_list(values: object) -> list[float]:
    if values is None:
        return []
    arr = np.atleast_1d(values)
    out: list[float] = []
    for value in arr.tolist():
        if isinstance(value, (list, tuple, np.ndarray)):
            flat = np.asarray(value).ravel()
            for item in flat.tolist():
                if item is None or item == "":
                    continue
                out.append(float(item))
        elif value is not None and value != "":
            out.append(float(value))
    return out


def _normalize_electrodes(raw: object) -> list[dict[str, object]]:
    if raw is None:
        return []
    electrodes = raw if isinstance(raw, list) else [raw]
    out = []
    for elec in electrodes:
        if not isinstance(elec, dict):
            continue
        out.append(
            {
                "centre": str(elec.get("centre", "")),
                "channelnr": int(elec.get("channelnr", 0)) if elec.get("channelnr", "") != "" else None,
                "shape": str(elec.get("shape", "")),
                "dimensions": _normalize_scalar_list(elec.get("dimensions")),
                "thickness": float(elec.get("thickness")) if elec.get("thickness", "") != "" else None,
            }
        )
    return out


def _normalize_conductivities(raw: object) -> dict[str, float]:
    if raw is None:
        return {}
    conds = raw if isinstance(raw, list) else [raw]
    out: dict[str, float] = {}
    for cond in conds:
        if not isinstance(cond, dict):
            continue
        name = cond.get("name")
        value = cond.get("value")
        if name is None:
            continue
        if isinstance(name, np.ndarray):
            if name.size == 0:
                continue
            flat_name = name.ravel().tolist()
            name = flat_name[0] if flat_name else None
        if name in ("", []):
            continue
        vals = _normalize_scalar_list(value)
        if len(vals) == 1:
            out[str(name)] = float(vals[0])
    return out


def _extract_sim_signature(mat_path: Path) -> dict[str, object] | None:
    try:
        from scipy.io import loadmat
    except Exception:
        log_event("missing_dep", kind="scipy", note="Skipping .mat simulation signature checks.")
        return None

    if not mat_path.exists():
        return None

    try:
        data = loadmat(str(mat_path), simplify_cells=True)
    except Exception as exc:
        log_event("mat_read_error", path=str(mat_path), error=str(exc))
        return None

    poslist = data.get("poslist", [])
    pos_items = poslist if isinstance(poslist, list) else [poslist]
    montages = []
    for pos in pos_items:
        if not isinstance(pos, dict):
            continue
        montages.append(
            {
                "currents": _normalize_scalar_list(pos.get("currents")),
                "solver_options": str(pos.get("solver_options", "")),
                "conductivities": _normalize_conductivities(pos.get("cond")),
                "electrodes": _normalize_electrodes(pos.get("electrode")),
            }
        )

    return {
        "map_to_vol": bool(np.asarray(data.get("map_to_vol", False)).squeeze()),
        "tissues_in_niftis": int(np.asarray(data.get("tissues_in_niftis", 0)).squeeze()),
        "montages": montages,
    }


def _load_repeat_signatures(repeat_anat_dirs: list[Path], subject: str) -> dict[str, object]:
    settings_hashes: dict[str, str | None] = {}
    t1_hashes: dict[str, str | None] = {}
    t2_hashes: dict[str, str | None] = {}
    seg_hashes: dict[str, str | None] = {}
    sim_signatures: dict[str, dict[str, object]] = {}

    for anat_dir in repeat_anat_dirs:
        repeat_tag = anat_dir.parent.parent.name
        settings_path = anat_dir / f"m2m_{subject}" / "settings.ini"
        t1_path = anat_dir / f"{subject}_T1w.nii"
        t2_path = anat_dir / f"{subject}_T2w.nii"
        seg_path = anat_dir / f"{subject}_T1w_ras_1mm_T1andT2_masks.nii"
        mat_candidates = sorted((anat_dir / "SimNIBS" / "Output" / subject).glob("simnibs_simulation_*.mat"))

        settings_hashes[repeat_tag] = _sha256_file(settings_path)
        t1_hashes[repeat_tag] = _sha256_file(t1_path)
        t2_hashes[repeat_tag] = _sha256_file(t2_path)
        seg_hashes[repeat_tag] = _sha256_file(seg_path)
        if mat_candidates:
            signature = _extract_sim_signature(mat_candidates[0])
            if signature is not None:
                sim_signatures[repeat_tag] = signature

    def _identical_or_none(values: dict[str, str | None]) -> bool | None:
        present = [v for v in values.values() if v is not None]
        if not present:
            return None
        return len(set(present)) == 1

    sim_signature_keys = {_json_string(v) for v in sim_signatures.values()}
    return {
        "settings_ini_hashes": settings_hashes,
        "t1_hashes": t1_hashes,
        "t2_hashes": t2_hashes,
        "segmentation_hashes": seg_hashes,
        "settings_ini_identical": _identical_or_none(settings_hashes),
        "t1_identical": _identical_or_none(t1_hashes),
        "t2_identical": _identical_or_none(t2_hashes),
        "segmentation_identical": _identical_or_none(seg_hashes),
        "simulation_signature_identical": len(sim_signature_keys) == 1 if sim_signatures else None,
        "simulation_signature_reference": next(iter(sim_signatures.values()), None),
    }


def _load_cohort_metric(
    cohort_root: Path,
    *,
    region_name: str | None,
    region_label: int | None,
    metric_name: str,
) -> dict[str, object]:
    values = []
    matched_subjects = []
    subject_value = None
    for subj_dir in sorted(cohort_root.iterdir()):
        if not subj_dir.is_dir():
            continue
        csv_path = subj_dir / "anat" / "post" / "region_stats_fastsurfer.csv"
        if not csv_path.is_file():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            matches = []
            for row in reader:
                label_ok = region_label is not None and int(row.get("label_id", -1)) == region_label
                name_ok = region_name is not None and row.get("label_name", "").strip().lower() == region_name.lower()
                if label_ok or name_ok:
                    matches.append(row)
            if len(matches) != 1:
                continue
            value = float(matches[0][metric_name])
            if np.isfinite(value):
                values.append(value)
                matched_subjects.append(subj_dir.name)

    stats = _metric_stats(values)
    return {
        "subjects": matched_subjects,
        "values": values,
        "stats": stats,
    }


def _percentile_rank(values: list[float], value: float) -> float:
    arr = _finite_values(np.asarray(values, dtype=np.float64))
    if arr.size == 0 or not np.isfinite(value):
        return float("nan")
    return float(100.0 * np.mean(arr <= value))


def _write_markdown_report(
    path: Path,
    *,
    subject: str,
    roi_name: str,
    roi_labels: list[int],
    reference_repeat: str,
    repeats: int,
    primary_repeatability: dict[str, float],
    repeatability_rows: list[dict[str, object]],
    parameter_consistency: dict[str, object],
    cohort_comparison: dict[str, object] | None,
) -> None:
    def _fmt_bool(value: bool | None) -> str:
        if value is None:
            return "not checked"
        return str(value)

    lines = [
        f"# Repeatability Report: {subject}",
        "",
        f"- ROI: {roi_name}",
        f"- ROI labels: {', '.join(str(x) for x in roi_labels)}",
        f"- Reference repeat: {reference_repeat}",
        f"- Repeats analysed: {repeats}",
        "",
        "## Primary Metric: Median TI in ROI Across Repeats",
        "",
        "| Mean | SD | Min | Max | CV |",
        "| --- | --- | --- | --- | --- |",
        (
            f"| {primary_repeatability['mean']:.6f} | {primary_repeatability['std']:.6f} | "
            f"{primary_repeatability['min']:.6f} | {primary_repeatability['max']:.6f} | "
            f"{primary_repeatability['cv_percent']:.2f}% |"
        ),
        "",
        "## Additional Repeatability Metrics",
        "",
        "| Metric | Mean | SD | Min | Max | CV |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in repeatability_rows:
        lines.append(
            f"| {row['metric']} | {float(row['mean']):.6f} | {float(row['std']):.6f} | "
            f"{float(row['min']):.6f} | {float(row['max']):.6f} | {float(row['cv_percent']):.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Parameter Consistency Across Repeats",
            "",
            f"- T1 identical: {_fmt_bool(parameter_consistency['t1_identical'])}",
            f"- T2 identical: {_fmt_bool(parameter_consistency['t2_identical'])}",
            f"- Segmentation identical: {_fmt_bool(parameter_consistency['segmentation_identical'])}",
            f"- CHARM settings identical: {_fmt_bool(parameter_consistency['settings_ini_identical'])}",
            f"- Electrode/current/conductivity signature identical: {_fmt_bool(parameter_consistency['simulation_signature_identical'])}",
        ]
    )

    if cohort_comparison is not None:
        lines.extend(
            [
                "",
                "## Cohort Comparison",
                "",
                f"- Cohort metric: {cohort_comparison['metric_name']}",
                f"- Cohort subjects matched: {cohort_comparison['cohort_subjects']}",
                f"- Within-subject SD: {cohort_comparison['within_std']:.6f}",
                f"- Between-subject SD: {cohort_comparison['between_std']:.6f}",
                f"- Between/within SD ratio: {cohort_comparison['between_to_within_sd_ratio']:.3f}",
                f"- Within-subject range: {cohort_comparison['within_range']:.6f}",
                f"- Between-subject range: {cohort_comparison['between_range']:.6f}",
                f"- Between/within range ratio: {cohort_comparison['between_to_within_range_ratio']:.3f}",
            ]
        )
        if cohort_comparison.get("subject_percentile") is not None and np.isfinite(cohort_comparison["subject_percentile"]):
            lines.append(f"- Subject percentile in cohort: {cohort_comparison['subject_percentile']:.1f}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_subject_analysis(
    args: argparse.Namespace,
    *,
    subject: str,
    roi_name: str,
    roi_labels: list[int],
    rootdir_override: Path | None = None,
    output_dir_override: Path | None = None,
) -> dict[str, object]:
    rootdir, repeats_root = _resolve_rootdir_and_repeats(
        subject,
        str(rootdir_override) if rootdir_override is not None else args.rootdir,
        args.repeats_dir,
    )

    log_event("repeat_root", subject=subject, path=str(repeats_root))
    repeat_anat_dirs = _find_repeat_dirs(repeats_root, subject)
    if not repeat_anat_dirs:
        raise SystemExit(f"No repeats found under {repeats_root}")

    if args.reference_repeat:
        ref_anat = repeats_root / args.reference_repeat / subject / "anat"
        if ref_anat not in repeat_anat_dirs:
            raise SystemExit(f"Reference repeat not found: {ref_anat}")
    else:
        ref_anat = repeat_anat_dirs[0]

    output_dir = output_dir_override or (
        Path(args.output_dir) if args.output_dir else repeats_root / "_analysis" / subject
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_event("output_dir", subject=subject, path=str(output_dir))
    repeat_qc_dir = output_dir / "repeat_qc"
    repeat_qc_dir.mkdir(parents=True, exist_ok=True)

    atlas_path = _resolve_atlas_path(subject, args.atlas, args.atlas_dir, rootdir)
    log_event("atlas_load", subject=subject, path=str(atlas_path))
    atlas_img = nib.load(atlas_path)
    t1_root = Path(args.t1_root) if args.t1_root else rootdir
    log_event("t1_root", subject=subject, path=str(t1_root))
    ref_t1_path = _resolve_t1_path(subject, ref_anat, t1_root)
    ref_t1 = nib.load(ref_t1_path)
    atlas_img = _ensure_label_grid(atlas_img, ref_t1, label="atlas_to_t1")
    atlas_data = np.asarray(atlas_img.dataobj).astype(np.int32, copy=False)

    roi_mask = np.isin(atlas_data, roi_labels)
    if not np.any(roi_mask):
        raise SystemExit(
            f"{roi_name} mask is empty after resampling; check label IDs and atlas alignment."
        )

    ref_label_path, _, ref_ti_path = _load_or_create_volumes(ref_anat, subject, ref_t1_path)
    ref_label_img = nib.load(ref_label_path)
    ref_label_img = _ensure_label_grid(ref_label_img, ref_t1, label="ref_labels_to_t1")
    ref_labels = np.asarray(ref_label_img.dataobj).astype(np.int32, copy=False)
    ref_ti_img = _ensure_scalar_grid(nib.load(ref_ti_path), ref_t1, label="ref_ti_to_t1")
    ref_ti_data = np.asarray(ref_ti_img.dataobj, dtype=np.float32)
    ref_head_mask = ref_labels > 0
    ref_ti_data, ref_ti_scale_factor = _normalize_ti_units(
        ref_ti_data,
        ref_head_mask,
        label="ref_ti_to_t1",
    )
    ref_peak_head = _peak_info(ref_ti_data, ref_head_mask, ref_t1.affine)
    ref_peak_roi = _peak_info(ref_ti_data, roi_mask, ref_t1.affine)
    ref_high_field_mask = _top_percentile_mask(ref_ti_data, ref_head_mask, args.spatial_percentile)
    ref_high_field_centroid = _centroid_xyz(ref_high_field_mask, ref_t1.affine)

    summary_rows = []
    skipped_repeats = []
    diff_counts = np.zeros(ref_labels.shape, dtype=np.int32)
    ti_sum = np.zeros(ref_labels.shape, dtype=np.float64)
    label_set = sorted(set(np.unique(ref_labels)))
    label_presence: dict[int, list[str]] = {}
    label_counts: dict[str, dict[int, int]] = {}

    if args.peak_percentile is not None:
        log_event("deprecated_arg_ignored", arg="--peak-percentile", value=args.peak_percentile)

    for anat_dir in repeat_anat_dirs:
        repeat_tag = anat_dir.parent.parent.name
        try:
            t1_path = _resolve_t1_path(subject, anat_dir, t1_root)
            label_path, _, ti_path = _load_or_create_volumes(anat_dir, subject, t1_path)
        except FileNotFoundError as exc:
            skipped_repeats.append({"repeat_tag": repeat_tag, "reason": str(exc)})
            log_event("repeat_skip", subject=subject, repeat_tag=repeat_tag, reason=str(exc))
            continue

        label_img = _ensure_label_grid(nib.load(label_path), ref_t1, label=f"{repeat_tag}_labels_to_t1")
        labels = np.asarray(label_img.dataobj).astype(np.int32, copy=False)

        ti_img = _ensure_scalar_grid(nib.load(ti_path), ref_t1, label=f"{repeat_tag}_ti_to_t1")
        base_data = np.asarray(ti_img.dataobj, dtype=np.float32)
        head_mask = labels > 0
        base_data, ti_scale_factor = _normalize_ti_units(
            base_data,
            head_mask,
            label=f"{repeat_tag}_ti_to_t1",
        )

        diff = labels != ref_labels
        diff_counts += diff.astype(np.int32)
        ti_sum += base_data.astype(np.float64, copy=False)
        _plot_roi_qc(
            ref_t1,
            roi_mask,
            repeat_qc_dir / f"{repeat_tag}_roi_outline_on_ti.png",
            title=(
                f"{repeat_tag} | {roi_name} outline on TI + T1 | "
                f"labels: {', '.join(str(x) for x in roi_labels)}"
            ),
            ti_data=base_data,
        )

        diff_fraction = float(diff.mean())
        diff_fraction_roi = float(diff[roi_mask].mean())

        roi_vals = _finite_values(base_data[roi_mask])
        mean_roi = float(np.nanmean(roi_vals)) if roi_vals.size else float("nan")
        median_roi = float(np.nanmedian(roi_vals)) if roi_vals.size else float("nan")
        peak_roi_info = _peak_info(base_data, roi_mask, ref_t1.affine)
        peak_roi = float(peak_roi_info["value"])

        head_vals = _finite_values(base_data[head_mask])
        mean_head = float(np.nanmean(head_vals)) if head_vals.size else float("nan")
        median_head = float(np.nanmedian(head_vals)) if head_vals.size else float("nan")
        peak_head_info = _peak_info(base_data, head_mask, ref_t1.affine)
        peak_head = float(peak_head_info["value"])
        high_field_mask = _top_percentile_mask(base_data, head_mask, args.spatial_percentile)
        high_field_dice_head = _dice(high_field_mask, ref_high_field_mask)
        high_field_centroid = _centroid_xyz(high_field_mask, ref_t1.affine)
        high_field_centroid_distance_mm = _distance_mm(high_field_centroid, ref_high_field_centroid)
        hotspot_distance_head_mm = _distance_mm(
            peak_head_info["xyz_mm"], ref_peak_head["xyz_mm"]
        )
        hotspot_distance_roi_mm = _distance_mm(
            peak_roi_info["xyz_mm"], ref_peak_roi["xyz_mm"]
        )

        dice_by_label = {}
        for lab in label_set:
            a = labels == lab
            b = ref_labels == lab
            dice_by_label[int(lab)] = _dice(a, b)

        label_ids = sorted(set(np.unique(labels)) - {0})
        for lab in label_ids:
            label_presence.setdefault(int(lab), []).append(repeat_tag)

        counts = np.bincount(labels.ravel(), minlength=max(TISSUE_LABELS) + 1)
        label_counts[repeat_tag] = {lab: int(counts[lab]) for lab in TISSUE_LABELS}

        ti_msh_path = anat_dir / "SimNIBS" / "Output" / subject / "TI.msh"
        if not ti_msh_path.exists():
            log_event("missing", subject=subject, kind="ti_msh", path=str(ti_msh_path))
            mesh_nodes = float("nan")
        else:
            mesh_nodes = _count_msh_nodes(ti_msh_path)

        summary_rows.append(
            {
                "repeat_tag": repeat_tag,
                "diff_fraction": diff_fraction,
                "diff_fraction_roi": diff_fraction_roi,
                "mean_roi": mean_roi,
                "median_roi": median_roi,
                "peak_roi": peak_roi,
                "mean_head": mean_head,
                "median_head": median_head,
                "peak_head": peak_head,
                "hotspot_distance_head_mm": hotspot_distance_head_mm,
                "hotspot_distance_roi_mm": hotspot_distance_roi_mm,
                "high_field_dice_head": high_field_dice_head,
                "high_field_centroid_distance_mm": high_field_centroid_distance_mm,
                "peak_head_ijk": peak_head_info["ijk"],
                "peak_head_xyz_mm": peak_head_info["xyz_mm"],
                "peak_roi_ijk": peak_roi_info["ijk"],
                "peak_roi_xyz_mm": peak_roi_info["xyz_mm"],
                "ti_scale_factor": ti_scale_factor,
                "mesh_nodes": mesh_nodes,
                "label_count": len(label_ids),
                "dice_by_label": dice_by_label,
                "diff_fraction_m1": diff_fraction_roi,
                "mean_m1": mean_roi,
                "median_m1": median_roi,
            }
        )

    if not summary_rows:
        raise SystemExit("No repeats were successfully analysed.")

    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    summary_fieldnames = [
        "repeat_tag",
        "diff_fraction",
        "diff_fraction_roi",
        "diff_fraction_m1",
        "mean_roi",
        "median_roi",
        "peak_roi",
        "mean_m1",
        "median_m1",
        "mean_head",
        "median_head",
        "peak_head",
        "high_field_dice_head",
        "high_field_centroid_distance_mm",
        "hotspot_distance_head_mm",
        "hotspot_distance_roi_mm",
        "peak_head_ijk",
        "peak_head_xyz_mm",
        "peak_roi_ijk",
        "peak_roi_xyz_mm",
        "ti_scale_factor",
        "mesh_nodes",
        "label_count",
        "dice_by_label",
    ]
    _save_csv(summary_csv, summary_rows, summary_fieldnames)
    with (output_dir / "skipped_repeats.json").open("w", encoding="utf-8") as f:
        json.dump(skipped_repeats, f, indent=2)

    mean_ti_data = (ti_sum / max(1, len(summary_rows))).astype(np.float32, copy=False)
    roi_mask_img = nib.Nifti1Image(roi_mask.astype(np.uint8), ref_t1.affine, ref_t1.header)
    roi_mask_img.header.set_data_dtype(np.uint8)
    nib.save(roi_mask_img, output_dir / "roi_mask_on_t1.nii.gz")
    _plot_roi_qc(
        ref_t1,
        roi_mask,
        output_dir / "roi_outline_on_t1.png",
        title=f"{roi_name} outline on T1 | labels: {', '.join(str(x) for x in roi_labels)}",
    )
    _plot_roi_qc(
        ref_t1,
        roi_mask,
        output_dir / "roi_outline_on_mean_ti.png",
        title=f"{roi_name} outline on mean TI + T1 | labels: {', '.join(str(x) for x in roi_labels)}",
        ti_data=mean_ti_data,
    )

    repeatability_metric_order = [
        "median_roi",
        "mean_roi",
        "peak_roi",
        "median_head",
        "mean_head",
        "peak_head",
        "high_field_dice_head",
        "high_field_centroid_distance_mm",
        "hotspot_distance_head_mm",
        "hotspot_distance_roi_mm",
        "diff_fraction",
        "diff_fraction_roi",
        "mesh_nodes",
    ]
    metric_stats_map = {
        metric: _metric_stats([row[metric] for row in summary_rows])
        for metric in repeatability_metric_order
    }
    repeatability_rows = [
        {"metric": metric, **metric_stats_map[metric]}
        for metric in repeatability_metric_order
    ]
    repeatability_json = output_dir / "repeatability_stats.json"
    repeatability_csv = output_dir / "repeatability_stats.csv"
    with repeatability_json.open("w", encoding="utf-8") as f:
        json.dump(repeatability_rows, f, indent=2)
    _save_csv(
        repeatability_csv,
        repeatability_rows,
        ["metric", "n", "mean", "std", "min", "max", "cv", "cv_percent"],
    )

    parameter_consistency = _load_repeat_signatures(repeat_anat_dirs, subject)
    parameter_json = output_dir / "parameter_consistency.json"
    with parameter_json.open("w", encoding="utf-8") as f:
        json.dump(parameter_consistency, f, indent=2)

    compare_metric_to_cohort = {
        "median_roi": "median",
        "mean_roi": "mean",
        "peak_roi": "max",
    }
    cohort_comparison = None
    if args.compare_cohort_root:
        cohort_metric = args.cohort_metric or compare_metric_to_cohort[args.compare_metric]
        cohort_region_name = args.cohort_region_name or roi_name
        cohort_data = _load_cohort_metric(
            Path(args.compare_cohort_root),
            region_name=cohort_region_name,
            region_label=args.cohort_region_label,
            metric_name=cohort_metric,
        )
        within_stats = metric_stats_map[args.compare_metric]
        between_stats = cohort_data["stats"]
        within_range = float(within_stats["max"] - within_stats["min"])
        between_range = float(between_stats["max"] - between_stats["min"])
        subject_percentile = float("nan")
        subject_value = None
        if subject in cohort_data["subjects"]:
            idx = cohort_data["subjects"].index(subject)
            subject_value = float(cohort_data["values"][idx])
            subject_percentile = _percentile_rank(cohort_data["values"], subject_value)
        cohort_comparison = {
            "metric_name": cohort_metric,
            "compare_metric": args.compare_metric,
            "cohort_region_name": cohort_region_name,
            "cohort_region_label": args.cohort_region_label,
            "cohort_subjects": len(cohort_data["subjects"]),
            "within_mean": within_stats["mean"],
            "within_std": within_stats["std"],
            "within_min": within_stats["min"],
            "within_max": within_stats["max"],
            "within_range": within_range,
            "between_mean": between_stats["mean"],
            "between_std": between_stats["std"],
            "between_min": between_stats["min"],
            "between_max": between_stats["max"],
            "between_range": between_range,
            "between_to_within_sd_ratio": (
                float(between_stats["std"] / within_stats["std"])
                if np.isfinite(within_stats["std"]) and within_stats["std"] not in (0.0, -0.0)
                else float("inf")
            ),
            "between_to_within_range_ratio": (
                float(between_range / within_range)
                if np.isfinite(within_range) and within_range not in (0.0, -0.0)
                else float("inf")
            ),
            "subject_value": subject_value,
            "subject_percentile": subject_percentile,
        }
        with (output_dir / "cohort_comparison.json").open("w", encoding="utf-8") as f:
            json.dump(cohort_comparison, f, indent=2)

    _write_markdown_report(
        output_dir / "repeatability_report.md",
        subject=subject,
        roi_name=roi_name,
        roi_labels=roi_labels,
        reference_repeat=ref_anat.parent.parent.name,
        repeats=len(summary_rows),
        primary_repeatability=metric_stats_map["median_roi"],
        repeatability_rows=repeatability_rows,
        parameter_consistency=parameter_consistency,
        cohort_comparison=cohort_comparison,
    )

    diff_freq = diff_counts.astype(np.float32) / max(1, len(summary_rows))
    diff_img = nib.Nifti1Image(diff_freq, ref_t1.affine, ref_t1.header)
    diff_img.header.set_data_dtype(np.float32)
    nib.save(diff_img, output_dir / "label_diff_frequency.nii.gz")
    overlay_title = f"Label diff frequency overlay | T1 ref: {ref_t1_path}"
    _plot_diff_overlay(
        ref_t1,
        diff_freq,
        output_dir / "label_diff_overlay.png",
        title=overlay_title,
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tags = [r["repeat_tag"] for r in summary_rows]

        def _plot_metric_line(metric: str, ylabel: str, fname: str) -> None:
            vals = [r[metric] for r in summary_rows]
            plt.figure(figsize=(max(8, len(tags) * 0.22), 4))
            plt.plot(tags, vals, marker="o")
            plt.xticks(rotation=90, fontsize=7)
            plt.ylabel(ylabel)
            plt.title(f"{ylabel} across repeats")
            plt.tight_layout()
            plt.savefig(output_dir / fname, dpi=150)
            plt.close()

        _plot_metric_line("mean_roi", "Mean ROI TI (V/m)", "mean_roi_by_repeat.png")
        _plot_metric_line("median_roi", "Median ROI TI (V/m)", "median_roi_by_repeat.png")
        _plot_metric_line("peak_roi", "Peak ROI TI (V/m)", "peak_roi_by_repeat.png")
        _plot_metric_line("mean_head", "Mean head TI (V/m)", "mean_head_by_repeat.png")
        _plot_metric_line("median_head", "Median head TI (V/m)", "median_head_by_repeat.png")
        _plot_metric_line("peak_head", "Peak head TI (V/m)", "peak_head_by_repeat.png")
        _plot_metric_line("high_field_dice_head", "High-field Dice", "high_field_dice_head_by_repeat.png")
        _plot_metric_line(
            "hotspot_distance_head_mm",
            "Hotspot distance (mm)",
            "hotspot_distance_head_by_repeat.png",
        )

        mesh_vals = [r["mesh_nodes"] for r in summary_rows]
        plt.figure(figsize=(max(9, len(tags) * 0.22), 4))
        plt.plot(tags, mesh_vals, marker="o")
        plt.xticks(rotation=90, fontsize=7)
        plt.ylabel("Mesh nodes")
        plt.title("Mesh node count by repeat")
        plt.tight_layout()
        plt.savefig(output_dir / "mesh_nodes_by_repeat.png", dpi=150)
        plt.close()

        plt.figure(figsize=(max(9, len(tags) * 0.22), 4))
        plt.bar(tags, mesh_vals)
        plt.xticks(rotation=90, fontsize=7)
        plt.ylabel("Mesh nodes")
        plt.title("Mesh node count by repeat")
        plt.tight_layout()
        plt.savefig(output_dir / "mesh_nodes_by_repeat_bar.png", dpi=150)
        plt.close()

        if label_presence:
            labs_sorted = sorted(label_presence.keys())
            matrix = np.zeros((len(labs_sorted), len(tags)), dtype=int)
            for i, lab in enumerate(labs_sorted):
                present = set(label_presence.get(lab, []))
                for j, tag in enumerate(tags):
                    matrix[i, j] = 1 if tag in present else 0
            plt.figure(figsize=(max(9, len(tags) * 0.22), max(3, len(labs_sorted) * 0.35)))
            plt.imshow(matrix, aspect="auto", interpolation="nearest")
            plt.yticks(np.arange(len(labs_sorted)), [f"{lab} {_label_name(lab)}" for lab in labs_sorted], fontsize=7)
            plt.xticks(np.arange(len(tags)), tags, rotation=90, fontsize=6)
            plt.title("Label presence by repeat")
            plt.xlabel("Repeat")
            plt.ylabel("Label")
            plt.tight_layout()
            plt.savefig(output_dir / "label_presence_by_repeat.png", dpi=150)
            plt.close()

        if label_counts:
            label_ids = TISSUE_LABELS
            x = np.arange(len(label_ids))
            total_width = 0.9
            width = total_width / max(1, len(tags))
            plt.figure(figsize=(max(9, len(label_ids) * 1.3), 5))
            cmap = plt.get_cmap("tab20", len(tags))
            for idx, tag in enumerate(tags):
                counts = [label_counts.get(tag, {}).get(lab, 0) for lab in label_ids]
                offsets = x - total_width / 2 + idx * width
                plt.bar(offsets, counts, width=width, color=cmap(idx), label=tag)
            plt.xlim(-0.5, len(label_ids) - 0.5)
            xlabels = [f"{lab} {_label_name(lab)}" for lab in label_ids]
            plt.xticks(x, xlabels, rotation=45, ha="right")
            plt.title("Tissue label voxel counts by repeat (labels 1-10)")
            plt.ylabel("Voxel count")
            plt.legend(fontsize=6, ncol=3, frameon=False)
            plt.tight_layout()
            plt.savefig(output_dir / "label_counts_by_repeat.png", dpi=150)
            plt.close()
    except Exception as exc:
        log_event("plot_error", subject=subject, error=str(exc))

    result = {
        "subject": subject,
        "output_dir": str(output_dir),
        "repeats": len(summary_rows),
        "discovered_repeats": len(repeat_anat_dirs),
        "skipped_repeats": len(skipped_repeats),
        "reference_repeat": ref_anat.parent.parent.name,
        "roi_name": roi_name,
        "roi_labels": ",".join(str(x) for x in roi_labels),
        "median_roi_mean": metric_stats_map["median_roi"]["mean"],
        "median_roi_std": metric_stats_map["median_roi"]["std"],
        "median_roi_cv_percent": metric_stats_map["median_roi"]["cv_percent"],
        "peak_roi_cv_percent": metric_stats_map["peak_roi"]["cv_percent"],
        "mean_roi_mean": metric_stats_map["mean_roi"]["mean"],
        "cohort_sd_ratio": (
            cohort_comparison["between_to_within_sd_ratio"] if cohort_comparison is not None else float("nan")
        ),
    }
    log_event(
        "done",
        subject=subject,
        repeats=len(summary_rows),
        discovered_repeats=len(repeat_anat_dirs),
        skipped_repeats=len(skipped_repeats),
        output_dir=str(output_dir),
        reference_repeat=ref_anat.parent.parent.name,
        roi_name=roi_name,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze repeatability of SimNIBS mesh outputs."
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--subject", default=None)
    target_group.add_argument(
        "--all-subjects",
        action="store_true",
        help="Run the analysis for every <subject>_repeatability study found under --rootdir.",
    )
    parser.add_argument(
        "--rootdir",
        default=None,
        help=(
            "Repeatability dataset root. If omitted, the script auto-detects a known "
            "repeatability root for the requested subject."
        ),
    )
    parser.add_argument(
        "--repeats-dir",
        default=None,
        help="Overrides rootdir/repeats if set.",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Optional cap for --all-subjects, useful for testing batch mode.",
    )
    parser.add_argument(
        "--t1-root",
        default=None,
        help="Root folder containing simulation-data/<subject>_repeatability/<subject>/anat.",
    )
    parser.add_argument(
        "--atlas",
        default=None,
        help="FreeSurfer atlas NIfTI (subject space). Auto-resolved if omitted.",
    )
    parser.add_argument(
        "--atlas-dir",
        default=None,
        help="Directory containing <subject>.nii.gz atlases. Used when --atlas is omitted.",
    )
    parser.add_argument(
        "--roi-name",
        default="ctx-lh-precentral",
        help="Human-readable ROI name for titles and reports.",
    )
    parser.add_argument(
        "--roi-labels",
        default=None,
        help="Comma-separated atlas label IDs for the ROI (e.g., 1022 for ctx-lh-precentral).",
    )
    parser.add_argument(
        "--m1-labels",
        default=None,
        help="Deprecated alias for --roi-labels.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for reports and plots. Defaults to repeats/_analysis/<subject>/.",
    )
    parser.add_argument(
        "--peak-percentile",
        type=float,
        default=None,
        help=(
            "Deprecated (ignored): legacy percentile-peak reporting option."
        ),
    )
    parser.add_argument(
        "--reference-repeat",
        default=None,
        help="Repeat tag to use as reference (e.g., repeat_001). Defaults to first repeat found.",
    )
    parser.add_argument(
        "--spatial-percentile",
        type=float,
        default=99.0,
        help="Percentile used to define the high-field region for spatial stability checks.",
    )
    parser.add_argument(
        "--compare-metric",
        default="median_roi",
        choices=["median_roi", "mean_roi", "peak_roi"],
        help="Repeat metric used for within-vs-between subject comparison.",
    )
    parser.add_argument(
        "--compare-cohort-root",
        default=None,
        help="Root directory containing cohort subject folders with anat/post/region_stats_fastsurfer.csv.",
    )
    parser.add_argument(
        "--cohort-region-name",
        default=None,
        help="ROI name in region_stats_fastsurfer.csv. Defaults to --roi-name.",
    )
    parser.add_argument(
        "--cohort-region-label",
        type=int,
        default=None,
        help="Optional numeric ROI label in region_stats_fastsurfer.csv.",
    )
    parser.add_argument(
        "--cohort-metric",
        default=None,
        choices=["mean", "median", "max", "p95", "std", "cv"],
        help="Metric to compare against the cohort table. Defaults to the closest match for --compare-metric.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional JSONL log file path. Disabled by default.",
    )

    args = parser.parse_args()
    global LOG_FILE
    LOG_FILE = Path(args.log_file) if args.log_file else None
    roi_label_arg = args.roi_labels or args.m1_labels
    if args.m1_labels and not args.roi_labels:
        log_event("deprecated_arg_alias", arg="--m1-labels", replacement="--roi-labels")
    roi_name = args.roi_name.strip() or "ROI"
    if roi_label_arg:
        roi_labels = _parse_int_list(roi_label_arg)
        log_event("roi_labels", source="explicit", labels=",".join(str(x) for x in roi_labels))
    else:
        inferred_roi_labels = _infer_roi_labels(roi_name)
        if not inferred_roi_labels:
            raise SystemExit(
                f"Could not infer label IDs for ROI '{roi_name}'. Provide --roi-labels."
            )
        roi_labels = inferred_roi_labels
        log_event("roi_labels", source="preset", roi_name=roi_name, labels=",".join(str(x) for x in roi_labels))

    if args.all_subjects:
        if args.repeats_dir:
            raise SystemExit("--repeats-dir is only supported in single-subject mode.")
        if args.atlas:
            raise SystemExit(
                "--atlas is only supported in single-subject mode; use --atlas-dir or auto-resolution for batch mode."
            )

        batch_root = _resolve_batch_root(args.rootdir)
        subjects = _discover_repeatability_subjects(batch_root)
        if args.max_subjects is not None:
            subjects = subjects[: args.max_subjects]
        if not subjects:
            raise SystemExit(f"No subjects found under {batch_root}")

        batch_output_root = Path(args.output_dir) if args.output_dir else batch_root / "_analysis"
        batch_output_root.mkdir(parents=True, exist_ok=True)
        log_event("batch_start", rootdir=str(batch_root), subjects=len(subjects), output_dir=str(batch_output_root))

        batch_results: list[dict[str, object]] = []
        batch_failures: list[dict[str, str]] = []
        for batch_subject in subjects:
            log_event("subject_start", subject=batch_subject)
            try:
                batch_results.append(
                    _run_subject_analysis(
                        args,
                        subject=batch_subject,
                        roi_name=roi_name,
                        roi_labels=roi_labels,
                        rootdir_override=batch_root,
                        output_dir_override=batch_output_root / batch_subject,
                    )
                )
            except (SystemExit, Exception) as exc:
                batch_failures.append({"subject": batch_subject, "error": str(exc)})
                log_event("subject_error", subject=batch_subject, error=str(exc))

        if batch_results:
            with (batch_output_root / "batch_summary.json").open("w", encoding="utf-8") as f:
                json.dump(batch_results, f, indent=2)
            _save_csv(
                batch_output_root / "batch_summary.csv",
                batch_results,
                [
                    "subject",
                    "output_dir",
                    "repeats",
                    "discovered_repeats",
                    "skipped_repeats",
                    "reference_repeat",
                    "roi_name",
                    "roi_labels",
                    "median_roi_mean",
                    "median_roi_std",
                    "median_roi_cv_percent",
                    "peak_roi_cv_percent",
                    "mean_roi_mean",
                    "cohort_sd_ratio",
                ],
            )
        if batch_failures:
            with (batch_output_root / "batch_failures.json").open("w", encoding="utf-8") as f:
                json.dump(batch_failures, f, indent=2)

        log_event(
            "batch_done",
            output_dir=str(batch_output_root),
            succeeded=len(batch_results),
            failed=len(batch_failures),
        )
        if not batch_results:
            raise SystemExit("Batch mode did not complete any subject successfully.")
        return

    subject = args.subject.strip()
    _run_subject_analysis(
        args,
        subject=subject,
        roi_name=roi_name,
        roi_labels=roi_labels,
    )
    return

    log_event("repeat_root", path=str(repeats_root))
    repeat_anat_dirs = _find_repeat_dirs(repeats_root, subject)
    if not repeat_anat_dirs:
        raise SystemExit(f"No repeats found under {repeats_root}")

    # Pick reference repeat
    if args.reference_repeat:
        ref_anat = repeats_root / args.reference_repeat / subject / "anat"
        if ref_anat not in repeat_anat_dirs:
            raise SystemExit(f"Reference repeat not found: {ref_anat}")
    else:
        ref_anat = repeat_anat_dirs[0]

    output_dir = Path(args.output_dir) if args.output_dir else repeats_root / "_analysis" / subject
    output_dir.mkdir(parents=True, exist_ok=True)
    log_event("output_dir", path=str(output_dir))
    repeat_qc_dir = output_dir / "repeat_qc"
    repeat_qc_dir.mkdir(parents=True, exist_ok=True)

    # Load atlas and reference T1
    atlas_path = _resolve_atlas_path(subject, args.atlas, args.atlas_dir, rootdir)
    log_event("atlas_load", path=str(atlas_path))
    atlas_img = nib.load(atlas_path)
    t1_root = Path(args.t1_root) if args.t1_root else rootdir
    log_event("t1_root", path=str(t1_root))
    ref_t1_path = _resolve_t1_path(subject, ref_anat, t1_root)
    ref_t1 = nib.load(ref_t1_path)
    atlas_img = _ensure_label_grid(atlas_img, ref_t1, label="atlas_to_t1")
    atlas_data = np.asarray(atlas_img.dataobj).astype(np.int32, copy=False)

    roi_mask = np.isin(atlas_data, roi_labels)
    if not np.any(roi_mask):
        raise SystemExit(
            f"{roi_name} mask is empty after resampling; check label IDs and atlas alignment."
        )

    # Load reference label volume
    ref_label_path, _, ref_ti_path = _load_or_create_volumes(ref_anat, subject, ref_t1_path)
    ref_label_img = nib.load(ref_label_path)
    ref_label_img = _ensure_label_grid(ref_label_img, ref_t1, label="ref_labels_to_t1")
    ref_labels = np.asarray(ref_label_img.dataobj).astype(np.int32, copy=False)
    ref_ti_img = _ensure_scalar_grid(nib.load(ref_ti_path), ref_t1, label="ref_ti_to_t1")
    ref_ti_data = np.asarray(ref_ti_img.dataobj, dtype=np.float32)
    ref_head_mask = ref_labels > 0
    ref_ti_data, ref_ti_scale_factor = _normalize_ti_units(
        ref_ti_data,
        ref_head_mask,
        label="ref_ti_to_t1",
    )
    ref_peak_head = _peak_info(ref_ti_data, ref_head_mask, ref_t1.affine)
    ref_peak_roi = _peak_info(ref_ti_data, roi_mask, ref_t1.affine)
    ref_high_field_mask = _top_percentile_mask(ref_ti_data, ref_head_mask, args.spatial_percentile)
    ref_high_field_centroid = _centroid_xyz(ref_high_field_mask, ref_t1.affine)

    # Aggregate metrics
    summary_rows = []
    skipped_repeats = []
    diff_counts = np.zeros(ref_labels.shape, dtype=np.int32)
    ti_sum = np.zeros(ref_labels.shape, dtype=np.float64)
    label_set = sorted(set(np.unique(ref_labels)))
    label_presence: dict[int, list[str]] = {}
    label_counts: dict[str, dict[int, int]] = {}

    if args.peak_percentile is not None:
        log_event("deprecated_arg_ignored", arg="--peak-percentile", value=args.peak_percentile)

    for anat_dir in repeat_anat_dirs:
        repeat_tag = anat_dir.parent.parent.name  # repeat_###
        try:
            t1_path = _resolve_t1_path(subject, anat_dir, t1_root)
            label_path, _, ti_path = _load_or_create_volumes(anat_dir, subject, t1_path)
        except FileNotFoundError as exc:
            skipped_repeats.append({"repeat_tag": repeat_tag, "reason": str(exc)})
            log_event("repeat_skip", repeat_tag=repeat_tag, reason=str(exc))
            continue

        label_img = _ensure_label_grid(nib.load(label_path), ref_t1, label=f"{repeat_tag}_labels_to_t1")
        labels = np.asarray(label_img.dataobj).astype(np.int32, copy=False)

        ti_img = _ensure_scalar_grid(nib.load(ti_path), ref_t1, label=f"{repeat_tag}_ti_to_t1")
        base_data = np.asarray(ti_img.dataobj, dtype=np.float32)
        head_mask = labels > 0
        base_data, ti_scale_factor = _normalize_ti_units(
            base_data,
            head_mask,
            label=f"{repeat_tag}_ti_to_t1",
        )

        diff = labels != ref_labels
        diff_counts += diff.astype(np.int32)
        ti_sum += base_data.astype(np.float64, copy=False)
        _plot_roi_qc(
            ref_t1,
            roi_mask,
            repeat_qc_dir / f"{repeat_tag}_roi_outline_on_ti.png",
            title=(
                f"{repeat_tag} | {roi_name} outline on TI + T1 | "
                f"labels: {', '.join(str(x) for x in roi_labels)}"
            ),
            ti_data=base_data,
        )

        diff_fraction = float(diff.mean())
        diff_fraction_roi = float(diff[roi_mask].mean())

        roi_vals = _finite_values(base_data[roi_mask])
        mean_roi = float(np.nanmean(roi_vals)) if roi_vals.size else float("nan")
        median_roi = float(np.nanmedian(roi_vals)) if roi_vals.size else float("nan")
        peak_roi_info = _peak_info(base_data, roi_mask, ref_t1.affine)
        peak_roi = float(peak_roi_info["value"])

        head_vals = _finite_values(base_data[head_mask])
        mean_head = float(np.nanmean(head_vals)) if head_vals.size else float("nan")
        median_head = float(np.nanmedian(head_vals)) if head_vals.size else float("nan")
        peak_head_info = _peak_info(base_data, head_mask, ref_t1.affine)
        peak_head = float(peak_head_info["value"])
        high_field_mask = _top_percentile_mask(base_data, head_mask, args.spatial_percentile)
        high_field_dice_head = _dice(high_field_mask, ref_high_field_mask)
        high_field_centroid = _centroid_xyz(high_field_mask, ref_t1.affine)
        high_field_centroid_distance_mm = _distance_mm(high_field_centroid, ref_high_field_centroid)
        hotspot_distance_head_mm = _distance_mm(
            peak_head_info["xyz_mm"], ref_peak_head["xyz_mm"]
        )
        hotspot_distance_roi_mm = _distance_mm(
            peak_roi_info["xyz_mm"], ref_peak_roi["xyz_mm"]
        )

        # Per-label Dice vs reference
        dice_by_label = {}
        for lab in label_set:
            a = labels == lab
            b = ref_labels == lab
            dice_by_label[int(lab)] = _dice(a, b)

        label_ids = sorted(set(np.unique(labels)) - {0})
        for lab in label_ids:
            label_presence.setdefault(int(lab), []).append(repeat_tag)

        counts = np.bincount(labels.ravel(), minlength=max(TISSUE_LABELS) + 1)
        label_counts[repeat_tag] = {lab: int(counts[lab]) for lab in TISSUE_LABELS}

        ti_msh_path = anat_dir / "SimNIBS" / "Output" / subject / "TI.msh"
        if not ti_msh_path.exists():
            log_event("missing", kind="ti_msh", path=str(ti_msh_path))
            mesh_nodes = float("nan")
        else:
            mesh_nodes = _count_msh_nodes(ti_msh_path)

        summary_rows.append(
            {
                "repeat_tag": repeat_tag,
                "diff_fraction": diff_fraction,
                "diff_fraction_roi": diff_fraction_roi,
                "mean_roi": mean_roi,
                "median_roi": median_roi,
                "peak_roi": peak_roi,
                "mean_head": mean_head,
                "median_head": median_head,
                "peak_head": peak_head,
                "hotspot_distance_head_mm": hotspot_distance_head_mm,
                "hotspot_distance_roi_mm": hotspot_distance_roi_mm,
                "high_field_dice_head": high_field_dice_head,
                "high_field_centroid_distance_mm": high_field_centroid_distance_mm,
                "peak_head_ijk": peak_head_info["ijk"],
                "peak_head_xyz_mm": peak_head_info["xyz_mm"],
                "peak_roi_ijk": peak_roi_info["ijk"],
                "peak_roi_xyz_mm": peak_roi_info["xyz_mm"],
                "ti_scale_factor": ti_scale_factor,
                "mesh_nodes": mesh_nodes,
                "label_count": len(label_ids),
                "dice_by_label": dice_by_label,
                # Deprecated aliases kept for compatibility with older downstream files.
                "diff_fraction_m1": diff_fraction_roi,
                "mean_m1": mean_roi,
                "median_m1": median_roi,
            }
        )

    if not summary_rows:
        raise SystemExit("No repeats were successfully analysed.")

    # Save summary JSON and CSV
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    summary_fieldnames = [
        "repeat_tag",
        "diff_fraction",
        "diff_fraction_roi",
        "diff_fraction_m1",
        "mean_roi",
        "median_roi",
        "peak_roi",
        "mean_m1",
        "median_m1",
        "mean_head",
        "median_head",
        "peak_head",
        "high_field_dice_head",
        "high_field_centroid_distance_mm",
        "hotspot_distance_head_mm",
        "hotspot_distance_roi_mm",
        "peak_head_ijk",
        "peak_head_xyz_mm",
        "peak_roi_ijk",
        "peak_roi_xyz_mm",
        "ti_scale_factor",
        "mesh_nodes",
        "label_count",
        "dice_by_label",
    ]
    _save_csv(summary_csv, summary_rows, summary_fieldnames)
    with (output_dir / "skipped_repeats.json").open("w", encoding="utf-8") as f:
        json.dump(skipped_repeats, f, indent=2)

    mean_ti_data = (ti_sum / max(1, len(summary_rows))).astype(np.float32, copy=False)
    roi_mask_img = nib.Nifti1Image(roi_mask.astype(np.uint8), ref_t1.affine, ref_t1.header)
    roi_mask_img.header.set_data_dtype(np.uint8)
    nib.save(roi_mask_img, output_dir / "roi_mask_on_t1.nii.gz")
    _plot_roi_qc(
        ref_t1,
        roi_mask,
        output_dir / "roi_outline_on_t1.png",
        title=f"{roi_name} outline on T1 | labels: {', '.join(str(x) for x in roi_labels)}",
    )
    _plot_roi_qc(
        ref_t1,
        roi_mask,
        output_dir / "roi_outline_on_mean_ti.png",
        title=f"{roi_name} outline on mean TI + T1 | labels: {', '.join(str(x) for x in roi_labels)}",
        ti_data=mean_ti_data,
    )

    repeatability_metric_order = [
        "median_roi",
        "mean_roi",
        "peak_roi",
        "median_head",
        "mean_head",
        "peak_head",
        "high_field_dice_head",
        "high_field_centroid_distance_mm",
        "hotspot_distance_head_mm",
        "hotspot_distance_roi_mm",
        "diff_fraction",
        "diff_fraction_roi",
        "mesh_nodes",
    ]
    metric_stats_map = {
        metric: _metric_stats([row[metric] for row in summary_rows])
        for metric in repeatability_metric_order
    }
    repeatability_rows = [
        {"metric": metric, **metric_stats_map[metric]}
        for metric in repeatability_metric_order
    ]
    repeatability_json = output_dir / "repeatability_stats.json"
    repeatability_csv = output_dir / "repeatability_stats.csv"
    with repeatability_json.open("w", encoding="utf-8") as f:
        json.dump(repeatability_rows, f, indent=2)
    _save_csv(
        repeatability_csv,
        repeatability_rows,
        ["metric", "n", "mean", "std", "min", "max", "cv", "cv_percent"],
    )

    parameter_consistency = _load_repeat_signatures(repeat_anat_dirs, subject)
    parameter_json = output_dir / "parameter_consistency.json"
    with parameter_json.open("w", encoding="utf-8") as f:
        json.dump(parameter_consistency, f, indent=2)

    compare_metric_to_cohort = {
        "median_roi": "median",
        "mean_roi": "mean",
        "peak_roi": "max",
    }
    cohort_comparison = None
    if args.compare_cohort_root:
        cohort_metric = args.cohort_metric or compare_metric_to_cohort[args.compare_metric]
        cohort_region_name = args.cohort_region_name or roi_name
        cohort_data = _load_cohort_metric(
            Path(args.compare_cohort_root),
            region_name=cohort_region_name,
            region_label=args.cohort_region_label,
            metric_name=cohort_metric,
        )
        within_stats = metric_stats_map[args.compare_metric]
        between_stats = cohort_data["stats"]
        within_range = float(within_stats["max"] - within_stats["min"])
        between_range = float(between_stats["max"] - between_stats["min"])
        subject_percentile = float("nan")
        subject_value = None
        if subject in cohort_data["subjects"]:
            idx = cohort_data["subjects"].index(subject)
            subject_value = float(cohort_data["values"][idx])
            subject_percentile = _percentile_rank(cohort_data["values"], subject_value)
        cohort_comparison = {
            "metric_name": cohort_metric,
            "compare_metric": args.compare_metric,
            "cohort_region_name": cohort_region_name,
            "cohort_region_label": args.cohort_region_label,
            "cohort_subjects": len(cohort_data["subjects"]),
            "within_mean": within_stats["mean"],
            "within_std": within_stats["std"],
            "within_min": within_stats["min"],
            "within_max": within_stats["max"],
            "within_range": within_range,
            "between_mean": between_stats["mean"],
            "between_std": between_stats["std"],
            "between_min": between_stats["min"],
            "between_max": between_stats["max"],
            "between_range": between_range,
            "between_to_within_sd_ratio": (
                float(between_stats["std"] / within_stats["std"])
                if np.isfinite(within_stats["std"]) and within_stats["std"] not in (0.0, -0.0)
                else float("inf")
            ),
            "between_to_within_range_ratio": (
                float(between_range / within_range)
                if np.isfinite(within_range) and within_range not in (0.0, -0.0)
                else float("inf")
            ),
            "subject_value": subject_value,
            "subject_percentile": subject_percentile,
        }
        with (output_dir / "cohort_comparison.json").open("w", encoding="utf-8") as f:
            json.dump(cohort_comparison, f, indent=2)

    _write_markdown_report(
        output_dir / "repeatability_report.md",
        subject=subject,
        roi_name=roi_name,
        roi_labels=roi_labels,
        reference_repeat=ref_anat.parent.parent.name,
        repeats=len(summary_rows),
        primary_repeatability=metric_stats_map["median_roi"],
        repeatability_rows=repeatability_rows,
        parameter_consistency=parameter_consistency,
        cohort_comparison=cohort_comparison,
    )

    # Save difference frequency map
    diff_freq = diff_counts.astype(np.float32) / max(1, len(summary_rows))
    diff_img = nib.Nifti1Image(diff_freq, ref_t1.affine, ref_t1.header)
    diff_img.header.set_data_dtype(np.float32)
    nib.save(diff_img, output_dir / "label_diff_frequency.nii.gz")
    overlay_title = f"Label diff frequency overlay | T1 ref: {ref_t1_path}"
    _plot_diff_overlay(
        ref_t1,
        diff_freq,
        output_dir / "label_diff_overlay.png",
        title=overlay_title,
    )

    # Basic plots (matplotlib imported lazily)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tags = [r["repeat_tag"] for r in summary_rows]
        mean_roi_vals = [r["mean_roi"] for r in summary_rows]
        median_roi_vals = [r["median_roi"] for r in summary_rows]
        peak_roi_vals = [r["peak_roi"] for r in summary_rows]
        mean_head_vals = [r["mean_head"] for r in summary_rows]
        median_head_vals = [r["median_head"] for r in summary_rows]
        peak_head_vals = [r["peak_head"] for r in summary_rows]
        dice_head_vals = [r["high_field_dice_head"] for r in summary_rows]
        hotspot_head_vals = [r["hotspot_distance_head_mm"] for r in summary_rows]
        ti_source_desc = "TI: anat/SimNIBS/ti_brain_only.nii.gz (per repeat)"

        plt.figure(figsize=(8, 4))
        plt.plot(tags, mean_roi_vals, marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Mean TI in {roi_name} | {ti_source_desc}")
        plt.ylabel("Mean TI in ROI (V/m)")
        plt.tight_layout()
        plt.savefig(output_dir / "mean_roi_by_repeat.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(tags, median_roi_vals, marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Median TI in {roi_name} | {ti_source_desc}")
        plt.ylabel("Median TI in ROI (V/m)")
        plt.tight_layout()
        plt.savefig(output_dir / "median_roi_by_repeat.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(tags, peak_roi_vals, marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Peak TI in {roi_name} | {ti_source_desc}")
        plt.ylabel("Peak TI in ROI (V/m)")
        plt.tight_layout()
        plt.savefig(output_dir / "peak_roi_by_repeat.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(tags, mean_head_vals, marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Mean TI in whole head model | {ti_source_desc}")
        plt.ylabel("Mean TI in whole head (V/m)")
        plt.tight_layout()
        plt.savefig(output_dir / "mean_head_by_repeat.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(tags, median_head_vals, marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Median TI in whole head model | {ti_source_desc}")
        plt.ylabel("Median TI in whole head (V/m)")
        plt.tight_layout()
        plt.savefig(output_dir / "median_head_by_repeat.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(tags, peak_head_vals, marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Peak TI in whole head model | {ti_source_desc}")
        plt.ylabel("Peak TI in whole head (V/m)")
        plt.tight_layout()
        plt.savefig(output_dir / "peak_head_by_repeat.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(tags, dice_head_vals, marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.title(
            f"High-field overlap vs reference (top {100.0 - args.spatial_percentile:.1f}% of head TI)"
        )
        plt.ylabel("Dice overlap")
        plt.tight_layout()
        plt.savefig(output_dir / "high_field_dice_head_by_repeat.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(tags, hotspot_head_vals, marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.title("Whole-head hotspot distance vs reference")
        plt.ylabel("Distance (mm)")
        plt.tight_layout()
        plt.savefig(output_dir / "hotspot_distance_head_by_repeat.png", dpi=150)
        plt.close()

        # Mesh node counts by repeat
        node_vals = [r["mesh_nodes"] for r in summary_rows]
        if not all(np.isnan(v) for v in node_vals):
            plt.figure(figsize=(8, 4))
            plt.plot(tags, node_vals, marker="o")
            plt.xticks(rotation=45, ha="right")
            plt.title("Head mesh node count by repeat")
            plt.ylabel("Node count")
            plt.tight_layout()
            plt.savefig(output_dir / "mesh_nodes_by_repeat.png", dpi=150)
            plt.close()

            # Bar chart with outlier-robust mean (mark outliers)
            plt.figure(figsize=(8, 4))

            node_arr = np.asarray(node_vals, dtype=float)
            valid_mask = ~np.isnan(node_arr)
            valid_vals = node_arr[valid_mask]
            outlier_mask = np.zeros_like(node_arr, dtype=bool)
            lower = upper = None
            if valid_vals.size >= 3:
                q1, q3 = np.percentile(valid_vals, [25, 75])
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_mask = (node_arr < lower) | (node_arr > upper)
                outlier_mask &= valid_mask

            colors = ["tab:red" if outlier_mask[i] else "tab:blue" for i in range(len(node_arr))]
            bars = plt.bar(tags, node_vals, color=colors)
            for i, bar in enumerate(bars):
                if outlier_mask[i]:
                    bar.set_hatch("//")

            plt.xticks(rotation=45, ha="right")
            plt.title("Head mesh node count by repeat (bar)")
            plt.ylabel("Node count")
            if valid_vals.size:
                y_min = float(np.min(valid_vals))
                y_max = float(np.max(valid_vals))
                y_span = y_max - y_min
                if y_span > 0:
                    y_pad = 0.05 * y_span
                else:
                    # Keep a small visual margin when all values are identical.
                    y_pad = max(1.0, 0.01 * max(abs(y_min), 1.0))
                plt.ylim(max(0.0, y_min - y_pad), y_max + y_pad)

            if lower is not None and upper is not None:
                inliers = node_arr[(node_arr >= lower) & (node_arr <= upper) & valid_mask]
                if inliers.size:
                    mean_val = float(np.mean(inliers))
                    plt.axhline(mean_val, color="red", linestyle="--", linewidth=1.5)
                    yticks = list(plt.yticks()[0]) + [mean_val]
                    yticks = sorted(set(yticks))
                    plt.yticks(yticks, [f"{y:.0f}" for y in yticks])
                    ax = plt.gca()
                    for tick, y in zip(ax.get_yticklabels(), ax.get_yticks()):
                        if abs(y - mean_val) < 1e-6:
                            tick.set_color("red")
                    plt.legend(
                        handles=[
                            plt.Rectangle((0, 0), 1, 1, color="tab:blue", label="Inlier"),
                            plt.Rectangle((0, 0), 1, 1, color="tab:red", hatch="//", label="Outlier"),
                        ],
                        fontsize=7,
                        frameon=False,
                    )

            plt.tight_layout()
            plt.savefig(output_dir / "mesh_nodes_by_repeat_bar.png", dpi=150)
            plt.close()
        else:
            log_event("plot_skip", reason="all_nan", plot="mesh_nodes_by_repeat")

        # Label presence heatmap by repeat (labels annotated with tissue names)
        all_labels = sorted(label_presence.keys())
        if all_labels:
            presence = np.zeros((len(tags), len(all_labels)), dtype=np.int32)
            tag_to_idx = {t: i for i, t in enumerate(tags)}
            for j, lab in enumerate(all_labels):
                for tag in label_presence.get(lab, []):
                    presence[tag_to_idx[tag], j] = 1

            plt.figure(figsize=(max(6, len(all_labels) * 0.4), 6))
            plt.imshow(presence, aspect="auto", cmap="Greys", interpolation="nearest")
            plt.yticks(range(len(tags)), tags)
            xlabels = [f"{lab} ({_label_name(lab)})" for lab in all_labels]
            plt.xticks(range(len(all_labels)), xlabels, rotation=45, ha="right")
            plt.title("Tissue label presence by repeat")
            plt.xlabel("Label ID (tissue)")
            plt.ylabel("Repeat")
            plt.tight_layout()
            plt.savefig(output_dir / "label_presence_by_repeat.png", dpi=150)
            plt.close()

        # Label voxel counts by repeat (grouped bars, labels 1-10)
        if label_counts:
            label_ids = TISSUE_LABELS
            x = np.arange(len(label_ids))
            total_width = 0.9
            width = total_width / max(1, len(tags))
            plt.figure(figsize=(max(9, len(label_ids) * 1.3), 5))
            cmap = plt.get_cmap("tab20", len(tags))
            for idx, tag in enumerate(tags):
                counts = [label_counts.get(tag, {}).get(lab, 0) for lab in label_ids]
                offsets = x - total_width / 2 + idx * width
                plt.bar(offsets, counts, width=width, color=cmap(idx), label=tag)
            plt.xlim(-0.5, len(label_ids) - 0.5)
            xlabels = [f"{lab} {_label_name(lab)}" for lab in label_ids]
            plt.xticks(x, xlabels, rotation=45, ha="right")
            plt.title("Tissue label voxel counts by repeat (labels 1-10)")
            plt.ylabel("Voxel count")
            plt.legend(fontsize=6, ncol=3, frameon=False)
            plt.tight_layout()
            plt.savefig(output_dir / "label_counts_by_repeat.png", dpi=150)
            plt.close()
    except Exception as exc:
        log_event("plot_error", error=str(exc))

    log_event(
        "done",
        repeats=len(summary_rows),
        discovered_repeats=len(repeat_anat_dirs),
        skipped_repeats=len(skipped_repeats),
        output_dir=str(output_dir),
        reference_repeat=ref_anat.parent.parent.name,
        roi_name=roi_name,
    )


if __name__ == "__main__":
    main()
