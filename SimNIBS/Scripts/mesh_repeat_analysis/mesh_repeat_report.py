#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repeatability analysis for SimNIBS mesh outputs.

Compares per-repeat volumetric label maps derived from TI.msh, and links
mesh differences to TI summary metrics (mean/median) within an ROI
(e.g., FreeSurfer M1 label) and across the whole head model.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to


REPEAT_PREFIX = "repeat_"
LOG_FILE: Path | None = None
TISSUE_LABELS = list(range(1, 11))


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
    for ax, (bg, fg, title) in zip(axes, slices, strict=False):
        ax.imshow(bg.T, cmap="gray", origin="lower")
        overlay = np.ma.masked_where(fg <= 0, fg)
        ax.imshow(overlay.T, cmap="hot", alpha=0.7, origin="lower", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze repeatability of SimNIBS mesh outputs."
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument(
        "--rootdir",
        default="/media/boyan/main/PhD/CamCan-SimNIBS_Repeatability/simulation-data",
    )
    parser.add_argument(
        "--repeats-dir",
        default=None,
        help="Overrides rootdir/repeats if set.",
    )
    parser.add_argument(
        "--t1-root",
        default=None,
        help="Root folder containing simulation-data/<subject>_repeatability/<subject>/anat.",
    )
    parser.add_argument(
        "--atlas",
        required=True,
        help="FreeSurfer atlas NIfTI (subject space).",
    )
    parser.add_argument(
        "--m1-labels",
        required=True,
        help="Comma-separated label IDs for M1 (e.g., 1024).",
    )
    parser.add_argument(
        "--output-dir",
        default="./",
        help="Output directory for reports and plots.",
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
        "--log-file",
        default=None,
        help="Optional JSONL log file path. Disabled by default.",
    )

    args = parser.parse_args()
    global LOG_FILE
    LOG_FILE = Path(args.log_file) if args.log_file else None
    subject = args.subject.strip()
    rootdir = Path(args.rootdir)
    repeats_root = Path(args.repeats_dir) if args.repeats_dir else (
        rootdir / f"{subject}_repeatability" / "repeats"
    )

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

    # Load atlas and reference T1
    log_event("atlas_load", path=str(args.atlas))
    atlas_img = nib.load(args.atlas)
    t1_root = Path(args.t1_root) if args.t1_root else rootdir
    log_event("t1_root", path=str(t1_root))
    ref_t1_path = _resolve_t1_path(subject, ref_anat, t1_root)
    ref_t1 = nib.load(ref_t1_path)
    atlas_img = _ensure_label_grid(atlas_img, ref_t1, label="atlas_to_t1")
    atlas_data = np.asarray(atlas_img.dataobj).astype(np.int32, copy=False)

    m1_labels = [int(x.strip()) for x in args.m1_labels.split(",") if x.strip()]
    m1_mask = np.isin(atlas_data, m1_labels)
    if not np.any(m1_mask):
        raise SystemExit("M1 mask is empty after resampling; check label IDs and atlas alignment.")

    # Load reference label volume
    ref_label_path, _, ref_ti_path = _load_or_create_volumes(ref_anat, subject, ref_t1_path)
    ref_label_img = nib.load(ref_label_path)
    ref_label_img = _ensure_label_grid(ref_label_img, ref_t1, label="ref_labels_to_t1")
    ref_labels = np.asarray(ref_label_img.dataobj).astype(np.int32, copy=False)

    # Aggregate metrics
    summary_rows = []
    diff_counts = np.zeros(ref_labels.shape, dtype=np.int32)
    label_set = sorted(set(np.unique(ref_labels)))
    label_presence: dict[int, list[str]] = {}
    label_counts: dict[str, dict[int, int]] = {}

    if args.peak_percentile is not None:
        log_event("deprecated_arg_ignored", arg="--peak-percentile", value=args.peak_percentile)

    for anat_dir in repeat_anat_dirs:
        repeat_tag = anat_dir.parent.parent.name  # repeat_###
        t1_path = _resolve_t1_path(subject, anat_dir, t1_root)
        label_path, _, ti_path = _load_or_create_volumes(anat_dir, subject, t1_path)

        label_img = _ensure_label_grid(nib.load(label_path), ref_t1, label=f"{repeat_tag}_labels_to_t1")
        labels = np.asarray(label_img.dataobj).astype(np.int32, copy=False)

        ti_img = _ensure_scalar_grid(nib.load(ti_path), ref_t1, label=f"{repeat_tag}_ti_to_t1")
        base_data = np.asarray(ti_img.dataobj, dtype=np.float32)

        diff = labels != ref_labels
        diff_counts += diff.astype(np.int32)

        diff_fraction = float(diff.mean())
        diff_fraction_m1 = float(diff[m1_mask].mean())

        m1_vals = base_data[m1_mask]
        mean_m1 = float(np.nanmean(m1_vals)) if m1_vals.size else float("nan")
        median_m1 = float(np.nanmedian(m1_vals)) if m1_vals.size else float("nan")

        head_vals = base_data[labels > 0]
        mean_head = float(np.nanmean(head_vals)) if head_vals.size else float("nan")
        median_head = float(np.nanmedian(head_vals)) if head_vals.size else float("nan")

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
                "diff_fraction_m1": diff_fraction_m1,
                "mean_m1": mean_m1,
                "median_m1": median_m1,
                "mean_head": mean_head,
                "median_head": median_head,
                "mesh_nodes": mesh_nodes,
                "label_count": len(label_ids),
                "dice_by_label": dice_by_label,
            }
        )

    # Save summary JSON and CSV
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    with summary_csv.open("w", encoding="utf-8") as f:
        f.write(
            "repeat_tag,diff_fraction,diff_fraction_m1,mean_m1,median_m1,mean_head,median_head,mesh_nodes,label_count\n"
        )
        for row in summary_rows:
            f.write(
                f"{row['repeat_tag']},{row['diff_fraction']},"
                f"{row['diff_fraction_m1']},{row['mean_m1']},{row['median_m1']},"
                f"{row['mean_head']},{row['median_head']},{row['mesh_nodes']},{row['label_count']}\n"
            )

    # Save difference frequency map
    diff_freq = diff_counts.astype(np.float32) / max(1, len(repeat_anat_dirs))
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
        scale_factor = 1.0 / 10000.0  # 2000 -> 0.2 V/m
        mean_m1_vals = [r["mean_m1"] * scale_factor for r in summary_rows]
        median_m1_vals = [r["median_m1"] * scale_factor for r in summary_rows]
        mean_head_vals = [r["mean_head"] * scale_factor for r in summary_rows]
        median_head_vals = [r["median_head"] * scale_factor for r in summary_rows]
        ti_source_desc = "TI: anat/SimNIBS/ti_brain_only.nii.gz (per repeat)"

        plt.figure(figsize=(8, 4))
        plt.plot(tags, mean_m1_vals, marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Mean TI in ROI (M1) | {ti_source_desc}")
        plt.ylabel("Mean TI in ROI (V/m)")
        plt.tight_layout()
        plt.savefig(output_dir / "mean_m1_by_repeat.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(tags, median_m1_vals, marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Median TI in ROI (M1) | {ti_source_desc}")
        plt.ylabel("Median TI in ROI (V/m)")
        plt.tight_layout()
        plt.savefig(output_dir / "median_m1_by_repeat.png", dpi=150)
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
        repeats=len(repeat_anat_dirs),
        output_dir=str(output_dir),
        reference_repeat=ref_anat.parent.parent.name,
    )


if __name__ == "__main__":
    main()
