from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .provenance import sha256_file, validate_sha256_sidecar, write_json_atomic
from .settings import ExperimentConfig
from .visualization_data import (
    CHARM_ARM,
    CHARM_MAP,
    VisualizationData,
    VisualizationDataError,
    load_visualization_data,
)


WORKFLOW = "simnibs_segmentation_determinism_visualizations"
OUTPUT_NAME = "visualizations_v1"
BLUE = "#0072B2"
ORANGE = "#D55E00"
TEAL = "#009E73"
GREY = "#6B7280"
FIGURE_STEMS = (
    "01_determinism_summary",
    "02_pairwise_disagreement_heatmap",
    "03_changed_voxel_distribution",
    "04_label_dice_distributions",
    "05_tissue_volume_variability",
    "06_spatial_disagreement_frequency",
)


class VisualizationBuildError(RuntimeError):
    """Raised when a visualization package cannot be built or reused safely."""


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VisualizationBuildError(f"Invalid JSON source: {path}") from exc
    if not isinstance(payload, dict):
        raise VisualizationBuildError(f"Expected JSON object: {path}")
    return payload


def _validate_source(config: ExperimentConfig) -> Mapping[str, object]:
    root = config.output_root.resolve()
    terminal = root / "_pipeline/complete.json"
    try:
        validate_sha256_sidecar(terminal)
    except (OSError, RuntimeError) as exc:
        raise VisualizationBuildError("Experiment terminal receipt is missing or invalid") from exc
    payload = _read_json(terminal)
    if (
        payload.get("status") != "complete"
        or payload.get("workflow") != "simnibs_segmentation_determinism"
        or payload.get("validated_repeats") != 20
    ):
        raise VisualizationBuildError("Experiment terminal receipt is not complete")

    artifacts = payload.get("analysis_artifacts")
    if not isinstance(artifacts, dict) or "determinism_manifest.json" not in artifacts:
        raise VisualizationBuildError("Terminal analysis-artifact inventory is invalid")
    for name, expected in artifacts.items():
        path = root / "analysis" / str(name)
        if not path.is_file() or sha256_file(path) != expected:
            raise VisualizationBuildError(f"Terminal analysis artifact mismatch: {path}")

    receipt_records = payload.get("repeat_receipts")
    if not isinstance(receipt_records, list) or len(receipt_records) != 20:
        raise VisualizationBuildError("Terminal repeat-receipt inventory is incomplete")
    charm_receipts = 0
    for record in receipt_records:
        if not isinstance(record, dict):
            raise VisualizationBuildError("Terminal repeat-receipt record is invalid")
        receipt = Path(str(record.get("path", "")))
        if not _inside(receipt, root) or not receipt.is_file():
            raise VisualizationBuildError(f"Repeat receipt is outside staging or missing: {receipt}")
        if sha256_file(receipt) != record.get("sha256"):
            raise VisualizationBuildError(f"Repeat receipt hash mismatch: {receipt}")
        receipt_payload = _read_json(receipt)
        if receipt_payload.get("arm") == CHARM_ARM:
            charm_receipts += 1
            artifact = receipt_payload.get("artifact")
            if not isinstance(artifact, dict) or artifact.get("relative_path") != CHARM_MAP:
                raise VisualizationBuildError(f"CHARM receipt artifact is invalid: {receipt}")
            label_map = receipt.parent / CHARM_MAP
            if not label_map.is_file() or sha256_file(label_map) != artifact.get("sha256"):
                raise VisualizationBuildError(f"CHARM label-map hash mismatch: {label_map}")
    if charm_receipts != 10:
        raise VisualizationBuildError(f"Expected 10 CHARM receipts, observed {charm_receipts}")

    manifest = root / "analysis/determinism_manifest.json"
    return {
        "terminal_receipt": {"path": str(terminal), "sha256": sha256_file(terminal)},
        "analysis_manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
    }


def _style_axes(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.65)
    axis.set_axisbelow(True)


def _save_figure(figure, output: Path, stem: str) -> None:
    figure.savefig(output / f"{stem}.png", dpi=200, bbox_inches="tight", facecolor="white")
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_summary(data: VisualizationData):
    rows = ("Successful first attempt", "Identical image geometry", "Identical tissue labels")
    arm_names = ("SimNIBS 3.2.6\nheadreco", "SimNIBS 4.0.1\nCHARM")
    arm_keys = ("simnibs326_headreco", CHARM_ARM)
    fields = (
        "operational_deterministic",
        "geometric_deterministic",
        "discrete_segmentation_deterministic",
    )
    values = np.asarray(
        [
            [bool(data.analysis_manifest["arms"][arm][field]) for arm in arm_keys]
            for field in fields
        ]
    )
    figure, axis = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    cmap = ListedColormap(["#F6C1B6", "#BFE3D5"])
    axis.imshow(values.astype(int), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(2), arm_names, fontsize=12, fontweight="bold")
    axis.set_yticks(range(3), rows, fontsize=11)
    for row in range(3):
        for column in range(2):
            axis.text(
                column,
                row,
                "Yes" if values[row, column] else "No — labels varied",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="#153243",
            )
    axis.set_title(
        "Do repeated segmentations reproduce the same result?",
        fontsize=15,
        fontweight="bold",
        pad=18,
    )
    axis.text(
        0.5,
        -0.14,
        "One participant; 10 independent runs per implementation",
        transform=axis.transAxes,
        ha="center",
        fontsize=10,
        color=GREY,
    )
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    return figure


def _plot_heatmap(data: VisualizationData):
    maximum = max(float(np.max(data.pairwise_changed_percent)), 0.01)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.3), constrained_layout=True)
    matrices = (np.zeros((10, 10)), data.pairwise_changed_percent)
    titles = (
        "SimNIBS 3.2.6 headreco\nAll retained labels identical",
        "SimNIBS 4.0.1 CHARM\nChanged labels between repeat pairs",
    )
    image = None
    for axis, matrix, title in zip(axes, matrices, titles):
        image = axis.imshow(matrix, cmap="magma", vmin=0, vmax=maximum, origin="lower")
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_xlabel("Repeat")
        axis.set_ylabel("Repeat")
        axis.set_xticks(range(10), range(1, 11))
        axis.set_yticks(range(10), range(1, 11))
        axis.set_aspect("equal")
    colorbar = figure.colorbar(image, ax=axes, shrink=0.82, pad=0.03)
    colorbar.set_label("Changed tissue labels (%)", fontsize=11)
    figure.suptitle("Pairwise segmentation disagreement", fontsize=15, fontweight="bold")
    return figure


def _plot_changed_distribution(data: VisualizationData):
    values = data.changed_percent
    figure, axis = plt.subplots(figsize=(8, 5.6), constrained_layout=True)
    offsets = np.linspace(-0.13, 0.13, values.size)
    axis.scatter(1 + offsets, values, color=ORANGE, s=38, alpha=0.78, edgecolor="white")
    median = float(np.median(values))
    axis.hlines(median, 0.72, 1.28, color="#1F2937", linewidth=2.4, label=f"Median: {median:.3f}%")
    axis.set_xlim(0.55, 1.45)
    axis.set_xticks([1], ["SimNIBS 4.0.1 CHARM\n45 repeat pairs"])
    axis.set_ylabel("Changed tissue labels (%)")
    axis.set_title("Magnitude of CHARM repeat-to-repeat differences", fontsize=15, fontweight="bold")
    axis.legend(frameon=False, loc="upper right")
    _style_axes(axis)
    return figure


def _label_order(labels: Sequence[str]):
    return sorted(labels, key=lambda label: int(label) if str(label).lstrip("-").isdigit() else str(label))


def _plot_dice(data: VisualizationData):
    labels = _label_order(tuple(data.dice_by_label))
    values = [data.dice_by_label[label] for label in labels]
    figure, axis = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    box = axis.boxplot(values, patch_artist=True, showfliers=False, widths=0.58)
    for patch in box["boxes"]:
        patch.set_facecolor("#F6C1B6")
        patch.set_edgecolor(ORANGE)
    for index, observations in enumerate(values, start=1):
        offsets = np.linspace(-0.18, 0.18, observations.size)
        axis.scatter(index + offsets, observations, color=ORANGE, s=15, alpha=0.55, linewidth=0)
    axis.axhline(1.0, color=BLUE, linestyle="--", linewidth=1.2, label="Exact agreement")
    lower = max(0.0, min(float(np.min(value)) for value in values) - 0.02)
    axis.set_ylim(lower, 1.005)
    axis.set_xticks(range(1, len(labels) + 1), labels)
    axis.set_xlabel("Tissue label ID")
    axis.set_ylabel("Dice similarity")
    axis.set_title("Tissue-label similarity across CHARM repeats", fontsize=15, fontweight="bold")
    axis.legend(frameon=False, loc="lower left")
    _style_axes(axis)
    return figure


def _plot_volumes(data: VisualizationData):
    labels = _label_order(tuple(data.volume_mm3_by_label))
    deviations = []
    for label in labels:
        values = data.volume_mm3_by_label[label]
        mean = np.mean(values)
        deviations.append((values - mean) / mean * 100.0 if mean else np.zeros_like(values))
    figure, axis = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    axis.axhline(0, color="#1F2937", linewidth=1)
    for index, values in enumerate(deviations, start=1):
        axis.plot(
            np.full(values.size, index),
            values,
            color=TEAL,
            marker="o",
            linestyle="none",
            markersize=5,
            alpha=0.72,
        )
        axis.vlines(index, float(np.min(values)), float(np.max(values)), color=TEAL, alpha=0.45)
    axis.set_xticks(range(1, len(labels) + 1), labels)
    axis.set_xlabel("Tissue label ID")
    axis.set_ylabel("Volume deviation from label mean (%)")
    axis.set_title("CHARM tissue-volume variation across 10 runs", fontsize=15, fontweight="bold")
    _style_axes(axis)
    return figure


def _plot_spatial(data: VisualizationData):
    frequency = data.disagreement_frequency
    if np.any(frequency > 0):
        coordinate = np.unravel_index(int(np.argmax(frequency)), frequency.shape)
    else:
        coordinate = tuple(size // 2 for size in frequency.shape)
    x, y, z = coordinate
    slices = (
        (data.modal_labels[x, :, :], frequency[x, :, :], f"Sagittal (x={x})"),
        (data.modal_labels[:, y, :], frequency[:, y, :], f"Coronal (y={y})"),
        (data.modal_labels[:, :, z], frequency[:, :, z], f"Axial (z={z})"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.9), constrained_layout=True)
    maximum = max(float(np.max(frequency) * 100.0), 1.0)
    overlay_image = None
    for axis, (modal, disagreement, title) in zip(axes, slices):
        modal_display = np.rot90(modal)
        disagreement_display = np.rot90(disagreement * 100.0)
        axis.imshow(modal_display, cmap="tab20", interpolation="nearest", alpha=0.55)
        overlay = np.ma.masked_where(disagreement_display <= 0, disagreement_display)
        overlay_image = axis.imshow(
            overlay,
            cmap="inferno",
            interpolation="nearest",
            vmin=0,
            vmax=maximum,
            alpha=0.9,
        )
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.set_axis_off()
    colorbar = figure.colorbar(overlay_image, ax=axes, shrink=0.78, pad=0.02)
    colorbar.set_label("Runs disagreeing with modal label (%)")
    figure.suptitle(
        "Where CHARM tissue labels vary across repeated runs",
        fontsize=15,
        fontweight="bold",
    )
    return figure


PLOTS: Mapping[str, Callable[[VisualizationData], object]] = {
    "01_determinism_summary": _plot_summary,
    "02_pairwise_disagreement_heatmap": _plot_heatmap,
    "03_changed_voxel_distribution": _plot_changed_distribution,
    "04_label_dice_distributions": _plot_dice,
    "05_tissue_volume_variability": _plot_volumes,
    "06_spatial_disagreement_frequency": _plot_spatial,
}


CAPTIONS = {
    "01_determinism_summary": "Within-implementation repeatability for one participant and ten independent runs. Exact tissue-label agreement means decoded voxel arrays and headers matched; this is not a comparison of tissue taxonomies between versions.",
    "02_pairwise_disagreement_heatmap": "Percentage of voxels whose tissue label differs for every repeat pair. Headreco retained outputs were exact across repeats; CHARM values use all 45 unique pairs. The diagonal compares a run with itself.",
    "03_changed_voxel_distribution": "Each point is one of the 45 unique CHARM repeat pairs. The vertical scale is the percentage of all voxels whose decoded tissue label changed; the horizontal line marks the median.",
    "04_label_dice_distributions": "Dice similarity for each numeric CHARM tissue label across all 45 repeat pairs. Values closer to one indicate greater overlap. Numeric IDs are retained because no authoritative LUT is inferred by the exporter.",
    "05_tissue_volume_variability": "Each point is one CHARM run. For each numeric tissue label, volume is expressed as percentage deviation from that label's mean across the ten runs.",
    "06_spatial_disagreement_frequency": "Orthogonal voxel-space slices through the location of maximum disagreement. The background is the modal tissue label; the overlay is the percentage of runs that differ from the modal label at each voxel. It is a repeat-disagreement frequency, not an error probability.",
}


def _write_captions(output: Path) -> None:
    lines = ["# Segmentation determinism figure captions", ""]
    for stem in FIGURE_STEMS:
        title = stem.split("_", 1)[1].replace("_", " ").title()
        lines.extend([f"## {title}", "", CAPTIONS[stem], ""])
    (output / "figure_captions.md").write_text("\n".join(lines), encoding="utf-8")


def _artifact_record(path: Path, root: Path) -> Mapping[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_package(output: Path, data: VisualizationData, source: Mapping[str, object]) -> None:
    for stem in FIGURE_STEMS:
        _save_figure(PLOTS[stem](data), output, stem)
    write_json_atomic(output / "summary_statistics.json", dict(data.summary_statistics))
    _write_captions(output)
    artifacts = [
        _artifact_record(path, output)
        for path in sorted(output.iterdir())
        if path.is_file()
    ]
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "workflow": WORKFLOW,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "figure_count": len(FIGURE_STEMS),
        "figures": [
            {
                "stem": stem,
                "png": f"{stem}.png",
                "pdf": f"{stem}.pdf",
                "caption": CAPTIONS[stem],
            }
            for stem in FIGURE_STEMS
        ],
        "artifacts": artifacts,
        "source_outputs_modified": False,
    }
    write_json_atomic(output / "figure_manifest.json", manifest)
    checksum_paths = sorted(path for path in output.iterdir() if path.is_file())
    lines = [f"{sha256_file(path)}  {path.name}" for path in checksum_paths]
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_existing(output: Path, source: Mapping[str, object]) -> Mapping[str, object]:
    manifest_path = output / "figure_manifest.json"
    sums_path = output / "SHA256SUMS"
    if not manifest_path.is_file() or not sums_path.is_file():
        raise VisualizationBuildError(
            f"Visualization destination is foreign or incomplete: {output}"
        )
    manifest = _read_json(manifest_path)
    if (
        manifest.get("status") != "complete"
        or manifest.get("workflow") != WORKFLOW
        or manifest.get("source") != source
        or manifest.get("figure_count") != len(FIGURE_STEMS)
    ):
        raise VisualizationBuildError("Existing visualization manifest is stale or foreign")
    expected_files = set()
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as exc:
            raise VisualizationBuildError("Existing SHA256SUMS is malformed") from exc
        path = output / relative
        if not _inside(path, output) or not path.is_file() or sha256_file(path) != expected:
            raise VisualizationBuildError(f"Existing visualization checksum mismatch: {path}")
        expected_files.add(relative)
    required = {
        *(f"{stem}.{suffix}" for stem in FIGURE_STEMS for suffix in ("png", "pdf")),
        "summary_statistics.json",
        "figure_captions.md",
        "figure_manifest.json",
    }
    if not required.issubset(expected_files):
        raise VisualizationBuildError("Existing visualization package is incomplete")
    return manifest


def build_visualizations(config: ExperimentConfig) -> Dict[str, object]:
    source = _validate_source(config)
    destination = config.output_root.resolve() / OUTPUT_NAME
    if destination.exists() or destination.is_symlink():
        manifest = _validate_existing(destination, source)
        return {
            "status": "complete",
            "action": "reused",
            "output_root": str(destination),
            "figure_count": manifest["figure_count"],
            "source_outputs_modified": False,
        }
    data = load_visualization_data(config.output_root)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{OUTPUT_NAME}.tmp-", dir=str(config.output_root.resolve()))
    )
    try:
        _write_package(temporary, data, source)
        os.replace(str(temporary), str(destination))
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "status": "complete",
        "action": "created",
        "output_root": str(destination),
        "figure_count": len(FIGURE_STEMS),
        "source_outputs_modified": False,
    }
