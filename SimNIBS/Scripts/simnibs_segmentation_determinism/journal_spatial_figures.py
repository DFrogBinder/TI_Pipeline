from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import nibabel as nib
import numpy as np

from .journal_spatial_data import (
    JournalSpatialData,
    JournalSpatialDataError,
    MAX_DISPLAY_DISAGREEMENT,
    ORIENTATIONS,
    _validated_staged_t1,
    prepare_journal_spatial_data,
)
from .provenance import sha256_file, write_json_atomic
from .settings import ExperimentConfig
from .visualizations import VisualizationBuildError, _validate_source


WORKFLOW = "simnibs_segmentation_determinism_journal_spatial_figures"
OUTPUT_NAME = "journal_spatial_figures_v1"
FIGURE_STEMS = (
    "01_charm_spatial_disagreement_overview",
    "02_charm_spatial_disagreement_atlas",
)
COLORBAR_LABEL = "Runs not matching the modal tissue label (out of 10)"


class JournalSpatialFigureError(RuntimeError):
    """Raised when the journal spatial package cannot be built or reused safely."""


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
        raise JournalSpatialFigureError(f"Invalid JSON source: {path}") from exc
    if not isinstance(payload, dict):
        raise JournalSpatialFigureError(f"Expected JSON object: {path}")
    return payload


def _source_records(config: ExperimentConfig) -> Mapping[str, object]:
    try:
        source = dict(_validate_source(config))
        t1_path, t1_sha256 = _validated_staged_t1(config)
    except (VisualizationBuildError, JournalSpatialDataError) as exc:
        raise JournalSpatialFigureError(str(exc)) from exc
    source["staged_t1"] = {
        "path": str(t1_path.resolve()),
        "sha256": t1_sha256,
    }
    return source


def _disagreement_cmap():
    colors = plt.get_cmap("magma")(np.linspace(0.28, 0.93, MAX_DISPLAY_DISAGREEMENT))
    return ListedColormap(colors, name="charm_disagreement_1_to_6")


def _disagreement_norm():
    return BoundaryNorm(
        np.arange(0.5, MAX_DISPLAY_DISAGREEMENT + 1.5, 1.0),
        MAX_DISPLAY_DISAGREEMENT,
    )


def _display_plane(volume: np.ndarray, orientation: str, index: int, bounds):
    x_bounds, y_bounds, z_bounds = bounds
    x = slice(*x_bounds)
    y = slice(*y_bounds)
    z = slice(*z_bounds)
    if orientation == "sagittal":
        return np.asarray(volume[index, y, z]).T
    if orientation == "coronal":
        return np.asarray(volume[x, index, z]).T
    if orientation == "axial":
        return np.asarray(volume[x, y, index]).T
    raise JournalSpatialFigureError(f"Unknown orientation: {orientation}")


def _display_projection(volume: np.ndarray, orientation: str, bounds):
    cropped = volume[tuple(slice(*bound) for bound in bounds)]
    if orientation == "sagittal":
        return np.max(cropped, axis=0).T
    if orientation == "coronal":
        return np.max(cropped, axis=1).T
    if orientation == "axial":
        return np.max(cropped, axis=2).T
    raise JournalSpatialFigureError(f"Unknown orientation: {orientation}")


def _boundary_mask(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return boundary & (labels != 0)


def _world_coordinate(data: JournalSpatialData, orientation: str, index: int) -> float:
    axis = ORIENTATIONS.index(orientation)
    centre = np.asarray([(start + stop - 1) / 2.0 for start, stop in data.crop_bounds])
    centre[axis] = index
    return float(nib.affines.apply_affine(data.affine, centre)[axis])


def _coordinate_label(orientation: str, coordinate: float) -> str:
    axis_name = {"sagittal": "x", "coronal": "y", "axial": "z"}[orientation]
    return f"{orientation.title()} plane ({axis_name} = {coordinate:.1f} mm RAS)"


def _compact_coordinate_label(orientation: str, coordinate: float) -> str:
    axis_name = {"sagittal": "x", "coronal": "y", "axial": "z"}[orientation]
    return f"{axis_name} = {coordinate:.1f} mm"


def _annotate_directions(axis, orientation: str) -> None:
    horizontal = {
        "sagittal": ("P", "A"),
        "coronal": ("L", "R"),
        "axial": ("L", "R"),
    }[orientation]
    vertical = {
        "sagittal": ("I", "S"),
        "coronal": ("I", "S"),
        "axial": ("P", "A"),
    }[orientation]
    style = dict(
        transform=axis.transAxes,
        color="#111827",
        fontsize=8,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.2},
    )
    axis.text(0.01, 0.5, horizontal[0], ha="left", va="center", **style)
    axis.text(0.99, 0.5, horizontal[1], ha="right", va="center", **style)
    axis.text(0.5, 0.01, vertical[0], ha="center", va="bottom", **style)
    axis.text(0.5, 0.99, vertical[1], ha="center", va="top", **style)


def _show_overlay(axis, background, labels, counts, orientation: str, title: str):
    axis.imshow(background, cmap="gray", origin="lower", interpolation="bilinear", vmin=0, vmax=1)
    boundary = _boundary_mask(labels)
    if np.any(boundary):
        axis.contour(
            boundary.astype(np.uint8),
            levels=[0.5],
            colors=["#F9FAFB"],
            linewidths=0.35,
            alpha=0.58,
            origin="lower",
        )
    overlay = np.ma.masked_where(counts == 0, counts)
    image = axis.imshow(
        overlay,
        cmap=_disagreement_cmap(),
        norm=_disagreement_norm(),
        origin="lower",
        interpolation="nearest",
        alpha=0.9,
    )
    axis.set_title(title, fontsize=9.5, fontweight="semibold", pad=6)
    axis.set_xticks([])
    axis.set_yticks([])
    _annotate_directions(axis, orientation)
    return image


def _show_projection(axis, background, counts, orientation: str):
    axis.imshow(background, cmap="gray", origin="lower", interpolation="bilinear", vmin=0, vmax=1)
    image = axis.imshow(
        np.ma.masked_where(counts == 0, counts),
        cmap=_disagreement_cmap(),
        norm=_disagreement_norm(),
        origin="lower",
        interpolation="nearest",
        alpha=0.9,
    )
    axis.set_title(f"{orientation.title()} maximum projection", fontsize=10, fontweight="semibold")
    axis.set_xticks([])
    axis.set_yticks([])
    _annotate_directions(axis, orientation)
    return image


def _shared_colorbar(figure, axes, image):
    colorbar = figure.colorbar(
        image,
        ax=np.asarray(axes).ravel().tolist(),
        ticks=range(1, MAX_DISPLAY_DISAGREEMENT + 1),
        fraction=0.025,
        pad=0.018,
    )
    colorbar.set_label(COLORBAR_LABEL, fontsize=10.5, labelpad=10)
    colorbar.ax.tick_params(labelsize=9)
    return colorbar


def _plot_spatial_overview(data: JournalSpatialData):
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 7.2), constrained_layout=True)
    overlay = None
    for column, orientation in enumerate(ORIENTATIONS):
        overlay = _show_projection(
            axes[0, column],
            _display_projection(data.t1_normalized, orientation, data.crop_bounds),
            _display_projection(data.disagreement_counts, orientation, data.crop_bounds),
            orientation,
        )
        index = data.main_slice_indices[orientation]
        overlay = _show_overlay(
            axes[1, column],
            _display_plane(data.t1_normalized, orientation, index, data.crop_bounds),
            _display_plane(data.modal_labels, orientation, index, data.crop_bounds),
            _display_plane(data.disagreement_counts, orientation, index, data.crop_bounds),
            orientation,
            _coordinate_label(orientation, _world_coordinate(data, orientation, index)),
        )
    _shared_colorbar(figure, axes, overlay)
    figure.suptitle(
        "Repeat-to-repeat CHARM tissue-label disagreement",
        fontsize=15,
        fontweight="bold",
    )
    return figure


def _plot_slice_atlas(data: JournalSpatialData):
    figure, axes = plt.subplots(3, 4, figsize=(10.5, 8.2), constrained_layout=True)
    overlay = None
    for row, orientation in enumerate(ORIENTATIONS):
        indices = data.atlas_slice_indices[orientation]
        if len(indices) != 4:
            raise JournalSpatialFigureError(
                f"Expected four {orientation} atlas slices, observed {len(indices)}"
            )
        for column, index in enumerate(indices):
            overlay = _show_overlay(
                axes[row, column],
                _display_plane(data.t1_normalized, orientation, index, data.crop_bounds),
                _display_plane(data.modal_labels, orientation, index, data.crop_bounds),
                _display_plane(data.disagreement_counts, orientation, index, data.crop_bounds),
                orientation,
                _compact_coordinate_label(
                    orientation, _world_coordinate(data, orientation, index)
                ),
            )
    _shared_colorbar(figure, axes, overlay)
    figure.suptitle(
        "Repeat-to-repeat CHARM tissue-label disagreement: whole-head atlas",
        fontsize=15,
        fontweight="bold",
    )
    return figure


FIGURE_BUILDERS: Mapping[str, Callable[[JournalSpatialData], object]] = {
    FIGURE_STEMS[0]: _plot_spatial_overview,
    FIGURE_STEMS[1]: _plot_slice_atlas,
}


CAPTIONS = {
    FIGURE_STEMS[0]: (
        "Spatial distribution of repeat-to-repeat tissue-label disagreement across ten "
        "SimNIBS 4.0.1 CHARM segmentations of one participant. The top row shows "
        "maximum-intensity projections of the disagreement count; the bottom row shows "
        "the sagittal, coronal, and axial planes with the greatest summed disagreement "
        "on the staged T1-weighted anatomy. Thin lines mark modal-label boundaries. "
        "The colour scale gives the number of runs not matching the modal tissue label "
        "at each voxel (out of 10); it is repeat disagreement, not segmentation error probability."
    ),
    FIGURE_STEMS[1]: (
        "Whole-head atlas of repeat-to-repeat tissue-label disagreement across ten "
        "SimNIBS 4.0.1 CHARM segmentations of one participant. Four disagreement-weighted "
        "slices are shown for each anatomical orientation on the staged T1-weighted anatomy. "
        "Thin lines mark modal-label boundaries, and coordinates are reported in RAS millimetres. "
        "The disagreement count is not segmentation error probability, and these one-participant "
        "results should not be generalized beyond this software and HPC configuration."
    ),
}


def _save_figure(figure, output: Path, stem: str) -> None:
    figure.savefig(
        output / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    figure.savefig(output / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _write_captions(output: Path) -> None:
    lines = ["# CHARM spatial-disagreement figure captions", ""]
    for stem in FIGURE_STEMS:
        lines.extend([f"## {stem}", "", CAPTIONS[stem], ""])
    (output / "figure_captions.md").write_text("\n".join(lines), encoding="utf-8")


def _artifact_record(path: Path, root: Path) -> Mapping[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_package(
    output: Path,
    data: JournalSpatialData,
    source: Mapping[str, object],
) -> None:
    for stem in FIGURE_STEMS:
        _save_figure(FIGURE_BUILDERS[stem](data), output, stem)
    write_json_atomic(output / "spatial_summary.json", dict(data.spatial_summary))
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
        "subject": data.subject,
        "repeat_count": data.repeat_count,
        "source": source,
        "figure_count": len(FIGURE_STEMS),
        "figures": [
            {
                "stem": stem,
                "png": f"{stem}.png",
                "pdf": f"{stem}.pdf",
                "svg": f"{stem}.svg",
                "caption": CAPTIONS[stem],
            }
            for stem in FIGURE_STEMS
        ],
        "artifacts": artifacts,
        "source_outputs_modified": False,
    }
    write_json_atomic(output / "figure_manifest.json", manifest)
    checksum_paths = sorted(path for path in output.iterdir() if path.is_file())
    (output / "SHA256SUMS").write_text(
        "\n".join(f"{sha256_file(path)}  {path.name}" for path in checksum_paths)
        + "\n",
        encoding="utf-8",
    )


def _validate_existing(
    output: Path,
    source: Mapping[str, object],
) -> Mapping[str, object]:
    manifest_path = output / "figure_manifest.json"
    sums_path = output / "SHA256SUMS"
    if not manifest_path.is_file() or not sums_path.is_file():
        raise JournalSpatialFigureError(
            f"Journal figure destination is foreign or incomplete: {output}"
        )
    manifest = _read_json(manifest_path)
    if (
        manifest.get("status") != "complete"
        or manifest.get("workflow") != WORKFLOW
        or manifest.get("source") != source
        or manifest.get("figure_count") != len(FIGURE_STEMS)
    ):
        raise JournalSpatialFigureError("Existing journal figure manifest is stale or foreign")
    checked = set()
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as exc:
            raise JournalSpatialFigureError("Existing SHA256SUMS is malformed") from exc
        path = output / relative
        if not _inside(path, output) or not path.is_file() or sha256_file(path) != expected:
            raise JournalSpatialFigureError(
                f"Existing journal figure checksum mismatch: {path}"
            )
        checked.add(relative)
    required = {
        *(f"{stem}.{suffix}" for stem in FIGURE_STEMS for suffix in ("png", "pdf", "svg")),
        "spatial_summary.json",
        "figure_captions.md",
        "figure_manifest.json",
    }
    if not required.issubset(checked):
        raise JournalSpatialFigureError("Existing journal figure package is incomplete")
    return manifest


def build_journal_spatial_figures(config: ExperimentConfig) -> Dict[str, object]:
    source = _source_records(config)
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
    try:
        data = prepare_journal_spatial_data(config)
    except JournalSpatialDataError as exc:
        raise JournalSpatialFigureError(str(exc)) from exc
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
