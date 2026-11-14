from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.image as mpimg
from matplotlib.collections import LineCollection
import nibabel as nib
import numpy as np
import pytest

import simnibs_segmentation_determinism.visualizations as visualizations
from simnibs_segmentation_determinism.provenance import (
    sha256_file,
    write_json_atomic,
    write_sha256_sidecar,
)
from simnibs_segmentation_determinism.tests.test_visualization_data import (
    _completed_analysis,
)
from simnibs_segmentation_determinism.visualizations import (
    VisualizationBuildError,
    _plot_changed_distribution,
    build_visualizations,
)


FIGURE_STEMS = (
    "01_determinism_summary",
    "02_pairwise_disagreement_heatmap",
    "03_changed_voxel_distribution",
    "04_label_dice_distributions",
    "05_tissue_volume_variability",
    "06_spatial_disagreement_frequency",
)


def _completed_source(
    config,
    *,
    label_arrays=None,
    label_affine=None,
    t1_data=None,
    t1_affine=None,
) -> Path:
    root = _completed_analysis(config.output_root)
    if label_arrays is not None:
        if len(label_arrays) != 10:
            raise ValueError("label_arrays must contain exactly ten repeats")
        affine = np.asarray(label_affine if label_affine is not None else np.eye(4))
        for repeat, labels in enumerate(label_arrays, start=1):
            path = (
                root
                / "simnibs401_charm"
                / f"repeat_{repeat:03d}"
                / "tissue_labeling_upsampled.nii.gz"
            )
            nib.save(nib.Nifti1Image(np.asarray(labels, dtype=np.uint8), affine), path)

    first_map = nib.load(
        str(root / "simnibs401_charm/repeat_001/tissue_labeling_upsampled.nii.gz")
    )
    if t1_data is None:
        t1_data = np.arange(np.prod(first_map.shape), dtype=np.float32).reshape(
            first_map.shape
        ) + 1.0
    if t1_affine is None:
        t1_affine = first_map.affine
    anat = root / "inputs" / config.subject / "anat"
    anat.mkdir(parents=True, exist_ok=True)
    t1 = anat / f"{config.subject}_T1w.nii.gz"
    t2 = anat / f"{config.subject}_T2w.nii.gz"
    nib.save(nib.Nifti1Image(np.asarray(t1_data, dtype=np.float32), t1_affine), t1)
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)), t2)
    inputs = {
        "t1": {"path": str(t1), "sha256": sha256_file(t1)},
        "t2": {"path": str(t2), "sha256": sha256_file(t2)},
    }

    repeat_records = []
    for arm in ("simnibs326_headreco", "simnibs401_charm"):
        for repeat in range(1, 11):
            repeat_root = root / arm / f"repeat_{repeat:03d}"
            repeat_root.mkdir(parents=True, exist_ok=True)
            receipt = repeat_root / "complete.json"
            payload = {
                "status": "complete",
                "arm": arm,
                "repeat_index": repeat,
                "inputs": inputs,
            }
            if arm == "simnibs401_charm":
                label_map = repeat_root / "tissue_labeling_upsampled.nii.gz"
                payload["artifact"] = {
                    "relative_path": label_map.name,
                    "size_bytes": label_map.stat().st_size,
                    "sha256": sha256_file(label_map),
                }
            write_json_atomic(receipt, payload)
            write_sha256_sidecar(receipt)
            repeat_records.append(
                {
                    "arm": arm,
                    "repeat_index": repeat,
                    "path": str(receipt),
                    "sha256": sha256_file(receipt),
                }
            )
    analysis_files = (
        "determinism_manifest.json",
        "changed_voxel_summary.tsv",
        "tissue_summary.tsv",
    )
    terminal = root / "_pipeline/complete.json"
    write_json_atomic(
        terminal,
        {
            "schema_version": 1,
            "status": "complete",
            "workflow": "simnibs_segmentation_determinism",
            "validated_repeats": 20,
            "repeat_receipts": repeat_records,
            "analysis_artifacts": {
                name: sha256_file(root / "analysis" / name) for name in analysis_files
            },
            "source_outputs_modified": False,
        },
    )
    write_sha256_sidecar(terminal)
    return root


def _digest_tree(root: Path, excluded: Path) -> dict:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and excluded not in path.parents
    }


def test_build_visualizations_creates_complete_bound_package(configured_experiment):
    root = _completed_source(configured_experiment)
    output = root / "visualizations_v1"
    before = _digest_tree(root, output)

    result = build_visualizations(configured_experiment)

    assert result["status"] == "complete"
    assert result["action"] == "created"
    assert result["figure_count"] == 6
    assert _digest_tree(root, output) == before
    for stem in FIGURE_STEMS:
        png = output / f"{stem}.png"
        pdf = output / f"{stem}.pdf"
        assert png.is_file() and pdf.is_file()
        assert mpimg.imread(png).shape[0] >= 900
        assert mpimg.imread(png).shape[1] >= 1200
    for name in (
        "summary_statistics.json",
        "figure_captions.md",
        "figure_manifest.json",
        "SHA256SUMS",
    ):
        assert (output / name).is_file()
    manifest = json.loads((output / "figure_manifest.json").read_text())
    assert manifest["source"]["analysis_manifest"]["sha256"] == sha256_file(
        root / "analysis/determinism_manifest.json"
    )
    assert manifest["source"]["terminal_receipt"]["sha256"] == sha256_file(
        root / "_pipeline/complete.json"
    )
    assert manifest["source_outputs_modified"] is False
    for line in (output / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert sha256_file(output / relative) == expected


def test_build_visualizations_reuses_verified_package(configured_experiment):
    _completed_source(configured_experiment)
    assert build_visualizations(configured_experiment)["action"] == "created"

    second = build_visualizations(configured_experiment)

    assert second["action"] == "reused"
    assert second["status"] == "complete"


def test_build_visualizations_refuses_foreign_or_partial_destination(
    configured_experiment,
):
    root = _completed_source(configured_experiment)
    destination = root / "visualizations_v1"
    destination.mkdir()
    (destination / "unowned.txt").write_text("foreign\n", encoding="utf-8")

    with pytest.raises(VisualizationBuildError, match="foreign or incomplete"):
        build_visualizations(configured_experiment)


def test_build_visualizations_rejects_source_drift_after_render(configured_experiment):
    root = _completed_source(configured_experiment)
    build_visualizations(configured_experiment)
    manifest = root / "analysis/determinism_manifest.json"
    manifest.write_text(manifest.read_text() + "\n", encoding="utf-8")

    with pytest.raises(VisualizationBuildError, match="analysis artifact mismatch"):
        build_visualizations(configured_experiment)


def test_changed_voxel_plot_only_marks_the_explained_median(configured_experiment):
    root = _completed_source(configured_experiment)
    from simnibs_segmentation_determinism.visualization_data import load_visualization_data

    figure = _plot_changed_distribution(load_visualization_data(root))

    line_collections = [
        collection
        for collection in figure.axes[0].collections
        if isinstance(collection, LineCollection)
    ]
    assert len(line_collections) == 1


def test_rendering_error_removes_temporary_package(
    configured_experiment, monkeypatch
):
    root = _completed_source(configured_experiment)

    def fail_render(data):
        del data
        raise RuntimeError("synthetic rendering failure")

    monkeypatch.setitem(visualizations.PLOTS, FIGURE_STEMS[0], fail_render)

    with pytest.raises(RuntimeError, match="synthetic rendering failure"):
        build_visualizations(configured_experiment)

    assert not (root / "visualizations_v1").exists()
    assert not tuple(root.glob(".visualizations_v1.tmp-*"))
