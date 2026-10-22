"""Render content-addressed head-model QC requests with deferred Matplotlib."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np


class QCRenderError(RuntimeError):
    """Raised when a render request or its source artifacts are invalid."""


def render_qc_request(request_path: str | Path) -> Path:
    request_path = Path(request_path).resolve(strict=True)
    try:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QCRenderError(f"Unreadable render request: {request_path}") from exc
    expected_hash = payload.pop("request_sha256", None)
    if expected_hash != _json_sha256(payload):
        raise QCRenderError("Render request hash mismatch")
    _validate_record(payload.get("t1"), "T1")
    _validate_record(payload.get("labels"), "labels")
    _validate_record(payload.get("headmodel_marker"), "head-model marker")
    t1 = nib.load(str(payload["t1"]["path"]))
    labels = nib.load(str(payload["labels"]["path"]))
    t1_data = np.asarray(t1.dataobj, dtype=float)
    label_data = np.asarray(labels.dataobj)
    if t1_data.ndim != 3 or label_data.ndim != 3 or t1_data.shape != label_data.shape:
        raise QCRenderError(
            f"T1/label grids differ: {t1_data.shape} versus {label_data.shape}"
        )
    if not np.all(np.isfinite(t1_data)):
        raise QCRenderError("T1 contains non-finite values")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    colors = [
        (0, 0, 0, 0),
        (0.90, 0.25, 0.25, 0.70),
        (0.25, 0.55, 0.95, 0.70),
        (0.20, 0.85, 0.90, 0.70),
        (0.95, 0.85, 0.25, 0.70),
        (0.95, 0.55, 0.20, 0.70),
        (0.55, 0.90, 0.35, 0.70),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0.65, 0.30, 0.85, 0.70),
    ]
    cmap = ListedColormap(colors)
    finite = t1_data[np.isfinite(t1_data)]
    vmin, vmax = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
    if vmax <= vmin:
        vmax = vmin + 1.0
    slices = payload.get("slices", {})
    views = (
        ("sagittal", 0, int(slices["sagittal"])),
        ("coronal", 1, int(slices["coronal"])),
        ("axial", 2, int(slices["axial"])),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, (name, dimension, index) in zip(axes, views, strict=True):
        base = _slice(t1_data, dimension, index)
        overlay = np.ma.masked_where(
            _slice(label_data, dimension, index) == 0,
            _slice(label_data, dimension, index),
        )
        axis.imshow(base, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        axis.imshow(overlay, cmap=cmap, origin="lower", vmin=0, vmax=11, interpolation="nearest")
        axis.set_title(f"{name.title()} (index {index})")
        axis.axis("off")
    figure.suptitle(f"{payload['subject']} corrected-label v3 head model", fontsize=13)
    output = Path(str(payload["output_png"])).resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=".tmp-", suffix=".png", dir=output.parent)
    os.close(descriptor)
    temporary = Path(raw)
    try:
        figure.savefig(temporary, dpi=140, facecolor="white")
        plt.close(figure)
        if not temporary.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise QCRenderError("Matplotlib did not create a valid PNG")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)
    return output


def _slice(data: np.ndarray, dimension: int, index: int) -> np.ndarray:
    if index < 0 or index >= data.shape[dimension]:
        raise QCRenderError(f"Requested slice {index} is outside axis {dimension}")
    if dimension == 0:
        return data[index, :, :].T
    if dimension == 1:
        return data[:, index, :].T
    return data[:, :, index].T


def _validate_record(record: object, label: str) -> None:
    if not isinstance(record, dict):
        raise QCRenderError(f"Render request lacks {label} record")
    path = Path(str(record.get("path", "")))
    if not path.is_file() or _sha256(path) != record.get("sha256"):
        raise QCRenderError(f"Render {label} hash mismatch: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args(argv)
    render_qc_request(args.request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
