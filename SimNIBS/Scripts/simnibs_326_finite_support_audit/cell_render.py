"""Render one audit cell using a plotting-capable Python environment."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import nibabel as nib

from .config import CONDITIONS, SCHEMA_VERSION
from .imaging import audit_nifti_stages
from .io_utils import sha256_file, utc_now, write_json_atomic


def _load_request(request_path: Path) -> dict[str, Any]:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if int(request.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError("Render request schema version mismatch")
    if set(request.get("conditions", {})) != set(CONDITIONS):
        raise RuntimeError("Render request must contain remesh and fixed_mesh")
    if not request.get("target") or not request.get("subject"):
        raise RuntimeError("Render request is missing cell identity")
    return request


def render_request(*, request_path: Path) -> dict[str, Any]:
    """Recompute repeat-001 image diagnostics and render both condition panels."""

    from . import render as render_module

    request_path = Path(request_path).resolve()
    request = _load_request(request_path)
    atlas_path = Path(str(request["atlas_path"]))
    atlas_image = nib.load(str(atlas_path))
    roi_cache: dict[tuple[Any, ...], tuple[Any, dict[str, Any], bool]] = {}
    figure_records: list[dict[str, Any]] = []

    for condition in CONDITIONS:
        condition_request = request["conditions"][condition]
        version_requests = condition_request.get("versions", {})
        if set(version_requests) != {"v3", "v4"}:
            raise RuntimeError(f"{condition}: render request must contain v3 and v4")
        versions = []
        for name in ("v3", "v4"):
            version = version_requests[name]
            audit = audit_nifti_stages(
                raw_field_path=Path(str(version["raw_field"])),
                masked_field_path=Path(str(version["masked_field"])),
                tissue_labels_path=Path(str(version["tissue_labels"])),
                atlas_path=atlas_path,
                roi=str(request["roi"]),
                atlas_labels=tuple(int(value) for value in request["atlas_labels"]),
                atlas_image=atlas_image,
                roi_cache=roi_cache,
            )
            audit["cause"] = str(version["cause"])
            versions.append(
                {"name": name, "t1_path": str(version["t1"]), "audit": audit}
            )
        output_path = Path(str(condition_request["output_path"])).resolve()
        render_module.render_paired_diagnostic(
            output_path=output_path,
            target_label=str(request["target_label"]),
            subject=str(request["subject"]),
            condition=condition,
            versions=versions,
        )
        figure_records.append(
            {
                "condition": condition,
                "path": str(output_path),
                "size_bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
            }
        )

    receipt = {
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "target": str(request["target"]),
        "subject": str(request["subject"]),
        "request": str(request_path),
        "request_sha256": sha256_file(request_path),
        "renderer_python": sys.executable,
        "matplotlib_version": render_module.matplotlib.__version__,
        "figures": figure_records,
        "source_outputs_modified": False,
    }
    write_json_atomic(request_path.parent / "render_complete.json", receipt)
    return receipt
