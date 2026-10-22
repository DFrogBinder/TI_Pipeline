"""Head-model build, validation, collection, and human-review promotion."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import nibabel as nib
import numpy as np

from .label_io import harmonize_subject_labels, temporary_surface_masks
from .reference_stage import validate_staged_reference
from .settings import (
    CampaignPaths,
    SUBJECTS,
    assert_root_isolation,
    subject_paths,
)
from .v3_meshing import (
    MeshValidation,
    SurfaceResult,
    V3MeshingAPI,
    build_surfaces,
    build_volume_mesh,
    load_v3_meshing_api,
    probe_v3_meshing_api,
    validate_mesh,
)


class HeadModelError(RuntimeError):
    """Raised when a head model cannot be proved complete and independent."""


def build_headmodel(
    paths: CampaignPaths,
    subject: str,
    api: V3MeshingAPI,
) -> dict[str, object]:
    """Build one corrected-label v3 head model; never mark a cohort complete."""
    assert_root_isolation(paths)
    resolved = subject_paths(paths, subject)
    if resolved.headmodel_receipt.exists():
        try:
            payload = validate_headmodel(paths, subject, api)
        except Exception as exc:
            raise HeadModelError(
                f"Refusing to replace invalid existing completion marker for {subject}"
            ) from exc
        return {"action": "reused", **payload}

    _clear_partial_generated_artifacts(resolved)
    staged = validate_staged_reference(resolved)
    label_receipt = harmonize_subject_labels(resolved)
    legacy_hashes = _legacy_artifact_hashes(resolved)
    scratch_root = paths.headmodel_root / "_pipeline" / "scratch" / subject
    scratch_root.mkdir(parents=True, exist_ok=True)

    with temporary_surface_masks(resolved.harmonized_label, scratch_root) as masks:
        api_probe = probe_v3_meshing_api(api)
        surface_result = build_surfaces(
            api,
            masks,
            resolved.staged_m2m,
            scratch_root,
            vertex_density=0.5,
        )
        mesh_validation = build_volume_mesh(
            api,
            subject,
            surface_result.surfaces,
            surface_result.relabel_surfaces,
            resolved.staged_m2m,
            resolved.head_mesh,
            keep_air=False,
        )
        mesh_validation = validate_mesh(api, resolved.head_mesh)
        node_count = _validate_finite_mesh_nodes(api, resolved.head_mesh)
        comparison = _reject_legacy_reuse(
            resolved.head_mesh,
            surface_result,
            legacy_hashes,
        )
        payload = _headmodel_payload(
            resolved=resolved,
            staged=staged,
            label_receipt=label_receipt,
            api_probe=api_probe,
            surfaces=surface_result,
            mesh_validation=mesh_validation,
            node_count=node_count,
            legacy_comparison=comparison,
        )
        _atomic_write_json(resolved.headmodel_receipt, payload)
        _after_marker_write(resolved.headmodel_receipt, payload)
    return {"action": "built", **payload}


def validate_headmodel(
    paths: CampaignPaths,
    subject: str,
    api: V3MeshingAPI,
) -> dict[str, object]:
    """Re-hash every artifact named by one head-model completion marker."""
    resolved = subject_paths(paths, subject)
    marker = resolved.headmodel_receipt
    if not marker.is_file():
        raise HeadModelError(f"Head-model completion marker is missing: {marker}")
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HeadModelError(f"Unreadable head-model marker: {marker}") from exc
    if payload.get("status") != "complete" or payload.get("subject") != subject:
        raise HeadModelError(f"Head-model marker is not complete for {subject}")
    for key in ("source", "harmonized_map", "harmonization_receipt"):
        record = payload.get(key)
        _validate_hash_record(record, key)
    staged = payload.get("staged_reference")
    if not isinstance(staged, dict):
        raise HeadModelError("Head-model marker lacks staged-reference provenance")
    receipt_path = Path(str(staged.get("receipt_path", "")))
    if receipt_path.resolve(strict=False) != resolved.reference_receipt.resolve(strict=False):
        raise HeadModelError("Head-model marker names a different staging receipt")
    if not receipt_path.is_file() or _sha256(receipt_path) != staged.get("receipt_sha256"):
        raise HeadModelError("Staged-reference receipt hash mismatch")
    surface_records = payload.get("surfaces")
    if not isinstance(surface_records, list) or len(surface_records) < 5:
        raise HeadModelError("Head-model marker has an incomplete surface inventory")
    surface_paths = []
    for record in surface_records:
        _validate_hash_record(record, "surface")
        surface_paths.append(Path(str(record["path"])))
    _require_primary_surfaces(surface_paths)
    mesh_record = payload.get("mesh")
    _validate_hash_record(mesh_record, "mesh")
    if Path(str(mesh_record["path"])).resolve() != resolved.head_mesh.resolve():
        raise HeadModelError("Head-model marker names a different canonical mesh")
    validation = validate_mesh(api, resolved.head_mesh)
    node_count = _validate_finite_mesh_nodes(api, resolved.head_mesh)
    expected_validation = payload.get("mesh_validation")
    if _json_sha256(expected_validation) != _json_sha256(asdict(validation)):
        raise HeadModelError("Current mesh validation differs from its marker")
    if int(payload.get("finite_node_count", -1)) != node_count:
        raise HeadModelError("Current mesh node count differs from its marker")
    api_probe = probe_v3_meshing_api(api)
    if _json_sha256(api_probe) != payload.get("api_probe_sha256"):
        raise HeadModelError("Current SimNIBS API probe differs from the build marker")
    return payload


def write_qc_request(paths: CampaignPaths, subject: str) -> Path:
    """Write a content-addressed request for the separate render interpreter."""
    resolved = subject_paths(paths, subject)
    if not resolved.headmodel_receipt.is_file():
        raise HeadModelError(f"Head-model marker is missing for QC: {subject}")
    marker = json.loads(resolved.headmodel_receipt.read_text(encoding="utf-8"))
    _validate_hash_record(marker.get("harmonized_map"), "harmonized map")
    t1_path = resolved.staged_m2m / "T1fs_conform.nii.gz"
    if not t1_path.is_file():
        raise HeadModelError(f"Staged T1fs reference is missing: {t1_path}")
    label_image = nib.load(resolved.harmonized_label)
    label_data = np.asanyarray(label_image.dataobj)
    support = np.argwhere(np.isin(label_data, [1, 2, 3, 4, 5, 6, 11]))
    if support.size == 0:
        raise HeadModelError(f"Harmonized label map is empty: {resolved.harmonized_label}")
    center = np.rint((support.min(axis=0) + support.max(axis=0)) / 2.0).astype(int)
    request_payload: dict[str, object] = {
        "schema_version": 1,
        "subject": subject,
        "t1": _file_record(t1_path),
        "labels": _file_record(resolved.harmonized_label),
        "headmodel_marker": _file_record(resolved.headmodel_receipt),
        "output_png": str(resolved.harmonization_figure.resolve(strict=False)),
        "views": ["sagittal", "coronal", "axial"],
        "slices": {
            "sagittal": int(center[0]),
            "coronal": int(center[1]),
            "axial": int(center[2]),
        },
        "label_names": {
            "1": "white matter",
            "2": "grey matter",
            "3": "CSF",
            "4": "bone",
            "5": "external soft tissue",
            "6": "eyes",
            "11": "internal air",
        },
    }
    request_payload["request_sha256"] = _json_sha256(request_payload)
    request = (
        paths.headmodel_root / "_pipeline" / "headmodels" / "qc_requests" / f"{subject}.json"
    )
    _atomic_write_json(request, request_payload)
    return request


def collect_headmodels(paths: CampaignPaths, render_python: str) -> dict[str, object]:
    """Validate and render all ten subjects, then write the cohort manifest."""
    assert_root_isolation(paths)
    api = load_v3_meshing_api()
    entries: list[dict[str, object]] = []
    artifact_paths: list[Path] = []
    tsv_rows = [
        "subject\tharmonized_map\tharmonized_sha256\tmesh\tmesh_sha256\t"
        "tetrahedra\ttissue_tags\tqc_panel\tqc_panel_sha256"
    ]
    for subject in SUBJECTS:
        resolved = subject_paths(paths, subject)
        try:
            marker = validate_headmodel(paths, subject, api)
            request = write_qc_request(paths, subject)
            _run_renderer(render_python, request)
            _validate_png(resolved.harmonization_figure)
        except Exception as exc:
            raise HeadModelError(
                f"Head-model collection failed for {subject}: {exc}"
            ) from exc
        artifacts = [
            _file_record(resolved.headmodel_receipt),
            _file_record(resolved.harmonized_label),
            _file_record(resolved.harmonization_qc),
            *[_file_record(Path(record["path"])) for record in marker["surfaces"]],
            _file_record(resolved.head_mesh),
            _file_record(request),
            _file_record(resolved.harmonization_figure),
        ]
        artifact_paths.extend(Path(str(record["path"])) for record in artifacts)
        validation = marker["mesh_validation"]
        entry = {
            "subject": subject,
            "marker_sha256": _sha256(resolved.headmodel_receipt),
            "harmonized_map_sha256": _sha256(resolved.harmonized_label),
            "mesh_sha256": _sha256(resolved.head_mesh),
            "panel_sha256": _sha256(resolved.harmonization_figure),
            "artifacts": artifacts,
        }
        entries.append(entry)
        tsv_rows.append(
            "\t".join(
                [
                    subject,
                    str(resolved.harmonized_label.resolve()),
                    str(entry["harmonized_map_sha256"]),
                    str(resolved.head_mesh.resolve()),
                    str(entry["mesh_sha256"]),
                    str(validation["tetrahedra"]),
                    ",".join(str(tag) for tag in validation["tissue_tags"]),
                    str(resolved.harmonization_figure.resolve()),
                    str(entry["panel_sha256"]),
                ]
            )
        )
    output_dir = paths.headmodel_root / "_pipeline" / "headmodels"
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = output_dir / "headmodels.tsv"
    _atomic_write_text(tsv_path, "\n".join(tsv_rows) + "\n")
    artifact_paths.append(tsv_path)
    checksum_path = output_dir / "checksums.sha256"
    checksum_lines = []
    unique_artifacts = sorted(
        {path.resolve() for path in artifact_paths}, key=lambda path: str(path)
    )
    for path in unique_artifacts:
        try:
            display = path.relative_to(paths.headmodel_root.resolve()).as_posix()
        except ValueError:
            display = str(path)
        checksum_lines.append(f"{_sha256(path)}  {display}")
    _atomic_write_text(checksum_path, "\n".join(checksum_lines) + "\n")
    manifest_path = output_dir / "manifest.json"
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "created_utc": _utc_now(),
        "git_commit": _git_commit(),
        "headmodel_root": str(paths.headmodel_root.resolve()),
        "subject_count": len(entries),
        "subject_order": list(SUBJECTS),
        "subjects": entries,
        "cohort_files": {
            "tsv": _file_record(tsv_path),
            "checksums": _file_record(checksum_path),
        },
        "source_label_maps_copied": False,
        "panels_required_for_promotion": len(entries),
    }
    _atomic_write_json(manifest_path, payload)
    return {**payload, "manifest_sha256": _sha256(manifest_path)}


def validate_headmodel_manifest(paths: CampaignPaths) -> dict[str, object]:
    manifest_path = paths.headmodel_root / "_pipeline" / "headmodels" / "manifest.json"
    if not manifest_path.is_file():
        raise HeadModelError(f"Head-model manifest is missing: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HeadModelError(f"Unreadable head-model manifest: {manifest_path}") from exc
    if payload.get("status") != "complete":
        raise HeadModelError("Head-model manifest is not complete")
    if payload.get("headmodel_root") != str(paths.headmodel_root.resolve()):
        raise HeadModelError("Head-model manifest is bound to a different root")
    if payload.get("subject_count") != len(SUBJECTS):
        raise HeadModelError("Head-model manifest does not contain ten subjects")
    if payload.get("subject_order") != list(SUBJECTS):
        raise HeadModelError("Head-model manifest subject order differs from the pin")
    entries = payload.get("subjects")
    if not isinstance(entries, list) or [entry.get("subject") for entry in entries] != list(SUBJECTS):
        raise HeadModelError("Head-model manifest subject entries are incomplete")
    for entry in entries:
        artifacts = entry.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise HeadModelError(f"Manifest artifacts missing for {entry.get('subject')}")
        for record in artifacts:
            _validate_hash_record(record, "manifest artifact")
        panel_records = [
            record for record in artifacts if str(record.get("path", "")).endswith(".png")
        ]
        if len(panel_records) != 1:
            raise HeadModelError(f"Manifest panel inventory is invalid for {entry.get('subject')}")
        _validate_png(Path(str(panel_records[0]["path"])))
    cohort_files = payload.get("cohort_files")
    if not isinstance(cohort_files, dict):
        raise HeadModelError("Manifest cohort-file records are missing")
    _validate_hash_record(cohort_files.get("tsv"), "cohort TSV")
    _validate_hash_record(cohort_files.get("checksums"), "checksum manifest")
    return {**payload, "manifest_sha256": _sha256(manifest_path)}


def promote_headmodels(
    paths: CampaignPaths,
    manifest_sha256: str,
    confirm_reviewed: bool,
) -> dict[str, object]:
    if confirm_reviewed is not True:
        raise HeadModelError(
            "Head-model promotion requires explicit confirm_reviewed=True"
        )
    manifest = validate_headmodel_manifest(paths)
    observed = str(manifest["manifest_sha256"])
    if manifest_sha256 != observed:
        raise HeadModelError(
            f"Requested manifest hash does not match current manifest hash: {observed}"
        )
    path = paths.headmodel_root / "_pipeline" / "headmodels" / "promoted.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HeadModelError(f"Existing promotion receipt is unreadable: {path}") from exc
        if existing.get("manifest_sha256") != observed:
            raise HeadModelError("Existing promotion receipt names a stale manifest")
        return {"action": "reused", **existing}
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "promoted",
        "promoted_utc": _utc_now(),
        "git_commit": _git_commit(),
        "manifest_path": str(
            (paths.headmodel_root / "_pipeline" / "headmodels" / "manifest.json").resolve()
        ),
        "manifest_sha256": observed,
        "confirm_reviewed": True,
        "panels_reviewed": len(SUBJECTS),
        "subjects": list(SUBJECTS),
    }
    _atomic_write_json(path, payload)
    return {"action": "created", **payload}


def _headmodel_payload(
    *,
    resolved,
    staged,
    label_receipt,
    api_probe: dict[str, object],
    surfaces: SurfaceResult,
    mesh_validation: MeshValidation,
    node_count: int,
    legacy_comparison: dict[str, object],
) -> dict[str, object]:
    surface_paths = tuple(surfaces.surfaces) + tuple(surfaces.relabel_surfaces)
    _require_primary_surfaces(list(surface_paths))
    return {
        "schema_version": 1,
        "status": "complete",
        "created_utc": _utc_now(),
        "git_commit": _git_commit(),
        "subject": resolved.subject,
        "source": _file_record(Path(label_receipt.source_path)),
        "harmonized_map": _file_record(Path(label_receipt.output_path)),
        "harmonization_receipt": _file_record(Path(label_receipt.receipt_path)),
        "staged_reference": {
            "receipt_path": str(Path(staged.receipt_path).resolve()),
            "receipt_sha256": _sha256(Path(staged.receipt_path)),
            "destination_root": staged.destination_root,
            "file_count": staged.file_count,
            "byte_count": staged.byte_count,
        },
        "surfaces": [_file_record(path) for path in surface_paths],
        "mesh": _file_record(resolved.head_mesh),
        "mesh_validation": asdict(mesh_validation),
        "finite_node_count": node_count,
        "api_probe": api_probe,
        "api_probe_sha256": _json_sha256(api_probe),
        "legacy_hash_comparison": legacy_comparison,
        "temporary_masks_retained": False,
    }


def _validate_finite_mesh_nodes(api: V3MeshingAPI, mesh_path: Path) -> int:
    mesh = api.mesh_io.read_msh(str(mesh_path), buffered=True)
    try:
        nodes = np.asarray(mesh.nodes.node_coord, dtype=float)
    except AttributeError as exc:
        raise HeadModelError("Mesh lacks node coordinates") from exc
    if nodes.ndim != 2 or nodes.shape[1] != 3 or nodes.shape[0] == 0:
        raise HeadModelError(f"Mesh node array has invalid shape: {nodes.shape}")
    if not np.all(np.isfinite(nodes)):
        raise HeadModelError("Mesh contains non-finite node coordinates")
    return int(nodes.shape[0])


def _clear_partial_generated_artifacts(resolved) -> None:
    if resolved.staged_m2m.is_dir():
        for path in resolved.staged_m2m.iterdir():
            if path.is_file() and path.suffix.lower() in {".stl", ".off", ".msh"}:
                path.unlink()
    for path in (resolved.head_mesh, resolved.compatibility_mesh):
        if path.is_file() or path.is_symlink():
            path.unlink()


def _legacy_artifact_hashes(resolved) -> dict[str, dict[str, str]]:
    surfaces = {}
    if resolved.legacy_m2m.is_dir():
        for path in sorted(resolved.legacy_m2m.iterdir()):
            if path.is_file() and path.suffix.lower() in {".stl", ".off"}:
                surfaces[str(path.resolve())] = _sha256(path)
    meshes = {}
    for path in (
        resolved.legacy_anat / f"{resolved.subject}.msh",
        resolved.legacy_m2m / f"{resolved.subject}.msh",
    ):
        if path.is_file():
            meshes[str(path.resolve())] = _sha256(path)
    return {"surfaces": surfaces, "meshes": meshes}


def _reject_legacy_reuse(
    mesh_path: Path,
    surfaces: SurfaceResult,
    legacy_hashes: dict[str, dict[str, str]],
) -> dict[str, object]:
    mesh_hash = _sha256(mesh_path)
    mesh_matches = [
        path for path, checksum in legacy_hashes["meshes"].items() if checksum == mesh_hash
    ]
    if mesh_matches:
        raise HeadModelError(
            f"New canonical mesh is byte-identical to a legacy mesh: {mesh_matches}"
        )
    new_surface_hashes = {
        _sha256(path) for path in (*surfaces.surfaces, *surfaces.relabel_surfaces)
    }
    reused_surfaces = sorted(
        path
        for path, checksum in legacy_hashes["surfaces"].items()
        if checksum in new_surface_hashes
    )
    if reused_surfaces:
        raise HeadModelError(
            f"New surfaces are byte-identical to legacy surfaces: {reused_surfaces}"
        )
    return {
        "legacy_mesh_count": len(legacy_hashes["meshes"]),
        "legacy_surface_count": len(legacy_hashes["surfaces"]),
        "mesh_reused": False,
        "surface_hashes_reused": [],
    }


def _require_primary_surfaces(paths: list[Path]) -> None:
    names = [path.name.lower() for path in paths]
    missing = [
        tissue
        for tissue in ("wm", "gm", "csf", "bone", "skin")
        if not any(tissue in name for name in names)
    ]
    if missing:
        raise HeadModelError(f"Required primary surfaces are missing: {missing}")


def _validate_hash_record(record: object, label: str) -> None:
    if not isinstance(record, dict):
        raise HeadModelError(f"Missing {label} hash record")
    path = Path(str(record.get("path", "")))
    if not path.is_file() or _sha256(path) != record.get("sha256"):
        raise HeadModelError(f"{label.title()} hash mismatch: {path}")


def _file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {
        "name": resolved.name,
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=path.parent)
    os.close(descriptor)
    temporary = Path(raw)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=".tmp-", suffix=".txt", dir=path.parent)
    os.close(descriptor)
    temporary = Path(raw)
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_renderer(render_python: str, request: Path) -> None:
    executable = Path(render_python).expanduser().resolve(strict=True)
    completed = subprocess.run(
        [
            str(executable),
            "-m",
            "simnibs_326_corrected_label_repeatability.render_qc",
            str(request),
        ],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise HeadModelError(
            "QC renderer failed for "
            f"{request}: stdout={completed.stdout!r}; stderr={completed.stderr!r}"
        )


def _validate_png(path: Path) -> None:
    if not path.is_file():
        raise HeadModelError(f"QC panel is missing: {path}")
    signature = path.read_bytes()[:8]
    if signature != b"\x89PNG\r\n\x1a\n":
        raise HeadModelError(f"QC panel is not a valid PNG: {path}")


def _after_marker_write(path: Path, payload: dict[str, object]) -> None:
    """Test seam for proving marker-before-mask-cleanup ordering."""
    del path, payload
