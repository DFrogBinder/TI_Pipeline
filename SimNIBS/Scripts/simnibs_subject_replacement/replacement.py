"""Read-only audit and atomic preparation for one replacement subject."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .settings import (
    ReplacementConfig,
    ReplacementPaths,
    assert_replacement_root_isolation,
    replacement_paths,
    replacement_scope,
    validate_candidate,
)
from .staging import audit_corrected_sources, stage_corrected_dataset


SCHEMA_VERSION = 1
WORKFLOW = "simnibs_subject_replacement"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required input is missing: {path}") from exc
    if not resolved.is_file() or resolved.is_symlink() or resolved.stat().st_size <= 0:
        raise ValueError(f"Required input is not a non-empty regular file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _legacy_reference_inventory(config: ReplacementConfig) -> dict[str, object]:
    anat = config.legacy_v3_root / config.subject / "anat"
    m2m = anat / f"m2m_{config.subject}"
    marker = anat / ".simnibs326_scaffold_ready.json"
    required_files = [
        m2m / "T1fs_conform.nii.gz",
        m2m / "T1fs_nu_conform.nii.gz",
        marker,
    ]
    for tree_name in ("eeg_positions", "toMNI"):
        tree = m2m / tree_name
        if not tree.is_dir() or tree.is_symlink():
            raise FileNotFoundError(
                f"Required legacy reference directory is missing or unsafe: {tree}"
            )
        files = sorted(path for path in tree.rglob("*") if path.is_file())
        if not files:
            raise ValueError(f"Required legacy reference directory is empty: {tree}")
        required_files.extend(files)
    return {
        "anat_root": str(anat.resolve(strict=False)),
        "m2m_root": str(m2m.resolve(strict=False)),
        "files": [_file_record(path) for path in required_files],
    }


def _paths_payload(paths: ReplacementPaths) -> dict[str, str]:
    return {
        name: str(getattr(paths, name))
        for name in (
            "root",
            "pipeline_root",
            "manifest",
            "singleton_subjects_file",
            "v4_corrected_root",
            "v3_headmodel_root",
            "v3_left_root",
            "v3_right_root",
            "v4_left_root",
            "v4_right_root",
        )
    }


def _corrected_inventory(config: ReplacementConfig) -> tuple[list[Any], list[dict[str, object]]]:
    ready, issues = audit_corrected_sources(
        config.corrected_v4_scaffold_root, config.subject
    )
    if issues:
        raise ValueError("Corrected-v4 source audit failed: " + "; ".join(issues))
    if len(ready) != 3 or {item.kind for item in ready} != {
        "t1",
        "t2",
        "segmentation",
    }:
        raise ValueError(
            f"Corrected-v4 source audit returned {len(ready)} files; expected three"
        )
    records = [
        {
            "kind": item.kind,
            "source_path": str(item.source_path.resolve()),
            "source_sha256": item.source_sha256,
            "expected_source_sha256": item.expected_source_sha256,
            "destination_name": item.destination_name,
            "transform": item.transform,
            "provenance_record": item.provenance_record,
        }
        for item in sorted(ready, key=lambda item: item.kind)
    ]
    return ready, records


def preflight(config: ReplacementConfig) -> dict[str, object]:
    """Audit all source evidence without creating any output path."""
    validate_candidate(config)
    paths = replacement_paths(config)
    isolation = assert_replacement_root_isolation(config, paths)
    if paths.root.exists() and not paths.manifest.is_file():
        if any(paths.root.iterdir()):
            raise RuntimeError(
                f"Replacement root is non-empty and unowned: {paths.root}"
            )
    ready, corrected = _corrected_inventory(config)
    del ready
    atlas = config.atlas_root / f"{config.subject}.nii.gz"
    inputs = {
        "authoritative_subjects": _file_record(config.final132_subjects_file),
        "corrected_v4": corrected,
        "legacy_reference": _legacy_reference_inventory(config),
        "atlas": _file_record(atlas),
        "targets_csv": _file_record(config.targets_csv),
    }
    scope = replacement_scope(config)
    scope["root_isolation"] = isolation
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "profile": "external1",
        "status": "ready",
        "subject": config.subject,
        "subjects": [config.subject],
        "replaces": config.replaces,
        "paths": _paths_payload(paths),
        "inputs": inputs,
        "scope": scope,
        "source_outputs_modified": False,
        "jobs_submitted": 0,
    }


def load_prepared_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Prepared replacement manifest is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Prepared replacement manifest is not a JSON object")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("workflow") != WORKFLOW
        or payload.get("profile") != "external1"
        or payload.get("status") != "prepared"
    ):
        raise RuntimeError("Prepared replacement manifest has a stale or invalid binding")
    paths = payload.get("paths")
    if not isinstance(paths, dict) or paths.get("manifest") != str(path.resolve()):
        raise RuntimeError("Prepared replacement manifest is bound to a different path")
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(config: ReplacementConfig) -> dict[str, object]:
    """Atomically create the singleton stage and immutable prepared manifest."""
    audited = preflight(config)
    expected = {**audited, "status": "prepared"}
    paths = replacement_paths(config)
    if paths.root.exists():
        if not paths.manifest.is_file():
            raise RuntimeError(f"Replacement root is non-empty and unowned: {paths.root}")
        current = load_prepared_manifest(paths.manifest)
        if current != expected:
            raise RuntimeError(
                "Existing prepared manifest is stale or mismatched; refusing adoption"
            )
        return {"action": "reused", **current}

    ready, issues = audit_corrected_sources(
        config.corrected_v4_scaffold_root, config.subject
    )
    if issues or len(ready) != 3:
        raise ValueError("Corrected-v4 source audit changed during preparation")
    work_root = config.work_root.expanduser().resolve(strict=False)
    work_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{config.subject}.prepare-", dir=work_root)
    )
    promoted = False
    try:
        temporary_pipeline = temporary / "_pipeline"
        temporary_subjects = temporary_pipeline / "subjects.txt"
        temporary_pipeline.mkdir(parents=True)
        temporary_subjects.write_text(f"{config.subject}\n", encoding="utf-8")
        stage_corrected_dataset(
            config.corrected_v4_scaffold_root,
            temporary / "inputs" / "corrected_v4",
            temporary_subjects,
            config.subject,
            ready,
        )
        _write_json(temporary_pipeline / "replacement_manifest.json", expected)
        os.replace(temporary, paths.root)
        promoted = True
    finally:
        if not promoted and temporary.exists():
            shutil.rmtree(temporary)
    return {"action": "created", **load_prepared_manifest(paths.manifest)}
