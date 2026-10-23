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


def _require_current_manifest(manifest: dict[str, object]) -> dict[str, str]:
    if (
        manifest.get("workflow") != WORKFLOW
        or manifest.get("profile") != "external1"
        or manifest.get("status") != "prepared"
    ):
        raise RuntimeError("Replacement manifest is not a prepared external1 manifest")
    paths = manifest.get("paths")
    if not isinstance(paths, dict):
        raise RuntimeError("Replacement manifest has no path binding")
    manifest_path = Path(str(paths.get("manifest", ""))).resolve(strict=False)
    current = load_prepared_manifest(manifest_path)
    if current != manifest:
        raise RuntimeError("Replacement manifest differs from its immutable on-disk copy")
    return {str(key): str(value) for key, value in paths.items()}


def require_v3_promotion(manifest: dict[str, object]) -> dict[str, object]:
    """Require a reviewed, hash-valid singleton v3 head-model promotion."""
    paths = _require_current_manifest(manifest)
    subject = str(manifest.get("subject", ""))
    collection = (
        Path(paths["v3_headmodel_root"])
        / "_pipeline"
        / "headmodels"
        / "external1"
    )
    headmodel_manifest_path = collection / "manifest.json"
    promotion_path = collection / "promoted.json"
    try:
        headmodel_manifest = json.loads(
            headmodel_manifest_path.read_text(encoding="utf-8")
        )
        promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Reviewed v3 head-model promotion is missing or unreadable: {promotion_path}"
        ) from exc
    if (
        not isinstance(headmodel_manifest, dict)
        or headmodel_manifest.get("status") != "complete"
        or headmodel_manifest.get("headmodel_root") != paths["v3_headmodel_root"]
        or headmodel_manifest.get("subject_count") != 1
        or headmodel_manifest.get("subject_order") != [subject]
    ):
        raise RuntimeError("V3 head-model manifest has a different subject/root binding")
    if not isinstance(promotion, dict) or promotion.get("status") != "promoted":
        raise RuntimeError("V3 head-model promotion is not complete")
    if promotion.get("subjects") != [subject]:
        raise RuntimeError("V3 head-model promotion has a different subject binding")
    if promotion.get("confirm_reviewed") is not True:
        raise RuntimeError("V3 head-model promotion does not prove human review")
    if Path(str(promotion.get("manifest_path", ""))).resolve(
        strict=False
    ) != headmodel_manifest_path.resolve():
        raise RuntimeError("V3 promotion names a different head-model manifest")
    observed = _sha256(headmodel_manifest_path)
    if promotion.get("manifest_sha256") != observed:
        raise RuntimeError("V3 promotion head-model manifest hash mismatch")
    return {
        "status": "ready",
        "subject": subject,
        "subjects": [subject],
        "promotion_path": str(promotion_path),
        "promotion_sha256": _sha256(promotion_path),
        "headmodel_manifest_path": str(headmodel_manifest_path),
        "headmodel_manifest_sha256": observed,
        "replacement_manifest_sha256": _sha256(Path(paths["manifest"])),
    }


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is missing or unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return payload


def require_left_complete(
    manifest: dict[str, object], version: str
) -> dict[str, object]:
    """Require the same-version left target to be terminal before right M1."""
    if version not in ("v3", "v4"):
        raise ValueError("version must be v3 or v4")
    require_v3_promotion(manifest)
    paths = _require_current_manifest(manifest)
    subject = str(manifest["subject"])
    root = Path(paths[f"{version}_left_root"])
    completion_path = root / "_pipeline" / "workflow" / "complete.json"
    completion = _load_json_object(
        completion_path, f"{version} left-hippocampus terminal receipt"
    )
    if completion.get("status") != "complete":
        raise RuntimeError(f"{version} left-hippocampus receipt is not complete")

    config_path = root / "_pipeline" / "configs" / "paired_analysis.json"
    config = _load_json_object(
        config_path, f"{version} left-hippocampus paired config"
    )
    if config.get("subjects") != [subject]:
        raise RuntimeError(f"{version} left-hippocampus config has a different subject")
    conditions = config.get("conditions")
    if not isinstance(conditions, list) or [
        (item.get("name"), item.get("repeat_count"))
        for item in conditions
        if isinstance(item, dict)
    ] != [("remesh", 40), ("fixed_mesh", 40)]:
        raise RuntimeError(
            f"{version} left-hippocampus config does not prove 40 remesh and fixed repeats"
        )

    if version == "v3":
        if completion.get("stage") != "final":
            raise RuntimeError("v3 left-hippocampus receipt is not the final stage")
        if Path(str(completion.get("config", ""))).resolve(
            strict=False
        ) != config_path.resolve() or completion.get("config_sha256") != _sha256(
            config_path
        ):
            raise RuntimeError("v3 left-hippocampus receipt/config hash mismatch")
        scope = completion.get("scope")
        validation = completion.get("validation")
        if (
            not isinstance(scope, dict)
            or scope.get("target") != "left-hippocampus"
            or scope.get("subject_count") != 1
            or not isinstance(validation, dict)
            or validation.get("total_tasks") != 80
        ):
            raise RuntimeError("v3 left-hippocampus receipt has the wrong scope")
    else:
        scope = completion.get("scope")
        if (
            not isinstance(scope, dict)
            or scope.get("roi") != "left-hippocampus"
            or scope.get("subjects") != [subject]
            or scope.get("subject_count") != 1
            or scope.get("repeats_per_condition") != 40
            or scope.get("total_simulation_tasks") != 80
        ):
            raise RuntimeError("v4 left-hippocampus receipt has the wrong scope")
    return {
        "status": "ready",
        "version": version,
        "subject": subject,
        "target": "left-hippocampus",
        "completion_path": str(completion_path),
        "completion_sha256": _sha256(completion_path),
    }
