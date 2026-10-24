"""Read-only audit, owned build, and validation for one v3.2.6 scaffold."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from simnibs_326_repeatability.cat12_compat import PATCH_ID
from simnibs_326_repeatability.scaffold_core import (
    ScaffoldRequest,
    build_scaffold,
    scaffold_paths,
    sha256_file,
    validate_scaffold,
)
from simnibs_326_repeatability.settings import (
    HEAD_MODEL_STRATEGY,
    MATLAB_MODULE,
    SEGMENTATION_PROVENANCE,
    SIMNIBS_MODULE,
)
from simnibs_subject_replacement.staging import audit_corrected_sources

from .settings import (
    CreatorConfig,
    CreatorPaths,
    assert_output_root_isolation,
    creator_paths,
    validate_subject,
)


SCHEMA_VERSION = 1
WORKFLOW = "simnibs_326_subject_scaffold_creator"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is missing or unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_record(path: Path, label: str) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise RuntimeError(f"{label} is missing or empty: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _corrected_inventory(
    config: CreatorConfig,
) -> tuple[list[Any], list[dict[str, object]]]:
    ready, issues = audit_corrected_sources(
        config.corrected_source_root, config.subject
    )
    if issues:
        raise ValueError("Corrected-v4 source audit failed: " + "; ".join(issues))
    expected = {"t1", "t2", "segmentation"}
    if len(ready) != 3 or {item.kind for item in ready} != expected:
        raise ValueError(
            f"Corrected-v4 source audit returned {len(ready)} files; expected three"
        )
    records: list[dict[str, object]] = []
    for item in sorted(ready, key=lambda item: item.kind):
        provenance_record = Path(item.provenance_record).expanduser().resolve()
        if not provenance_record.is_file() or provenance_record.stat().st_size <= 0:
            raise ValueError(
                "Corrected-v4 provenance receipt is missing or empty: "
                f"{provenance_record}"
            )
        records.append(
            {
                "kind": item.kind,
                "source_path": str(item.source_path.resolve()),
                "source_sha256": item.source_sha256,
                "expected_source_sha256": item.expected_source_sha256,
                "destination_name": item.destination_name,
                "transform": item.transform,
                "provenance_record": str(provenance_record),
                "provenance_record_sha256": sha256_file(provenance_record),
            }
        )
    return ready, records


def _paths_payload(paths: CreatorPaths) -> dict[str, str]:
    return {
        "subject_anat": str(paths.subject_anat),
        "module_receipt": str(paths.module_receipt),
        "owner_receipt": str(paths.owner_receipt),
        "completion_receipt": str(paths.completion_receipt),
        "log_dir": str(paths.log_dir),
    }


def _owner_binding(
    config: CreatorConfig, source_records: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "subject": config.subject,
        "output_root": str(config.output_root.expanduser().resolve(strict=False)),
        "simnibs_module": SIMNIBS_MODULE,
        "cat12_compatibility_patch": PATCH_ID,
        "head_model_strategy": HEAD_MODEL_STRATEGY,
        "segmentation_provenance": SEGMENTATION_PROVENANCE,
        "authoritative_subjects": _file_record(
            config.final132_subjects_file, "authoritative final-132 subject list"
        ),
        "sources": source_records,
    }


def _matches_binding(payload: dict[str, object], binding: dict[str, object]) -> bool:
    return all(payload.get(key) == value for key, value in binding.items())


def _request(config: CreatorConfig, ready: list[Any]) -> ScaffoldRequest:
    sources = {item.kind: item.source_path.resolve() for item in ready}
    return ScaffoldRequest(
        subject=config.subject,
        t1=sources["t1"],
        t2=sources["t2"],
        corrected_labels=sources["segmentation"],
        scaffold_root=config.output_root.expanduser().resolve(strict=False),
        simnibs_module=SIMNIBS_MODULE,
        cat12_compat_script=(
            Path(__file__).resolve().parent.parent
            / "simnibs_326_repeatability"
            / "cat12_compat.py"
        ),
        cat12_patch_id=PATCH_ID,
        head_model_strategy=HEAD_MODEL_STRATEGY,
        segmentation_provenance=SEGMENTATION_PROVENANCE,
        workflow=WORKFLOW,
    )


def preflight(config: CreatorConfig) -> dict[str, object]:
    """Audit membership, sources, isolation, and ownership without mutation."""
    cohort = validate_subject(config)
    paths = creator_paths(config)
    isolation = assert_output_root_isolation(config)
    ready, source_records = _corrected_inventory(config)
    del ready
    binding = _owner_binding(config, source_records)
    owner_present = paths.owner_receipt.is_file()
    if owner_present:
        owner = _read_json(paths.owner_receipt, "creator owner receipt")
        if not _matches_binding(owner, binding):
            raise RuntimeError(
                "Existing creator owner receipt has a mismatch or different binding"
            )
    if paths.subject_anat.exists() and not owner_present:
        raise RuntimeError(
            f"Existing subject scaffold is unowned; refusing adoption: "
            f"{paths.subject_anat}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "status": "ready",
        "subject": config.subject,
        "final132_subject_count": len(cohort),
        "simnibs_module": SIMNIBS_MODULE,
        "matlab_module": MATLAB_MODULE,
        "headreco_command": "headreco all --noclean",
        "paths": _paths_payload(paths),
        "sources": source_records,
        "authoritative_subjects": binding["authoritative_subjects"],
        "root_isolation": isolation,
        "owner_state": "matched" if owner_present else "absent",
        "source_outputs_modified": False,
        "jobs_submitted": 0,
    }


def _validate_module_payload(
    payload: dict[str, object], label: str
) -> dict[str, object]:
    compatibility = payload.get("cat12_compatibility")
    if (
        payload.get("status") != "ready"
        or payload.get("expected_module") != SIMNIBS_MODULE
        or payload.get("expected_matlab_module") != MATLAB_MODULE
        or not isinstance(compatibility, dict)
        or compatibility.get("status") != "ready"
        or compatibility.get("patch_id") != PATCH_ID
    ):
        raise RuntimeError(f"{label} is stale or bound to a different runtime")
    return payload


def _module_receipt(paths: CreatorPaths) -> dict[str, object]:
    payload = _read_json(paths.module_receipt, "module preflight receipt")
    return _validate_module_payload(payload, "Module preflight receipt")


def _completion_binding(
    config: CreatorConfig,
    paths: CreatorPaths,
    source_records: list[dict[str, object]],
    request: ScaffoldRequest,
) -> dict[str, object]:
    scaffold = scaffold_paths(request)
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "status": "ready",
        "subject": config.subject,
        "output_root": str(config.output_root.expanduser().resolve(strict=False)),
        "simnibs_module": SIMNIBS_MODULE,
        "matlab_module": MATLAB_MODULE,
        "sources": source_records,
        "owner_receipt": str(paths.owner_receipt),
        "owner_receipt_sha256": sha256_file(paths.owner_receipt),
        "module_receipt": str(paths.module_receipt),
        "ready_marker": str(scaffold.ready_marker),
        "ready_marker_sha256": sha256_file(scaffold.ready_marker),
        "native_mesh": str(scaffold.native_mesh),
        "native_mesh_sha256": sha256_file(scaffold.native_mesh),
        "compatibility_mesh": str(scaffold.compatibility_mesh),
        "compatibility_mesh_sha256": sha256_file(scaffold.compatibility_mesh),
    }


def validate(config: CreatorConfig) -> dict[str, object]:
    audited = preflight(config)
    paths = creator_paths(config)
    _module_receipt(paths)
    ready, source_records = _corrected_inventory(config)
    request = _request(config, ready)
    scaffold = validate_scaffold(request)
    if scaffold.get("status") != "ready":
        raise RuntimeError(
            "SimNIBS 3.2.6 scaffold is incomplete or stale: "
            + json.dumps(scaffold, sort_keys=True)
        )
    owner = _read_json(paths.owner_receipt, "creator owner receipt")
    if not _matches_binding(owner, _owner_binding(config, source_records)):
        raise RuntimeError("Creator owner receipt differs from the current request")
    completion = _read_json(paths.completion_receipt, "creator completion receipt")
    expected = _completion_binding(config, paths, source_records, request)
    if not _matches_binding(completion, expected):
        raise RuntimeError(
            "Creator completion receipt is stale or bound to different evidence"
        )
    historical_module_sha256 = completion.get("module_receipt_sha256")
    module_evidence = completion.get("module_receipt_evidence")
    module_evidence_sha256 = completion.get("module_receipt_evidence_sha256")
    if (
        not isinstance(historical_module_sha256, str)
        or len(historical_module_sha256) != 64
        or not isinstance(module_evidence, dict)
        or not isinstance(module_evidence_sha256, str)
        or module_evidence_sha256 != _json_sha256(module_evidence)
    ):
        raise RuntimeError(
            "Creator completion receipt has invalid historical module evidence"
        )
    _validate_module_payload(module_evidence, "Historical module evidence")
    return {
        **audited,
        "status": "ready",
        "scaffold": scaffold,
        "owner_receipt_sha256": sha256_file(paths.owner_receipt),
        "completion_receipt_sha256": sha256_file(paths.completion_receipt),
        "module_receipt_sha256": sha256_file(paths.module_receipt),
        "historical_module_receipt_sha256": historical_module_sha256,
    }


def build(config: CreatorConfig) -> dict[str, object]:
    preflight(config)
    paths = creator_paths(config)
    module_evidence = _module_receipt(paths)
    if paths.completion_receipt.is_file():
        try:
            result = validate(config)
        except RuntimeError:
            # A matching owner receipt authorizes rebuilding stale or partial
            # creator output. Source-audit failures already fail in preflight.
            pass
        else:
            result["action"] = "reused"
            return result

    ready, source_records = _corrected_inventory(config)
    binding = _owner_binding(config, source_records)
    if not paths.owner_receipt.exists():
        _atomic_json(paths.owner_receipt, {**binding, "created_utc": _utc_now()})
    owner = _read_json(paths.owner_receipt, "creator owner receipt")
    if not _matches_binding(owner, binding):
        raise RuntimeError("Creator owner receipt differs from the current request")

    request = _request(config, ready)
    scaffold = build_scaffold(request)
    if scaffold.get("status") != "ready":
        raise RuntimeError("Shared scaffold builder did not return ready")
    completion = {
        **_completion_binding(config, paths, source_records, request),
        "module_receipt_sha256": sha256_file(paths.module_receipt),
        "module_receipt_evidence": module_evidence,
        "module_receipt_evidence_sha256": _json_sha256(module_evidence),
        "created_utc": _utc_now(),
    }
    _atomic_json(paths.completion_receipt, completion)
    result = validate(config)
    result["action"] = str(scaffold.get("action", "built"))
    return result


def status(config: CreatorConfig) -> dict[str, object]:
    """Report filesystem state without creating or modifying any path."""
    paths = creator_paths(config)
    try:
        audited = preflight(config)
    except Exception as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "workflow": WORKFLOW,
            "status": "invalid",
            "subject": config.subject,
            "paths": _paths_payload(paths),
            "error": str(exc),
            "owner_receipt_present": paths.owner_receipt.is_file(),
            "subject_anat_present": paths.subject_anat.exists(),
            "completion_receipt_present": paths.completion_receipt.is_file(),
        }
    if not paths.completion_receipt.is_file():
        return {
            **audited,
            "status": "not-built",
            "owner_receipt_present": paths.owner_receipt.is_file(),
            "subject_anat_present": paths.subject_anat.exists(),
            "completion_receipt_present": False,
        }
    try:
        return validate(config)
    except Exception as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "workflow": WORKFLOW,
            "status": "incomplete",
            "subject": config.subject,
            "paths": _paths_payload(paths),
            "error": str(exc),
            "owner_receipt_present": paths.owner_receipt.is_file(),
            "subject_anat_present": paths.subject_anat.exists(),
            "completion_receipt_present": True,
        }
