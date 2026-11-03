"""Guarded, receipt-bound workflow for the final combined analysis package."""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from .assembly import (
    AssemblyError,
    AssemblyPlan,
    build_assembly_plan,
    materialize_inputs,
)
from .established import canonical_code_manifest, run_established
from .receipts import ReceiptError, validate_terminal_receipts
from .render import render_comprehensive_outputs
from .settings import AnalysisConfig, CONDITIONS, TARGETS, VERSIONS


WORKFLOW = "repeatability_combined_analysis"
OWNER_RELATIVE = Path("_workflow/owner.json")
PLAN_RELATIVE = Path("_workflow/input_plan.json")


class WorkflowError(RuntimeError):
    """Raised when workflow ownership, provenance, or artifacts do not match."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_payload(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise WorkflowError(f"Missing or unsafe {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkflowError(f"Unreadable {label}: {path}") from error
    if not isinstance(payload, dict):
        raise WorkflowError(f"Invalid {label}: {path}")
    return payload


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and len(commit) == 40 else "unknown"


def _scope(config: AnalysisConfig) -> dict[str, int]:
    return {
        "subjects": len(config.subjects),
        "versions": len(VERSIONS),
        "targets": len(TARGETS),
        "conditions": len(CONDITIONS),
        "participant_cells": config.expected_cells,
        "run_level_rows": config.expected_rows,
        "meshing_tasks": 0,
        "fem_simulation_tasks": 0,
    }


def _snapshot(config: AnalysisConfig) -> tuple[AssemblyPlan, dict[str, Any]]:
    try:
        plan = build_assembly_plan(config)
        receipts = validate_terminal_receipts(config)
    except (AssemblyError, ReceiptError) as error:
        raise WorkflowError(str(error)) from error
    selections = []
    for selection in plan.selections:
        record = asdict(selection)
        record["path"] = str(selection.path.resolve())
        record["conditions"] = list(selection.conditions)
        record["subjects"] = list(selection.subjects)
        selections.append(record)
    core = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "output_root": str(config.output_root.resolve(strict=False)),
        "scope": _scope(config),
        "subjects": list(config.subjects),
        "repeat_count": config.repeat_count,
        "sources": selections,
        "terminal_receipts": receipts,
        "canonical_table_rows": {
            table.key: table.row_count for table in plan.tables
        },
        "total_rows": plan.total_rows,
    }
    return plan, {**core, "plan_sha256": _sha256_payload(core)}


def preflight(config: AnalysisConfig) -> dict[str, Any]:
    """Validate all analysis inputs and receipts without writing anything."""

    plan, snapshot = _snapshot(config)
    return {
        "status": "ready",
        "workflow": WORKFLOW,
        "output_root": str(config.output_root),
        "scope": _scope(config),
        "plan_sha256": snapshot["plan_sha256"],
        "source_count": len(plan.selections),
        "terminal_receipt_count": len(snapshot["terminal_receipts"]),
        "source_outputs_modified": False,
    }


def _owned_records(config: AnalysisConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    root = config.output_root
    if root.is_symlink():
        raise WorkflowError(f"Refusing symbolic-link analysis root: {root}")
    if not root.is_dir():
        raise WorkflowError(f"Analysis root is absent: {root}")
    owner_path = root / OWNER_RELATIVE
    plan_path = root / PLAN_RELATIVE
    if not owner_path.exists() and not plan_path.exists():
        raise WorkflowError(f"Refusing unowned analysis root: {root}")
    if not owner_path.is_file() or not plan_path.is_file():
        raise WorkflowError(f"Refusing incomplete owned analysis root: {root}")
    owner = _read_json(owner_path, "owner receipt")
    snapshot = _read_json(plan_path, "input plan")
    claimed_plan_hash = snapshot.get("plan_sha256")
    snapshot_core = {
        key: value for key, value in snapshot.items() if key != "plan_sha256"
    }
    if (
        not isinstance(claimed_plan_hash, str)
        or _sha256_payload(snapshot_core) != claimed_plan_hash
    ):
        raise WorkflowError("Input plan hash does not match its contents")
    if owner.get("workflow") != WORKFLOW:
        raise WorkflowError(f"Refusing unowned analysis root: {root}")
    if owner.get("output_root") != str(root.resolve()):
        raise WorkflowError("Existing owner is bound to a different output root")
    if owner.get("plan_sha256") != snapshot.get("plan_sha256"):
        raise WorkflowError("Existing owner is bound to a different plan")
    return owner, snapshot


def _require_current_snapshot(config: AnalysisConfig) -> tuple[AssemblyPlan, dict[str, Any]]:
    _, prepared = _owned_records(config)
    plan, current = _snapshot(config)
    if prepared.get("plan_sha256") != current.get("plan_sha256"):
        raise WorkflowError("An analysis source changed after preparation")
    if prepared != current:
        raise WorkflowError("The prepared input plan differs from current inputs")
    return plan, current


def prepare(config: AnalysisConfig, confirmed_root: str | None) -> dict[str, Any]:
    """Atomically create or validate an exactly confirmed owned root."""

    if confirmed_root != str(config.output_root):
        raise WorkflowError(
            f"--confirm-output-root must exactly equal: {config.output_root}"
        )
    plan, snapshot = _snapshot(config)
    root = config.output_root
    if root.exists() or root.is_symlink():
        owner, existing = _owned_records(config)
        if existing.get("plan_sha256") != snapshot.get("plan_sha256"):
            raise WorkflowError("Existing owner is bound to a different plan")
        if existing != snapshot:
            raise WorkflowError("Existing immutable input plan differs")
        other_entries = [
            path for path in root.iterdir() if path.name != "_workflow"
        ]
        if other_entries:
            if (root / "analysis_manifest.json").is_file():
                completed = _validate_complete(config)
                return {**completed, "action": "reused"}
            raise WorkflowError(
                f"Refusing incomplete owned analysis root: {root}"
            )
        return {
            "status": "prepared",
            "action": "reused",
            "workflow": WORKFLOW,
            "output_root": str(root),
            "plan_sha256": owner["plan_sha256"],
            "scope": _scope(config),
        }
    if not root.parent.is_dir() or root.parent.is_symlink():
        raise WorkflowError(f"Analysis parent is missing or unsafe: {root.parent}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.prepare-", dir=root.parent)
    )
    promoted = False
    try:
        owner = {
            "schema_version": 1,
            "status": "owned",
            "workflow": WORKFLOW,
            "created_utc": _utc_now(),
            "git_commit": _git_commit(),
            "output_root": str(root.resolve(strict=False)),
            "plan_sha256": snapshot["plan_sha256"],
        }
        _write_json(temporary / OWNER_RELATIVE, owner)
        _write_json(temporary / PLAN_RELATIVE, snapshot)
        os.replace(temporary, root)
        promoted = True
    finally:
        if not promoted and temporary.exists():
            shutil.rmtree(temporary)
    return {
        "status": "prepared",
        "action": "created",
        "workflow": WORKFLOW,
        "output_root": str(root),
        "plan_sha256": snapshot["plan_sha256"],
        "scope": _scope(config),
    }


def _combined_frame(inputs: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for key in sorted(inputs):
        version, target = key.split("_", 1)
        frame = pd.read_csv(inputs[key])
        frame.insert(0, "target", target)
        frame.insert(0, "version", version)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if len(combined) != 3200:
        raise WorkflowError(
            f"Expected 3,200 assembled run rows, observed {len(combined)}"
        )
    return combined


def _analysis_code_manifest() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    local = {}
    for path in sorted(package_root.glob("*.py")):
        local[path.name] = {"path": str(path), "sha256": _sha256_file(path)}
    return {"local": local, "canonical": canonical_code_manifest()}


def _environment() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _readme() -> str:
    return """# Final-132 combined repeatability analysis v1

Source-bound analysis of the canonical nine retained participants plus
`sub-CC510415`, across SimNIBS 3.2.6 and SimNIBS 4, left hippocampus and right
M1, and remesh and spherical-median fixed-mesh conditions. Each participant
cell contains 40 technical repeats; participant is the inferential unit.

The package contains four immutable canonical input tables, established
analysis outputs, the complete condition-inclusive comparison, provenance,
checksums, and a portable archive. It performs no meshing or FEM simulation.
"""


def _artifact_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in {"analysis_manifest.json", "SHA256SUMS"}:
            continue
        if relative.startswith(".") or relative.startswith("_workflow/"):
            continue
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return records


def _write_checksums(root: Path) -> None:
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS"
        and not path.relative_to(root).as_posix().startswith(".")
        and not path.relative_to(root).as_posix().startswith("_workflow/")
        and not path.name.endswith(".tar.gz")
        and not path.name.endswith(".tar.gz.sha256")
    )
    text = "".join(
        f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
        for path in candidates
    )
    _write_text(root / "SHA256SUMS", text)


def _verify_checksums(root: Path) -> int:
    checksum_path = root / "SHA256SUMS"
    if not checksum_path.is_file() or checksum_path.is_symlink():
        raise WorkflowError("Missing SHA256SUMS")
    count = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        try:
            expected, relative_text = line.split("  ", 1)
        except ValueError as error:
            raise WorkflowError("Invalid checksum line") from error
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise WorkflowError("Unsafe checksum path")
        path = root / relative
        if not path.is_file() or path.is_symlink() or _sha256_file(path) != expected:
            raise WorkflowError(f"Artifact checksum mismatch: {relative_text}")
        count += 1
    if count == 0:
        raise WorkflowError("SHA256SUMS is empty")
    return count


def _write_archive(
    root: Path, package_name: str | None = None
) -> tuple[Path, Path]:
    archive = root / f"{package_name or root.name}.tar.gz"
    temporary = root / f".{archive.name}.tmp-{os.getpid()}"
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.relative_to(root).as_posix().startswith(".")
        and not path.relative_to(root).as_posix().startswith("_workflow/")
        and path not in {archive, archive.with_suffix(archive.suffix + ".sha256")}
    )
    with tarfile.open(temporary, "w:gz") as handle:
        for path in candidates:
            handle.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    os.replace(temporary, archive)
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    _write_text(sidecar, f"{_sha256_file(archive)}  {archive.name}\n")
    return archive, sidecar


def _validate_archive(root: Path, package_name: str | None = None) -> None:
    archive = root / f"{package_name or root.name}.tar.gz"
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    if not archive.is_file() or not sidecar.is_file():
        raise WorkflowError("Analysis archive or sidecar is missing")
    try:
        expected, name = sidecar.read_text(encoding="utf-8").strip().split("  ", 1)
    except ValueError as error:
        raise WorkflowError("Invalid archive checksum sidecar") from error
    if name != archive.name or expected != _sha256_file(archive):
        raise WorkflowError("Analysis archive checksum mismatch")
    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
    if not names or any(
        Path(name).is_absolute()
        or ".." in Path(name).parts
        or "_workflow" in Path(name).parts
        or any(part.startswith(".attempt-") for part in Path(name).parts)
        or Path(name).name in {archive.name, sidecar.name}
        for name in names
    ):
        raise WorkflowError("Analysis archive contains unsafe or internal members")


def _validate_complete(config: AnalysisConfig) -> dict[str, Any]:
    root = config.output_root
    manifest = _read_json(root / "analysis_manifest.json", "analysis manifest")
    if manifest.get("status") != "complete" or manifest.get("workflow") != WORKFLOW:
        raise WorkflowError("Analysis completion manifest is invalid")
    if manifest.get("output_root") != str(root.resolve()):
        raise WorkflowError("Analysis manifest is bound to another root")
    if manifest.get("scope") != _scope(config):
        raise WorkflowError("Analysis manifest scope is invalid")
    _verify_checksums(root)
    _validate_archive(root)
    return manifest


def execute(config: AnalysisConfig) -> dict[str, Any]:
    """Execute through a private attempt and promote only a verified package."""

    _owned_records(config)
    manifest_path = config.output_root / "analysis_manifest.json"
    if manifest_path.exists():
        manifest = _validate_complete(config)
        return {**manifest, "action": "reused"}
    plan, snapshot = _require_current_snapshot(config)
    root = config.output_root
    for old_attempt in root.glob(".attempt-*"):
        if old_attempt.is_dir() and not old_attempt.is_symlink():
            shutil.rmtree(old_attempt)
        else:
            raise WorkflowError(f"Unsafe prior attempt path: {old_attempt}")
    attempt = Path(tempfile.mkdtemp(prefix=".attempt-", dir=root))
    try:
        inputs = materialize_inputs(plan, attempt / "inputs")
        frame = _combined_frame(inputs)
        established_result = run_established(inputs, attempt / "established")
        comprehensive_result = render_comprehensive_outputs(
            frame, attempt / "comprehensive"
        )
        _write_text(attempt / "README.md", _readme())
        _write_text(attempt / "subjects.txt", "\n".join(config.subjects) + "\n")
        owner = _read_json(root / OWNER_RELATIVE, "owner receipt")
        _write_json(attempt / "provenance/owner.json", owner)
        _write_json(
            attempt / "provenance/source_receipts.json",
            {
                "status": "complete",
                "receipts": snapshot["terminal_receipts"],
            },
        )
        code_manifest = _analysis_code_manifest()
        environment = _environment()
        _write_json(attempt / "provenance/analysis_code.json", code_manifest)
        _write_json(attempt / "provenance/run_environment.json", environment)
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "workflow": WORKFLOW,
            "created_utc": _utc_now(),
            "git_commit": _git_commit(),
            "output_root": str(root.resolve()),
            "plan_sha256": snapshot["plan_sha256"],
            "scope": _scope(config),
            "source_outputs_modified": False,
            "analysis_code": code_manifest,
            "environment": environment,
            "established": established_result,
            "comprehensive": comprehensive_result,
            "artifacts": _artifact_records(attempt),
        }
        _write_json(attempt / "analysis_manifest.json", manifest)
        _write_checksums(attempt)
        _verify_checksums(attempt)
        _write_archive(attempt, root.name)
        _validate_archive(attempt, root.name)
    except WorkflowError:
        # Keep the private attempt as evidence of an interrupted or failed run.
        raise
    except Exception as error:
        # Keep the private attempt as evidence of an interrupted or failed run.
        raise WorkflowError(f"Analysis execution failed: {error}") from error

    promoted_names = sorted(path.name for path in attempt.iterdir())
    for name in promoted_names:
        destination = root / name
        if destination.exists() or destination.is_symlink():
            raise WorkflowError(f"Refusing to overwrite promoted path: {destination}")
    for name in promoted_names:
        os.replace(attempt / name, root / name)
    attempt.rmdir()
    completed = _validate_complete(config)
    return {**completed, "action": "created"}


def status(config: AnalysisConfig) -> dict[str, Any]:
    """Independently classify absent, prepared, incomplete, complete, or invalid."""

    root = config.output_root
    if not root.exists() and not root.is_symlink():
        return {"status": "ok", "state": "absent", "output_root": str(root)}
    try:
        _owned_records(config)
        if (root / "analysis_manifest.json").exists():
            manifest = _validate_complete(config)
            return {
                "status": "ok",
                "state": "complete",
                "output_root": str(root),
                "plan_sha256": manifest["plan_sha256"],
            }
        attempts = list(root.glob(".attempt-*"))
        scientific_entries = [
            path.name for path in root.iterdir() if path.name != "_workflow"
        ]
        state = (
            "running-or-incomplete"
            if attempts or scientific_entries
            else "prepared"
        )
        return {"status": "ok", "state": state, "output_root": str(root)}
    except WorkflowError as error:
        return {
            "status": "error",
            "state": "invalid",
            "output_root": str(root),
            "error": str(error),
        }


def rebind_preflight(
    config: AnalysisConfig, dataset_root: Path
) -> dict[str, Any]:
    """Read-only proof that dataset destinations match all metric sources."""

    dataset_root = Path(dataset_root)
    receipt = _read_json(
        dataset_root / "_consolidation/complete.json",
        "dataset completion receipt",
    )
    if receipt.get("status") != "complete" or receipt.get("workflow") != "repeatability_dataset_consolidation":
        raise WorkflowError("Dataset completion receipt is not complete")
    mapping_path = dataset_root / "provenance/source_mapping.tsv"
    if not mapping_path.is_file() or mapping_path.is_symlink():
        raise WorkflowError("Dataset source mapping is missing")
    try:
        with mapping_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    except OSError as error:
        raise WorkflowError("Dataset source mapping is unreadable") from error
    by_source: dict[str, dict[str, str]] = {}
    for row in rows:
        source = row.get("source", "")
        if not source or source in by_source:
            raise WorkflowError("Dataset source mapping has duplicate or empty sources")
        by_source[source] = row
    plan, _ = _snapshot(config)
    bindings = []
    resolved_dataset = dataset_root.resolve()
    for selection in plan.selections:
        source = str(selection.path.resolve())
        row = by_source.get(source)
        if row is None:
            raise WorkflowError(f"No exact dataset source mapping for: {source}")
        destination = Path(row.get("destination", ""))
        try:
            resolved_destination = destination.resolve(strict=True)
        except OSError as error:
            raise WorkflowError(
                f"Mapped destination is missing: {destination}"
            ) from error
        if (
            resolved_dataset != resolved_destination
            and resolved_dataset not in resolved_destination.parents
        ):
            raise WorkflowError(
                f"Mapped destination escapes dataset root: {destination}"
            )
        if not destination.is_file() or destination.is_symlink():
            raise WorkflowError(f"Mapped destination is not a regular file: {destination}")
        observed = _sha256_file(destination)
        if observed != selection.sha256:
            raise WorkflowError(f"Mapped destination hash differs: {destination}")
        if row.get("verification_status") != "checksum_matched":
            raise WorkflowError(f"Mapped destination lacks checksum evidence: {destination}")
        bindings.append(
            {
                "source": source,
                "destination": str(resolved_destination),
                "sha256": observed,
            }
        )
    return {
        "status": "matched",
        "workflow": WORKFLOW,
        "dataset_root": str(resolved_dataset),
        "bindings": len(bindings),
        "mapped_sources": bindings,
        "writes_performed": 0,
    }

