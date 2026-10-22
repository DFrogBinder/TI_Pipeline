"""Safe, allowlisted staging of legacy v3 coordinate references."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
from typing import Any

from .settings import CampaignPaths, SUBJECTS, SubjectPaths, subject_paths


PROVENANCE_MARKER_NAME = "legacy_scaffold_ready.provenance.json"
ROOT_REFERENCE_FILES = ("T1fs_conform.nii.gz", "T1fs_nu_conform.nii.gz")
REFERENCE_TREES = ("eeg_positions", "toMNI")
FORBIDDEN_SUFFIXES = (".msh", ".stl", ".off")


class ReferenceStageError(RuntimeError):
    """Raised when reference staging cannot prove an exact safe copy."""


@dataclass(frozen=True)
class StageReceipt:
    subject: str
    source_root: str
    destination_root: str
    receipt_path: str
    file_count: int
    byte_count: int
    reused: bool = False


@dataclass(frozen=True)
class _PlannedFile:
    source: Path
    source_relative: str
    destination_relative: str
    byte_count: int
    source_sha256: str


def stage_subject_reference(subject_paths: SubjectPaths) -> StageReceipt:
    """Copy the immutable reference allowlist into one atomically promoted tree."""
    _validate_roots(subject_paths)
    destination = subject_paths.staged_m2m.resolve(strict=False)
    receipt_path = subject_paths.reference_receipt
    if destination.exists() or receipt_path.exists():
        if destination.is_dir() and receipt_path.is_file():
            try:
                return replace(validate_staged_reference(subject_paths), reused=True)
            except ReferenceStageError as exc:
                raise ReferenceStageError(
                    f"Existing staged reference is incomplete or invalid: {destination}"
                ) from exc
        raise ReferenceStageError(
            f"Existing staged reference is incomplete; refusing replacement: {destination}"
        )

    planned = _enumerate_allowlist(subject_paths)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    )
    promoted = False
    try:
        records: list[dict[str, Any]] = []
        for item in planned:
            target = _safe_destination(temporary, item.destination_relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy_file(item.source, target)
            destination_hash = _sha256(target)
            if destination_hash != item.source_sha256:
                raise ReferenceStageError(
                    f"Copied file hash mismatch for {item.source}: {destination_hash}"
                )
            records.append(
                {
                    "source_path": str(item.source),
                    "source_relative": item.source_relative,
                    "destination_relative": item.destination_relative,
                    "byte_count": item.byte_count,
                    "source_sha256": item.source_sha256,
                    "destination_sha256": destination_hash,
                }
            )
        for item in planned:
            if _sha256(item.source) != item.source_sha256:
                raise ReferenceStageError(
                    f"Legacy source changed while it was staged: {item.source}"
                )
        os.replace(temporary, destination)
        promoted = True
        payload = {
            "schema_version": 1,
            "status": "complete",
            "created_utc": _utc_now(),
            "git_commit": _git_commit(),
            "subject": subject_paths.subject,
            "source_root": str(subject_paths.legacy_m2m.resolve()),
            "destination_root": str(destination),
            "allowlist": {
                "root_files": list(ROOT_REFERENCE_FILES),
                "recursive_trees": list(REFERENCE_TREES),
                "legacy_marker_destination": PROVENANCE_MARKER_NAME,
            },
            "file_count": len(records),
            "byte_count": sum(int(record["byte_count"]) for record in records),
            "files": records,
        }
        _atomic_write_json(receipt_path, payload)
        return validate_staged_reference(subject_paths)
    except Exception:
        if promoted:
            shutil.rmtree(destination, ignore_errors=True)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def validate_staged_reference(
    subject_paths: SubjectPaths,
    *,
    verify_source: bool = True,
) -> StageReceipt:
    """Validate the staged tree, optionally re-reading the legacy source."""
    if verify_source:
        _validate_roots(subject_paths)
    receipt_path = subject_paths.reference_receipt
    destination = subject_paths.staged_m2m.resolve(strict=False)
    if not receipt_path.is_file() or not destination.is_dir():
        raise ReferenceStageError("Staged reference tree or receipt is missing")
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceStageError(f"Unreadable staging receipt: {receipt_path}") from exc
    expected_header = {
        "status": "complete",
        "subject": subject_paths.subject,
        "source_root": str(subject_paths.legacy_m2m.resolve()),
        "destination_root": str(destination),
    }
    for key, expected in expected_header.items():
        if payload.get(key) != expected:
            raise ReferenceStageError(
                f"Staging receipt field {key!r} is not bound to this subject/root"
            )
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ReferenceStageError("Staging receipt contains no file records")
    expected_destinations: set[str] = set()
    byte_count = 0
    for record in files:
        if not isinstance(record, dict):
            raise ReferenceStageError("Invalid staging file record")
        relative = str(record.get("destination_relative", ""))
        destination_file = _safe_destination(destination, relative)
        source_file = Path(str(record.get("source_path", "")))
        if verify_source:
            _assert_regular_source(source_file)
            if _sha256(source_file) != record.get("source_sha256"):
                raise ReferenceStageError(f"Legacy source hash changed: {source_file}")
        if not destination_file.is_file() or destination_file.is_symlink():
            raise ReferenceStageError(f"Staged file is missing or unsafe: {destination_file}")
        destination_hash = _sha256(destination_file)
        if destination_hash != record.get("destination_sha256"):
            raise ReferenceStageError(f"Staged file hash mismatch: {destination_file}")
        if destination_hash != record.get("source_sha256"):
            raise ReferenceStageError(f"Source/copy receipt mismatch: {destination_file}")
        expected_destinations.add(relative)
        byte_count += destination_file.stat().st_size
    actual_destinations = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    }
    if actual_destinations != expected_destinations:
        raise ReferenceStageError(
            "Staged tree differs from its exact allowlist receipt: "
            f"missing={sorted(expected_destinations - actual_destinations)}, "
            f"extra={sorted(actual_destinations - expected_destinations)}"
        )
    if int(payload.get("file_count", -1)) != len(files):
        raise ReferenceStageError("Staging receipt file count is inconsistent")
    if int(payload.get("byte_count", -1)) != byte_count:
        raise ReferenceStageError("Staging receipt byte count is inconsistent")
    return StageReceipt(
        subject=subject_paths.subject,
        source_root=expected_header["source_root"],
        destination_root=str(destination),
        receipt_path=str(receipt_path.resolve()),
        file_count=len(files),
        byte_count=byte_count,
    )


def validate_collected_reference(subject_paths: SubjectPaths) -> StageReceipt:
    """Validate one staged subject against the source-verified cohort receipt."""
    collection_path = (
        subject_paths.reference_receipt.parents[2]
        / "reference_stage_collection.json"
    )
    try:
        collection = json.loads(collection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceStageError(
            f"Reference-stage collection receipt is missing or unreadable: {collection_path}"
        ) from exc
    if (
        collection.get("status") != "complete"
        or collection.get("subject_count") != len(SUBJECTS)
        or collection.get("headmodel_root")
        != str(subject_paths.staged_anat.parents[1].resolve())
        or collection.get("legacy_reference_dependency_released") is not True
    ):
        raise ReferenceStageError("Reference-stage collection receipt is incomplete")
    entries = collection.get("subjects")
    if not isinstance(entries, list):
        raise ReferenceStageError("Reference-stage collection has no subject inventory")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("subject") == subject_paths.subject
    ]
    if len(matches) != 1:
        raise ReferenceStageError(
            f"Reference-stage collection does not contain {subject_paths.subject} exactly once"
        )
    entry = matches[0]
    receipt_path = subject_paths.reference_receipt.resolve()
    if entry.get("receipt_path") != str(receipt_path):
        raise ReferenceStageError("Collected reference names a different subject receipt")
    if not receipt_path.is_file() or entry.get("receipt_sha256") != _sha256(receipt_path):
        raise ReferenceStageError("Collected subject receipt hash mismatch")
    return validate_staged_reference(subject_paths, verify_source=False)


def collect_reference_stage(paths: CampaignPaths) -> dict[str, object]:
    """Validate all ten staged references before releasing the legacy dependency."""
    subjects: list[dict[str, object]] = []
    for subject in SUBJECTS:
        resolved = subject_paths(paths, subject)
        try:
            receipt = validate_staged_reference(resolved)
        except ReferenceStageError as exc:
            raise ReferenceStageError(
                f"Reference staging is incomplete for {subject}: {exc}"
            ) from exc
        subjects.append(
            {
                "subject": subject,
                "destination_root": receipt.destination_root,
                "receipt_path": receipt.receipt_path,
                "receipt_sha256": _sha256(Path(receipt.receipt_path)),
                "file_count": receipt.file_count,
                "byte_count": receipt.byte_count,
            }
        )
    binding = json.dumps(subjects, sort_keys=True, separators=(",", ":")).encode()
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "created_utc": _utc_now(),
        "git_commit": _git_commit(),
        "headmodel_root": str(paths.headmodel_root.resolve()),
        "subject_count": len(subjects),
        "subjects": subjects,
        "subjects_manifest_sha256": hashlib.sha256(binding).hexdigest(),
        "legacy_reference_dependency_released": True,
    }
    collection = paths.headmodel_root / "_pipeline" / "reference_stage_collection.json"
    _atomic_write_json(collection, payload)
    return payload


def _validate_roots(subject_paths: SubjectPaths) -> None:
    try:
        legacy_anat = subject_paths.legacy_anat.resolve(strict=True)
        source_root = subject_paths.legacy_m2m.resolve(strict=True)
        marker = subject_paths.legacy_ready_marker.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ReferenceStageError(
            f"Required legacy reference path is missing for {subject_paths.subject}: {exc.filename}"
        ) from exc
    if not _is_within(source_root, legacy_anat):
        raise ReferenceStageError(
            f"Legacy m2m path escapes its subject anatomy root: {source_root}"
        )
    if not _is_within(marker, legacy_anat):
        raise ReferenceStageError(f"Legacy marker path escapes anatomy root: {marker}")
    destination = subject_paths.staged_m2m.resolve(strict=False)
    staged_anat = subject_paths.staged_anat.resolve(strict=False)
    if not _is_within(destination, staged_anat):
        raise ReferenceStageError(
            f"Staged m2m path escapes its subject anatomy root: {destination}"
        )


def _enumerate_allowlist(subject_paths: SubjectPaths) -> list[_PlannedFile]:
    source_root = subject_paths.legacy_m2m.resolve(strict=True)
    planned: list[tuple[Path, str, str]] = []
    for name in ROOT_REFERENCE_FILES:
        source = source_root / name
        _assert_regular_source(source)
        planned.append((source, name, name))
    for tree_name in REFERENCE_TREES:
        tree = source_root / tree_name
        if tree.is_symlink() or not tree.is_dir():
            raise ReferenceStageError(f"Required reference directory is unsafe: {tree}")
        for root, directory_names, file_names in os.walk(tree, followlinks=False):
            root_path = Path(root)
            for directory_name in directory_names:
                directory = root_path / directory_name
                if directory.is_symlink():
                    raise ReferenceStageError(f"Symlink in reference allowlist: {directory}")
            for file_name in file_names:
                source = root_path / file_name
                _assert_regular_source(source)
                if source.suffix.lower() in FORBIDDEN_SUFFIXES:
                    raise ReferenceStageError(
                        f"Forbidden mesh/surface file inside reference tree: {source}"
                    )
                relative = source.relative_to(source_root).as_posix()
                planned.append((source, relative, relative))
    marker = subject_paths.legacy_ready_marker
    _assert_regular_source(marker)
    planned.append((marker, marker.name, PROVENANCE_MARKER_NAME))
    results = []
    for source, source_relative, destination_relative in sorted(
        planned, key=lambda item: item[2]
    ):
        resolved_source = source.resolve(strict=True)
        expected_parent = (
            subject_paths.legacy_anat.resolve(strict=True)
            if source == marker
            else source_root
        )
        if not _is_within(resolved_source, expected_parent):
            raise ReferenceStageError(f"Allowlisted source path escapes root: {source}")
        results.append(
            _PlannedFile(
                source=resolved_source,
                source_relative=source_relative,
                destination_relative=destination_relative,
                byte_count=resolved_source.stat().st_size,
                source_sha256=_sha256(resolved_source),
            )
        )
    return results


def _assert_regular_source(path: Path) -> None:
    if path.is_symlink():
        raise ReferenceStageError(f"Symlink in reference allowlist: {path}")
    try:
        mode = path.stat(follow_symlinks=False).st_mode
    except FileNotFoundError as exc:
        raise ReferenceStageError(f"Required reference file is missing: {path}") from exc
    if not stat.S_ISREG(mode):
        raise ReferenceStageError(f"Reference path is not a regular file: {path}")


def _safe_destination(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts or relative in {"", "."}:
        raise ReferenceStageError(f"Unsafe relative staging path: {relative!r}")
    destination = (root / relative_path).resolve(strict=False)
    if not _is_within(destination, root.resolve(strict=False)):
        raise ReferenceStageError(f"Staging path escapes destination root: {relative!r}")
    return destination


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return path != root
    except ValueError:
        return False


def _copy_file(source: Path, destination: Path) -> None:
    shutil.copy2(source, destination, follow_symlinks=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=".tmp-", suffix=".json", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw_temporary)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
