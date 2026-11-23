"""Immutable checksummed package builder for v2 handoff artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from typing import Iterable, Mapping


REQUIRED_CATEGORIES = ("simulation", "metrics", "figures", "manuscript")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_checksummed_package(
    *, package_root: Path, artifacts: Mapping[str, Iterable[Path]]
) -> dict[str, object]:
    if set(artifacts) != set(REQUIRED_CATEGORIES):
        raise ValueError(
            f"Package categories must be exactly {sorted(REQUIRED_CATEGORIES)}"
        )
    archive = package_root.with_name(f"{package_root.name}.tar.gz")
    if package_root.exists() or package_root.is_symlink() or archive.exists():
        raise ValueError("Versioned package destination must be new")
    materialized = {
        category: tuple(artifacts[category]) for category in REQUIRED_CATEGORIES
    }
    for category, paths in materialized.items():
        if not paths:
            raise ValueError(f"Package category {category} is empty")
        names = [source.name for source in paths]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate artifact name in {category}")
        for source in paths:
            if not source.is_file() or source.is_symlink():
                raise FileNotFoundError(f"Missing or unsafe package artifact: {source}")
    package_root.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for category in REQUIRED_CATEGORIES:
        for source in materialized[category]:
            destination = package_root / category / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            records.append(
                {
                    "category": category,
                    "source_path": str(source.resolve()),
                    "package_path": str(destination.relative_to(package_root)),
                    "size_bytes": destination.stat().st_size,
                    "sha256": _sha256(destination),
                }
            )
    manifest = {
        "schema_version": 2,
        "status": "validated",
        "categories": list(REQUIRED_CATEGORIES),
        "artifacts": records,
        "historical_sources_included": False,
        "deletion_performed": False,
    }
    manifest_path = package_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_records = [
        *records,
        {
            "package_path": "manifest.json",
            "sha256": _sha256(manifest_path),
        },
    ]
    checksum_path = package_root / "SHA256SUMS"
    checksum_path.write_text(
        "".join(
            f"{record['sha256']}  {record['package_path']}\n"
            for record in checksum_records
        ),
        encoding="utf-8",
    )
    with tarfile.open(archive, "w:gz") as handle:
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                handle.add(path, arcname=str(path.relative_to(package_root)))
    archive_sha256 = _sha256(archive)
    for path in package_root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    archive.chmod(0o444)
    return {
        **manifest,
        "package_root": str(package_root.resolve()),
        "archive": str(archive.resolve()),
        "archive_sha256": archive_sha256,
        "manifest_sha256": _sha256(manifest_path),
        "checksums_sha256": _sha256(checksum_path),
    }


def verify_checksummed_package(
    *, package_root: Path, archive: Path
) -> dict[str, object]:
    """Independently validate copied files, declared checksums, and archive bytes."""

    for path, label in ((package_root, "package root"), (archive, "archive")):
        if path.is_symlink():
            raise ValueError(f"Refusing symbolic-link {label}: {path}")
    if not package_root.is_dir() or not archive.is_file():
        raise FileNotFoundError("Package root and archive must both exist")
    manifest_path = package_root / "manifest.json"
    checksum_path = package_root / "SHA256SUMS"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "validated":
        raise ValueError("Package manifest is not validated")
    declared: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise ValueError(f"Malformed SHA256SUMS line: {line!r}") from error
        if relative in declared or len(digest) != 64:
            raise ValueError("SHA256SUMS contains a duplicate path or invalid digest")
        declared[relative] = digest
    expected_declared = {
        str(record["package_path"]) for record in manifest["artifacts"]
    } | {"manifest.json"}
    if set(declared) != expected_declared:
        raise ValueError("SHA256SUMS paths disagree with the package manifest")
    for relative, expected in declared.items():
        path = package_root / relative
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected:
            raise ValueError(f"Package checksum validation failed: {relative}")
    expected_archive = expected_declared | {"SHA256SUMS"}
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        if any(not member.isfile() for member in members):
            raise ValueError("Package archive may contain regular files only")
        if len({member.name for member in members}) != len(members):
            raise ValueError("Package archive contains duplicate members")
        if {member.name for member in members} != expected_archive:
            raise ValueError("Package archive contents disagree with the package root")
        for member in members:
            extracted = handle.extractfile(member)
            if extracted is None:
                raise ValueError(f"Cannot read package archive member: {member.name}")
            archive_digest = hashlib.sha256(extracted.read()).hexdigest()
            root_digest = _sha256(package_root / member.name)
            if archive_digest != root_digest:
                raise ValueError(f"Archive member differs from package root: {member.name}")
    return {
        "schema_version": 2,
        "status": "independently_validated",
        "artifact_count": len(manifest["artifacts"]),
        "archive_sha256": _sha256(archive),
        "manifest_sha256": _sha256(manifest_path),
        "checksums_sha256": _sha256(checksum_path),
    }

