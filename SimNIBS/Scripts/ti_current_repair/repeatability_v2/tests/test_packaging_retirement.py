from __future__ import annotations

import json
from pathlib import Path
import tarfile

import pytest

from ti_current_repair.repeatability_v2.packaging import (
    build_checksummed_package,
    verify_checksummed_package,
)
from ti_current_repair.repeatability_v2.retirement import (
    RetirementCandidate,
    build_retirement_candidate_manifest,
    retirement_approval_gate,
)


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def test_release_package_contains_all_categories_and_independent_checksums(
    tmp_path: Path,
) -> None:
    artifacts = {
        "simulation": [_write(tmp_path / "artifacts" / "simulation.json", "sim")],
        "metrics": [_write(tmp_path / "artifacts" / "metrics.csv", "metric")],
        "figures": [_write(tmp_path / "artifacts" / "figure.png", "figure")],
        "manuscript": [_write(tmp_path / "artifacts" / "notes.md", "notes")],
    }
    result = build_checksummed_package(
        package_root=tmp_path / "package-v2",
        artifacts=artifacts,
    )

    assert result["status"] == "validated"
    assert set(result["categories"]) == set(artifacts)
    archive = Path(result["archive"])
    assert archive.is_file()
    assert len(result["archive_sha256"]) == 64
    assert archive.stat().st_mode & 0o222 == 0
    with tarfile.open(archive, "r:gz") as handle:
        names = set(handle.getnames())
    assert "manifest.json" in names
    assert "SHA256SUMS" in names
    verification = verify_checksummed_package(
        package_root=Path(result["package_root"]), archive=archive
    )
    assert verification["status"] == "independently_validated"
    assert verification["archive_sha256"] == result["archive_sha256"]


def test_retirement_gate_never_deletes_and_requires_user_and_supervisor(tmp_path: Path) -> None:
    v4 = _write(tmp_path / "historical" / "v401" / "TI.msh", "v4")
    v3 = _write(tmp_path / "historical" / "v326-fixed" / "TI.msh", "v3")
    manifest_path = tmp_path / "retirement-candidates.json"
    manifest = build_retirement_candidate_manifest(
        candidates=[
            RetirementCandidate(v4.parent, "obsolete_simnibs401"),
            RetirementCandidate(v3.parent, "superseded_v3_fixed"),
        ],
        replacement_package_sha256="a" * 64,
        output=manifest_path,
    )
    assert manifest["automatic_deletion"] is False
    assert v4.is_file() and v3.is_file()

    with pytest.raises(ValueError, match="user and supervisory"):
        retirement_approval_gate(
            manifest=manifest,
            user_approval=True,
            supervisor_approval=False,
        )
    eligible = retirement_approval_gate(
        manifest=manifest,
        user_approval=True,
        supervisor_approval=True,
    )
    assert eligible["status"] == "eligible_for_separate_manual_retirement"
    assert eligible["deletion_performed"] is False
    assert v4.is_file() and v3.is_file()


def test_retirement_manifest_rejects_non_sha256_replacement_identity(
    tmp_path: Path,
) -> None:
    candidate = _write(tmp_path / "historical" / "TI.msh", "source").parent
    with pytest.raises(ValueError, match="SHA-256"):
        build_retirement_candidate_manifest(
            candidates=[RetirementCandidate(candidate, "obsolete_simnibs401")],
            replacement_package_sha256="not-a-digest".ljust(64, "x"),
            output=tmp_path / "retirement.json",
        )
