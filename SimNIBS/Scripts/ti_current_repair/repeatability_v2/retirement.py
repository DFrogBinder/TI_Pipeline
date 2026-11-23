"""No-delete retirement candidate inventory and dual-approval gate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Mapping


@dataclass(frozen=True)
class RetirementCandidate:
    path: Path
    reason: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_retirement_candidate_manifest(
    *,
    candidates: Iterable[RetirementCandidate],
    replacement_package_sha256: str,
    output: Path,
) -> dict[str, object]:
    materialized = tuple(candidates)
    if not materialized:
        raise ValueError("Retirement candidate manifest cannot be empty")
    if re.fullmatch(r"[0-9a-f]{64}", replacement_package_sha256) is None:
        raise ValueError("Replacement package SHA-256 is invalid")
    if output.exists() or output.is_symlink():
        raise ValueError("Retirement candidate manifest destination must be new")
    records = []
    for candidate in materialized:
        root = candidate.path
        if not root.is_dir() or root.is_symlink():
            raise FileNotFoundError(f"Missing or unsafe retirement candidate: {root}")
        files = []
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Retirement inventory refuses symbolic link: {path}")
            if path.is_file():
                files.append(
                    {
                        "relative_path": str(path.relative_to(root)),
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
        if not files:
            raise ValueError(f"Retirement candidate is empty: {root}")
        records.append(
            {
                "path": str(root.resolve()),
                "reason": candidate.reason,
                "file_count": len(files),
                "files": files,
            }
        )
    manifest = {
        "schema_version": 2,
        "status": "candidates_only",
        "replacement_package_sha256": replacement_package_sha256,
        "candidates": records,
        "automatic_deletion": False,
        "deletion_performed": False,
        "required_approvals": ["user", "supervisor"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def retirement_approval_gate(
    *,
    manifest: Mapping[str, object],
    user_approval: bool,
    supervisor_approval: bool,
) -> dict[str, object]:
    if manifest.get("status") != "candidates_only" or manifest.get(
        "automatic_deletion"
    ) is not False:
        raise ValueError("Retirement manifest is not a no-delete candidate manifest")
    if not (user_approval and supervisor_approval):
        raise ValueError("Retirement requires explicit user and supervisory approval")
    return {
        "status": "eligible_for_separate_manual_retirement",
        "user_approval": True,
        "supervisor_approval": True,
        "candidate_count": len(manifest["candidates"]),
        "deletion_performed": False,
        "next_action": "run_a_separately_reviewed_manual_retirement_workflow",
    }
