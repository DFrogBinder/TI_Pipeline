"""Read-only historical source inventory and canonical source-map validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

from .config import CANONICAL_SUBJECTS, CONDITIONS, TARGETS


SOURCE_GROUP_BY_SUBJECT = {
    subject: "replacement" if subject == "sub-CC510415" else "base9"
    for subject in CANONICAL_SUBJECTS
}


@dataclass(frozen=True)
class HistoricalRun:
    version: str
    subject: str
    target: str
    condition: str
    repeat: int
    pair1_msh: Path
    pair2_msh: Path
    ti_msh: Path
    completion_receipt: Path
    source_group: str

    @property
    def key(self) -> tuple[str, str, str, str, int]:
        return self.version, self.subject, self.target, self.condition, self.repeat


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_historical_sources(
    runs: Iterable[HistoricalRun],
    *,
    expected_per_version: int = 1600,
    expected_versions: tuple[str, ...] = ("3.2.6", "4.0.1"),
) -> dict[str, object]:
    """Hash a complete historical inventory without creating any output path."""

    records = tuple(runs)
    keys = [run.key for run in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate historical simulation key")
    counts = Counter(run.version for run in records)
    unexpected = sorted(set(counts).difference(expected_versions))
    if unexpected:
        raise ValueError(f"Unexpected historical versions: {unexpected}")
    for version in expected_versions:
        observed = counts.get(version, 0)
        if observed != expected_per_version:
            raise ValueError(
                f"Historical version {version} has {observed} runs; "
                f"expected {expected_per_version}"
            )
    files: list[dict[str, object]] = []
    for run in sorted(records, key=lambda item: item.key):
        if run.subject not in SOURCE_GROUP_BY_SUBJECT:
            raise ValueError(
                f"Historical inventory subject is outside the canonical 9+1 cohort: "
                f"{run.subject}"
            )
        expected_group = SOURCE_GROUP_BY_SUBJECT[run.subject]
        if run.source_group != expected_group:
            raise ValueError(
                f"Historical source_group for {run.subject} must be {expected_group}, "
                f"observed {run.source_group}"
            )
        if run.target not in TARGETS or run.condition not in CONDITIONS:
            raise ValueError(
                f"Historical run has noncanonical target or condition: {run.key}"
            )
        if not 1 <= run.repeat <= 40:
            raise ValueError(f"Historical run repeat is outside 1..40: {run.key}")
        for role, path in (
            ("pair1_msh", run.pair1_msh),
            ("pair2_msh", run.pair2_msh),
            ("ti_msh", run.ti_msh),
            ("completion_receipt", run.completion_receipt),
        ):
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(f"Missing or unsafe {role}: {path}")
            files.append(
                {
                    "version": run.version,
                    "subject": run.subject,
                    "target": run.target,
                    "condition": run.condition,
                    "repeat": run.repeat,
                    "source_group": run.source_group,
                    "role": role,
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema_version": 2,
        "status": "complete",
        "mode": "read_only_historical_inventory",
        "counts_by_version": {version: counts[version] for version in expected_versions},
        "files": files,
        "source_outputs_modified": False,
    }

