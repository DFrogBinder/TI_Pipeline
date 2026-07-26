#!/usr/bin/env python3
"""Audit or atomically stage the minimal NIfTI repeatability dataset."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_INPUTS = (
    ("t1", "_T1w.nii"),
    ("t2", "_T2w.nii"),
    ("segmentation", "_T1w_ras_1mm_T1andT2_masks.nii"),
)
SOURCE_LAYOUTS = ("legacy", "final132-scaffold")


@dataclass(frozen=True)
class StageInput:
    subject: str
    kind: str
    source_path: Path
    destination_name: str
    transform: str
    source_sha256: str
    expected_source_sha256: str
    provenance_record: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_subjects(path: Path) -> list[str]:
    subjects = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(subjects) != 10:
        raise ValueError(f"{path} contains {len(subjects)} subjects; expected 10.")
    if len(subjects) != len(set(subjects)):
        raise ValueError(f"{path} contains duplicate subjects.")
    invalid = [subject for subject in subjects if not subject.startswith("sub-CC")]
    if invalid:
        raise ValueError(f"Invalid subject IDs: {invalid}")
    return subjects


def normalize_scaffold_root(source_root: Path) -> Path:
    """Accept either the scaffold root or its ``subjects`` directory."""
    if source_root.name == "subjects" and (source_root.parent / "results").is_dir():
        return source_root.parent
    return source_root


def _load_result_record(path: Path, subject: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{subject}: invalid scaffold result {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{subject}: scaffold result is not a JSON object: {path}")
    if payload.get("status") != "complete":
        raise ValueError(
            f"{subject}: scaffold result status is {payload.get('status')!r}, expected 'complete': {path}"
        )
    if payload.get("subject") != subject:
        raise ValueError(
            f"{subject}: scaffold result subject is {payload.get('subject')!r}: {path}"
        )
    return payload


def _record_path(payload: dict[str, object], field: str, subject: str) -> Path:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{subject}: scaffold result is missing {field!r}")
    return Path(value).expanduser()


def _record_hash(payload: dict[str, object], field: str, subject: str) -> str:
    value = payload.get(field)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{subject}: scaffold result has invalid {field!r}")
    return value.lower()


def _require_subject_source(path: Path, subject_anat: Path, subject: str, kind: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(subject_anat.resolve(strict=True))
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ValueError(
            f"{subject}: {kind} source is missing or outside its canonical scaffold anat directory: {path}"
        ) from exc
    return resolved


def _legacy_specs(source_root: Path, subject: str) -> list[tuple[str, Path, str, str, str]]:
    anat_dir = source_root / subject / "anat"
    return [
        (kind, anat_dir / f"{subject}{suffix}", f"{subject}{suffix}", "copy", "")
        for kind, suffix in REQUIRED_INPUTS
    ]


def _scaffold_specs(
    source_root: Path, subject: str
) -> tuple[list[tuple[str, Path, str, str, str]], Path]:
    scaffold_root = normalize_scaffold_root(source_root)
    record_path = scaffold_root / "results" / f"{subject}.json"
    payload = _load_result_record(record_path, subject)
    subject_anat = scaffold_root / "subjects" / subject / "anat"
    fields = (
        ("t1", "task_t1_path", "source_t1_sha256", "_T1w.nii"),
        ("t2", "task_t2_path", "source_t2_sha256", "_T2w.nii"),
        (
            "segmentation",
            "installed_label",
            "installed_label_sha256",
            "_T1w_ras_1mm_T1andT2_masks.nii",
        ),
    )
    specs: list[tuple[str, Path, str, str, str]] = []
    for kind, path_field, hash_field, destination_suffix in fields:
        source_path = _require_subject_source(
            _record_path(payload, path_field, subject),
            subject_anat,
            subject,
            kind,
        )
        transform = "gunzip" if source_path.suffix == ".gz" else "copy"
        specs.append(
            (
                kind,
                source_path,
                f"{subject}{destination_suffix}",
                transform,
                _record_hash(payload, hash_field, subject),
            )
        )
    return specs, record_path.resolve()


def _verify_gzip(path: Path) -> None:
    with gzip.open(path, "rb") as handle:
        for _ in iter(lambda: handle.read(1024 * 1024), b""):
            pass


def audit(
    source_root: Path,
    subjects: list[str],
    source_layout: str = "legacy",
) -> tuple[list[StageInput], list[str]]:
    if source_layout not in SOURCE_LAYOUTS:
        raise ValueError(f"Unsupported source layout: {source_layout}")
    ready: list[StageInput] = []
    issues: list[str] = []
    for subject in subjects:
        try:
            if source_layout == "final132-scaffold":
                specs, record_path = _scaffold_specs(source_root, subject)
                provenance_record = str(record_path)
            else:
                specs = _legacy_specs(source_root, subject)
                provenance_record = ""
        except ValueError as exc:
            issues.append(str(exc))
            continue

        for kind, source_path, destination_name, transform, expected_hash in specs:
            if not source_path.is_file():
                issues.append(f"{subject}: missing {source_path}")
                continue
            if source_path.stat().st_size <= 0:
                issues.append(f"{subject}: empty {source_path}")
                continue
            try:
                source_hash = sha256_file(source_path)
                if expected_hash and source_hash != expected_hash:
                    issues.append(
                        f"{subject}: {kind} SHA-256 mismatch for {source_path}; "
                        f"expected {expected_hash}, observed {source_hash}"
                    )
                    continue
                if transform == "gunzip":
                    _verify_gzip(source_path)
            except (OSError, EOFError, gzip.BadGzipFile) as exc:
                issues.append(f"{subject}: unreadable {kind} source {source_path}: {exc}")
                continue
            ready.append(
                StageInput(
                    subject=subject,
                    kind=kind,
                    source_path=source_path,
                    destination_name=destination_name,
                    transform=transform,
                    source_sha256=source_hash,
                    expected_source_sha256=expected_hash,
                    provenance_record=provenance_record,
                )
            )
    return ready, issues


def _materialize(source: StageInput, destination: Path) -> None:
    if source.transform == "copy":
        shutil.copy2(source.source_path, destination)
        return
    if source.transform == "gunzip":
        with gzip.open(source.source_path, "rb") as source_handle:
            with destination.open("wb") as destination_handle:
                shutil.copyfileobj(source_handle, destination_handle, 1024 * 1024)
        return
    raise ValueError(f"Unsupported transform: {source.transform}")


def stage(
    source_root: Path,
    output_root: Path,
    subjects_file: Path,
    subjects: list[str],
    ready: list[StageInput],
    source_layout: str = "legacy",
) -> None:
    if output_root.exists():
        raise FileExistsError(
            f"Output root already exists; refusing to overwrite: {output_root}"
        )
    expected_count = len(subjects) * len(REQUIRED_INPUTS)
    if len(ready) != expected_count:
        raise ValueError(f"Refusing to stage {len(ready)} inputs; expected {expected_count}.")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.parent / f".{output_root.name}.staging-{uuid.uuid4().hex}"
    rows: list[dict[str, str | int]] = []
    try:
        for item in ready:
            destination = staging_root / item.subject / "anat" / item.destination_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            _materialize(item, destination)
            if destination.stat().st_size <= 0:
                raise OSError(f"Staged output is empty: {destination}")
            destination_hash = sha256_file(destination)
            if item.transform == "copy" and item.source_sha256 != destination_hash:
                raise OSError(f"Checksum mismatch after copying {item.source_path}")
            rows.append(
                {
                    "subject": item.subject,
                    "kind": item.kind,
                    "source": str(item.source_path),
                    "source_sha256": item.source_sha256,
                    "expected_source_sha256": item.expected_source_sha256,
                    "transform": item.transform,
                    "destination": str(destination.relative_to(staging_root)),
                    "size_bytes": destination.stat().st_size,
                    "sha256": destination_hash,
                    "provenance_record": item.provenance_record,
                }
            )
        shutil.copy2(subjects_file, staging_root / "subjects.txt")
        fieldnames = [
            "subject",
            "kind",
            "source",
            "source_sha256",
            "expected_source_sha256",
            "transform",
            "destination",
            "size_bytes",
            "sha256",
            "provenance_record",
        ]
        with (staging_root / "dataset_manifest.tsv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        metadata = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_layout": source_layout,
            "source_root": str(source_root),
            "subject_count": len(subjects),
            "required_file_count": len(rows),
            "source_hashes_verified": source_layout == "final132-scaffold",
            "subjects": subjects,
        }
        (staging_root / "dataset_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging_root, output_root)
    except BaseException:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source-layout",
        choices=SOURCE_LAYOUTS,
        default="legacy",
        help=(
            "Use 'final132-scaffold' for CamCan_Corrected_v4_Scaffolds; "
            "'legacy' expects the runner's three filenames directly."
        ),
    )
    parser.add_argument(
        "--subjects-file",
        type=Path,
        default=Path(__file__).with_name("subjects.txt"),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Stage the validated files. Without this flag, run a read-only audit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    if args.source_layout == "final132-scaffold":
        source_root = normalize_scaffold_root(source_root)
    output_root = args.output_root.resolve()
    subjects_file = args.subjects_file.resolve()
    if not source_root.is_dir():
        print(f"[ERROR] Source root is not a directory: {source_root}", file=sys.stderr)
        return 2
    subjects = read_subjects(subjects_file)
    ready, issues = audit(source_root, subjects, args.source_layout)
    ready_subjects = {
        subject
        for subject in subjects
        if sum(item.subject == subject for item in ready) == len(REQUIRED_INPUTS)
    }
    print(f"source_layout={args.source_layout}")
    print(f"subjects_requested={len(subjects)}")
    print(f"subjects_ready={len(ready_subjects)}")
    print(f"required_files_expected={len(subjects) * len(REQUIRED_INPUTS)}")
    print(f"required_files_ready={len(ready)}")
    print(
        "source_hashes_verified="
        f"{'yes' if args.source_layout == 'final132-scaffold' and not issues else 'no'}"
    )
    print(f"issues={len(issues)}")
    for issue in issues:
        print(f"[ERROR] {issue}", file=sys.stderr)
    if issues:
        return 2
    if not args.apply:
        print("[READY] Audit passed. Re-run with --apply to create the physical dataset.")
        return 0
    stage(
        source_root,
        output_root,
        subjects_file,
        subjects,
        ready,
        args.source_layout,
    )
    print(f"[OK] Staged dataset: {output_root}")
    print(f"[OK] Manifest: {output_root / 'dataset_manifest.tsv'}")
    print(f"[OK] Metadata: {output_root / 'dataset_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
