#!/usr/bin/env python3
"""Extract and collect repeatability mesh-element metrics without rerunning FEM.

The extractor reads the retained ``TI.msh`` files produced by a completed
repeatability experiment.  It writes to an isolated output directory and never
modifies the simulations or the existing repeatability analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PIPELINE_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from experiment_config import (  # noqa: E402
    load_experiment_config,
    repeat_tag,
    subject_condition_repeats_root,
)


SCHEMA_VERSION = 1
ROW_FIELDS = [
    "schema_version",
    "subject",
    "condition",
    "repeat_tag",
    "ti_msh_path",
    "ti_msh_size_bytes",
    "mesh_nodes",
    "mesh_elements",
    "mesh_elements_by_tissue",
    "mesh_volume_mm3_by_tissue",
    "mesh_total_volume_mm3",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv_atomic(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _completion_scope(config) -> dict[str, object]:
    receipt_path = config.experiment_root / "_pipeline" / "workflow" / "complete.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"Missing completion receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "complete":
        raise RuntimeError(f"Experiment is not complete: {receipt_path}")
    scope = receipt.get("scope")
    if not isinstance(scope, dict):
        raise RuntimeError(f"Completion receipt has no scope: {receipt_path}")

    expected_subjects = len(config.subjects)
    expected_repeats = {
        condition.name: int(condition.repeat_count)
        for condition in config.conditions
    }
    expected_rows = expected_subjects * sum(expected_repeats.values())
    if int(scope.get("subject_count", -1)) != expected_subjects:
        raise RuntimeError(
            "Completion receipt subject count does not match the experiment config"
        )
    if len(set(expected_repeats.values())) != 1:
        raise RuntimeError("Conditions do not share one repeat count")
    repeat_count = next(iter(expected_repeats.values()))
    if int(scope.get("repeats_per_condition", -1)) != repeat_count:
        raise RuntimeError(
            "Completion receipt repeat count does not match the experiment config"
        )
    if int(scope.get("expected_ti_msh", -1)) != expected_rows:
        raise RuntimeError(
            "Completion receipt expected TI.msh count does not match the config"
        )
    return {
        "receipt": str(receipt_path),
        "receipt_sha256": _sha256_file(receipt_path),
        "subject_count": expected_subjects,
        "conditions": expected_repeats,
        "expected_meshes": expected_rows,
        "roi": scope.get("roi"),
    }


def preflight(*, config_path: Path, output_root: Path) -> dict[str, object]:
    config = load_experiment_config(config_path, validate_paths=False)
    scope = _completion_scope(config)
    if not config.experiment_root.is_dir():
        raise FileNotFoundError(config.experiment_root)
    return {
        "status": "ready",
        "mode": "read_only_mesh_metric_extraction",
        "config": str(config.config_path),
        "config_sha256": _sha256_file(config.config_path),
        "experiment_root": str(config.experiment_root),
        "output_root": str(output_root),
        "source_outputs_modified": False,
        "meshing_tasks": 0,
        "fem_tasks": 0,
        **scope,
    }


def _ti_msh_path(
    *,
    config,
    subject: str,
    condition: str,
    repeat_number: int,
) -> Path:
    tag = repeat_tag(repeat_number)
    return (
        subject_condition_repeats_root(config, subject, condition)
        / tag
        / subject
        / "anat"
        / "SimNIBS"
        / "Output"
        / subject
        / "TI.msh"
    )


def extract_subject(
    *,
    config_path: Path,
    subject_index: int,
    output_root: Path,
) -> dict[str, object]:
    config = load_experiment_config(config_path, validate_paths=False)
    scope = _completion_scope(config)
    if subject_index < 0 or subject_index >= len(config.subjects):
        raise IndexError(
            f"subject index {subject_index} outside 0..{len(config.subjects) - 1}"
        )
    subject = config.subjects[subject_index]

    # Import only inside the compute stage.  Stanage login-node preflight uses
    # base Python, whereas array tasks load the SimNIBS scientific environment.
    from post.mesh_repeat_report import _mesh_statistics

    rows: list[dict[str, object]] = []
    for condition in config.conditions:
        for repeat_number in range(1, int(condition.repeat_count) + 1):
            tag = repeat_tag(repeat_number)
            ti_msh = _ti_msh_path(
                config=config,
                subject=subject,
                condition=condition.name,
                repeat_number=repeat_number,
            )
            if not ti_msh.is_file():
                raise FileNotFoundError(ti_msh)
            (
                mesh_nodes,
                mesh_elements,
                elements_by_tissue,
                volume_by_tissue,
            ) = _mesh_statistics(ti_msh)
            if (
                not math.isfinite(mesh_nodes)
                or not math.isfinite(mesh_elements)
                or mesh_nodes <= 0
                or mesh_elements <= 0
                or not elements_by_tissue
                or not volume_by_tissue
            ):
                raise RuntimeError(f"Incomplete mesh statistics: {ti_msh}")
            element_count = int(round(mesh_elements))
            normalized_counts = {
                int(key): int(value)
                for key, value in elements_by_tissue.items()
            }
            normalized_volumes = {
                int(key): float(value)
                for key, value in volume_by_tissue.items()
            }
            if sum(normalized_counts.values()) != element_count:
                raise RuntimeError(
                    f"Tissue element counts do not sum to total: {ti_msh}"
                )
            if any(value < 0 for value in normalized_volumes.values()):
                raise RuntimeError(f"Negative tissue volume: {ti_msh}")
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "subject": subject,
                    "condition": condition.name,
                    "repeat_tag": tag,
                    "ti_msh_path": str(ti_msh),
                    "ti_msh_size_bytes": ti_msh.stat().st_size,
                    "mesh_nodes": int(round(mesh_nodes)),
                    "mesh_elements": element_count,
                    "mesh_elements_by_tissue": json.dumps(
                        normalized_counts,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "mesh_volume_mm3_by_tissue": json.dumps(
                        normalized_volumes,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "mesh_total_volume_mm3": sum(
                        normalized_volumes.values()
                    ),
                }
            )

    expected_rows = sum(int(item.repeat_count) for item in config.conditions)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"{subject}: expected {expected_rows} rows, extracted {len(rows)}"
        )
    subject_csv = output_root / "subjects" / f"{subject}.csv"
    _write_csv_atomic(subject_csv, rows, ROW_FIELDS)
    receipt = {
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "config": str(config.config_path),
        "config_sha256": _sha256_file(config.config_path),
        "source_completion_receipt": scope["receipt"],
        "source_completion_receipt_sha256": scope["receipt_sha256"],
        "subject_index": subject_index,
        "subject": subject,
        "rows": len(rows),
        "conditions": {
            condition.name: int(condition.repeat_count)
            for condition in config.conditions
        },
        "subject_csv": str(subject_csv),
        "subject_csv_sha256": _sha256_file(subject_csv),
        "source_outputs_modified": False,
    }
    receipt_path = output_root / "subjects" / f"{subject}.json"
    _write_json_atomic(receipt_path, receipt)
    return receipt


def _validate_collected_rows(
    *,
    rows: list[dict[str, str]],
    config,
) -> dict[str, object]:
    expected_by_condition = {
        condition.name: int(condition.repeat_count)
        for condition in config.conditions
    }
    expected_total = len(config.subjects) * sum(expected_by_condition.values())
    if len(rows) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} mesh rows, collected {len(rows)}"
        )
    keys = [
        (row["subject"], row["condition"], row["repeat_tag"])
        for row in rows
    ]
    if len(set(keys)) != len(keys):
        raise RuntimeError("Duplicate subject/condition/repeat mesh records")

    fixed_mesh_unique_counts: dict[str, int] = {}
    for subject in config.subjects:
        for condition, expected in expected_by_condition.items():
            selected = [
                row
                for row in rows
                if row["subject"] == subject
                and row["condition"] == condition
            ]
            if len(selected) != expected:
                raise RuntimeError(
                    f"{subject}/{condition}: expected {expected}, got {len(selected)}"
                )
            repeat_tags = {row["repeat_tag"] for row in selected}
            expected_tags = {
                repeat_tag(index)
                for index in range(1, expected + 1)
            }
            if repeat_tags != expected_tags:
                raise RuntimeError(
                    f"{subject}/{condition}: repeat tag mismatch"
                )
            for row in selected:
                if int(row["schema_version"]) != SCHEMA_VERSION:
                    raise RuntimeError("Mesh metric schema mismatch")
                element_count = int(row["mesh_elements"])
                counts = json.loads(row["mesh_elements_by_tissue"])
                volumes = json.loads(row["mesh_volume_mm3_by_tissue"])
                if element_count <= 0:
                    raise RuntimeError("Non-positive tetrahedral element count")
                if sum(int(value) for value in counts.values()) != element_count:
                    raise RuntimeError("Tissue counts do not sum to total")
                if not volumes or any(
                    float(value) < 0 for value in volumes.values()
                ):
                    raise RuntimeError("Invalid tissue-volume mapping")
            if condition == "fixed_mesh":
                fixed_mesh_unique_counts[subject] = len(
                    {int(row["mesh_elements"]) for row in selected}
                )
    return {
        "rows": len(rows),
        "expected_rows": expected_total,
        "fixed_mesh_unique_element_counts_by_subject": (
            fixed_mesh_unique_counts
        ),
        "fixed_mesh_constant_within_subject": all(
            value == 1 for value in fixed_mesh_unique_counts.values()
        ),
    }


def collect(
    *,
    config_path: Path,
    output_root: Path,
    archive_path: Path | None,
) -> dict[str, object]:
    config = load_experiment_config(config_path, validate_paths=False)
    scope = _completion_scope(config)
    rows: list[dict[str, str]] = []
    subject_files: list[Path] = []
    receipt_files: list[Path] = []
    for subject in config.subjects:
        subject_csv = output_root / "subjects" / f"{subject}.csv"
        receipt_path = output_root / "subjects" / f"{subject}.json"
        if not subject_csv.is_file() or not receipt_path.is_file():
            raise FileNotFoundError(
                f"Missing subject mesh metrics for {subject}: "
                f"{subject_csv} / {receipt_path}"
            )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "complete":
            raise RuntimeError(f"Incomplete subject receipt: {receipt_path}")
        if receipt.get("subject_csv_sha256") != _sha256_file(subject_csv):
            raise RuntimeError(f"Subject CSV checksum mismatch: {subject_csv}")
        rows.extend(_read_csv(subject_csv))
        subject_files.append(subject_csv)
        receipt_files.append(receipt_path)

    rows.sort(
        key=lambda row: (
            config.subjects.index(row["subject"]),
            [item.name for item in config.conditions].index(row["condition"]),
            int(row["repeat_tag"].rsplit("_", 1)[-1]),
        )
    )
    validation = _validate_collected_rows(rows=rows, config=config)
    combined_csv = output_root / "mesh_metrics.csv"
    _write_csv_atomic(combined_csv, rows, ROW_FIELDS)

    manifest = {
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "mode": "read_only_mesh_metric_extraction",
        "config": str(config.config_path),
        "config_sha256": _sha256_file(config.config_path),
        "experiment_root": str(config.experiment_root),
        "output_root": str(output_root),
        "roi": scope["roi"],
        "subjects": list(config.subjects),
        "subject_count": len(config.subjects),
        "conditions": scope["conditions"],
        "expected_meshes": scope["expected_meshes"],
        "source_completion_receipt": scope["receipt"],
        "source_completion_receipt_sha256": scope["receipt_sha256"],
        "source_outputs_modified": False,
        "meshing_tasks": 0,
        "fem_tasks": 0,
        "combined_csv": str(combined_csv),
        "combined_csv_sha256": _sha256_file(combined_csv),
        "validation": validation,
    }
    manifest_path = output_root / "manifest.json"
    _write_json_atomic(manifest_path, manifest)

    checksum_paths = [
        combined_csv,
        manifest_path,
        *subject_files,
        *receipt_files,
    ]
    checksum_path = output_root / "checksums.sha256"
    checksum_lines = [
        f"{_sha256_file(path)}  {path.relative_to(output_root)}"
        for path in checksum_paths
    ]
    checksum_path.write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )

    if archive_path is not None:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive_path.with_name(
            f".{archive_path.name}.tmp-{os.getpid()}"
        )
        with tarfile.open(temporary, "w:gz") as archive:
            archive.add(combined_csv, arcname=combined_csv.name)
            archive.add(manifest_path, arcname=manifest_path.name)
            archive.add(checksum_path, arcname=checksum_path.name)
            for path in [*subject_files, *receipt_files]:
                archive.add(
                    path,
                    arcname=str(path.relative_to(output_root)),
                )
        temporary.replace(archive_path)
        archive_sha256 = _sha256_file(archive_path)
        archive_path.with_suffix(archive_path.suffix + ".sha256").write_text(
            f"{archive_sha256}  {archive_path}\n",
            encoding="utf-8",
        )
        manifest["archive"] = str(archive_path)
        manifest["archive_sha256"] = archive_sha256
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("preflight", "extract-subject", "collect"):
        child = subparsers.add_parser(name)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--output-root", type=Path, required=True)
        if name == "extract-subject":
            child.add_argument("--subject-index", type=int, required=True)
        if name == "collect":
            child.add_argument("--archive", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if args.command == "preflight":
        payload = preflight(
            config_path=config_path,
            output_root=output_root,
        )
    elif args.command == "extract-subject":
        payload = extract_subject(
            config_path=config_path,
            subject_index=args.subject_index,
            output_root=output_root,
        )
    else:
        payload = collect(
            config_path=config_path,
            output_root=output_root,
            archive_path=(
                args.archive.expanduser().resolve()
                if args.archive
                else None
            ),
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
