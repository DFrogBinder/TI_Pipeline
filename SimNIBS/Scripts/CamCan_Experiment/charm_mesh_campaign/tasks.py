"""Normalize segmentation inventories into immutable CHARM mesh tasks."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .config import CampaignConfig


IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
TASK_FIELDS = (
    "task_id",
    "subject_id",
    "realization_id",
    "task_key",
    "label_path",
    "label_sha256",
    "label_bytes",
    "mesh_path",
    "result_path",
    "config_sha256",
    "status",
    "message",
)


@dataclass(frozen=True)
class TaskRecord:
    task_id: int
    subject_id: str
    realization_id: str
    task_key: str
    label_path: Path
    label_sha256: str
    label_bytes: int
    mesh_path: Path
    result_path: Path
    config_sha256: str
    status: str
    message: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identifier(value: str, context: str) -> str:
    if not IDENTIFIER.fullmatch(value) or "/" in value or "\\" in value:
        raise ValueError(f"invalid {context}: {value!r}")
    return value


def task_key(subject_id: str, realization_id: str) -> str:
    return f"{_identifier(subject_id, 'subject_id')}__{_identifier(realization_id, 'realization_id')}"


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _render_output_path(
    output_root: Path, template: str, values: Mapping[str, object]
) -> Path:
    rendered = template.format(**values)
    candidate = (output_root / rendered).resolve()
    if not _within(candidate, output_root.resolve()):
        raise ValueError(f"rendered output is outside output root: {candidate}")
    return candidate


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_tasks(path: Path, tasks: list[TaskRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TASK_FIELDS, delimiter="\t")
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    "task_id": task.task_id,
                    "subject_id": task.subject_id,
                    "realization_id": task.realization_id,
                    "task_key": task.task_key,
                    "label_path": task.label_path,
                    "label_sha256": task.label_sha256,
                    "label_bytes": task.label_bytes,
                    "mesh_path": task.mesh_path,
                    "result_path": task.result_path,
                    "config_sha256": task.config_sha256,
                    "status": task.status,
                    "message": task.message,
                }
            )
    os.replace(temporary, path)


def read_task_manifest(path: str | Path) -> list[TaskRecord]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != TASK_FIELDS:
            raise ValueError(f"unexpected task manifest header: {path}")
        rows = list(reader)
    tasks: list[TaskRecord] = []
    for row in rows:
        tasks.append(
            TaskRecord(
                task_id=int(row["task_id"]),
                subject_id=row["subject_id"],
                realization_id=row["realization_id"],
                task_key=row["task_key"],
                label_path=Path(row["label_path"]),
                label_sha256=row["label_sha256"],
                label_bytes=int(row["label_bytes"]),
                mesh_path=Path(row["mesh_path"]),
                result_path=Path(row["result_path"]),
                config_sha256=row["config_sha256"],
                status=row["status"],
                message=row["message"],
            )
        )
    return tasks


def build_task_manifest(config: CampaignConfig) -> dict[str, object]:
    """Validate all source maps and write the normalized task manifest."""

    manifest_path = config.input.manifest.resolve(strict=True)
    delimiter = "\t" if config.input.delimiter == "tab" else ","
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        source_rows = list(reader)
        header = set(reader.fieldnames or ())
    columns = config.input.columns
    required_columns = {
        columns.subject_id,
        columns.label_path,
        columns.label_sha256,
        columns.label_bytes,
        columns.status,
    }
    if columns.realization_id is not None:
        required_columns.add(columns.realization_id)
    missing_columns = sorted(required_columns - header)
    if missing_columns:
        raise ValueError(
            "input manifest has missing configured column(s): "
            + ", ".join(missing_columns)
        )

    prepared: list[tuple[dict[str, str], str, str, str]] = []
    all_keys: set[str] = set()
    for row in source_rows:
        subject = _identifier(row[columns.subject_id].strip(), "subject_id")
        realization = (
            _identifier(row[columns.realization_id].strip(), "realization_id")
            if columns.realization_id is not None
            else "seg_001"
        )
        key = task_key(subject, realization)
        if key in all_keys:
            raise ValueError(f"duplicate task_key in input manifest: {key}")
        all_keys.add(key)
        prepared.append((row, subject, realization, key))

    exclusions = set(config.input.excluded_task_keys)
    absent_exclusions = sorted(exclusions - all_keys)
    if absent_exclusions:
        raise ValueError(
            "excluded task_key(s) absent from input manifest: "
            + ", ".join(absent_exclusions)
        )
    prepared = [item for item in prepared if item[3] not in exclusions]

    tasks: list[TaskRecord] = []
    mesh_outputs: set[Path] = set()
    result_outputs: set[Path] = set()
    count_matches = len(prepared) == config.execution.expected_tasks
    for index, (row, subject, realization, key) in enumerate(prepared):
        messages: list[str] = []
        status_value = row[columns.status].strip()
        if status_value != config.input.ready_status:
            messages.append(
                f"input status is {status_value!r}, expected {config.input.ready_status!r}"
            )
        raw_label = row[columns.label_path].strip()
        label = Path(raw_label)
        if not label.is_absolute():
            label = manifest_path.parent / label
        source_is_symlink = label.is_symlink()
        label = label.resolve()
        recorded_hash = row[columns.label_sha256].strip().lower()
        try:
            recorded_bytes = int(row[columns.label_bytes])
        except ValueError:
            recorded_bytes = 0
            messages.append(f"invalid label byte count: {row[columns.label_bytes]!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", recorded_hash):
            messages.append("label SHA-256 is missing or malformed")
        if source_is_symlink:
            messages.append(f"refusing symlinked label map: {label}")
        elif not label.is_file():
            messages.append(f"label map is missing: {label}")
        else:
            actual_bytes = label.stat().st_size
            if actual_bytes <= 0:
                messages.append("label map is empty")
            if actual_bytes != recorded_bytes:
                messages.append(
                    f"label byte count differs: {actual_bytes} != {recorded_bytes}"
                )
            actual_hash = _sha256(label)
            if actual_hash != recorded_hash:
                messages.append(
                    f"label SHA-256 differs: {actual_hash} != {recorded_hash}"
                )
        values = {
            "task_id": index,
            "subject_id": subject,
            "realization_id": realization,
            "task_key": key,
        }
        mesh_path = _render_output_path(
            config.output.root, config.output.mesh_template, values
        )
        result_path = _render_output_path(
            config.output.root, config.output.result_template, values
        )
        if mesh_path in mesh_outputs:
            raise ValueError(f"duplicate mesh output path: {mesh_path}")
        if result_path in result_outputs:
            raise ValueError(f"duplicate result output path: {result_path}")
        mesh_outputs.add(mesh_path)
        result_outputs.add(result_path)
        if not count_matches:
            messages.append(
                f"task count is {len(prepared)}, expected {config.execution.expected_tasks}"
            )
        tasks.append(
            TaskRecord(
                task_id=index,
                subject_id=subject,
                realization_id=realization,
                task_key=key,
                label_path=label,
                label_sha256=recorded_hash,
                label_bytes=recorded_bytes,
                mesh_path=mesh_path,
                result_path=result_path,
                config_sha256=config.sha256,
                status="ready" if not messages else "blocked",
                message="ready" if not messages else "; ".join(messages),
            )
        )

    ready = sum(task.status == "ready" for task in tasks)
    _write_tasks(config.output.task_manifest, tasks)
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": (
            "ready"
            if ready == config.execution.expected_tasks == len(tasks)
            else "blocked"
        ),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "campaign_id": config.campaign_id,
        "config_path": str(config.source_path),
        "config_sha256": config.sha256,
        "input_manifest": str(manifest_path),
        "input_rows": len(source_rows),
        "excluded_task_keys": sorted(exclusions),
        "tasks": len(tasks),
        "tasks_expected": config.execution.expected_tasks,
        "ready": ready,
        "blocked": len(tasks) - ready,
        "task_manifest": str(config.output.task_manifest),
        "output_root": str(config.output.root),
        "segmentation_rerun": False,
        "surfaces_requested": False,
        "simulation_requested": False,
    }
    _atomic_text(
        config.output.preflight_summary,
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    return summary
