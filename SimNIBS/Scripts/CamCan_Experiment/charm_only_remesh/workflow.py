#!/usr/bin/env python3
"""Audit/remove ROAST maps, remesh, and validate CHARM-only CamCan datasets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.mesh_reuse import candidate_mesh_paths, resolve_existing_mesh  # noqa: E402
from utils.camcan_dataset import (  # noqa: E402
    CAMCAN_ROI_CONFIGS,
    parse_dataset_name,
    sha256_file,
)


TARGET_BASENAME = "tissue_labeling_upsampled.nii.gz"
READY_INSTALL_STATUSES = {"installed", "already_current"}
REMESH_FIELDNAMES = (
    "task_id",
    "dataset_name",
    "roi_prefix",
    "repeat_id",
    "dataset_root",
    "subject",
    "anat_dir",
    "m2m_dir",
    "label_path",
    "source_label_path",
    "expected_label_sha256",
    "mesh_path",
    "status",
    "message",
)
ROAST_REMOVAL_FIELDNAMES = (
    "dataset_name",
    "subject",
    "path",
    "sha256",
    "bytes",
    "action",
    "message",
)
RESULT_SCHEMA_VERSION = 1


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, object]],
) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: str(row.get(field, "")).replace("\t", " ").replace("\n", " ")
                    for field in fieldnames
                }
            )
    return destination


def write_json_atomic(path: str | Path, payload: object) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def roast_segmentation_candidates(anat_dir: Path, subject: str) -> tuple[Path, Path]:
    stem = f"{subject}_T1w_ras_1mm_T1andT2_masks"
    return anat_dir / f"{stem}.nii", anat_dir / f"{stem}.nii.gz"


def _validate_install_manifest_header(rows: list[dict[str, str]]) -> None:
    required = {
        "dataset",
        "subject",
        "source",
        "destination",
        "installed_sha256",
        "status",
    }
    if not rows:
        raise ValueError("Installation manifest contains no task rows.")
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(
            "Installation manifest is missing required column(s): " + ", ".join(missing)
        )


def _dataset_root_from_label(label_path: Path, dataset_name: str) -> Path:
    for parent in label_path.parents:
        if parent.name == dataset_name:
            return parent
    raise ValueError(
        f"Label path {label_path} is not contained in its dataset directory {dataset_name}."
    )


def select_install_rows(
    install_rows: list[dict[str, str]],
    *,
    roi_prefix: str | None,
    expected_targets: int,
    expected_repeats: int = 10,
    expected_subjects: int = 175,
) -> list[dict[str, str]]:
    """Select and validate an exact ROI scope from the full install manifest."""
    if roi_prefix is None:
        return install_rows

    allowed = {config.dataset_prefix for config in CAMCAN_ROI_CONFIGS}
    if roi_prefix not in allowed:
        raise ValueError(
            f"unsupported ROI prefix {roi_prefix!r}; expected one of: "
            + ", ".join(sorted(allowed))
        )
    if expected_repeats <= 0 or expected_subjects <= 0:
        raise ValueError("expected repeats and subjects must be positive")

    selected: list[dict[str, str]] = []
    subjects_by_dataset: dict[str, set[str]] = {}
    row_counts: dict[str, int] = {}
    for row in install_rows:
        dataset_name = row["dataset"].strip()
        config, _ = parse_dataset_name(dataset_name)
        if config.dataset_prefix != roi_prefix:
            continue
        selected.append(row)
        subject = row["subject"].strip()
        subjects_by_dataset.setdefault(dataset_name, set()).add(subject)
        row_counts[dataset_name] = row_counts.get(dataset_name, 0) + 1

    expected_datasets = {
        f"{roi_prefix}_Data_{repeat:02d}"
        for repeat in range(1, expected_repeats + 1)
    }
    actual_datasets = set(subjects_by_dataset)
    if actual_datasets != expected_datasets:
        missing = sorted(expected_datasets - actual_datasets)
        extra = sorted(actual_datasets - expected_datasets)
        raise ValueError(
            f"ROI dataset scope mismatch for {roi_prefix}: "
            f"missing={missing}, extra={extra}"
        )
    for dataset_name in sorted(expected_datasets):
        unique_subjects = len(subjects_by_dataset[dataset_name])
        rows = row_counts[dataset_name]
        if unique_subjects != expected_subjects or rows != expected_subjects:
            raise ValueError(
                f"dataset {dataset_name} has rows={rows}, "
                f"unique_subjects={unique_subjects}; expected {expected_subjects}"
            )
    reference_dataset = min(expected_datasets)
    reference_subjects = subjects_by_dataset[reference_dataset]
    for dataset_name in sorted(expected_datasets - {reference_dataset}):
        subjects = subjects_by_dataset[dataset_name]
        if subjects != reference_subjects:
            raise ValueError(
                f"dataset {dataset_name} has a different subject set from "
                f"{reference_dataset}: missing={sorted(reference_subjects - subjects)}, "
                f"extra={sorted(subjects - reference_subjects)}"
            )
    if len(selected) != expected_targets:
        raise ValueError(
            f"ROI scope {roi_prefix} has {len(selected)} rows; "
            f"expected {expected_targets}"
        )
    return selected


def build_remesh_manifest(
    *,
    install_manifest: str | Path,
    roi_root: str | Path,
    manifest: str | Path,
    summary: str | Path,
    expected_targets: int = 7000,
    roi_prefix: str | None = None,
    expected_repeats: int = 10,
    expected_subjects: int = 175,
) -> dict[str, object]:
    roi_root_path = Path(roi_root).expanduser().resolve()
    install_rows = read_tsv(install_manifest)
    _validate_install_manifest_header(install_rows)
    install_rows = select_install_rows(
        install_rows,
        roi_prefix=roi_prefix,
        expected_targets=expected_targets,
        expected_repeats=expected_repeats,
        expected_subjects=expected_subjects,
    )
    selected_roi_prefix = roi_prefix
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()

    for install_row in install_rows:
        dataset_name = install_row["dataset"].strip()
        subject = install_row["subject"].strip()
        messages: list[str] = []
        row_roi_prefix = ""
        repeat_id = ""
        dataset_root: Path | None = None
        anat_dir: Path | None = None
        m2m_dir: Path | None = None
        label_path = Path(install_row["destination"]).expanduser()
        source_label_path = Path(install_row.get("source", "")).expanduser()
        expected_hash = install_row["installed_sha256"].strip()
        mesh_path: Path | str = ""

        key = (dataset_name, subject)
        if key in seen:
            messages.append("duplicate dataset/subject row in installation manifest")
        seen.add(key)

        try:
            config, repeat_id = parse_dataset_name(dataset_name)
            row_roi_prefix = config.dataset_prefix
        except ValueError as exc:
            messages.append(str(exc))

        if install_row["status"].strip() not in READY_INSTALL_STATUSES:
            messages.append(f"installation status is {install_row['status']!r}")
        if not expected_hash:
            messages.append("installed_sha256 is empty")
        try:
            source_label_path = source_label_path.resolve(strict=True)
            if sha256_file(source_label_path) != expected_hash:
                messages.append("source CHARM label hash differs from installed_sha256")
        except (FileNotFoundError, OSError) as exc:
            messages.append(f"source CHARM label is unavailable: {exc}")

        try:
            label_path = label_path.resolve(strict=True)
            if not label_path.is_relative_to(roi_root_path):
                messages.append(f"label is outside ROI root: {label_path}")
            if label_path.name != TARGET_BASENAME:
                messages.append(f"unexpected label basename: {label_path.name}")
            dataset_root = _dataset_root_from_label(label_path, dataset_name)
            anat_dir = dataset_root / subject / "anat"
            if label_path.parent.name != "label_prep":
                messages.append(f"label is not under label_prep: {label_path}")
            else:
                m2m_dir = label_path.parent.parent
            actual_hash = sha256_file(label_path)
            if expected_hash and actual_hash != expected_hash:
                messages.append(
                    f"installed label hash mismatch: {actual_hash} != {expected_hash}"
                )
        except (FileNotFoundError, OSError, ValueError) as exc:
            messages.append(str(exc))

        if anat_dir is not None:
            live_roast = [
                path for path in roast_segmentation_candidates(anat_dir, subject) if path.is_file()
            ]
            if live_roast:
                messages.append(
                    "forbidden ROAST/custom segmentation remains: "
                    + ";".join(str(path) for path in live_roast)
                )
            existing_mesh = resolve_existing_mesh(anat_dir, subject)
            if existing_mesh is not None and existing_mesh.is_symlink():
                messages.append(f"existing mesh is a symlink: {existing_mesh}")
            mesh_path = existing_mesh or (
                m2m_dir / f"{subject}.msh" if m2m_dir is not None else ""
            )

        rows.append(
            {
                "task_id": len(rows),
                "dataset_name": dataset_name,
                "roi_prefix": row_roi_prefix,
                "repeat_id": repeat_id,
                "dataset_root": dataset_root or "",
                "subject": subject,
                "anat_dir": anat_dir or "",
                "m2m_dir": m2m_dir or "",
                "label_path": label_path,
                "source_label_path": source_label_path,
                "expected_label_sha256": expected_hash,
                "mesh_path": mesh_path,
                "status": "ready" if not messages else "blocked",
                "message": "ready" if not messages else "; ".join(messages),
            }
        )

    if len(rows) != expected_targets:
        global_message = f"task count mismatch: found {len(rows)}, expected {expected_targets}"
        for row in rows:
            row["status"] = "blocked"
            row["message"] = (
                global_message
                if row["message"] == "ready"
                else f"{row['message']}; {global_message}"
            )

    manifest_path = write_tsv(manifest, REMESH_FIELDNAMES, rows)
    ready = sum(row["status"] == "ready" for row in rows)
    payload = {
        "status": "ready" if ready == expected_targets else "blocked",
        "install_manifest": str(Path(install_manifest).expanduser().resolve()),
        "roi_root": str(roi_root_path),
        "roi_prefix": selected_roi_prefix,
        "manifest": str(manifest_path.resolve()),
        "targets_found": len(rows),
        "targets_expected": expected_targets,
        "ready": ready,
        "blocked": len(rows) - ready,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json_atomic(summary, payload)
    return payload


def remove_roast_segmentations(
    *,
    install_manifest: str | Path,
    roi_root: str | Path,
    report: str | Path,
    apply_delete: bool = False,
    external_backup_confirmed: bool = False,
    expected_targets: int = 7000,
    roi_prefix: str | None = None,
    expected_repeats: int = 10,
    expected_subjects: int = 175,
) -> dict[str, object]:
    roi_root_path = Path(roi_root).expanduser().resolve()
    if apply_delete and not external_backup_confirmed:
        raise ValueError(
            "--confirm-external-backup is required with --apply-delete"
        )

    install_rows = read_tsv(install_manifest)
    _validate_install_manifest_header(install_rows)
    install_rows = select_install_rows(
        install_rows,
        roi_prefix=roi_prefix,
        expected_targets=expected_targets,
        expected_repeats=expected_repeats,
        expected_subjects=expected_subjects,
    )
    if len(install_rows) != expected_targets:
        raise ValueError(
            f"installation manifest has {len(install_rows)} rows; expected {expected_targets}"
        )

    report_rows: list[dict[str, object]] = []
    candidates: list[tuple[Path, dict[str, object]]] = []
    seen_rows: set[tuple[str, str]] = set()
    for install_row in install_rows:
        dataset_name = install_row["dataset"].strip()
        subject = install_row["subject"].strip()
        key = (dataset_name, subject)
        if key in seen_rows:
            raise ValueError(
                f"duplicate dataset/subject row in installation manifest: {key}"
            )
        seen_rows.add(key)
        if install_row["status"].strip() not in READY_INSTALL_STATUSES:
            raise ValueError(
                f"installation status for {dataset_name}/{subject} is "
                f"{install_row['status']!r}"
            )
        label_path = Path(install_row["destination"]).expanduser().resolve(strict=True)
        if not label_path.is_relative_to(roi_root_path):
            raise ValueError(f"label is outside ROI root: {label_path}")
        if label_path.name != TARGET_BASENAME:
            raise ValueError(f"unexpected label basename: {label_path.name}")
        dataset_root = _dataset_root_from_label(label_path, dataset_name)
        anat_dir = dataset_root / subject / "anat"
        for source in roast_segmentation_candidates(anat_dir, subject):
            if not source.is_file():
                continue
            if source.is_symlink():
                raise ValueError(f"refusing to delete ROAST/custom symlink: {source}")
            resolved_source = source.resolve(strict=True)
            if not resolved_source.is_relative_to(roi_root_path):
                raise ValueError(
                    f"ROAST/custom map is outside ROI root: {resolved_source}"
                )
            digest = sha256_file(source)
            byte_count = source.stat().st_size
            row: dict[str, object] = {
                "dataset_name": dataset_name,
                "subject": subject,
                "path": source,
                "sha256": digest,
                "bytes": byte_count,
                "action": "would_delete",
                "message": "dry run",
            }
            candidates.append((source, row))
            report_rows.append(row)

    deleted = 0
    failed = 0
    if apply_delete:
        for source, row in candidates:
            try:
                source.unlink()
                if source.exists():
                    raise IOError(f"ROAST/custom map still exists after unlink: {source}")
                row["action"] = "deleted"
                row["message"] = "deleted after external-backup confirmation"
                deleted += 1
            except Exception as exc:
                row["action"] = "failed"
                row["message"] = str(exc)
                failed += 1
                break
        if failed:
            for _, row in candidates:
                if row["action"] == "would_delete":
                    row["action"] = "not_attempted"
                    row["message"] = "stopped after an earlier deletion failure"

    report_path = write_tsv(report, ROAST_REMOVAL_FIELDNAMES, report_rows)
    payload = {
        "status": "failed" if failed else ("complete" if apply_delete else "audit"),
        "mode": "apply_delete" if apply_delete else "audit",
        "install_manifest": str(Path(install_manifest).expanduser().resolve()),
        "roi_root": str(roi_root_path),
        "roi_prefix": roi_prefix,
        "report": str(report_path.resolve()),
        "found": len(candidates),
        "deleted": deleted,
        "failed": failed,
    }
    write_json_atomic(report_path.with_suffix(".summary.json"), payload)
    return payload


def _prune_empty_tree(root: Path) -> bool:
    if not root.is_dir():
        return False
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        return False
    return True


def remove_installation_backups(
    *,
    install_manifest: str | Path,
    backup_root: str | Path,
    report: str | Path,
    apply_delete: bool = False,
    external_backup_confirmed: bool = False,
    expected_backups: int = 7000,
    roi_prefix: str | None = None,
    expected_repeats: int = 10,
    expected_subjects: int = 175,
) -> dict[str, object]:
    if apply_delete and not external_backup_confirmed:
        raise ValueError(
            "--confirm-external-backup is required with --apply-delete"
        )
    root = Path(backup_root).expanduser().resolve()
    if root == Path(root.anchor):
        raise ValueError("backup root cannot be a filesystem root")

    install_rows = read_tsv(install_manifest)
    _validate_install_manifest_header(install_rows)
    install_rows = select_install_rows(
        install_rows,
        roi_prefix=roi_prefix,
        expected_targets=expected_backups,
        expected_repeats=expected_repeats,
        expected_subjects=expected_subjects,
    )
    required = {"backup", "before_sha256"}
    missing_columns = sorted(required - set(install_rows[0]))
    if missing_columns:
        raise ValueError(
            "Installation manifest is missing backup column(s): "
            + ", ".join(missing_columns)
        )

    report_rows: list[dict[str, object]] = []
    candidates: list[tuple[Path, dict[str, object]]] = []
    seen_paths: set[Path] = set()
    issues = 0
    for install_row in install_rows:
        backup_text = install_row["backup"].strip()
        expected_hash = install_row["before_sha256"].strip()
        row: dict[str, object] = {
            "dataset_name": install_row["dataset"].strip(),
            "subject": install_row["subject"].strip(),
            "path": backup_text,
            "sha256": expected_hash,
            "bytes": "",
            "action": "blocked",
            "message": "",
        }
        try:
            if install_row["status"].strip() not in READY_INSTALL_STATUSES:
                raise ValueError(
                    f"installation status is not complete: {install_row['status']!r}"
                )
            if not backup_text:
                raise ValueError("installation manifest backup path is empty")
            unresolved = Path(backup_text).expanduser()
            if unresolved.is_symlink():
                raise ValueError(f"backup is a symlink: {unresolved}")
            backup_path = unresolved.resolve(strict=True)
            if not backup_path.is_relative_to(root):
                raise ValueError(f"backup is outside declared root: {backup_path}")
            if not backup_path.is_file():
                raise ValueError(f"backup is not a regular file: {backup_path}")
            if backup_path in seen_paths:
                raise ValueError(f"duplicate backup path in manifest: {backup_path}")
            seen_paths.add(backup_path)
            actual_hash = sha256_file(backup_path)
            if not expected_hash or actual_hash != expected_hash:
                raise ValueError(
                    f"backup hash mismatch: {actual_hash} != {expected_hash}"
                )
            row.update(
                {
                    "path": backup_path,
                    "sha256": actual_hash,
                    "bytes": backup_path.stat().st_size,
                    "action": "would_delete",
                    "message": "dry run",
                }
            )
            candidates.append((backup_path, row))
        except (FileNotFoundError, OSError, ValueError) as exc:
            row["message"] = str(exc)
            issues += 1
        report_rows.append(row)

    if len(candidates) != expected_backups:
        issues += 1
        count_message = (
            f"backup count mismatch: found {len(candidates)}, expected {expected_backups}"
        )
        for _, row in candidates:
            row["action"] = "blocked"
            row["message"] = count_message

    deleted = 0
    if apply_delete and issues == 0:
        for backup_path, row in candidates:
            try:
                backup_path.unlink()
                if backup_path.exists():
                    raise IOError(f"backup still exists after unlink: {backup_path}")
                row["action"] = "deleted"
                row["message"] = "deleted after external-backup confirmation"
                deleted += 1
            except OSError as exc:
                row["action"] = "failed"
                row["message"] = str(exc)
                issues += 1
                break
        if issues:
            for _, row in candidates:
                if row["action"] == "would_delete":
                    row["action"] = "not_attempted"
                    row["message"] = "stopped after an earlier deletion failure"

    report_path = write_tsv(report, ROAST_REMOVAL_FIELDNAMES, report_rows)
    root_removed = _prune_empty_tree(root) if apply_delete and issues == 0 else False
    payload = {
        "status": (
            "blocked"
            if issues and not apply_delete
            else ("failed" if issues else ("complete" if apply_delete else "audit"))
        ),
        "mode": "apply_delete" if apply_delete else "audit",
        "install_manifest": str(Path(install_manifest).expanduser().resolve()),
        "backup_root": str(root),
        "roi_prefix": roi_prefix,
        "report": str(report_path.resolve()),
        "found": len(candidates),
        "expected": expected_backups,
        "deleted": deleted,
        "issues": issues,
        "backup_root_removed": root_removed,
    }
    write_json_atomic(report_path.with_suffix(".summary.json"), payload)
    return payload


def result_path_for_task(result_dir: str | Path, dataset_name: str, subject: str) -> Path:
    safe_dataset = "".join(char if char.isalnum() or char in "_.-" else "_" for char in dataset_name)
    safe_subject = "".join(char if char.isalnum() or char in "_.-" else "_" for char in subject)
    return Path(result_dir) / safe_dataset / f"{safe_subject}.json"


def validate_mesh_payload(mesh_path: Path, *, load_mesh: bool = True) -> dict[str, object]:
    if not mesh_path.is_file():
        raise FileNotFoundError(f"mesh not found: {mesh_path}")
    size = mesh_path.stat().st_size
    if size <= 0:
        raise ValueError(f"mesh is empty: {mesh_path}")
    payload: dict[str, object] = {
        "mesh_path": str(mesh_path),
        "mesh_bytes": size,
        "mesh_sha256": sha256_file(mesh_path),
    }
    if load_mesh:
        from simnibs import mesh_io

        mesh = mesh_io.read_msh(str(mesh_path))
        element_types = mesh.elm.elm_type
        tetra_count = int((element_types == 4).sum())
        if tetra_count <= 0:
            raise ValueError(f"mesh has no tetrahedral elements: {mesh_path}")
        payload["tetrahedra"] = tetra_count
        payload["tissue_tags"] = sorted(
            {int(value) for value in mesh.elm.tag1[element_types == 4]}
        )
    return payload


def _remove_existing_mesh(mesh_path: Path) -> dict[str, object]:
    if mesh_path.is_symlink():
        raise ValueError(f"refusing to remove mesh symlink: {mesh_path}")
    if not mesh_path.is_file():
        return {
            "removed_previous_mesh_path": None,
            "removed_previous_mesh_sha256": None,
            "removed_previous_mesh_bytes": 0,
        }
    payload = {
        "removed_previous_mesh_path": str(mesh_path),
        "removed_previous_mesh_sha256": sha256_file(mesh_path),
        "removed_previous_mesh_bytes": mesh_path.stat().st_size,
    }
    mesh_path.unlink()
    if mesh_path.exists():
        raise IOError(f"could not remove obsolete mesh: {mesh_path}")
    return payload


def _remove_failed_subject_meshes(anat_dir: Path, subject: str) -> None:
    for mesh_path in candidate_mesh_paths(anat_dir, subject):
        if mesh_path.is_symlink():
            raise ValueError(f"refusing to remove failed mesh symlink: {mesh_path}")
        if mesh_path.is_file():
            mesh_path.unlink()
        if mesh_path.exists():
            raise IOError(f"failed remesh output could not be removed: {mesh_path}")


def _restore_installed_label(source: Path, destination: Path, expected_hash: str) -> None:
    if sha256_file(source) != expected_hash:
        raise IOError(f"CHARM source label hash changed: {source}")
    temporary = destination.with_name(f".{destination.name}.restore-{os.getpid()}")
    shutil.copy2(source, temporary)
    if sha256_file(temporary) != expected_hash:
        temporary.unlink(missing_ok=True)
        raise IOError(f"CHARM label rollback verification failed: {destination}")
    os.replace(temporary, destination)


def run_remesh_task(
    *,
    manifest: str | Path,
    task_index: int,
    result_dir: str | Path,
    charm_bin: str = "charm",
    load_mesh: bool = True,
) -> dict[str, object]:
    rows = read_tsv(manifest)
    if task_index < 0 or task_index >= len(rows):
        raise IndexError(f"task index {task_index} is outside manifest range 0..{len(rows)-1}")
    row = rows[task_index]
    if row["status"] != "ready":
        raise ValueError(f"task {task_index} is not ready: {row['message']}")

    dataset_name = row["dataset_name"]
    subject = row["subject"]
    if not subject.startswith("sub-") or "/" in subject:
        raise ValueError(f"unexpected subject identifier: {subject!r}")
    dataset_root = Path(row["dataset_root"]).expanduser().resolve(strict=True)
    anat_dir = Path(row["anat_dir"]).expanduser().resolve(strict=True)
    if anat_dir != dataset_root / subject / "anat":
        raise ValueError(
            f"anat directory does not match dataset/subject identity: {anat_dir}"
        )
    label_path = Path(row["label_path"]).expanduser().resolve(strict=True)
    if not label_path.is_relative_to(anat_dir) or label_path.name != TARGET_BASENAME:
        raise ValueError(f"unexpected installed label path: {label_path}")
    source_label_path = Path(row["source_label_path"])
    expected_hash = row["expected_label_sha256"]
    result_path = result_path_for_task(result_dir, dataset_name, subject)

    if result_path.is_file():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        current_mesh = Path(previous.get("mesh_path", ""))
        if (
            previous.get("status") == "complete"
            and label_path.is_file()
            and sha256_file(label_path) == expected_hash
            and current_mesh.is_file()
            and sha256_file(current_mesh) == previous.get("mesh_sha256")
        ):
            previous["status"] = "already_complete"
            return previous

    live_roast = [
        path for path in roast_segmentation_candidates(anat_dir, subject) if path.is_file()
    ]
    if live_roast:
        raise ValueError(
            "refusing remesh while ROAST/custom segmentation remains: "
            + ";".join(str(path) for path in live_roast)
        )
    before_hash = sha256_file(label_path)
    if before_hash != expected_hash:
        raise ValueError(f"label hash mismatch before remesh: {before_hash} != {expected_hash}")
    source_hash = sha256_file(source_label_path)
    if source_hash != expected_hash:
        raise ValueError(
            f"source CHARM label hash mismatch before remesh: {source_hash} != {expected_hash}"
        )

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    command = [charm_bin, subject, "--mesh"]
    prior_mesh_path = Path(row["mesh_path"]).expanduser()
    allowed_mesh_paths = set(candidate_mesh_paths(anat_dir, subject))
    if not prior_mesh_path.is_absolute() or prior_mesh_path not in allowed_mesh_paths:
        raise ValueError(
            f"manifest mesh path is outside the allowed subject locations: {prior_mesh_path}"
        )
    removed_mesh = _remove_existing_mesh(prior_mesh_path)
    try:
        print(json.dumps({"event": "charm_only_remesh_start", "command": command, "cwd": str(anat_dir)}))
        completed = subprocess.run(command, cwd=anat_dir, capture_output=True, text=True)
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="" if completed.stderr.endswith("\n") else "\n")

        after_hash = sha256_file(label_path)
        if after_hash != before_hash:
            raise RuntimeError(
                f"CRITICAL: CHARM changed the installed segmentation map: {after_hash} != {before_hash}"
            )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, command)

        mesh_path = resolve_existing_mesh(anat_dir, subject)
        if mesh_path is None:
            raise FileNotFoundError(
                f"CHARM completed but no mesh was found for {subject} under {anat_dir}"
            )
        if mesh_path.is_symlink() or not mesh_path.resolve(strict=True).is_relative_to(
            anat_dir
        ):
            raise ValueError(f"new mesh is not a regular in-subject path: {mesh_path}")
        mesh_payload = validate_mesh_payload(mesh_path, load_mesh=load_mesh)
    except Exception:
        _remove_failed_subject_meshes(anat_dir, subject)
        if not label_path.is_file() or sha256_file(label_path) != expected_hash:
            _restore_installed_label(source_label_path, label_path, expected_hash)
        raise
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "task_index": task_index,
        "dataset_name": dataset_name,
        "subject": subject,
        "anat_dir": str(anat_dir),
        "label_path": str(label_path),
        "source_label_path": str(source_label_path),
        "label_sha256_before": before_hash,
        "label_sha256_after": after_hash,
        "command": command,
        **removed_mesh,
        "started_at": started_at,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **mesh_payload,
    }
    write_json_atomic(result_path, payload)
    print(json.dumps({"event": "charm_only_remesh_complete", **payload}, default=str))
    return payload


def validate_remesh_results(
    *,
    manifest: str | Path,
    result_dir: str | Path,
    summary: str | Path,
    load_mesh: bool = True,
    task_indices: Sequence[int] | None = None,
) -> dict[str, object]:
    rows = read_tsv(manifest)
    if task_indices is not None:
        invalid = sorted(index for index in task_indices if index < 0 or index >= len(rows))
        if invalid:
            raise IndexError(f"validation task indices are outside the manifest: {invalid}")
        rows = [rows[index] for index in dict.fromkeys(task_indices)]
    validation_rows: list[dict[str, object]] = []
    complete = 0
    for row in rows:
        messages: list[str] = []
        if row["status"] != "ready":
            messages.append(f"manifest task is {row['status']}: {row['message']}")
        result_path = result_path_for_task(result_dir, row["dataset_name"], row["subject"])
        result: dict[str, object] = {}
        if not result_path.is_file():
            messages.append(f"result missing: {result_path}")
        else:
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if result.get("status") != "complete":
                    messages.append(f"result status is {result.get('status')!r}")
                if (
                    result.get("dataset_name") != row["dataset_name"]
                    or result.get("subject") != row["subject"]
                ):
                    messages.append("result identity differs from manifest")
                label_path = Path(row["label_path"])
                actual_label_hash = sha256_file(label_path)
                if actual_label_hash != row["expected_label_sha256"]:
                    messages.append("installed segmentation hash changed")
                if result.get("label_sha256_after") != row["expected_label_sha256"]:
                    messages.append("result label hash differs from manifest")
                if sha256_file(Path(row["source_label_path"])) != row["expected_label_sha256"]:
                    messages.append("source CHARM segmentation hash changed")
                mesh_path = Path(str(result.get("mesh_path", "")))
                resolved_mesh = resolve_existing_mesh(Path(row["anat_dir"]), row["subject"])
                if resolved_mesh is None or mesh_path.resolve() != resolved_mesh.resolve():
                    messages.append("result mesh is not the current subject mesh")
                mesh_payload = validate_mesh_payload(mesh_path, load_mesh=load_mesh)
                if mesh_payload["mesh_sha256"] != result.get("mesh_sha256"):
                    messages.append("mesh hash differs from task result")
            except Exception as exc:
                messages.append(str(exc))
        status = "complete" if not messages else "incomplete"
        complete += status == "complete"
        validation_rows.append(
            {
                "dataset_name": row["dataset_name"],
                "subject": row["subject"],
                "status": status,
                "result": result_path,
                "message": "; ".join(messages) if messages else "complete",
            }
        )

    summary_path = Path(summary).expanduser()
    write_tsv(
        summary_path,
        ("dataset_name", "subject", "status", "result", "message"),
        validation_rows,
    )
    payload = {
        "status": "complete" if complete == len(rows) else "incomplete",
        "tasks": len(rows),
        "complete": complete,
        "incomplete": len(rows) - complete,
        "summary": str(summary_path.resolve()),
    }
    write_json_atomic(summary_path.with_suffix(".json"), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    remove_roast = subparsers.add_parser("remove-roast")
    remove_roast.add_argument("--install-manifest", required=True)
    remove_roast.add_argument("--roi-root", required=True)
    remove_roast.add_argument("--report", required=True)
    remove_roast.add_argument("--expected-targets", type=int, default=7000)
    remove_roast.add_argument("--roi-prefix")
    remove_roast.add_argument("--expected-repeats", type=int, default=10)
    remove_roast.add_argument("--expected-subjects", type=int, default=175)
    remove_roast.add_argument("--apply-delete", action="store_true")
    remove_roast.add_argument("--confirm-external-backup", action="store_true")

    remove_backups = subparsers.add_parser("remove-install-backups")
    remove_backups.add_argument("--install-manifest", required=True)
    remove_backups.add_argument("--backup-root", required=True)
    remove_backups.add_argument("--report", required=True)
    remove_backups.add_argument("--expected-backups", type=int, default=7000)
    remove_backups.add_argument("--roi-prefix")
    remove_backups.add_argument("--expected-repeats", type=int, default=10)
    remove_backups.add_argument("--expected-subjects", type=int, default=175)
    remove_backups.add_argument("--apply-delete", action="store_true")
    remove_backups.add_argument("--confirm-external-backup", action="store_true")

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--install-manifest", required=True)
    preflight.add_argument("--roi-root", required=True)
    preflight.add_argument("--manifest", required=True)
    preflight.add_argument("--summary", required=True)
    preflight.add_argument("--expected-targets", type=int, default=7000)
    preflight.add_argument("--roi-prefix")
    preflight.add_argument("--expected-repeats", type=int, default=10)
    preflight.add_argument("--expected-subjects", type=int, default=175)

    run_task = subparsers.add_parser("run-task")
    run_task.add_argument("--manifest", required=True)
    run_task.add_argument("--task-index", type=int, required=True)
    run_task.add_argument("--result-dir", required=True)
    run_task.add_argument("--charm-bin", default="charm")
    run_task.add_argument("--skip-mesh-load", action="store_true")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--result-dir", required=True)
    validate.add_argument("--summary", required=True)
    validate.add_argument(
        "--task-index",
        type=int,
        action="append",
        help="Validate only this task index; repeat for a smoke-test subset.",
    )
    validate.add_argument("--skip-mesh-load", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "remove-roast":
        result = remove_roast_segmentations(
            install_manifest=args.install_manifest,
            roi_root=args.roi_root,
            report=args.report,
            apply_delete=args.apply_delete,
            external_backup_confirmed=args.confirm_external_backup,
            expected_targets=args.expected_targets,
            roi_prefix=args.roi_prefix,
            expected_repeats=args.expected_repeats,
            expected_subjects=args.expected_subjects,
        )
    elif args.command == "remove-install-backups":
        result = remove_installation_backups(
            install_manifest=args.install_manifest,
            backup_root=args.backup_root,
            report=args.report,
            apply_delete=args.apply_delete,
            external_backup_confirmed=args.confirm_external_backup,
            expected_backups=args.expected_backups,
            roi_prefix=args.roi_prefix,
            expected_repeats=args.expected_repeats,
            expected_subjects=args.expected_subjects,
        )
    elif args.command == "preflight":
        result = build_remesh_manifest(
            install_manifest=args.install_manifest,
            roi_root=args.roi_root,
            manifest=args.manifest,
            summary=args.summary,
            expected_targets=args.expected_targets,
            roi_prefix=args.roi_prefix,
            expected_repeats=args.expected_repeats,
            expected_subjects=args.expected_subjects,
        )
    elif args.command == "run-task":
        result = run_remesh_task(
            manifest=args.manifest,
            task_index=args.task_index,
            result_dir=args.result_dir,
            charm_bin=args.charm_bin,
            load_mesh=not args.skip_mesh_load,
        )
    elif args.command == "validate":
        result = validate_remesh_results(
            manifest=args.manifest,
            result_dir=args.result_dir,
            summary=args.summary,
            load_mesh=not args.skip_mesh_load,
            task_indices=args.task_index,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("status") not in {"blocked", "failed", "incomplete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
