#!/usr/bin/env python3
"""Seed a Repeatability fixed-mesh cache from selected remesh m2m folders.

This script is intentionally self-contained for Stanage use. It reads the
geometry-preserving median-repeat selection CSV and copies each selected
``m2m_<subject>`` directory into the fixed-mesh cache layout expected by the
Repeatability runner:

    <new-root>/<subject>_repeatability/fixed_mesh/mesh_cache/<subject>/anat/

The source paths are taken directly from ``selected_m2m_dir`` in the CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MANIFEST_NAME = "fixed_median_mesh_dataset_manifest.csv"


@dataclass(frozen=True)
class Selection:
    subject: str
    status: str
    repeat_tag: str
    selected_m2m_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_selection_csv(path: Path) -> list[Selection]:
    rows: list[Selection] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                Selection(
                    subject=str(raw.get("subject", "")).strip(),
                    status=str(raw.get("selection_status", "")).strip(),
                    repeat_tag=str(raw.get("selected_repeat_tag", "")).strip(),
                    selected_m2m_dir=Path(str(raw.get("selected_m2m_dir", "")).strip()),
                )
            )
    return rows


def dest_anat_dir(new_root: Path, subject: str) -> Path:
    return new_root / f"{subject}_repeatability" / "fixed_mesh" / "mesh_cache" / subject / "anat"


def dest_m2m_dir(new_root: Path, subject: str) -> Path:
    return dest_anat_dir(new_root, subject) / f"m2m_{subject}"


def ready_marker_path(new_root: Path, subject: str) -> Path:
    return dest_anat_dir(new_root, subject) / ".mesh_ready.json"


def remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def write_ready_marker(path: Path, *, subject: str, mesh_path: Path, source_m2m_dir: Path) -> None:
    payload = {
        "subject": subject,
        "mesh_path": str(mesh_path),
        "status": "mesh_ready",
        "source_selected_m2m_dir": str(source_m2m_dir),
        "created_at": time.time(),
        "selection_source": "scaled_geometry_preserving_median_repeats",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def manifest_fields() -> list[str]:
    return [
        "subject",
        "repeat_tag",
        "status",
        "source_m2m_dir",
        "source_mesh_path",
        "dest_m2m_dir",
        "dest_mesh_path",
        "ready_marker_path",
        "message",
    ]


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields())
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in manifest_fields()})


def manifest_row(selection: Selection, new_root: Path, status: str, message: str) -> dict[str, str]:
    source_mesh = selection.selected_m2m_dir / f"{selection.subject}.msh"
    destination_m2m = dest_m2m_dir(new_root, selection.subject)
    destination_mesh = destination_m2m / f"{selection.subject}.msh"
    return {
        "subject": selection.subject,
        "repeat_tag": selection.repeat_tag,
        "status": status,
        "source_m2m_dir": str(selection.selected_m2m_dir),
        "source_mesh_path": str(source_mesh),
        "dest_m2m_dir": str(destination_m2m),
        "dest_mesh_path": str(destination_mesh),
        "ready_marker_path": str(ready_marker_path(new_root, selection.subject)),
        "message": message,
    }


def seed_dataset(selection_csv: Path, new_root: Path, manifest: Path, overwrite: bool, dry_run: bool) -> None:
    rows: list[dict[str, str]] = []
    selections = read_selection_csv(selection_csv)

    for selection in selections:
        if selection.status != "selected":
            rows.append(manifest_row(selection, new_root, "skipped_nonselected", selection.status))
            continue
        if not selection.subject or not selection.repeat_tag:
            rows.append(manifest_row(selection, new_root, "skipped_invalid_row", "Missing subject or repeat tag."))
            continue

        source_m2m = selection.selected_m2m_dir
        source_mesh = source_m2m / f"{selection.subject}.msh"
        destination_m2m = dest_m2m_dir(new_root, selection.subject)
        destination_mesh = destination_m2m / f"{selection.subject}.msh"
        marker = ready_marker_path(new_root, selection.subject)

        if not source_m2m.is_dir() or not source_mesh.is_file():
            rows.append(manifest_row(selection, new_root, "missing_source", "Source m2m directory or mesh missing."))
            write_manifest(manifest, rows)
            raise FileNotFoundError(source_mesh)

        if dry_run:
            rows.append(manifest_row(selection, new_root, "dry_run", "Source exists; no files copied."))
            continue

        if destination_m2m.exists() or destination_m2m.is_symlink():
            if not overwrite:
                rows.append(manifest_row(selection, new_root, "destination_exists", "Use --overwrite to replace."))
                write_manifest(manifest, rows)
                raise FileExistsError(destination_m2m)
            remove_existing(destination_m2m)

        destination_m2m.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_m2m, destination_m2m, symlinks=True)
        write_ready_marker(marker, subject=selection.subject, mesh_path=destination_mesh, source_m2m_dir=source_m2m)
        rows.append(manifest_row(selection, new_root, "seeded", "Copied selected m2m directory."))

    write_manifest(manifest, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"wrote manifest: {manifest}")
    print(", ".join(f"{key}={counts[key]}" for key in sorted(counts)))


def main() -> int:
    args = parse_args()
    selection_csv = args.selection_csv.expanduser().resolve()
    new_root = args.new_root.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve() if args.manifest else new_root / DEFAULT_MANIFEST_NAME
    seed_dataset(
        selection_csv=selection_csv,
        new_root=new_root,
        manifest=manifest,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
