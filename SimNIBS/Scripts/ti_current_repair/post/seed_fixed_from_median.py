#!/usr/bin/env python3
"""Seed fixed-mesh workspaces by physically copying selected remesh anatomy."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline import provenance


EXCLUDED_NAMES = {
    "SimNIBS",
    ".mesh_build.lock",
    ".mesh_cache_prepare.lock",
    ".mesh_force_reset.json",
    ".mesh_ready.json",
    "task_manifest.json",
}
EXCLUDED_PREFIXES = ("slurm-",)
DEFAULT_MANIFEST = "_pipeline/fixed_seed_manifest.csv"


class SymlinkValidationError(RuntimeError):
    pass


def manifest_fields() -> list[str]:
    return [
        "subject",
        "selected_repeat_tag",
        "metric",
        "metric_value",
        "mesh_nodes",
        "mesh_checksum",
        "source_anat_dir",
        "source_mesh_path",
        "cache_anat_dir",
        "repeat_anat_dirs",
        "copy_mode",
        "validation_result",
        "checksum_matches",
        "message",
    ]


def _repeat_tag(index: int) -> str:
    return f"repeat_{index:03d}"


def _fixed_cache_anat(experiment_root: Path, subject: str) -> Path:
    return experiment_root / f"{subject}_repeatability" / "fixed_mesh" / "mesh_cache" / subject / "anat"


def _fixed_repeat_anat(experiment_root: Path, subject: str, repeat_tag: str) -> Path:
    return (
        experiment_root
        / f"{subject}_repeatability"
        / "fixed_mesh"
        / "repeats"
        / repeat_tag
        / subject
        / "anat"
    )


def _remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _ignore(_directory: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        if name in EXCLUDED_NAMES:
            ignored.add(name)
        elif name.startswith(EXCLUDED_PREFIXES):
            ignored.add(name)
        elif name.endswith(".tmp") or name.endswith(".lock"):
            ignored.add(name)
    return ignored


def _copy_anat(source_anat: Path, dest_anat: Path, *, overwrite: bool) -> None:
    if dest_anat.exists() or dest_anat.is_symlink():
        if not overwrite:
            raise FileExistsError(dest_anat)
        _remove_existing(dest_anat)
    dest_anat.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_anat, dest_anat, symlinks=False, ignore=_ignore)


def validate_no_symlinks(root: Path) -> None:
    symlinks = provenance.find_symlinks(root)
    if symlinks:
        preview = ", ".join(str(path) for path in symlinks[:8])
        raise SymlinkValidationError(f"Symlinks are not allowed under {root}: {preview}")


def _write_ready_marker(
    anat_dir: Path,
    *,
    subject: str,
    mesh_path: Path,
    mesh_checksum: str,
    source_anat: Path,
    selected_repeat_tag: str,
) -> None:
    marker = {
        "subject": subject,
        "mesh_path": str(mesh_path),
        "mesh_checksum": mesh_checksum,
        "status": "mesh_ready",
        "source_anat_dir": str(source_anat),
        "selected_repeat_tag": selected_repeat_tag,
        "copy_mode": "physical_copy",
        "created_at": time.time(),
    }
    (anat_dir / ".mesh_ready.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


def _read_selected_rows(selection_csv: Path) -> list[dict[str, str]]:
    with selection_csv.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields())
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in manifest_fields()})


def _seed_one(
    row: dict[str, str],
    *,
    experiment_root: Path,
    repeat_count: int,
    overwrite: bool,
) -> dict[str, str]:
    subject = row.get("subject", "").strip()
    repeat_tag = row.get("selected_repeat_tag", "").strip()
    source_m2m = Path(row.get("selected_m2m_dir", "").strip())
    source_anat = source_m2m.parent
    source_mesh = source_m2m / f"{subject}.msh"
    selected_checksum = row.get("mesh_checksum", "").strip()
    if row.get("selection_status") != "selected":
        return {
            "subject": subject,
            "selected_repeat_tag": repeat_tag,
            "copy_mode": "physical_copy",
            "validation_result": "skipped_nonselected",
            "message": row.get("selection_status", ""),
        }
    if not source_mesh.is_file():
        raise FileNotFoundError(source_mesh)
    actual_checksum = provenance.file_sha256(source_mesh)
    if selected_checksum and selected_checksum != actual_checksum:
        raise ValueError(f"Selection checksum mismatch for {subject}: {source_mesh}")

    destinations = [_fixed_cache_anat(experiment_root, subject)] + [
        _fixed_repeat_anat(experiment_root, subject, _repeat_tag(index))
        for index in range(1, repeat_count + 1)
    ]
    for dest_anat in destinations:
        _copy_anat(source_anat, dest_anat, overwrite=overwrite)
        dest_mesh = dest_anat / f"m2m_{subject}" / f"{subject}.msh"
        if not dest_mesh.is_file():
            raise FileNotFoundError(dest_mesh)
        if provenance.file_sha256(dest_mesh) != actual_checksum:
            raise ValueError(f"Copied mesh checksum mismatch: {dest_mesh}")
        validate_no_symlinks(dest_anat)
        _write_ready_marker(
            dest_anat,
            subject=subject,
            mesh_path=dest_mesh,
            mesh_checksum=actual_checksum,
            source_anat=source_anat,
            selected_repeat_tag=repeat_tag,
        )

    return {
        "subject": subject,
        "selected_repeat_tag": repeat_tag,
        "metric": row.get("metric", ""),
        "metric_value": row.get("metric_value", ""),
        "mesh_nodes": row.get("mesh_nodes", ""),
        "mesh_checksum": actual_checksum,
        "source_anat_dir": str(source_anat),
        "source_mesh_path": str(source_mesh),
        "cache_anat_dir": str(destinations[0]),
        "repeat_anat_dirs": ";".join(str(path) for path in destinations[1:]),
        "copy_mode": "physical_copy",
        "validation_result": "ok",
        "checksum_matches": "true",
        "message": f"seeded {len(destinations)} destinations",
    }


def seed_fixed_meshes(
    *,
    experiment_root: Path,
    selection_csv: Path,
    repeat_count: int,
    manifest: Path | None = None,
    overwrite: bool = False,
) -> Path:
    if repeat_count < 1:
        raise ValueError("repeat_count must be >= 1")
    manifest = manifest or experiment_root / DEFAULT_MANIFEST
    rows = [
        _seed_one(row, experiment_root=experiment_root, repeat_count=repeat_count, overwrite=overwrite)
        for row in _read_selected_rows(selection_csv)
    ]
    _write_manifest(manifest, rows)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--repeat-count", type=int, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = seed_fixed_meshes(
        experiment_root=args.experiment_root.expanduser().resolve(),
        selection_csv=args.selection_csv.expanduser().resolve(),
        repeat_count=args.repeat_count,
        manifest=args.manifest.expanduser().resolve() if args.manifest else None,
        overwrite=args.overwrite,
    )
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
