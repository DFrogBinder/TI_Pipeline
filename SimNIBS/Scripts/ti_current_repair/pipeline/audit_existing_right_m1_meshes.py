#!/usr/bin/env python3
"""Read-only content audit of existing v4 and v3 right-M1 head meshes.

The command writes only JSON to stdout. It never creates, edits, or deletes a
file under either experiment root. Redirect stdout to a local file if a saved
receipt is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


SUBJECTS = (
    "sub-CC110174",
    "sub-CC121144",
    "sub-CC310407",
    "sub-CC320616",
    "sub-CC420071",
    "sub-CC410432",
    "sub-CC520083",
    "sub-CC520127",
    "sub-CC610631",
    "sub-CC720941",
)
DEFAULT_V4_ROOT = Path(
    "/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_right_m1"
)
DEFAULT_V3_ROOT = Path(
    "/mnt/parscratch/users/cop23bi/"
    "final_132_repeatability_balanced_10_simnibs326_right_m1_v1"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remesh_path(root: Path, subject: str, repeat_index: int) -> Path:
    return (
        root
        / f"{subject}_repeatability"
        / "remesh"
        / "repeats"
        / f"repeat_{repeat_index:03d}"
        / subject
        / "anat"
        / f"m2m_{subject}"
        / f"{subject}.msh"
    )


def fixed_cache_path(root: Path, subject: str) -> Path:
    return (
        root
        / f"{subject}_repeatability"
        / "fixed_mesh"
        / "mesh_cache"
        / subject
        / "anat"
        / f"m2m_{subject}"
        / f"{subject}.msh"
    )


def _mesh_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "present": False}
    stat = path.stat()
    record: dict[str, Any] = {
        "path": str(path),
        "present": True,
        "size_bytes": stat.st_size,
        "is_symlink": path.is_symlink(),
    }
    try:
        record["sha256"] = file_sha256(path)
        record["readable"] = True
    except OSError as exc:
        record["readable"] = False
        record["read_error"] = f"{type(exc).__name__}: {exc}"
    return record


def audit_subject(
    *,
    root: Path,
    subject: str,
    repeat_count: int,
    expected_unique_remesh: int,
) -> dict[str, Any]:
    remesh = [
        _mesh_record(remesh_path(root, subject, index))
        for index in range(1, repeat_count + 1)
    ]
    fixed = _mesh_record(fixed_cache_path(root, subject))
    present = [record for record in remesh if record["present"]]
    readable = [record for record in present if record.get("readable")]
    hashes = [str(record["sha256"]) for record in readable]
    unique_hashes = sorted(set(hashes))
    sizes = [int(record["size_bytes"]) for record in present]
    missing = [str(record["path"]) for record in remesh if not record["present"]]
    checks = {
        "all_remesh_meshes_present": len(present) == repeat_count,
        "all_remesh_meshes_readable": len(readable) == repeat_count,
        "expected_unique_remesh_hashes": len(unique_hashes) == expected_unique_remesh,
        "fixed_cache_mesh_present": bool(fixed["present"]),
        "fixed_cache_hash_is_remesh_hash": bool(
            fixed["present"] and fixed.get("sha256") in unique_hashes
        ),
    }
    return {
        "subject": subject,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "remesh": {
            "expected": repeat_count,
            "present": len(present),
            "readable": len(readable),
            "missing": missing,
            "unique_sha256_count": len(unique_hashes),
            "unique_sha256": unique_hashes,
            "minimum_size_bytes": min(sizes) if sizes else None,
            "maximum_size_bytes": max(sizes) if sizes else None,
            "symlink_count": sum(bool(record.get("is_symlink")) for record in present),
            "mesh_records": remesh,
        },
        "fixed_cache": fixed,
    }


def audit_version(
    *,
    label: str,
    root: Path,
    subjects: tuple[str, ...],
    repeat_count: int,
    expected_unique_remesh: int,
) -> dict[str, Any]:
    results = []
    for index, subject in enumerate(subjects, start=1):
        print(
            f"[{label}] hashing subject {index}/{len(subjects)}: {subject}",
            file=sys.stderr,
            flush=True,
        )
        results.append(
            audit_subject(
                root=root,
                subject=subject,
                repeat_count=repeat_count,
                expected_unique_remesh=expected_unique_remesh,
            )
        )
    passed = sum(result["status"] == "pass" for result in results)
    return {
        "label": label,
        "root": str(root),
        "root_present": root.is_dir(),
        "expected_unique_remesh_meshes_per_subject": expected_unique_remesh,
        "subjects_expected": len(subjects),
        "subjects_passed": passed,
        "status": "pass" if passed == len(subjects) else "fail",
        "subjects": results,
    }


def run(args: argparse.Namespace) -> int:
    subjects = tuple(item.strip() for item in args.subjects.split(",") if item.strip())
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError("--subjects must contain unique comma-separated subject IDs.")
    if args.repeat_count <= 0:
        raise ValueError("--repeat-count must be positive.")

    versions = []
    if args.version in ("both", "v4"):
        versions.append(
            audit_version(
                label="simnibs_v4",
                root=args.v4_root.expanduser().resolve(),
                subjects=subjects,
                repeat_count=args.repeat_count,
                expected_unique_remesh=args.repeat_count,
            )
        )
    if args.version in ("both", "v3"):
        versions.append(
            audit_version(
                label="simnibs_v3_2_6",
                root=args.v3_root.expanduser().resolve(),
                subjects=subjects,
                repeat_count=args.repeat_count,
                expected_unique_remesh=1,
            )
        )

    status = "pass" if all(item["status"] == "pass" for item in versions) else "fail"
    receipt = {
        "schema_version": 1,
        "audit": "right-M1 existing mesh inputs for the pending tDCS repeatability study",
        "mode": "read_only_stdout_only",
        "status": status,
        "interpretation": {
            "simnibs_v4": f"{args.repeat_count} content-distinct remesh heads per subject plus one fixed-cache head selected from them.",
            "simnibs_v3_2_6": "One content-unique deterministic head per subject; this is a control, not a remeshing-variability arm.",
        },
        "versions": versions,
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if status == "pass" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-root", type=Path, default=DEFAULT_V4_ROOT)
    parser.add_argument("--v3-root", type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument("--subjects", default=",".join(SUBJECTS))
    parser.add_argument("--repeat-count", type=int, default=40)
    parser.add_argument("--version", choices=("both", "v4", "v3"), default="both")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
