#!/usr/bin/env python3
"""Build the ten reusable SimNIBS 3.2.6 headreco scaffolds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from settings import (
    HEAD_MODEL_STRATEGY,
    SEGMENTATION_PROVENANCE,
    SIMNIBS_MODULE,
    SUBJECTS,
    assert_output_root_isolation,
    scaffold_subject_root,
    source_paths,
)


READY_MARKER = ".simnibs326_scaffold_ready.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _simnibs_version() -> str:
    import simnibs  # type: ignore

    return str(getattr(simnibs, "__version__", "unknown"))


def _subject_from_index(index: int) -> str:
    if index < 0 or index >= len(SUBJECTS):
        raise IndexError(f"subject index {index} outside 0..{len(SUBJECTS) - 1}")
    return SUBJECTS[index]


def scaffold_paths(subject: str) -> dict[str, Path]:
    anat = scaffold_subject_root(subject)
    m2m = anat / f"m2m_{subject}"
    return {
        "anat": anat,
        "m2m": m2m,
        "native_mesh": anat / f"{subject}.msh",
        "compatibility_mesh": m2m / f"{subject}.msh",
        "ready_marker": anat / READY_MARKER,
    }


def validate_scaffold(subject: str) -> dict[str, object]:
    paths = scaffold_paths(subject)
    t1, t2, corrected_labels = source_paths(subject)
    surfaces = sorted(paths["m2m"].glob("*.stl")) if paths["m2m"].is_dir() else []
    cap = paths["m2m"] / "eeg_positions" / "EEG10-10_UI_Jurak_2007.csv"
    required = {
        "source_t1": t1.is_file(),
        "source_t2": t2.is_file(),
        "corrected_v4_labels_provenance_only": corrected_labels.is_file(),
        "m2m_directory": paths["m2m"].is_dir(),
        "t1fs_conform": (paths["m2m"] / "T1fs_conform.nii.gz").is_file(),
        "t1fs_nu_conform": (paths["m2m"] / "T1fs_nu_conform.nii.gz").is_file(),
        "to_mni": (paths["m2m"] / "toMNI").is_dir(),
        "eeg_cap": cap.is_file(),
        "surface_meshes": len(surfaces) >= 5,
        "native_mesh": paths["native_mesh"].is_file(),
        "compatibility_mesh": paths["compatibility_mesh"].is_file(),
        "ready_marker": paths["ready_marker"].is_file(),
    }
    checksums_match = False
    if required["native_mesh"] and required["compatibility_mesh"]:
        checksums_match = _sha256(paths["native_mesh"]) == _sha256(
            paths["compatibility_mesh"]
        )
    required["mesh_copies_match"] = checksums_match
    return {
        "subject": subject,
        "status": "ready" if all(required.values()) else "incomplete",
        "checks": required,
        "surface_count": len(surfaces),
        "paths": {key: str(value) for key, value in paths.items()},
    }


def build_scaffold(subject: str) -> dict[str, object]:
    assert_output_root_isolation()
    t1, t2, corrected_labels = source_paths(subject)
    missing = [path for path in (t1, t2, corrected_labels) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing source inputs for " + subject + ": " + ", ".join(map(str, missing))
        )

    paths = scaffold_paths(subject)
    existing = validate_scaffold(subject)
    if existing["status"] == "ready":
        existing["action"] = "reused"
        return existing

    # This is a dedicated, pipeline-owned scaffold directory.  A partial
    # headreco tree is not a reusable checkpoint, so retries rebuild it rather
    # than mixing files from two attempts.
    if paths["anat"].exists():
        shutil.rmtree(paths["anat"])
    paths["anat"].mkdir(parents=True, exist_ok=True)
    command = [
        "headreco",
        "all",
        "--noclean",
        subject,
        str(t1),
        str(t2),
    ]
    print(json.dumps({"event": "headreco_scaffold_start", "command": command}))
    subprocess.run(command, cwd=paths["anat"], check=True)

    if not paths["native_mesh"].is_file():
        raise FileNotFoundError(
            f"headreco completed without its native mesh: {paths['native_mesh']}"
        )
    if not paths["m2m"].is_dir():
        raise FileNotFoundError(f"headreco completed without m2m folder: {paths['m2m']}")
    shutil.copy2(paths["native_mesh"], paths["compatibility_mesh"])

    marker = {
        "schema_version": 1,
        "status": "ready",
        "created_utc": _utc_now(),
        "subject": subject,
        "simnibs_module": SIMNIBS_MODULE,
        "simnibs_version": _simnibs_version(),
        "loaded_modules": os.environ.get("LOADEDMODULES", ""),
        "head_model_strategy": HEAD_MODEL_STRATEGY,
        "segmentation_provenance": SEGMENTATION_PROVENANCE,
        "source_t1": str(t1),
        "source_t1_sha256": _sha256(t1),
        "source_t2": str(t2),
        "source_t2_sha256": _sha256(t2),
        "corrected_v4_labels_provenance_only": str(corrected_labels),
        "corrected_v4_labels_sha256": _sha256(corrected_labels),
        "native_mesh": str(paths["native_mesh"]),
        "native_mesh_sha256": _sha256(paths["native_mesh"]),
        "compatibility_mesh": str(paths["compatibility_mesh"]),
        "command": command,
    }
    paths["ready_marker"].write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = validate_scaffold(subject)
    if result["status"] != "ready":
        raise RuntimeError(f"Scaffold validation failed: {json.dumps(result)}")
    result["action"] = "built"
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--json", action="store_true")

    for name in ("build", "validate"):
        child = subparsers.add_parser(name)
        group = child.add_mutually_exclusive_group(required=True)
        group.add_argument("--subject-index", type=int)
        group.add_argument("--subject", choices=SUBJECTS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        payload = {
            "simnibs_module": SIMNIBS_MODULE,
            "task_count": len(SUBJECTS),
            "subjects": list(SUBJECTS),
            "head_model_strategy": HEAD_MODEL_STRATEGY,
            "segmentation_provenance": SEGMENTATION_PROVENANCE,
        }
        print(json.dumps(payload, indent=2) if args.json else len(SUBJECTS))
        return 0

    subject = args.subject or _subject_from_index(args.subject_index)
    payload = (
        build_scaffold(subject)
        if args.command == "build"
        else validate_scaffold(subject)
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
