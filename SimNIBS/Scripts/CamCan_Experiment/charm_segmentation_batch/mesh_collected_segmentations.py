#!/usr/bin/env python3
"""Create CHARM tetrahedral meshes from collected tissue-label NIfTI files.

This workflow intentionally runs only the meshing portion of CHARM. It does
not repeat segmentation and does not require a retained ``m2m_*`` work tree.
Each task verifies the collected map against the collector manifest, invokes
the same SimNIBS mesher and ``charm.ini`` mesh settings used by ``charm
--mesh``, reloads the written mesh, and records SHA-256 provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


MAP_SUFFIX = "_CHARM_tissue_labeling_upsampled.nii.gz"
SUBJECT_PATTERN = re.compile(r"^sub-[A-Za-z0-9][A-Za-z0-9._-]*$")
MANIFEST_FIELDS = (
    "task_id",
    "subject",
    "label_path",
    "label_sha256",
    "label_bytes",
    "mesh_path",
    "result_path",
    "status",
    "message",
)
VALIDATION_FIELDS = (
    "task_id",
    "subject",
    "status",
    "mesh_path",
    "result_path",
    "message",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_tsv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _safe_subject(subject: str) -> str:
    if not SUBJECT_PATTERN.fullmatch(subject) or "/" in subject:
        raise ValueError(f"invalid subject identifier: {subject!r}")
    return subject


def mesh_path_for_subject(mesh_root: Path, subject: str) -> Path:
    subject = _safe_subject(subject)
    return mesh_root / "subjects" / subject / "anat" / f"m2m_{subject}" / f"{subject}.msh"


def result_path_for_subject(mesh_root: Path, subject: str) -> Path:
    subject = _safe_subject(subject)
    return mesh_root / "results" / f"{subject}.json"


def build_preflight_manifest(
    *,
    collection_manifest: str | Path,
    mesh_root: str | Path,
    manifest: str | Path,
    summary: str | Path,
    expected_subjects: int,
    excluded_subjects: Sequence[str] = (),
) -> dict[str, object]:
    if expected_subjects <= 0:
        raise ValueError("expected_subjects must be positive")
    collection_path = Path(collection_manifest).expanduser().resolve(strict=True)
    output_root = Path(mesh_root).expanduser().resolve()
    collection_rows = read_tsv(collection_path)
    expected_header = {
        "subject",
        "status",
        "source_map",
        "collected_map",
        "sha256",
        "bytes",
        "message",
    }
    if collection_rows and not expected_header.issubset(collection_rows[0]):
        raise ValueError(f"unexpected collection manifest header: {collection_path}")

    exclusions = tuple(dict.fromkeys(_safe_subject(subject) for subject in excluded_subjects))
    collection_subjects = [row.get("subject", "").strip() for row in collection_rows]
    missing_exclusions = sorted(set(exclusions) - set(collection_subjects))
    if missing_exclusions:
        raise ValueError(
            "excluded subject(s) are absent from the collection manifest: "
            + ", ".join(missing_exclusions)
        )
    rows = [row for row in collection_rows if row.get("subject", "").strip() not in exclusions]

    manifest_rows: list[dict[str, object]] = []
    seen: set[str] = set()
    ready = 0
    for task_id, row in enumerate(rows):
        messages: list[str] = []
        subject = row.get("subject", "").strip()
        try:
            _safe_subject(subject)
        except ValueError as exc:
            messages.append(str(exc))
        if subject in seen:
            messages.append("duplicate subject in collection manifest")
        seen.add(subject)
        if row.get("status", "").strip() != "complete":
            messages.append(
                f"collection status is {row.get('status', '').strip()!r}, not 'complete'"
            )

        raw_map = row.get("collected_map", "").strip() or row.get(
            "source_map", ""
        ).strip()
        label_path = Path(raw_map).expanduser() if raw_map else Path(".")
        actual_hash = ""
        actual_bytes = 0
        if not raw_map:
            messages.append("collection manifest does not provide a map path")
        elif not label_path.is_absolute():
            messages.append(f"map path is not absolute: {label_path}")
        elif label_path.is_symlink():
            messages.append(f"refusing symlinked collected map: {label_path}")
        elif not label_path.is_file():
            messages.append(f"collected map is missing: {label_path}")
        else:
            label_path = label_path.resolve(strict=True)
            expected_name = f"{subject}{MAP_SUFFIX}"
            if subject and label_path.name != expected_name:
                messages.append(
                    f"unexpected map filename: {label_path.name!r} != {expected_name!r}"
                )
            actual_bytes = label_path.stat().st_size
            if actual_bytes <= 0:
                messages.append("collected map is empty")
            actual_hash = sha256_file(label_path)
            recorded_hash = row.get("sha256", "").strip()
            if not recorded_hash or actual_hash != recorded_hash:
                messages.append(
                    f"map SHA-256 differs from collection manifest: "
                    f"{actual_hash} != {recorded_hash}"
                )
            recorded_bytes = row.get("bytes", "").strip()
            if recorded_bytes:
                try:
                    if int(recorded_bytes) != actual_bytes:
                        messages.append(
                            f"map size differs from collection manifest: "
                            f"{actual_bytes} != {recorded_bytes}"
                        )
                except ValueError:
                    messages.append(f"invalid recorded map size: {recorded_bytes!r}")

        if SUBJECT_PATTERN.fullmatch(subject):
            mesh_path = mesh_path_for_subject(output_root, subject)
            result_path = result_path_for_subject(output_root, subject)
        else:
            mesh_path = output_root / "invalid" / f"task-{task_id}.msh"
            result_path = output_root / "results" / f"invalid-task-{task_id}.json"
        status = "ready" if not messages else "blocked"
        ready += status == "ready"
        manifest_rows.append(
            {
                "task_id": task_id,
                "subject": subject,
                "label_path": str(label_path),
                "label_sha256": actual_hash,
                "label_bytes": actual_bytes,
                "mesh_path": str(mesh_path),
                "result_path": str(result_path),
                "status": status,
                "message": "ready" if not messages else "; ".join(messages),
            }
        )

    if len(rows) != expected_subjects:
        count_message = (
            f"collection contains {len(rows)} row(s), expected {expected_subjects}"
        )
        for row in manifest_rows:
            if row["status"] == "ready":
                row["status"] = "blocked"
                row["message"] = count_message
        ready = 0

    manifest_path = Path(manifest).expanduser().resolve()
    summary_path = Path(summary).expanduser().resolve()
    write_tsv(manifest_path, MANIFEST_FIELDS, manifest_rows)
    blocked = len(manifest_rows) - ready
    payload = {
        "status": "ready" if ready == expected_subjects else "blocked",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "collection_manifest": str(collection_path),
        "collection_rows_total": len(collection_rows),
        "excluded_subjects": list(exclusions),
        "excluded_subjects_count": len(exclusions),
        "mesh_root": str(output_root),
        "manifest": str(manifest_path),
        "subjects_expected": expected_subjects,
        "subjects_found": len(rows),
        "ready": ready,
        "blocked": blocked,
        "mesh_outputs_expected": expected_subjects,
        "segmentation_rerun": False,
        "simulation_requested": False,
    }
    write_json_atomic(summary_path, payload)
    return payload


def _load_meshing_api() -> dict[str, Any]:
    import nibabel as nib
    import numpy as np
    import simnibs
    from simnibs import SIMNIBSDIR
    from simnibs.mesh_tools import mesh_io
    from simnibs.mesh_tools.meshing import create_mesh
    from simnibs.utils import settings_reader
    from simnibs.utils.transformations import crop_vol

    return {
        "nib": nib,
        "np": np,
        "simnibs": simnibs,
        "SIMNIBSDIR": SIMNIBSDIR,
        "mesh_io": mesh_io,
        "create_mesh": create_mesh,
        "settings_reader": settings_reader,
        "crop_vol": crop_vol,
    }


def _mesh_settings(
    settings: dict[str, Any],
    supported_parameters: set[str] | None = None,
) -> dict[str, Any]:
    """Build CHARM meshing arguments for the loaded SimNIBS API.

    SimNIBS 4.0.1 predates ``apply_cream``, ``mmg_noinsert``,
    ``num_threads``, and the newer debug arguments.  Later releases expose
    those arguments and store the first two in ``charm.ini``.  Keep the
    settings shared by every supported CHARM release, and enable
    version-specific arguments only when both the installed configuration and
    ``create_mesh`` signature support them.

    ``supported_parameters=None`` represents a callable accepting arbitrary
    keyword arguments (primarily useful for wrappers and tests).
    """
    mesh = settings["mesh"]
    skin_facet_size = mesh["skin_facet_size"] or None
    skin_tag = mesh["skin_tag"] or None
    hierarchy = mesh["hierarchy"] or None
    options = {
        "elem_sizes": mesh["elem_sizes"],
        "smooth_size_field": mesh["smooth_size_field"],
        "skin_facet_size": skin_facet_size,
        "facet_distances": mesh["facet_distances"],
        "optimize": mesh["optimize"],
        "remove_spikes": mesh["remove_spikes"],
        "skin_tag": skin_tag,
        "hierarchy": hierarchy,
        "smooth_steps": mesh["smooth_steps"],
        "skin_care": mesh["skin_care"],
    }

    def supports(name: str) -> bool:
        return supported_parameters is None or name in supported_parameters

    unsupported_common = sorted(name for name in options if not supports(name))
    if unsupported_common:
        raise RuntimeError(
            "SimNIBS create_mesh does not support required CHARM options: "
            + ", ".join(unsupported_common)
        )

    for name in ("apply_cream", "mmg_noinsert"):
        if name in mesh and supports(name):
            options[name] = mesh[name]

    if supports("debug_path"):
        options["debug_path"] = None
        if supports("debug"):
            options["debug"] = False
    elif supports("DEBUG_FN"):
        # SimNIBS 4.0.1 uses this legacy name. CHARM passes None when debug is
        # disabled, which is also the mode used by this batch workflow.
        options["DEBUG_FN"] = None

    if supports("num_threads"):
        options["num_threads"] = max(
            1, int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
        )
    return options


def _validate_mesh_object(mesh: Any) -> tuple[int, list[int]]:
    element_types = mesh.elm.elm_type
    tetrahedra = int((element_types == 4).sum())
    if tetrahedra <= 0:
        raise ValueError("mesh contains no tetrahedral elements")
    tissue_tags = sorted({int(value) for value in mesh.elm.tag1[element_types == 4]})
    if not tissue_tags:
        raise ValueError("mesh contains no tetrahedral tissue tags")
    return tetrahedra, tissue_tags


def _copy_mesh_atomic(source: Path, destination: Path, expected_hash: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        copied_hash = sha256_file(temporary)
        if copied_hash != expected_hash:
            raise IOError(
                f"mesh copy SHA-256 mismatch: {copied_hash} != {expected_hash}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _completed_result_is_current(
    result_path: Path,
    *,
    subject: str,
    label_path: Path,
    expected_label_hash: str,
    mesh_path: Path,
) -> dict[str, object] | None:
    if not result_path.is_file() or not mesh_path.is_file():
        return None
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            payload.get("status") != "complete"
            or payload.get("subject") != subject
            or payload.get("label_sha256_after") != expected_label_hash
            or payload.get("mesh_path") != str(mesh_path)
            or mesh_path.stat().st_size != payload.get("mesh_bytes")
            or sha256_file(label_path) != expected_label_hash
            or sha256_file(mesh_path) != payload.get("mesh_sha256")
        ):
            return None
        return payload
    except (OSError, ValueError, TypeError):
        return None


def run_mesh_task(
    *,
    manifest: str | Path,
    task_index: int,
    settings_path: str | Path | None = None,
) -> dict[str, object]:
    rows = read_tsv(manifest)
    if task_index < 0 or task_index >= len(rows):
        raise IndexError(
            f"task index {task_index} is outside manifest range 0..{len(rows) - 1}"
        )
    row = rows[task_index]
    if row.get("status") != "ready":
        raise ValueError(f"task {task_index} is not ready: {row.get('message')}")
    if int(row["task_id"]) != task_index:
        raise ValueError(
            f"manifest task_id {row['task_id']} does not match index {task_index}"
        )

    subject = _safe_subject(row["subject"])
    label_path = Path(row["label_path"]).expanduser().resolve(strict=True)
    expected_label_hash = row["label_sha256"]
    mesh_path = Path(row["mesh_path"]).expanduser().resolve()
    result_path = Path(row["result_path"]).expanduser().resolve()
    expected_mesh_path = mesh_path_for_subject(mesh_path.parents[4], subject)
    if mesh_path != expected_mesh_path:
        raise ValueError(f"unexpected mesh output path: {mesh_path}")
    if label_path.is_symlink() or mesh_path.is_symlink():
        raise ValueError("refusing symlinked label or mesh path")
    if label_path.name != f"{subject}{MAP_SUFFIX}":
        raise ValueError(f"collected map does not match subject identity: {label_path}")

    previous = _completed_result_is_current(
        result_path,
        subject=subject,
        label_path=label_path,
        expected_label_hash=expected_label_hash,
        mesh_path=mesh_path,
    )
    if previous is not None:
        print(
            json.dumps(
                {
                    "event": "collected_charm_mesh_reused",
                    "subject": subject,
                    "mesh_path": str(mesh_path),
                    "mesh_sha256": previous["mesh_sha256"],
                }
            ),
            flush=True,
        )
        return previous

    label_hash_before = sha256_file(label_path)
    if label_hash_before != expected_label_hash:
        raise ValueError(
            f"collected map SHA-256 changed: {label_hash_before} != {expected_label_hash}"
        )

    api = _load_meshing_api()
    nib = api["nib"]
    np = api["np"]
    mesh_io = api["mesh_io"]
    create_mesh = api["create_mesh"]
    crop_vol = api["crop_vol"]
    settings_reader = api["settings_reader"]
    resolved_settings = (
        Path(settings_path).expanduser().resolve(strict=True)
        if settings_path
        else Path(api["SIMNIBSDIR"]) / "charm.ini"
    )
    settings = settings_reader.read_ini(str(resolved_settings))
    signature = inspect.signature(create_mesh)
    supports_arbitrary_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    supported = None if supports_arbitrary_keywords else set(signature.parameters)
    create_kwargs = _mesh_settings(settings, supported)

    label_image = nib.load(str(label_path))
    if len(label_image.shape) != 3:
        raise ValueError(f"expected a 3D label image, got shape {label_image.shape}")
    label_buffer = np.round(np.asanyarray(label_image.dataobj)).astype(np.uint16)
    if not np.any(label_buffer > 0):
        raise ValueError("collected label image contains no nonzero tissue voxels")
    input_tissue_tags = sorted(int(value) for value in np.unique(label_buffer) if value)
    label_buffer, label_affine, _ = crop_vol(
        label_buffer,
        label_image.affine,
        label_buffer > 0,
        thickness_boundary=5,
    )

    mesh_root = mesh_path.parents[4]
    staging_root = mesh_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    task_stage = Path(tempfile.mkdtemp(prefix=f"{subject}-", dir=staging_root))
    staged_mesh = task_stage / f"{subject}.msh"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(
        json.dumps(
            {
                "event": "collected_charm_mesh_start",
                "subject": subject,
                "label_path": str(label_path),
                "label_sha256": label_hash_before,
                "settings_path": str(resolved_settings),
                "mesh_path": str(mesh_path),
                "input_tissue_tags": input_tissue_tags,
            }
        ),
        flush=True,
    )
    try:
        final_mesh = create_mesh(
            label_buffer,
            label_affine,
            **create_kwargs,
        )
        tetrahedra, tissue_tags = _validate_mesh_object(final_mesh)
        mesh_io.write_msh(final_mesh, str(staged_mesh))
        if not staged_mesh.is_file() or staged_mesh.stat().st_size <= 0:
            raise IOError(f"mesher did not write a non-empty mesh: {staged_mesh}")
        reloaded = mesh_io.read_msh(str(staged_mesh))
        loaded_tetrahedra, loaded_tissue_tags = _validate_mesh_object(reloaded)
        if loaded_tetrahedra != tetrahedra or loaded_tissue_tags != tissue_tags:
            raise ValueError("reloaded mesh metadata differs from in-memory mesh")
        staged_hash = sha256_file(staged_mesh)
        _copy_mesh_atomic(staged_mesh, mesh_path, staged_hash)
        final_hash = sha256_file(mesh_path)
        if final_hash != staged_hash:
            raise IOError(f"installed mesh SHA-256 changed: {final_hash} != {staged_hash}")
    finally:
        shutil.rmtree(task_stage, ignore_errors=True)

    label_hash_after = sha256_file(label_path)
    if label_hash_after != label_hash_before:
        mesh_path.unlink(missing_ok=True)
        raise RuntimeError(
            "CRITICAL: meshing changed the collected segmentation map: "
            f"{label_hash_after} != {label_hash_before}"
        )

    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "task_index": task_index,
        "subject": subject,
        "label_path": str(label_path),
        "label_bytes": label_path.stat().st_size,
        "label_sha256_before": label_hash_before,
        "label_sha256_after": label_hash_after,
        "input_tissue_tags": input_tissue_tags,
        "mesh_path": str(mesh_path),
        "mesh_bytes": mesh_path.stat().st_size,
        "mesh_sha256": final_hash,
        "tetrahedra": tetrahedra,
        "tissue_tags": tissue_tags,
        "settings_path": str(resolved_settings),
        "simnibs_version": getattr(api["simnibs"], "__version__", "unknown"),
        "meshing_mode": "direct_charm_create_mesh",
        "segmentation_rerun": False,
        "surfaces_requested": False,
        "simulation_requested": False,
        "started_at": started_at,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json_atomic(result_path, payload)
    print(
        json.dumps({"event": "collected_charm_mesh_complete", **payload}),
        flush=True,
    )
    return payload


def validate_results(
    *,
    manifest: str | Path,
    summary: str | Path,
    verify_hashes: bool = True,
) -> dict[str, object]:
    rows = read_tsv(manifest)
    validation_rows: list[dict[str, object]] = []
    complete = 0
    for row in rows:
        messages: list[str] = []
        result_path = Path(row["result_path"])
        mesh_path = Path(row["mesh_path"])
        result: dict[str, object] = {}
        if row.get("status") != "ready":
            messages.append(f"manifest row is {row.get('status')}: {row.get('message')}")
        if not result_path.is_file():
            messages.append(f"result is missing: {result_path}")
        else:
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if result.get("status") != "complete":
                    messages.append(f"result status is {result.get('status')!r}")
                if result.get("subject") != row["subject"]:
                    messages.append("result subject differs from manifest")
                if result.get("label_sha256_after") != row["label_sha256"]:
                    messages.append("result label hash differs from manifest")
                if result.get("mesh_path") != str(mesh_path):
                    messages.append("result mesh path differs from manifest")
                if not mesh_path.is_file() or mesh_path.stat().st_size <= 0:
                    messages.append(f"mesh is missing or empty: {mesh_path}")
                elif mesh_path.stat().st_size != result.get("mesh_bytes"):
                    messages.append("mesh size differs from task result")
                elif verify_hashes and sha256_file(mesh_path) != result.get("mesh_sha256"):
                    messages.append("mesh SHA-256 differs from task result")
                if verify_hashes:
                    label_path = Path(row["label_path"])
                    if sha256_file(label_path) != row["label_sha256"]:
                        messages.append("collected map SHA-256 changed")
            except (OSError, ValueError, TypeError) as exc:
                messages.append(str(exc))
        status = "complete" if not messages else "incomplete"
        complete += status == "complete"
        validation_rows.append(
            {
                "task_id": row["task_id"],
                "subject": row["subject"],
                "status": status,
                "mesh_path": str(mesh_path),
                "result_path": str(result_path),
                "message": "complete" if not messages else "; ".join(messages),
            }
        )

    summary_path = Path(summary).expanduser().resolve()
    write_tsv(summary_path, VALIDATION_FIELDS, validation_rows)
    payload = {
        "status": "complete" if complete == len(rows) else "incomplete",
        "tasks": len(rows),
        "complete": complete,
        "incomplete": len(rows) - complete,
        "hashes_verified": verify_hashes,
        "summary": str(summary_path),
    }
    write_json_atomic(summary_path.with_suffix(".json"), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--collection-manifest", required=True)
    preflight.add_argument("--mesh-root", required=True)
    preflight.add_argument("--manifest", required=True)
    preflight.add_argument("--summary", required=True)
    preflight.add_argument("--expected-subjects", type=int, required=True)
    preflight.add_argument(
        "--exclude-subject",
        action="append",
        default=[],
        help="Explicitly exclude this collection subject; repeat for multiple subjects.",
    )

    run_task = subparsers.add_parser("run-task")
    run_task.add_argument("--manifest", required=True)
    run_task.add_argument("--task-index", type=int, required=True)
    run_task.add_argument("--settings")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--summary", required=True)
    validate.add_argument("--skip-hashes", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "preflight":
            payload = build_preflight_manifest(
                collection_manifest=args.collection_manifest,
                mesh_root=args.mesh_root,
                manifest=args.manifest,
                summary=args.summary,
                expected_subjects=args.expected_subjects,
                excluded_subjects=args.exclude_subject,
            )
        elif args.command == "run-task":
            payload = run_mesh_task(
                manifest=args.manifest,
                task_index=args.task_index,
                settings_path=args.settings,
            )
        else:
            payload = validate_results(
                manifest=args.manifest,
                summary=args.summary,
                verify_hashes=not args.skip_hashes,
            )
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0 if payload.get("status") in {"ready", "complete"} else 1
    except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
