#!/usr/bin/env python3
"""Correct collected CHARM labels without modifying the reviewed source maps.

The correction mirrors the cumulative-mask hierarchy supplied by the study
supervisor:

* close the cumulative WM+GM+CSF envelope with a 5-voxel radius;
* close the cumulative whole-head tissue envelope with a 10-voxel radius;
* remove disconnected WM and cumulative GM/brain components; and
* reconstruct one non-overlapping CHARM label image using the supplied tissue
  priority order.

Every output is written atomically to a separate root and accompanied by
SHA-256 provenance and voxel-transition metrics. Source maps are hash-checked
before and after processing and are never opened for writing.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import nibabel as nib
import numpy as np
from scipy import ndimage


MAP_SUFFIX = "_CHARM_tissue_labeling_upsampled.nii.gz"
SUBJECT_PATTERN = re.compile(r"^sub-[A-Za-z0-9][A-Za-z0-9._-]*$")
ALGORITHM = "cumulative-charm-cleanup-v1"
CSF_COMPONENT_ALGORITHM = "cumulative-charm-cleanup-v2-csf-components"
CSF_EXCLUDES_BLOOD_ALGORITHM = "cumulative-charm-cleanup-v3-csf-excludes-blood"
KNOWN_LABELS = frozenset((0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11))
REQUIRED_LABELS = frozenset((1, 2, 3, 5))
MANIFEST_FIELDS = (
    "task_id",
    "subject",
    "source_map",
    "source_sha256",
    "source_bytes",
    "corrected_map",
    "result_path",
    "status",
    "message",
)
VALIDATION_FIELDS = (
    "task_id",
    "subject",
    "status",
    "source_map",
    "corrected_map",
    "result_path",
    "changed_voxels",
    "message",
)
COLLECTION_FIELDS = (
    "subject",
    "status",
    "source_map",
    "collected_map",
    "sha256",
    "bytes",
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


def subject_from_map(path: Path) -> str:
    if not path.name.endswith(MAP_SUFFIX):
        raise ValueError(f"unexpected CHARM map filename: {path.name}")
    return _safe_subject(path.name[: -len(MAP_SUFFIX)])


def correction_parameters(
    *,
    csf_radius: int,
    skin_radius: int,
    include_blood_in_csf: bool,
    csf_component_policy: str,
    component_policy: str,
    wm_min_component_voxels: int,
    gm_min_component_voxels: int,
    connectivity: int,
) -> dict[str, object]:
    if csf_radius < 0 or skin_radius < 0:
        raise ValueError("CSF and skin radii must be non-negative")
    if csf_component_policy not in {"none", "largest"}:
        raise ValueError("csf_component_policy must be 'none' or 'largest'")
    if component_policy not in {"largest", "min-size"}:
        raise ValueError("component_policy must be 'largest' or 'min-size'")
    if connectivity not in {6, 18, 26}:
        raise ValueError("connectivity must be 6, 18, or 26")
    if wm_min_component_voxels < 0 or gm_min_component_voxels < 0:
        raise ValueError("component thresholds must be non-negative")
    if component_policy == "min-size" and (
        wm_min_component_voxels <= 0 or gm_min_component_voxels <= 0
    ):
        raise ValueError(
            "min-size policy requires positive WM and GM component thresholds"
        )
    if include_blood_in_csf:
        algorithm = (
            CSF_COMPONENT_ALGORITHM
            if csf_component_policy != "none"
            else ALGORITHM
        )
    else:
        algorithm = CSF_EXCLUDES_BLOOD_ALGORITHM
    parameters: dict[str, object] = {
        "algorithm": algorithm,
        "csf_closing_radius_voxels": int(csf_radius),
        "skin_closing_radius_voxels": int(skin_radius),
        "component_policy": component_policy,
        "wm_min_component_voxels": int(wm_min_component_voxels),
        "gm_min_component_voxels": int(gm_min_component_voxels),
        "connectivity": int(connectivity),
    }
    if csf_component_policy != "none":
        parameters["csf_component_policy"] = csf_component_policy
    if not include_blood_in_csf:
        parameters["csf_source_labels"] = [1, 2, 3]
        parameters["blood_restored_after_csf"] = True
    return parameters


def _connectivity_structure(connectivity: int) -> np.ndarray:
    rank = {6: 1, 18: 2, 26: 3}[connectivity]
    return ndimage.generate_binary_structure(3, rank)


def _filter_components(
    mask: np.ndarray,
    *,
    policy: str,
    min_voxels: int,
    connectivity: int,
) -> tuple[np.ndarray, dict[str, object]]:
    labeled, component_count = ndimage.label(
        mask, structure=_connectivity_structure(connectivity)
    )
    counts = np.bincount(labeled.ravel())
    sizes = counts[1:].astype(np.int64, copy=False)
    if component_count == 0:
        raise ValueError("required tissue mask has no connected components")

    if policy == "largest":
        keep_ids = np.array([int(np.argmax(sizes)) + 1], dtype=np.int64)
    else:
        keep_ids = np.flatnonzero(sizes >= min_voxels).astype(np.int64) + 1
        if keep_ids.size == 0:
            raise ValueError(
                f"component threshold {min_voxels} would remove the entire tissue mask"
            )
    filtered = np.isin(labeled, keep_ids)
    kept_sizes = sizes[keep_ids - 1]
    return filtered, {
        "components_before": int(component_count),
        "components_kept": int(keep_ids.size),
        "components_removed": int(component_count - keep_ids.size),
        "largest_component_voxels": int(sizes.max()),
        "smallest_kept_component_voxels": int(kept_sizes.min()),
        "voxels_before": int(mask.sum()),
        "voxels_after": int(filtered.sum()),
        "voxels_removed": int(mask.sum() - filtered.sum()),
    }


def _ball_closing(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary closing with an isotropic Euclidean ball measured in voxels.

    Euclidean distance transforms avoid the prohibitive cost of convolving a
    21x21x21 structuring element over a full CHARM volume for radius 10.
    Padding also gives explicit, symmetric behavior at volume boundaries.
    """

    if radius == 0:
        return np.array(mask, dtype=bool, copy=True)
    padding = radius + 1
    padded = np.pad(mask.astype(bool, copy=False), padding, constant_values=False)
    dilated = ndimage.distance_transform_edt(~padded) <= float(radius)
    closed = ndimage.distance_transform_edt(dilated) > float(radius)
    crop = tuple(slice(padding, -padding) for _ in range(3))
    return np.asarray(closed[crop], dtype=bool)


def _label_counts(labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def _transition_counts(before: np.ndarray, after: np.ndarray) -> dict[str, int]:
    changed = before != after
    if not np.any(changed):
        return {}
    pairs = np.stack((before[changed], after[changed]), axis=1)
    values, counts = np.unique(pairs, axis=0, return_counts=True)
    return {
        f"{int(pair[0])}->{int(pair[1])}": int(count)
        for pair, count in zip(values, counts)
    }


def correct_label_array(
    labels: np.ndarray,
    *,
    csf_radius: int = 5,
    skin_radius: int = 10,
    include_blood_in_csf: bool = False,
    csf_component_policy: str = "none",
    component_policy: str = "largest",
    wm_min_component_voxels: int = 0,
    gm_min_component_voxels: int = 0,
    connectivity: int = 26,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return a corrected single-label CHARM array and audit metrics."""

    parameters = correction_parameters(
        csf_radius=csf_radius,
        skin_radius=skin_radius,
        include_blood_in_csf=include_blood_in_csf,
        csf_component_policy=csf_component_policy,
        component_policy=component_policy,
        wm_min_component_voxels=wm_min_component_voxels,
        gm_min_component_voxels=gm_min_component_voxels,
        connectivity=connectivity,
    )
    array = np.asarray(labels)
    if array.ndim != 3:
        raise ValueError(f"expected a 3D CHARM map, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("CHARM map contains non-finite values")
    rounded = np.rint(array)
    if not np.array_equal(array, rounded):
        raise ValueError("CHARM map contains non-integer label values")
    original = rounded.astype(np.int32, copy=False)
    if np.any(original < 0) or np.any(original > np.iinfo(np.uint16).max):
        raise ValueError("CHARM labels fall outside uint16 range")
    labels_present = {int(value) for value in np.unique(original)}
    missing = sorted(REQUIRED_LABELS - labels_present)
    if missing:
        raise ValueError(f"CHARM map is missing required label(s): {missing}")

    wm = original == 1
    gm_envelope = np.isin(original, (1, 2))
    csf_source_labels = (1, 2, 3, 9) if include_blood_in_csf else (1, 2, 3)
    csf_envelope = np.isin(original, csf_source_labels)
    head_envelope = (original >= 1) & (original <= 10)

    wm_clean, wm_metrics = _filter_components(
        wm,
        policy=component_policy,
        min_voxels=wm_min_component_voxels,
        connectivity=connectivity,
    )
    gm_clean, gm_metrics = _filter_components(
        gm_envelope,
        policy=component_policy,
        min_voxels=gm_min_component_voxels,
        connectivity=connectivity,
    )
    csf_for_closing = csf_envelope
    csf_component_metrics: dict[str, object] | None = None
    if csf_component_policy == "largest":
        csf_for_closing, csf_component_metrics = _filter_components(
            csf_envelope,
            policy="largest",
            min_voxels=0,
            connectivity=connectivity,
        )
    csf_closed = _ball_closing(csf_for_closing, csf_radius)
    head_closed = _ball_closing(head_envelope, skin_radius)

    # Reconstruct using the supplied low-to-high priority order. Masks are
    # cumulative for skin, compact bone, CSF, GM, and WM so edits cannot
    # overlap in the final single-label volume.
    corrected = np.zeros(original.shape, dtype=np.uint16)
    corrected[head_closed] = 5
    corrected[original == 10] = 10
    corrected[original == 6] = 6
    corrected[np.isin(original, (1, 2, 3, 7, 8))] = 7
    corrected[original == 8] = 8
    corrected[csf_closed] = 3
    corrected[original == 9] = 9
    corrected[gm_clean] = 2
    corrected[wm_clean] = 1
    corrected[original == 11] = 11

    # Preserve unrecognized/extended labels byte-for-label rather than
    # silently discarding them as the supplied MATLAB recombination does.
    known_mask = np.isin(original, tuple(KNOWN_LABELS))
    corrected[~known_mask] = original[~known_mask].astype(np.uint16)

    changed = original != corrected
    metrics: dict[str, object] = {
        "parameters": parameters,
        "shape": [int(value) for value in original.shape],
        "labels_before": sorted(labels_present),
        "labels_after": sorted(int(value) for value in np.unique(corrected)),
        "label_voxels_before": _label_counts(original),
        "label_voxels_after": _label_counts(corrected),
        "changed_voxels": int(changed.sum()),
        "changed_fraction": float(changed.mean()),
        "transitions": _transition_counts(original, corrected),
        "wm_components": wm_metrics,
        "gm_cumulative_components": gm_metrics,
        "csf_cumulative_voxels_before": int(csf_envelope.sum()),
        "csf_cumulative_voxels_after": int(csf_closed.sum()),
        "skin_cumulative_voxels_before": int(head_envelope.sum()),
        "skin_cumulative_voxels_after": int(head_closed.sum()),
        "unknown_labels_preserved": sorted(labels_present - KNOWN_LABELS),
    }
    if csf_component_metrics is not None:
        metrics["csf_cumulative_voxels_after_component_filter"] = int(
            csf_for_closing.sum()
        )
        metrics["csf_cumulative_components"] = csf_component_metrics
    return corrected, metrics


def build_preflight_manifest(
    *,
    maps_root: str | Path,
    output_root: str | Path,
    manifest: str | Path,
    summary: str | Path,
    expected_subjects: int,
) -> dict[str, object]:
    if expected_subjects <= 0:
        raise ValueError("expected_subjects must be positive")
    source_root = Path(maps_root).expanduser().resolve(strict=True)
    destination_root = Path(output_root).expanduser().resolve()
    corrected_root = destination_root / "maps"
    results_root = destination_root / "results"
    if corrected_root == source_root:
        raise ValueError("corrected maps root must differ from the source maps root")

    candidates = sorted(source_root.glob(f"sub-*{MAP_SUFFIX}"))
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    ready = 0
    for task_id, source in enumerate(candidates):
        messages: list[str] = []
        subject = ""
        source_hash = ""
        source_bytes = 0
        try:
            subject = subject_from_map(source)
        except ValueError as exc:
            messages.append(str(exc))
        if subject in seen:
            messages.append("duplicate subject map")
        seen.add(subject)
        if source.is_symlink():
            messages.append(f"refusing symlinked source map: {source}")
        elif not source.is_file():
            messages.append(f"source map is missing: {source}")
        else:
            source_bytes = source.stat().st_size
            if source_bytes <= 0:
                messages.append("source map is empty")
            source_hash = sha256_file(source)
        corrected = corrected_root / source.name
        result = results_root / f"{subject or f'invalid-{task_id}'}.json"
        status = "ready" if not messages else "blocked"
        ready += status == "ready"
        rows.append(
            {
                "task_id": task_id,
                "subject": subject,
                "source_map": str(source.resolve()),
                "source_sha256": source_hash,
                "source_bytes": source_bytes,
                "corrected_map": str(corrected),
                "result_path": str(result),
                "status": status,
                "message": "ready" if not messages else "; ".join(messages),
            }
        )

    if len(rows) != expected_subjects:
        message = f"discovered {len(rows)} map(s), expected {expected_subjects}"
        for row in rows:
            if row["status"] == "ready":
                row["status"] = "blocked"
                row["message"] = message
        ready = 0

    manifest_path = Path(manifest).expanduser().resolve()
    summary_path = Path(summary).expanduser().resolve()
    write_tsv(manifest_path, MANIFEST_FIELDS, rows)
    payload: dict[str, object] = {
        "status": "ready" if ready == expected_subjects else "blocked",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_maps_root": str(source_root),
        "output_root": str(destination_root),
        "corrected_maps_root": str(corrected_root),
        "results_root": str(results_root),
        "manifest": str(manifest_path),
        "subjects_expected": expected_subjects,
        "subjects_found": len(rows),
        "ready": ready,
        "blocked": len(rows) - ready,
        "source_maps_modified": False,
        "corrected_maps_expected": expected_subjects,
    }
    write_json_atomic(summary_path, payload)
    return payload


def _row_for_task(manifest: str | Path, task_index: int) -> dict[str, str]:
    rows = read_tsv(manifest)
    if task_index < 0 or task_index >= len(rows):
        raise IndexError(
            f"task index {task_index} is outside manifest range 0..{len(rows) - 1}"
        )
    row = rows[task_index]
    if int(row["task_id"]) != task_index:
        raise ValueError("manifest task index does not match row position")
    if row.get("status") != "ready":
        raise ValueError(f"task is blocked: {row.get('message', '')}")
    return row


def _current_result(
    *,
    row: dict[str, str],
    parameters: dict[str, object],
) -> dict[str, object] | None:
    result_path = Path(row["result_path"])
    corrected_path = Path(row["corrected_map"])
    if not result_path.is_file() or not corrected_path.is_file():
        return None
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            payload.get("status") != "complete"
            or payload.get("subject") != row["subject"]
            or payload.get("source_sha256_after") != row["source_sha256"]
            or payload.get("parameters") != parameters
            or payload.get("corrected_map") != str(corrected_path)
            or payload.get("corrected_bytes") != corrected_path.stat().st_size
            or payload.get("corrected_sha256") != sha256_file(corrected_path)
        ):
            return None
        return payload
    except (OSError, ValueError, TypeError):
        return None


def _save_corrected_image(
    source_image: nib.spatialimages.SpatialImage,
    corrected: np.ndarray,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name[:-7]}.tmp-{os.getpid()}.nii.gz"
        if destination.name.endswith(".nii.gz")
        else f".{destination.name}.tmp-{os.getpid()}"
    )
    header = source_image.header.copy()
    header.set_data_dtype(np.uint16)
    output = nib.Nifti1Image(corrected.astype(np.uint16, copy=False), source_image.affine, header)
    qform, qcode = source_image.get_qform(coded=True)
    sform, scode = source_image.get_sform(coded=True)
    if qform is not None:
        output.set_qform(qform, int(qcode))
    if sform is not None:
        output.set_sform(sform, int(scode))
    try:
        nib.save(output, str(temporary))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def run_task(
    *,
    manifest: str | Path,
    task_index: int,
    csf_radius: int = 5,
    skin_radius: int = 10,
    include_blood_in_csf: bool = False,
    csf_component_policy: str = "none",
    component_policy: str = "largest",
    wm_min_component_voxels: int = 0,
    gm_min_component_voxels: int = 0,
    connectivity: int = 26,
) -> dict[str, object]:
    row = _row_for_task(manifest, task_index)
    parameters = correction_parameters(
        csf_radius=csf_radius,
        skin_radius=skin_radius,
        include_blood_in_csf=include_blood_in_csf,
        csf_component_policy=csf_component_policy,
        component_policy=component_policy,
        wm_min_component_voxels=wm_min_component_voxels,
        gm_min_component_voxels=gm_min_component_voxels,
        connectivity=connectivity,
    )
    previous = _current_result(row=row, parameters=parameters)
    if previous is not None:
        print(
            json.dumps(
                {
                    "event": "charm_cleanup_already_current",
                    "subject": row["subject"],
                    "corrected_map": row["corrected_map"],
                    "corrected_sha256": previous["corrected_sha256"],
                }
            ),
            flush=True,
        )
        return previous

    result_path = Path(row["result_path"]).resolve()
    if result_path.exists():
        raise ValueError(
            "existing cleanup result does not match the requested source/parameters; "
            "use a new versioned output root instead of overwriting it"
        )

    subject = _safe_subject(row["subject"])
    source = Path(row["source_map"]).resolve(strict=True)
    corrected_path = Path(row["corrected_map"]).resolve()
    if source.is_symlink() or corrected_path.is_symlink() or result_path.is_symlink():
        raise ValueError("refusing symlinked source or output path")
    if source.name != f"{subject}{MAP_SUFFIX}" or corrected_path.name != source.name:
        raise ValueError("manifest map path does not match subject identity")
    source_hash_before = sha256_file(source)
    if source_hash_before != row["source_sha256"]:
        raise ValueError("source map hash has changed since preflight")
    if source.stat().st_size != int(row["source_bytes"]):
        raise ValueError("source map size has changed since preflight")

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    image = nib.load(str(source))
    raw = np.asanyarray(image.dataobj)
    corrected, metrics = correct_label_array(
        raw,
        csf_radius=csf_radius,
        skin_radius=skin_radius,
        include_blood_in_csf=include_blood_in_csf,
        csf_component_policy=csf_component_policy,
        component_policy=component_policy,
        wm_min_component_voxels=wm_min_component_voxels,
        gm_min_component_voxels=gm_min_component_voxels,
        connectivity=connectivity,
    )
    _save_corrected_image(image, corrected, corrected_path)
    reloaded = nib.load(str(corrected_path))
    if reloaded.shape != image.shape or not np.allclose(reloaded.affine, image.affine):
        corrected_path.unlink(missing_ok=True)
        raise ValueError("corrected map grid or affine differs from its source")
    corrected_hash = sha256_file(corrected_path)
    source_hash_after = sha256_file(source)
    if source_hash_after != source_hash_before:
        corrected_path.unlink(missing_ok=True)
        raise RuntimeError("CRITICAL: source CHARM map changed during cleanup")

    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "task_index": task_index,
        "subject": subject,
        "source_map": str(source),
        "source_bytes": source.stat().st_size,
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "corrected_map": str(corrected_path),
        "corrected_bytes": corrected_path.stat().st_size,
        "corrected_sha256": corrected_hash,
        "parameters": parameters,
        "zooms": [float(value) for value in image.header.get_zooms()[:3]],
        "source_modified": False,
        "metrics": metrics,
        "started_at": started_at,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json_atomic(result_path, payload)
    print(
        json.dumps(
            {
                "event": "charm_cleanup_complete",
                "subject": subject,
                "corrected_map": str(corrected_path),
                "corrected_sha256": corrected_hash,
                "changed_voxels": metrics["changed_voxels"],
            }
        ),
        flush=True,
    )
    return payload


def _run_task_from_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
    return run_task(**kwargs)


def run_all(
    *,
    manifest: str | Path,
    workers: int,
    **parameters: object,
) -> dict[str, object]:
    rows = read_tsv(manifest)
    if workers <= 0:
        raise ValueError("workers must be positive")
    tasks = [
        {"manifest": str(manifest), "task_index": index, **parameters}
        for index in range(len(rows))
    ]
    completed = 0
    failed: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_task_from_kwargs, task): task["task_index"]
            for task in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            task_index = futures[future]
            try:
                future.result()
                completed += 1
            except Exception as exc:  # noqa: BLE001 - aggregate worker failures
                failed.append({"task_index": task_index, "error": str(exc)})
    return {
        "status": "complete" if not failed else "incomplete",
        "tasks": len(tasks),
        "complete": completed,
        "failed": len(failed),
        "failures": failed,
    }


def validate_results(
    *,
    manifest: str | Path,
    validation: str | Path,
    summary: str | Path,
    collection_manifest: str | Path,
    checksums: str | Path,
    skip_nifti_load: bool = False,
) -> dict[str, object]:
    manifest_path = Path(manifest).expanduser().resolve(strict=True)
    rows = read_tsv(manifest_path)
    validation_rows: list[dict[str, object]] = []
    collection_rows: list[dict[str, object]] = []
    complete = 0
    total_changed = 0
    for row in rows:
        messages: list[str] = []
        source = Path(row["source_map"])
        corrected = Path(row["corrected_map"])
        result = Path(row["result_path"])
        payload: dict[str, Any] = {}
        try:
            if row.get("status") != "ready":
                raise ValueError(f"preflight status is {row.get('status')}")
            if not result.is_file():
                raise FileNotFoundError(f"missing result marker: {result}")
            payload = json.loads(result.read_text(encoding="utf-8"))
            if payload.get("status") != "complete":
                raise ValueError("result marker is not complete")
            if payload.get("subject") != row["subject"]:
                raise ValueError("result subject differs from manifest")
            if payload.get("source_sha256_before") != row["source_sha256"] or payload.get(
                "source_sha256_after"
            ) != row["source_sha256"]:
                raise ValueError("result source hash differs from manifest")
            if payload.get("corrected_map") != str(corrected):
                raise ValueError("result corrected-map path differs from manifest")
            if not source.is_file() or sha256_file(source) != row["source_sha256"]:
                raise ValueError("source map no longer matches preflight hash")
            if not corrected.is_file() or corrected.stat().st_size <= 0:
                raise FileNotFoundError(f"missing corrected map: {corrected}")
            corrected_hash = sha256_file(corrected)
            if corrected_hash != payload.get("corrected_sha256"):
                raise ValueError("corrected map hash differs from result marker")
            if corrected.stat().st_size != payload.get("corrected_bytes"):
                raise ValueError("corrected map size differs from result marker")
            if not skip_nifti_load:
                source_image = nib.load(str(source))
                corrected_image = nib.load(str(corrected))
                if source_image.shape != corrected_image.shape or not np.allclose(
                    source_image.affine, corrected_image.affine
                ):
                    raise ValueError("corrected image grid differs from source")
                corrected_data = np.asanyarray(corrected_image.dataobj)
                if corrected_data.ndim != 3 or not np.all(np.isfinite(corrected_data)):
                    raise ValueError("corrected image data are invalid")
                if not np.array_equal(corrected_data, np.rint(corrected_data)):
                    raise ValueError("corrected image contains non-integer labels")
            complete += 1
            total_changed += int(payload.get("metrics", {}).get("changed_voxels", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            messages.append(str(exc))

        status = "complete" if not messages else "incomplete"
        changed_voxels = (
            payload.get("metrics", {}).get("changed_voxels", "") if payload else ""
        )
        validation_rows.append(
            {
                "task_id": row["task_id"],
                "subject": row["subject"],
                "status": status,
                "source_map": str(source),
                "corrected_map": str(corrected),
                "result_path": str(result),
                "changed_voxels": changed_voxels,
                "message": "ok" if not messages else "; ".join(messages),
            }
        )
        collection_rows.append(
            {
                "subject": row["subject"],
                "status": status,
                "source_map": str(source),
                "collected_map": str(corrected) if status == "complete" else "",
                "sha256": payload.get("corrected_sha256", "") if status == "complete" else "",
                "bytes": payload.get("corrected_bytes", "") if status == "complete" else "",
                "message": "corrected" if status == "complete" else "; ".join(messages),
            }
        )

    validation_path = Path(validation).expanduser().resolve()
    collection_path = Path(collection_manifest).expanduser().resolve()
    checksums_path = Path(checksums).expanduser().resolve()
    summary_path = Path(summary).expanduser().resolve()
    write_tsv(validation_path, VALIDATION_FIELDS, validation_rows)
    write_tsv(collection_path, COLLECTION_FIELDS, collection_rows)
    checksums_path.parent.mkdir(parents=True, exist_ok=True)
    checksums_path.write_text(
        "".join(
            f"{row['sha256']}  {row['collected_map']}\n"
            for row in collection_rows
            if row["status"] == "complete"
        ),
        encoding="utf-8",
    )
    incomplete = len(rows) - complete
    payload = {
        "status": "complete" if incomplete == 0 else "incomplete",
        "manifest": str(manifest_path),
        "validation": str(validation_path),
        "collection_manifest": str(collection_path),
        "checksums": str(checksums_path),
        "tasks": len(rows),
        "complete": complete,
        "incomplete": incomplete,
        "total_changed_voxels": total_changed,
        "source_maps_modified": False,
        "maps_are_byte_for_byte_copies": False,
        "nifti_load_skipped": bool(skip_nifti_load),
    }
    write_json_atomic(summary_path, payload)
    return payload


def _add_parameter_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--csf-radius", type=int, default=5)
    parser.add_argument("--skin-radius", type=int, default=10)
    parser.add_argument(
        "--include-blood-in-csf",
        action="store_true",
        help="Reproduce the legacy CSF mask that included label 9 before closing.",
    )
    parser.add_argument(
        "--csf-component-policy", choices=("none", "largest"), default="none"
    )
    parser.add_argument(
        "--component-policy", choices=("largest", "min-size"), default="largest"
    )
    parser.add_argument("--wm-min-component-voxels", type=int, default=0)
    parser.add_argument("--gm-min-component-voxels", type=int, default=0)
    parser.add_argument("--connectivity", type=int, choices=(6, 18, 26), default=26)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--maps-root", required=True)
    preflight.add_argument("--output-root", required=True)
    preflight.add_argument("--manifest", required=True)
    preflight.add_argument("--summary", required=True)
    preflight.add_argument("--expected-subjects", type=int, required=True)

    task = subparsers.add_parser("run-task")
    task.add_argument("--manifest", required=True)
    task.add_argument("--task-index", type=int, required=True)
    _add_parameter_arguments(task)

    all_tasks = subparsers.add_parser("run-all")
    all_tasks.add_argument("--manifest", required=True)
    all_tasks.add_argument("--workers", type=int, default=1)
    _add_parameter_arguments(all_tasks)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--validation", required=True)
    validate.add_argument("--summary", required=True)
    validate.add_argument("--collection-manifest", required=True)
    validate.add_argument("--checksums", required=True)
    validate.add_argument("--skip-nifti-load", action="store_true")
    return parser.parse_args(argv)


def _parameter_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "csf_radius": args.csf_radius,
        "skin_radius": args.skin_radius,
        "include_blood_in_csf": args.include_blood_in_csf,
        "csf_component_policy": args.csf_component_policy,
        "component_policy": args.component_policy,
        "wm_min_component_voxels": args.wm_min_component_voxels,
        "gm_min_component_voxels": args.gm_min_component_voxels,
        "connectivity": args.connectivity,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "preflight":
            payload = build_preflight_manifest(
                maps_root=args.maps_root,
                output_root=args.output_root,
                manifest=args.manifest,
                summary=args.summary,
                expected_subjects=args.expected_subjects,
            )
        elif args.command == "run-task":
            payload = run_task(
                manifest=args.manifest,
                task_index=args.task_index,
                **_parameter_kwargs(args),
            )
        elif args.command == "run-all":
            payload = run_all(
                manifest=args.manifest,
                workers=args.workers,
                **_parameter_kwargs(args),
            )
        else:
            payload = validate_results(
                manifest=args.manifest,
                validation=args.validation,
                summary=args.summary,
                collection_manifest=args.collection_manifest,
                checksums=args.checksums,
                skip_nifti_load=args.skip_nifti_load,
            )
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0 if payload.get("status") in {"ready", "complete"} else 1
    except (OSError, ValueError, TypeError, IndexError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
