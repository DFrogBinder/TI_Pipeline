#!/usr/bin/env python3
"""Build reusable scaffolds, independent meshes, and four-ROI FEM results.

The workflow is cohort-driven rather than hard-coded to one subject count.
Scaffolds are stored once per corrected segmentation map set and reused as a
cohort expands. Every ROI/repeat receives a physical m2m scaffold copy and an
independent direct-CHARM mesh generated from the selected corrected map.
Meshing and FEM are deliberately separate stages.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence


CAMCAN_ROOT = Path(__file__).resolve().parents[1]
if str(CAMCAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CAMCAN_ROOT))

from approved_wave import workflow as approved  # noqa: E402
from charm_segmentation_batch import mesh_collected_segmentations as direct_mesh  # noqa: E402
from simulation.validate_simulation_outputs import (  # noqa: E402
    validate_subject_outputs,
)
from utils.camcan_dataset import (  # noqa: E402
    CAMCAN_ROI_CONFIGS,
    CONFIRMED_TARGETS_SHA256,
    electrode_names_from_target_row,
    electrode_names_for_config,
    load_individualized_target_row,
    sha256_file,
    validate_individualized_target_table,
    validate_dataset_montage,
)


MAP_BASENAME = approved.MAP_BASENAME
MAP_SUFFIX = approved.MAP_SUFFIX
CAP_BASENAME = approved.CAP_BASENAME
DEFAULT_REPEATS = tuple(f"{value:02d}" for value in range(1, 11))
DEFAULT_ROIS = tuple(config.dataset_prefix for config in CAMCAN_ROI_CONFIGS)
ROI_BY_PREFIX = {config.dataset_prefix: config for config in CAMCAN_ROI_CONFIGS}

SCAFFOLD_FIELDS = (
    "task_id",
    "subject",
    "mode",
    "source_anat",
    "source_t1",
    "source_t1_sha256",
    "source_t2",
    "source_t2_sha256",
    "corrected_label",
    "corrected_label_sha256",
    "legacy_anat_dir",
    "legacy_m2m_dir",
    "legacy_result_path",
    "scaffold_anat_dir",
    "scaffold_m2m_dir",
    "result_path",
    "required_electrodes",
    "status",
    "message",
)

TASK_FIELDS = (
    "task_id",
    "roi",
    "montage_preset",
    "dataset_name",
    "repeat_id",
    "dataset_root",
    "subject",
    "source_t1_sha256",
    "source_t2_sha256",
    "anat_dir",
    "m2m_dir",
    "mesh_path",
    "corrected_label",
    "corrected_label_sha256",
    "scaffold_anat_dir",
    "scaffold_m2m_dir",
    "scaffold_result_path",
    "mesh_result_path",
    "simulation_result_path",
    "required_electrodes",
    "status",
    "message",
)

VALIDATION_FIELDS = (
    "task_id",
    "stage",
    "roi",
    "repeat_id",
    "subject",
    "status",
    "result_path",
    "message",
)


def _load_json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _safe_cohort_id(value: str) -> str:
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in value):
        raise ValueError(f"invalid cohort id: {value!r}")
    return value


def _resolve_config_relative(config_path: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    return candidate.resolve(strict=True)


def _read_collection(path: Path) -> dict[str, dict[str, str]]:
    rows = approved.read_tsv(path)
    required = {
        "subject",
        "status",
        "source_map",
        "collected_map",
        "sha256",
        "bytes",
        "message",
    }
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"unexpected corrected-map manifest header: {path}")
    by_subject: dict[str, dict[str, str]] = {}
    for row in rows:
        subject = row.get("subject", "").strip()
        if subject in by_subject:
            raise ValueError(f"duplicate corrected-map subject: {subject}")
        by_subject[subject] = row
    return by_subject


def _required_electrodes(
    rois: Sequence[str],
    repeats: Sequence[str],
    subjects: Sequence[str],
    targets_csv: Path,
    individualized_targets: Mapping[tuple[str, str], Mapping[str, str]] | None,
) -> tuple[dict[tuple[str, str], tuple[str, ...]], tuple[str, ...]]:
    by_subject_roi: dict[tuple[str, str], tuple[str, ...]] = {}
    combined: set[str] = set()
    for roi in rois:
        try:
            config = ROI_BY_PREFIX[roi]
        except KeyError as exc:
            raise ValueError(f"unsupported ROI: {roi}") from exc
        dataset_names = [f"{roi}_Data_{repeat}" for repeat in repeats]
        validated = validate_dataset_montage(dataset_names, config.montage_preset)
        if individualized_targets is None:
            names = tuple(electrode_names_for_config(validated, targets_csv))
            for subject in subjects:
                by_subject_roi[(subject, roi)] = names
            combined.update(names)
        else:
            for subject in subjects:
                row = individualized_targets[(subject, roi)]
                names = tuple(electrode_names_from_target_row(row))
                by_subject_roi[(subject, roi)] = names
                combined.update(names)
    return by_subject_roi, tuple(sorted(combined))


def _legacy_scaffolds(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    rows = approved.read_tsv(path)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        subject = row.get("subject", "")
        if subject and subject not in result:
            result[subject] = row
    return result


def _legacy_scaffold_is_usable(
    row: Mapping[str, str] | None,
    subject: str,
    required_electrodes: Sequence[str],
) -> bool:
    if row is None:
        return False
    marker = approved._load_json(Path(row.get("result_path", "")))
    m2m = Path(row.get("m2m_dir", ""))
    anat = Path(row.get("anat_dir", ""))
    cap = m2m / "eeg_positions" / CAP_BASENAME
    if marker is None or marker.get("status") != "complete":
        return False
    try:
        return bool(
            marker.get("subject") == subject
            and marker.get("eeg_cap_sha256") == sha256_file(cap)
            and anat.is_dir()
            and m2m.is_dir()
            and not anat.is_symlink()
            and not m2m.is_symlink()
            and set(required_electrodes).issubset(approved._cap_names(cap))
        )
    except (OSError, UnicodeError, csv.Error):
        return False


def scaffold_result_is_current(
    row: Mapping[str, str],
) -> dict[str, object] | None:
    marker = approved._load_json(Path(row["result_path"]))
    if marker is None or marker.get("status") != "complete":
        return None
    subject = row["subject"]
    anat = Path(row["scaffold_anat_dir"])
    m2m = Path(row["scaffold_m2m_dir"])
    label = m2m / "label_prep" / MAP_BASENAME
    cap = m2m / "eeg_positions" / CAP_BASENAME
    mesh = m2m / f"{subject}.msh"
    try:
        checks = (
            marker.get("subject") == subject,
            marker.get("corrected_label_sha256")
            == row["corrected_label_sha256"],
            marker.get("installed_label_sha256")
            == row["corrected_label_sha256"],
            marker.get("source_t1_sha256") == row["source_t1_sha256"],
            marker.get("source_t2_sha256") == row["source_t2_sha256"],
            sha256_file(label) == row["corrected_label_sha256"],
            sha256_file(cap) == marker.get("eeg_cap_sha256"),
            sha256_file(Path(marker["task_t1_path"])) == row["source_t1_sha256"],
            sha256_file(Path(marker["task_t2_path"])) == row["source_t2_sha256"],
            set(row["required_electrodes"].split(",")).issubset(
                approved._cap_names(cap)
            ),
            anat.is_dir(),
            m2m.is_dir(),
            not anat.is_symlink(),
            not m2m.is_symlink(),
            not mesh.exists(),
        )
        return marker if all(checks) else None
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        csv.Error,
    ):
        return None


def _completed_bootstrap_can_be_recovered(
    row: Mapping[str, str],
    anat: Path,
    m2m: Path,
    task_t1: Path,
    task_t2: Path,
) -> bool:
    """Detect the checkpoint left after a successful CHARM segmentation.

    The corrected label is installed only after the segmentation command
    returns successfully.  Its exact hash, together with the copied source
    scans, therefore distinguishes a completed segmentation from a partial
    bootstrap.  This permits a retry to create the missing EEG cap without
    deleting and rerunning the completed segmentation.
    """

    installed_label = m2m / "label_prep" / MAP_BASENAME
    try:
        checks = (
            anat.is_dir(),
            m2m.is_dir(),
            not anat.is_symlink(),
            not m2m.is_symlink(),
            sha256_file(task_t1) == row["source_t1_sha256"],
            sha256_file(task_t2) == row["source_t2_sha256"],
            sha256_file(installed_label) == row["corrected_label_sha256"],
        )
        return all(checks)
    except (FileNotFoundError, OSError):
        return False


def _manifest_paths(campaign_root: Path) -> tuple[Path, Path, Path, Path]:
    return (
        campaign_root / "scaffold_tasks.tsv",
        campaign_root / "mesh_tasks.tsv",
        campaign_root / "simulation_tasks.tsv",
        campaign_root / "preflight.json",
    )


def build_manifests(
    *,
    study_config: str | Path,
    cohort_config: str | Path,
    campaign_root: str | Path,
    targets_csv: str | Path,
    map_manifest: str | Path | None = None,
    study_root: str | Path | None = None,
    scaffold_root: str | Path | None = None,
    source_roots: Sequence[str | Path] | None = None,
    legacy_scaffold_manifest: str | Path | None = None,
    max_array_elements: int = 1000,
    mesh_workers: int = 2,
) -> dict[str, object]:
    study_path = Path(study_config).expanduser().resolve(strict=True)
    cohort_path = Path(cohort_config).expanduser().resolve(strict=True)
    study = _load_json(study_path)
    cohort = _load_json(cohort_path)
    if int(study.get("schema_version", 0)) != 1:
        raise ValueError("unsupported study config schema")
    if int(cohort.get("schema_version", 0)) != 1:
        raise ValueError("unsupported cohort config schema")
    cohort_id = _safe_cohort_id(str(cohort.get("cohort_id", "")))
    subjects_path = _resolve_config_relative(
        cohort_path, str(cohort.get("subjects_file", ""))
    )
    subjects = approved.read_subjects(subjects_path)
    expected_subjects = int(cohort.get("expected_subjects", 0))
    if not 1 <= expected_subjects <= 200:
        raise ValueError("expected_subjects must be between 1 and 200")
    if len(subjects) != expected_subjects:
        raise ValueError(
            f"cohort contains {len(subjects)} subjects, expected {expected_subjects}"
        )
    if max_array_elements < 1 or mesh_workers < 1:
        raise ValueError("array and mesh-worker limits must be positive")

    configured_rois = tuple(str(value) for value in study.get("rois", DEFAULT_ROIS))
    configured_repeats = tuple(
        str(value) for value in study.get("repeats", DEFAULT_REPEATS)
    )
    if set(configured_rois) != set(DEFAULT_ROIS) or len(configured_rois) != 4:
        raise ValueError("study must contain the four established CamCan ROIs")
    if configured_repeats != DEFAULT_REPEATS:
        raise ValueError("study repeats must be exactly 01 through 10")

    resolved_study_root = Path(
        study_root or str(study["hpc_study_root"])
    ).expanduser().resolve()
    resolved_scaffold_root = Path(
        scaffold_root or str(study["hpc_scaffold_root"])
    ).expanduser().resolve()
    resolved_map_manifest = Path(
        map_manifest or str(study["hpc_corrected_map_manifest"])
    ).expanduser().resolve(strict=True)
    raw_source_roots = source_roots or tuple(study.get("hpc_source_roots", ()))
    resolved_source_roots = [
        Path(value).expanduser().resolve(strict=True) for value in raw_source_roots
    ]
    raw_legacy = legacy_scaffold_manifest
    if raw_legacy is None:
        raw_legacy = str(study.get("hpc_legacy_scaffold_manifest", ""))
    resolved_legacy = (
        Path(raw_legacy).expanduser().resolve()
        if raw_legacy
        else None
    )
    targets = Path(targets_csv).expanduser().resolve(strict=True)
    targets_hash = sha256_file(targets)
    expected_targets_hash = str(
        study.get("targets_csv_sha256", CONFIRMED_TARGETS_SHA256)
    )
    if targets_hash != expected_targets_hash:
        raise ValueError(
            f"targets.csv hash mismatch: {targets_hash} != {expected_targets_hash}"
        )

    individualized_targets_path: Path | None = None
    individualized_targets_hash: str | None = None
    individualized_targets: (
        dict[tuple[str, str], dict[str, str]] | None
    ) = None
    raw_individualized_targets = str(
        cohort.get("individualized_targets_csv", "")
    ).strip()
    if raw_individualized_targets:
        individualized_targets_path = _resolve_config_relative(
            cohort_path, raw_individualized_targets
        )
        individualized_targets_hash = sha256_file(
            individualized_targets_path
        )
        expected_individualized_hash = str(
            cohort.get("individualized_targets_csv_sha256", "")
        ).strip()
        if (
            not expected_individualized_hash
            or individualized_targets_hash != expected_individualized_hash
        ):
            raise ValueError(
                "individualized targets CSV hash mismatch: "
                f"{individualized_targets_hash} != "
                f"{expected_individualized_hash}"
            )
        individualized_targets = validate_individualized_target_table(
            individualized_targets_path,
            subjects=subjects,
            configs=tuple(ROI_BY_PREFIX[roi] for roi in configured_rois),
        )
    elif str(cohort.get("individualized_targets_csv_sha256", "")).strip():
        raise ValueError(
            "cohort declares an individualized targets hash without a CSV"
        )

    by_subject_roi_electrodes, all_electrodes = _required_electrodes(
        configured_rois,
        configured_repeats,
        subjects,
        targets,
        individualized_targets,
    )
    corrected_maps = _read_collection(resolved_map_manifest)
    legacy = _legacy_scaffolds(resolved_legacy)
    campaign = Path(campaign_root).expanduser().resolve()
    scaffold_manifest, mesh_manifest, simulation_manifest, summary_path = (
        _manifest_paths(campaign)
    )
    scaffold_results = resolved_scaffold_root / "results"
    shared_results = resolved_study_root / "results"

    scaffold_rows: list[dict[str, object]] = []
    for task_id, subject in enumerate(subjects):
        messages: list[str] = []
        source_anat: Path | str = ""
        source_t1: Path | str = ""
        source_t2: Path | str = ""
        t1_hash = ""
        t2_hash = ""
        try:
            source_anat, source_t1, source_t2 = approved.resolve_source_inputs(
                resolved_source_roots, subject
            )
            t1_hash = sha256_file(Path(source_t1))
            t2_hash = sha256_file(Path(source_t2))
        except (FileNotFoundError, OSError, ValueError) as exc:
            messages.append(str(exc))

        collection = corrected_maps.get(subject)
        corrected_label: Path | str = ""
        corrected_hash = ""
        if collection is None:
            messages.append("subject is absent from corrected-map collection")
        elif collection.get("status") != "complete":
            messages.append(
                f"corrected-map collection status is {collection.get('status')!r}"
            )
        else:
            raw_label = collection.get("collected_map") or collection.get("source_map")
            corrected_label = Path(str(raw_label)).expanduser()
            corrected_hash = collection.get("sha256", "")
            try:
                if corrected_label.is_symlink():
                    raise ValueError("corrected label is symlinked")
                corrected_label = corrected_label.resolve(strict=True)
                if sha256_file(corrected_label) != corrected_hash:
                    raise ValueError("corrected label hash differs from collection")
                if corrected_label.stat().st_size <= 0:
                    raise ValueError("corrected label is empty")
            except (OSError, ValueError) as exc:
                messages.append(str(exc))

        scaffold_anat = (
            resolved_scaffold_root / "subjects" / subject / "anat"
        )
        scaffold_m2m = scaffold_anat / f"m2m_{subject}"
        result_path = scaffold_results / f"{subject}.json"
        legacy_row = legacy.get(subject)
        row: dict[str, object] = {
            "task_id": task_id,
            "subject": subject,
            "mode": "bootstrap",
            "source_anat": source_anat,
            "source_t1": source_t1,
            "source_t1_sha256": t1_hash,
            "source_t2": source_t2,
            "source_t2_sha256": t2_hash,
            "corrected_label": corrected_label,
            "corrected_label_sha256": corrected_hash,
            "legacy_anat_dir": legacy_row.get("anat_dir", "") if legacy_row else "",
            "legacy_m2m_dir": legacy_row.get("m2m_dir", "") if legacy_row else "",
            "legacy_result_path": legacy_row.get("result_path", "") if legacy_row else "",
            "scaffold_anat_dir": scaffold_anat,
            "scaffold_m2m_dir": scaffold_m2m,
            "result_path": result_path,
            "required_electrodes": ",".join(all_electrodes),
            "status": "ready" if not messages else "blocked",
            "message": "ready" if not messages else "; ".join(messages),
        }
        if not messages:
            if scaffold_result_is_current(row) is not None:
                row["mode"] = "reuse"
            elif _legacy_scaffold_is_usable(
                legacy_row, subject, all_electrodes
            ):
                row["mode"] = "import"
        scaffold_rows.append(row)

    scaffold_by_subject = {
        str(row["subject"]): row for row in scaffold_rows
    }
    task_rows: list[dict[str, object]] = []
    for roi in configured_rois:
        config = ROI_BY_PREFIX[roi]
        for repeat in configured_repeats:
            dataset_name = f"{roi}_Data_{repeat}"
            dataset_root = (
                resolved_study_root / "runs" / f"{roi}_Runs" / dataset_name
            )
            for subject in subjects:
                scaffold = scaffold_by_subject[subject]
                anat = dataset_root / subject / "anat"
                m2m = anat / f"m2m_{subject}"
                task_rows.append(
                    {
                        "task_id": len(task_rows),
                        "roi": roi,
                        "montage_preset": config.montage_preset,
                        "dataset_name": dataset_name,
                        "repeat_id": repeat,
                        "dataset_root": dataset_root,
                        "subject": subject,
                        "source_t1_sha256": scaffold["source_t1_sha256"],
                        "source_t2_sha256": scaffold["source_t2_sha256"],
                        "anat_dir": anat,
                        "m2m_dir": m2m,
                        "mesh_path": m2m / f"{subject}.msh",
                        "corrected_label": scaffold["corrected_label"],
                        "corrected_label_sha256": scaffold[
                            "corrected_label_sha256"
                        ],
                        "scaffold_anat_dir": scaffold["scaffold_anat_dir"],
                        "scaffold_m2m_dir": scaffold["scaffold_m2m_dir"],
                        "scaffold_result_path": scaffold["result_path"],
                        "mesh_result_path": (
                            shared_results
                            / "meshes"
                            / roi
                            / repeat
                            / f"{subject}.json"
                        ),
                        "simulation_result_path": (
                            shared_results
                            / "simulations"
                            / roi
                            / repeat
                            / f"{subject}.json"
                        ),
                        "required_electrodes": ",".join(
                            by_subject_roi_electrodes[(subject, roi)]
                        ),
                        "status": scaffold["status"],
                        "message": scaffold["message"],
                    }
                )

    expected_tasks = expected_subjects * len(configured_rois) * len(configured_repeats)
    ready_scaffolds = sum(row["status"] == "ready" for row in scaffold_rows)
    ready_tasks = sum(row["status"] == "ready" for row in task_rows)
    if len(task_rows) != expected_tasks:
        raise RuntimeError("internal task-count mismatch")
    mesh_elements = (expected_tasks + mesh_workers - 1) // mesh_workers
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": (
            "ready"
            if ready_scaffolds == expected_subjects
            and ready_tasks == expected_tasks
            else "blocked"
        ),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "study_id": study.get("study_id"),
        "cohort_id": cohort_id,
        "cohort_status": cohort.get("status", "unspecified"),
        "subjects_file": str(subjects_path),
        "subjects_file_sha256": sha256_file(subjects_path),
        "subjects": expected_subjects,
        "rois": list(configured_rois),
        "repeats": list(configured_repeats),
        "corrected_map_manifest": str(resolved_map_manifest),
        "corrected_map_set": study.get("corrected_map_set"),
        "study_root": str(resolved_study_root),
        "scaffold_root": str(resolved_scaffold_root),
        "legacy_scaffold_manifest": (
            str(resolved_legacy) if resolved_legacy else None
        ),
        "targets_csv": str(targets),
        "targets_csv_sha256": targets_hash,
        "individualized_targets_csv": (
            str(individualized_targets_path)
            if individualized_targets_path is not None
            else None
        ),
        "individualized_targets_csv_sha256": individualized_targets_hash,
        "montage_mode": (
            "subject_roi_individualized"
            if individualized_targets is not None
            else "fixed_roi"
        ),
        "individualized_target_rows": (
            len(individualized_targets)
            if individualized_targets is not None
            else 0
        ),
        "scaffold_manifest": str(scaffold_manifest),
        "mesh_manifest": str(mesh_manifest),
        "simulation_manifest": str(simulation_manifest),
        "scaffold_tasks": expected_subjects,
        "scaffold_ready": ready_scaffolds,
        "scaffold_reuse": sum(row["mode"] == "reuse" for row in scaffold_rows),
        "scaffold_import": sum(row["mode"] == "import" for row in scaffold_rows),
        "scaffold_bootstrap": sum(
            row["mode"] == "bootstrap" for row in scaffold_rows
        ),
        "full_charm_segmentations_expected": sum(
            row["mode"] == "bootstrap" and row["status"] == "ready"
            for row in scaffold_rows
        ),
        "temporary_scaffold_mesh_runs_expected": sum(
            row["mode"] == "bootstrap" and row["status"] == "ready"
            for row in scaffold_rows
        ),
        "mesh_tasks": expected_tasks,
        "mesh_ready": ready_tasks,
        "mesh_workers_per_array_element": mesh_workers,
        "mesh_array_elements": mesh_elements,
        "mesh_array_chunks": (
            mesh_elements + max_array_elements - 1
        )
        // max_array_elements,
        "simulation_tasks": expected_tasks,
        "simulation_ready": ready_tasks,
        "simulation_array_chunks": (
            expected_tasks + max_array_elements - 1
        )
        // max_array_elements,
        "max_array_elements": max_array_elements,
        "independent_meshes_expected": expected_tasks,
        "validated_fem_results_expected": expected_tasks,
        "roast_involvement": False,
        "source_maps_modified": False,
        "execution": (
            "reusable scaffold -> physical repeat copy -> corrected-map direct "
            "mesh -> ROI FEM"
        ),
    }
    approved.write_tsv(scaffold_manifest, SCAFFOLD_FIELDS, scaffold_rows)
    approved.write_tsv(mesh_manifest, TASK_FIELDS, task_rows)
    approved.write_tsv(simulation_manifest, TASK_FIELDS, task_rows)
    approved.write_json_atomic(summary_path, payload)
    return payload


def _clear_subject_root(anat_dir: Path, allowed_root: Path) -> None:
    subject_root = anat_dir.parent
    if not subject_root.exists() and not subject_root.is_symlink():
        return
    if subject_root.is_symlink():
        raise ValueError(f"refusing symlinked subject root: {subject_root}")
    resolved = subject_root.resolve()
    if not resolved.is_relative_to(allowed_root.resolve()):
        raise ValueError(f"refusing to clear subject outside allowed root: {resolved}")
    shutil.rmtree(subject_root)


def run_scaffold_task(
    *,
    manifest: str | Path,
    task_index: int,
    charm_bin: str = "charm",
    command_runner: Callable[..., None] = approved._run_command,
) -> dict[str, object]:
    rows = approved.read_tsv(manifest)
    if task_index < 0 or task_index >= len(rows):
        raise IndexError(f"scaffold task {task_index} is outside the manifest")
    row = rows[task_index]
    if int(row["task_id"]) != task_index or row["status"] != "ready":
        raise ValueError(f"scaffold task is blocked or misindexed: {row['message']}")
    current = scaffold_result_is_current(row)
    if current is not None:
        print(json.dumps({"event": "cohort_scaffold_reused", **current}), flush=True)
        return current

    for path_key, hash_key in (
        ("source_t1", "source_t1_sha256"),
        ("source_t2", "source_t2_sha256"),
        ("corrected_label", "corrected_label_sha256"),
    ):
        actual = sha256_file(Path(row[path_key]))
        if actual != row[hash_key]:
            raise ValueError(f"{path_key} hash changed: {actual} != {row[hash_key]}")

    subject = row["subject"]
    anat = Path(row["scaffold_anat_dir"])
    m2m = Path(row["scaffold_m2m_dir"])
    scaffold_root = m2m.parents[3]
    source_t1 = Path(row["source_t1"])
    source_t2 = Path(row["source_t2"])
    task_t1 = anat / source_t1.name
    task_t2 = anat / source_t2.name

    execution_mode = row["mode"]
    if execution_mode == "reuse":
        execution_mode = (
            "import"
            if _legacy_scaffold_is_usable(
                {
                    "anat_dir": row["legacy_anat_dir"],
                    "m2m_dir": row["legacy_m2m_dir"],
                    "result_path": row["legacy_result_path"],
                },
                subject,
                row["required_electrodes"].split(","),
            )
            else "bootstrap"
        )

    bootstrap_recovered = (
        execution_mode == "bootstrap"
        and _completed_bootstrap_can_be_recovered(
            row,
            anat,
            m2m,
            task_t1,
            task_t2,
        )
    )
    if not bootstrap_recovered:
        _clear_subject_root(anat, scaffold_root)
        anat.mkdir(parents=True, exist_ok=True)
        approved._copy_verified(
            source_t1,
            task_t1,
            row["source_t1_sha256"],
        )
        approved._copy_verified(
            source_t2,
            task_t2,
            row["source_t2_sha256"],
        )
    Path(row["result_path"]).unlink(missing_ok=True)

    segmentation_runs = 0
    temporary_mesh_runs = 0
    segmentation_command: list[str] | None = None
    temporary_mesh_command: list[str] | None = None
    temporary_mesh_bytes = 0
    temporary_mesh_sha256: str | None = None
    generated_label_hash: str | None = None
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if execution_mode == "import":
        legacy_m2m = Path(row["legacy_m2m_dir"]).resolve(strict=True)

        def ignore_mesh(directory: str, names: list[str]) -> set[str]:
            if Path(directory).resolve() == legacy_m2m:
                return {f"{subject}.msh"}.intersection(names)
            return set()

        shutil.copytree(
            legacy_m2m,
            m2m,
            symlinks=False,
            ignore=ignore_mesh,
        )
        source_description = str(legacy_m2m)
    elif execution_mode == "bootstrap":
        if bootstrap_recovered:
            source_description = "recovered_completed_charm_bootstrap"
        else:
            segmentation_command = [
                charm_bin,
                subject,
                str(task_t1),
                str(task_t2),
                "--registerT2",
                "--initatlas",
                "--segment",
                "--forceqform",
            ]
            command_runner(segmentation_command, cwd=anat)
            segmentation_runs = 1
            generated_label_hash = sha256_file(
                m2m / "label_prep" / MAP_BASENAME
            )
            source_description = "new_charm_bootstrap"
    else:
        raise ValueError(f"unexpected non-current scaffold mode: {execution_mode}")

    installed_label = m2m / "label_prep" / MAP_BASENAME
    approved._copy_verified(
        Path(row["corrected_label"]),
        installed_label,
        row["corrected_label_sha256"],
        replace_existing=True,
    )
    if sha256_file(installed_label) != row["corrected_label_sha256"]:
        raise RuntimeError("corrected-v4 scaffold label installation failed")

    mesh = m2m / f"{subject}.msh"
    mesh.unlink(missing_ok=True)
    cap = m2m / "eeg_positions" / CAP_BASENAME
    required = set(row["required_electrodes"].split(","))
    try:
        cap_is_usable = required.issubset(approved._cap_names(cap))
    except (FileNotFoundError, OSError, UnicodeError, csv.Error):
        cap_is_usable = False

    if execution_mode == "bootstrap" and not cap_is_usable:
        temporary_mesh_command = [charm_bin, subject, "--mesh"]
        command_runner(temporary_mesh_command, cwd=anat)
        temporary_mesh_runs = 1
        if sha256_file(installed_label) != row["corrected_label_sha256"]:
            mesh.unlink(missing_ok=True)
            raise RuntimeError("temporary scaffold meshing changed corrected-v4 label")
        if not mesh.is_file() or mesh.stat().st_size == 0:
            raise RuntimeError(
                "temporary CHARM mesh did not produce a non-empty mesh"
            )
        temporary_mesh_bytes = mesh.stat().st_size
        temporary_mesh_sha256 = sha256_file(mesh)

    missing = sorted(required - approved._cap_names(cap))
    if missing:
        mesh.unlink(missing_ok=True)
        raise ValueError("scaffold EEG cap lacks: " + ",".join(missing))
    mesh.unlink(missing_ok=True)

    payload: dict[str, object] = {
        "schema_version": 2,
        "status": "complete",
        "task_index": task_index,
        "subject": subject,
        "scaffold_mode": execution_mode,
        "scaffold_source": source_description,
        "source_t1_sha256": row["source_t1_sha256"],
        "source_t2_sha256": row["source_t2_sha256"],
        "task_t1_path": str(task_t1),
        "task_t2_path": str(task_t2),
        "corrected_label": row["corrected_label"],
        "corrected_label_sha256": row["corrected_label_sha256"],
        "installed_label": str(installed_label),
        "installed_label_sha256": sha256_file(installed_label),
        "eeg_cap_path": str(cap),
        "eeg_cap_sha256": sha256_file(cap),
        "required_electrodes": sorted(required),
        "segmentation_runs_for_scaffold": segmentation_runs,
        "segmentation_command": segmentation_command,
        "generated_label_sha256_before_corrected_install": generated_label_hash,
        "completed_segmentation_recovered": bootstrap_recovered,
        "temporary_mesh_runs_for_eeg_cap": temporary_mesh_runs,
        "temporary_mesh_command": temporary_mesh_command,
        "temporary_mesh_bytes_before_removal": temporary_mesh_bytes,
        "temporary_mesh_sha256_before_removal": temporary_mesh_sha256,
        "temporary_mesh_removed": not mesh.exists(),
        "mesh_created_in_scaffold_stage": False,
        "roast_involvement": False,
        "started_at": started_at,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    approved.write_json_atomic(Path(row["result_path"]), payload)
    print(json.dumps({"event": "cohort_scaffold_complete", **payload}), flush=True)
    return payload


def _task_row(manifest: str | Path, task_index: int) -> dict[str, str]:
    rows = approved.read_tsv(manifest)
    if task_index < 0 or task_index >= len(rows):
        raise IndexError(f"task {task_index} is outside the manifest")
    row = rows[task_index]
    if int(row["task_id"]) != task_index or row["status"] != "ready":
        raise ValueError(f"task is blocked or misindexed: {row['message']}")
    return row


def _validate_scaffold_for_task(
    row: Mapping[str, str],
) -> dict[str, object]:
    scaffold_row = {
        "subject": row["subject"],
        "source_t1_sha256": row["source_t1_sha256"],
        "source_t2_sha256": row["source_t2_sha256"],
        "corrected_label_sha256": row["corrected_label_sha256"],
        "scaffold_anat_dir": row["scaffold_anat_dir"],
        "scaffold_m2m_dir": row["scaffold_m2m_dir"],
        "result_path": row["scaffold_result_path"],
        "required_electrodes": row["required_electrodes"],
    }
    marker = approved._load_json(Path(row["scaffold_result_path"]))
    if marker is None:
        raise ValueError(f"scaffold marker is absent: {row['scaffold_result_path']}")
    current = scaffold_result_is_current(scaffold_row)
    if current is None:
        raise ValueError(f"scaffold is invalid: {row['scaffold_result_path']}")
    return current


def _materialize_repeat_scaffold(
    row: Mapping[str, str],
    scaffold: Mapping[str, object],
) -> None:
    anat = Path(row["anat_dir"])
    m2m = Path(row["m2m_dir"])
    installed_label = m2m / "label_prep" / MAP_BASENAME
    cap = m2m / "eeg_positions" / CAP_BASENAME
    canonical_t1 = Path(str(scaffold["task_t1_path"]))
    canonical_t2 = Path(str(scaffold["task_t2_path"]))
    task_t1 = anat / canonical_t1.name
    task_t2 = anat / canonical_t2.name
    try:
        existing_ok = bool(
            anat.is_dir()
            and not anat.is_symlink()
            and m2m.is_dir()
            and not m2m.is_symlink()
            and sha256_file(installed_label) == row["corrected_label_sha256"]
            and sha256_file(cap) == scaffold["eeg_cap_sha256"]
            and task_t1.is_symlink()
            and task_t1.resolve(strict=True) == canonical_t1.resolve(strict=True)
            and sha256_file(task_t1) == row["source_t1_sha256"]
            and task_t2.is_symlink()
            and task_t2.resolve(strict=True) == canonical_t2.resolve(strict=True)
            and sha256_file(task_t2) == row["source_t2_sha256"]
        )
    except (FileNotFoundError, OSError):
        existing_ok = False
    if existing_ok:
        return

    dataset_root = Path(row["dataset_root"])
    _clear_subject_root(anat, dataset_root)
    anat.mkdir(parents=True, exist_ok=True)
    subject = row["subject"]
    for canonical_input, expected_hash in (
        (canonical_t1, row["source_t1_sha256"]),
        (canonical_t2, row["source_t2_sha256"]),
    ):
        if sha256_file(canonical_input) != expected_hash:
            raise ValueError(f"canonical input hash changed: {canonical_input}")
        approved._ensure_relative_symlink(
            canonical_input, anat / canonical_input.name
        )

    canonical_m2m = Path(row["scaffold_m2m_dir"]).resolve(strict=True)

    def ignore_mesh(directory: str, names: list[str]) -> set[str]:
        if Path(directory).resolve() == canonical_m2m:
            return {f"{subject}.msh"}.intersection(names)
        return set()

    staging = anat / f".m2m_{subject}.copy-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        shutil.copytree(
            canonical_m2m,
            staging,
            symlinks=False,
            ignore=ignore_mesh,
        )
        os.replace(staging, m2m)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if (
        sha256_file(installed_label) != row["corrected_label_sha256"]
        or sha256_file(cap) != scaffold["eeg_cap_sha256"]
    ):
        raise RuntimeError("physical scaffold copy failed validation")


def run_mesh_task(
    *,
    manifest: str | Path,
    task_index: int,
    settings_path: str | Path | None = None,
    staging_root: str | Path | None = None,
) -> dict[str, object]:
    row = _task_row(manifest, task_index)
    scaffold = _validate_scaffold_for_task(row)
    _materialize_repeat_scaffold(row, scaffold)
    installed_label = (
        Path(row["m2m_dir"]) / "label_prep" / MAP_BASENAME
    )
    provenance = {
        "cohort_schema_version": 1,
        "roi": row["roi"],
        "montage_preset": row["montage_preset"],
        "dataset_name": row["dataset_name"],
        "repeat_id": row["repeat_id"],
        "scaffold_result_path": row["scaffold_result_path"],
        "scaffold_eeg_cap_sha256": scaffold["eeg_cap_sha256"],
        "scaffold_copy_mode": "physical",
        "independent_repeat_mesh": True,
        "roast_involvement": False,
    }
    return direct_mesh.create_mesh_from_label(
        subject=row["subject"],
        label_path=installed_label,
        expected_label_hash=row["corrected_label_sha256"],
        mesh_path=row["mesh_path"],
        result_path=row["mesh_result_path"],
        task_index=task_index,
        settings_path=settings_path,
        staging_root=staging_root,
        provenance=provenance,
    )


def mesh_result_is_current(
    row: Mapping[str, str],
    *,
    verify_hash: bool = True,
) -> dict[str, object] | None:
    marker = approved._load_json(Path(row["mesh_result_path"]))
    if marker is None:
        return None
    mesh = Path(row["mesh_path"])
    label = Path(row["m2m_dir"]) / "label_prep" / MAP_BASENAME
    cap = Path(row["m2m_dir"]) / "eeg_positions" / CAP_BASENAME
    scaffold = approved._load_json(Path(row["scaffold_result_path"]))
    try:
        checks = (
            marker.get("status") == "complete",
            marker.get("subject") == row["subject"],
            marker.get("roi") == row["roi"],
            marker.get("repeat_id") == row["repeat_id"],
            marker.get("dataset_name") == row["dataset_name"],
            marker.get("label_sha256_after") == row["corrected_label_sha256"],
            marker.get("independent_repeat_mesh") is True,
            marker.get("scaffold_copy_mode") == "physical",
            marker.get("roast_involvement") is False,
            mesh.is_file(),
            mesh.stat().st_size == marker.get("mesh_bytes"),
            sha256_file(label) == row["corrected_label_sha256"],
            scaffold is not None,
            sha256_file(cap) == scaffold.get("eeg_cap_sha256"),
        )
        if not all(checks):
            return None
        if verify_hash and sha256_file(mesh) != marker.get("mesh_sha256"):
            return None
        return marker
    except (FileNotFoundError, OSError, TypeError):
        return None


def run_simulation_task(
    *,
    manifest: str | Path,
    task_index: int,
    targets_csv: str | Path,
    expected_targets_sha256: str,
    simulation_runner: str | Path,
    simulation_validator: str | Path,
    individualized_targets_csv: str | Path | None = None,
    expected_individualized_targets_sha256: str | None = None,
    python_bin: str = "python",
    command_runner: Callable[..., None] = approved._run_command,
) -> dict[str, object]:
    row = _task_row(manifest, task_index)
    targets = Path(targets_csv).expanduser().resolve(strict=True)
    targets_hash = sha256_file(targets)
    if targets_hash != expected_targets_sha256:
        raise ValueError("targets.csv hash mismatch")
    config = validate_dataset_montage(
        [row["dataset_name"]], row["montage_preset"]
    )
    individualized_path: Path | None = None
    individualized_hash: str | None = None
    individualized_row: dict[str, str] | None = None
    if individualized_targets_csv:
        individualized_path = (
            Path(individualized_targets_csv)
            .expanduser()
            .resolve(strict=True)
        )
        individualized_hash = sha256_file(individualized_path)
        if (
            not expected_individualized_targets_sha256
            or individualized_hash
            != expected_individualized_targets_sha256
        ):
            raise ValueError("individualized targets CSV hash mismatch")
        individualized_row = load_individualized_target_row(
            individualized_path,
            subject=row["subject"],
            dataset_roi=row["roi"],
        )
        if (
            individualized_row["roi"].strip() != config.targets_roi
            or individualized_row["pareto_selection"].strip()
            != "TI_free.Emin"
        ):
            raise ValueError(
                "individualized target row does not match the task ROI or "
                "TI_free.Emin selection"
            )
        required = set(
            electrode_names_from_target_row(individualized_row)
        )
    elif expected_individualized_targets_sha256:
        raise ValueError(
            "individualized targets hash supplied without a CSV"
        )
    else:
        required = set(electrode_names_for_config(config, targets))
    mesh_marker = mesh_result_is_current(row)
    if mesh_marker is None:
        raise ValueError(f"mesh result is invalid: {row['mesh_result_path']}")
    cap = Path(row["m2m_dir"]) / "eeg_positions" / CAP_BASENAME
    missing = sorted(required - approved._cap_names(cap))
    if missing:
        raise ValueError("repeat EEG cap lacks: " + ",".join(missing))

    result_path = Path(row["simulation_result_path"])
    existing = approved._load_json(result_path)
    reusable = bool(
        existing
        and existing.get("status") == "complete"
        and existing.get("subject") == row["subject"]
        and existing.get("roi") == row["roi"]
        and existing.get("repeat_id") == row["repeat_id"]
        and existing.get("mesh_sha256") == mesh_marker.get("mesh_sha256")
        and existing.get("corrected_label_sha256")
        == row["corrected_label_sha256"]
        and existing.get("montage_preset") == row["montage_preset"]
        and existing.get("targets_csv_sha256") == targets_hash
        and existing.get("individualized_targets_csv_sha256")
        == individualized_hash
        and validate_subject_outputs(
            row["dataset_root"],
            row["subject"],
            check_nifti=True,
        ).ok
    )
    environment = os.environ.copy()
    environment["TI_SIM_ROOT"] = row["dataset_root"]
    validation_command = [
        python_bin,
        "-u",
        str(Path(simulation_validator).expanduser().resolve(strict=True)),
        "--root",
        row["dataset_root"],
        "--subject",
        row["subject"],
    ]
    if reusable:
        command_runner(
            validation_command, cwd=Path(row["anat_dir"]), env=environment
        )
        existing["status"] = "already_complete"
        print(
            json.dumps({"event": "cohort_simulation_reused", **existing}),
            flush=True,
        )
        return existing

    simulation_command = [
        python_bin,
        "-u",
        str(Path(simulation_runner).expanduser().resolve(strict=True)),
        "--subject",
        row["subject"],
        "--montage-preset",
        row["montage_preset"],
        "--reuse-existing-mesh",
    ]
    if individualized_path is not None:
        simulation_command.extend(
            [
                "--individualized-targets-csv",
                str(individualized_path),
                "--expected-individualized-targets-sha256",
                str(individualized_hash),
            ]
        )
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    command_runner(
        simulation_command, cwd=Path(row["anat_dir"]), env=environment
    )
    command_runner(
        validation_command, cwd=Path(row["anat_dir"]), env=environment
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "task_index": task_index,
        "roi": row["roi"],
        "dataset_name": row["dataset_name"],
        "repeat_id": row["repeat_id"],
        "subject": row["subject"],
        "mesh_path": row["mesh_path"],
        "mesh_sha256": mesh_marker["mesh_sha256"],
        "corrected_label_sha256": row["corrected_label_sha256"],
        "montage_preset": row["montage_preset"],
        "targets_csv": str(targets),
        "targets_csv_sha256": targets_hash,
        "individualized_targets_csv": (
            str(individualized_path)
            if individualized_path is not None
            else None
        ),
        "individualized_targets_csv_sha256": individualized_hash,
        "pareto_selection": (
            individualized_row.get("pareto_selection")
            if individualized_row is not None
            else None
        ),
        "optimized_configuration": (
            int(individualized_row["configuration"])
            if individualized_row is not None
            else None
        ),
        "optimized_e_target_v_per_m": (
            float(individualized_row["E_target"])
            if individualized_row is not None
            else None
        ),
        "optimized_stimulated_volume": (
            float(individualized_row["stimulated_volume"])
            if individualized_row is not None
            else None
        ),
        "optimized_pair1": (
            individualized_row.get("pair1")
            if individualized_row is not None
            else None
        ),
        "optimized_pair2": (
            individualized_row.get("pair2")
            if individualized_row is not None
            else None
        ),
        "optimized_current1_ma": (
            float(individualized_row["current1"])
            if individualized_row is not None
            else None
        ),
        "optimized_current2_ma": (
            float(individualized_row["current2"])
            if individualized_row is not None
            else None
        ),
        "optimized_source_target_id": (
            individualized_row.get("source_target_id")
            if individualized_row is not None
            else None
        ),
        "optimized_source_mat": (
            individualized_row.get("source_mat")
            if individualized_row is not None
            else None
        ),
        "optimized_source_mat_sha256": (
            individualized_row.get("source_mat_sha256")
            if individualized_row is not None
            else None
        ),
        "simulation_command": simulation_command,
        "validation_command": validation_command,
        "segmentation_in_simulation_task": False,
        "mesh_created_in_simulation_task": False,
        "independent_repeat_mesh": True,
        "roast_involvement": False,
        "started_at": started_at,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    approved.write_json_atomic(result_path, payload)
    print(json.dumps({"event": "cohort_simulation_complete", **payload}), flush=True)
    return payload


def validate_stage(
    *,
    stage: str,
    manifest: str | Path,
    summary: str | Path,
    verify_hashes: bool = True,
) -> dict[str, object]:
    rows = approved.read_tsv(manifest)
    validation_rows: list[dict[str, object]] = []
    complete = 0
    for row in rows:
        messages: list[str] = []
        roi = row.get("roi", "")
        repeat = row.get("repeat_id", "")
        if stage == "scaffolds":
            result_path = Path(row["result_path"])
            current = scaffold_result_is_current(row)
        elif stage == "meshes":
            result_path = Path(row["mesh_result_path"])
            current = mesh_result_is_current(row, verify_hash=verify_hashes)
        elif stage == "simulations":
            result_path = Path(row["simulation_result_path"])
            current = approved._load_json(result_path)
            mesh = mesh_result_is_current(row, verify_hash=verify_hashes)
            outputs = validate_subject_outputs(
                row["dataset_root"],
                row["subject"],
                check_nifti=verify_hashes,
            )
            if not (
                current
                and current.get("status") == "complete"
                and current.get("subject") == row["subject"]
                and current.get("roi") == row["roi"]
                and current.get("repeat_id") == row["repeat_id"]
                and current.get("montage_preset") == row["montage_preset"]
                and current.get("corrected_label_sha256")
                == row["corrected_label_sha256"]
                and mesh is not None
                and current.get("mesh_sha256") == mesh.get("mesh_sha256")
                and outputs.ok
            ):
                current = None
        else:
            raise ValueError(f"unsupported validation stage: {stage}")
        if current is None:
            messages.append(f"{stage} result is missing or invalid")
        status = "complete" if not messages else "incomplete"
        complete += status == "complete"
        validation_rows.append(
            {
                "task_id": row["task_id"],
                "stage": stage,
                "roi": roi,
                "repeat_id": repeat,
                "subject": row["subject"],
                "status": status,
                "result_path": str(result_path),
                "message": "complete" if not messages else "; ".join(messages),
            }
        )
    summary_path = Path(summary).expanduser().resolve()
    approved.write_tsv(summary_path, VALIDATION_FIELDS, validation_rows)
    payload = {
        "status": "complete" if complete == len(rows) else "incomplete",
        "stage": stage,
        "tasks": len(rows),
        "complete": complete,
        "incomplete": len(rows) - complete,
        "hashes_verified": verify_hashes,
        "summary": str(summary_path),
    }
    approved.write_json_atomic(summary_path.with_suffix(".json"), payload)
    return payload


def create_cohort(
    *,
    cohort_id: str,
    subjects_file: str | Path,
    output_root: str | Path,
    status: str,
    note: str,
) -> dict[str, object]:
    cohort_id = _safe_cohort_id(cohort_id)
    source = Path(subjects_file).expanduser().resolve(strict=True)
    subjects = approved.read_subjects(source)
    if not 1 <= len(subjects) <= 200:
        raise ValueError("cohort must contain between 1 and 200 subjects")
    destination = Path(output_root).expanduser().resolve() / cohort_id
    destination.mkdir(parents=True, exist_ok=True)
    target_subjects = destination / "subjects.txt"
    target_subjects.write_text("\n".join(subjects) + "\n", encoding="utf-8")
    payload = {
        "schema_version": 1,
        "cohort_id": cohort_id,
        "status": status,
        "subjects_file": "subjects.txt",
        "expected_subjects": len(subjects),
        "source_subjects_file": str(source),
        "subjects_sha256": sha256_file(target_subjects),
        "note": note,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    approved.write_json_atomic(destination / "cohort.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--study-config", required=True)
    preflight.add_argument("--cohort-config", required=True)
    preflight.add_argument("--campaign-root", required=True)
    preflight.add_argument("--targets-csv", required=True)
    preflight.add_argument("--map-manifest")
    preflight.add_argument("--study-root")
    preflight.add_argument("--scaffold-root")
    preflight.add_argument("--source-root", action="append")
    preflight.add_argument("--legacy-scaffold-manifest")
    preflight.add_argument("--max-array-elements", type=int, default=1000)
    preflight.add_argument("--mesh-workers", type=int, default=2)

    scaffold = subparsers.add_parser("scaffold-task")
    scaffold.add_argument("--manifest", required=True)
    scaffold.add_argument("--task-index", type=int, required=True)
    scaffold.add_argument("--charm-bin", default="charm")

    mesh = subparsers.add_parser("mesh-task")
    mesh.add_argument("--manifest", required=True)
    mesh.add_argument("--task-index", type=int, required=True)
    mesh.add_argument("--settings")
    mesh.add_argument("--staging-root")

    simulation = subparsers.add_parser("simulation-task")
    simulation.add_argument("--manifest", required=True)
    simulation.add_argument("--task-index", type=int, required=True)
    simulation.add_argument("--targets-csv", required=True)
    simulation.add_argument(
        "--expected-targets-sha256",
        default=CONFIRMED_TARGETS_SHA256,
    )
    simulation.add_argument("--individualized-targets-csv")
    simulation.add_argument(
        "--expected-individualized-targets-sha256"
    )
    simulation.add_argument("--simulation-runner", required=True)
    simulation.add_argument("--simulation-validator", required=True)
    simulation.add_argument("--python-bin", default="python")

    validate = subparsers.add_parser("validate")
    validate.add_argument(
        "--stage",
        choices=("scaffolds", "meshes", "simulations"),
        required=True,
    )
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--summary", required=True)
    validate.add_argument("--skip-hashes", action="store_true")

    register = subparsers.add_parser("create-cohort")
    register.add_argument("--cohort-id", required=True)
    register.add_argument("--subjects-file", required=True)
    register.add_argument("--output-root", required=True)
    register.add_argument("--status", default="provisional")
    register.add_argument("--note", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            payload = build_manifests(
                study_config=args.study_config,
                cohort_config=args.cohort_config,
                campaign_root=args.campaign_root,
                targets_csv=args.targets_csv,
                map_manifest=args.map_manifest,
                study_root=args.study_root,
                scaffold_root=args.scaffold_root,
                source_roots=args.source_root,
                legacy_scaffold_manifest=args.legacy_scaffold_manifest,
                max_array_elements=args.max_array_elements,
                mesh_workers=args.mesh_workers,
            )
        elif args.command == "scaffold-task":
            payload = run_scaffold_task(
                manifest=args.manifest,
                task_index=args.task_index,
                charm_bin=args.charm_bin,
            )
        elif args.command == "mesh-task":
            payload = run_mesh_task(
                manifest=args.manifest,
                task_index=args.task_index,
                settings_path=args.settings,
                staging_root=args.staging_root,
            )
        elif args.command == "simulation-task":
            payload = run_simulation_task(
                manifest=args.manifest,
                task_index=args.task_index,
                targets_csv=args.targets_csv,
                expected_targets_sha256=args.expected_targets_sha256,
                simulation_runner=args.simulation_runner,
                simulation_validator=args.simulation_validator,
                individualized_targets_csv=args.individualized_targets_csv,
                expected_individualized_targets_sha256=(
                    args.expected_individualized_targets_sha256
                ),
                python_bin=args.python_bin,
            )
        elif args.command == "validate":
            payload = validate_stage(
                stage=args.stage,
                manifest=args.manifest,
                summary=args.summary,
                verify_hashes=not args.skip_hashes,
            )
        else:
            payload = create_cohort(
                cohort_id=args.cohort_id,
                subjects_file=args.subjects_file,
                output_root=args.output_root,
                status=args.status,
                note=args.note,
            )
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0 if payload.get("status") in {
            "ready",
            "complete",
            "provisional",
            "final",
        } else 1
    except (
        IndexError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
