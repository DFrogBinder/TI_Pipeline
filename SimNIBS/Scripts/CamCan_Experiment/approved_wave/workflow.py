#!/usr/bin/env python3
"""Bootstrap CHARM once per approved subject, then simulate all repeats.

The bootstrap stage reconstructs one complete ``m2m_*`` tree per subject,
replaces CHARM's generated tissue map with the exact supervisor-reviewed map,
and meshes that map once.  The simulation stage reuses this immutable subject
support for every requested repeat.  No repeat simulation runs segmentation or
meshing, and ROAST/custom segmentation is never accepted.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.camcan_dataset import (  # noqa: E402
    CONFIRMED_TARGETS_SHA256,
    electrode_names_for_config,
    sha256_file,
    validate_dataset_montage,
)


SUBJECT_RE = re.compile(r"^sub-[A-Za-z0-9][A-Za-z0-9._-]*$")
MAP_BASENAME = "tissue_labeling_upsampled.nii.gz"
MAP_SUFFIX = "_CHARM_tissue_labeling_upsampled.nii.gz"
CAP_BASENAME = "EEG10-10_UI_Jurak_2007.csv"
PREP_FIELDS = (
    "task_id",
    "dataset_name",
    "subject",
    "source_anat",
    "source_t1",
    "source_t1_sha256",
    "source_t2",
    "source_t2_sha256",
    "approved_label",
    "approved_label_sha256",
    "dataset_root",
    "anat_dir",
    "m2m_dir",
    "mesh_path",
    "result_path",
    "status",
    "message",
)
SIMULATION_FIELDS = (
    "task_id",
    "dataset_name",
    "repeat_id",
    "dataset_root",
    "subject",
    "anat_dir",
    "canonical_anat_dir",
    "canonical_m2m_dir",
    "prep_result_path",
    "result_path",
    "status",
    "message",
)


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


def read_subjects(path: str | Path) -> list[str]:
    subjects: list[str] = []
    seen: set[str] = set()
    for raw in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        subject = raw.strip()
        if not subject or subject.startswith("#"):
            continue
        if not SUBJECT_RE.fullmatch(subject) or "/" in subject:
            raise ValueError(f"invalid subject identifier: {subject!r}")
        if subject in seen:
            raise ValueError(f"duplicate subject: {subject}")
        seen.add(subject)
        subjects.append(subject)
    return subjects


def _nifti_candidates(anat: Path, stem: str) -> tuple[Path, Path]:
    return anat / f"{stem}.nii", anat / f"{stem}.nii.gz"


def _resolve_single_nifti(anat: Path, stem: str) -> Path:
    matches = [path for path in _nifti_candidates(anat, stem) if path.is_file()]
    if len(matches) != 1:
        checked = ";".join(str(path) for path in _nifti_candidates(anat, stem))
        raise FileNotFoundError(
            f"expected exactly one {stem} NIfTI, found {len(matches)}; checked {checked}"
        )
    return matches[0].resolve(strict=True)


def resolve_source_inputs(
    source_roots: Sequence[Path], subject: str
) -> tuple[Path, Path, Path]:
    matches: list[tuple[Path, Path, Path]] = []
    problems: list[str] = []
    for root in source_roots:
        anat = root / subject / "anat"
        if not anat.is_dir():
            continue
        try:
            t1 = _resolve_single_nifti(anat, f"{subject}_T1w")
            t2 = _resolve_single_nifti(anat, f"{subject}_T2w")
        except FileNotFoundError as exc:
            problems.append(f"{anat}: {exc}")
            continue
        matches.append((anat.resolve(strict=True), t1, t2))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            "subject has complete MRI inputs under multiple source roots: "
            + ";".join(str(match[0]) for match in matches)
        )
    detail = "; ".join(problems) if problems else "subject directory was not found"
    raise FileNotFoundError(f"no complete T1/T2 source: {detail}")


def _cap_names(path: Path) -> set[str]:
    names: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if row:
                names.add(row[-1].strip())
    return names


def build_manifests(
    *,
    subjects_file: str | Path,
    source_roots: Sequence[str | Path],
    map_root: str | Path,
    output_root: str | Path,
    prep_result_dir: str | Path,
    simulation_result_dir: str | Path,
    prep_manifest: str | Path,
    simulation_manifest: str | Path,
    summary: str | Path,
    dataset_prefix: str,
    repeats: Sequence[str],
    expected_subjects: int,
    expected_prep_tasks: int,
    expected_simulation_tasks: int,
) -> dict[str, object]:
    if not repeats or len(repeats) != len(set(repeats)):
        raise ValueError("repeat identifiers must be non-empty and unique")
    subjects = read_subjects(subjects_file)
    roots = [Path(root).expanduser().resolve(strict=True) for root in source_roots]
    maps = Path(map_root).expanduser().resolve(strict=True)
    output = Path(output_root).expanduser().resolve()
    prep_results = Path(prep_result_dir).expanduser().resolve()
    simulation_results = Path(simulation_result_dir).expanduser().resolve()
    first_repeat = repeats[0]
    first_dataset_name = f"{dataset_prefix}_Data_{first_repeat}"
    first_dataset_root = output / first_dataset_name
    prep_rows: list[dict[str, object]] = []

    for task_id, subject in enumerate(subjects):
        messages: list[str] = []
        source_anat: Path | str = ""
        t1: Path | str = ""
        t2: Path | str = ""
        t1_hash = ""
        t2_hash = ""
        label: Path | str = maps / f"{subject}{MAP_SUFFIX}"
        label_hash = ""
        try:
            source_anat, t1, t2 = resolve_source_inputs(roots, subject)
            t1_hash = sha256_file(Path(t1))
            t2_hash = sha256_file(Path(t2))
        except (FileNotFoundError, OSError, ValueError) as exc:
            messages.append(str(exc))
        try:
            label = Path(label).resolve(strict=True)
            if not Path(label).is_file() or Path(label).is_symlink():
                raise ValueError(f"approved label is not a regular file: {label}")
            label_hash = sha256_file(Path(label))
            if Path(label).stat().st_size <= 0:
                raise ValueError(f"approved label is empty: {label}")
        except (FileNotFoundError, OSError, ValueError) as exc:
            messages.append(str(exc))
        anat_dir = first_dataset_root / subject / "anat"
        m2m_dir = anat_dir / f"m2m_{subject}"
        prep_rows.append(
            {
                "task_id": task_id,
                "dataset_name": first_dataset_name,
                "subject": subject,
                "source_anat": source_anat,
                "source_t1": t1,
                "source_t1_sha256": t1_hash,
                "source_t2": t2,
                "source_t2_sha256": t2_hash,
                "approved_label": label,
                "approved_label_sha256": label_hash,
                "dataset_root": first_dataset_root,
                "anat_dir": anat_dir,
                "m2m_dir": m2m_dir,
                "mesh_path": m2m_dir / f"{subject}.msh",
                "result_path": prep_results / f"{subject}.json",
                "status": "ready" if not messages else "blocked",
                "message": "ready" if not messages else "; ".join(messages),
            }
        )

    prep_by_subject = {str(row["subject"]): row for row in prep_rows}
    simulation_rows: list[dict[str, object]] = []
    for repeat in repeats:
        dataset_name = f"{dataset_prefix}_Data_{repeat}"
        dataset_root = output / dataset_name
        for subject in subjects:
            prep = prep_by_subject[subject]
            anat_dir = dataset_root / subject / "anat"
            simulation_rows.append(
                {
                    "task_id": len(simulation_rows),
                    "dataset_name": dataset_name,
                    "repeat_id": repeat,
                    "dataset_root": dataset_root,
                    "subject": subject,
                    "anat_dir": anat_dir,
                    "canonical_anat_dir": prep["anat_dir"],
                    "canonical_m2m_dir": prep["m2m_dir"],
                    "prep_result_path": prep["result_path"],
                    "result_path": simulation_results
                    / dataset_name
                    / f"{subject}.json",
                    "status": prep["status"],
                    "message": prep["message"],
                }
            )

    global_messages: list[str] = []
    expected_counts = (
        ("subject", len(subjects), expected_subjects),
        ("preparation task", len(prep_rows), expected_prep_tasks),
        ("simulation task", len(simulation_rows), expected_simulation_tasks),
    )
    for label, found, expected in expected_counts:
        if found != expected:
            global_messages.append(
                f"{label} count mismatch: found {found}, expected {expected}"
            )
    if global_messages:
        message = "; ".join(global_messages)
        for row in (*prep_rows, *simulation_rows):
            row["status"] = "blocked"
            row["message"] = "; ".join((str(row["message"]), message))

    prep_ready = sum(row["status"] == "ready" for row in prep_rows)
    simulation_ready = sum(row["status"] == "ready" for row in simulation_rows)
    ready = (
        prep_ready == expected_prep_tasks
        and simulation_ready == expected_simulation_tasks
    )
    payload: dict[str, object] = {
        "status": "ready" if ready else "blocked",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "subjects_file": str(Path(subjects_file).expanduser().resolve(strict=True)),
        "source_roots": [str(root) for root in roots],
        "map_root": str(maps),
        "output_root": str(output),
        "prep_manifest": str(Path(prep_manifest).expanduser().resolve()),
        "simulation_manifest": str(
            Path(simulation_manifest).expanduser().resolve()
        ),
        "subjects_found": len(subjects),
        "subjects_expected": expected_subjects,
        "repeats": list(repeats),
        "prep_tasks_found": len(prep_rows),
        "prep_tasks_expected": expected_prep_tasks,
        "prep_ready": prep_ready,
        "simulation_tasks_found": len(simulation_rows),
        "simulation_tasks_expected": expected_simulation_tasks,
        "simulation_ready": simulation_ready,
        "charm_segmentation_runs_expected": expected_subjects,
        "meshes_expected": expected_subjects,
        "fem_simulations_expected": expected_simulation_tasks,
        "segmentation_used_for_mesh": "exact supervisor-reviewed CHARM label",
        "roast_involvement": False,
        "execution": "one CHARM bootstrap and mesh per subject; FEM for all repeats",
    }
    write_tsv(Path(prep_manifest).expanduser().resolve(), PREP_FIELDS, prep_rows)
    write_tsv(
        Path(simulation_manifest).expanduser().resolve(),
        SIMULATION_FIELDS,
        simulation_rows,
    )
    write_json_atomic(Path(summary).expanduser().resolve(), payload)
    return payload


def _copy_verified(
    source: Path,
    destination: Path,
    expected_hash: str,
    *,
    replace_existing: bool = False,
) -> None:
    if sha256_file(source) != expected_hash:
        raise ValueError(f"source hash changed: {source}")
    if destination.is_file():
        if sha256_file(destination) == expected_hash:
            return
        if not replace_existing:
            raise ValueError(f"existing destination differs from source: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        if sha256_file(temporary) != expected_hash:
            raise IOError(f"staged copy hash mismatch: {temporary}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _default_mesh_validator(mesh_path: Path) -> dict[str, object]:
    from charm_only_remesh.workflow import validate_mesh_payload

    return validate_mesh_payload(mesh_path, load_mesh=True)


def _run_command(
    command: Sequence[str], *, cwd: Path, env: dict[str, str] | None = None
) -> None:
    print(
        json.dumps(
            {"event": "run_command", "command": list(command), "cwd": str(cwd)}
        ),
        flush=True,
    )
    subprocess.run(list(command), cwd=cwd, env=env, check=True)


def prep_result_is_current(
    row: dict[str, str], required_electrodes: Sequence[str]
) -> dict[str, object] | None:
    payload = _load_json(Path(row["result_path"]))
    if payload is None or payload.get("status") != "complete":
        return None
    label = Path(row["m2m_dir"]) / "label_prep" / MAP_BASENAME
    cap = Path(row["m2m_dir"]) / "eeg_positions" / CAP_BASENAME
    mesh = Path(row["mesh_path"])
    try:
        t1_path = Path(str(payload["task_t1_path"]))
        t2_path = Path(str(payload["task_t2_path"]))
        checks = (
            sha256_file(label) == row["approved_label_sha256"],
            sha256_file(mesh) == payload.get("mesh_sha256"),
            sha256_file(cap) == payload.get("eeg_cap_sha256"),
            sha256_file(t1_path) == payload.get("task_t1_sha256"),
            sha256_file(t2_path) == payload.get("task_t2_sha256"),
            set(required_electrodes).issubset(_cap_names(cap)),
        )
        if not all(checks):
            return None
    except (FileNotFoundError, KeyError, OSError, UnicodeError, csv.Error):
        return None
    return payload


def run_prep_task(
    *,
    manifest: str | Path,
    task_index: int,
    montage_preset: str,
    targets_csv: str | Path,
    expected_targets_sha256: str,
    charm_bin: str = "charm",
    command_runner: Callable[..., None] = _run_command,
    mesh_validator: Callable[[Path], dict[str, object]] = _default_mesh_validator,
) -> dict[str, object]:
    rows = read_tsv(manifest)
    if task_index < 0 or task_index >= len(rows):
        raise IndexError(f"prep task index {task_index} is outside {len(rows)} rows")
    row = rows[task_index]
    if row.get("status") != "ready" or int(row["task_id"]) != task_index:
        raise ValueError(f"prep task is blocked or misindexed: {row.get('message', '')}")
    targets = Path(targets_csv).expanduser().resolve(strict=True)
    actual_targets_hash = sha256_file(targets)
    if actual_targets_hash != expected_targets_sha256:
        raise ValueError(
            f"targets.csv hash mismatch: {actual_targets_hash} != {expected_targets_sha256}"
        )
    config = validate_dataset_montage([row["dataset_name"]], montage_preset)
    required_electrodes = electrode_names_for_config(config, targets)
    current = prep_result_is_current(row, required_electrodes)
    if current is not None:
        print(json.dumps({"event": "approved_prep_reused", **current}), flush=True)
        return current

    sources = (
        (Path(row["source_t1"]), row["source_t1_sha256"], "T1"),
        (Path(row["source_t2"]), row["source_t2_sha256"], "T2"),
        (Path(row["approved_label"]), row["approved_label_sha256"], "label"),
    )
    for source, expected_hash, label in sources:
        actual = sha256_file(source)
        if actual != expected_hash:
            raise ValueError(f"{label} source hash changed: {actual} != {expected_hash}")

    anat_dir = Path(row["anat_dir"])
    m2m_dir = Path(row["m2m_dir"])
    if m2m_dir.is_symlink():
        raise ValueError(f"refusing symlinked bootstrap m2m directory: {m2m_dir}")
    subject_root = anat_dir.parent
    dataset_root = Path(row["dataset_root"]).resolve()
    if subject_root.exists():
        if not subject_root.resolve().is_relative_to(dataset_root):
            raise ValueError(f"refusing to clear bootstrap subject: {subject_root}")
        shutil.rmtree(subject_root)
    Path(row["result_path"]).unlink(missing_ok=True)
    anat_dir.mkdir(parents=True, exist_ok=True)

    source_t1 = Path(row["source_t1"])
    source_t2 = Path(row["source_t2"])
    t1_suffix = ".nii.gz" if source_t1.name.endswith(".nii.gz") else ".nii"
    t2_suffix = ".nii.gz" if source_t2.name.endswith(".nii.gz") else ".nii"
    task_t1 = anat_dir / f"{row['subject']}_T1w{t1_suffix}"
    task_t2 = anat_dir / f"{row['subject']}_T2w{t2_suffix}"
    _copy_verified(source_t1, task_t1, row["source_t1_sha256"])
    _copy_verified(source_t2, task_t2, row["source_t2_sha256"])

    segment_command = [
        charm_bin,
        row["subject"],
        str(task_t1),
        str(task_t2),
        "--registerT2",
        "--initatlas",
        "--segment",
        "--forceqform",
    ]
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    command_runner(segment_command, cwd=anat_dir)
    installed_label = m2m_dir / "label_prep" / MAP_BASENAME
    generated_label_hash = sha256_file(installed_label)
    _copy_verified(
        Path(row["approved_label"]),
        installed_label,
        row["approved_label_sha256"],
        replace_existing=True,
    )
    if sha256_file(installed_label) != row["approved_label_sha256"]:
        raise RuntimeError("approved label installation failed")

    mesh_command = [charm_bin, row["subject"], "--mesh"]
    command_runner(mesh_command, cwd=anat_dir)
    if sha256_file(installed_label) != row["approved_label_sha256"]:
        Path(row["mesh_path"]).unlink(missing_ok=True)
        raise RuntimeError("meshing changed the approved label")
    mesh_payload = mesh_validator(Path(row["mesh_path"]))
    cap = m2m_dir / "eeg_positions" / CAP_BASENAME
    missing = sorted(set(required_electrodes) - _cap_names(cap))
    if missing:
        raise ValueError("transformed EEG cap lacks: " + ",".join(missing))

    payload: dict[str, object] = {
        "schema_version": 2,
        "status": "complete",
        "task_index": task_index,
        "dataset_name": row["dataset_name"],
        "subject": row["subject"],
        "approved_label": row["approved_label"],
        "approved_label_sha256": row["approved_label_sha256"],
        "generated_label_sha256_before_approved_install": generated_label_hash,
        "task_t1_path": str(task_t1),
        "task_t1_sha256": sha256_file(task_t1),
        "task_t2_path": str(task_t2),
        "task_t2_sha256": sha256_file(task_t2),
        "installed_label": str(installed_label),
        "installed_label_sha256": sha256_file(installed_label),
        "mesh_sha256": sha256_file(Path(row["mesh_path"])),
        "eeg_cap_path": str(cap),
        "eeg_cap_sha256": sha256_file(cap),
        "required_electrodes": list(required_electrodes),
        "segmentation_command": segment_command,
        "mesh_command": mesh_command,
        "segmentation_runs_for_subject": 1,
        "roast_involvement": False,
        "started_at": started_at,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **mesh_payload,
    }
    write_json_atomic(Path(row["result_path"]), payload)
    print(json.dumps({"event": "approved_prep_complete", **payload}), flush=True)
    return payload


def _ensure_relative_symlink(source: Path, destination: Path) -> None:
    resolved_source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve(strict=True) == resolved_source:
            return
        destination.unlink()
    elif destination.exists():
        raise ValueError(f"refusing to replace non-symlink path: {destination}")
    relative = os.path.relpath(resolved_source, destination.parent.resolve())
    destination.symlink_to(relative, target_is_directory=resolved_source.is_dir())


def materialize_repeat_subject(row: dict[str, str]) -> None:
    anat_dir = Path(row["anat_dir"])
    canonical_anat = Path(row["canonical_anat_dir"])
    if anat_dir.resolve() == canonical_anat.resolve():
        return
    subject = row["subject"]
    t1_candidates = sorted(canonical_anat.glob(f"{subject}_T1w.nii*"))
    t2_candidates = sorted(canonical_anat.glob(f"{subject}_T2w.nii*"))
    if len(t1_candidates) != 1 or len(t2_candidates) != 1:
        raise FileNotFoundError("canonical T1/T2 inputs are missing or ambiguous")
    _ensure_relative_symlink(t1_candidates[0], anat_dir / t1_candidates[0].name)
    _ensure_relative_symlink(t2_candidates[0], anat_dir / t2_candidates[0].name)
    _ensure_relative_symlink(
        Path(row["canonical_m2m_dir"]), anat_dir / f"m2m_{subject}"
    )


def run_simulation_task(
    *,
    manifest: str | Path,
    task_index: int,
    montage_preset: str,
    targets_csv: str | Path,
    expected_targets_sha256: str,
    simulation_runner: str | Path,
    simulation_validator: str | Path,
    python_bin: str = "python",
    command_runner: Callable[..., None] = _run_command,
) -> dict[str, object]:
    rows = read_tsv(manifest)
    if task_index < 0 or task_index >= len(rows):
        raise IndexError(
            f"simulation task index {task_index} is outside {len(rows)} rows"
        )
    row = rows[task_index]
    if row.get("status") != "ready" or int(row["task_id"]) != task_index:
        raise ValueError(
            f"simulation task is blocked or misindexed: {row.get('message', '')}"
        )
    targets = Path(targets_csv).expanduser().resolve(strict=True)
    actual_targets_hash = sha256_file(targets)
    if actual_targets_hash != expected_targets_sha256:
        raise ValueError(
            f"targets.csv hash mismatch: {actual_targets_hash} != {expected_targets_sha256}"
        )
    config = validate_dataset_montage([row["dataset_name"]], montage_preset)
    required_electrodes = electrode_names_for_config(config, targets)
    prep_marker = _load_json(Path(row["prep_result_path"]))
    if prep_marker is None:
        raise ValueError(f"bootstrap marker is absent: {row['prep_result_path']}")
    prep_row = {
        "result_path": row["prep_result_path"],
        "m2m_dir": row["canonical_m2m_dir"],
        "mesh_path": str(Path(row["canonical_m2m_dir"]) / f"{row['subject']}.msh"),
        "approved_label_sha256": str(prep_marker.get("approved_label_sha256", "")),
    }
    prep_payload = prep_result_is_current(prep_row, required_electrodes)
    if prep_payload is None:
        raise ValueError(
            f"bootstrap result is absent or invalid: {row['prep_result_path']}"
        )
    materialize_repeat_subject(row)

    result_path = Path(row["result_path"])
    simulation_env = os.environ.copy()
    simulation_env["TI_SIM_ROOT"] = row["dataset_root"]
    validation_command = [
        python_bin,
        "-u",
        str(Path(simulation_validator).expanduser().resolve(strict=True)),
        "--root",
        row["dataset_root"],
        "--subject",
        row["subject"],
    ]
    existing = _load_json(result_path)
    reusable = bool(
        existing
        and existing.get("status") == "complete"
        and existing.get("dataset_name") == row["dataset_name"]
        and existing.get("repeat_id") == row["repeat_id"]
        and existing.get("subject") == row["subject"]
        and existing.get("mesh_sha256") == prep_payload.get("mesh_sha256")
        and existing.get("approved_label_sha256")
        == prep_payload.get("approved_label_sha256")
        and existing.get("montage_preset") == montage_preset
        and existing.get("targets_csv_sha256") == actual_targets_hash
    )
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if reusable:
        command_runner(
            validation_command, cwd=Path(row["anat_dir"]), env=simulation_env
        )
        existing["status"] = "already_complete"
        print(
            json.dumps({"event": "approved_simulation_reused", **existing}),
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
        montage_preset,
        "--reuse-existing-mesh",
    ]
    command_runner(
        simulation_command, cwd=Path(row["anat_dir"]), env=simulation_env
    )
    command_runner(
        validation_command, cwd=Path(row["anat_dir"]), env=simulation_env
    )
    payload: dict[str, object] = {
        "schema_version": 2,
        "status": "complete",
        "task_index": task_index,
        "dataset_name": row["dataset_name"],
        "repeat_id": row["repeat_id"],
        "subject": row["subject"],
        "mesh_path": prep_payload["mesh_path"],
        "mesh_sha256": prep_payload["mesh_sha256"],
        "approved_label_sha256": prep_payload["approved_label_sha256"],
        "montage_preset": montage_preset,
        "targets_csv": str(targets),
        "targets_csv_sha256": actual_targets_hash,
        "simulation_command": simulation_command,
        "validation_command": validation_command,
        "segmentation_or_meshing_in_simulation_task": False,
        "started_at": started_at,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json_atomic(result_path, payload)
    print(json.dumps({"event": "approved_simulation_complete", **payload}), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--subjects-file", required=True)
    preflight.add_argument("--source-root", action="append", required=True)
    preflight.add_argument("--map-root", required=True)
    preflight.add_argument("--output-root", required=True)
    preflight.add_argument("--prep-result-dir", required=True)
    preflight.add_argument("--simulation-result-dir", required=True)
    preflight.add_argument("--prep-manifest", required=True)
    preflight.add_argument("--simulation-manifest", required=True)
    preflight.add_argument("--summary", required=True)
    preflight.add_argument("--dataset-prefix", default="Left_Hippocampus")
    preflight.add_argument(
        "--repeats", nargs="+", default=[f"{i:02d}" for i in range(1, 11)]
    )
    preflight.add_argument("--expected-subjects", type=int, default=89)
    preflight.add_argument("--expected-prep-tasks", type=int, default=89)
    preflight.add_argument("--expected-simulation-tasks", type=int, default=890)

    prep = subparsers.add_parser("prep-task")
    prep.add_argument("--manifest", required=True)
    prep.add_argument("--task-index", type=int, required=True)
    prep.add_argument("--montage-preset", default="left-hippocampus")
    prep.add_argument("--targets-csv", required=True)
    prep.add_argument(
        "--expected-targets-sha256", default=CONFIRMED_TARGETS_SHA256
    )
    prep.add_argument("--charm-bin", default="charm")

    simulation = subparsers.add_parser("simulation-task")
    simulation.add_argument("--manifest", required=True)
    simulation.add_argument("--task-index", type=int, required=True)
    simulation.add_argument("--montage-preset", default="left-hippocampus")
    simulation.add_argument("--targets-csv", required=True)
    simulation.add_argument(
        "--expected-targets-sha256", default=CONFIRMED_TARGETS_SHA256
    )
    simulation.add_argument("--simulation-runner", required=True)
    simulation.add_argument("--simulation-validator", required=True)
    simulation.add_argument("--python-bin", default="python")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "preflight":
        payload = build_manifests(
            subjects_file=args.subjects_file,
            source_roots=args.source_root,
            map_root=args.map_root,
            output_root=args.output_root,
            prep_result_dir=args.prep_result_dir,
            simulation_result_dir=args.simulation_result_dir,
            prep_manifest=args.prep_manifest,
            simulation_manifest=args.simulation_manifest,
            summary=args.summary,
            dataset_prefix=args.dataset_prefix,
            repeats=args.repeats,
            expected_subjects=args.expected_subjects,
            expected_prep_tasks=args.expected_prep_tasks,
            expected_simulation_tasks=args.expected_simulation_tasks,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "ready" else 2
    if args.command == "prep-task":
        payload = run_prep_task(
            manifest=args.manifest,
            task_index=args.task_index,
            montage_preset=args.montage_preset,
            targets_csv=args.targets_csv,
            expected_targets_sha256=args.expected_targets_sha256,
            charm_bin=args.charm_bin,
        )
    else:
        payload = run_simulation_task(
            manifest=args.manifest,
            task_index=args.task_index,
            montage_preset=args.montage_preset,
            targets_csv=args.targets_csv,
            expected_targets_sha256=args.expected_targets_sha256,
            simulation_runner=args.simulation_runner,
            simulation_validator=args.simulation_validator,
            python_bin=args.python_bin,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
