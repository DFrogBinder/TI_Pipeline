#!/usr/bin/env python3
"""Prepare and run a fresh approved-subject CHARM-only simulation wave.

Each task creates a self-contained repeat workspace from T1/T2, runs only the
CHARM prerequisites needed for a simulation-ready ``m2m_*`` tree, atomically
installs the exact QC-approved CHARM tissue map, meshes it, and invokes the
existing validated CamCan TI simulation runner.  Completed mesh provenance is
retained independently from simulation completion so a requeued simulation
does not silently generate a different repeat mesh.
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
MANIFEST_FIELDS = (
    "task_id",
    "dataset_name",
    "repeat_id",
    "dataset_root",
    "subject",
    "source_anat",
    "source_t1",
    "source_t1_sha256",
    "source_t2",
    "source_t2_sha256",
    "approved_label",
    "approved_label_sha256",
    "anat_dir",
    "m2m_dir",
    "mesh_path",
    "mesh_result_path",
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


def write_tsv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_subjects(path: str | Path) -> list[str]:
    subjects: list[str] = []
    for raw in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        subject = raw.strip()
        if not subject or subject.startswith("#"):
            continue
        if not SUBJECT_RE.fullmatch(subject) or "/" in subject:
            raise ValueError(f"invalid subject identifier: {subject!r}")
        subjects.append(subject)
    duplicates = sorted({subject for subject in subjects if subjects.count(subject) > 1})
    if duplicates:
        raise ValueError("duplicate subject(s): " + ", ".join(duplicates))
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


def resolve_source_inputs(source_roots: Sequence[Path], subject: str) -> tuple[Path, Path, Path]:
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


def build_manifest(
    *,
    subjects_file: str | Path,
    source_roots: Sequence[str | Path],
    map_root: str | Path,
    output_root: str | Path,
    result_dir: str | Path,
    manifest: str | Path,
    summary: str | Path,
    dataset_prefix: str,
    repeats: Sequence[str],
    expected_subjects: int,
    expected_tasks: int,
) -> dict[str, object]:
    subjects = read_subjects(subjects_file)
    roots = [Path(root).expanduser().resolve(strict=True) for root in source_roots]
    maps = Path(map_root).expanduser().resolve(strict=True)
    output = Path(output_root).expanduser().resolve()
    results = Path(result_dir).expanduser().resolve()
    rows: list[dict[str, object]] = []
    subject_inputs: dict[str, dict[str, object]] = {}

    for subject in subjects:
        messages: list[str] = []
        source_anat: Path | str = ""
        t1: Path | str = ""
        t2: Path | str = ""
        t1_hash = ""
        t2_hash = ""
        label = maps / f"{subject}{MAP_SUFFIX}"
        label_hash = ""
        try:
            source_anat, t1, t2 = resolve_source_inputs(roots, subject)
            t1_hash = sha256_file(Path(t1))
            t2_hash = sha256_file(Path(t2))
        except (FileNotFoundError, OSError, ValueError) as exc:
            messages.append(str(exc))
        try:
            label = label.resolve(strict=True)
            if not label.is_file() or label.is_symlink():
                raise ValueError(f"approved label is not a regular file: {label}")
            label_hash = sha256_file(label)
            if label.stat().st_size <= 0:
                raise ValueError(f"approved label is empty: {label}")
        except (FileNotFoundError, OSError, ValueError) as exc:
            messages.append(str(exc))
        subject_inputs[subject] = {
            "source_anat": source_anat,
            "t1": t1,
            "t2": t2,
            "t1_hash": t1_hash,
            "t2_hash": t2_hash,
            "label": label,
            "label_hash": label_hash,
            "messages": messages,
        }

    for repeat in repeats:
        dataset_name = f"{dataset_prefix}_Data_{repeat}"
        dataset_root = output / dataset_name
        for subject in subjects:
            inputs = subject_inputs[subject]
            anat_dir = dataset_root / subject / "anat"
            m2m_dir = anat_dir / f"m2m_{subject}"
            task_result_dir = results / dataset_name
            messages = list(inputs["messages"])
            rows.append(
                {
                    "task_id": len(rows),
                    "dataset_name": dataset_name,
                    "repeat_id": repeat,
                    "dataset_root": dataset_root,
                    "subject": subject,
                    "source_anat": inputs["source_anat"],
                    "source_t1": inputs["t1"],
                    "source_t1_sha256": inputs["t1_hash"],
                    "source_t2": inputs["t2"],
                    "source_t2_sha256": inputs["t2_hash"],
                    "approved_label": inputs["label"],
                    "approved_label_sha256": inputs["label_hash"],
                    "anat_dir": anat_dir,
                    "m2m_dir": m2m_dir,
                    "mesh_path": m2m_dir / f"{subject}.msh",
                    "mesh_result_path": task_result_dir / f"{subject}.mesh.json",
                    "result_path": task_result_dir / f"{subject}.json",
                    "status": "ready" if not messages else "blocked",
                    "message": "ready" if not messages else "; ".join(messages),
                }
            )

    global_messages: list[str] = []
    if len(subjects) != expected_subjects:
        global_messages.append(
            f"subject count mismatch: found {len(subjects)}, expected {expected_subjects}"
        )
    if len(rows) != expected_tasks:
        global_messages.append(
            f"task count mismatch: found {len(rows)}, expected {expected_tasks}"
        )
    if global_messages:
        for row in rows:
            row["status"] = "blocked"
            row["message"] = "; ".join(
                [message for message in (str(row["message"]), *global_messages) if message]
            )

    ready = sum(row["status"] == "ready" for row in rows)
    payload: dict[str, object] = {
        "status": "ready" if ready == expected_tasks else "blocked",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "subjects_file": str(Path(subjects_file).expanduser().resolve(strict=True)),
        "source_roots": [str(root) for root in roots],
        "map_root": str(maps),
        "output_root": str(output),
        "result_dir": str(results),
        "manifest": str(Path(manifest).expanduser().resolve()),
        "dataset_prefix": dataset_prefix,
        "subjects_found": len(subjects),
        "subjects_expected": expected_subjects,
        "repeats": list(repeats),
        "tasks_found": len(rows),
        "tasks_expected": expected_tasks,
        "ready": ready,
        "blocked": len(rows) - ready,
        "execution": "approved_label -> CHARM support -> mesh -> Left Hippocampus simulation",
        "segmentation_used_for_mesh": "exact approved flat CHARM label",
        "roast_involvement": False,
    }
    write_tsv(Path(manifest).expanduser().resolve(), rows)
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


def _cap_names(path: Path) -> set[str]:
    names: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if row:
                names.add(row[-1].strip())
    return names


def _load_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _mesh_marker_is_current(
    row: dict[str, str], required_electrodes: Sequence[str]
) -> dict[str, object] | None:
    marker_path = Path(row["mesh_result_path"])
    marker = _load_json(marker_path)
    if marker is None or marker.get("status") != "complete":
        return None
    label = Path(row["m2m_dir"]) / "label_prep" / MAP_BASENAME
    mesh = Path(row["mesh_path"])
    cap = Path(row["m2m_dir"]) / "eeg_positions" / CAP_BASENAME
    try:
        if sha256_file(label) != row["approved_label_sha256"]:
            return None
        if sha256_file(mesh) != marker.get("mesh_sha256"):
            return None
        if sha256_file(cap) != marker.get("eeg_cap_sha256"):
            return None
        if not set(required_electrodes).issubset(_cap_names(cap)):
            return None
    except (FileNotFoundError, OSError):
        return None
    return marker


def _default_mesh_validator(mesh_path: Path) -> dict[str, object]:
    from charm_only_remesh.workflow import validate_mesh_payload

    return validate_mesh_payload(mesh_path, load_mesh=True)


def _run_command(command: Sequence[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print(json.dumps({"event": "run_command", "command": list(command), "cwd": str(cwd)}), flush=True)
    subprocess.run(list(command), cwd=cwd, env=env, check=True)


def run_task(
    *,
    manifest: str | Path,
    task_index: int,
    montage_preset: str,
    targets_csv: str | Path,
    expected_targets_sha256: str,
    simulation_runner: str | Path,
    simulation_validator: str | Path,
    charm_bin: str = "charm",
    python_bin: str = "python",
    command_runner: Callable[..., None] = _run_command,
    mesh_validator: Callable[[Path], dict[str, object]] = _default_mesh_validator,
) -> dict[str, object]:
    rows = read_tsv(manifest)
    if task_index < 0 or task_index >= len(rows):
        raise IndexError(f"task index {task_index} is outside manifest with {len(rows)} rows")
    row = rows[task_index]
    if row.get("status") != "ready":
        raise ValueError(f"task is blocked: {row.get('message', '')}")
    if int(row["task_id"]) != task_index:
        raise ValueError(f"manifest task_id mismatch at index {task_index}")

    targets = Path(targets_csv).expanduser().resolve(strict=True)
    actual_targets_hash = sha256_file(targets)
    if actual_targets_hash != expected_targets_sha256:
        raise ValueError(
            f"targets.csv hash mismatch: {actual_targets_hash} != {expected_targets_sha256}"
        )
    config = validate_dataset_montage([row["dataset_name"]], montage_preset)
    required_electrodes = electrode_names_for_config(config, targets)

    result_path = Path(row["result_path"])
    existing_result = _load_json(result_path)
    mesh_marker = _mesh_marker_is_current(row, required_electrodes)
    result_reusable = bool(
        existing_result
        and existing_result.get("status") == "complete"
        and mesh_marker is not None
        and existing_result.get("mesh_sha256") == mesh_marker.get("mesh_sha256")
    )

    source_t1 = Path(row["source_t1"])
    source_t2 = Path(row["source_t2"])
    approved_label = Path(row["approved_label"])
    for source, expected_hash, label in (
        (source_t1, row["source_t1_sha256"], "T1"),
        (source_t2, row["source_t2_sha256"], "T2"),
        (approved_label, row["approved_label_sha256"], "approved_label"),
    ):
        actual = sha256_file(source)
        if actual != expected_hash:
            raise ValueError(f"{label} source hash changed: {actual} != {expected_hash}")

    anat_dir = Path(row["anat_dir"])
    m2m_dir = Path(row["m2m_dir"])
    if m2m_dir.is_symlink():
        raise ValueError(f"refusing symlinked m2m directory: {m2m_dir}")
    anat_dir.mkdir(parents=True, exist_ok=True)
    t1_suffix = ".nii.gz" if source_t1.name.endswith(".nii.gz") else ".nii"
    t2_suffix = ".nii.gz" if source_t2.name.endswith(".nii.gz") else ".nii"
    task_t1 = anat_dir / f"{row['subject']}_T1w{t1_suffix}"
    task_t2 = anat_dir / f"{row['subject']}_T2w{t2_suffix}"
    _copy_verified(source_t1, task_t1, row["source_t1_sha256"])
    _copy_verified(source_t2, task_t2, row["source_t2_sha256"])

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if mesh_marker is None:
        result_path.unlink(missing_ok=True)
        if m2m_dir.exists():
            if not m2m_dir.resolve().is_relative_to(anat_dir.resolve()):
                raise ValueError(f"refusing to remove m2m directory outside anat: {m2m_dir}")
            shutil.rmtree(m2m_dir)
        Path(row["mesh_result_path"]).unlink(missing_ok=True)

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
        command_runner(segment_command, cwd=anat_dir)
        installed_label = m2m_dir / "label_prep" / MAP_BASENAME
        generated_label_hash = sha256_file(installed_label)
        _copy_verified(
            approved_label,
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
            raise RuntimeError("CHARM meshing changed the approved label")
        mesh_payload = mesh_validator(Path(row["mesh_path"]))
        cap_path = m2m_dir / "eeg_positions" / CAP_BASENAME
        cap_names = _cap_names(cap_path)
        missing_electrodes = sorted(set(required_electrodes) - cap_names)
        if missing_electrodes:
            raise ValueError(
                "transformed EEG cap is missing optimized electrode(s): "
                + ",".join(missing_electrodes)
            )
        mesh_marker = {
            "schema_version": 1,
            "status": "complete",
            "task_index": task_index,
            "dataset_name": row["dataset_name"],
            "repeat_id": row["repeat_id"],
            "subject": row["subject"],
            "source_t1": str(source_t1),
            "source_t1_sha256": row["source_t1_sha256"],
            "source_t2": str(source_t2),
            "source_t2_sha256": row["source_t2_sha256"],
            "approved_label": str(approved_label),
            "approved_label_sha256": row["approved_label_sha256"],
            "generated_label_sha256_before_approved_install": generated_label_hash,
            "installed_label": str(installed_label),
            "installed_label_sha256": sha256_file(installed_label),
            "segmentation_command": segment_command,
            "mesh_command": mesh_command,
            "eeg_cap_path": str(cap_path),
            "eeg_cap_sha256": sha256_file(cap_path),
            "required_electrodes": list(required_electrodes),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **mesh_payload,
        }
        write_json_atomic(Path(row["mesh_result_path"]), mesh_marker)
        print(json.dumps({"event": "approved_wave_mesh_complete", **mesh_marker}), flush=True)
    else:
        print(
            json.dumps(
                {
                    "event": "approved_wave_mesh_reused",
                    "task_index": task_index,
                    "dataset_name": row["dataset_name"],
                    "subject": row["subject"],
                    "mesh_path": row["mesh_path"],
                    "mesh_sha256": mesh_marker["mesh_sha256"],
                }
            ),
            flush=True,
        )

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
    if result_reusable and existing_result is not None:
        command_runner(validation_command, cwd=anat_dir, env=simulation_env)
        existing_result["status"] = "already_complete"
        print(
            json.dumps({"event": "approved_wave_task_reused", **existing_result}),
            flush=True,
        )
        return existing_result

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
    command_runner(simulation_command, cwd=anat_dir, env=simulation_env)
    command_runner(validation_command, cwd=anat_dir, env=simulation_env)

    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "task_index": task_index,
        "dataset_name": row["dataset_name"],
        "repeat_id": row["repeat_id"],
        "subject": row["subject"],
        "approved_label": row["approved_label"],
        "approved_label_sha256": row["approved_label_sha256"],
        "mesh_path": row["mesh_path"],
        "mesh_sha256": mesh_marker["mesh_sha256"],
        "mesh_result_path": row["mesh_result_path"],
        "montage_preset": montage_preset,
        "targets_csv": str(targets),
        "targets_csv_sha256": actual_targets_hash,
        "simulation_command": simulation_command,
        "validation_command": validation_command,
        "started_at": started_at,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json_atomic(result_path, payload)
    print(json.dumps({"event": "approved_wave_task_complete", **payload}), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--subjects-file", required=True)
    preflight.add_argument("--source-root", action="append", required=True)
    preflight.add_argument("--map-root", required=True)
    preflight.add_argument("--output-root", required=True)
    preflight.add_argument("--result-dir", required=True)
    preflight.add_argument("--manifest", required=True)
    preflight.add_argument("--summary", required=True)
    preflight.add_argument("--dataset-prefix", default="Left_Hippocampus")
    preflight.add_argument(
        "--repeats",
        nargs="+",
        default=[f"{repeat:02d}" for repeat in range(1, 11)],
    )
    preflight.add_argument("--expected-subjects", type=int, default=89)
    preflight.add_argument("--expected-tasks", type=int, default=890)

    task = subparsers.add_parser("run-task")
    task.add_argument("--manifest", required=True)
    task.add_argument("--task-index", type=int, required=True)
    task.add_argument("--montage-preset", default="left-hippocampus")
    task.add_argument("--targets-csv", required=True)
    task.add_argument(
        "--expected-targets-sha256", default=CONFIRMED_TARGETS_SHA256
    )
    task.add_argument("--simulation-runner", required=True)
    task.add_argument("--simulation-validator", required=True)
    task.add_argument("--charm-bin", default="charm")
    task.add_argument("--python-bin", default="python")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "preflight":
        payload = build_manifest(
            subjects_file=args.subjects_file,
            source_roots=args.source_root,
            map_root=args.map_root,
            output_root=args.output_root,
            result_dir=args.result_dir,
            manifest=args.manifest,
            summary=args.summary,
            dataset_prefix=args.dataset_prefix,
            repeats=args.repeats,
            expected_subjects=args.expected_subjects,
            expected_tasks=args.expected_tasks,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "ready" else 2
    payload = run_task(
        manifest=args.manifest,
        task_index=args.task_index,
        montage_preset=args.montage_preset,
        targets_csv=args.targets_csv,
        expected_targets_sha256=args.expected_targets_sha256,
        simulation_runner=args.simulation_runner,
        simulation_validator=args.simulation_validator,
        charm_bin=args.charm_bin,
        python_bin=args.python_bin,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
