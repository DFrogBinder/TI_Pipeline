#!/usr/bin/env python3
"""Prepare, clean, and validate in-place CamCan mesh-reuse reruns."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.mesh_reuse import resolve_existing_mesh  # noqa: E402
from simulation.validate_simulation_outputs import validate_subject_outputs  # noqa: E402


REPEAT_DATASET_PATTERN = re.compile(r"^(?P<roi_prefix>.+)_Data_(?P<repeat>\d+)$")
DEFAULT_DATASET_GLOB = "*_Data_*"
DEFAULT_SUBJECT_GLOB = "sub-*"
TASK_FIELDNAMES = (
    "task_id",
    "dataset_name",
    "repeat_id",
    "dataset_root",
    "subject",
    "anat_dir",
    "mesh_path",
    "status",
    "message",
)
CLEANUP_FIELDNAMES = (
    "scope",
    "dataset_name",
    "subject",
    "path",
    "kind",
    "action",
    "message",
)
VALIDATION_FIELDNAMES = (
    "task_id",
    "dataset_name",
    "repeat_id",
    "dataset_root",
    "subject",
    "status",
    "missing_outputs",
    "failure_reasons",
)


@dataclass(frozen=True)
class RepeatDataset:
    root: Path
    name: str
    roi_prefix: str
    repeat_id: str


def _parse_repeat(value: str | int) -> int:
    try:
        parsed = int(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"Invalid repeat identifier {value!r}; expected an integer.") from exc
    if parsed < 0:
        raise ValueError(f"Repeat identifier must be non-negative, got {parsed}.")
    return parsed


def _sanitize_cell(value: object) -> str:
    return str(value).replace("\t", " ").replace("\n", " ").strip()


def _join_cells(values: Iterable[object]) -> str:
    return ";".join(_sanitize_cell(value) for value in values)


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: str | Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> Path:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _sanitize_cell(row.get(field, "")) for field in fieldnames})
    return out_path


def discover_repeat_datasets(
    roi_root: str | Path,
    *,
    dataset_glob: str = DEFAULT_DATASET_GLOB,
    repeats: Iterable[str] | None = None,
) -> tuple[RepeatDataset, ...]:
    root = Path(roi_root).expanduser().resolve()
    selected_repeats = None if repeats is None else {_parse_repeat(item) for item in repeats}
    datasets: list[RepeatDataset] = []

    for path in root.glob(dataset_glob):
        if not path.is_dir():
            continue
        match = REPEAT_DATASET_PATTERN.match(path.name)
        if not match:
            continue
        repeat_value = _parse_repeat(match.group("repeat"))
        if selected_repeats is not None and repeat_value not in selected_repeats:
            continue
        datasets.append(
            RepeatDataset(
                root=path.resolve(),
                name=path.name,
                roi_prefix=match.group("roi_prefix"),
                repeat_id=match.group("repeat"),
            )
        )

    datasets.sort(
        key=lambda item: (
            item.roi_prefix.casefold(),
            _parse_repeat(item.repeat_id),
            item.name.casefold(),
        )
    )
    return tuple(datasets)


def _read_subjects_file(path: str | Path) -> tuple[str, ...]:
    subjects: list[str] = []
    seen: set[str] = set()
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            subject = line.strip()
            if not subject or subject.startswith("#"):
                continue
            if " " in subject or "\t" in subject:
                raise ValueError(f"Invalid subject entry at {path}:{line_number}: {subject!r}")
            if subject not in seen:
                seen.add(subject)
                subjects.append(subject)
    return tuple(subjects)


def discover_expected_subjects(
    datasets: Sequence[RepeatDataset],
    *,
    subjects: Sequence[str] | None = None,
    subjects_file: str | Path | None = None,
    subject_glob: str = DEFAULT_SUBJECT_GLOB,
) -> tuple[str, ...]:
    if subjects is not None:
        return tuple(dict.fromkeys(subject.strip() for subject in subjects if subject.strip()))
    if subjects_file is not None:
        return _read_subjects_file(subjects_file)

    discovered: set[str] = set()
    for dataset in datasets:
        for path in dataset.root.glob(subject_glob):
            if path.is_dir():
                discovered.add(path.name)
    return tuple(sorted(discovered))


def _first_existing(candidates: Sequence[Path]) -> Path | None:
    for path in candidates:
        if path.is_file():
            return path
    return None


def _required_input_checks(anat_dir: Path, subject: str) -> tuple[list[Path], list[str]]:
    t1 = _first_existing(
        (
            anat_dir / f"{subject}_T1w.nii",
            anat_dir / f"{subject}_T1w.nii.gz",
        )
    )
    t2 = _first_existing(
        (
            anat_dir / f"{subject}_T2w.nii",
            anat_dir / f"{subject}_T2w.nii.gz",
        )
    )
    seg = _first_existing(
        (
            anat_dir / f"{subject}_T1w_ras_1mm_T1andT2_masks.nii",
            anat_dir / f"{subject}_T1w_ras_1mm_T1andT2_masks.nii.gz",
        )
    )
    mesh = resolve_existing_mesh(anat_dir, subject)

    present = [path for path in (t1, t2, seg, mesh) if path is not None]
    missing: list[str] = []
    if t1 is None:
        missing.append(str(anat_dir / f"{subject}_T1w.nii[.gz]"))
    if t2 is None:
        missing.append(str(anat_dir / f"{subject}_T2w.nii[.gz]"))
    if seg is None:
        missing.append(str(anat_dir / f"{subject}_T1w_ras_1mm_T1andT2_masks.nii[.gz]"))
    if mesh is None:
        missing.append(f"existing mesh under {anat_dir}/m2m_*")
    return present, missing


def build_task_rows(
    datasets: Sequence[RepeatDataset],
    subjects: Sequence[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for subject in subjects:
            anat_dir = dataset.root / subject / "anat"
            mesh_path = resolve_existing_mesh(anat_dir, subject)
            _, missing = _required_input_checks(anat_dir, subject)
            status = "ready" if not missing else "blocked"
            rows.append(
                {
                    "task_id": len(rows),
                    "dataset_name": dataset.name,
                    "repeat_id": dataset.repeat_id,
                    "dataset_root": dataset.root,
                    "subject": subject,
                    "anat_dir": anat_dir,
                    "mesh_path": mesh_path or "",
                    "status": status,
                    "message": "ready" if status == "ready" else f"missing: {_join_cells(missing)}",
                }
            )
    return rows


def _generated_subject_paths(dataset_root: Path, subject: str) -> tuple[Path, ...]:
    anat = dataset_root / subject / "anat"
    return (
        anat / "SimNIBS",
        anat / "post",
        anat / f"{subject}_T1w_ras_1mm_T1andT2_masks_clipped.nii",
        anat / f"{subject}_T1w_ras_1mm_T1andT2_masks_merged.nii",
        anat / "skin_mask.nii.gz",
    )


def _generated_dataset_paths(dataset_root: Path) -> tuple[Path, ...]:
    return (
        dataset_root / "population_analysis",
        dataset_root / "subject_metrics_analysis",
        dataset_root / "_analysis",
    )


def _kind(path: Path) -> str:
    if path.is_symlink():
        return "symlink"
    if path.is_dir():
        return "dir"
    if path.is_file():
        return "file"
    return "missing"


def _remove_generated_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def cleanup_generated_outputs(
    datasets: Sequence[RepeatDataset],
    subjects: Sequence[str],
    *,
    apply: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[Path] = set()

    def record(path: Path, *, scope: str, dataset_name: str, subject: str = "") -> None:
        if path in seen:
            return
        seen.add(path)
        before_kind = _kind(path)
        if before_kind == "missing":
            action = "missing"
            message = "path not present"
        elif apply:
            _remove_generated_path(path)
            action = "deleted"
            message = "deleted generated output"
        else:
            action = "would_delete"
            message = "dry run"
        rows.append(
            {
                "scope": scope,
                "dataset_name": dataset_name,
                "subject": subject,
                "path": path,
                "kind": before_kind,
                "action": action,
                "message": message,
            }
        )

    for dataset in datasets:
        for path in _generated_dataset_paths(dataset.root):
            record(path, scope="dataset", dataset_name=dataset.name)
        for subject in subjects:
            for path in _generated_subject_paths(dataset.root, subject):
                record(path, scope="subject", dataset_name=dataset.name, subject=subject)

    return rows


def run_preflight(
    roi_root: str | Path,
    *,
    manifest: str | Path,
    cleanup_manifest: str | Path | None = None,
    apply: bool = False,
    dataset_glob: str = DEFAULT_DATASET_GLOB,
    repeats: Iterable[str] | None = None,
    subjects: Sequence[str] | None = None,
    subjects_file: str | Path | None = None,
    subject_glob: str = DEFAULT_SUBJECT_GLOB,
) -> dict[str, int | str]:
    datasets = discover_repeat_datasets(roi_root, dataset_glob=dataset_glob, repeats=repeats)
    expected_subjects = discover_expected_subjects(
        datasets,
        subjects=subjects,
        subjects_file=subjects_file,
        subject_glob=subject_glob,
    )
    task_rows = build_task_rows(datasets, expected_subjects)
    cleanup_rows = cleanup_generated_outputs(datasets, expected_subjects, apply=apply)

    manifest_path = write_tsv(manifest, TASK_FIELDNAMES, task_rows)
    cleanup_path = Path(cleanup_manifest).expanduser() if cleanup_manifest else manifest_path.with_name("cleanup_manifest.tsv")
    write_tsv(cleanup_path, CLEANUP_FIELDNAMES, cleanup_rows)

    ready = sum(1 for row in task_rows if row["status"] == "ready")
    blocked = sum(1 for row in task_rows if row["status"] == "blocked")
    deleted = sum(1 for row in cleanup_rows if row["action"] == "deleted")
    would_delete = sum(1 for row in cleanup_rows if row["action"] == "would_delete")
    return {
        "datasets": len(datasets),
        "subjects": len(expected_subjects),
        "total_tasks": len(task_rows),
        "ready_tasks": ready,
        "blocked_tasks": blocked,
        "cleanup_deleted": deleted,
        "cleanup_would_delete": would_delete,
        "manifest": str(manifest_path),
        "cleanup_manifest": str(cleanup_path),
    }


def validate_manifest(
    manifest: str | Path,
    *,
    summary_path: str | Path | None = None,
    check_nifti: bool = True,
) -> dict[str, int | str]:
    manifest_path = Path(manifest).expanduser().resolve()
    rows = read_tsv(manifest_path)
    summary_rows: list[dict[str, object]] = []

    complete = 0
    incomplete = 0
    skipped = 0
    for row in rows:
        if row.get("status") != "ready":
            skipped += 1
            summary_rows.append(
                {
                    "task_id": row.get("task_id", ""),
                    "dataset_name": row.get("dataset_name", ""),
                    "repeat_id": row.get("repeat_id", ""),
                    "dataset_root": row.get("dataset_root", ""),
                    "subject": row.get("subject", ""),
                    "status": "skipped",
                    "missing_outputs": "",
                    "failure_reasons": row.get("message", ""),
                }
            )
            continue

        result = validate_subject_outputs(
            row["dataset_root"],
            row["subject"],
            check_nifti=check_nifti,
        )
        missing_outputs = [check.name for check in result.checks if not check.ok]
        failure_reasons = [
            f"{check.name}:{check.reason}"
            for check in result.checks
            if not check.ok
        ]
        if result.ok:
            complete += 1
            status = "complete"
        else:
            incomplete += 1
            status = "incomplete"
        summary_rows.append(
            {
                "task_id": row.get("task_id", ""),
                "dataset_name": row.get("dataset_name", ""),
                "repeat_id": row.get("repeat_id", ""),
                "dataset_root": row.get("dataset_root", ""),
                "subject": row.get("subject", ""),
                "status": status,
                "missing_outputs": _join_cells(missing_outputs),
                "failure_reasons": _join_cells(failure_reasons),
            }
        )

    out_path = Path(summary_path).expanduser() if summary_path else manifest_path.with_name("validation_summary.tsv")
    write_tsv(out_path, VALIDATION_FIELDNAMES, summary_rows)
    return {
        "total_tasks": len(rows),
        "complete_tasks": complete,
        "incomplete_tasks": incomplete,
        "skipped_tasks": skipped,
        "summary_path": str(out_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight",
        help="Discover tasks and optionally delete generated outputs.",
    )
    preflight.add_argument("--roi-root", required=True, help="ROI root containing *_Data_* repeat folders.")
    preflight.add_argument("--manifest", required=True, help="Path for the task manifest TSV.")
    preflight.add_argument("--cleanup-manifest", help="Path for the cleanup manifest TSV.")
    preflight.add_argument("--dataset-glob", default=DEFAULT_DATASET_GLOB)
    preflight.add_argument("--repeats", nargs="+", help="Optional repeat ids to include.")
    preflight.add_argument("--subjects", nargs="+", help="Optional subject ids to include.")
    preflight.add_argument("--subjects-file", help="Optional newline-delimited subject list.")
    preflight.add_argument("--subject-glob", default=DEFAULT_SUBJECT_GLOB)
    preflight.add_argument(
        "--apply",
        action="store_true",
        help="Delete generated outputs. Omit for dry-run cleanup planning.",
    )

    validate = subparsers.add_parser(
        "validate",
        help="Validate final outputs for every ready row in a task manifest.",
    )
    validate.add_argument("--manifest", required=True, help="Task manifest TSV from preflight.")
    validate.add_argument("--summary", help="Validation summary TSV path.")
    validate.add_argument(
        "--skip-nifti-load",
        action="store_true",
        help="Check presence and size only; do not load NIfTI payloads.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "preflight":
        if args.subjects and args.subjects_file:
            parser.error("--subjects and --subjects-file are mutually exclusive.")
        result = run_preflight(
            args.roi_root,
            manifest=args.manifest,
            cleanup_manifest=args.cleanup_manifest,
            apply=args.apply,
            dataset_glob=args.dataset_glob,
            repeats=args.repeats,
            subjects=args.subjects,
            subjects_file=args.subjects_file,
            subject_glob=args.subject_glob,
        )
    elif args.command == "validate":
        result = validate_manifest(
            args.manifest,
            summary_path=args.summary,
            check_nifti=not args.skip_nifti_load,
        )
    else:
        raise AssertionError(args.command)

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
