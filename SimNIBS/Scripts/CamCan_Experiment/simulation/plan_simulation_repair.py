#!/usr/bin/env python3
"""Plan subject-level reruns needed to repair repeated simulation datasets.

Given a parent directory containing repeat dataset roots such as
``Left_Hippocampus_Data_01`` through ``Left_Hippocampus_Data_10``, this script
has two stages:

- ``discovery``: check each expected subject in each repeat, write findings,
  statistics, and per-repeat repair plans, but submit nothing;
- ``simulate``: submit one Slurm repair array per repeat using the plans written
  by discovery.

The task plan is designed for ``HPC_scripts/repair_jobArray.slurm``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.validate_simulation_outputs import validate_subject_outputs


REPEAT_DATASET_PATTERN = re.compile(r"^(?P<roi_prefix>.+)_Data_(?P<repeat>\d+)$")
DEFAULT_DATASET_GLOB = "*_Data_*"
DEFAULT_SUBJECT_GLOB = "sub-*"
DEFAULT_OUTPUT_DIR_NAME = "simulation_repair_plan"
DEFAULT_PER_REPEAT_PLAN_DIR_NAME = "per_repeat_repair_plans"
DEFAULT_SUBMISSION_SUMMARY_NAME = "submission_summary.json"
REPAIR_PLAN_FIELDNAMES = (
    "task_id",
    "dataset_name",
    "repeat_id",
    "dataset_root",
    "subject",
    "missing_outputs",
    "failure_reasons",
)


@dataclass(frozen=True)
class RepeatDataset:
    root: Path
    name: str
    roi_prefix: str
    repeat_id: str


@dataclass(frozen=True)
class SubjectRunStatus:
    dataset_name: str
    repeat_id: str
    dataset_root: Path
    subject: str
    status: str
    output_ok: bool
    repairable: bool
    missing_outputs: tuple[str, ...]
    failure_reasons: tuple[str, ...]
    missing_inputs: tuple[str, ...]


@dataclass(frozen=True)
class RepairScanResult:
    batch_root: Path
    datasets: tuple[RepeatDataset, ...]
    expected_subjects: tuple[str, ...]
    statuses: tuple[SubjectRunStatus, ...]

    @property
    def complete_statuses(self) -> tuple[SubjectRunStatus, ...]:
        return tuple(item for item in self.statuses if item.status == "complete")

    @property
    def repairable_statuses(self) -> tuple[SubjectRunStatus, ...]:
        return tuple(item for item in self.statuses if item.status == "repairable")

    @property
    def blocked_statuses(self) -> tuple[SubjectRunStatus, ...]:
        return tuple(item for item in self.statuses if item.status == "blocked")

    @property
    def incomplete_statuses(self) -> tuple[SubjectRunStatus, ...]:
        return tuple(item for item in self.statuses if item.status != "complete")


def _parse_repeat_value(value: str | int) -> int:
    try:
        repeat_value = int(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"Invalid repeat identifier {value!r}; expected an integer.") from exc
    if repeat_value < 0:
        raise ValueError(f"Repeat identifier must be non-negative, got {repeat_value}.")
    return repeat_value


def _sanitize_cell(value: object) -> str:
    return str(value).replace("\t", " ").replace("\n", " ").strip()


def _join_cells(values: Iterable[object]) -> str:
    return ";".join(_sanitize_cell(value) for value in values)


def _resolve_output_dir(batch_root: str | Path, out_dir: str | Path | None = None) -> Path:
    batch_root_path = Path(batch_root).expanduser().resolve()
    output_dir = (
        Path(out_dir).expanduser()
        if out_dir is not None
        else batch_root_path / DEFAULT_OUTPUT_DIR_NAME
    )
    if not output_dir.is_absolute():
        output_dir = batch_root_path / output_dir
    return output_dir


def _read_subjects_file(path: Path) -> tuple[str, ...]:
    subjects: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            subject = line.strip()
            if not subject or subject.startswith("#"):
                continue
            if "\t" in subject or " " in subject:
                raise SystemExit(
                    f"Invalid subject entry at {path}:{line_number}: {subject!r}"
                )
            if subject not in seen:
                seen.add(subject)
                subjects.append(subject)
    if not subjects:
        raise SystemExit(f"No subjects found in expected-subjects file: {path}")
    return tuple(subjects)


def discover_repeat_datasets(
    batch_root: Path,
    *,
    dataset_glob: str = DEFAULT_DATASET_GLOB,
    repeats: Optional[Iterable[str]] = None,
) -> tuple[RepeatDataset, ...]:
    selected_repeats = None
    if repeats is not None:
        selected_repeats = {_parse_repeat_value(repeat) for repeat in repeats}

    datasets: list[RepeatDataset] = []
    for path in batch_root.glob(dataset_glob):
        if not path.is_dir():
            continue
        match = REPEAT_DATASET_PATTERN.match(path.name)
        if not match:
            continue

        repeat_value = _parse_repeat_value(match.group("repeat"))
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
            _parse_repeat_value(item.repeat_id),
            item.name.casefold(),
        )
    )
    return tuple(datasets)


def discover_expected_subjects(
    datasets: Sequence[RepeatDataset],
    *,
    subject_glob: str = DEFAULT_SUBJECT_GLOB,
    expected_subjects_file: str | Path | None = None,
) -> tuple[str, ...]:
    if expected_subjects_file is not None:
        return _read_subjects_file(Path(expected_subjects_file).expanduser())

    subjects: set[str] = set()
    for dataset in datasets:
        for path in dataset.root.glob(subject_glob):
            if path.is_dir():
                subjects.add(path.name)

    if not subjects:
        roots = ", ".join(str(dataset.root) for dataset in datasets)
        raise SystemExit(f"No subject directories matching {subject_glob!r} under: {roots}")
    return tuple(sorted(subjects))


def required_runner_inputs(dataset_root: Path, subject: str) -> tuple[Path, ...]:
    """Inputs needed by the current subject-specific simulation runner."""
    anat_dir = dataset_root / subject / "anat"
    return (
        dataset_root / subject,
        anat_dir / f"{subject}_T1w.nii",
        anat_dir / f"{subject}_T2w.nii",
        anat_dir / f"{subject}_T1w_ras_1mm_T1andT2_masks.nii",
    )


def missing_runner_inputs(dataset_root: Path, subject: str) -> tuple[str, ...]:
    return tuple(str(path) for path in required_runner_inputs(dataset_root, subject) if not path.exists())


def evaluate_subject_run(
    dataset: RepeatDataset,
    subject: str,
    *,
    min_bytes: int = 1,
    check_nifti: bool = True,
    require_inputs: bool = True,
) -> SubjectRunStatus:
    validation = validate_subject_outputs(
        dataset.root,
        subject,
        min_bytes=min_bytes,
        check_nifti=check_nifti,
    )

    missing_outputs = tuple(check.name for check in validation.checks if not check.ok)
    failure_reasons = tuple(
        f"{check.name}:{check.reason}" for check in validation.checks if not check.ok
    )

    if validation.ok:
        return SubjectRunStatus(
            dataset_name=dataset.name,
            repeat_id=dataset.repeat_id,
            dataset_root=dataset.root,
            subject=subject,
            status="complete",
            output_ok=True,
            repairable=False,
            missing_outputs=(),
            failure_reasons=(),
            missing_inputs=(),
        )

    missing_inputs = missing_runner_inputs(dataset.root, subject) if require_inputs else ()
    status = "blocked" if missing_inputs else "repairable"
    return SubjectRunStatus(
        dataset_name=dataset.name,
        repeat_id=dataset.repeat_id,
        dataset_root=dataset.root,
        subject=subject,
        status=status,
        output_ok=False,
        repairable=(status == "repairable"),
        missing_outputs=missing_outputs,
        failure_reasons=failure_reasons,
        missing_inputs=missing_inputs,
    )


def scan_repair_needs(
    *,
    batch_root: str | Path,
    dataset_glob: str = DEFAULT_DATASET_GLOB,
    repeats: Optional[Iterable[str]] = None,
    subject_glob: str = DEFAULT_SUBJECT_GLOB,
    expected_subjects_file: str | Path | None = None,
    min_bytes: int = 1,
    check_nifti: bool = True,
    require_inputs: bool = True,
) -> RepairScanResult:
    batch_root_path = Path(batch_root).expanduser().resolve()
    if not batch_root_path.is_dir():
        raise SystemExit(f"Batch root directory not found: {batch_root_path}")

    datasets = discover_repeat_datasets(
        batch_root_path,
        dataset_glob=dataset_glob,
        repeats=repeats,
    )
    if not datasets:
        raise SystemExit(
            f"No repeat datasets matched {dataset_glob!r} under {batch_root_path}. "
            "Expected names such as 'Left_Hippocampus_Data_01'."
        )

    subjects = discover_expected_subjects(
        datasets,
        subject_glob=subject_glob,
        expected_subjects_file=expected_subjects_file,
    )

    statuses = [
        evaluate_subject_run(
            dataset,
            subject,
            min_bytes=min_bytes,
            check_nifti=check_nifti,
            require_inputs=require_inputs,
        )
        for dataset in datasets
        for subject in subjects
    ]
    return RepairScanResult(
        batch_root=batch_root_path,
        datasets=datasets,
        expected_subjects=subjects,
        statuses=tuple(statuses),
    )


def _status_row(status: SubjectRunStatus) -> dict[str, object]:
    return {
        "dataset_name": status.dataset_name,
        "repeat_id": status.repeat_id,
        "dataset_root": str(status.dataset_root),
        "subject": status.subject,
        "status": status.status,
        "output_ok": str(status.output_ok).lower(),
        "repairable": str(status.repairable).lower(),
        "missing_outputs": _join_cells(status.missing_outputs),
        "failure_reasons": _join_cells(status.failure_reasons),
        "missing_inputs": _join_cells(status.missing_inputs),
    }


def build_repair_plan_rows(statuses: Sequence[SubjectRunStatus]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task_id, status in enumerate(item for item in statuses if item.status == "repairable"):
        rows.append(
            {
                "task_id": task_id,
                "dataset_name": status.dataset_name,
                "repeat_id": status.repeat_id,
                "dataset_root": str(status.dataset_root),
                "subject": status.subject,
                "missing_outputs": _join_cells(status.missing_outputs),
                "failure_reasons": _join_cells(status.failure_reasons),
            }
        )
    return rows


def build_per_repeat_plan_rows(
    statuses: Sequence[SubjectRunStatus],
) -> dict[str, list[dict[str, object]]]:
    rows_by_dataset: dict[str, list[SubjectRunStatus]] = {}
    for status in statuses:
        if status.status != "repairable":
            continue
        rows_by_dataset.setdefault(status.dataset_name, []).append(status)

    return {
        dataset_name: build_repair_plan_rows(dataset_statuses)
        for dataset_name, dataset_statuses in sorted(rows_by_dataset.items())
    }


def build_blocked_rows(statuses: Sequence[SubjectRunStatus]) -> list[dict[str, object]]:
    return [_status_row(status) for status in statuses if status.status == "blocked"]


def build_status_rows(statuses: Sequence[SubjectRunStatus]) -> list[dict[str, object]]:
    return [_status_row(status) for status in statuses]


def build_subject_count_rows(statuses: Sequence[SubjectRunStatus]) -> list[dict[str, object]]:
    by_subject: dict[str, list[SubjectRunStatus]] = {}
    for status in statuses:
        by_subject.setdefault(status.subject, []).append(status)

    rows: list[dict[str, object]] = []
    for subject in sorted(by_subject):
        subject_statuses = by_subject[subject]
        incomplete = [status for status in subject_statuses if status.status != "complete"]
        if not incomplete:
            continue
        repairable = [status for status in incomplete if status.status == "repairable"]
        blocked = [status for status in incomplete if status.status == "blocked"]
        rows.append(
            {
                "subject": subject,
                "n_missing_runs": len(incomplete),
                "n_repairable_runs": len(repairable),
                "n_blocked_runs": len(blocked),
                "missing_repeats": _join_cells(status.repeat_id for status in incomplete),
                "repairable_repeats": _join_cells(status.repeat_id for status in repairable),
                "blocked_repeats": _join_cells(status.repeat_id for status in blocked),
                "missing_datasets": _join_cells(status.dataset_name for status in incomplete),
                "repairable_datasets": _join_cells(status.dataset_name for status in repairable),
                "blocked_datasets": _join_cells(status.dataset_name for status in blocked),
            }
        )
    return rows


def _write_tsv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_per_repeat_repair_plans(
    *,
    result: RepairScanResult,
    output_dir: Path,
) -> list[dict[str, object]]:
    per_repeat_dir = output_dir / DEFAULT_PER_REPEAT_PLAN_DIR_NAME
    per_repeat_dir.mkdir(parents=True, exist_ok=True)

    rows_by_dataset = build_per_repeat_plan_rows(result.statuses)
    dataset_lookup = {dataset.name: dataset for dataset in result.datasets}
    plan_entries: list[dict[str, object]] = []

    for dataset_name, rows in rows_by_dataset.items():
        if not rows:
            continue
        dataset = dataset_lookup[dataset_name]
        plan_path = per_repeat_dir / f"{dataset.name}_repair_plan.tsv"
        _write_tsv(plan_path, rows, REPAIR_PLAN_FIELDNAMES)
        plan_entries.append(
            {
                "dataset_name": dataset.name,
                "repeat_id": dataset.repeat_id,
                "dataset_root": str(dataset.root),
                "repairable_subject_runs": len(rows),
                "repair_plan": str(plan_path),
            }
        )

    return plan_entries


def write_repair_outputs(result: RepairScanResult, out_dir: str | Path | None = None) -> dict[str, str]:
    output_dir = _resolve_output_dir(result.batch_root, out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repair_plan_rows = build_repair_plan_rows(result.statuses)
    per_repeat_plan_entries = _write_per_repeat_repair_plans(
        result=result,
        output_dir=output_dir,
    )
    blocked_rows = build_blocked_rows(result.statuses)
    subject_count_rows = build_subject_count_rows(result.statuses)
    status_rows = build_status_rows(result.statuses)

    repair_plan_path = output_dir / "repair_plan.tsv"
    subject_counts_path = output_dir / "repair_subject_counts.tsv"
    blocked_path = output_dir / "blocked_repair_tasks.tsv"
    status_path = output_dir / "subject_run_status.tsv"
    summary_path = output_dir / "scan_summary.json"

    _write_tsv(
        repair_plan_path,
        repair_plan_rows,
        REPAIR_PLAN_FIELDNAMES,
    )
    _write_tsv(
        subject_counts_path,
        subject_count_rows,
        (
            "subject",
            "n_missing_runs",
            "n_repairable_runs",
            "n_blocked_runs",
            "missing_repeats",
            "repairable_repeats",
            "blocked_repeats",
            "missing_datasets",
            "repairable_datasets",
            "blocked_datasets",
        ),
    )
    _write_tsv(
        blocked_path,
        blocked_rows,
        (
            "dataset_name",
            "repeat_id",
            "dataset_root",
            "subject",
            "status",
            "output_ok",
            "repairable",
            "missing_outputs",
            "failure_reasons",
            "missing_inputs",
        ),
    )
    _write_tsv(
        status_path,
        status_rows,
        (
            "dataset_name",
            "repeat_id",
            "dataset_root",
            "subject",
            "status",
            "output_ok",
            "repairable",
            "missing_outputs",
            "failure_reasons",
            "missing_inputs",
        ),
    )

    summary = {
        "batch_root": str(result.batch_root),
        "dataset_count": len(result.datasets),
        "datasets": [
            {
                "name": dataset.name,
                "repeat_id": dataset.repeat_id,
                "roi_prefix": dataset.roi_prefix,
                "root": str(dataset.root),
            }
            for dataset in result.datasets
        ],
        "expected_subject_count": len(result.expected_subjects),
        "subject_run_count": len(result.statuses),
        "complete_subject_runs": len(result.complete_statuses),
        "incomplete_subject_runs": len(result.incomplete_statuses),
        "repairable_subject_runs": len(result.repairable_statuses),
        "blocked_subject_runs": len(result.blocked_statuses),
        "subjects_needing_repair": len(subject_count_rows),
        "output_dir": str(output_dir),
        "repair_plan": str(repair_plan_path),
        "repair_subject_counts": str(subject_counts_path),
        "blocked_repair_tasks": str(blocked_path),
        "subject_run_status": str(status_path),
        "per_repeat_repair_plans_dir": str(output_dir / DEFAULT_PER_REPEAT_PLAN_DIR_NAME),
        "per_repeat_repair_plans": per_repeat_plan_entries,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "repair_plan": str(repair_plan_path),
        "repair_subject_counts": str(subject_counts_path),
        "blocked_repair_tasks": str(blocked_path),
        "subject_run_status": str(status_path),
        "per_repeat_repair_plans_dir": str(output_dir / DEFAULT_PER_REPEAT_PLAN_DIR_NAME),
        "scan_summary": str(summary_path),
    }


def _read_scan_summary(output_dir: Path) -> dict[str, object]:
    summary_path = output_dir / "scan_summary.json"
    if not summary_path.is_file():
        raise SystemExit(
            f"Discovery summary not found: {summary_path}. "
            "Run --stage discovery before --stage simulate."
        )
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid discovery summary JSON: {summary_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Discovery summary is not a JSON object: {summary_path}")
    return payload


def _load_per_repeat_plan_entries(output_dir: Path) -> list[dict[str, object]]:
    summary = _read_scan_summary(output_dir)
    entries = summary.get("per_repeat_repair_plans")
    if not isinstance(entries, list):
        raise SystemExit(
            f"Discovery summary does not contain per-repeat repair plans: {output_dir / 'scan_summary.json'}"
        )
    return [entry for entry in entries if isinstance(entry, dict)]


def submit_repair_jobs(
    *,
    output_dir: str | Path,
    submit_script: str | Path | None = None,
    dry_run: bool = False,
    submit_args: Sequence[str] = (),
) -> list[dict[str, object]]:
    output_dir_path = Path(output_dir).expanduser().resolve()
    entries = _load_per_repeat_plan_entries(output_dir_path)
    if not entries:
        print(f"[INFO] No per-repeat repair plans found under: {output_dir_path}")
        return []

    submit_script_path = (
        Path(submit_script).expanduser()
        if submit_script is not None
        else ROOT / "HPC_scripts" / "submit_repair_jobArray.sh"
    )
    if not submit_script_path.is_file():
        raise SystemExit(f"Repair submit script not found: {submit_script_path}")

    submission_results: list[dict[str, object]] = []
    for entry in entries:
        plan_path = Path(str(entry.get("repair_plan", ""))).expanduser()
        if not plan_path.is_file():
            raise SystemExit(f"Per-repeat repair plan not found: {plan_path}")

        env = os.environ.copy()
        env["TI_REPAIR_PLAN_FILE"] = str(plan_path)
        cmd = [str(submit_script_path), *submit_args]
        printable_cmd = " ".join([f"TI_REPAIR_PLAN_FILE={plan_path}", *cmd])
        print(
            "[INFO] "
            + ("Would submit" if dry_run else "Submitting")
            + f" repair array for {entry.get('dataset_name')} ({entry.get('repairable_subject_runs')} task(s)):"
        )
        print(f"       {printable_cmd}")

        result_payload: dict[str, object] = {
            "dataset_name": entry.get("dataset_name"),
            "repeat_id": entry.get("repeat_id"),
            "dataset_root": entry.get("dataset_root"),
            "repairable_subject_runs": entry.get("repairable_subject_runs"),
            "repair_plan": str(plan_path),
            "command": cmd,
            "dry_run": dry_run,
        }

        if dry_run:
            result_payload["status"] = "dry_run"
            submission_results.append(result_payload)
            continue

        completed = subprocess.run(
            cmd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)

        result_payload.update(
            {
                "status": "submitted" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        submission_results.append(result_payload)
        if completed.returncode != 0:
            summary_path = output_dir_path / DEFAULT_SUBMISSION_SUMMARY_NAME
            summary_path.write_text(json.dumps(submission_results, indent=2), encoding="utf-8")
            raise SystemExit(
                f"Repair submission failed for {entry.get('dataset_name')} "
                f"with exit code {completed.returncode}."
            )

    summary_path = output_dir_path / DEFAULT_SUBMISSION_SUMMARY_NAME
    summary_path.write_text(json.dumps(submission_results, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote submission summary to: {summary_path}")
    return submission_results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument(
        "--stage",
        choices=("discovery", "simulate", "all"),
        default="discovery",
        help=(
            "Workflow stage to run. 'discovery' scans and reports only; "
            "'simulate' submits one repair array per repeat from a previous discovery; "
            "'all' runs discovery and then submits. Default: discovery"
        ),
    )
    parser.add_argument(
        "--batch-root",
        required=True,
        help="Parent directory containing repeat datasets such as Left_Hippocampus_Data_01.",
    )
    parser.add_argument(
        "--dataset-glob",
        default=DEFAULT_DATASET_GLOB,
        help=f"Glob used within batch-root to find repeat datasets. Default: {DEFAULT_DATASET_GLOB}",
    )
    parser.add_argument(
        "--repeats",
        nargs="*",
        default=None,
        help="Optional repeat identifiers to scan, for example: --repeats 01 02 10",
    )
    parser.add_argument(
        "--expected-subjects-file",
        default=None,
        help=(
            "Optional one-subject-per-line manifest. If omitted, the expected cohort "
            "is the union of subject directories found across selected repeats."
        ),
    )
    parser.add_argument(
        "--subject-glob",
        default=DEFAULT_SUBJECT_GLOB,
        help=f"Subject directory glob used when no manifest is provided. Default: {DEFAULT_SUBJECT_GLOB}",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory for plan files. Relative paths are resolved under batch-root. "
            f"Default: <batch-root>/{DEFAULT_OUTPUT_DIR_NAME}"
        ),
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=1,
        help="Minimum acceptable byte size for each required output. Default: 1",
    )
    parser.add_argument(
        "--skip-nifti-load",
        action="store_true",
        help="Only check output file presence/size; do not load NIfTI payloads during the scan.",
    )
    parser.add_argument(
        "--skip-input-check",
        action="store_true",
        help=(
            "Put incomplete subject-runs into the repair plan even if the current "
            "runner inputs are missing. Use only if inputs will be restored before submission."
        ),
    )
    parser.add_argument(
        "--expected-repeat-count",
        type=int,
        default=None,
        help="Optional expected repeat directory count; emits a warning if discovery differs.",
    )
    parser.add_argument(
        "--submit-script",
        default=None,
        help=(
            "Submit wrapper used by --stage simulate. Default: "
            "HPC_scripts/submit_repair_jobArray.sh"
        ),
    )
    parser.add_argument(
        "--submit-arg",
        action="append",
        default=[],
        help="Extra argument passed through to the repair submit wrapper. Repeat as needed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="For --stage simulate/all, print per-repeat submission commands without running sbatch.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paths: dict[str, str] | None = None

    if args.stage in ("discovery", "all"):
        result = scan_repair_needs(
            batch_root=args.batch_root,
            dataset_glob=args.dataset_glob,
            repeats=args.repeats,
            subject_glob=args.subject_glob,
            expected_subjects_file=args.expected_subjects_file,
            min_bytes=args.min_bytes,
            check_nifti=not args.skip_nifti_load,
            require_inputs=not args.skip_input_check,
        )

        if args.expected_repeat_count is not None and len(result.datasets) != args.expected_repeat_count:
            print(
                f"[WARN] Expected {args.expected_repeat_count} repeat dataset(s), "
                f"found {len(result.datasets)}.",
                file=sys.stderr,
            )

        paths = write_repair_outputs(result, out_dir=args.out_dir)
        print(f"[INFO] Discovery stage complete.")
        print(f"[INFO] Repeat datasets scanned: {len(result.datasets)}")
        print(f"[INFO] Expected subjects: {len(result.expected_subjects)}")
        print(f"[INFO] Complete subject-runs: {len(result.complete_statuses)}")
        print(f"[INFO] Repairable subject-runs: {len(result.repairable_statuses)}")
        print(f"[INFO] Blocked incomplete subject-runs: {len(result.blocked_statuses)}")
        print(f"[INFO] Combined repair plan: {paths['repair_plan']}")
        print(f"[INFO] Per-repeat repair plans: {paths['per_repeat_repair_plans_dir']}")
        print(f"[INFO] Subject counts: {paths['repair_subject_counts']}")
        if result.blocked_statuses:
            print(f"[WARN] Blocked rows need input restoration before repair: {paths['blocked_repair_tasks']}")
        print("[INFO] Submit per-repeat repair arrays with:")
        print(f"       python {Path(__file__).as_posix()} --stage simulate --batch-root {result.batch_root}")

    if args.stage in ("simulate", "all"):
        output_dir = (
            Path(paths["output_dir"]).expanduser()
            if paths is not None
            else _resolve_output_dir(args.batch_root, args.out_dir)
        )
        submit_repair_jobs(
            output_dir=output_dir,
            submit_script=args.submit_script,
            dry_run=args.dry_run,
            submit_args=args.submit_arg,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
