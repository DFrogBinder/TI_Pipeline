#!/usr/bin/env python3
"""Remove an existing CamCAN cohort from a larger subject directory.

The reference directory is used only to discover subject IDs.  Only matching,
direct child directories of the full dataset can be removed.  The default mode
is a read-only audit; permanent removal requires ``--apply``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
import stat
import sys
from typing import TextIO


SUBJECT_PATTERN = re.compile(r"sub-CC[0-9]{6}\Z")


@dataclass(frozen=True)
class CleanupPlan:
    reference_root: Path
    full_dataset_root: Path
    reference_ids: tuple[str, ...]
    full_dataset_ids: tuple[str, ...]
    targets: tuple[Path, ...]
    missing_ids: tuple[str, ...]

    @property
    def remaining_subjects(self) -> int:
        return len(self.full_dataset_ids) - len(self.targets)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit or permanently remove CamCAN subjects found in a reference "
            "cohort from a larger dataset directory."
        )
    )
    parser.add_argument(
        "reference_cohort",
        type=Path,
        help="extracted original cohort containing direct sub-CC###### directories",
    )
    parser.add_argument(
        "full_dataset",
        type=Path,
        help="full dataset containing direct sub-CC###### directories",
    )
    parser.add_argument(
        "--expected-subjects",
        type=int,
        metavar="N",
        help=(
            "optionally require an exact number of IDs in the reference cohort"
        ),
    )
    parser.add_argument(
        "--expected-full-subjects",
        type=int,
        metavar="N",
        help="optionally require an exact pre-cleanup full-dataset subject count",
    )
    parser.add_argument(
        "--allow-missing-targets",
        action="store_true",
        help=(
            "allow reference IDs already absent from the full dataset; intended "
            "only for resuming an interrupted/partially completed cleanup"
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="show counts without printing every matching subject path",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="permanently delete the planned subject directories",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the interactive confirmation (valid only with --apply)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "apply-mode event manifest path (default: a timestamped TSV beside "
            "the full dataset)"
        ),
    )
    return parser.parse_args(argv)


def is_relative_to(path: Path, possible_parent: Path) -> bool:
    """Compatibility helper for Python versions before Path.is_relative_to."""
    try:
        path.relative_to(possible_parent)
    except ValueError:
        return False
    return True


def resolve_directory(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} cannot be resolved: {path} ({error})") from error
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def validate_roots(reference_root: Path, full_dataset_root: Path) -> None:
    if reference_root == full_dataset_root:
        raise ValueError("reference cohort and full dataset must be different directories")
    if is_relative_to(reference_root, full_dataset_root):
        raise ValueError("reference cohort must not be inside the full dataset")
    if is_relative_to(full_dataset_root, reference_root):
        raise ValueError("full dataset must not be inside the reference cohort")


def discover_subjects(root: Path, label: str) -> dict[str, Path]:
    subjects: dict[str, Path] = {}
    invalid_entries: list[str] = []

    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if not entry.name.startswith("sub-"):
                    continue
                if SUBJECT_PATTERN.fullmatch(entry.name) is None:
                    invalid_entries.append(f"{entry.name} (invalid CamCAN subject ID)")
                    continue
                if entry.is_symlink():
                    invalid_entries.append(f"{entry.name} (symbolic link not allowed)")
                    continue
                if not entry.is_dir(follow_symlinks=False):
                    invalid_entries.append(f"{entry.name} (not a directory)")
                    continue
                subjects[entry.name] = Path(entry.path)
    except OSError as error:
        raise ValueError(f"could not scan {label}: {root} ({error})") from error

    if invalid_entries:
        details = ", ".join(sorted(invalid_entries))
        raise ValueError(f"unsafe subject-like entries in {label}: {details}")
    return subjects


def build_plan(
    reference_cohort: Path,
    full_dataset: Path,
    expected_subjects: int | None,
    expected_full_subjects: int | None,
    allow_missing_targets: bool,
) -> CleanupPlan:
    if expected_subjects is not None and expected_subjects < 1:
        raise ValueError("--expected-subjects must be at least 1")
    if expected_full_subjects is not None and expected_full_subjects < 1:
        raise ValueError("--expected-full-subjects must be at least 1")

    reference_root = resolve_directory(reference_cohort, "reference cohort")
    full_dataset_root = resolve_directory(full_dataset, "full dataset")
    validate_roots(reference_root, full_dataset_root)

    reference_subjects = discover_subjects(reference_root, "reference cohort")
    full_subjects = discover_subjects(full_dataset_root, "full dataset")
    reference_ids = tuple(sorted(reference_subjects))
    full_dataset_ids = tuple(sorted(full_subjects))

    if not reference_ids:
        raise ValueError("reference cohort contains no CamCAN subject directories")
    if expected_subjects is not None and len(reference_ids) != expected_subjects:
        raise ValueError(
            f"reference cohort has {len(reference_ids)} subjects; "
            f"expected exactly {expected_subjects}"
        )
    if (
        expected_full_subjects is not None
        and len(full_dataset_ids) != expected_full_subjects
    ):
        raise ValueError(
            f"full dataset has {len(full_dataset_ids)} subjects; "
            f"expected exactly {expected_full_subjects}"
        )

    missing_ids = tuple(sorted(set(reference_ids) - set(full_dataset_ids)))
    if missing_ids and not allow_missing_targets:
        raise ValueError(
            f"{len(missing_ids)} reference subjects are absent from the full dataset; "
            "refusing a partial cleanup (use --allow-missing-targets only to resume)"
        )

    matching_ids = sorted(set(reference_ids) & set(full_dataset_ids))
    if not matching_ids:
        raise ValueError("no reference subjects exist in the full dataset")
    targets = tuple(full_subjects[subject_id] for subject_id in matching_ids)

    return CleanupPlan(
        reference_root=reference_root,
        full_dataset_root=full_dataset_root,
        reference_ids=reference_ids,
        full_dataset_ids=full_dataset_ids,
        targets=targets,
        missing_ids=missing_ids,
    )


def print_plan(plan: CleanupPlan, summary_only: bool) -> None:
    print(f"Reference cohort:       {plan.reference_root}")
    print(f"Full dataset:           {plan.full_dataset_root}")
    print(f"Reference subjects:     {len(plan.reference_ids)}")
    print(f"Full dataset subjects:  {len(plan.full_dataset_ids)}")
    print(f"Matching directories:   {len(plan.targets)}")
    print(f"Missing reference IDs:  {len(plan.missing_ids)}")
    print(f"Subjects after cleanup: {plan.remaining_subjects}")

    if plan.missing_ids:
        print("\nAlready absent from the full dataset:")
        for subject_id in plan.missing_ids:
            print(f"  {subject_id}")

    if not summary_only:
        print("\nPlanned permanent deletions:")
        for target in plan.targets:
            print(f"  DELETE {target}")


def confirm_deletion(plan: CleanupPlan) -> bool:
    if not sys.stdin.isatty():
        print(
            "ERROR: --apply requires an interactive terminal or the explicit --yes flag",
            file=sys.stderr,
        )
        return False
    phrase = f"DELETE {len(plan.targets)} SUBJECTS"
    print("\nWARNING: this permanently removes every directory listed above.")
    try:
        answer = input(f'Type exactly "{phrase}" to continue: ')
    except EOFError:
        return False
    if answer != phrase:
        print("Confirmation did not match; nothing was deleted.", file=sys.stderr)
        return False
    return True


def default_manifest_path(plan: CleanupPlan) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{plan.full_dataset_root.name}_cleanup_{timestamp}.tsv"
    return plan.full_dataset_root.parent / filename


def open_manifest(path: Path, plan: CleanupPlan) -> tuple[TextIO, csv.writer]:
    manifest = path.expanduser().resolve(strict=False)
    if not manifest.parent.is_dir():
        raise ValueError(f"manifest parent directory does not exist: {manifest.parent}")
    try:
        stream = manifest.open("x", encoding="utf-8", newline="")
    except OSError as error:
        raise ValueError(f"cannot create manifest: {manifest} ({error})") from error

    stream.write(f"# reference_cohort\t{plan.reference_root}\n")
    stream.write(f"# full_dataset\t{plan.full_dataset_root}\n")
    stream.write(f"# reference_subjects\t{len(plan.reference_ids)}\n")
    stream.write(f"# full_dataset_subjects_before\t{len(plan.full_dataset_ids)}\n")
    stream.write(f"# planned_deletions\t{len(plan.targets)}\n")
    stream.write(f"# projected_subjects_after\t{plan.remaining_subjects}\n")
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(["timestamp_utc", "subject_id", "target", "status", "detail"])
    for target in plan.targets:
        writer.writerow([utc_now(), target.name, target, "planned", ""])
    stream.flush()
    os.fsync(stream.fileno())
    return stream, writer


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_event(
    stream: TextIO,
    writer: csv.writer,
    target: Path,
    status_value: str,
    detail: str = "",
) -> None:
    safe_detail = detail.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    writer.writerow([utc_now(), target.name, target, status_value, safe_detail])
    stream.flush()
    os.fsync(stream.fileno())


def validate_target_again(target: Path, full_dataset_root: Path) -> None:
    if target.parent != full_dataset_root:
        raise ValueError("target is not a direct child of the full dataset")
    if SUBJECT_PATTERN.fullmatch(target.name) is None:
        raise ValueError("target name is not a valid CamCAN subject ID")
    try:
        target_stat = target.lstat()
    except OSError as error:
        raise ValueError(f"target cannot be inspected: {error}") from error
    if stat.S_ISLNK(target_stat.st_mode):
        raise ValueError("target became a symbolic link")
    if not stat.S_ISDIR(target_stat.st_mode):
        raise ValueError("target is no longer a directory")
    if target.resolve(strict=True).parent != full_dataset_root:
        raise ValueError("resolved target escaped the full dataset")


def apply_plan(plan: CleanupPlan, manifest_path: Path) -> int:
    try:
        stream, writer = open_manifest(manifest_path, plan)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    deleted = 0
    failures = 0
    resolved_manifest = Path(stream.name)
    print(f"\nManifest: {resolved_manifest}")
    with stream:
        for target in plan.targets:
            try:
                validate_target_again(target, plan.full_dataset_root)
                shutil.rmtree(target)
            except (OSError, ValueError) as error:
                failures += 1
                record_event(stream, writer, target, "failed", str(error))
                print(f"FAILED  {target}: {error}", file=sys.stderr)
            else:
                deleted += 1
                record_event(stream, writer, target, "deleted")
                print(f"DELETED {target}")

    try:
        remaining_subjects = discover_subjects(plan.full_dataset_root, "full dataset")
    except ValueError as error:
        print(f"ERROR: post-cleanup verification failed: {error}", file=sys.stderr)
        return 1
    planned_ids = {target.name for target in plan.targets}
    still_present = sorted(planned_ids & set(remaining_subjects))

    print(
        f"\nFinished: {deleted} deleted, {failures} failed, "
        f"{len(remaining_subjects)} subjects remain."
    )
    if still_present:
        print(
            f"ERROR: {len(still_present)} planned subjects are still present",
            file=sys.stderr,
        )
        return 1
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.yes and not args.apply:
        print("ERROR: --yes is valid only with --apply", file=sys.stderr)
        return 2
    if args.manifest is not None and not args.apply:
        print("ERROR: --manifest is valid only with --apply", file=sys.stderr)
        return 2

    try:
        plan = build_plan(
            args.reference_cohort,
            args.full_dataset,
            args.expected_subjects,
            args.expected_full_subjects,
            args.allow_missing_targets,
        )
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print_plan(plan, args.summary_only)
    if not args.apply:
        print("\nAUDIT ONLY: nothing was deleted. Re-run with --apply after review.")
        return 0

    if not args.yes and not confirm_deletion(plan):
        return 2
    manifest_path = args.manifest or default_manifest_path(plan)
    return apply_plan(plan, manifest_path)


if __name__ == "__main__":
    raise SystemExit(main())
