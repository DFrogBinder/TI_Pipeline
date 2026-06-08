#!/usr/bin/env python3
"""Collect compact source-side inputs for mechanistic repeatability figures."""

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional


MANIFEST_NAME = "mechanistic_figure_collection_manifest.csv"
COVERAGE_SELECTION_NAME = "coverage_selection.csv"
COMPUTED_COVERAGE_NAME = "coverage_values.csv"
DEFAULT_FIXED_REFERENCE_REPEAT = "repeat_001"


class CoverageRow(NamedTuple):
    subject: str
    condition: str
    repeat_tag: str
    coverage_0p2: float
    finite_roi_voxels: int
    suprathreshold_voxels: int


class SelectedRepeat(NamedTuple):
    subject: str
    condition: str
    repeat_tag: str
    coverage_0p2: float
    rank_label: str


class CoverageSelection(NamedTuple):
    subject: str
    low: SelectedRepeat
    median: SelectedRepeat
    high: SelectedRepeat
    coverage_range: float


class ManifestRow(NamedTuple):
    role: str
    subject: str
    condition: str
    repeat_tag: str
    source_path: str
    dest_path: str
    status: str
    size_bytes: int
    message: str


class CollectionResult(NamedTuple):
    manifest_rows: List[ManifestRow]
    bytes_planned: int
    bytes_copied: int


def _finite_float(value: object) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def read_coverage_rows(path: Path) -> List[CoverageRow]:
    rows: List[CoverageRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            coverage = _finite_float(raw.get("coverage_0p2"))
            if not math.isfinite(coverage):
                continue
            subject = str(raw.get("subject", "")).strip()
            repeat_tag = str(raw.get("repeat_tag", "")).strip()
            if not subject or not repeat_tag:
                continue
            rows.append(
                CoverageRow(
                    subject=subject,
                    condition=str(raw.get("condition", "")).strip() or "remesh",
                    repeat_tag=repeat_tag,
                    coverage_0p2=coverage,
                    finite_roi_voxels=int(float(raw.get("finite_roi_voxels") or 0)),
                    suprathreshold_voxels=int(float(raw.get("suprathreshold_voxels") or 0)),
                )
            )
    return rows


def _import_nifti_dependencies():
    missing = []
    first_exception: Optional[BaseException] = None
    try:
        import nibabel as nib  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only when environment lacks dependency
        missing.append(exc.name or "nibabel")
        first_exception = exc

    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only when environment lacks dependency
        missing.append(exc.name or "numpy")
        if first_exception is None:
            first_exception = exc

    if missing:
        package_list = ", ".join(dict.fromkeys(missing))
        raise RuntimeError(
            "Computing coverage from NIfTI files requires nibabel and numpy. "
            f"Missing Python package(s): {package_list}. "
            "On Stanage, load the same environment used by the Slurm launchers: "
            'module purge; module use "$HOME/modules"; module load SimNIBS/4.0.1-foss-2023a. '
            "Alternatively, pass --coverage-csv to use precomputed coverage and skip NIfTI imports."
        ) from first_exception
    return nib, np


def _subject_from_repeatability_dir(path: Path) -> str:
    suffix = "_repeatability"
    return path.name[: -len(suffix)] if path.name.endswith(suffix) else path.name


def discover_subjects(base_output_dir: Path, preferred_subject: Optional[str] = None) -> List[str]:
    if preferred_subject is not None:
        return [preferred_subject]
    subjects = [
        _subject_from_repeatability_dir(path)
        for path in sorted(base_output_dir.glob("sub-*_repeatability"))
        if path.is_dir()
    ]
    return subjects


def roi_mask_path(base_output_dir: Path, subject: str) -> Path:
    return base_output_dir / "_analysis" / subject / "remesh" / "roi_mask_on_reference_ti.nii.gz"


def repeat_dirs(base_output_dir: Path, subject: str, condition: str) -> List[Path]:
    root = base_output_dir / f"{subject}_repeatability" / condition / "repeats"
    return sorted((path for path in root.glob("repeat_*") if path.is_dir()), key=lambda path: _repeat_index(path.name))


def compute_coverage_rows(
    base_output_dir: Path,
    *,
    threshold: float = 0.2,
    preferred_subject: Optional[str] = None,
) -> List[CoverageRow]:
    nib, np = _import_nifti_dependencies()
    rows: List[CoverageRow] = []
    for subject in discover_subjects(base_output_dir, preferred_subject=preferred_subject):
        mask_file = roi_mask_path(base_output_dir, subject)
        if not mask_file.is_file():
            continue
        mask_data = np.asanyarray(nib.load(mask_file).dataobj) > 0
        for repeat_dir in repeat_dirs(base_output_dir, subject, "remesh"):
            ti_file = repeat_ti_volume_path(base_output_dir, subject, "remesh", repeat_dir.name)
            if not ti_file.is_file():
                continue
            ti_data = np.asanyarray(nib.load(ti_file).dataobj, dtype=float)
            if ti_data.shape != mask_data.shape:
                raise ValueError(
                    f"Shape mismatch for {subject} {repeat_dir.name}: "
                    f"TI shape {ti_data.shape}, ROI mask shape {mask_data.shape}"
                )
            roi_values = ti_data[mask_data]
            finite = np.isfinite(roi_values)
            finite_count = int(np.count_nonzero(finite))
            if finite_count == 0:
                continue
            suprathreshold = int(np.count_nonzero(roi_values[finite] > threshold))
            rows.append(
                CoverageRow(
                    subject=subject,
                    condition="remesh",
                    repeat_tag=repeat_dir.name,
                    coverage_0p2=suprathreshold / finite_count,
                    finite_roi_voxels=finite_count,
                    suprathreshold_voxels=suprathreshold,
                )
            )
    return rows


def _repeat_index(repeat_tag: str) -> int:
    digits = "".join(ch for ch in repeat_tag if ch.isdigit())
    return int(digits) if digits else 10**9


def _selected(row: CoverageRow, rank_label: str) -> SelectedRepeat:
    return SelectedRepeat(
        subject=row.subject,
        condition=row.condition,
        repeat_tag=row.repeat_tag,
        coverage_0p2=row.coverage_0p2,
        rank_label=rank_label,
    )


def select_subject_and_repeats(rows: List[CoverageRow], preferred_subject: Optional[str] = None) -> CoverageSelection:
    by_subject: Dict[str, List[CoverageRow]] = {}
    for row in rows:
        if row.condition != "remesh":
            continue
        by_subject.setdefault(row.subject, []).append(row)
    if not by_subject:
        raise ValueError("No finite remesh coverage rows were found.")

    if preferred_subject is not None:
        if preferred_subject not in by_subject:
            raise ValueError(f"Preferred subject has no finite remesh coverage rows: {preferred_subject}")
        subject = preferred_subject
    else:
        subject = max(
            by_subject,
            key=lambda item: max(row.coverage_0p2 for row in by_subject[item])
            - min(row.coverage_0p2 for row in by_subject[item]),
        )

    ordered = sorted(by_subject[subject], key=lambda row: (row.coverage_0p2, _repeat_index(row.repeat_tag)))
    low = ordered[0]
    high = ordered[-1]
    if len(ordered) % 2:
        median_target = ordered[len(ordered) // 2].coverage_0p2
    else:
        upper = len(ordered) // 2
        median_target = 0.5 * (ordered[upper - 1].coverage_0p2 + ordered[upper].coverage_0p2)
    median = min(ordered, key=lambda row: (abs(row.coverage_0p2 - median_target), _repeat_index(row.repeat_tag)))
    return CoverageSelection(
        subject=subject,
        low=_selected(low, "low"),
        median=_selected(median, "median"),
        high=_selected(high, "high"),
        coverage_range=high.coverage_0p2 - low.coverage_0p2,
    )


def repeat_anat_dir(base_output_dir: Path, subject: str, condition: str, repeat_tag: str) -> Path:
    return base_output_dir / f"{subject}_repeatability" / condition / "repeats" / repeat_tag / subject / "anat"


def repeat_mesh_path(base_output_dir: Path, subject: str, condition: str, repeat_tag: str) -> Path:
    return repeat_anat_dir(base_output_dir, subject, condition, repeat_tag) / f"m2m_{subject}" / f"{subject}.msh"


def repeat_ti_volume_path(base_output_dir: Path, subject: str, condition: str, repeat_tag: str) -> Path:
    return repeat_anat_dir(base_output_dir, subject, condition, repeat_tag) / "SimNIBS" / "ti_brain_only.nii.gz"


def fixed_mesh_cache_path(base_output_dir: Path, subject: str) -> Path:
    return (
        base_output_dir
        / f"{subject}_repeatability"
        / "fixed_mesh"
        / "mesh_cache"
        / subject
        / "anat"
        / f"m2m_{subject}"
        / f"{subject}.msh"
    )


def analysis_file(base_output_dir: Path, subject: str, condition: str, filename: str) -> Path:
    return base_output_dir / "_analysis" / subject / condition / filename


def _manifest_row_for_file(
    *,
    role: str,
    subject: str,
    condition: str,
    repeat_tag: str,
    source: Path,
    dest: Path,
) -> ManifestRow:
    if source.is_file():
        return ManifestRow(
            role=role,
            subject=subject,
            condition=condition,
            repeat_tag=repeat_tag,
            source_path=str(source),
            dest_path=str(dest),
            status="planned",
            size_bytes=source.stat().st_size,
            message="file found",
        )
    return ManifestRow(
        role=role,
        subject=subject,
        condition=condition,
        repeat_tag=repeat_tag,
        source_path=str(source),
        dest_path=str(dest),
        status="missing",
        size_bytes=0,
        message="source file missing",
    )


def _selected_repeat_files(base_output_dir: Path, out_dir: Path, repeat: SelectedRepeat) -> List[ManifestRow]:
    mesh_source = repeat_mesh_path(base_output_dir, repeat.subject, repeat.condition, repeat.repeat_tag)
    ti_source = repeat_ti_volume_path(base_output_dir, repeat.subject, repeat.condition, repeat.repeat_tag)
    base_dest = out_dir / "selected_repeats" / repeat.rank_label / repeat.condition / repeat.repeat_tag
    return [
        _manifest_row_for_file(
            role="selected_remesh_mesh",
            subject=repeat.subject,
            condition=repeat.condition,
            repeat_tag=repeat.repeat_tag,
            source=mesh_source,
            dest=base_dest / mesh_source.name,
        ),
        _manifest_row_for_file(
            role="selected_remesh_ti_volume",
            subject=repeat.subject,
            condition=repeat.condition,
            repeat_tag=repeat.repeat_tag,
            source=ti_source,
            dest=base_dest / ti_source.name,
        ),
    ]


def plan_manifest(
    base_output_dir: Path,
    out_dir: Path,
    selection: CoverageSelection,
    fixed_reference_repeat: str = DEFAULT_FIXED_REFERENCE_REPEAT,
) -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    for repeat in (selection.low, selection.median, selection.high):
        rows.extend(_selected_repeat_files(base_output_dir, out_dir, repeat))

    for repeat_dir in repeat_dirs(base_output_dir, selection.subject, "remesh"):
        source = repeat_ti_volume_path(base_output_dir, selection.subject, "remesh", repeat_dir.name)
        dest = out_dir / "all_ti_volumes" / selection.subject / "remesh" / repeat_dir.name / source.name
        rows.append(
            _manifest_row_for_file(
                role="repeat_ti_volume",
                subject=selection.subject,
                condition="remesh",
                repeat_tag=repeat_dir.name,
                source=source,
                dest=dest,
            )
        )

    for repeat_dir in repeat_dirs(base_output_dir, selection.subject, "fixed_mesh"):
        source = repeat_ti_volume_path(base_output_dir, selection.subject, "fixed_mesh", repeat_dir.name)
        dest = out_dir / "all_ti_volumes" / selection.subject / "fixed_mesh" / repeat_dir.name / source.name
        rows.append(
            _manifest_row_for_file(
                role="fixed_repeat_ti_volume",
                subject=selection.subject,
                condition="fixed_mesh",
                repeat_tag=repeat_dir.name,
                source=source,
                dest=dest,
            )
        )

    fixed_mesh_source = repeat_mesh_path(base_output_dir, selection.subject, "fixed_mesh", fixed_reference_repeat)
    if not fixed_mesh_source.is_file():
        fixed_mesh_source = fixed_mesh_cache_path(base_output_dir, selection.subject)
    fixed_ti_source = repeat_ti_volume_path(base_output_dir, selection.subject, "fixed_mesh", fixed_reference_repeat)
    fixed_dest = out_dir / "fixed_mesh_reference" / fixed_reference_repeat
    rows.append(
        _manifest_row_for_file(
            role="fixed_mesh_reference_mesh",
            subject=selection.subject,
            condition="fixed_mesh",
            repeat_tag=fixed_reference_repeat,
            source=fixed_mesh_source,
            dest=fixed_dest / fixed_mesh_source.name,
        )
    )
    rows.append(
        _manifest_row_for_file(
            role="fixed_mesh_reference_ti_volume",
            subject=selection.subject,
            condition="fixed_mesh",
            repeat_tag=fixed_reference_repeat,
            source=fixed_ti_source,
            dest=fixed_dest / fixed_ti_source.name,
        )
    )

    for condition in ("remesh", "fixed_mesh"):
        for filename, role in (
            ("summary.csv", "condition_summary"),
            ("summary.json", "condition_summary_json"),
            ("repeatability_stats.csv", "repeatability_stats"),
            ("repeatability_stats.json", "repeatability_stats_json"),
            ("label_diff_overlay.png", "label_diff_overlay"),
            ("label_diff_frequency.nii.gz", "label_diff_frequency"),
            ("roi_mask_on_reference_ti.nii.gz", "roi_mask_reference_ti"),
            ("roi_mask_on_t1.nii.gz", "roi_mask_t1"),
            ("roi_outline_on_t1.png", "roi_outline_t1"),
            ("roi_outline_on_mean_ti.png", "roi_outline_mean_ti"),
        ):
            source = analysis_file(base_output_dir, selection.subject, condition, filename)
            dest = out_dir / "_analysis" / selection.subject / condition / filename
            rows.append(
                _manifest_row_for_file(
                    role=role,
                    subject=selection.subject,
                    condition=condition,
                    repeat_tag="",
                    source=source,
                    dest=dest,
                )
            )
    return rows


def write_manifest(path: Path, rows: List[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "role",
        "subject",
        "condition",
        "repeat_tag",
        "source_path",
        "dest_path",
        "status",
        "size_bytes",
        "message",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def write_coverage_selection(path: Path, selection: CoverageSelection) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subject", "rank_label", "condition", "repeat_tag", "coverage_0p2", "coverage_range"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for repeat in (selection.low, selection.median, selection.high):
            writer.writerow(
                {
                    "subject": selection.subject,
                    "rank_label": repeat.rank_label,
                    "condition": repeat.condition,
                    "repeat_tag": repeat.repeat_tag,
                    "coverage_0p2": repeat.coverage_0p2,
                    "coverage_range": selection.coverage_range,
                }
            )


def write_coverage_rows(path: Path, rows: List[CoverageRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject",
        "condition",
        "repeat_tag",
        "coverage_0p2",
        "finite_roi_voxels",
        "suprathreshold_voxels",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def collect_mechanistic_inputs(
    *,
    base_output_dir: Path,
    out_dir: Path,
    coverage_csv: Optional[Path],
    dry_run: bool,
    copy_files: bool,
    max_total_bytes: int,
    allow_large: bool,
    threshold: float = 0.2,
    preferred_subject: Optional[str] = None,
    fixed_reference_repeat: str = DEFAULT_FIXED_REFERENCE_REPEAT,
) -> CollectionResult:
    if coverage_csv is None:
        coverage_rows = compute_coverage_rows(
            base_output_dir,
            threshold=threshold,
            preferred_subject=preferred_subject,
        )
        write_coverage_rows(out_dir / COMPUTED_COVERAGE_NAME, coverage_rows)
        if not coverage_rows:
            subject_text = f" for {preferred_subject}" if preferred_subject else ""
            raise ValueError(
                f"No finite remesh coverage rows were computed{subject_text}. "
                "Check that _analysis/<subject>/remesh/roi_mask_on_reference_ti.nii.gz exists, "
                "that remesh repeats contain anat/SimNIBS/ti_brain_only.nii.gz, "
                "and that the ROI mask has finite overlap with the TI volumes."
            )
    else:
        coverage_rows = read_coverage_rows(coverage_csv)
    selection = select_subject_and_repeats(coverage_rows, preferred_subject=preferred_subject)
    manifest = plan_manifest(
        base_output_dir=base_output_dir,
        out_dir=out_dir,
        selection=selection,
        fixed_reference_repeat=fixed_reference_repeat,
    )
    bytes_planned = sum(row.size_bytes for row in manifest if row.status == "planned")
    write_manifest(out_dir / MANIFEST_NAME, manifest)
    write_coverage_selection(out_dir / COVERAGE_SELECTION_NAME, selection)

    if copy_files and not dry_run and bytes_planned > max_total_bytes and not allow_large:
        raise RuntimeError(f"planned bundle size {bytes_planned} exceeds --max-total-bytes {max_total_bytes}")

    bytes_copied = 0
    if copy_files and not dry_run:
        for row in manifest:
            if row.status != "planned":
                continue
            source = Path(row.source_path)
            dest = Path(row.dest_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            bytes_copied += row.size_bytes

    return CollectionResult(manifest_rows=manifest, bytes_planned=bytes_planned, bytes_copied=bytes_copied)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-output-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--check-nifti-dependencies",
        action="store_true",
        help="Import nibabel and numpy, print a status line, and exit.",
    )
    parser.add_argument(
        "--coverage-csv",
        type=Path,
        default=None,
        help="Optional precomputed coverage CSV. If omitted, coverage is computed from ti_brain_only.nii.gz files.",
    )
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--preferred-subject", default=None)
    parser.add_argument("--fixed-reference-repeat", default=DEFAULT_FIXED_REFERENCE_REPEAT)
    parser.add_argument("--max-total-bytes", type=int, default=5_000_000_000)
    parser.add_argument("--allow-large", action="store_true")
    parser.add_argument("--copy", action="store_true", help="Copy planned files. Without this flag, only manifests are written.")
    parser.add_argument("--dry-run", action="store_true", help="Write manifests without copying files.")
    args = parser.parse_args()
    if not args.check_nifti_dependencies:
        if args.base_output_dir is None:
            parser.error("--base-output-dir is required unless --check-nifti-dependencies is used.")
        if args.out_dir is None:
            parser.error("--out-dir is required unless --check-nifti-dependencies is used.")
    return args


def main() -> None:
    args = parse_args()
    if args.check_nifti_dependencies:
        _import_nifti_dependencies()
        print("nifti_dependencies=ok")
        return

    dry_run = args.dry_run or not args.copy
    result = collect_mechanistic_inputs(
        base_output_dir=args.base_output_dir,
        out_dir=args.out_dir,
        coverage_csv=args.coverage_csv,
        dry_run=dry_run,
        copy_files=args.copy,
        max_total_bytes=args.max_total_bytes,
        allow_large=args.allow_large,
        threshold=args.threshold,
        preferred_subject=args.preferred_subject,
        fixed_reference_repeat=args.fixed_reference_repeat,
    )
    mode = "dry-run" if dry_run else "copy"
    print(f"mode={mode}")
    print(f"manifest={args.out_dir / MANIFEST_NAME}")
    print(f"coverage_selection={args.out_dir / COVERAGE_SELECTION_NAME}")
    print(f"bytes_planned={result.bytes_planned}")
    print(f"bytes_copied={result.bytes_copied}")


if __name__ == "__main__":
    main()
