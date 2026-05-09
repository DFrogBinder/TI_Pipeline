#!/usr/bin/env python3
"""Validate that a subject-level TI simulation reached usable final outputs."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paths import sim_output_dir, ti_brain_path


DEFAULT_ROOT = "/mnt/parscratch/users/cop23bi/LM1"


@dataclass(frozen=True)
class OutputCheck:
    name: str
    path: str
    ok: bool
    reason: str
    size_bytes: int | None = None


@dataclass(frozen=True)
class ValidationResult:
    root: str
    subject: str
    ok: bool
    checks: tuple[OutputCheck, ...]
    warnings: tuple[str, ...] = ()


def _is_nifti(path: Path) -> bool:
    return path.name.endswith(".nii") or path.name.endswith(".nii.gz")


def _nifti_candidates(directory: Path, prefix: str) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.name.startswith(prefix) and _is_nifti(path)
    )


def _check_file(path: Path, *, name: str, min_bytes: int) -> OutputCheck:
    if not path.is_file():
        return OutputCheck(name=name, path=str(path), ok=False, reason="missing")

    size = path.stat().st_size
    if size < min_bytes:
        return OutputCheck(
            name=name,
            path=str(path),
            ok=False,
            reason=f"too small: {size} bytes < {min_bytes} bytes",
            size_bytes=size,
        )

    return OutputCheck(
        name=name,
        path=str(path),
        ok=True,
        reason="present",
        size_bytes=size,
    )


def _check_first_candidate(
    candidates: Iterable[Path],
    *,
    directory: Path,
    prefix: str,
    name: str,
    min_bytes: int,
) -> OutputCheck:
    candidates = list(candidates)
    if not candidates:
        return OutputCheck(
            name=name,
            path=str(directory / f"{prefix}*.nii*"),
            ok=False,
            reason="missing",
        )
    return _check_file(candidates[0], name=name, min_bytes=min_bytes)


def _check_nifti_payload(path: Path, *, require_finite_values: bool) -> tuple[bool, str]:
    try:
        import nibabel as nib
        import numpy as np
    except Exception as exc:
        return True, f"NIfTI payload check skipped; import failed: {exc}"

    try:
        img = nib.load(str(path))
    except Exception as exc:
        return False, f"unreadable NIfTI: {exc}"

    if not img.shape or any(int(dim) <= 0 for dim in img.shape):
        return False, f"invalid shape: {img.shape}"

    if require_finite_values:
        try:
            data = np.asanyarray(img.dataobj)
        except Exception as exc:
            return False, f"could not read NIfTI data: {exc}"
        if data.size == 0:
            return False, "empty NIfTI data"
        if not np.isfinite(data).any():
            return False, "no finite NIfTI values"

    return True, f"loadable NIfTI shape={tuple(int(dim) for dim in img.shape)}"


def _with_nifti_check(
    check: OutputCheck,
    *,
    enabled: bool,
    require_finite_values: bool,
) -> OutputCheck:
    if not enabled or not check.ok or not _is_nifti(Path(check.path)):
        return check

    ok, reason = _check_nifti_payload(
        Path(check.path),
        require_finite_values=require_finite_values,
    )
    return OutputCheck(
        name=check.name,
        path=check.path,
        ok=ok,
        reason=f"{check.reason}; {reason}",
        size_bytes=check.size_bytes,
    )


def validate_subject_outputs(
    root: str | Path,
    subject: str,
    *,
    min_bytes: int = 1,
    check_nifti: bool = True,
) -> ValidationResult:
    root_path = Path(root).expanduser()
    output_dir = sim_output_dir(str(root_path), subject)
    volume_base_dir = output_dir / "Volume_Base"
    volume_labels_dir = output_dir / "Volume_Labels"

    checks = [
        _check_file(output_dir / "TI.msh", name="ti_mesh", min_bytes=min_bytes),
        _check_first_candidate(
            _nifti_candidates(volume_base_dir, "TI_Volumetric_"),
            directory=volume_base_dir,
            prefix="TI_Volumetric_",
            name="ti_volume",
            min_bytes=min_bytes,
        ),
        _check_first_candidate(
            _nifti_candidates(volume_labels_dir, "TI_Volumetric_"),
            directory=volume_labels_dir,
            prefix="TI_Volumetric_",
            name="ti_labels",
            min_bytes=min_bytes,
        ),
        _check_file(
            ti_brain_path(str(root_path), subject),
            name="ti_brain_only",
            min_bytes=min_bytes,
        ),
    ]
    checks = [
        _with_nifti_check(
            check,
            enabled=check_nifti,
            require_finite_values=(check.name == "ti_brain_only"),
        )
        for check in checks
    ]

    warnings = []
    if not root_path.is_dir():
        warnings.append(f"simulation root does not exist: {root_path}")

    return ValidationResult(
        root=str(root_path),
        subject=subject,
        ok=all(check.ok for check in checks),
        checks=tuple(checks),
        warnings=tuple(warnings),
    )


def _print_result(result: ValidationResult) -> None:
    for warning in result.warnings:
        print(f"[WARN] {warning}")

    for check in result.checks:
        status = "OK" if check.ok else "FAIL"
        size = "" if check.size_bytes is None else f" ({check.size_bytes} bytes)"
        print(f"[{status}] {check.name}: {check.path}{size} - {check.reason}")

    if result.ok:
        print(
            f"[INFO] Simulation output validation passed for {result.subject} "
            f"under {result.root}."
        )
    else:
        print(
            f"[ERROR] Simulation output validation failed for {result.subject} "
            f"under {result.root}."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether a subject simulation produced the final files needed "
            "by post-processing."
        )
    )
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Simulation dataset root.")
    parser.add_argument("--subject", required=True, help="Subject ID, e.g. sub-CC110056.")
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=1,
        help="Minimum acceptable byte size for each required output.",
    )
    parser.add_argument(
        "--skip-nifti-load",
        action="store_true",
        help="Only check file presence and size; do not load NIfTI payloads.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_subject_outputs(
        args.root,
        args.subject,
        min_bytes=args.min_bytes,
        check_nifti=not args.skip_nifti_load,
    )
    _print_result(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
