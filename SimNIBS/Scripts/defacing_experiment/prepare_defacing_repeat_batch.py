#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


DEFAULT_TARGETS = ("left-hippocampus", "left-m1")
TASK_FIELDNAMES = (
    "task_id",
    "dataset_name",
    "repeat_id",
    "dataset_root",
    "subject",
    "anat_dir",
    "status",
    "message",
)
EXPERIMENT_FIELDNAMES = (
    "target",
    "condition",
    "parent_root",
    "dataset_name",
    "repeat_id",
    "dataset_root",
    "subject",
    "t1_source",
    "t2_source",
)


@dataclass(frozen=True)
class RepeatBatchSpec:
    target: str
    condition: str
    parent_root: Path
    repeat_ids: tuple[str, ...]


def _condition_suffix(condition: str) -> str:
    normalized = condition.strip().lower()
    if normalized == "intact":
        return "Intact"
    if normalized == "defaced":
        return "Defaced"
    raise ValueError(f"Unsupported condition: {condition}")


def _target_prefix(target: str) -> str:
    parts = [part for part in target.strip().replace("_", "-").split("-") if part]
    if not parts:
        raise ValueError(f"Invalid target: {target!r}")
    return "_".join(part.capitalize() for part in parts)


def build_repeat_batch_specs(
    *,
    out_root: str | Path,
    subject: str,
    repeats: int,
    targets: Sequence[str] = DEFAULT_TARGETS,
) -> tuple[RepeatBatchSpec, ...]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    _ = subject  # kept in signature so callers can validate one subject workflow explicitly
    repeat_ids = tuple(f"{idx:02d}" for idx in range(1, repeats + 1))
    root = Path(out_root).expanduser().resolve()
    specs: list[RepeatBatchSpec] = []
    for target in targets:
        prefix = _target_prefix(target)
        for condition in ("intact", "defaced"):
            specs.append(
                RepeatBatchSpec(
                    target=target,
                    condition=condition,
                    parent_root=root / f"{prefix}_{_condition_suffix(condition)}",
                    repeat_ids=repeat_ids,
                )
            )
    return tuple(specs)


def resample_keep_mask_to_target(
    keep_mask_img: nib.spatialimages.SpatialImage,
    target_img: nib.spatialimages.SpatialImage,
) -> nib.Nifti1Image:
    if keep_mask_img.shape == target_img.shape and np.allclose(
        keep_mask_img.affine,
        target_img.affine,
        atol=1e-5,
    ):
        data = (np.asanyarray(keep_mask_img.dataobj) > 0.5).astype(np.uint8, copy=False)
        out = nib.Nifti1Image(data, target_img.affine, target_img.header)
        out.set_data_dtype(np.uint8)
        return out

    resampled = resample_from_to(keep_mask_img, target_img, order=0)
    data = (np.asanyarray(resampled.dataobj) > 0.5).astype(np.uint8, copy=False)
    out = nib.Nifti1Image(data, target_img.affine, target_img.header)
    out.set_data_dtype(np.uint8)
    return out


def apply_keep_mask(
    source_img: nib.spatialimages.SpatialImage,
    keep_mask_img: nib.spatialimages.SpatialImage,
) -> nib.Nifti1Image:
    if source_img.shape != keep_mask_img.shape or not np.allclose(
        source_img.affine,
        keep_mask_img.affine,
        atol=1e-5,
    ):
        keep_mask_img = resample_keep_mask_to_target(keep_mask_img, source_img)

    keep = np.asanyarray(keep_mask_img.dataobj) > 0.5
    source = np.asanyarray(source_img.dataobj)
    out_data = np.where(keep, source, 0).astype(source.dtype, copy=False)
    out = nib.Nifti1Image(out_data, source_img.affine, source_img.header)
    out.set_data_dtype(source_img.get_data_dtype())
    return out


def _nifti_suffix(path: str | Path) -> str:
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    if name.endswith(".nii"):
        return ".nii"
    raise ValueError(f"Expected a NIfTI filename ending in .nii or .nii.gz: {path}")


def _write_tsv(path: str | Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> Path:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: str(row.get(field, "")) for field in fieldnames})
    return out


def _pydeface_mask_sidecar(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    suffix = _nifti_suffix(path)
    stem = path.name[: -len(suffix)]
    return path.with_name(f"{stem}_pydeface_mask.nii.gz")


def _pydeface_mat_sidecar(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    suffix = _nifti_suffix(path)
    stem = path.name[: -len(suffix)]
    return path.with_name(f"{stem}_pydeface.mat")


def run_pydeface(
    *,
    intact_t1: str | Path,
    defaced_t1: str | Path,
    keep_mask_path: str | Path,
    pydeface_bin: str,
    force: bool,
) -> None:
    intact_t1 = Path(intact_t1).expanduser().resolve()
    defaced_t1 = Path(defaced_t1).expanduser().resolve()
    keep_mask_path = Path(keep_mask_path).expanduser().resolve()

    if defaced_t1.exists() and keep_mask_path.exists() and not force:
        return

    defaced_t1.parent.mkdir(parents=True, exist_ok=True)
    keep_mask_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_pydeface = shutil.which(pydeface_bin)
    if resolved_pydeface is None:
        raise RuntimeError(f"pydeface executable not found: {pydeface_bin}")

    work_dir = defaced_t1.parent / "_pydeface_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    temp_input_t1 = work_dir / intact_t1.name
    temp_mask_path = _pydeface_mask_sidecar(temp_input_t1)
    temp_mat_path = _pydeface_mat_sidecar(temp_input_t1)

    if temp_input_t1.exists():
        temp_input_t1.unlink()
    if temp_mask_path.exists():
        temp_mask_path.unlink()
    if temp_mat_path.exists():
        temp_mat_path.unlink()

    shutil.copy2(intact_t1, temp_input_t1)
    if defaced_t1.exists() and force:
        defaced_t1.unlink()
    if keep_mask_path.exists() and force:
        keep_mask_path.unlink()

    cmd = [
        resolved_pydeface,
        str(temp_input_t1),
        "--outfile",
        str(defaced_t1),
        "--force",
        "--nocleanup",
    ]

    print(f"[CMD] {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    pydeface_dir = str(Path(resolved_pydeface).parent)
    env["PATH"] = f"{pydeface_dir}:{env.get('PATH', '')}" if env.get("PATH") else pydeface_dir
    result = subprocess.run(cmd, env=env)
    try:
        if result.returncode != 0:
            raise RuntimeError(f"pydeface failed with exit code {result.returncode}")
        if not temp_mask_path.exists():
            raise RuntimeError(f"pydeface did not produce expected mask: {temp_mask_path}")
        shutil.move(str(temp_mask_path), str(keep_mask_path))
    finally:
        if temp_input_t1.exists():
            temp_input_t1.unlink()
        if temp_mask_path.exists():
            temp_mask_path.unlink()
        if temp_mat_path.exists():
            temp_mat_path.unlink()
        if work_dir.exists():
            try:
                work_dir.rmdir()
            except OSError:
                pass


def generate_defaced_modalities(
    *,
    subject: str,
    intact_t1: str | Path,
    intact_t2: str | Path,
    generated_root: str | Path,
    pydeface_bin: str = "pydeface",
    force: bool = False,
) -> tuple[Path, Path, Path, Path]:
    generated_root = Path(generated_root).expanduser().resolve()
    anat_dir = generated_root / subject / "anat"
    anat_dir.mkdir(parents=True, exist_ok=True)

    t1_suffix = _nifti_suffix(intact_t1)
    t2_suffix = _nifti_suffix(intact_t2)

    defaced_t1 = anat_dir / f"{subject}_desc-deface_T1w{t1_suffix}"
    keep_mask_t1 = anat_dir / f"{subject}_desc-deface_mask_T1w{t1_suffix}"
    defaced_t2 = anat_dir / f"{subject}_desc-deface_T2w{t2_suffix}"
    keep_mask_t2 = anat_dir / f"{subject}_desc-deface_mask_T2w{t2_suffix}"

    run_pydeface(
        intact_t1=intact_t1,
        defaced_t1=defaced_t1,
        keep_mask_path=keep_mask_t1,
        pydeface_bin=pydeface_bin,
        force=force,
    )

    if defaced_t2.exists() and keep_mask_t2.exists() and not force:
        return defaced_t1, defaced_t2, keep_mask_t1, keep_mask_t2

    keep_mask_t1_img = nib.load(str(keep_mask_t1))
    intact_t2_img = nib.load(str(intact_t2))
    keep_mask_t2_img = resample_keep_mask_to_target(keep_mask_t1_img, intact_t2_img)
    nib.save(keep_mask_t2_img, str(keep_mask_t2))

    defaced_t2_img = apply_keep_mask(intact_t2_img, keep_mask_t2_img)
    nib.save(defaced_t2_img, str(defaced_t2))
    return defaced_t1, defaced_t2, keep_mask_t1, keep_mask_t2


def _copy_pair_into_repeat(
    *,
    source_t1: str | Path,
    source_t2: str | Path,
    dest_anat_dir: str | Path,
    dest_t1_name: str,
    dest_t2_name: str,
    force: bool,
) -> None:
    dest_anat_dir = Path(dest_anat_dir)
    dest_anat_dir.mkdir(parents=True, exist_ok=True)
    dest_t1 = dest_anat_dir / dest_t1_name
    dest_t2 = dest_anat_dir / dest_t2_name

    if dest_t1.exists() and not force:
        pass
    else:
        shutil.copy2(source_t1, dest_t1)
    if dest_t2.exists() and not force:
        pass
    else:
        shutil.copy2(source_t2, dest_t2)


def stage_repeat_batches(
    *,
    out_root: str | Path,
    subject: str,
    intact_t1: str | Path,
    intact_t2: str | Path,
    defaced_t1: str | Path,
    defaced_t2: str | Path,
    repeats: int,
    targets: Sequence[str] = DEFAULT_TARGETS,
    force: bool = False,
) -> tuple[Path, ...]:
    specs = build_repeat_batch_specs(
        out_root=out_root,
        subject=subject,
        repeats=repeats,
        targets=targets,
    )
    intact_t1 = Path(intact_t1).expanduser().resolve()
    intact_t2 = Path(intact_t2).expanduser().resolve()
    defaced_t1 = Path(defaced_t1).expanduser().resolve()
    defaced_t2 = Path(defaced_t2).expanduser().resolve()

    experiment_rows: list[dict[str, object]] = []
    parent_roots: list[Path] = []

    for spec in specs:
        parent_roots.append(spec.parent_root)
        manifest_rows: list[dict[str, object]] = []
        for task_id, repeat_id in enumerate(spec.repeat_ids):
            dataset_name = f"{_target_prefix(spec.target)}_Data_{repeat_id}"
            dataset_root = spec.parent_root / dataset_name
            anat_dir = dataset_root / subject / "anat"

            if spec.condition == "intact":
                source_t1 = intact_t1
                source_t2 = intact_t2
            else:
                source_t1 = defaced_t1
                source_t2 = defaced_t2

            _copy_pair_into_repeat(
                source_t1=source_t1,
                source_t2=source_t2,
                dest_anat_dir=anat_dir,
                dest_t1_name=intact_t1.name,
                dest_t2_name=intact_t2.name,
                force=force,
            )

            manifest_rows.append(
                {
                    "task_id": task_id,
                    "dataset_name": dataset_name,
                    "repeat_id": repeat_id,
                    "dataset_root": dataset_root,
                    "subject": subject,
                    "anat_dir": anat_dir,
                    "status": "ready",
                    "message": "staged",
                }
            )
            experiment_rows.append(
                {
                    "target": spec.target,
                    "condition": spec.condition,
                    "parent_root": spec.parent_root,
                    "dataset_name": dataset_name,
                    "repeat_id": repeat_id,
                    "dataset_root": dataset_root,
                    "subject": subject,
                    "t1_source": source_t1,
                    "t2_source": source_t2,
                }
            )

        _write_tsv(spec.parent_root / "slurm" / "manifest.tsv", TASK_FIELDNAMES, manifest_rows)

    _write_tsv(
        Path(out_root).expanduser().resolve() / "experiment_manifest.tsv",
        EXPERIMENT_FIELDNAMES,
        experiment_rows,
    )
    return tuple(parent_roots)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate defaced T1/T2 inputs and stage 10-repeat intact/defaced CamCan experiment roots."
    )
    parser.add_argument("--subject", default="sub-CCMe", help="Subject ID. Default: sub-CCMe")
    parser.add_argument("--intact-t1", type=Path, required=True, help="Path to intact T1 NIfTI")
    parser.add_argument("--intact-t2", type=Path, required=True, help="Path to intact T2 NIfTI")
    parser.add_argument("--out-root", type=Path, required=True, help="Output root for generated defaced files and staged experiment roots")
    parser.add_argument("--repeats", type=int, default=10, help="Number of repeats per arm. Default: 10")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(DEFAULT_TARGETS),
        help="Target presets to stage. Default: left-hippocampus left-m1",
    )
    parser.add_argument("--pydeface-bin", default="pydeface", help="Path or name for pydeface")
    parser.add_argument("--force", action="store_true", help="Overwrite generated defaced files and staged repeats")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.intact_t1.exists():
        raise SystemExit(f"--intact-t1 does not exist: {args.intact_t1}")
    if not args.intact_t2.exists():
        raise SystemExit(f"--intact-t2 does not exist: {args.intact_t2}")

    generated_root = args.out_root.expanduser().resolve() / "_generated_defaced"
    defaced_t1, defaced_t2, keep_mask_t1, keep_mask_t2 = generate_defaced_modalities(
        subject=args.subject,
        intact_t1=args.intact_t1,
        intact_t2=args.intact_t2,
        generated_root=generated_root,
        pydeface_bin=args.pydeface_bin,
        force=args.force,
    )

    parent_roots = stage_repeat_batches(
        out_root=args.out_root,
        subject=args.subject,
        intact_t1=args.intact_t1,
        intact_t2=args.intact_t2,
        defaced_t1=defaced_t1,
        defaced_t2=defaced_t2,
        repeats=args.repeats,
        targets=args.targets,
        force=args.force,
    )

    print("[INFO] Generated defaced outputs:", flush=True)
    print(f"  T1: {defaced_t1}", flush=True)
    print(f"  T2: {defaced_t2}", flush=True)
    print(f"  T1 keep-mask: {keep_mask_t1}", flush=True)
    print(f"  T2 keep-mask: {keep_mask_t2}", flush=True)
    print("[INFO] Staged parent roots:", flush=True)
    for parent_root in parent_roots:
        print(f"  {parent_root}", flush=True)


if __name__ == "__main__":
    main()
