#!/usr/bin/env python3
"""
FastSurfer-backed brain-only anonymization (face removed) in one script.

What it does
------------
1) For each input T1w NIfTI, ensure a FastSurfer brain mask exists.
   - FastSurfer typically writes:  <subjects_dir>/<sid>/mri/mask.mgz
   - FreeSurfer typically writes:  <subjects_dir>/<sid>/mri/brainmask.mgz
   If either exists, FastSurfer is SKIPPED.
2) Apply that mask to the original T1 to create a brain-only NIfTI:
   - voxels outside mask set to 0
   - output named with "_desc-brainonly"

Key behavior
------------
- FastSurfer is only run if no existing mask is found.
- --force overwrites the *final output image* only (does NOT force FastSurfer rerun).
- If you want to force FastSurfer rerun, use --rerun-fastsurfer.

Requirements
------------
- FastSurfer repo/installation and launcher script (run_fastsurfer.sh) available.
- Python deps: nibabel, numpy
  pip install nibabel numpy

Example (single file)
---------------------
python deface_batch.py \
  --t1 ~/sandbox/Jake_Data/tmp/defacing/sub-CCMe_T1w.nii.gz \
  --sid sub-CCMe \
  --out-root ~/sandbox/Jake_Data/tmp/defacing_out \
  --subjects-dir ~/sandbox/Jake_Data/tmp/fastsurfer_subjects \
  --fastsurfer-sh /home/boyan/sandbox/utils/FastSurfer/run_fastsurfer.sh \
  --threads 8 \
  --n-jobs 1 \
  --force

Example (batch folder)
----------------------
python deface_batch.py \
  --in-root ~/sandbox/Jake_Data/tmp/defacing \
  --pattern "**/*_T1w.nii.gz" \
  --out-root ~/sandbox/Jake_Data/tmp/defacing_out \
  --subjects-dir ~/sandbox/Jake_Data/tmp/fastsurfer_subjects \
  --fastsurfer-sh /home/boyan/sandbox/utils/FastSurfer/run_fastsurfer.sh \
  --threads 8 \
  --n-jobs 1 \
  --force
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to


@dataclass(frozen=True)
class Job:
    t1_path: Path
    sid: str
    out_path: Path


def run_cmd(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    """
    Run command and stream output. Raise on non-zero return.
    """
    print(f"[CMD] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")


def infer_sid_from_t1(t1_path: Path) -> str:
    """
    Infer subject ID from filename using BIDS-ish pattern 'sub-XXXX'.
    Fallback to filename stem.
    """
    m = re.search(r"(sub-[A-Za-z0-9]+)", t1_path.name)
    if m:
        return m.group(1)

    name = t1_path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return t1_path.stem


def detect_fastsurfer_t1_flag(fastsurfer_sh: Path, env: dict[str, str] | None) -> str:
    """
    FastSurfer CLI differs between versions:
      - some accept --t1
      - others accept --i
    Detect via --help output.
    """
    out = subprocess.run(
        [str(fastsurfer_sh), "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        check=False,
    ).stdout
    return "--t1" if "--t1" in out else "--i"


def find_existing_mask(subjects_dir: Path, sid: str) -> Path | None:
    """
    Return an existing brain mask file if present, else None.

    FastSurfer commonly produces: mri/mask.mgz
    FreeSurfer commonly produces: mri/brainmask.mgz
    """
    subj_dir = subjects_dir / sid / "mri"
    mask_fs = subj_dir / "mask.mgz"
    mask_fs_alt = subj_dir / "brainmask.mgz"

    if mask_fs.exists():
        return mask_fs
    if mask_fs_alt.exists():
        return mask_fs_alt
    return None


def ensure_fastsurfer_mask(
    *,
    fastsurfer_sh: Path,
    t1_path: Path,
    sid: str,
    subjects_dir: Path,
    threads: int,
    tmpdir: Path | None,
    rerun_fastsurfer: bool,
) -> Path:
    """
    Ensure a brain mask exists for sid in subjects_dir.
    Runs FastSurfer only if no mask exists or if rerun_fastsurfer=True.
    """
    subjects_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if tmpdir is not None:
        env["TMPDIR"] = str(tmpdir)
        env["TMP"] = str(tmpdir)
        env["TEMP"] = str(tmpdir)

    existing = find_existing_mask(subjects_dir, sid)
    if existing is not None and not rerun_fastsurfer:
        print(f"[INFO] Found existing mask: {existing} (skipping FastSurfer)", flush=True)
        return existing

    # If rerun requested, we still keep old files; FastSurfer will overwrite.
    if existing is not None and rerun_fastsurfer:
        print(f"[INFO] Rerun requested: will re-run FastSurfer for {sid} (existing {existing})", flush=True)

    t1_flag = detect_fastsurfer_t1_flag(fastsurfer_sh, env)

    cmd = [
        str(fastsurfer_sh),
        t1_flag, str(t1_path),
        "--sid", sid,
        "--sd", str(subjects_dir),
        "--seg_only",
        "--threads", str(int(threads)),
    ]
    run_cmd(cmd, env=env)

    mask = find_existing_mask(subjects_dir, sid)
    if mask is None:
        raise FileNotFoundError(
            f"FastSurfer finished but no mask found for {sid}. "
            f"Expected one of: {subjects_dir}/{sid}/mri/mask.mgz or brainmask.mgz"
        )
    return mask


def make_brain_only(
    *,
    t1_path: Path,
    mask_path: Path,
    out_path: Path,
    threshold: float,
) -> None:
    """
    Apply mask to T1 and save brain-only NIfTI. Resample mask to T1 grid if needed.
    """
    t1_img = nib.load(str(t1_path))
    m_img = nib.load(str(mask_path))  # .mgz supported by nibabel

    # Resample mask to T1 grid if shapes/affines differ (FastSurfer uses conformed spaces).
    if m_img.shape != t1_img.shape or not np.allclose(m_img.affine, t1_img.affine, atol=1e-3):
        m_rs = resample_from_to(m_img, t1_img, order=0)  # nearest-neighbor
    else:
        m_rs = m_img

    m_data = np.asanyarray(m_rs.dataobj)
    mask = m_data > float(threshold)

    t1_data = np.asanyarray(t1_img.dataobj)
    out_data = np.where(mask, t1_data, 0).astype(t1_data.dtype, copy=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(out_data, affine=t1_img.affine, header=t1_img.header)
    out_img.set_data_dtype(t1_img.get_data_dtype())
    nib.save(out_img, str(out_path))


def output_name_for(t1_path: Path) -> str:
    """
    Create an explicit brain-only output name, preserving BIDS-ish suffixes.
    """
    name = t1_path.name
    if name.endswith("_T1w.nii.gz"):
        return name.replace("_T1w.nii.gz", "_desc-brainonly_T1w.nii.gz")
    if name.endswith(".nii.gz"):
        return name.replace(".nii.gz", "_desc-brainonly.nii.gz")
    if name.endswith(".nii"):
        return name.replace(".nii", "_desc-brainonly.nii")
    return name + "_desc-brainonly.nii.gz"


def build_jobs(
    *,
    t1_single: Path | None,
    in_root: Path | None,
    pattern: str,
    out_root: Path,
    sid_override: str | None,
) -> list[Job]:
    if t1_single is not None:
        sid = sid_override or infer_sid_from_t1(t1_single)
        out_path = out_root / output_name_for(t1_single)
        return [Job(t1_path=t1_single, sid=sid, out_path=out_path)]

    assert in_root is not None
    t1_files = sorted(in_root.glob(pattern))
    if not t1_files:
        raise SystemExit(f"No files found under {in_root} matching pattern: {pattern}")

    jobs: list[Job] = []
    for t1 in t1_files:
        sid = sid_override or infer_sid_from_t1(t1)
        rel = t1.relative_to(in_root)
        out_path = (out_root / rel).with_name(output_name_for(Path(rel.name)))
        jobs.append(Job(t1_path=t1, sid=sid, out_path=out_path))
    return jobs


def process_one(
    job: Job,
    *,
    fastsurfer_sh: Path,
    subjects_dir: Path,
    threads: int,
    threshold: float,
    force: bool,
    tmpdir: Path | None,
    rerun_fastsurfer: bool,
) -> None:
    print(f"[INFO] Subject {job.sid}: {job.t1_path}", flush=True)

    # Only overwrite final output if requested
    if job.out_path.exists() and not force:
        print(f"[SKIP] Output exists (use --force to overwrite): {job.out_path}", flush=True)
        return

    mask = ensure_fastsurfer_mask(
        fastsurfer_sh=fastsurfer_sh,
        t1_path=job.t1_path,
        sid=job.sid,
        subjects_dir=subjects_dir,
        threads=threads,
        tmpdir=tmpdir,
        rerun_fastsurfer=rerun_fastsurfer,
    )

    make_brain_only(
        t1_path=job.t1_path,
        mask_path=mask,
        out_path=job.out_path,
        threshold=threshold,
    )

    print(f"[OK] Wrote: {job.out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--t1", type=Path, help="Single T1 NIfTI (.nii or .nii.gz)")
    g.add_argument("--in-root", type=Path, help="Root folder to search for T1 files")

    ap.add_argument("--pattern", type=str, default="**/*_T1w.nii.gz", help="Glob under --in-root")
    ap.add_argument("--out-root", type=Path, required=True, help="Output root folder")
    ap.add_argument("--subjects-dir", type=Path, required=True, help="FastSurfer subjects dir (output)")
    ap.add_argument("--sid", type=str, default=None, help="Override subject id (useful for single-file mode)")

    ap.add_argument(
        "--fastsurfer-sh",
        type=Path,
        required=True,
        help="Path to FastSurfer launcher (e.g., .../FastSurfer/run_fastsurfer.sh)",
    )
    ap.add_argument("--threads", type=int, default=8, help="Threads for FastSurfer per subject")
    ap.add_argument("--n-jobs", type=int, default=1, help="Parallel subjects (keep small; FastSurfer is heavy)")
    ap.add_argument("--threshold", type=float, default=0.5, help="Mask threshold")
    ap.add_argument("--force", action="store_true", help="Overwrite final brain-only output if it exists")
    ap.add_argument(
        "--rerun-fastsurfer",
        action="store_true",
        help="Force rerun FastSurfer even if an existing mask is found",
    )
    ap.add_argument("--tmpdir", type=Path, default=Path("/tmp"), help="Temp dir to use (default /tmp)")

    args = ap.parse_args()

    if not args.fastsurfer_sh.exists():
        raise SystemExit(f"--fastsurfer-sh does not exist: {args.fastsurfer_sh}")

    if args.in_root is not None and not args.in_root.exists():
        raise SystemExit(f"--in-root does not exist: {args.in_root}")
    if args.t1 is not None and not args.t1.exists():
        raise SystemExit(f"--t1 does not exist: {args.t1}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.subjects_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(
        t1_single=args.t1,
        in_root=args.in_root,
        pattern=args.pattern,
        out_root=args.out_root,
        sid_override=args.sid,
    )

    print(
        f"[INFO] Jobs: {len(jobs)} | n_jobs={args.n_jobs} | threads/subject={args.threads} | tmpdir={args.tmpdir}",
        flush=True,
    )

    tmpdir = args.tmpdir if args.tmpdir else None

    # Avoid oversubscription unless you know you have the cores
    if args.n_jobs <= 1 or len(jobs) == 1:
        for j in jobs:
            process_one(
                j,
                fastsurfer_sh=args.fastsurfer_sh,
                subjects_dir=args.subjects_dir,
                threads=args.threads,
                threshold=args.threshold,
                force=args.force,
                tmpdir=tmpdir,
                rerun_fastsurfer=args.rerun_fastsurfer,
            )
        return

    with ThreadPoolExecutor(max_workers=args.n_jobs) as ex:
        futures = [
            ex.submit(
                process_one,
                j,
                fastsurfer_sh=args.fastsurfer_sh,
                subjects_dir=args.subjects_dir,
                threads=args.threads,
                threshold=args.threshold,
                force=args.force,
                tmpdir=tmpdir,
                rerun_fastsurfer=args.rerun_fastsurfer,
            )
            for j in jobs
        ]
        done = 0
        for fut in as_completed(futures):
            fut.result()
            done += 1
            print(f"[INFO] Completed {done}/{len(futures)}", flush=True)


if __name__ == "__main__":
    main()
