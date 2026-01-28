#!/usr/bin/env python3
"""
Run FastSurfer (seg_only) to generate brainmask.mgz, then create a brain-only NIfTI.

What you get:
  - A brain-only image where voxels outside the brain mask are set to 0.
  - This removes facial anatomy because everything outside the brain is removed.

Requirements:
  - FastSurfer installed and fastsurfer.sh available on PATH, or provide --fastsurfer-sh
  - Python packages: nibabel, numpy

Install Python deps:
  pip install nibabel numpy

Example (single file):
  python fastsurfer_brainonly_batch.py \
    --t1 /data/sub-01_T1w.nii.gz \
    --out-root /data/anon \
    --subjects-dir /data/fastsurfer_subjects \
    --sid sub-01 \
    --threads 8 \
    --n-jobs 1 \
    --force

Example (batch folder):
  python fastsurfer_brainonly_batch.py \
    --in-root /data/bids \
    --pattern "**/*_T1w.nii.gz" \
    --out-root /data/anon \
    --subjects-dir /data/fastsurfer_subjects \
    --threads 8 \
    --n-jobs 2 \
    --force
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
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
    brainmask_path: Path


def run_cmd(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    """
    Run command and stream output. If it fails, raise with command and exit code.
    """
    print(f"[CMD] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")


def infer_sid_from_t1(t1_path: Path) -> str:
    """
    Infer a subject id from filename using BIDS-ish convention:
      sub-XXXX[_...]_T1w.nii.gz -> sub-XXXX
    If not found, fallback to stem.
    """
    m = re.search(r"(sub-[a-zA-Z0-9]+)", t1_path.name)
    if m:
        return m.group(1)
    # fallback: use filename stem without extensions
    name = t1_path.name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return name


def ensure_fast_surfer_mask(
    *,
    fastsurfer_sh: Path,
    t1_path: Path,
    sid: str,
    subjects_dir: Path,
    threads: int,
    tmpdir: Path | None,
) -> Path:
    """
    Run FastSurfer seg_only if brainmask does not exist. Return brainmask path.
    """
    subj_dir = subjects_dir / sid
    brainmask = subj_dir / "mri" / "brainmask.mgz"
    if brainmask.exists():
        return brainmask

    subjects_dir.mkdir(parents=True, exist_ok=True)

    # Encourage local temp if provided (helps on slow / network FS).
    env = os.environ.copy()
    if tmpdir is not None:
        env["TMPDIR"] = str(tmpdir)
        env["TMP"] = str(tmpdir)
        env["TEMP"] = str(tmpdir)

    cmd = [
        str(fastsurfer_sh),
        "--t1",
        str(t1_path),
        "--sid",
        sid,
        "--sd",
        str(subjects_dir),
        "--seg_only",
        "--threads",
        str(int(threads)),
    ]
    run_cmd(cmd, env=env)

    if not brainmask.exists():
        raise FileNotFoundError(
            f"FastSurfer finished but brainmask.mgz not found at: {brainmask}"
        )
    return brainmask


def make_brain_only(
    *,
    t1_path: Path,
    brainmask_mgz: Path,
    out_path: Path,
    threshold: float,
) -> None:
    """
    Apply brainmask to T1 and save brain-only NIfTI. Resample mask if needed.
    """
    t1_img = nib.load(str(t1_path))
    bm_img = nib.load(str(brainmask_mgz))  # mgz supported by nibabel

    # Resample brainmask to T1 grid if shapes/affines differ (common due to conformed space).
    if bm_img.shape != t1_img.shape or not np.allclose(bm_img.affine, t1_img.affine, atol=1e-3):
        bm_rs = resample_from_to(bm_img, t1_img, order=0)  # nearest neighbor
    else:
        bm_rs = bm_img

    bm_data = np.asanyarray(bm_rs.dataobj)
    mask = bm_data > threshold

    t1_data = np.asanyarray(t1_img.dataobj)
    out_data = np.where(mask, t1_data, 0).astype(t1_data.dtype, copy=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(out_data, affine=t1_img.affine, header=t1_img.header)
    out_img.set_data_dtype(t1_img.get_data_dtype())
    nib.save(out_img, str(out_path))


def build_jobs(
    *,
    t1_single: Path | None,
    in_root: Path | None,
    pattern: str,
    out_root: Path,
    subjects_dir: Path,
    sid_override: str | None,
) -> list[Job]:
    if t1_single is not None:
        sid = sid_override or infer_sid_from_t1(t1_single)
        rel = Path(t1_single.name)  # single-file mode: keep just name
        out_path = out_root / rel.name.replace("_T1w.nii.gz", "_desc-brainonly_T1w.nii.gz")
        if out_path.name == rel.name:  # if pattern doesn't match, just append
            out_path = out_root / (rel.stem + "_desc-brainonly.nii.gz")
        bm = subjects_dir / sid / "mri" / "brainmask.mgz"
        return [Job(t1_path=t1_single, sid=sid, out_path=out_path, brainmask_path=bm)]

    assert in_root is not None
    t1_files = sorted(in_root.glob(pattern))
    if not t1_files:
        raise SystemExit(f"No files found under {in_root} matching pattern: {pattern}")

    jobs: list[Job] = []
    for t1 in t1_files:
        sid = sid_override or infer_sid_from_t1(t1)
        rel = t1.relative_to(in_root)
        # Keep folder structure under out_root
        out_path = out_root / rel
        # Name output as desc-brainonly while preserving suffix
        name = out_path.name
        if name.endswith("_T1w.nii.gz"):
            name = name.replace("_T1w.nii.gz", "_desc-brainonly_T1w.nii.gz")
        elif name.endswith(".nii.gz"):
            name = name.replace(".nii.gz", "_desc-brainonly.nii.gz")
        elif name.endswith(".nii"):
            name = name.replace(".nii", "_desc-brainonly.nii")
        out_path = out_path.with_name(name)

        bm = subjects_dir / sid / "mri" / "brainmask.mgz"
        jobs.append(Job(t1_path=t1, sid=sid, out_path=out_path, brainmask_path=bm))

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
) -> None:
    if job.out_path.exists() and not force:
        print(f"[SKIP] exists: {job.out_path}", flush=True)
        return

    print(f"[INFO] Subject {job.sid}: {job.t1_path}", flush=True)

    bm = ensure_fast_surfer_mask(
        fastsurfer_sh=fastsurfer_sh,
        t1_path=job.t1_path,
        sid=job.sid,
        subjects_dir=subjects_dir,
        threads=threads,
        tmpdir=tmpdir,
    )

    make_brain_only(
        t1_path=job.t1_path,
        brainmask_mgz=bm,
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
    ap.add_argument("--sid", type=str, default=None, help="Override subject id (single-subject or uniform)")

    ap.add_argument("--fastsurfer-sh", type=Path, default=None, help="Path to fastsurfer.sh (optional)")
    ap.add_argument("--threads", type=int, default=8, help="Threads for FastSurfer per subject")
    ap.add_argument("--n-jobs", type=int, default=1, help="Parallel subjects")
    ap.add_argument("--threshold", type=float, default=0.5, help="Mask threshold")
    ap.add_argument("--force", action="store_true", help="Overwrite outputs")
    ap.add_argument("--tmpdir", type=Path, default=Path("/tmp"), help="Temp dir to use (default /tmp)")

    args = ap.parse_args()

    fastsurfer_sh = args.fastsurfer_sh
    if fastsurfer_sh is None:
        found = shutil.which("fastsurfer.sh")
        if found is None:
            raise SystemExit(
                "fastsurfer.sh not found on PATH. Provide --fastsurfer-sh /path/to/fastsurfer.sh"
            )
        fastsurfer_sh = Path(found)

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
        subjects_dir=args.subjects_dir,
        sid_override=args.sid,
    )

    # IMPORTANT: FastSurfer is heavy; don't oversubscribe.
    # n_jobs should be modest unless you have many cores.
    print(f"[INFO] Jobs: {len(jobs)} | n_jobs={args.n_jobs} | threads/subject={args.threads}", flush=True)

    tmpdir = args.tmpdir if args.tmpdir else None

    if args.n_jobs <= 1 or len(jobs) == 1:
        for j in jobs:
            process_one(
                j,
                fastsurfer_sh=fastsurfer_sh,
                subjects_dir=args.subjects_dir,
                threads=args.threads,
                threshold=args.threshold,
                force=args.force,
                tmpdir=tmpdir,
            )
        return

    # Parallel execution across subjects
    with ThreadPoolExecutor(max_workers=args.n_jobs) as ex:
        futures = [
            ex.submit(
                process_one,
                j,
                fastsurfer_sh=fastsurfer_sh,
                subjects_dir=args.subjects_dir,
                threads=args.threads,
                threshold=args.threshold,
                force=args.force,
                tmpdir=tmpdir,
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
