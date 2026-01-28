#!/usr/bin/env python3
"""
Batch deface T1w images using PyDeface.

Requirements:
  - pydeface installed and on PATH (pip install pydeface)
  - FSL installed (PyDeface uses FSL FLIRT internally in typical setups) :contentReference[oaicite:5]{index=5}

Example:
  python pydeface_batch.py \
    --in-root /data/bids/subs \
    --pattern "**/*_T1w.nii.gz" \
    --out-root /data/bids_defaced \
    --n-jobs 4 \
    --force
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  {' '.join(cmd)}\n\n"
            f"STDOUT:\n{proc.stdout}\n\n"
            f"STDERR:\n{proc.stderr}\n"
        )


def deface_one(
    in_path: Path,
    out_path: Path,
    *,
    force: bool,
    applyto: list[Path] | None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["pydeface", str(in_path), "--outfile", str(out_path)]
    if force:
        cmd.append("--force")
    if applyto:
        # Apply the created face mask to other images (paths must exist).
        # PyDeface supports --applyto multiple args. :contentReference[oaicite:6]{index=6}
        cmd += ["--applyto"] + [str(p) for p in applyto]

    run_cmd(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", type=Path, required=True, help="Input root folder")
    ap.add_argument("--pattern", type=str, default="**/*_T1w.nii.gz", help="Glob pattern under in-root")
    ap.add_argument("--out-root", type=Path, required=True, help="Output root folder")
    ap.add_argument("--n-jobs", type=int, default=4, help="Parallel workers (threads)")
    ap.add_argument("--force", action="store_true", help="Overwrite outputs if they exist")
    ap.add_argument(
        "--applyto-suffixes",
        nargs="*",
        default=[],
        help=(
            "Optional: for each T1w, also deface sibling images by suffix replacement. "
            "Example: --applyto-suffixes _T2w.nii.gz _PDw.nii.gz "
            "(will look for matching files next to the T1w)."
        ),
    )
    args = ap.parse_args()

    if shutil.which("pydeface") is None:
        raise SystemExit("pydeface not found on PATH. Install with: pip install pydeface")

    # Heuristic warning: PyDeface commonly relies on FSL (via FLIRT). :contentReference[oaicite:7]{index=7}
    if os.environ.get("FSLDIR") is None:
        print("[WARN] FSLDIR is not set. If pydeface fails with FLIRT-related errors, install/source FSL.")

    in_files = sorted(args.in_root.glob(args.pattern))
    if not in_files:
        raise SystemExit(f"No files found under {args.in_root} matching pattern: {args.pattern}")

    futures = []
    with ThreadPoolExecutor(max_workers=args.n_jobs) as ex:
        for in_path in in_files:
            rel = in_path.relative_to(args.in_root)
            out_path = args.out_root / rel

            applyto = []
            for suf in args.applyto_suffixes:
                # Replace T1w suffix with desired suffix in same directory
                # (only if the T1w filename ends with _T1w.nii.gz).
                name = in_path.name
                if name.endswith("_T1w.nii.gz"):
                    sib = in_path.with_name(name.replace("_T1w.nii.gz", suf))
                    if sib.exists():
                        applyto.append(sib)

            futures.append(ex.submit(deface_one, in_path, out_path, force=args.force, applyto=applyto or None))

        done = 0
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 10 == 0 or done == len(futures):
                print(f"[INFO] Defaced {done}/{len(futures)}")

    print("[INFO] Completed.")


if __name__ == "__main__":
    main()

