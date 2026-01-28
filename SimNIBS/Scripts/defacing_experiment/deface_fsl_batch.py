#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(f"[CMD] {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed (exit {p.returncode}): {' '.join(cmd)}")


def deface_one(in_path: Path, out_path: Path, *, force: bool, tmpdir: Path | None) -> None:
    if out_path.exists() and not force:
        print(f"[SKIP] exists: {out_path}", flush=True)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if tmpdir is not None:
        env["TMPDIR"] = str(tmpdir)
        env["TMP"] = str(tmpdir)
        env["TEMP"] = str(tmpdir)

    # fsl_deface <input> <output>
    cmd = ["fsl_deface", str(in_path), str(out_path)]
    run_cmd(cmd, env=env)

    print(f"[OK] {in_path.name} -> {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Deface T1 NIfTI files using FSL fsl_deface (face only).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--t1", type=Path, help="Single input T1 (.nii or .nii.gz)")
    g.add_argument("--in-root", type=Path, help="Input root directory")

    ap.add_argument("--pattern", type=str, default="**/*_T1w.nii.gz", help="Glob pattern under --in-root")
    ap.add_argument("--out-root", type=Path, required=True, help="Output root directory")
    ap.add_argument("--n-jobs", type=int, default=2, help="Parallel jobs (2 is safe on a laptop)")
    ap.add_argument("--force", action="store_true", help="Overwrite outputs")
    ap.add_argument("--tmpdir", type=Path, default=Path("/tmp"), help="Temp directory (default /tmp)")
    args = ap.parse_args()

    # Quick sanity check that fsl_deface is available
    if subprocess.run(["bash", "-lc", "command -v fsl_deface >/dev/null 2>&1"]).returncode != 0:
        raise SystemExit(
            "fsl_deface not found on PATH. Source FSL first (e.g., source $FSLDIR/etc/fslconf/fsl.sh) "
            "or ensure FSL is installed and on PATH."
        )

    args.out_root.mkdir(parents=True, exist_ok=True)

    if args.t1 is not None:
        inputs = [args.t1]
        rels = [Path(args.t1.name)]
    else:
        if not args.in_root.exists():
            raise SystemExit(f"--in-root does not exist: {args.in_root}")
        inputs = sorted(args.in_root.glob(args.pattern))
        if not inputs:
            raise SystemExit(f"No files found under {args.in_root} matching {args.pattern}")
        rels = [p.relative_to(args.in_root) for p in inputs]

    # Output naming: preserve path, add desc-deface
    out_paths: list[Path] = []
    for rel in rels:
        name = rel.name
        if name.endswith("_T1w.nii.gz"):
            name = name.replace("_T1w.nii.gz", "_desc-deface_T1w.nii.gz")
        elif name.endswith(".nii.gz"):
            name = name.replace(".nii.gz", "_desc-deface.nii.gz")
        elif name.endswith(".nii"):
            name = name.replace(".nii", "_desc-deface.nii")
        out_paths.append((args.out_root / rel).with_name(name))

    print(f"[INFO] Files: {len(inputs)} | n_jobs={args.n_jobs}", flush=True)

    tmpdir = args.tmpdir if args.tmpdir else None

    if args.n_jobs <= 1 or len(inputs) == 1:
        for ip, op in zip(inputs, out_paths):
            deface_one(ip, op, force=args.force, tmpdir=tmpdir)
        return

    with ThreadPoolExecutor(max_workers=args.n_jobs) as ex:
        futs = [ex.submit(deface_one, ip, op, force=args.force, tmpdir=tmpdir) for ip, op in zip(inputs, out_paths)]
        for fut in as_completed(futs):
            fut.result()


if __name__ == "__main__":
    main()
