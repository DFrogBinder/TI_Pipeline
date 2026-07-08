#!/usr/bin/env python3
"""Stage defacing repeat batches from already-created intact/defaced images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

try:
    from defacing_experiment.prepare_defacing_repeat_batch import DEFAULT_REPEATS, DEFAULT_TARGETS, stage_repeat_batches
except ImportError:  # pragma: no cover - supports direct execution as a file on HPC.
    from prepare_defacing_repeat_batch import DEFAULT_REPEATS, DEFAULT_TARGETS, stage_repeat_batches


def _existing_path(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"{label} does not exist: {path}")
    return path


def _count_manifest_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage the intact/defaced repeat-batch experiment when defaced T1/T2 "
            "images already exist. This does not run pydeface."
        )
    )
    parser.add_argument("--subject", required=True, help="Subject ID, e.g. sub-IXI025")
    parser.add_argument("--intact-t1", type=Path, required=True, help="Canonical intact T1w NIfTI")
    parser.add_argument("--intact-t2", type=Path, required=True, help="Canonical intact T2w NIfTI")
    parser.add_argument("--defaced-t1", type=Path, required=True, help="Defaced T1w NIfTI")
    parser.add_argument("--defaced-t2", type=Path, required=True, help="Defaced T2w NIfTI")
    parser.add_argument("--out-root", type=Path, required=True, help="Output root for staged experiment arms")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help=f"Repeats per arm. Default: {DEFAULT_REPEATS}")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(DEFAULT_TARGETS),
        help="Target presets to stage. Default: left-hippocampus left-m1",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite staged repeat input files")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.repeats < 1:
        raise SystemExit(f"--repeats must be positive; got {args.repeats}")

    intact_t1 = _existing_path(args.intact_t1, "--intact-t1")
    intact_t2 = _existing_path(args.intact_t2, "--intact-t2")
    defaced_t1 = _existing_path(args.defaced_t1, "--defaced-t1")
    defaced_t2 = _existing_path(args.defaced_t2, "--defaced-t2")
    out_root = args.out_root.expanduser().resolve()

    parent_roots = stage_repeat_batches(
        out_root=out_root,
        subject=args.subject,
        intact_t1=intact_t1,
        intact_t2=intact_t2,
        defaced_t1=defaced_t1,
        defaced_t2=defaced_t2,
        repeats=args.repeats,
        targets=args.targets,
        force=args.force,
    )

    print("[INFO] Staged existing-defaced repeat batch", flush=True)
    print(f"  Subject: {args.subject}", flush=True)
    print(f"  Out root: {out_root}", flush=True)
    print(f"  Repeats per arm: {args.repeats}", flush=True)
    print("[INFO] Staged parent roots:", flush=True)
    for parent_root in parent_roots:
        manifest = parent_root / "slurm" / "manifest.tsv"
        print(f"  {parent_root} ({_count_manifest_rows(manifest)} manifest rows)", flush=True)
    print(f"[INFO] Experiment manifest: {out_root / 'experiment_manifest.tsv'}", flush=True)


if __name__ == "__main__":
    main()
