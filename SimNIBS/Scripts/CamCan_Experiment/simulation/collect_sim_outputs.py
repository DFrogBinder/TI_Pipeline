#!/usr/bin/env python3
"""
Collect (scrape) only selected simulation outputs from a large directory tree
into a slim export directory, using rsync include/exclude filters.

Typical use (on HPC):
  python collect_sim_outputs.py \
    --src /mnt/parscratch/users/cop23bi/full-ti-dataset \
    --dst /mnt/parscratch/users/cop23bi/export_sim_outputs \
    --compress zstd

Notes:
- This script shells out to rsync (must be available).
- Compression uses either zstd or gzip (optional).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional


# -----------------------------
# EDIT THESE LISTS
# -----------------------------

# Directories that should always be traversable / included (relative to SRC root).
# For SimNIBS, output artifacts commonly live under SimNIBS/Output.
INCLUDE_DIRS: List[str] = [
    "SimNIBS/Output",
]

INCLUDE_SPECIAL: List[str] = [
    "**/m2m_*/[!/]*.msh",
]

# # File patterns you want to keep anywhere under the included trees.
# INCLUDE_FILES: List[str] = [
#     "*.msh",
#     "*.nii",
#     "*.nii.gz",
#     "*.csv",
#     "*.tsv",
#     "*.json",
#     "*.mat",
#     "*.pkl",
#     "*.npz",
#     "*.txt",
#     "*.log",
#     "*.out",
#     "*.err",
# ]

# File patterns you want to keep anywhere under the included trees.
INCLUDE_FILES: List[str] = [
    "*.msh",
    "*.nii",
    "*.nii.gz"
]

# Patterns to exclude (often large or regenerable).
# These use rsync filter wildcard syntax.
EXCLUDE_DIRS: List[str] = [
    "**/m2m_*",
    "**/Volume_Base/**",
    "**/Volume_Labels/**",
    "**/Volume_Maks/**",
    "**/tmp/**",
    "**/.git/**",
    "**/__pycache__/**",
    "**/coregistration/**",
    "**/intermediate/**",
    "**/cache/**",
]


# =============================================================================
# Helpers
# =============================================================================

def which_or_exit(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        print(f"[ERROR] Required executable not found in PATH: {binary}", file=sys.stderr)
        sys.exit(2)
    return path


def format_bytes(num_bytes: int) -> str:
    gb = 1024 ** 3
    mb = 1024 ** 2
    if num_bytes >= gb:
        return f"{num_bytes / gb:.2f} GB"
    if num_bytes >= mb:
        return f"{num_bytes / mb:.2f} MB"
    return f"{num_bytes} bytes"


def parse_rsync_total_size_bytes(rsync_output: str) -> int | None:
    """
    Parse rsync --info=stats2 output and extract a useful estimate of transfer size.

    We prefer:
      - 'Total transferred file size: X bytes'
    Fallback:
      - 'Total file size: X bytes' (less precise for "would transfer", but informative)
    """
    patterns = [
        r"Total transferred file size:\s+([\d,]+)\s+bytes",
        r"Total file size:\s+([\d,]+)\s+bytes",
    ]
    for pat in patterns:
        m = re.search(pat, rsync_output)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


def get_directory_size_bytes(path: Path) -> int:
    """
    Python-native directory size calculation (works even if `du` is unavailable).
    """
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                # Ignore unreadable files (rare on HPC scratch but possible)
                pass
    return total


def shlex_quote(s: str) -> str:
    import shlex
    return shlex.quote(s)


# =============================================================================
# rsync filter generation and execution
# =============================================================================

def build_rsync_filter_file(path: Path) -> None:
    """
    Create an rsync filter file that:
      - allows directory traversal
      - includes INCLUDE_DIRS trees
      - includes INCLUDE_SPECIAL patterns (e.g., the single .msh in m2m_*)
      - includes INCLUDE_FILES patterns
      - excludes EXCLUDE_DIRS
      - excludes everything else
    """
    lines: List[str] = []
    lines.append("# Auto-generated rsync filter rules")
    lines.append("")
    lines.append("# Always allow directory traversal")
    lines.append("+ */")
    lines.append("")

    lines.append("# Include key output directories (full subtree)")
    for d in INCLUDE_DIRS:
        d = d.strip("/")
        lines.append(f"+ {d}/***")
    lines.append("")

    lines.append("# Special-case includes (must come before excludes)")
    for pat in INCLUDE_SPECIAL:
        lines.append(f"+ {pat}")
    lines.append("")

    lines.append("# Include relevant file types (anywhere)")
    for pat in INCLUDE_FILES:
        lines.append(f"+ **/{pat}")
    lines.append("")

    lines.append("# Exclude known large/unneeded directories")
    for xd in EXCLUDE_DIRS:
        lines.append(f"- {xd}")
    lines.append("")

    lines.append("# Exclude everything else")
    lines.append("- **")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def run_rsync(
    src_root: Path,
    dst_root: Path,
    filter_file: Path,
    dry_run: bool = False,
    verbose: bool = True,
) -> str:
    """
    Run rsync with a filter file to copy only allowed files into dst_root.
    Returns rsync combined stdout/stderr text so we can parse stats in dry-run mode.
    """
    which_or_exit("rsync")

    dst_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rsync",
        "-a",
        "--prune-empty-dirs",
        f"--filter=merge {str(filter_file)}",
        "--info=stats2",
    ]

    if verbose:
        cmd.append("-v")
    if dry_run:
        cmd.append("--dry-run")

    # Trailing slashes ensure rsync copies CONTENTS of src_root into dst_root
    cmd.extend([str(src_root) + "/", str(dst_root) + "/"])

    print("[INFO] Running rsync command:")
    print("       " + " ".join(cmd))
    print()

    proc = subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Show rsync output (useful for debugging and confirming matches)
    print(proc.stdout)
    return proc.stdout


# =============================================================================
# Compression (optional)
# =============================================================================

def compress_export(
    export_dir: Path,
    out_path: Optional[Path],
    method: str,
) -> Path:
    """
    Compress export_dir into a tar archive.
    method: 'zstd' or 'gzip'
    """
    export_dir = export_dir.resolve()
    parent = export_dir.parent
    name = export_dir.name

    if out_path is None:
        if method == "zstd":
            out_path = parent / f"{name}.tar.zst"
        elif method == "gzip":
            out_path = parent / f"{name}.tar.gz"
        else:
            raise ValueError("Unsupported compression method")

    out_path = out_path.resolve()

    if method == "zstd":
        which_or_exit("zstd")
        which_or_exit("tar")

        cmd = (
            f"tar -C {shlex_quote(str(parent))} -cf - {shlex_quote(name)}"
            f" | zstd -T0 -19 -o {shlex_quote(str(out_path))}"
        )
        print("[INFO] Compressing with zstd:")
        print(f"       {cmd}")
        subprocess.run(cmd, shell=True, check=True)

    elif method == "gzip":
        which_or_exit("tar")

        cmd = [
            "tar",
            "-C",
            str(parent),
            "-czf",
            str(out_path),
            name,
        ]
        print("[INFO] Compressing with gzip:")
        print("       " + " ".join(cmd))
        subprocess.run(cmd, check=True)

    else:
        raise ValueError("Compression method must be 'zstd' or 'gzip'.")

    print(f"[INFO] Created archive: {out_path}")
    return out_path


# =============================================================================
# CLI / Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Collect only simulation outputs into a slim export folder (rsync filters)."
    )
    p.add_argument("--src", required=True, help="Source root directory (huge dataset root).")
    p.add_argument("--dst", required=True, help="Destination export directory (slim output).")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied, without copying.",
    )
    p.add_argument(
        "--no-verbose",
        action="store_true",
        help="Less rsync output.",
    )
    p.add_argument(
        "--write-filter-to",
        default=None,
        help="Optional path to write the filter file for inspection/editing.",
    )
    p.add_argument(
        "--compress",
        choices=["none", "zstd", "gzip"],
        default="none",
        help="Optionally compress the export folder after copying.",
    )
    p.add_argument(
        "--archive-path",
        default=None,
        help="Optional explicit output path for the archive (tar.zst or tar.gz).",
    )

    args = p.parse_args()

    src_root = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()

    if not src_root.exists():
        print(f"[ERROR] Source does not exist: {src_root}", file=sys.stderr)
        sys.exit(2)

    # Create filter file
    if args.write_filter_to:
        filter_path = Path(args.write_filter_to).expanduser().resolve()
        filter_path.parent.mkdir(parents=True, exist_ok=True)
        build_rsync_filter_file(filter_path)
        print(f"[INFO] Wrote rsync filter file to: {filter_path}")
    else:
        tmp = tempfile.NamedTemporaryFile(prefix="rsync_filter_", suffix=".rules", delete=False)
        filter_path = Path(tmp.name)
        tmp.close()
        build_rsync_filter_file(filter_path)

    print("[INFO] rsync filter rules:")
    print(filter_path.read_text(encoding="utf-8"))
    print()

    try:
        rsync_text = run_rsync(
            src_root=src_root,
            dst_root=dst_root,
            filter_file=filter_path,
            dry_run=args.dry_run,
            verbose=not args.no_verbose,
        )
    finally:
        # Only delete if we created a temp file
        if not args.write_filter_to:
            try:
                filter_path.unlink(missing_ok=True)
            except Exception:
                pass

    if args.dry_run:
        est = parse_rsync_total_size_bytes(rsync_text)
        print("[INFO] Dry-run complete. No files were copied.")
        if est is not None:
            print(f"[INFO] Estimated transfer size (dry-run): {format_bytes(est)}  ({est:,} bytes)")
        else:
            print("[WARN] Could not parse rsync stats for estimated transfer size.")
        return

    # Real run: compute actual export directory size (Python-native)
    size_bytes = get_directory_size_bytes(dst_root)
    print("[INFO] Export size:")
    print(f"       {format_bytes(size_bytes)}  ({size_bytes:,} bytes)")
    print()

    if args.compress != "none":
        archive_path = Path(args.archive_path).expanduser().resolve() if args.archive_path else None
        created = compress_export(dst_root, archive_path, args.compress)

        if created.exists():
            archive_size = created.stat().st_size
            print("[INFO] Archive size:")
            print(f"       {format_bytes(archive_size)}  ({archive_size:,} bytes)")


if __name__ == "__main__":
    main()