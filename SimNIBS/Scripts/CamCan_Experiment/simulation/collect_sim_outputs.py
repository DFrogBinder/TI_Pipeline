#!/usr/bin/env python3
"""
collect_sim_outputs.py

Collect only selected simulation outputs from a large directory tree
into a slim export directory, using rsync include/exclude filters.

Features:
- Precise rsync filtering
- Dry-run with MB/GB transfer size estimate
- Keeps ONLY the top-level .msh file inside m2m_* folders
- Allows exclusion of specific filename patterns
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional


# =============================================================================
# ========================= USER-EDITABLE FILTERS ==============================
# =============================================================================

# Output directories that must be traversable
INCLUDE_DIRS: List[str] = [
    "SimNIBS/Output",
]

# File extensions you generally want
INCLUDE_FILES: List[str] = [
    "*.msh",
    "*.nii",
    "*.nii.gz",
    "*.vtk",
    "*.vtp"
]

# Special-case include:
# Keep ONLY the subject head mesh inside m2m_* folders
INCLUDE_SPECIAL: List[str] = [
    "**/m2m_*/[!/]*.msh",
]

# Exclude everything under m2m_* except the mesh above
EXCLUDE_DIRS: List[str] = [
    "**/m2m_*/**",
    "**/tmp/**",
    "**/surfaces/**",
    "**/segmentation/**",
    "**/Volume_Maks/**",
    "**/Volume_Labels/**",
    "**/Volume_Base/**",
    "**/.git/**",
    "**/toMNI/**",
    "**/__pycache__/**",
    "**/coregistration/**",
    "**/intermediate/**",
    "**/cache/**",
]

# 🚨 NEW: exclude specific FILE NAME patterns (overrides INCLUDE_FILES)
# Example: exclude all mask volumes
EXCLUDE_FILE_PATTERNS: List[str] = [
    "*_mask_*.nii.gz",
    "*_brainmask_*.nii.gz",
    "*_corrected.nii.gz",
    "*_nonl.nii.gz",
    "*_mask.nii.gz",
]


# =============================================================================
# =============================== HELPERS ======================================
# =============================================================================

def which_or_exit(binary: str) -> None:
    if not shutil.which(binary):
        print(f"[ERROR] Required executable not found: {binary}", file=sys.stderr)
        sys.exit(2)


def format_bytes(num_bytes: int) -> str:
    gb = 1024 ** 3
    mb = 1024 ** 2
    if num_bytes >= gb:
        return f"{num_bytes / gb:.2f} GB"
    if num_bytes >= mb:
        return f"{num_bytes / mb:.2f} MB"
    return f"{num_bytes} bytes"


def parse_rsync_total_size_bytes(text: str) -> int | None:
    patterns = [
        r"Total transferred file size:\s+([\d,]+)\s+bytes",
        r"Total file size:\s+([\d,]+)\s+bytes",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


def get_directory_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


# =============================================================================
# ============================ RSYNC LOGIC =====================================
# =============================================================================

def build_rsync_filter_file(path: Path) -> None:
    lines: List[str] = []

    lines += [
        "# Auto-generated rsync filter rules",
        "",
        "# Always allow directory traversal",
        "+ */",
        "",
        "# Include main output directories",
    ]

    for d in INCLUDE_DIRS:
        lines.append(f"+ {d.strip('/')}/***")

    lines += ["", "# Special-case includes (must precede excludes)"]
    for pat in INCLUDE_SPECIAL:
        lines.append(f"+ {pat}")

    lines += ["", "# Include general file types"]
    for pat in INCLUDE_FILES:
        lines.append(f"+ **/{pat}")

    lines += ["", "# Exclude specific filename patterns"]
    for pat in EXCLUDE_FILE_PATTERNS:
        lines.append(f"- **/{pat}")

    lines += ["", "# Exclude unwanted directories"]
    for pat in EXCLUDE_DIRS:
        lines.append(f"- {pat}")

    lines += ["", "# Exclude everything else", "- **", ""]

    path.write_text("\n".join(lines), encoding="utf-8")


def run_rsync(
    src: Path,
    dst: Path,
    filter_file: Path,
    dry_run: bool,
    verbose: bool,
) -> str:
    which_or_exit("rsync")
    dst.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rsync",
        "-a",
        "--prune-empty-dirs",
        f"--filter=merge {filter_file}",
        "--info=stats2",
    ]

    if verbose:
        cmd.append("-v")
    if dry_run:
        cmd.append("--dry-run")

    cmd += [str(src) + "/", str(dst) + "/"]

    print("[INFO] Running rsync:")
    print("       " + " ".join(cmd))
    print()

    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    print(proc.stdout)
    return proc.stdout


# =============================================================================
# ================================ MAIN ========================================
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-verbose", action="store_true")
    ap.add_argument("--compress", choices=["none", "zstd", "gzip"], default="none")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()

    if not src.exists():
        print(f"[ERROR] Source not found: {src}", file=sys.stderr)
        sys.exit(2)

    with tempfile.NamedTemporaryFile("w+", suffix=".rules", delete=False) as f:
        filter_path = Path(f.name)
        build_rsync_filter_file(filter_path)

    print("[INFO] rsync filter rules:")
    print(filter_path.read_text())
    print()

    try:
        rsync_output = run_rsync(
            src, dst, filter_path, args.dry_run, not args.no_verbose
        )
    finally:
        filter_path.unlink(missing_ok=True)

    if args.dry_run:
        est = parse_rsync_total_size_bytes(rsync_output)
        print("[INFO] Dry-run complete.")
        if est:
            print(f"[INFO] Estimated transfer size: {format_bytes(est)}  ({est:,} bytes)")
        return

    size = get_directory_size_bytes(dst)
    print("[INFO] Export size:")
    print(f"       {format_bytes(size)}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
