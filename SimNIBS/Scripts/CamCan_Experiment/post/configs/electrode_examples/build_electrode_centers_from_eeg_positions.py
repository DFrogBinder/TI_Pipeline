#!/usr/bin/env python3
"""
Build a consolidated electrode-centres CSV from per-subject eeg_positions files.

This example mirrors the parser used by post-processing:
- CSV eeg_positions files with columns name,x,y,z are accepted.
- Whitespace-delimited files with rows like "AF4 28.4 73.2 61.8" are accepted.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_eeg_positions(path: Path) -> dict[str, tuple[float, float, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", newline="", encoding="utf-8") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        if "," in sample:
            reader = csv.DictReader(handle)
            required = {"name", "x", "y", "z"}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                raise ValueError(f"{path} must contain columns name,x,y,z")
            return {
                row["name"]: (float(row["x"]), float(row["y"]), float(row["z"]))
                for row in reader
            }

    positions: dict[str, tuple[float, float, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        positions[parts[0]] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return positions


def build_rows(
    subjects_root: Path,
    subjects: list[str],
    electrode_names: list[str],
    eeg_positions_template: str,
) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    missing: list[str] = []
    for subject in subjects:
        eeg_path = Path(
            eeg_positions_template.format(root=subjects_root, subject=subject)
        ).expanduser()
        positions = read_eeg_positions(eeg_path)
        for electrode in electrode_names:
            if electrode not in positions:
                missing.append(f"{subject}:{electrode}:{eeg_path}")
                continue
            x, y, z = positions[electrode]
            rows.append(
                {
                    "subject": subject,
                    "electrode": electrode,
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )
    if missing:
        details = "\n".join(missing)
        raise SystemExit(f"Missing required electrode positions:\n{details}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects-root", required=True, type=Path)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--electrode-names", nargs="+", required=True)
    parser.add_argument(
        "--eeg-positions-template",
        default="{root}/{subject}/anat/m2m_{subject}/eeg_positions.csv",
        help="Template with {root} and {subject} placeholders.",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows = build_rows(
        subjects_root=args.subjects_root,
        subjects=args.subjects,
        electrode_names=args.electrode_names,
        eeg_positions_template=args.eeg_positions_template,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", "electrode", "x", "y", "z"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
