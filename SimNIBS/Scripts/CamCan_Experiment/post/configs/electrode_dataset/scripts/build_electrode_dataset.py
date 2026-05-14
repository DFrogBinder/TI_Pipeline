#!/usr/bin/env python3
"""Build per-ROI/per-subject electrode CSV files for post-processing."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_POST_DATA_ROOT = Path(
    "/home/boyan/sandbox/Jake_Data/Simulation-Target-Data/v8-Post-Data"
)
DEFAULT_CAP_CSV = Path(
    "/home/boyan/sandbox/simnibs4_exmaples/m2m_MNI152/eeg_positions/"
    "EEG10-10_UI_Jurak_2007.csv"
)
DEFAULT_ROI_ELECTRODE_SETS = (
    Path(__file__).resolve().parents[1].parent
    / "electrode_examples"
    / "roi_electrode_sets.csv"
)
DEFAULT_OUT = Path(__file__).resolve().parents[1]


def slug(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def read_roi_electrode_sets(path: Path) -> dict[str, dict[str, str | list[str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[str, dict[str, str | list[str]]] = {}
    for row in rows:
        roi_alias = slug(row["roi_alias"])
        out[roi_alias] = {
            "roi_alias": roi_alias,
            "montage_preset": row["montage_preset"],
            "electrode_names": row["electrode_names"].split(),
            "notes": row.get("notes", ""),
        }
    return out


def read_simnibs_cap(path: Path) -> dict[str, tuple[float, float, float]]:
    """Read SimNIBS cap rows: type,x,y,z,name with no header."""
    positions: dict[str, tuple[float, float, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 5 or row[0] != "Electrode":
                continue
            positions[row[4]] = (float(row[1]), float(row[2]), float(row[3]))
    return positions


def discover_subjects(post_data_root: Path, roi_alias: str) -> list[str]:
    roi_root = post_data_root / roi_alias
    if not roi_root.is_dir():
        return []

    subjects: set[str] = set()
    for dataset_root in roi_root.glob("*_Data_*"):
        if dataset_root.is_dir():
            subjects.update(path.name for path in dataset_root.glob("sub-*") if path.is_dir())
    return sorted(subjects)


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject",
        "electrode",
        "x",
        "y",
        "z",
        "roi_alias",
        "montage_preset",
        "coordinate_source",
        "source_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(
    *,
    post_data_root: Path,
    cap_csv: Path,
    roi_electrode_sets_csv: Path,
    out_dir: Path,
    selected_rois: set[str] | None,
) -> dict[str, object]:
    roi_sets = read_roi_electrode_sets(roi_electrode_sets_csv)
    cap_positions = read_simnibs_cap(cap_csv)
    manifest_rows: list[dict[str, str | int]] = []

    if not cap_positions:
        raise SystemExit(f"No electrode positions were parsed from {cap_csv}")

    roi_aliases = sorted(path.name for path in post_data_root.iterdir() if path.is_dir())
    if selected_rois is not None:
        roi_aliases = [roi for roi in roi_aliases if roi in selected_rois]

    for roi_alias in roi_aliases:
        preset = roi_sets.get(roi_alias)
        if preset is None:
            continue

        electrode_names = list(preset["electrode_names"])
        missing_electrodes = [name for name in electrode_names if name not in cap_positions]
        if missing_electrodes:
            raise SystemExit(
                f"Missing electrode(s) in cap file for {roi_alias}: "
                + ", ".join(missing_electrodes)
            )

        subject_rows: list[dict[str, str | float]] = []
        subjects = discover_subjects(post_data_root, roi_alias)
        for subject in subjects:
            rows: list[dict[str, str | float]] = []
            for electrode in electrode_names:
                x, y, z = cap_positions[electrode]
                rows.append(
                    {
                        "subject": subject,
                        "electrode": electrode,
                        "x": x,
                        "y": y,
                        "z": z,
                        "roi_alias": roi_alias,
                        "montage_preset": str(preset["montage_preset"]),
                        "coordinate_source": "MNI152_EEG10-10_UI_Jurak_2007",
                        "source_file": str(cap_csv),
                    }
                )
            subject_rows.extend(rows)
            write_csv(out_dir / roi_alias / subject / "electrodes.csv", rows)

        write_csv(out_dir / roi_alias / "electrode_centers.csv", subject_rows)
        manifest_rows.append(
            {
                "roi_alias": roi_alias,
                "montage_preset": str(preset["montage_preset"]),
                "subjects": len(subjects),
                "electrodes_per_subject": len(electrode_names),
                "rows": len(subject_rows),
                "electrode_names": " ".join(electrode_names),
            }
        )

    manifest_path = out_dir / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "roi_alias",
                "montage_preset",
                "subjects",
                "electrodes_per_subject",
                "rows",
                "electrode_names",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "post_data_root": str(post_data_root),
        "cap_csv": str(cap_csv),
        "roi_electrode_sets_csv": str(roi_electrode_sets_csv),
        "out_dir": str(out_dir),
        "roi_count": len(manifest_rows),
        "subject_files": sum(int(row["subjects"]) for row in manifest_rows),
        "rows": sum(int(row["rows"]) for row in manifest_rows),
        "manifest": str(manifest_path),
    }
    (out_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post-data-root", type=Path, default=DEFAULT_POST_DATA_ROOT)
    parser.add_argument("--cap-csv", type=Path, default=DEFAULT_CAP_CSV)
    parser.add_argument("--roi-electrode-sets-csv", type=Path, default=DEFAULT_ROI_ELECTRODE_SETS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--roi", action="append", help="ROI alias to build, e.g. right-dlpc.")
    args = parser.parse_args()

    selected_rois = {slug(value) for value in args.roi} if args.roi else None
    summary = build_dataset(
        post_data_root=args.post_data_root.expanduser(),
        cap_csv=args.cap_csv.expanduser(),
        roi_electrode_sets_csv=args.roi_electrode_sets_csv.expanduser(),
        out_dir=args.out_dir.expanduser(),
        selected_rois=selected_rois,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
