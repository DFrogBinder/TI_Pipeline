#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import median

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from mesh_repeat_analysis.post.mesh_qc.discovery import MeshRecord, discover_meshes
    from mesh_repeat_analysis.post.mesh_qc.geometry_qc import QCMetrics, compute_qc_metrics
    from mesh_repeat_analysis.post.mesh_qc.loaders import load_surface_arrays
    from mesh_repeat_analysis.post.mesh_qc.rendering import make_mosaic, render_mesh_png
else:
    from .discovery import MeshRecord, discover_meshes
    from .geometry_qc import QCMetrics, compute_qc_metrics
    from .loaders import load_surface_arrays
    from .rendering import make_mosaic, render_mesh_png


FOUND_FIELDS = ("roi", "subject", "repeat", "path")
SUMMARY_FIELDS = (
    "roi",
    "subject",
    "repeat",
    "path",
    "status",
    "flags",
    "n_points",
    "n_faces",
    "degenerate_faces",
    "boundary_edges",
    "nonmanifold_edges",
    "connected_components",
    "x_size",
    "y_size",
    "z_size",
)


class ProgressReporter:
    def __init__(self, phase: str, total: int, *, every: int = 1) -> None:
        self.phase = phase
        self.total = max(total, 0)
        self.every = max(every, 1)
        self.start = time.monotonic()

    def update(self, current: int, detail: str = "") -> None:
        if current != self.total and current % self.every != 0:
            return
        elapsed = max(time.monotonic() - self.start, 1e-9)
        rate = current / elapsed if current else 0.0
        remaining = self.total - current
        eta = remaining / rate if rate > 0 else 0.0
        suffix = f" | {detail}" if detail else ""
        print(
            f"[{self.phase}] {current}/{self.total} "
            f"elapsed={_format_seconds(elapsed)} "
            f"rate={rate:.2f}/s eta={_format_seconds(eta)}{suffix}",
            flush=True,
        )

    def complete(self) -> None:
        elapsed = max(time.monotonic() - self.start, 1e-9)
        print(f"[{self.phase}] Complete in {_format_seconds(elapsed)}", flush=True)


def _format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _record_row(record: MeshRecord) -> dict[str, object]:
    return {
        "roi": record.roi,
        "subject": record.subject,
        "repeat": record.repeat,
        "path": str(record.path),
    }


def _summary_row(record: MeshRecord, metrics: QCMetrics) -> dict[str, object]:
    row = _record_row(record)
    row.update(metrics.as_row())
    return row


def _read_fail_row(record: MeshRecord, exc: Exception) -> dict[str, object]:
    row = _record_row(record)
    row.update(
        {
            "status": "FAIL",
            "flags": f"READ_FAIL:{exc}",
            "n_points": 0,
            "n_faces": 0,
            "degenerate_faces": 0,
            "boundary_edges": 0,
            "nonmanifold_edges": 0,
            "connected_components": 0,
            "x_size": 0.0,
            "y_size": 0.0,
            "z_size": 0.0,
        }
    )
    return row


def _as_float(row: dict[str, object], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return 0.0


def _append_flag(row: dict[str, object], flag: str) -> None:
    flags = str(row.get("flags", ""))
    current = [item for item in flags.split(";") if item]
    if flag not in current:
        current.append(flag)
    row["flags"] = ";".join(current)
    row["status"] = "FAIL"


def _apply_bounds_outlier_flags(rows: list[dict[str, object]], *, multiplier: float = 4.0) -> None:
    ok_rows = [row for row in rows if row.get("status") == "OK"]
    rows_by_roi: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in ok_rows:
        rows_by_roi[str(row["roi"])].append(row)

    for roi_rows in rows_by_roi.values():
        if len(roi_rows) < 4:
            continue
        for key in ("x_size", "y_size", "z_size"):
            values = sorted(_as_float(row, key) for row in roi_rows)
            q1 = median(values[: len(values) // 2])
            q3 = median(values[(len(values) + 1) // 2 :])
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            for row in roi_rows:
                value = _as_float(row, key)
                if value < lower or value > upper:
                    _append_flag(row, "BOUNDS_OUTLIER")


def _safe_name(value: str) -> str:
    keep = []
    for char in value:
        keep.append(char if char.isalnum() or char in ("-", "_") else "_")
    return "".join(keep).strip("_") or "unknown"


def _render_outputs(records: list[MeshRecord], out_dir: Path, args: argparse.Namespace) -> None:
    rendered_by_roi: dict[str, list[Path]] = defaultdict(list)
    all_rendered: list[Path] = []
    failures: list[str] = []
    progress = ProgressReporter("RENDER", len(records), every=args.progress_every)

    for idx, record in enumerate(records, start=1):
        roi_name = _safe_name(record.roi)
        stem = "__".join(
            _safe_name(part)
            for part in (record.roi, record.subject, record.repeat, record.path.stem)
        )
        out_png = out_dir / "renders" / roi_name / f"{idx:05d}__{stem}.png"
        label = f"{record.roi}\n{record.subject}\n{record.repeat}"
        try:
            render_mesh_png(record.path, out_png, label=label, image_size=args.image_size)
        except Exception as exc:
            failures.append(f"{record.path}\t{exc}")
            progress.update(idx, f"FAILED {record.subject} {record.repeat} {record.roi}")
            continue
        rendered_by_roi[record.roi].append(out_png)
        all_rendered.append(out_png)
        progress.update(idx, f"{record.subject} {record.repeat} {record.roi}")

    progress.complete()

    for roi, images in sorted(rendered_by_roi.items()):
        print(f"[MOSAIC] Building ROI wall for {roi} ({len(images)} tiles)", flush=True)
        make_mosaic(
            images,
            out_dir / "mosaics" / f"{_safe_name(roi)}_wall.png",
            cols=args.cols,
            tile_size=args.tile_size,
        )

    if all_rendered:
        print(f"[MOSAIC] Building combined all-ROI wall ({len(all_rendered)} tiles)", flush=True)
        make_mosaic(
            all_rendered,
            out_dir / "mosaics" / "all_roi_wall.png",
            cols=args.cols,
            tile_size=args.tile_size,
        )

    if failures:
        fail_path = out_dir / "render_failures.txt"
        fail_path.write_text("\n".join(failures) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan repeat-experiment .msh files for geometry QC flags and render ROI mosaic walls."
    )
    parser.add_argument("--root", required=True, help="Root containing ROI/subject/repeat mesh outputs.")
    parser.add_argument("--out", required=True, help="Output directory for CSV reports, renders, and mosaics.")
    parser.add_argument(
        "--mesh-glob",
        default=None,
        help=(
            "Recursive mesh glob. If omitted, only .msh files inside m2m* "
            "directories are scanned. Pass '*.msh' to scan all meshes."
        ),
    )
    parser.add_argument("--roi-regex", default=None, help="Optional regex for ROI inference; first group is used.")
    parser.add_argument("--subject-regex", default=None, help="Optional regex for subject inference; first group is used.")
    parser.add_argument("--repeat-regex", default=None, help="Optional regex for repeat inference; first group is used.")
    parser.add_argument("--image-size", type=int, default=600, help="Individual render size in pixels.")
    parser.add_argument("--tile-size", type=int, default=220, help="Mosaic tile size in pixels.")
    parser.add_argument("--cols", type=int, default=None, help="Mosaic columns; default uses square-ish grid.")
    parser.add_argument("--skip-renders", action="store_true", help="Write CSV QC reports without PNG rendering.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N meshes during QC/rendering. Use 1 for every mesh.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    records = discover_meshes(
        root,
        mesh_glob=args.mesh_glob,
        roi_regex=args.roi_regex,
        subject_regex=args.subject_regex,
        repeat_regex=args.repeat_regex,
    )
    if not records:
        print(f"No meshes found under {root} matching {args.mesh_glob}", file=sys.stderr)
        return 2

    print(f"[DISCOVERY] Found {len(records)} mesh(es) under {root}", flush=True)
    _write_csv(out_dir / "found_meshes.csv", [_record_row(r) for r in records], FOUND_FIELDS)

    summary_rows: list[dict[str, object]] = []
    progress = ProgressReporter("QC", len(records), every=args.progress_every)
    for idx, record in enumerate(records, start=1):
        try:
            surface = load_surface_arrays(record.path)
            metrics = compute_qc_metrics(surface)
            summary_rows.append(_summary_row(record, metrics))
            detail = f"{metrics.status} {record.subject} {record.repeat} {record.roi}"
        except Exception as exc:
            summary_rows.append(_read_fail_row(record, exc))
            detail = f"READ_FAIL {record.subject} {record.repeat} {record.roi}"
        progress.update(idx, detail)
    progress.complete()

    _apply_bounds_outlier_flags(summary_rows)
    _write_csv(out_dir / "qc_summary.csv", summary_rows, SUMMARY_FIELDS)
    flag_rows = [row for row in summary_rows if row["status"] != "OK"]
    _write_csv(out_dir / "qc_flags.csv", flag_rows, SUMMARY_FIELDS)

    if not args.skip_renders:
        _render_outputs(records, out_dir, args)

    print(f"Wrote QC outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
