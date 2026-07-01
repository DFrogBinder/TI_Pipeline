#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from statistics import median

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from mesh_repeat_analysis.post.mesh_qc.discovery import DiscoveryStats, MeshRecord, discover_meshes
    from mesh_repeat_analysis.post.mesh_qc.geometry_qc import QCMetrics, compute_qc_metrics
    from mesh_repeat_analysis.post.mesh_qc.loaders import load_surface_arrays
    from mesh_repeat_analysis.post.mesh_qc.rendering import make_mosaic, render_mesh_png
else:
    from .discovery import DiscoveryStats, MeshRecord, discover_meshes
    from .geometry_qc import QCMetrics, compute_qc_metrics
    from .loaders import load_surface_arrays
    from .rendering import make_mosaic, render_mesh_png

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


FOUND_FIELDS = ("mesh_id", "subject", "repeat", "roi", "path")
SUMMARY_FIELDS = (
    "mesh_id",
    "subject",
    "repeat",
    "roi",
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


class TqdmProgressReporter:
    def __init__(self, phase: str, total: int) -> None:
        if tqdm is None:
            raise RuntimeError("tqdm is not available")
        self.bar = tqdm(total=total, desc=phase, unit="mesh", dynamic_ncols=True)
        self.current = 0

    def update(self, current: int, detail: str = "") -> None:
        delta = current - self.current
        if delta > 0:
            self.bar.update(delta)
            self.current = current
        if detail:
            self.bar.set_postfix_str(detail)

    def complete(self) -> None:
        self.bar.close()


class NoopProgressReporter:
    def update(self, current: int, detail: str = "") -> None:
        return

    def complete(self) -> None:
        return


class TqdmDiscoveryReporter:
    def __init__(self) -> None:
        if tqdm is None:
            raise RuntimeError("tqdm is not available")
        self.bar = tqdm(total=None, desc="Discovery", unit="dir", dynamic_ncols=True)
        self.dirs_seen = 0

    def update(self, stats: DiscoveryStats) -> None:
        delta = stats.dirs_scanned - self.dirs_seen
        if delta > 0:
            self.bar.update(delta)
            self.dirs_seen = stats.dirs_scanned
        self.bar.set_postfix_str(f"files={stats.files_seen} matches={stats.matches}")

    def complete(self) -> None:
        self.bar.close()


def _use_tqdm_progress(args: argparse.Namespace) -> bool:
    if args.progress == "tqdm":
        if tqdm is None:
            print("[PROGRESS] tqdm requested but not installed; using text progress", flush=True)
            return False
        return True
    if args.progress == "auto":
        return tqdm is not None
    return False


def _make_progress_reporter(args: argparse.Namespace, phase: str, total: int):
    if args.progress == "none":
        return NoopProgressReporter()
    if _use_tqdm_progress(args):
        return TqdmProgressReporter(phase, total)
    return ProgressReporter(phase.upper(), total, every=args.progress_every)


def _format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _print_discovery_progress(stats: DiscoveryStats) -> None:
    print(
        f"[DISCOVERY] dirs={stats.dirs_scanned} "
        f"files={stats.files_seen} matches={stats.matches} "
        f"current={stats.current_dir}",
        flush=True,
    )


def _make_discovery_callback(args: argparse.Namespace):
    if args.progress == "none":
        return None, None
    if _use_tqdm_progress(args):
        reporter = TqdmDiscoveryReporter()
        return reporter.update, reporter
    return _print_discovery_progress, None


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _record_row(record: MeshRecord) -> dict[str, object]:
    return {
        "mesh_id": record.mesh_id,
        "subject": record.subject,
        "repeat": record.repeat,
        "roi": record.roi,
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
    if len(ok_rows) < 4:
        return
    for key in ("x_size", "y_size", "z_size"):
        values = sorted(_as_float(row, key) for row in ok_rows)
        q1 = median(values[: len(values) // 2])
        q3 = median(values[(len(values) + 1) // 2 :])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        for row in ok_rows:
            value = _as_float(row, key)
            if value < lower or value > upper:
                _append_flag(row, "BOUNDS_OUTLIER")


def _safe_name(value: str) -> str:
    keep = []
    for char in value:
        keep.append(char if char.isalnum() or char in ("-", "_") else "_")
    return "".join(keep).strip("_") or "unknown"


def _mesh_detail(record: MeshRecord, status: str | None = None) -> str:
    prefix = f"{status} " if status else ""
    return f"{prefix}{record.subject} {record.repeat} {record.mesh_id} {record.path.name}"


def _resolve_worker_count(requested_workers: int) -> int:
    if requested_workers > 0:
        return requested_workers

    slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus_per_task:
        try:
            value = int(slurm_cpus_per_task)
            if value > 0:
                return value
        except ValueError:
            pass

    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = os.sched_getaffinity(0)
            if affinity:
                return len(affinity)
        except Exception:
            pass

    return os.cpu_count() or 1


def _status_label(row: dict[str, object]) -> str:
    status = str(row.get("status", "UNKNOWN"))
    flags = str(row.get("flags", ""))
    if flags:
        first_flag = flags.split(";", 1)[0]
        if first_flag.startswith("READ_FAIL"):
            first_flag = "READ_FAIL"
        return f"{status}:{first_flag}"
    return status


def _qc_record_worker(record: MeshRecord, check_components: bool) -> dict[str, object]:
    try:
        surface = load_surface_arrays(record.path)
        metrics = compute_qc_metrics(surface, check_components=check_components)
        return _summary_row(record, metrics)
    except Exception as exc:
        return _read_fail_row(record, exc)


def _run_qc(records: list[MeshRecord], args: argparse.Namespace) -> list[dict[str, object]]:
    workers = max(1, _resolve_worker_count(args.workers))

    summary_rows: list[dict[str, object] | None] = [None] * len(records)
    progress = _make_progress_reporter(args, "QC", len(records))

    if workers == 1:
        for idx, record in enumerate(records, start=1):
            row = _qc_record_worker(record, args.check_components)
            summary_rows[idx - 1] = row
            progress.update(idx, _mesh_detail(record, _status_label(row)))
        progress.complete()
        return [row for row in summary_rows if row is not None]

    print(f"[QC] Using {workers} worker processes", flush=True)
    completed = 0
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
            futures = {
                executor.submit(_qc_record_worker, record, args.check_components): (idx, record)
                for idx, record in enumerate(records)
            }
            for future in as_completed(futures):
                idx, record = futures[future]
                row = future.result()
                summary_rows[idx] = row
                completed += 1
                progress.update(completed, _mesh_detail(record, _status_label(row)))
    except Exception as exc:
        progress.complete()
        raise RuntimeError(
            f"Parallel QC failed with {workers} workers ({exc}). "
            "Retry with a smaller worker count such as --workers 8."
        ) from exc
    progress.complete()
    return [row for row in summary_rows if row is not None]


def _load_found_records(found_path: Path) -> list[MeshRecord]:
    rows = _read_csv(found_path)
    records = []
    for row in rows:
        records.append(
            MeshRecord(
                path=Path(row["path"]).expanduser(),
                roi=row["roi"],
                subject=row["subject"],
                repeat=row["repeat"],
                mesh_id=row["mesh_id"],
            )
        )
    return records


def _render_record_worker(
    record: MeshRecord,
    out_png: Path,
    *,
    image_size: int,
    renderer: str,
) -> str | None:
    label = f"{record.subject}\n{record.repeat}\n{record.mesh_id}"
    try:
        render_mesh_png(
            record.path,
            out_png,
            label=label,
            image_size=image_size,
            renderer=renderer,
        )
        return None
    except Exception as exc:
        return str(exc)


def _render_outputs(
    records: list[MeshRecord],
    summary_rows: list[dict[str, object]],
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    rows_by_path = {str(row["path"]): row for row in summary_rows}
    render_records = [
        record
        for record in records
        if not str(rows_by_path.get(str(record.path), {}).get("flags", "")).startswith("READ_FAIL")
    ]
    skipped = len(records) - len(render_records)
    if skipped:
        print(f"[RENDER] Skipping {skipped} mesh(es) that failed QC loading", flush=True)
    if not render_records:
        print("[RENDER] No QC-loadable meshes to render", flush=True)
        return

    workers = max(1, min(_resolve_worker_count(args.workers), len(render_records)))
    task_specs = []
    for idx, record in enumerate(render_records, start=1):
        stem = "__".join(
            _safe_name(part)
            for part in (record.subject, record.repeat, record.mesh_id, record.path.stem)
        )
        out_png = out_dir / "renders" / "meshes" / f"{idx:05d}__{stem}.png"
        task_specs.append((idx - 1, record, out_png))

    rendered_slots: list[Path | None] = [None] * len(task_specs)
    failures: list[str] = []
    progress = _make_progress_reporter(args, "Render", len(render_records))

    if workers == 1:
        for completed, (slot_idx, record, out_png) in enumerate(task_specs, start=1):
            error = _render_record_worker(
                record,
                out_png,
                image_size=args.image_size,
                renderer=args.renderer,
            )
            if error is not None:
                failures.append(f"{record.path}\t{error}")
                progress.update(completed, _mesh_detail(record, "FAILED"))
                continue
            rendered_slots[slot_idx] = out_png
            progress.update(completed, _mesh_detail(record))
    else:
        print(f"[RENDER] Using {workers} worker processes", flush=True)
        completed = 0
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
                futures = {
                    executor.submit(
                        _render_record_worker,
                        record,
                        out_png,
                        image_size=args.image_size,
                        renderer=args.renderer,
                    ): (slot_idx, record, out_png)
                    for slot_idx, record, out_png in task_specs
                }
                for future in as_completed(futures):
                    slot_idx, record, out_png = futures[future]
                    error = future.result()
                    completed += 1
                    if error is not None:
                        failures.append(f"{record.path}\t{error}")
                        progress.update(completed, _mesh_detail(record, "FAILED"))
                        continue
                    rendered_slots[slot_idx] = out_png
                    progress.update(completed, _mesh_detail(record))
        except Exception as exc:
            progress.complete()
            raise RuntimeError(
                f"Parallel rendering failed with {workers} workers ({exc}). "
                "Retry with a smaller worker count such as --workers 8."
            ) from exc

    progress.complete()

    successful = [
        (record, out_png)
        for (_, record, _), out_png in zip(task_specs, rendered_slots)
        if out_png is not None
    ]
    rendered_by_roi: dict[str, list[Path]] = defaultdict(list)
    all_rendered: list[Path] = []
    for record, out_png in successful:
        rendered_by_roi[record.roi].append(out_png)
        all_rendered.append(out_png)

    if args.roi_walls:
        for roi, images in sorted(rendered_by_roi.items()):
            print(f"[MOSAIC] Building ROI wall for {roi} ({len(images)} tiles)", flush=True)
            make_mosaic(
                images,
                out_dir / "mosaics" / f"{_safe_name(roi)}_wall.png",
                cols=args.cols,
                tile_size=args.tile_size,
            )

    if all_rendered:
        print(f"[MOSAIC] Building combined mesh wall ({len(all_rendered)} tiles)", flush=True)
        make_mosaic(
            all_rendered,
            out_dir / "mosaics" / "all_mesh_wall.png",
            cols=args.cols,
            tile_size=args.tile_size,
        )

    if failures:
        fail_path = out_dir / "render_failures.txt"
        fail_path.write_text("\n".join(failures) + "\n", encoding="utf-8")


def _validate_stage_args(args: argparse.Namespace) -> None:
    if args.render_only and (args.qc_only or args.skip_renders):
        raise ValueError("--render-only cannot be combined with --qc-only or --skip-renders")


def _run_discovery(root: Path, args: argparse.Namespace, progress_mode: str) -> list[MeshRecord]:
    if args.mesh_glob is None:
        print(
            f"[DISCOVERY] Scanning {root} for .msh files inside m2m* directories "
            f"(progress={progress_mode})",
            flush=True,
        )
    else:
        print(
            f"[DISCOVERY] Scanning {root} for mesh glob {args.mesh_glob!r} "
            f"(progress={progress_mode})",
            flush=True,
        )
    discovery_callback, discovery_reporter = _make_discovery_callback(args)
    records = discover_meshes(
        root,
        mesh_glob=args.mesh_glob,
        roi_regex=args.roi_regex,
        subject_regex=args.subject_regex,
        repeat_regex=args.repeat_regex,
        progress_callback=discovery_callback,
        progress_interval_sec=args.discovery_progress_seconds,
    )
    if discovery_reporter is not None:
        discovery_reporter.complete()
    return records


def _run_full_or_qc_only(root: Path, out_dir: Path, args: argparse.Namespace, progress_mode: str) -> int:
    records = _run_discovery(root, args, progress_mode)
    if not records:
        print(f"No meshes found under {root} matching {args.mesh_glob}", file=sys.stderr)
        return 2

    print(f"[DISCOVERY] Found {len(records)} mesh(es) under {root}", flush=True)
    _write_csv(out_dir / "found_meshes.csv", [_record_row(r) for r in records], FOUND_FIELDS)

    summary_rows = _run_qc(records, args)

    _apply_bounds_outlier_flags(summary_rows)
    _write_csv(out_dir / "qc_summary.csv", summary_rows, SUMMARY_FIELDS)
    flag_rows = [row for row in summary_rows if row["status"] != "OK"]
    _write_csv(out_dir / "qc_flags.csv", flag_rows, SUMMARY_FIELDS)

    if not (args.skip_renders or args.qc_only):
        _render_outputs(records, summary_rows, out_dir, args)

    print(f"Wrote QC outputs to {out_dir}")
    return 0


def _run_render_only(out_dir: Path, args: argparse.Namespace) -> int:
    found_path = out_dir / "found_meshes.csv"
    summary_path = out_dir / "qc_summary.csv"
    if not found_path.exists():
        print(f"Missing render input: {found_path}", file=sys.stderr)
        return 2
    if not summary_path.exists():
        print(f"Missing render input: {summary_path}", file=sys.stderr)
        return 2

    print(f"[RENDER] Loading prior QC outputs from {out_dir}", flush=True)
    records = _load_found_records(found_path)
    summary_rows = _read_csv(summary_path)
    if not records:
        print(f"No meshes found in {found_path}", file=sys.stderr)
        return 2

    print(
        f"[RENDER] Loaded {len(records)} discovered mesh(es) and {len(summary_rows)} QC row(s)",
        flush=True,
    )
    _render_outputs(records, summary_rows, out_dir, args)
    print(f"Wrote render outputs to {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan generated m2m .msh files for geometry QC flags and render mesh mosaic walls."
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
    parser.add_argument("--image-size", type=int, default=1200, help="Individual render size in pixels.")
    parser.add_argument("--tile-size", type=int, default=220, help="Mosaic tile size in pixels.")
    parser.add_argument("--cols", type=int, default=None, help="Mosaic columns; default uses square-ish grid.")
    parser.add_argument(
        "--renderer",
        choices=("auto", "pyvista", "pillow"),
        default="pillow",
        help="PNG renderer. Default is the pure Pillow software renderer.",
    )
    parser.add_argument(
        "--check-components",
        action="store_true",
        help=(
            "Run disconnected-component analysis. This is slower and is disabled "
            "by default for large HPC batches."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel worker processes for QC. Default 0 uses all available CPUs.",
    )
    parser.add_argument("--skip-renders", action="store_true", help="Write CSV QC reports without PNG rendering.")
    parser.add_argument(
        "--qc-only",
        action="store_true",
        help="Run discovery and QC only, then stop before rendering.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Skip discovery and QC, and render from existing found_meshes.csv and qc_summary.csv in --out.",
    )
    parser.add_argument(
        "--roi-walls",
        action="store_true",
        help="Also write separate ROI wall mosaics. Default writes only all_mesh_wall.png.",
    )
    parser.add_argument(
        "--progress",
        choices=("auto", "tqdm", "text", "none"),
        default="auto",
        help="Progress display mode. auto uses tqdm if installed, otherwise text.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="In text progress mode, print every N meshes during QC/rendering. Use 1 for every mesh.",
    )
    parser.add_argument(
        "--discovery-progress-seconds",
        type=float,
        default=5.0,
        help="Print discovery progress at least this often while walking the filesystem.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    try:
        _validate_stage_args(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    use_tqdm = _use_tqdm_progress(args)
    if args.progress != "none" and not use_tqdm:
        progress_mode = "text"
    elif use_tqdm:
        progress_mode = "tqdm"
    else:
        progress_mode = "none"
    if args.render_only:
        return _run_render_only(out_dir, args)
    return _run_full_or_qc_only(root, out_dir, args, progress_mode)


if __name__ == "__main__":
    raise SystemExit(main())
