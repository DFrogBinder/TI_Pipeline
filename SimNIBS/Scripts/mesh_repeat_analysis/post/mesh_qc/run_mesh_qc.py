#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import inspect
import json
import os
import resource
import select
import shutil
import socket
import subprocess
import sys
import time
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from mesh_repeat_analysis.post.mesh_qc.discovery import (
        DiscoveryStats,
        MeshRecord,
        discover_meshes,
        infer_record,
    )
    from mesh_repeat_analysis.post.mesh_qc.geometry_qc import (
        QCMetrics,
        compute_qc_metrics,
    )
    from mesh_repeat_analysis.post.mesh_qc.loaders import (
        iter_tissue_surface_arrays,
        load_surface_arrays,
    )
    from mesh_repeat_analysis.post.mesh_qc.rendering import (
        make_mosaic,
        render_mesh_png,
        render_surface_png,
    )
else:
    from .discovery import DiscoveryStats, MeshRecord, discover_meshes, infer_record
    from .geometry_qc import QCMetrics, compute_qc_metrics
    from .loaders import iter_tissue_surface_arrays, load_surface_arrays
    from .rendering import make_mosaic, render_mesh_png, render_surface_png

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
EXCEPTION_FIELDS = (
    "stage",
    "subject",
    "repeat",
    "roi",
    "mesh_id",
    "path",
    "output_path",
    "error_type",
    "error_message",
    "traceback",
)
RENDER_MANIFEST_FIELDS = (
    "mesh_id",
    "subject",
    "repeat",
    "roi",
    "path",
    "output_path",
    "requested_renderer",
    "actual_renderer",
    "resumed",
)
RENDER_COMPLETENESS_FIELDS = (
    "roi",
    "status",
    "discovered_meshes",
    "qc_loadable_meshes",
    "rendered_meshes",
    "qc_read_failures",
    "render_failures",
    "missing_tiles",
)
TISSUE_PRESENCE_FIELDS = FOUND_FIELDS + (
    "tissue_tag",
    "tissue_name",
    "tissue_slug",
)
TISSUE_RENDER_MANIFEST_FIELDS = FOUND_FIELDS + (
    "tissue_tag",
    "tissue_name",
    "tissue_slug",
    "view",
    "output_path",
    "requested_renderer",
    "actual_renderer",
    "resumed",
)
TISSUE_EXCEPTION_FIELDS = EXCEPTION_FIELDS + (
    "tissue_tag",
    "tissue_name",
    "tissue_slug",
    "view",
)
TISSUE_RENDER_COMPLETENESS_FIELDS = (
    "tissue_tag",
    "tissue_name",
    "tissue_slug",
    "view",
    "status",
    "qc_loadable_meshes",
    "present_meshes",
    "rendered_meshes",
    "missing_from_meshes",
    "render_failures",
    "missing_tiles",
)
STANDARD_TISSUE_VIEWS = ("front", "back")
COMPACT_BONE_TAG = 7
COMPACT_BONE_EXTRA_VIEWS = ("top",)
TISSUE_VIEWS = STANDARD_TISSUE_VIEWS + COMPACT_BONE_EXTRA_VIEWS
LEGACY_TISSUE_VIEW_CONVENTION = {
    "version": "ras_anatomical_orthographic_v2",
    "front": "camera from RAS +Y toward the origin",
    "back": "camera from RAS -Y toward the origin",
    "up": "RAS +Z",
    "projection": "orthographic",
}
TISSUE_VIEW_CONVENTION = {
    "version": "ras_anatomical_orthographic_compact_top_v3",
    "front": "camera from RAS +Y toward the origin",
    "back": "camera from RAS -Y toward the origin",
    "front_back_up": "RAS +Z",
    "compact_bone_top": "camera from RAS +Z toward the origin; RAS +Y toward image top",
    "projection": "orthographic",
}
TISSUE_VIEW_CONVENTION_FILENAME = "tissue_view_convention.json"


_RUN_LOGGER = None


class RunLogger:
    def __init__(self, out_dir: Path, *, run_label: str = "") -> None:
        self.out_dir = Path(out_dir)
        self.log_dir = self.out_dir / "logs"
        suffix = f"_{_safe_name(run_label)}" if run_label else ""
        self.text_path = self.log_dir / f"mesh_qc{suffix}.log"
        self.context_path = self.log_dir / f"run_context{suffix}.json"
        self.fatal_path = self.log_dir / f"fatal_error{suffix}.txt"
        self.available = True
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.available = False
            print(
                f"[LOG] Could not create log directory {self.log_dir}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    def log(self, level: str, stage: str, message: str, **fields) -> None:
        if not self.available:
            return
        stamp = _utc_now()
        suffix = ""
        if fields:
            parts = [f"{key}={fields[key]}" for key in sorted(fields)]
            suffix = " | " + " ".join(parts)
        line = f"{stamp} [{level}] [{stage}] {message}{suffix}\n"
        try:
            with self.text_path.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            return

    def write_context(self, payload: dict[str, object]) -> None:
        if not self.available:
            return
        try:
            with self.context_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
        except Exception:
            return

    def write_fatal(self, *, stage: str, exc: Exception, traceback_text: str) -> None:
        if not self.available:
            return
        lines = [
            f"time: {_utc_now()}",
            f"stage: {stage}",
            f"error_type: {type(exc).__name__}",
            f"error_message: {exc}",
            "",
            traceback_text.rstrip(),
            "",
        ]
        try:
            with self.fatal_path.open("w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception:
            return


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


class VirtualDisplaySession:
    def __init__(
        self,
        *,
        width: int = 1600,
        height: int = 1600,
        depth: int = 24,
        startup_timeout: float = 10.0,
    ) -> None:
        self.width = max(width, 800)
        self.height = max(height, 800)
        self.depth = max(depth, 24)
        self.startup_timeout = max(float(startup_timeout), 1.0)
        self.proc: subprocess.Popen | None = None
        self.previous_display = os.environ.get("DISPLAY")
        self._set_libgl = False

    def start(self) -> str:
        xvfb_bin = shutil.which("Xvfb")
        if xvfb_bin is None:
            raise RuntimeError("Xvfb executable was not found in PATH.")

        read_fd, write_fd = os.pipe()
        try:
            cmd = [
                xvfb_bin,
                "-displayfd",
                str(write_fd),
                "-screen",
                "0",
                f"{self.width}x{self.height}x{self.depth}",
                "+extension",
                "GLX",
                "+render",
                "-nolisten",
                "tcp",
            ]
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                pass_fds=(write_fd,),
                close_fds=True,
            )
        finally:
            os.close(write_fd)

        try:
            ready, _, _ = select.select([read_fd], [], [], self.startup_timeout)
            if ready:
                with os.fdopen(read_fd, "r", encoding="utf-8", closefd=True) as pipe:
                    display_number = pipe.readline().strip()
            else:
                os.close(read_fd)
                display_number = ""
        except Exception:
            display_number = ""

        if not display_number:
            detail = f"Xvfb did not report a display within {self.startup_timeout:.0f} seconds"
            if self.proc is not None:
                try:
                    _, stderr = self.proc.communicate(timeout=1)
                    detail = stderr.strip() or detail
                except Exception:
                    self.proc.kill()
                    _, stderr = self.proc.communicate()
                    detail = stderr.strip() or detail
            self.stop()
            raise RuntimeError(_short_error(detail))

        os.environ["DISPLAY"] = f":{display_number}"
        if "LIBGL_ALWAYS_SOFTWARE" not in os.environ:
            os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
            self._set_libgl = True
        return os.environ["DISPLAY"]

    def stop(self) -> None:
        if self.previous_display is None:
            os.environ.pop("DISPLAY", None)
        else:
            os.environ["DISPLAY"] = self.previous_display
        if self._set_libgl:
            os.environ.pop("LIBGL_ALWAYS_SOFTWARE", None)
            self._set_libgl = False

        if self.proc is None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
            self.proc.wait(timeout=5)
        finally:
            self.proc = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log_info(stage: str, message: str, **fields) -> None:
    if _RUN_LOGGER is not None:
        _RUN_LOGGER.log("INFO", stage, message, **fields)


def _log_warning(stage: str, message: str, **fields) -> None:
    if _RUN_LOGGER is not None:
        _RUN_LOGGER.log("WARNING", stage, message, **fields)


def _log_error(stage: str, message: str, **fields) -> None:
    if _RUN_LOGGER is not None:
        _RUN_LOGGER.log("ERROR", stage, message, **fields)


def _short_error(exc: Exception | str, *, limit: int = 1200) -> str:
    text = str(exc).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ... [truncated]"


def _write_optional_csv(
    path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]
) -> None:
    if rows:
        _write_csv(path, rows, fieldnames)
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _resolved_stage_name(args: argparse.Namespace) -> str:
    if args.tissue_collect_only:
        return "tissue_collect"
    if args.tissue_only:
        if args.tissue_shard_count > 1:
            return "tissue_shard"
        return "tissue"
    if args.render_only:
        return "render"
    if args.qc_only or args.skip_renders:
        return "qc"
    return "full"


def _affinity_cpu_count() -> int | None:
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = os.sched_getaffinity(0)
            if affinity:
                return len(affinity)
        except Exception:
            return None
    return None


def _maxrss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value
    return value * 1024


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(size)}{unit}"
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{int(value)}B"


def _parse_memory_value_to_bytes(
    value: str | None, *, default_unit: str = "M"
) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "unlimited", "max"}:
        return None
    suffix = text[-1].upper()
    if suffix.isalpha():
        number = text[:-1].strip()
        unit = suffix
    else:
        number = text
        unit = default_unit.upper()
    try:
        base_value = float(number)
    except ValueError:
        return None
    units = {
        "B": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
    }
    multiplier = units.get(unit)
    if multiplier is None:
        return None
    return int(base_value * multiplier)


def _resolve_memory_budget_bytes(visible_workers: int) -> int | None:
    per_node = _parse_memory_value_to_bytes(
        os.environ.get("SLURM_MEM_PER_NODE"), default_unit="M"
    )
    if per_node:
        return per_node

    per_cpu = _parse_memory_value_to_bytes(
        os.environ.get("SLURM_MEM_PER_CPU"), default_unit="M"
    )
    if per_cpu:
        return per_cpu * max(visible_workers, 1)

    for candidate in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        path = Path(candidate)
        if not path.exists():
            continue
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        limit = _parse_memory_value_to_bytes(raw, default_unit="B")
        if limit and limit < (1 << 60):
            return limit
    return None


def _choose_stage_worker_count(
    *,
    stage: str,
    task_count: int,
    visible_workers: int,
    memory_budget_bytes: int | None,
    sampled_worker_rss_bytes: int | None,
) -> tuple[int, dict[str, object]]:
    upper = max(1, min(visible_workers, max(task_count, 1)))
    chosen = upper
    limited_by_memory = False
    usable_memory_bytes = None
    per_worker_budget_bytes = None
    if (
        memory_budget_bytes
        and sampled_worker_rss_bytes
        and sampled_worker_rss_bytes > 0
    ):
        usable_memory_bytes = int(memory_budget_bytes * 0.70)
        per_worker_budget_bytes = max(
            int(sampled_worker_rss_bytes * 1.40), 256 * 1024**2
        )
        chosen = max(1, min(chosen, usable_memory_bytes // per_worker_budget_bytes))
        limited_by_memory = chosen < upper
    details = {
        "stage": stage,
        "visible_workers": upper,
        "memory_budget_bytes": memory_budget_bytes,
        "sampled_worker_rss_bytes": sampled_worker_rss_bytes,
        "usable_memory_bytes": usable_memory_bytes,
        "per_worker_budget_bytes": per_worker_budget_bytes,
        "limited_by_memory": limited_by_memory,
    }
    return chosen, details


def _process_pool_executor_kwargs(workers: int) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "max_workers": workers,
        "mp_context": mp.get_context("spawn"),
    }
    try:
        supports_recycling = (
            "max_tasks_per_child" in inspect.signature(ProcessPoolExecutor).parameters
        )
    except (TypeError, ValueError):
        supports_recycling = True
    if supports_recycling:
        kwargs["max_tasks_per_child"] = 1
    return kwargs


def _build_run_context(
    root: Path, out_dir: Path, args: argparse.Namespace, argv: list[str] | None
) -> dict[str, object]:
    visible_workers = _resolve_worker_count(args.workers)
    return {
        "time_utc": _utc_now(),
        "argv": list(argv) if argv is not None else sys.argv[1:],
        "cwd": str(Path.cwd()),
        "script_path": str(Path(__file__).resolve()),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "root": str(root),
        "out": str(out_dir),
        "stage": _resolved_stage_name(args),
        "renderer": args.renderer,
        "mesh_glob": args.mesh_glob,
        "workers_requested": args.workers,
        "workers_visible": visible_workers,
        "memory_budget_bytes": _resolve_memory_budget_bytes(visible_workers),
        "cpu_count": os.cpu_count(),
        "affinity_cpu_count": _affinity_cpu_count(),
        "progress": args.progress,
        "image_size": args.image_size,
        "tile_size": args.tile_size,
        "cols": args.cols,
        "roi_walls": args.roi_walls,
        "tissue_walls": args.tissue_walls or args.tissue_only,
        "tissue_only": args.tissue_only,
        "tissue_collect_only": args.tissue_collect_only,
        "tissue_shard_count": args.tissue_shard_count,
        "tissue_shard_index": args.tissue_shard_index,
        "run_label": args.run_label,
        "check_components": args.check_components,
        "render_only": args.render_only,
        "qc_only": args.qc_only,
        "skip_renders": args.skip_renders,
        "display": os.environ.get("DISPLAY", ""),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE", ""),
    }


def _use_tqdm_progress(args: argparse.Namespace) -> bool:
    if args.progress == "tqdm":
        if tqdm is None:
            print(
                "[PROGRESS] tqdm requested but not installed; using text progress",
                flush=True,
            )
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


def _write_csv(
    path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]
) -> None:
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


def _exception_row(
    record: MeshRecord,
    *,
    stage: str,
    exc: Exception,
    traceback_text: str,
    output_path: Path | None = None,
) -> dict[str, object]:
    return {
        "stage": stage,
        "subject": record.subject,
        "repeat": record.repeat,
        "roi": record.roi,
        "mesh_id": record.mesh_id,
        "path": str(record.path),
        "output_path": "" if output_path is None else str(output_path),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback_text,
    }


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


def _apply_bounds_outlier_flags(
    rows: list[dict[str, object]], *, multiplier: float = 4.0
) -> None:
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
    return (
        f"{prefix}{record.subject} {record.repeat} {record.mesh_id} {record.path.name}"
    )


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


def _qc_record_worker(
    record: MeshRecord, check_components: bool
) -> tuple[dict[str, object], dict[str, object] | None]:
    try:
        surface = load_surface_arrays(record.path)
        metrics = compute_qc_metrics(surface, check_components=check_components)
        return _summary_row(record, metrics), None
    except Exception as exc:
        return _read_fail_row(record, exc), _exception_row(
            record,
            stage="qc",
            exc=exc,
            traceback_text=traceback.format_exc(),
        )


def _qc_record_worker_profiled(
    record: MeshRecord, check_components: bool
) -> tuple[dict[str, object], dict[str, object] | None, int]:
    row, exception_row = _qc_record_worker(record, check_components)
    return row, exception_row, _maxrss_bytes()


def _isolated_worker_entry(queue, fn, args) -> None:
    queue.put(fn(*args))


def _run_isolated_worker(fn, *args):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_isolated_worker_entry, args=(queue, fn, args))
    proc.start()
    proc.join()
    payload = None
    if proc.exitcode == 0:
        try:
            payload = queue.get_nowait()
        except Exception:
            payload = None
    queue.close()
    queue.join_thread()
    return payload, int(proc.exitcode or 0)


def _qc_worker_exit_payload(
    record: MeshRecord,
    exitcode: int,
) -> tuple[dict[str, object], dict[str, object], int]:
    message = (
        f"worker process exited before returning a QC result (exitcode={exitcode})"
    )
    row = _record_row(record)
    row.update(
        {
            "status": "FAIL",
            "flags": f"READ_FAIL:WORKER_EXIT_{exitcode}",
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
    exception_row = _exception_row(
        record,
        stage="qc",
        exc=RuntimeError(message),
        traceback_text=message,
    )
    return row, exception_row, 0


def _run_qc_isolated_once(
    record: MeshRecord,
    check_components: bool,
) -> tuple[dict[str, object], dict[str, object] | None, int]:
    payload, exitcode = _run_isolated_worker(
        _qc_record_worker_profiled, record, check_components
    )
    if exitcode != 0 or payload is None:
        return _qc_worker_exit_payload(record, exitcode)
    return payload


def _run_qc_direct_once(
    record: MeshRecord,
    check_components: bool,
) -> tuple[dict[str, object], dict[str, object] | None, int]:
    row, exception_row = _qc_record_worker(record, check_components)
    return row, exception_row, 0


def _run_qc_warmup_samples(
    records: list[MeshRecord],
    args: argparse.Namespace,
    summary_rows: list[dict[str, object] | None],
    exception_rows: list[dict[str, object]],
    progress,
) -> tuple[int, int | None, list[int]]:
    warmup_count = 0
    visible_workers = max(1, min(_resolve_worker_count(args.workers), len(records)))
    if visible_workers > 1 and len(records) >= 4:
        warmup_count = min(2, len(records))
    completed = 0
    peak_rss = 0
    for idx in range(warmup_count):
        record = records[idx]
        row, exception_row, rss_bytes = _run_qc_isolated_once(
            record, args.check_components
        )
        summary_rows[idx] = row
        if exception_row is not None:
            exception_rows.append(exception_row)
        peak_rss = max(peak_rss, rss_bytes)
        completed += 1
        progress.update(completed, _mesh_detail(record, _status_label(row)))
    remaining_indices = list(range(warmup_count, len(records)))
    return completed, (peak_rss or None), remaining_indices


def _run_qc_parallel_attempt(
    records: list[MeshRecord],
    indices: list[int],
    args: argparse.Namespace,
    workers: int,
    summary_rows: list[dict[str, object] | None],
    exception_rows: list[dict[str, object]],
    progress,
    completed: int,
) -> tuple[int, list[int], int, Exception | None]:
    in_flight: dict[object, int] = {}
    next_pos = 0
    peak_rss = 0
    backlog = max(workers * 2, workers)
    current_idx: int | None = None
    try:
        with ProcessPoolExecutor(**_process_pool_executor_kwargs(workers)) as executor:
            while next_pos < len(indices) and len(in_flight) < backlog:
                idx = indices[next_pos]
                next_pos += 1
                in_flight[
                    executor.submit(
                        _qc_record_worker_profiled, records[idx], args.check_components
                    )
                ] = idx

            while in_flight:
                future = next(iter(as_completed(tuple(in_flight))))
                current_idx = in_flight.pop(future)
                row, exception_row, rss_bytes = future.result()
                summary_rows[current_idx] = row
                if exception_row is not None:
                    exception_rows.append(exception_row)
                peak_rss = max(peak_rss, rss_bytes)
                completed += 1
                progress.update(
                    completed, _mesh_detail(records[current_idx], _status_label(row))
                )
                current_idx = None

                while next_pos < len(indices) and len(in_flight) < backlog:
                    next_idx = indices[next_pos]
                    next_pos += 1
                    in_flight[
                        executor.submit(
                            _qc_record_worker_profiled,
                            records[next_idx],
                            args.check_components,
                        )
                    ] = next_idx
    except Exception as exc:
        remaining = []
        if current_idx is not None:
            remaining.append(current_idx)
        remaining.extend(in_flight.values())
        remaining.extend(indices[next_pos:])
        dedup_remaining = list(dict.fromkeys(remaining))
        return completed, dedup_remaining, peak_rss, exc

    return completed, [], peak_rss, None


def _run_qc_detailed(
    records: list[MeshRecord],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object] | None] = [None] * len(records)
    exception_rows: list[dict[str, object]] = []
    progress = _make_progress_reporter(args, "QC", len(records))
    visible_workers = max(
        1, min(_resolve_worker_count(args.workers), len(records) or 1)
    )
    memory_budget_bytes = _resolve_memory_budget_bytes(visible_workers)
    _log_info(
        "QC",
        "Starting QC stage",
        meshes=len(records),
        visible_workers=visible_workers,
        memory_budget=_format_bytes(memory_budget_bytes),
    )

    completed, sampled_worker_rss_bytes, remaining_indices = _run_qc_warmup_samples(
        records,
        args,
        summary_rows,
        exception_rows,
        progress,
    )
    workers, worker_details = _choose_stage_worker_count(
        stage="QC",
        task_count=len(remaining_indices),
        visible_workers=visible_workers,
        memory_budget_bytes=memory_budget_bytes,
        sampled_worker_rss_bytes=sampled_worker_rss_bytes,
    )
    _log_info(
        "QC",
        "Selected worker count",
        workers=workers,
        sampled_worker_rss=_format_bytes(sampled_worker_rss_bytes),
        memory_limited=worker_details["limited_by_memory"],
        per_worker_budget=_format_bytes(worker_details["per_worker_budget_bytes"]),
    )

    recovery_mode = False
    while remaining_indices:
        if workers <= 1:
            if recovery_mode:
                _log_warning(
                    "QC",
                    "Falling back to isolated single-mesh execution",
                    remaining=len(remaining_indices),
                )
                print(
                    f"[QC] Falling back to isolated single-mesh execution for {len(remaining_indices)} mesh(es)",
                    flush=True,
                )
            for idx in remaining_indices:
                if recovery_mode:
                    row, exception_row, rss_bytes = _run_qc_isolated_once(
                        records[idx], args.check_components
                    )
                else:
                    row, exception_row, rss_bytes = _run_qc_direct_once(
                        records[idx], args.check_components
                    )
                summary_rows[idx] = row
                if exception_row is not None:
                    exception_rows.append(exception_row)
                completed += 1
                sampled_worker_rss_bytes = (
                    max(sampled_worker_rss_bytes or 0, rss_bytes)
                    or sampled_worker_rss_bytes
                )
                progress.update(
                    completed, _mesh_detail(records[idx], _status_label(row))
                )
            remaining_indices = []
            break

        print(f"[QC] Using {workers} worker processes", flush=True)
        completed, remaining_indices, attempt_peak_rss, attempt_error = (
            _run_qc_parallel_attempt(
                records,
                remaining_indices,
                args,
                workers,
                summary_rows,
                exception_rows,
                progress,
                completed,
            )
        )
        sampled_worker_rss_bytes = (
            max(sampled_worker_rss_bytes or 0, attempt_peak_rss)
            or sampled_worker_rss_bytes
        )
        if attempt_error is None:
            break

        pending = [_mesh_detail(records[idx]) for idx in remaining_indices[:10]]
        _log_error(
            "QC",
            "Parallel QC crashed; retrying with fewer workers",
            workers=workers,
            completed=completed,
            remaining=len(remaining_indices),
            error_type=type(attempt_error).__name__,
            error_message=str(attempt_error),
            pending=" || ".join(pending),
        )
        next_workers = max(1, workers // 2)
        print(
            f"[QC] Worker pool crashed at {workers} workers; retrying remaining {len(remaining_indices)} mesh(es) with {next_workers}",
            flush=True,
        )
        if workers == next_workers == 1 and remaining_indices:
            break
        recovery_mode = True
        workers = next_workers

    progress.complete()
    final_rows = [row for row in summary_rows if row is not None]
    _log_info(
        "QC",
        "Completed QC stage",
        meshes=len(final_rows),
        exceptions=len(exception_rows),
        failed=sum(1 for row in final_rows if row["status"] != "OK"),
    )
    return final_rows, exception_rows


def _run_qc(
    records: list[MeshRecord], args: argparse.Namespace
) -> list[dict[str, object]]:
    rows, _ = _run_qc_detailed(records, args)
    return rows


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


def _refresh_loaded_metadata(
    records: list[MeshRecord],
    summary_rows: list[dict[str, str]],
    *,
    root: Path,
    args: argparse.Namespace,
) -> tuple[list[MeshRecord], int]:
    refreshed_records: list[MeshRecord] = []
    force_roi = bool(args.roi_regex)
    force_subject = bool(args.subject_regex)
    force_repeat = bool(args.repeat_regex)
    changed = 0

    for record in records:
        inferred = infer_record(
            record.path,
            root=root,
            roi_regex=args.roi_regex,
            subject_regex=args.subject_regex,
            repeat_regex=args.repeat_regex,
        )
        roi = (
            inferred.roi
            if force_roi or record.roi in {"", "unknown_roi"}
            else record.roi
        )
        subject = (
            inferred.subject
            if force_subject or record.subject in {"", "unknown_subject"}
            else record.subject
        )
        repeat = (
            inferred.repeat
            if force_repeat or record.repeat in {"", "unknown_repeat"}
            else record.repeat
        )
        mesh_id = (
            inferred.mesh_id
            if record.mesh_id in {"", "unknown_mesh"}
            else record.mesh_id
        )
        refreshed = MeshRecord(
            path=record.path,
            roi=roi,
            subject=subject,
            repeat=repeat,
            mesh_id=mesh_id,
        )
        if refreshed != record:
            changed += 1
        refreshed_records.append(refreshed)

    records_by_path = {str(record.path): record for record in refreshed_records}
    for row in summary_rows:
        record = records_by_path.get(str(Path(row["path"]).expanduser()))
        if record is None:
            continue
        for field in ("mesh_id", "subject", "repeat", "roi"):
            value = getattr(record, field)
            if row.get(field) != value:
                row[field] = value

    return refreshed_records, changed


def _render_record_worker(
    record: MeshRecord,
    out_png: Path,
    *,
    image_size: int,
    renderer: str,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    label = f"{record.subject}\n{record.repeat}\n{record.mesh_id}"
    try:
        actual_renderer = (
            render_mesh_png(
                record.path,
                out_png,
                label=label,
                image_size=image_size,
                renderer=renderer,
            )
            or renderer
        )
        return None, _render_manifest_row(
            record,
            out_png,
            requested_renderer=renderer,
            actual_renderer=str(actual_renderer),
            resumed=False,
        )
    except Exception as exc:
        return (
            _exception_row(
                record,
                stage="render",
                exc=exc,
                traceback_text=traceback.format_exc(),
                output_path=out_png,
            ),
            None,
        )


def _render_manifest_row(
    record: MeshRecord,
    out_png: Path,
    *,
    requested_renderer: str,
    actual_renderer: str,
    resumed: bool,
) -> dict[str, object]:
    row = _record_row(record)
    row.update(
        {
            "output_path": str(out_png),
            "requested_renderer": requested_renderer,
            "actual_renderer": actual_renderer,
            "resumed": "1" if resumed else "0",
        }
    )
    return row


def _render_record_worker_profiled(
    record: MeshRecord,
    out_png: Path,
    image_size: int,
    renderer: str,
) -> tuple[dict[str, object] | None, dict[str, object] | None, int]:
    failure_row, manifest_row = _render_record_worker(
        record,
        out_png,
        image_size=image_size,
        renderer=renderer,
    )
    return failure_row, manifest_row, _maxrss_bytes()


def _render_worker_exit_payload(
    record: MeshRecord,
    out_png: Path,
    exitcode: int,
) -> tuple[dict[str, object], None, int]:
    message = (
        f"worker process exited before returning a render result (exitcode={exitcode})"
    )
    return (
        _exception_row(
            record,
            stage="render",
            exc=RuntimeError(message),
            traceback_text=message,
            output_path=out_png,
        ),
        None,
        0,
    )


def _run_render_isolated_once(
    record: MeshRecord,
    out_png: Path,
    *,
    image_size: int,
    renderer: str,
) -> tuple[dict[str, object] | None, dict[str, object] | None, int]:
    payload, exitcode = _run_isolated_worker(
        _render_record_worker_profiled,
        record,
        out_png,
        image_size,
        renderer,
    )
    if exitcode != 0 or payload is None:
        return _render_worker_exit_payload(record, out_png, exitcode)
    return payload


def _run_render_direct_once(
    record: MeshRecord,
    out_png: Path,
    *,
    image_size: int,
    renderer: str,
) -> tuple[dict[str, object] | None, dict[str, object] | None, int]:
    failure_row, manifest_row = _render_record_worker(
        record,
        out_png,
        image_size=image_size,
        renderer=renderer,
    )
    return failure_row, manifest_row, 0


def _run_render_warmup_samples(
    task_specs: list[tuple[int, MeshRecord, Path]],
    args: argparse.Namespace,
    rendered_slots: list[Path | None],
    failure_rows: list[dict[str, object]],
    manifest_rows: list[dict[str, object]],
    progress,
) -> tuple[int, int | None, list[tuple[int, MeshRecord, Path]]]:
    warmup_count = 0
    visible_workers = max(1, min(_resolve_worker_count(args.workers), len(task_specs)))
    if visible_workers > 1 and len(task_specs) >= 4:
        warmup_count = 1
    completed = 0
    peak_rss = 0
    for slot_idx, record, out_png in task_specs[:warmup_count]:
        failure_row, manifest_row, rss_bytes = _run_render_isolated_once(
            record,
            out_png,
            image_size=args.image_size,
            renderer=args.renderer,
        )
        if failure_row is not None:
            failure_rows.append(failure_row)
            progress.update(completed + 1, _mesh_detail(record, "FAILED"))
        else:
            rendered_slots[slot_idx] = out_png
            if manifest_row is not None:
                manifest_rows.append(manifest_row)
            progress.update(completed + 1, _mesh_detail(record))
        peak_rss = max(peak_rss, rss_bytes)
        completed += 1
    return completed, (peak_rss or None), task_specs[warmup_count:]


def _run_render_parallel_attempt(
    task_specs: list[tuple[int, MeshRecord, Path]],
    args: argparse.Namespace,
    workers: int,
    rendered_slots: list[Path | None],
    failure_rows: list[dict[str, object]],
    manifest_rows: list[dict[str, object]],
    progress,
    completed: int,
) -> tuple[int, list[tuple[int, MeshRecord, Path]], int, Exception | None]:
    in_flight: dict[object, tuple[int, MeshRecord, Path]] = {}
    next_pos = 0
    peak_rss = 0
    backlog = max(workers * 2, workers)
    current_task: tuple[int, MeshRecord, Path] | None = None
    try:
        with ProcessPoolExecutor(**_process_pool_executor_kwargs(workers)) as executor:
            while next_pos < len(task_specs) and len(in_flight) < backlog:
                slot_idx, record, out_png = task_specs[next_pos]
                next_pos += 1
                in_flight[
                    executor.submit(
                        _render_record_worker_profiled,
                        record,
                        out_png,
                        args.image_size,
                        args.renderer,
                    )
                ] = (slot_idx, record, out_png)

            while in_flight:
                future = next(iter(as_completed(tuple(in_flight))))
                current_task = in_flight.pop(future)
                slot_idx, record, out_png = current_task
                failure_row, manifest_row, rss_bytes = future.result()
                peak_rss = max(peak_rss, rss_bytes)
                completed += 1
                if failure_row is not None:
                    failure_rows.append(failure_row)
                    progress.update(completed, _mesh_detail(record, "FAILED"))
                else:
                    rendered_slots[slot_idx] = out_png
                    if manifest_row is not None:
                        manifest_rows.append(manifest_row)
                    progress.update(completed, _mesh_detail(record))
                current_task = None

                while next_pos < len(task_specs) and len(in_flight) < backlog:
                    next_slot_idx, next_record, next_out_png = task_specs[next_pos]
                    next_pos += 1
                    in_flight[
                        executor.submit(
                            _render_record_worker_profiled,
                            next_record,
                            next_out_png,
                            args.image_size,
                            args.renderer,
                        )
                    ] = (next_slot_idx, next_record, next_out_png)
    except Exception as exc:
        remaining = []
        if current_task is not None:
            remaining.append(current_task)
        remaining.extend(in_flight.values())
        remaining.extend(task_specs[next_pos:])
        dedup_remaining = list(dict.fromkeys(remaining))
        return completed, dedup_remaining, peak_rss, exc

    return completed, [], peak_rss, None


@contextmanager
def _render_display_context(args: argparse.Namespace):
    wants_gmsh = args.renderer in {"auto", "gmsh"}
    if not wants_gmsh:
        yield
        return

    if os.environ.get("MESH_QC_USE_EXISTING_DISPLAY") == "1" and os.environ.get(
        "DISPLAY"
    ):
        yield
        return

    session = VirtualDisplaySession(
        width=max(args.image_size, 1200),
        height=max(args.image_size, 1200),
    )
    try:
        display = session.start()
    except Exception as exc:
        error_message = _short_error(exc)
        if args.renderer == "gmsh":
            _log_error(
                "RENDER",
                "Failed to start Xvfb for gmsh renderer",
                error_message=error_message,
            )
            raise RuntimeError(
                "Gmsh rendering requires a DISPLAY or a working Xvfb installation. "
                f"Automatic Xvfb startup failed: {error_message}"
            ) from exc
        existing_display = os.environ.get("DISPLAY")
        if existing_display:
            _log_warning(
                "RENDER",
                "Could not start managed Xvfb; falling back to existing DISPLAY",
                display=existing_display,
                error_message=error_message,
            )
            print(
                "[RENDER] Could not start managed Xvfb; "
                f"falling back to existing DISPLAY={existing_display} ({error_message})",
                flush=True,
            )
            yield
            return
        _log_warning(
            "RENDER",
            "Could not start Xvfb for auto renderer",
            error_message=error_message,
        )
        print(
            "[RENDER] Could not start Xvfb for auto renderer; "
            f"Gmsh may be skipped and auto will fall back ({error_message})",
            flush=True,
        )
        yield
        return

    os.environ["DISPLAY"] = display
    print(f"[RENDER] Started virtual display {display} for Gmsh rendering", flush=True)
    _log_info("RENDER", "Started virtual display", display=display)
    try:
        yield
    finally:
        session.stop()
        _log_info("RENDER", "Stopped virtual display")


def _write_render_failure_outputs(
    out_dir: Path, failure_rows: list[dict[str, object]]
) -> None:
    _write_optional_csv(
        out_dir / "render_exception_details.csv", failure_rows, EXCEPTION_FIELDS
    )
    if failure_rows:
        fail_path = out_dir / "render_failures.txt"
        fail_lines = [
            f"{row['path']}\t{row['error_type']}: {row['error_message']}"
            for row in failure_rows
        ]
        fail_path.write_text("\n".join(fail_lines) + "\n", encoding="utf-8")
    else:
        try:
            (out_dir / "render_failures.txt").unlink()
        except FileNotFoundError:
            pass


def _write_render_completeness_outputs(
    out_dir: Path,
    *,
    records: list[MeshRecord],
    render_records: list[MeshRecord],
    successful: list[tuple[MeshRecord, Path | None]],
    failure_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    discovered = Counter(record.roi for record in records)
    qc_loadable = Counter(record.roi for record in render_records)
    rendered = Counter(
        record.roi for record, out_png in successful if out_png is not None
    )
    render_failures = Counter(
        str(row.get("roi", "unknown_roi")) for row in failure_rows
    )

    rois = sorted(
        set(discovered) | set(qc_loadable) | set(rendered) | set(render_failures)
    )
    rows = []
    for roi in rois:
        discovered_count = discovered[roi]
        qc_loadable_count = qc_loadable[roi]
        rendered_count = rendered[roi]
        qc_read_failures = max(0, discovered_count - qc_loadable_count)
        missing_tiles = max(0, discovered_count - rendered_count)
        rows.append(
            {
                "roi": roi,
                "status": "OK" if missing_tiles == 0 else "INCOMPLETE",
                "discovered_meshes": discovered_count,
                "qc_loadable_meshes": qc_loadable_count,
                "rendered_meshes": rendered_count,
                "qc_read_failures": qc_read_failures,
                "render_failures": render_failures[roi],
                "missing_tiles": missing_tiles,
            }
        )

    _write_csv(out_dir / "render_completeness.csv", rows, RENDER_COMPLETENESS_FIELDS)
    incomplete = [row for row in rows if row["status"] != "OK"]
    if incomplete:
        for row in incomplete:
            print(
                "[WARN] Incomplete mesh wall for "
                f"{row['roi']}: rendered {row['rendered_meshes']}/"
                f"{row['discovered_meshes']} tiles "
                f"(qc_read_failures={row['qc_read_failures']}, "
                f"render_failures={row['render_failures']}).",
                flush=True,
            )
        _log_warning(
            "RENDER",
            "One or more mesh walls are incomplete",
            details="; ".join(
                f"{row['roi']} {row['rendered_meshes']}/{row['discovered_meshes']}"
                for row in incomplete
            ),
            path=str(out_dir / "render_completeness.csv"),
        )
    return rows


def _run_forced_gmsh_preflight(
    task_specs: list[tuple[int, MeshRecord, Path]],
    args: argparse.Namespace,
    rendered_slots: list[Path | None],
    failure_rows: list[dict[str, object]],
    manifest_rows: list[dict[str, object]],
    progress,
    out_dir: Path,
) -> tuple[int, int | None, list[tuple[int, MeshRecord, Path]]]:
    if args.renderer != "gmsh" or not task_specs:
        return 0, None, task_specs

    slot_idx, record, out_png = task_specs[0]
    print(
        f"[RENDER] Running forced-Gmsh preflight on {_mesh_detail(record)}", flush=True
    )
    failure_row, manifest_row, rss_bytes = _run_render_isolated_once(
        record,
        out_png,
        image_size=args.image_size,
        renderer=args.renderer,
    )
    if failure_row is not None:
        failure_rows.append(failure_row)
        progress.update(1, _mesh_detail(record, "FAILED"))
        _write_render_failure_outputs(out_dir, failure_rows)
        _log_error(
            "RENDER",
            "Forced Gmsh preflight failed",
            mesh=_mesh_detail(record),
            output_path=str(out_png),
            error_type=failure_row["error_type"],
            error_message=failure_row["error_message"],
        )
        raise RuntimeError(
            "Forced Gmsh render preflight failed for "
            f"{record.path}. First error: {failure_row['error_type']}: {failure_row['error_message']}. "
            f"See {out_dir / 'render_exception_details.csv'}."
        )

    rendered_slots[slot_idx] = out_png
    if manifest_row is not None:
        manifest_rows.append(manifest_row)
    progress.update(1, _mesh_detail(record))
    return 1, (rss_bytes or None), task_specs[1:]


def _tissue_metadata_row(record: MeshRecord, tissue) -> dict[str, object]:
    row = _record_row(record)
    row.update(
        {
            "tissue_tag": tissue.tag,
            "tissue_name": tissue.name,
            "tissue_slug": tissue.slug,
        }
    )
    return row


def _tissue_exception_row(
    record: MeshRecord,
    *,
    stage: str,
    exc: Exception,
    traceback_text: str,
    output_path: Path | None = None,
    tissue=None,
    view: str = "",
) -> dict[str, object]:
    row = _exception_row(
        record,
        stage=stage,
        exc=exc,
        traceback_text=traceback_text,
        output_path=output_path,
    )
    row.update(
        {
            "tissue_tag": "" if tissue is None else tissue.tag,
            "tissue_name": "" if tissue is None else tissue.name,
            "tissue_slug": "" if tissue is None else tissue.slug,
            "view": view,
        }
    )
    return row


def _tissue_manifest_row(
    record: MeshRecord,
    tissue,
    out_png: Path,
    *,
    view: str,
    requested_renderer: str,
    actual_renderer: str,
    resumed: bool,
) -> dict[str, object]:
    row = _tissue_metadata_row(record, tissue)
    row.update(
        {
            "view": view,
            "output_path": str(out_png),
            "requested_renderer": requested_renderer,
            "actual_renderer": actual_renderer,
            "resumed": "1" if resumed else "0",
        }
    )
    return row


def _tissue_render_path(
    tissue_root: Path,
    tissue_slug: str,
    view: str,
    tile_filename: str,
) -> Path:
    if view == "front":
        view_root = tissue_root
    elif view == "back":
        view_root = tissue_root.parent / "tissues_back"
    elif view == "top":
        view_root = tissue_root.parent / "tissues_top"
    else:
        raise ValueError(f"Unsupported tissue view: {view}")
    return view_root / tissue_slug / tile_filename


def _views_for_tissue_tag(tag: int) -> tuple[str, ...]:
    if int(tag) == COMPACT_BONE_TAG:
        return TISSUE_VIEWS
    return STANDARD_TISSUE_VIEWS


def _tissue_record_worker(
    record: MeshRecord,
    tile_filename: str,
    tissue_root: Path,
    image_size: int,
    renderer: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    manifest_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    presence_rows: list[dict[str, object]] = []
    try:
        for tissue in iter_tissue_surface_arrays(record.path):
            presence_rows.append(_tissue_metadata_row(record, tissue))
            for view in _views_for_tissue_tag(tissue.tag):
                out_png = _tissue_render_path(
                    tissue_root,
                    tissue.slug,
                    view,
                    tile_filename,
                )
                try:
                    if out_png.is_file() and out_png.stat().st_size > 0:
                        manifest_rows.append(
                            _tissue_manifest_row(
                                record,
                                tissue,
                                out_png,
                                view=view,
                                requested_renderer=renderer,
                                actual_renderer="unknown_existing",
                                resumed=True,
                            )
                        )
                        continue
                except OSError:
                    pass

                label = (
                    f"{record.subject}\n{record.repeat}\n{record.mesh_id}\n"
                    f"Tag {tissue.tag}: {tissue.name}\n{view.title()} view"
                )
                try:
                    actual_renderer = (
                        render_surface_png(
                            tissue.surface,
                            out_png,
                            label=label,
                            view=view,
                            image_size=image_size,
                            renderer=renderer,
                        )
                        or renderer
                    )
                    manifest_rows.append(
                        _tissue_manifest_row(
                            record,
                            tissue,
                            out_png,
                            view=view,
                            requested_renderer=renderer,
                            actual_renderer=str(actual_renderer),
                            resumed=False,
                        )
                    )
                except Exception as exc:
                    failure_rows.append(
                        _tissue_exception_row(
                            record,
                            stage="tissue_render",
                            exc=exc,
                            traceback_text=traceback.format_exc(),
                            output_path=out_png,
                            tissue=tissue,
                            view=view,
                        )
                    )
    except Exception as exc:
        failure_rows.append(
            _tissue_exception_row(
                record,
                stage="tissue_load",
                exc=exc,
                traceback_text=traceback.format_exc(),
                output_path=tissue_root,
            )
        )
    return manifest_rows, failure_rows, presence_rows


def _tissue_record_worker_profiled(
    record: MeshRecord,
    tile_filename: str,
    tissue_root: Path,
    image_size: int,
    renderer: str,
) -> tuple[
    list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], int
]:
    manifest_rows, failure_rows, presence_rows = _tissue_record_worker(
        record,
        tile_filename,
        tissue_root,
        image_size,
        renderer,
    )
    return manifest_rows, failure_rows, presence_rows, _maxrss_bytes()


def _tissue_worker_exit_payload(
    record: MeshRecord,
    tissue_root: Path,
    exitcode: int,
) -> tuple[
    list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], int
]:
    message = (
        f"worker process exited before returning tissue renders (exitcode={exitcode})"
    )
    failure_row = _tissue_exception_row(
        record,
        stage="tissue_render_worker",
        exc=RuntimeError(message),
        traceback_text=message,
        output_path=tissue_root,
    )
    return [], [failure_row], [], 0


def _run_tissue_isolated_once(
    task: tuple[MeshRecord, str],
    *,
    tissue_root: Path,
    image_size: int,
    renderer: str,
):
    record, tile_filename = task
    payload, exitcode = _run_isolated_worker(
        _tissue_record_worker_profiled,
        record,
        tile_filename,
        tissue_root,
        image_size,
        renderer,
    )
    if exitcode != 0 or payload is None:
        return _tissue_worker_exit_payload(record, tissue_root, exitcode)
    return payload


def _merge_tissue_worker_result(
    result,
    manifest_rows: list[dict[str, object]],
    failure_rows: list[dict[str, object]],
    presence_rows: list[dict[str, object]],
) -> int:
    result_manifest, result_failures, result_presence, rss_bytes = result
    manifest_rows.extend(result_manifest)
    failure_rows.extend(result_failures)
    presence_rows.extend(result_presence)
    return rss_bytes


def _run_tissue_parallel_attempt(
    task_specs: list[tuple[MeshRecord, str]],
    args: argparse.Namespace,
    workers: int,
    tissue_root: Path,
    manifest_rows: list[dict[str, object]],
    failure_rows: list[dict[str, object]],
    presence_rows: list[dict[str, object]],
    progress,
    completed: int,
) -> tuple[int, list[tuple[MeshRecord, str]], int, Exception | None]:
    in_flight: dict[object, tuple[MeshRecord, str]] = {}
    next_pos = 0
    peak_rss = 0
    backlog = max(workers * 2, workers)
    current_task: tuple[MeshRecord, str] | None = None
    try:
        with ProcessPoolExecutor(**_process_pool_executor_kwargs(workers)) as executor:
            while next_pos < len(task_specs) and len(in_flight) < backlog:
                task = task_specs[next_pos]
                next_pos += 1
                record, tile_filename = task
                in_flight[
                    executor.submit(
                        _tissue_record_worker_profiled,
                        record,
                        tile_filename,
                        tissue_root,
                        args.image_size,
                        args.renderer,
                    )
                ] = task

            while in_flight:
                future = next(iter(as_completed(tuple(in_flight))))
                current_task = in_flight.pop(future)
                record, _ = current_task
                result = future.result()
                peak_rss = max(
                    peak_rss,
                    _merge_tissue_worker_result(
                        result,
                        manifest_rows,
                        failure_rows,
                        presence_rows,
                    ),
                )
                completed += 1
                result_failures = result[1]
                detail = f"{len(result[2])} tissue(s)"
                if result_failures:
                    detail += f", {len(result_failures)} failure(s)"
                progress.update(completed, _mesh_detail(record, detail))
                current_task = None

                while next_pos < len(task_specs) and len(in_flight) < backlog:
                    next_task = task_specs[next_pos]
                    next_pos += 1
                    next_record, next_tile_filename = next_task
                    in_flight[
                        executor.submit(
                            _tissue_record_worker_profiled,
                            next_record,
                            next_tile_filename,
                            tissue_root,
                            args.image_size,
                            args.renderer,
                        )
                    ] = next_task
    except Exception as exc:
        remaining = []
        if current_task is not None:
            remaining.append(current_task)
        remaining.extend(in_flight.values())
        remaining.extend(task_specs[next_pos:])
        return completed, list(dict.fromkeys(remaining)), peak_rss, exc
    return completed, [], peak_rss, None


def _write_tissue_completeness(
    out_dir: Path,
    *,
    render_records: list[MeshRecord],
    presence_rows: list[dict[str, object]],
    manifest_rows: list[dict[str, object]],
    failure_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    metadata_by_tag: dict[int, tuple[str, str]] = {}
    present: set[tuple[int, str]] = set()
    rendered: set[tuple[int, str, str]] = set()
    failures = Counter()

    for row in presence_rows:
        tag = int(row["tissue_tag"])
        metadata_by_tag[tag] = (str(row["tissue_name"]), str(row["tissue_slug"]))
        present.add((tag, str(row["path"])))
    for row in manifest_rows:
        rendered.add((int(row["tissue_tag"]), str(row["path"]), str(row["view"])))
    for row in failure_rows:
        raw_tag = row.get("tissue_tag", "")
        view = str(row.get("view", ""))
        if raw_tag not in ("", None) and view in TISSUE_VIEWS:
            failures[(int(raw_tag), view)] += 1

    rows: list[dict[str, object]] = []
    total_meshes = len(render_records)
    for tag in sorted(metadata_by_tag):
        name, slug = metadata_by_tag[tag]
        present_count = sum(1 for present_tag, _ in present if present_tag == tag)
        for view in _views_for_tissue_tag(tag):
            rendered_count = sum(
                1
                for rendered_tag, _, rendered_view in rendered
                if rendered_tag == tag and rendered_view == view
            )
            missing_from_meshes = max(0, total_meshes - present_count)
            missing_tiles = max(0, present_count - rendered_count)
            if missing_from_meshes and missing_tiles:
                status = "MISSING_TISSUE_AND_INCOMPLETE_RENDER"
            elif missing_from_meshes:
                status = "MISSING_TISSUE"
            elif missing_tiles:
                status = "INCOMPLETE_RENDER"
            else:
                status = "OK"
            rows.append(
                {
                    "tissue_tag": tag,
                    "tissue_name": name,
                    "tissue_slug": slug,
                    "view": view,
                    "status": status,
                    "qc_loadable_meshes": total_meshes,
                    "present_meshes": present_count,
                    "rendered_meshes": rendered_count,
                    "missing_from_meshes": missing_from_meshes,
                    "render_failures": failures[(tag, view)],
                    "missing_tiles": missing_tiles,
                }
            )
    _write_csv(
        out_dir / "tissue_render_completeness.csv",
        rows,
        TISSUE_RENDER_COMPLETENESS_FIELDS,
    )
    return rows


def _prepare_tissue_view_convention_unlocked(out_dir: Path) -> None:
    marker_path = out_dir / TISSUE_VIEW_CONVENTION_FILENAME
    if marker_path.is_file():
        try:
            existing = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"Could not read tissue view marker {marker_path}: {exc}"
            ) from exc
        if existing == TISSUE_VIEW_CONVENTION:
            return
        if existing == LEGACY_TISSUE_VIEW_CONVENTION:
            top_root = out_dir / "renders" / "tissues_top"
            if top_root.is_dir():
                for png_path in top_root.rglob("*.png"):
                    try:
                        if png_path.is_file() and png_path.stat().st_size > 0:
                            raise RuntimeError(
                                f"Existing top-view tissue tile {png_path} is not covered by "
                                "the legacy front/back marker. Use a fresh output directory."
                            )
                    except OSError:
                        continue
            marker_path.write_text(
                json.dumps(TISSUE_VIEW_CONVENTION, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return
        raise RuntimeError(
            f"Tissue renders in {out_dir} use a different view convention. "
            "Use a fresh output directory to avoid mixing incompatible tiles."
        )

    for tile_root in (
        out_dir / "renders" / "tissues",
        out_dir / "renders" / "tissues_back",
        out_dir / "renders" / "tissues_top",
    ):
        if not tile_root.is_dir():
            continue
        for png_path in tile_root.rglob("*.png"):
            try:
                if png_path.is_file() and png_path.stat().st_size > 0:
                    raise RuntimeError(
                        f"Existing tissue tile {png_path} predates the current "
                        "anatomical view convention. Use a fresh output directory; these "
                        "tiles must not be resumed."
                    )
            except OSError:
                continue

    out_dir.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps(TISSUE_VIEW_CONVENTION, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _prepare_tissue_view_convention(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / ".tissue_view_convention.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            _prepare_tissue_view_convention_unlocked(out_dir)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _build_tissue_task_specs(
    records: list[MeshRecord],
) -> list[tuple[MeshRecord, str]]:
    task_specs: list[tuple[MeshRecord, str]] = []
    for idx, record in enumerate(records, start=1):
        stem = "__".join(
            _safe_name(part)
            for part in (
                record.subject,
                record.repeat,
                record.mesh_id,
                record.path.stem,
            )
        )
        task_specs.append((record, f"{idx:05d}__{stem}.png"))
    return task_specs


def _tissue_shard_report_dir(
    out_dir: Path,
    *,
    shard_count: int,
    shard_index: int,
) -> Path:
    return (
        out_dir
        / "shards"
        / "tissue_render"
        / f"shard_{shard_index:05d}_of_{shard_count:05d}"
    )


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _run_tissue_outputs(
    render_records: list[MeshRecord],
    out_dir: Path,
    args: argparse.Namespace,
    *,
    task_specs: list[tuple[MeshRecord, str]] | None = None,
    report_dir: Path | None = None,
    build_mosaics: bool = True,
) -> dict[str, int]:
    if not render_records:
        return {"meshes": 0, "renders": 0, "failures": 0, "walls": 0, "incomplete": 0}
    _prepare_tissue_view_convention(out_dir)
    report_dir = out_dir if report_dir is None else report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    tissue_root = out_dir / "renders" / "tissues"
    if task_specs is None:
        task_specs = _build_tissue_task_specs(render_records)
    if len(task_specs) != len(render_records):
        raise ValueError(
            "tissue task specification count differs from render record count"
        )

    visible_workers = max(1, min(_resolve_worker_count(args.workers), len(task_specs)))
    memory_budget_bytes = _resolve_memory_budget_bytes(visible_workers)
    manifest_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    presence_rows: list[dict[str, object]] = []
    progress = _make_progress_reporter(args, "Tissue render", len(task_specs))
    _log_info(
        "TISSUE_RENDER",
        "Starting tissue wall stage",
        meshes=len(task_specs),
        visible_workers=visible_workers,
        memory_budget=_format_bytes(memory_budget_bytes),
        renderer=args.renderer,
    )

    completed = 0
    sampled_worker_rss_bytes: int | None = None
    remaining_specs = task_specs
    with _render_display_context(args):
        if visible_workers > 1 and len(task_specs) >= 4:
            first_task = task_specs[0]
            result = _run_tissue_isolated_once(
                first_task,
                tissue_root=tissue_root,
                image_size=args.image_size,
                renderer=args.renderer,
            )
            sampled_worker_rss_bytes = (
                _merge_tissue_worker_result(
                    result,
                    manifest_rows,
                    failure_rows,
                    presence_rows,
                )
                or None
            )
            if args.renderer == "gmsh" and result[1]:
                _write_csv(
                    report_dir / "tissue_render_exception_details.csv",
                    failure_rows,
                    TISSUE_EXCEPTION_FIELDS,
                )
                first_failure = result[1][0]
                raise RuntimeError(
                    "Forced Gmsh tissue-render preflight failed for "
                    f"{first_task[0].path}: {first_failure['error_type']}: "
                    f"{first_failure['error_message']}. See "
                    f"{report_dir / 'tissue_render_exception_details.csv'}."
                )
            completed = 1
            progress.update(
                completed, _mesh_detail(first_task[0], f"{len(result[2])} tissue(s)")
            )
            remaining_specs = task_specs[1:]

        workers, worker_details = _choose_stage_worker_count(
            stage="TISSUE_RENDER",
            task_count=len(remaining_specs),
            visible_workers=visible_workers,
            memory_budget_bytes=memory_budget_bytes,
            sampled_worker_rss_bytes=sampled_worker_rss_bytes,
        )
        _log_info(
            "TISSUE_RENDER",
            "Selected worker count",
            workers=workers,
            sampled_worker_rss=_format_bytes(sampled_worker_rss_bytes),
            memory_limited=worker_details["limited_by_memory"],
            per_worker_budget=_format_bytes(worker_details["per_worker_budget_bytes"]),
        )

        recovery_mode = False
        while remaining_specs:
            if workers <= 1:
                if recovery_mode:
                    print(
                        "[TISSUE RENDER] Falling back to isolated single-mesh execution for "
                        f"{len(remaining_specs)} mesh(es)",
                        flush=True,
                    )
                for task in remaining_specs:
                    if recovery_mode:
                        result = _run_tissue_isolated_once(
                            task,
                            tissue_root=tissue_root,
                            image_size=args.image_size,
                            renderer=args.renderer,
                        )
                    else:
                        record, tile_filename = task
                        result_manifest, result_failures, result_presence = (
                            _tissue_record_worker(
                                record,
                                tile_filename,
                                tissue_root,
                                args.image_size,
                                args.renderer,
                            )
                        )
                        result = (result_manifest, result_failures, result_presence, 0)
                    sampled_worker_rss_bytes = (
                        max(
                            sampled_worker_rss_bytes or 0,
                            _merge_tissue_worker_result(
                                result,
                                manifest_rows,
                                failure_rows,
                                presence_rows,
                            ),
                        )
                        or sampled_worker_rss_bytes
                    )
                    completed += 1
                    detail = f"{len(result[2])} tissue(s)"
                    if result[1]:
                        detail += f", {len(result[1])} failure(s)"
                    progress.update(completed, _mesh_detail(task[0], detail))
                remaining_specs = []
                break

            print(f"[TISSUE RENDER] Using {workers} worker processes", flush=True)
            completed, remaining_specs, peak_rss, attempt_error = (
                _run_tissue_parallel_attempt(
                    remaining_specs,
                    args,
                    workers,
                    tissue_root,
                    manifest_rows,
                    failure_rows,
                    presence_rows,
                    progress,
                    completed,
                )
            )
            sampled_worker_rss_bytes = (
                max(sampled_worker_rss_bytes or 0, peak_rss) or sampled_worker_rss_bytes
            )
            if attempt_error is None:
                break
            next_workers = max(1, workers // 2)
            _log_error(
                "TISSUE_RENDER",
                "Parallel tissue rendering crashed; retrying with fewer workers",
                workers=workers,
                completed=completed,
                remaining=len(remaining_specs),
                next_workers=next_workers,
                error_type=type(attempt_error).__name__,
                error_message=str(attempt_error),
            )
            print(
                f"[TISSUE RENDER] Worker pool crashed at {workers} workers; "
                f"retrying {len(remaining_specs)} mesh(es) with {next_workers}",
                flush=True,
            )
            recovery_mode = True
            workers = next_workers
    progress.complete()

    presence_rows.sort(key=lambda row: (int(row["tissue_tag"]), str(row["path"])))
    manifest_rows.sort(
        key=lambda row: (
            int(row["tissue_tag"]),
            TISSUE_VIEWS.index(str(row["view"])),
            str(row["output_path"]),
        )
    )
    _write_csv(
        report_dir / "tissue_presence.csv", presence_rows, TISSUE_PRESENCE_FIELDS
    )
    _write_csv(
        report_dir / "tissue_render_manifest.csv",
        manifest_rows,
        TISSUE_RENDER_MANIFEST_FIELDS,
    )
    _write_csv(
        report_dir / "tissue_render_exception_details.csv",
        failure_rows,
        TISSUE_EXCEPTION_FIELDS,
    )
    completeness_rows = _write_tissue_completeness(
        report_dir,
        render_records=render_records,
        presence_rows=presence_rows,
        manifest_rows=manifest_rows,
        failure_rows=failure_rows,
    )
    if not presence_rows:
        _log_error(
            "TISSUE_RENDER",
            "No tagged tetrahedral tissues could be extracted",
            meshes=len(render_records),
            failures=len(failure_rows),
            details_path=str(report_dir / "tissue_render_exception_details.csv"),
        )
        raise RuntimeError(
            "No tagged tetrahedral tissues could be extracted from any loadable mesh. "
            f"See {report_dir / 'tissue_render_exception_details.csv'}."
        )

    images_by_tissue: dict[tuple[int, str, str, str], list[Path]] = defaultdict(list)
    for row in manifest_rows:
        key = (
            int(row["tissue_tag"]),
            str(row["tissue_name"]),
            str(row["tissue_slug"]),
            str(row["view"]),
        )
        images_by_tissue[key].append(Path(str(row["output_path"])))

    mosaic_failures: list[dict[str, object]] = []
    if build_mosaics:
        for (tag, name, slug, view), images in sorted(
            images_by_tissue.items(),
            key=lambda item: (item[0][0], TISSUE_VIEWS.index(item[0][3])),
        ):
            view_suffix = {"front": "", "back": "_back", "top": "_top"}[view]
            out_png = out_dir / "mosaics" / "tissues" / f"{slug}{view_suffix}_wall.png"
            print(
                f"[MOSAIC] Building {view} tissue wall for tag {tag} {name} "
                f"({len(images)} tiles)",
                flush=True,
            )
            try:
                make_mosaic(images, out_png, cols=args.cols, tile_size=args.tile_size)
            except Exception as exc:
                mosaic_failures.append(
                    {
                        "stage": "tissue_mosaic",
                        "subject": "",
                        "repeat": "",
                        "roi": "",
                        "mesh_id": slug,
                        "path": "",
                        "output_path": str(out_png),
                        "error_type": type(exc).__name__,
                        "error_message": _short_error(exc),
                        "traceback": traceback.format_exc(),
                        "tissue_tag": tag,
                        "tissue_name": name,
                        "tissue_slug": slug,
                        "view": view,
                    }
                )
        _write_csv(
            out_dir / "tissue_mosaic_exception_details.csv",
            mosaic_failures,
            TISSUE_EXCEPTION_FIELDS,
        )
    incomplete = [row for row in completeness_rows if row["status"] != "OK"]
    if incomplete:
        print(
            f"[TISSUE RENDER] {len(incomplete)} tissue wall(s) are incomplete; "
            f"see {report_dir / 'tissue_render_completeness.csv'}",
            flush=True,
        )
    _log_info(
        "TISSUE_RENDER",
        "Completed tissue wall stage",
        meshes=len(render_records),
        tissues=len({key[0] for key in images_by_tissue}),
        walls=len(images_by_tissue) if build_mosaics else 0,
        renders=len(manifest_rows),
        failures=len(failure_rows),
        incomplete_tissues=len(incomplete),
    )
    if mosaic_failures:
        raise RuntimeError(
            f"{len(mosaic_failures)} tissue mosaic(s) failed; see "
            f"{out_dir / 'tissue_mosaic_exception_details.csv'}"
        )
    return {
        "meshes": len(render_records),
        "renders": len(manifest_rows),
        "failures": len(failure_rows),
        "walls": len(images_by_tissue) if build_mosaics else 0,
        "incomplete": len(incomplete),
    }


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
        if not str(rows_by_path.get(str(record.path), {}).get("flags", "")).startswith(
            "READ_FAIL"
        )
    ]
    skipped = len(records) - len(render_records)
    if skipped:
        print(
            f"[RENDER] Skipping {skipped} mesh(es) that failed QC loading", flush=True
        )
    if not render_records:
        print("[RENDER] No QC-loadable meshes to render", flush=True)
        _write_optional_csv(
            out_dir / "render_exception_details.csv", [], EXCEPTION_FIELDS
        )
        return

    visible_workers = max(
        1, min(_resolve_worker_count(args.workers), len(render_records))
    )
    memory_budget_bytes = _resolve_memory_budget_bytes(visible_workers)
    task_specs = []
    for idx, record in enumerate(render_records, start=1):
        stem = "__".join(
            _safe_name(part)
            for part in (
                record.subject,
                record.repeat,
                record.mesh_id,
                record.path.stem,
            )
        )
        out_png = out_dir / "renders" / "meshes" / f"{idx:05d}__{stem}.png"
        task_specs.append((idx - 1, record, out_png))

    rendered_slots: list[Path | None] = [None] * len(task_specs)
    manifest_rows: list[dict[str, object]] = []
    remaining_task_specs = []
    resumed_count = 0
    for slot_idx, record, out_png in task_specs:
        try:
            if out_png.is_file() and out_png.stat().st_size > 0:
                rendered_slots[slot_idx] = out_png
                manifest_rows.append(
                    _render_manifest_row(
                        record,
                        out_png,
                        requested_renderer=args.renderer,
                        actual_renderer="unknown_existing",
                        resumed=True,
                    )
                )
                resumed_count += 1
                continue
        except OSError:
            pass
        remaining_task_specs.append((slot_idx, record, out_png))
    if resumed_count:
        print(f"[RENDER] Reusing {resumed_count} existing render(s)", flush=True)
    failure_rows: list[dict[str, object]] = []
    progress = _make_progress_reporter(args, "Render", len(render_records))
    _log_info(
        "RENDER",
        "Starting render stage",
        meshes=len(render_records),
        existing_renders=resumed_count,
        pending_renders=len(remaining_task_specs),
        skipped_qc_read_failures=skipped,
        visible_workers=visible_workers,
        memory_budget=_format_bytes(memory_budget_bytes),
        renderer=args.renderer,
    )

    with _render_display_context(args):
        completed = resumed_count
        if resumed_count:
            progress.update(completed, f"reused {resumed_count} existing render(s)")

        preflight_completed, preflight_rss_bytes, remaining_after_preflight = (
            _run_forced_gmsh_preflight(
                remaining_task_specs,
                args,
                rendered_slots,
                failure_rows,
                manifest_rows,
                progress,
                out_dir,
            )
        )
        completed += preflight_completed

        warmup_completed, sampled_worker_rss_bytes, remaining_specs = (
            _run_render_warmup_samples(
                remaining_after_preflight,
                args,
                rendered_slots,
                failure_rows,
                manifest_rows,
                progress,
            )
        )
        completed += warmup_completed
        sampled_worker_rss_bytes = (
            max(preflight_rss_bytes or 0, sampled_worker_rss_bytes or 0) or None
        )
        workers, worker_details = _choose_stage_worker_count(
            stage="RENDER",
            task_count=len(remaining_specs),
            visible_workers=visible_workers,
            memory_budget_bytes=memory_budget_bytes,
            sampled_worker_rss_bytes=sampled_worker_rss_bytes,
        )
        _log_info(
            "RENDER",
            "Selected worker count",
            workers=workers,
            sampled_worker_rss=_format_bytes(sampled_worker_rss_bytes),
            memory_limited=worker_details["limited_by_memory"],
            per_worker_budget=_format_bytes(worker_details["per_worker_budget_bytes"]),
        )

        recovery_mode = False
        while remaining_specs:
            if workers <= 1:
                if recovery_mode:
                    _log_warning(
                        "RENDER",
                        "Falling back to isolated single-mesh execution",
                        remaining=len(remaining_specs),
                    )
                    print(
                        f"[RENDER] Falling back to isolated single-mesh execution for {len(remaining_specs)} mesh(es)",
                        flush=True,
                    )
                for slot_idx, record, out_png in remaining_specs:
                    if recovery_mode:
                        failure_row, manifest_row, rss_bytes = (
                            _run_render_isolated_once(
                                record,
                                out_png,
                                image_size=args.image_size,
                                renderer=args.renderer,
                            )
                        )
                    else:
                        failure_row, manifest_row, rss_bytes = _run_render_direct_once(
                            record,
                            out_png,
                            image_size=args.image_size,
                            renderer=args.renderer,
                        )
                    sampled_worker_rss_bytes = (
                        max(sampled_worker_rss_bytes or 0, rss_bytes)
                        or sampled_worker_rss_bytes
                    )
                    completed += 1
                    if failure_row is not None:
                        failure_rows.append(failure_row)
                        progress.update(completed, _mesh_detail(record, "FAILED"))
                    else:
                        rendered_slots[slot_idx] = out_png
                        if manifest_row is not None:
                            manifest_rows.append(manifest_row)
                        progress.update(completed, _mesh_detail(record))
                remaining_specs = []
                break

            print(f"[RENDER] Using {workers} worker processes", flush=True)
            completed, remaining_specs, attempt_peak_rss, attempt_error = (
                _run_render_parallel_attempt(
                    remaining_specs,
                    args,
                    workers,
                    rendered_slots,
                    failure_rows,
                    manifest_rows,
                    progress,
                    completed,
                )
            )
            sampled_worker_rss_bytes = (
                max(sampled_worker_rss_bytes or 0, attempt_peak_rss)
                or sampled_worker_rss_bytes
            )
            if attempt_error is None:
                break

            pending = [_mesh_detail(record) for _, record, _ in remaining_specs[:10]]
            _log_error(
                "RENDER",
                "Parallel render crashed; retrying with fewer workers",
                workers=workers,
                completed=completed,
                remaining=len(remaining_specs),
                renderer=args.renderer,
                error_type=type(attempt_error).__name__,
                error_message=str(attempt_error),
                pending=" || ".join(pending),
            )
            next_workers = max(1, workers // 2)
            print(
                f"[RENDER] Worker pool crashed at {workers} workers; retrying remaining {len(remaining_specs)} mesh(es) with {next_workers}",
                flush=True,
            )
            recovery_mode = True
            workers = next_workers

    progress.complete()

    successful = [
        (record, rendered_slots[slot_idx])
        for slot_idx, record, _ in task_specs
        if rendered_slots[slot_idx] is not None
    ]
    rendered_by_roi: dict[str, list[Path]] = defaultdict(list)
    all_rendered: list[Path] = []
    for record, out_png in successful:
        rendered_by_roi[record.roi].append(out_png)
        all_rendered.append(out_png)

    _write_render_completeness_outputs(
        out_dir,
        records=records,
        render_records=render_records,
        successful=successful,
        failure_rows=failure_rows,
    )

    manifest_rows = sorted(manifest_rows, key=lambda row: str(row["output_path"]))
    _write_render_failure_outputs(out_dir, failure_rows)
    _write_optional_csv(
        out_dir / "render_manifest.csv", manifest_rows, RENDER_MANIFEST_FIELDS
    )
    renderer_counts = Counter(str(row["actual_renderer"]) for row in manifest_rows)
    if renderer_counts:
        _log_info(
            "RENDER",
            "Render manifest written",
            path=str(out_dir / "render_manifest.csv"),
            renderer_counts=" ".join(
                f"{key}:{renderer_counts[key]}" for key in sorted(renderer_counts)
            ),
        )

    mosaic_exception_rows: list[dict[str, object]] = []

    def build_mosaic(name: str, images: list[Path], out_png: Path) -> None:
        try:
            make_mosaic(
                images,
                out_png,
                cols=args.cols,
                tile_size=args.tile_size,
            )
        except Exception as exc:
            tb = traceback.format_exc()
            mosaic_exception_rows.append(
                {
                    "stage": "mosaic",
                    "subject": "",
                    "repeat": "",
                    "roi": name,
                    "mesh_id": name,
                    "path": "",
                    "output_path": str(out_png),
                    "error_type": type(exc).__name__,
                    "error_message": _short_error(exc),
                    "traceback": tb,
                }
            )
            _log_error(
                "MOSAIC",
                "Mosaic assembly failed",
                name=name,
                output_path=str(out_png),
                tiles=len(images),
                error_type=type(exc).__name__,
                error_message=_short_error(exc),
            )
            print(f"[MOSAIC] FAILED {name}: {_short_error(exc, limit=240)}", flush=True)

    if args.roi_walls:
        for roi, images in sorted(rendered_by_roi.items()):
            print(
                f"[MOSAIC] Building ROI wall for {roi} ({len(images)} tiles)",
                flush=True,
            )
            build_mosaic(
                roi,
                images,
                out_dir / "mosaics" / f"{_safe_name(roi)}_wall.png",
            )

    if all_rendered:
        print(
            f"[MOSAIC] Building combined mesh wall ({len(all_rendered)} tiles)",
            flush=True,
        )
        build_mosaic(
            "all",
            all_rendered,
            out_dir / "mosaics" / "all_mesh_wall.png",
        )

    _write_optional_csv(
        out_dir / "mosaic_exception_details.csv",
        mosaic_exception_rows,
        EXCEPTION_FIELDS,
    )
    if mosaic_exception_rows:
        failed_names = ", ".join(str(row["roi"]) for row in mosaic_exception_rows[:8])
        if len(mosaic_exception_rows) > 8:
            failed_names += f", ... ({len(mosaic_exception_rows)} total)"
        raise RuntimeError(
            "One or more mosaic assemblies failed after per-mesh rendering completed: "
            f"{failed_names}. See {out_dir / 'mosaic_exception_details.csv'}."
        )

    if args.tissue_walls:
        _run_tissue_outputs(render_records, out_dir, args)

    _log_info(
        "RENDER",
        "Completed render stage",
        attempted=len(render_records),
        rendered=len(all_rendered),
        failures=len(failure_rows),
        roi_walls=args.roi_walls,
        tissue_walls=args.tissue_walls,
    )


def _validate_stage_args(args: argparse.Namespace) -> None:
    if args.render_only and (args.qc_only or args.skip_renders):
        raise ValueError(
            "--render-only cannot be combined with --qc-only or --skip-renders"
        )
    if args.tissue_only and (args.render_only or args.qc_only or args.skip_renders):
        raise ValueError(
            "--tissue-only cannot be combined with --render-only, --qc-only, or --skip-renders"
        )
    if args.tissue_collect_only and (
        args.tissue_only or args.render_only or args.qc_only or args.skip_renders
    ):
        raise ValueError(
            "--tissue-collect-only cannot be combined with another stage selector"
        )
    if args.tissue_shard_count < 1:
        raise ValueError("--tissue-shard-count must be at least 1")
    if args.tissue_shard_index is not None and not (
        0 <= args.tissue_shard_index < args.tissue_shard_count
    ):
        raise ValueError(
            "--tissue-shard-index must be in the range 0..tissue-shard-count-1"
        )
    if (
        args.tissue_only
        and args.tissue_shard_count > 1
        and args.tissue_shard_index is None
    ):
        raise ValueError(
            "sharded --tissue-only execution requires --tissue-shard-index"
        )
    if args.tissue_collect_only and args.tissue_shard_count <= 1:
        raise ValueError(
            "--tissue-collect-only requires --tissue-shard-count greater than 1"
        )


def _run_discovery(
    root: Path, args: argparse.Namespace, progress_mode: str
) -> list[MeshRecord]:
    _log_info(
        "DISCOVERY",
        "Starting discovery",
        root=str(root),
        mesh_glob=args.mesh_glob or "m2m_only",
        progress=progress_mode,
    )
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
    _log_info("DISCOVERY", "Completed discovery", meshes=len(records))
    return records


def _run_full_or_qc_only(
    root: Path, out_dir: Path, args: argparse.Namespace, progress_mode: str
) -> int:
    records = _run_discovery(root, args, progress_mode)
    if not records:
        _log_warning(
            "DISCOVERY",
            "No meshes found",
            root=str(root),
            mesh_glob=args.mesh_glob or "m2m_only",
        )
        print(
            f"No meshes found under {root} matching {args.mesh_glob}", file=sys.stderr
        )
        return 2

    print(f"[DISCOVERY] Found {len(records)} mesh(es) under {root}", flush=True)
    _write_csv(
        out_dir / "found_meshes.csv", [_record_row(r) for r in records], FOUND_FIELDS
    )

    summary_rows, qc_exception_rows = _run_qc_detailed(records, args)

    _apply_bounds_outlier_flags(summary_rows)
    _write_csv(out_dir / "qc_summary.csv", summary_rows, SUMMARY_FIELDS)
    flag_rows = [row for row in summary_rows if row["status"] != "OK"]
    _write_csv(out_dir / "qc_flags.csv", flag_rows, SUMMARY_FIELDS)
    _write_optional_csv(
        out_dir / "qc_exception_details.csv", qc_exception_rows, EXCEPTION_FIELDS
    )
    _log_info(
        "QC",
        "Wrote QC outputs",
        summary_rows=len(summary_rows),
        flagged_rows=len(flag_rows),
        exception_rows=len(qc_exception_rows),
    )

    if not (args.skip_renders or args.qc_only):
        _render_outputs(records, summary_rows, out_dir, args)

    _log_info("MAIN", "Run outputs written", out=str(out_dir))
    print(f"Wrote QC outputs to {out_dir}")
    return 0


def _run_tissue_only(
    root: Path, out_dir: Path, args: argparse.Namespace, progress_mode: str
) -> int:
    records = _run_discovery(root, args, progress_mode)
    if not records:
        _log_warning(
            "DISCOVERY",
            "No meshes found",
            root=str(root),
            mesh_glob=args.mesh_glob or "m2m_only",
        )
        print(
            f"No meshes found under {root} matching {args.mesh_glob}", file=sys.stderr
        )
        return 2

    print(f"[DISCOVERY] Found {len(records)} mesh(es) under {root}", flush=True)
    task_specs = _build_tissue_task_specs(records)
    if args.tissue_shard_count > 1:
        shard_index = int(args.tissue_shard_index)
        task_specs = [
            spec
            for global_index, spec in enumerate(task_specs)
            if global_index % args.tissue_shard_count == shard_index
        ]
        records = [record for record, _ in task_specs]
        report_dir = _tissue_shard_report_dir(
            out_dir,
            shard_count=args.tissue_shard_count,
            shard_index=shard_index,
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        completion_marker = report_dir / "shard_complete.json"
        completion_marker.unlink(missing_ok=True)
        _write_csv(
            report_dir / "found_meshes.csv",
            [_record_row(record) for record in records],
            FOUND_FIELDS,
        )
        print(
            f"[TISSUE SHARD] Rendering shard {shard_index}/{args.tissue_shard_count - 1} "
            f"with {len(records)} mesh(es)",
            flush=True,
        )
    else:
        report_dir = out_dir
        completion_marker = None
        _write_csv(
            out_dir / "found_meshes.csv",
            [_record_row(record) for record in records],
            FOUND_FIELDS,
        )
    _log_info(
        "TISSUE_RENDER",
        "Skipping geometry QC and whole-mesh rendering for tissue-only run",
        meshes=len(records),
    )
    print(
        "[TISSUE RENDER] Tissue-only mode: skipping geometry QC and whole-mesh renders",
        flush=True,
    )
    if args.tissue_shard_count > 1:
        result = _run_tissue_outputs(
            records,
            out_dir,
            args,
            task_specs=task_specs,
            report_dir=report_dir,
            build_mosaics=False,
        )
    else:
        result = _run_tissue_outputs(records, out_dir, args)
    if completion_marker is not None:
        _write_json_atomic(
            completion_marker,
            {
                "status": "complete",
                "shard_count": args.tissue_shard_count,
                "shard_index": int(args.tissue_shard_index),
                "meshes": len(records),
                "renders": result["renders"],
                "render_failures": result["failures"],
                "incomplete_tissue_views": result["incomplete"],
                "view_version": TISSUE_VIEW_CONVENTION["version"],
                "completed_at": _utc_now(),
            },
        )
        print(f"[TISSUE SHARD] Completion marker: {completion_marker}", flush=True)
    _log_info("MAIN", "Tissue-only outputs written", out=str(out_dir))
    print(f"Wrote tissue-only outputs to {out_dir}")
    return 0


def _deduplicate_rows(
    rows: list[dict[str, str]],
    *,
    keys: tuple[str, ...],
) -> list[dict[str, str]]:
    deduplicated: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        deduplicated[tuple(str(row[key]) for key in keys)] = row
    return list(deduplicated.values())


def _run_tissue_collect_only(
    root: Path,
    out_dir: Path,
    args: argparse.Namespace,
    progress_mode: str,
) -> int:
    records = _run_discovery(root, args, progress_mode)
    if not records:
        print(
            f"No meshes found under {root} matching {args.mesh_glob}", file=sys.stderr
        )
        return 2

    _prepare_tissue_view_convention(out_dir)
    expected_specs = _build_tissue_task_specs(records)
    expected_paths = {str(record.path) for record, _ in expected_specs}
    all_presence: list[dict[str, str]] = []
    all_manifest: list[dict[str, str]] = []
    all_failures: list[dict[str, str]] = []
    covered_paths: set[str] = set()

    print(
        f"[TISSUE COLLECT] Validating {args.tissue_shard_count} shard(s) for "
        f"{len(records)} mesh(es)",
        flush=True,
    )
    for shard_index in range(args.tissue_shard_count):
        report_dir = _tissue_shard_report_dir(
            out_dir,
            shard_count=args.tissue_shard_count,
            shard_index=shard_index,
        )
        marker_path = report_dir / "shard_complete.json"
        if not marker_path.is_file():
            raise RuntimeError(f"Missing tissue-render shard marker: {marker_path}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("status") != "complete"
            or marker.get("shard_count") != args.tissue_shard_count
            or marker.get("shard_index") != shard_index
            or marker.get("view_version") != TISSUE_VIEW_CONVENTION["version"]
        ):
            raise RuntimeError(f"Invalid tissue-render shard marker: {marker_path}")

        shard_found = _read_csv(report_dir / "found_meshes.csv")
        shard_paths = {row["path"] for row in shard_found}
        expected_shard_paths = {
            str(record.path)
            for global_index, (record, _) in enumerate(expected_specs)
            if global_index % args.tissue_shard_count == shard_index
        }
        if shard_paths != expected_shard_paths:
            missing = sorted(expected_shard_paths - shard_paths)
            unexpected = sorted(shard_paths - expected_shard_paths)
            raise RuntimeError(
                f"Tissue shard {shard_index} coverage mismatch: "
                f"missing={missing[:3]} unexpected={unexpected[:3]}"
            )
        covered_paths.update(shard_paths)
        all_presence.extend(_read_csv(report_dir / "tissue_presence.csv"))
        all_manifest.extend(_read_csv(report_dir / "tissue_render_manifest.csv"))
        all_failures.extend(
            _read_csv(report_dir / "tissue_render_exception_details.csv")
        )

    if covered_paths != expected_paths:
        raise RuntimeError(
            "Combined tissue-render shard coverage differs from the discovered cohort"
        )

    presence_rows = _deduplicate_rows(
        all_presence,
        keys=("path", "tissue_tag"),
    )
    manifest_rows = _deduplicate_rows(
        all_manifest,
        keys=("path", "tissue_tag", "view"),
    )
    failure_rows = _deduplicate_rows(
        all_failures,
        keys=("path", "tissue_tag", "view", "error_type", "error_message"),
    )
    presence_rows.sort(key=lambda row: (int(row["tissue_tag"]), row["path"]))
    manifest_rows.sort(
        key=lambda row: (
            int(row["tissue_tag"]),
            TISSUE_VIEWS.index(row["view"]),
            row["output_path"],
        )
    )
    missing_tile_paths = []
    for row in manifest_rows:
        tile_path = Path(row["output_path"])
        try:
            valid = tile_path.is_file() and tile_path.stat().st_size > 0
        except OSError:
            valid = False
        if not valid:
            missing_tile_paths.append(str(tile_path))
    if missing_tile_paths:
        raise RuntimeError(
            f"Shard manifests reference {len(missing_tile_paths)} missing/empty tile(s): "
            f"{missing_tile_paths[:3]}"
        )

    _write_csv(
        out_dir / "found_meshes.csv",
        [_record_row(record) for record in records],
        FOUND_FIELDS,
    )
    _write_csv(out_dir / "tissue_presence.csv", presence_rows, TISSUE_PRESENCE_FIELDS)
    _write_csv(
        out_dir / "tissue_render_manifest.csv",
        manifest_rows,
        TISSUE_RENDER_MANIFEST_FIELDS,
    )
    _write_csv(
        out_dir / "tissue_render_exception_details.csv",
        failure_rows,
        TISSUE_EXCEPTION_FIELDS,
    )
    completeness_rows = _write_tissue_completeness(
        out_dir,
        render_records=records,
        presence_rows=presence_rows,
        manifest_rows=manifest_rows,
        failure_rows=failure_rows,
    )

    images_by_tissue: dict[tuple[int, str, str, str], list[Path]] = defaultdict(list)
    for row in manifest_rows:
        key = (
            int(row["tissue_tag"]),
            row["tissue_name"],
            row["tissue_slug"],
            row["view"],
        )
        images_by_tissue[key].append(Path(row["output_path"]))

    mosaic_failures: list[dict[str, object]] = []
    for (tag, name, slug, view), images in sorted(
        images_by_tissue.items(),
        key=lambda item: (item[0][0], TISSUE_VIEWS.index(item[0][3])),
    ):
        view_suffix = {"front": "", "back": "_back", "top": "_top"}[view]
        out_png = out_dir / "mosaics" / "tissues" / f"{slug}{view_suffix}_wall.png"
        print(
            f"[MOSAIC] Building {view} tissue wall for tag {tag} {name} "
            f"({len(images)} tiles)",
            flush=True,
        )
        try:
            make_mosaic(images, out_png, cols=args.cols, tile_size=args.tile_size)
        except Exception as exc:
            mosaic_failures.append(
                {
                    "stage": "tissue_mosaic",
                    "subject": "",
                    "repeat": "",
                    "roi": "",
                    "mesh_id": slug,
                    "path": "",
                    "output_path": str(out_png),
                    "error_type": type(exc).__name__,
                    "error_message": _short_error(exc),
                    "traceback": traceback.format_exc(),
                    "tissue_tag": tag,
                    "tissue_name": name,
                    "tissue_slug": slug,
                    "view": view,
                }
            )
    _write_csv(
        out_dir / "tissue_mosaic_exception_details.csv",
        mosaic_failures,
        TISSUE_EXCEPTION_FIELDS,
    )
    incomplete = [row for row in completeness_rows if row["status"] != "OK"]
    summary = {
        "status": (
            "complete"
            if not incomplete and not mosaic_failures and len(images_by_tissue) == 19
            else "incomplete"
        ),
        "meshes": len(records),
        "shards": args.tissue_shard_count,
        "tissue_tiles": len(manifest_rows),
        "tissue_walls": len(images_by_tissue),
        "render_failures": len(failure_rows),
        "incomplete_tissue_views": len(incomplete),
        "mosaic_failures": len(mosaic_failures),
        "view_version": TISSUE_VIEW_CONVENTION["version"],
        "completed_at": _utc_now(),
    }
    _write_json_atomic(out_dir / "accelerated_tissue_wall_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if summary["status"] != "complete":
        raise RuntimeError(
            "Accelerated tissue-wall collection is incomplete; see "
            f"{out_dir / 'accelerated_tissue_wall_summary.json'}"
        )
    print(f"Wrote accelerated tissue walls to {out_dir}", flush=True)
    return 0


def _run_render_only(out_dir: Path, args: argparse.Namespace) -> int:
    found_path = out_dir / "found_meshes.csv"
    summary_path = out_dir / "qc_summary.csv"
    if not found_path.exists():
        _log_error("RENDER", "Missing render-only input file", path=str(found_path))
        print(f"Missing render input: {found_path}", file=sys.stderr)
        return 2
    if not summary_path.exists():
        _log_error("RENDER", "Missing render-only input file", path=str(summary_path))
        print(f"Missing render input: {summary_path}", file=sys.stderr)
        return 2

    print(f"[RENDER] Loading prior QC outputs from {out_dir}", flush=True)
    records = _load_found_records(found_path)
    summary_rows = _read_csv(summary_path)
    if not records:
        _log_warning("RENDER", "Render-only input had no records", path=str(found_path))
        print(f"No meshes found in {found_path}", file=sys.stderr)
        return 2

    records, refreshed_metadata = _refresh_loaded_metadata(
        records,
        summary_rows,
        root=Path(args.root).expanduser(),
        args=args,
    )
    if refreshed_metadata:
        _write_csv(
            found_path, [_record_row(record) for record in records], FOUND_FIELDS
        )
        _write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
        _write_csv(
            out_dir / "qc_flags.csv",
            [row for row in summary_rows if row["status"] != "OK"],
            SUMMARY_FIELDS,
        )
        print(
            f"[RENDER] Refreshed path-derived metadata for {refreshed_metadata} mesh(es)",
            flush=True,
        )
        _log_info(
            "RENDER",
            "Refreshed path-derived metadata",
            meshes=refreshed_metadata,
            found_path=str(found_path),
            summary_path=str(summary_path),
        )

    print(
        f"[RENDER] Loaded {len(records)} discovered mesh(es) and {len(summary_rows)} QC row(s)",
        flush=True,
    )
    _log_info(
        "RENDER",
        "Loaded prior QC outputs",
        discovered_meshes=len(records),
        qc_rows=len(summary_rows),
    )
    _render_outputs(records, summary_rows, out_dir, args)
    print(f"Wrote render outputs to {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan generated m2m .msh files for geometry QC flags and render mesh mosaic walls."
    )
    parser.add_argument(
        "--root", required=True, help="Root containing ROI/subject/repeat mesh outputs."
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for CSV reports, renders, and mosaics.",
    )
    parser.add_argument(
        "--mesh-glob",
        default=None,
        help=(
            "Recursive mesh glob. If omitted, only .msh files inside m2m* "
            "directories are scanned. Pass '*.msh' to scan all meshes."
        ),
    )
    parser.add_argument(
        "--roi-regex",
        default=None,
        help="Optional regex for ROI inference; first group is used.",
    )
    parser.add_argument(
        "--subject-regex",
        default=None,
        help="Optional regex for subject inference; first group is used.",
    )
    parser.add_argument(
        "--repeat-regex",
        default=None,
        help="Optional regex for repeat inference; first group is used.",
    )
    parser.add_argument(
        "--image-size", type=int, default=1200, help="Individual render size in pixels."
    )
    parser.add_argument(
        "--tile-size", type=int, default=220, help="Mosaic tile size in pixels."
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Mosaic columns; default uses square-ish grid.",
    )
    parser.add_argument(
        "--renderer",
        choices=("auto", "gmsh", "pyvista", "pillow"),
        default="auto",
        help="PNG renderer. auto prefers Gmsh, then PyVista, then Pillow.",
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
        help="Number of parallel worker processes for QC and rendering. Default 0 uses all visible CPUs.",
    )
    parser.add_argument(
        "--skip-renders",
        action="store_true",
        help="Write CSV QC reports without PNG rendering.",
    )
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
        "--tissue-only",
        action="store_true",
        help=(
            "Discover meshes and generate tissue tiles/walls only, skipping geometry QC and all "
            "whole-mesh renders. Existing non-empty tissue tiles are reused."
        ),
    )
    parser.add_argument(
        "--tissue-collect-only",
        action="store_true",
        help=(
            "Merge completed tissue-render shard reports, validate full-cohort coverage, "
            "and build tissue walls without running Gmsh."
        ),
    )
    parser.add_argument(
        "--tissue-shard-count",
        type=int,
        default=1,
        help="Total number of disjoint tissue-render shards. Default 1 disables sharding.",
    )
    parser.add_argument(
        "--tissue-shard-index",
        type=int,
        default=None,
        help="Zero-based tissue-render shard index assigned to this process.",
    )
    parser.add_argument(
        "--run-label",
        default="",
        help="Optional suffix for run logs so concurrent shards never overwrite one another.",
    )
    parser.add_argument(
        "--roi-walls",
        action="store_true",
        help="Also write separate ROI wall mosaics. Default writes only all_mesh_wall.png.",
    )
    parser.add_argument(
        "--tissue-walls",
        action="store_true",
        help=(
            "Also render each tagged tetrahedral tissue and write front/back cohort walls, plus "
            "a top wall for compact bone tag 7. Each worker loads one mesh once and processes "
            "all of its tissues."
        ),
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
    global _RUN_LOGGER
    args = build_parser().parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    _RUN_LOGGER = RunLogger(out_dir, run_label=args.run_label)
    _RUN_LOGGER.write_context(_build_run_context(root, out_dir, args, argv))
    _log_info(
        "MAIN",
        "Starting mesh QC run",
        run_stage=_resolved_stage_name(args),
        script_path=str(Path(__file__).resolve()),
        root=str(root),
        out=str(out_dir),
        renderer=args.renderer,
        workers=args.workers,
    )
    try:
        try:
            _validate_stage_args(args)
        except ValueError as exc:
            _log_error("MAIN", "Argument validation failed", error_message=str(exc))
            print(str(exc), file=sys.stderr)
            return 2

        use_tqdm = _use_tqdm_progress(args)
        if args.progress != "none" and not use_tqdm:
            progress_mode = "text"
        elif use_tqdm:
            progress_mode = "tqdm"
        else:
            progress_mode = "none"
        if args.tissue_collect_only:
            rc = _run_tissue_collect_only(root, out_dir, args, progress_mode)
        elif args.tissue_only:
            rc = _run_tissue_only(root, out_dir, args, progress_mode)
        elif args.render_only:
            rc = _run_render_only(out_dir, args)
        else:
            rc = _run_full_or_qc_only(root, out_dir, args, progress_mode)
        _log_info("MAIN", "Finished mesh QC run", return_code=rc)
        return rc
    except Exception as exc:
        tb = traceback.format_exc()
        _log_error(
            "MAIN",
            "Fatal pipeline error",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        if _RUN_LOGGER is not None:
            _RUN_LOGGER.write_fatal(stage="main", exc=exc, traceback_text=tb)
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        _RUN_LOGGER = None


if __name__ == "__main__":
    raise SystemExit(main())
