import csv
import json
import os
from contextlib import nullcontext
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pytest

from mesh_repeat_analysis.post.mesh_qc import run_mesh_qc
from mesh_repeat_analysis.post.mesh_qc.geometry_qc import SurfaceArrays
from mesh_repeat_analysis.post.mesh_qc.loaders import TissueSurface


class FakeTqdm:
    instances = []

    def __init__(self, *, total=None, desc=None, unit=None, dynamic_ncols=None):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.n = 0
        self.postfix = {}
        self.closed = False
        FakeTqdm.instances.append(self)

    def update(self, amount):
        self.n += amount

    def set_postfix_str(self, value):
        self.postfix["text"] = value

    def close(self):
        self.closed = True


def test_cli_writes_csv_reports_in_skip_render_mode(tmp_path, monkeypatch):
    mesh_path = tmp_path / "Pallidum" / "sub-CC110056" / "repeat_01" / "head.msh"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text("$MeshFormat\n", encoding="utf-8")
    out_dir = tmp_path / "qc"

    def fake_loader(path):
        assert path == mesh_path
        return SurfaceArrays(
            points=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            faces=np.array(
                [
                    [0, 2, 1],
                    [0, 1, 3],
                    [1, 2, 3],
                    [2, 0, 3],
                ]
            ),
        )

    monkeypatch.setattr(run_mesh_qc, "load_surface_arrays", fake_loader)

    rc = run_mesh_qc.main(
        [
            "--root",
            str(tmp_path),
            "--out",
            str(out_dir),
            "--mesh-glob",
            "*.msh",
            "--workers",
            "1",
            "--skip-renders",
        ]
    )

    assert rc == 0
    with (out_dir / "qc_summary.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["roi"] == "Pallidum"
    assert rows[0]["mesh_id"] == "head"
    assert rows[0]["subject"] == "sub-CC110056"
    assert rows[0]["status"] == "OK"


def test_parser_defaults_to_auto_and_all_cpus():
    args = run_mesh_qc.build_parser().parse_args(["--root", "/tmp/in", "--out", "/tmp/out"])
    gmsh_args = run_mesh_qc.build_parser().parse_args(
        ["--root", "/tmp/in", "--out", "/tmp/out", "--renderer", "gmsh"]
    )

    assert args.renderer == "auto"
    assert args.workers == 0
    assert args.image_size == 1200
    assert args.qc_only is False
    assert args.render_only is False
    assert args.tissue_walls is False
    assert gmsh_args.renderer == "gmsh"


def test_tissue_walls_render_each_present_tissue_and_report_missing_labels(tmp_path, monkeypatch):
    records = [
        run_mesh_qc.MeshRecord(
            path=tmp_path / f"sub-CC{idx}.msh",
            roi="unknown_roi",
            subject=f"sub-CC{idx}",
            repeat="repeat_01",
            mesh_id=f"m2m_sub-CC{idx}",
        )
        for idx in (1, 2)
    ]
    surface = SurfaceArrays(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        faces=np.array([[0, 1, 2]]),
    )

    def fake_tissues(path):
        if path == records[0].path:
            yield TissueSurface(1, "White Matter", "tag_01_white_matter", surface)
        yield TissueSurface(5, "Scalp", "tag_05_scalp", surface)

    rendered = []

    def fake_render(surface_arg, out_png, *, label, image_size, renderer):
        rendered.append((out_png, label))
        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_png.write_bytes(b"tissue png")
        return "pillow"

    def fake_mosaic(image_paths, out_png, *, cols, tile_size):
        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_png.write_bytes(b"wall")

    monkeypatch.setattr(run_mesh_qc, "iter_tissue_surface_arrays", fake_tissues)
    monkeypatch.setattr(run_mesh_qc, "render_surface_png", fake_render)
    monkeypatch.setattr(run_mesh_qc, "make_mosaic", fake_mosaic)
    args = run_mesh_qc.build_parser().parse_args(
        [
            "--root",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            "--workers",
            "1",
            "--progress",
            "none",
            "--renderer",
            "pillow",
            "--tissue-walls",
        ]
    )
    out_dir = tmp_path / "out"

    run_mesh_qc._run_tissue_outputs(records, out_dir, args)

    assert len(rendered) == 3
    assert (out_dir / "mosaics" / "tissues" / "tag_01_white_matter_wall.png").exists()
    assert (out_dir / "mosaics" / "tissues" / "tag_05_scalp_wall.png").exists()
    with (out_dir / "tissue_render_completeness.csv").open(newline="", encoding="utf-8") as f:
        rows = {int(row["tissue_tag"]): row for row in csv.DictReader(f)}
    assert rows[1]["status"] == "MISSING_TISSUE"
    assert rows[1]["present_meshes"] == "1"
    assert rows[5]["status"] == "OK"
    assert rows[5]["present_meshes"] == "2"


def test_tissue_walls_fail_clearly_when_no_tissue_can_be_extracted(tmp_path, monkeypatch):
    record = run_mesh_qc.MeshRecord(
        path=tmp_path / "sub-CC1.msh",
        roi="unknown_roi",
        subject="sub-CC1",
        repeat="repeat_01",
        mesh_id="m2m_sub-CC1",
    )

    def fail_tissues(path):
        raise RuntimeError("neither SimNIBS nor meshio is available")
        yield

    monkeypatch.setattr(run_mesh_qc, "iter_tissue_surface_arrays", fail_tissues)
    args = run_mesh_qc.build_parser().parse_args(
        [
            "--root",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            "--workers",
            "1",
            "--progress",
            "none",
            "--renderer",
            "pillow",
            "--tissue-walls",
        ]
    )
    out_dir = tmp_path / "out"

    with pytest.raises(RuntimeError, match="No tagged tetrahedral tissues"):
        run_mesh_qc._run_tissue_outputs([record], out_dir, args)

    with (out_dir / "tissue_render_exception_details.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["stage"] == "tissue_load"
    assert "neither SimNIBS nor meshio" in rows[0]["error_message"]


def test_resolve_auto_workers_prefers_slurm_cpu_allocation(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "20")
    monkeypatch.setattr(run_mesh_qc.os, "cpu_count", lambda: 64)

    assert run_mesh_qc._resolve_worker_count(0) == 20


def test_resolve_auto_workers_uses_affinity_when_available(monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setattr(run_mesh_qc.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(run_mesh_qc.os, "sched_getaffinity", lambda pid: set(range(12)))

    assert run_mesh_qc._resolve_worker_count(0) == 12


def test_choose_stage_worker_count_respects_memory_budget():
    workers, details = run_mesh_qc._choose_stage_worker_count(
        stage="QC",
        task_count=100,
        visible_workers=32,
        memory_budget_bytes=64 * 1024**3,
        sampled_worker_rss_bytes=2 * 1024**3,
    )

    assert workers == 16
    assert details["limited_by_memory"] is True


def test_process_pool_executor_kwargs_skip_recycling_when_unsupported(monkeypatch):
    class LegacyExecutor:
        def __init__(self, max_workers, mp_context):
            self.max_workers = max_workers
            self.mp_context = mp_context

    monkeypatch.setattr(run_mesh_qc, "ProcessPoolExecutor", LegacyExecutor)

    kwargs = run_mesh_qc._process_pool_executor_kwargs(4)

    assert kwargs["max_workers"] == 4
    assert "mp_context" in kwargs
    assert "max_tasks_per_child" not in kwargs


def test_cli_emits_progress_messages(tmp_path, monkeypatch, capsys):
    mesh_path = tmp_path / "M1" / "sub-CC110056" / "repeat_01" / "m2m_sub-CC110056" / "head.msh"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text("$MeshFormat\n", encoding="utf-8")
    out_dir = tmp_path / "qc"

    monkeypatch.setattr(
        run_mesh_qc,
        "load_surface_arrays",
        lambda path: SurfaceArrays(
            points=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            faces=np.array(
                [
                    [0, 2, 1],
                    [0, 1, 3],
                    [1, 2, 3],
                    [2, 0, 3],
                ]
            ),
        ),
    )

    rc = run_mesh_qc.main(
        [
            "--root",
            str(tmp_path),
            "--out",
            str(out_dir),
            "--workers",
            "1",
            "--skip-renders",
            "--progress",
            "text",
            "--progress-every",
            "1",
        ]
    )

    captured = capsys.readouterr()

    assert rc == 0
    assert "[DISCOVERY] Found 1 mesh" in captured.out
    assert "[QC] 1/1" in captured.out
    assert "[QC] Complete" in captured.out


def test_cli_progress_shows_geometry_flag_names(tmp_path, monkeypatch, capsys):
    mesh_path = tmp_path / "M1" / "sub-CC110056" / "repeat_01" / "m2m_sub-CC110056" / "head.msh"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text("$MeshFormat\n", encoding="utf-8")
    out_dir = tmp_path / "qc"

    monkeypatch.setattr(
        run_mesh_qc,
        "load_surface_arrays",
        lambda path: SurfaceArrays(
            points=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            faces=np.array([[0, 1, 2], [0, 2, 3]]),
        ),
    )

    rc = run_mesh_qc.main(
        [
            "--root",
            str(tmp_path),
            "--out",
            str(out_dir),
            "--workers",
            "1",
            "--skip-renders",
            "--progress",
            "text",
            "--progress-every",
            "1",
        ]
    )

    captured = capsys.readouterr()

    assert rc == 0
    assert "FAIL:BOUNDARY_EDGES" in captured.out


def test_cli_uses_tqdm_when_requested(tmp_path, monkeypatch):
    FakeTqdm.instances = []
    mesh_path = tmp_path / "M1" / "sub-CC110056" / "repeat_01" / "m2m_sub-CC110056" / "head.msh"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text("$MeshFormat\n", encoding="utf-8")
    out_dir = tmp_path / "qc"

    monkeypatch.setattr(run_mesh_qc, "tqdm", FakeTqdm)
    monkeypatch.setattr(
        run_mesh_qc,
        "load_surface_arrays",
        lambda path: SurfaceArrays(
            points=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            faces=np.array(
                [
                    [0, 2, 1],
                    [0, 1, 3],
                    [1, 2, 3],
                    [2, 0, 3],
                ]
            ),
        ),
    )

    rc = run_mesh_qc.main(
        [
            "--root",
            str(tmp_path),
            "--out",
            str(out_dir),
            "--workers",
            "1",
            "--skip-renders",
            "--progress",
            "tqdm",
        ]
    )

    assert rc == 0
    assert [bar.desc for bar in FakeTqdm.instances] == ["Discovery", "QC"]
    assert all(bar.closed for bar in FakeTqdm.instances)
    assert FakeTqdm.instances[-1].n == 1


def test_run_qc_preserves_order_with_parallel_workers(tmp_path):
    records = [
        run_mesh_qc.MeshRecord(
            path=tmp_path / f"mesh_{idx}.msh",
            roi="unknown_roi",
            subject=f"sub-CC{idx}",
            repeat="repeat_01",
            mesh_id=f"m2m_sub-CC{idx}",
        )
        for idx in (2, 1)
    ]
    args = run_mesh_qc.build_parser().parse_args(
        ["--root", str(tmp_path), "--out", str(tmp_path / "out"), "--workers", "2", "--progress", "none"]
    )

    rows = run_mesh_qc._run_qc(records, args)

    assert [row["subject"] for row in rows] == ["sub-CC2", "sub-CC1"]
    assert all(str(row["flags"]).startswith("READ_FAIL") for row in rows)


def test_run_qc_detailed_retries_with_fewer_workers_after_pool_crash(tmp_path, monkeypatch):
    records = [
        run_mesh_qc.MeshRecord(
            path=tmp_path / f"mesh_{idx}.msh",
            roi="unknown_roi",
            subject=f"sub-CC{idx}",
            repeat="repeat_01",
            mesh_id=f"m2m_sub-CC{idx}",
        )
        for idx in range(4)
    ]
    args = run_mesh_qc.build_parser().parse_args(
        ["--root", str(tmp_path), "--out", str(tmp_path / "out"), "--workers", "0", "--progress", "none"]
    )

    monkeypatch.setattr(run_mesh_qc, "_resolve_worker_count", lambda requested: 4)
    monkeypatch.setattr(run_mesh_qc, "_resolve_memory_budget_bytes", lambda visible_workers: None)
    monkeypatch.setattr(
        run_mesh_qc,
        "_run_qc_warmup_samples",
        lambda records, args, summary_rows, exception_rows, progress: (0, None, list(range(len(records)))),
    )
    monkeypatch.setattr(
        run_mesh_qc,
        "_qc_record_worker_profiled",
        lambda record, check_components: (
            {
                "mesh_id": record.mesh_id,
                "subject": record.subject,
                "repeat": record.repeat,
                "roi": record.roi,
                "path": str(record.path),
                "status": "OK",
                "flags": "",
            },
            None,
            123,
        ),
    )
    monkeypatch.setattr(run_mesh_qc, "as_completed", lambda futures: list(futures))

    attempts = []

    class FakeFuture:
        def __init__(self, max_workers, fn, args, kwargs):
            self.max_workers = max_workers
            self.fn = fn
            self.args = args
            self.kwargs = kwargs

        def result(self):
            if self.max_workers > 2:
                raise BrokenProcessPool("worker died")
            return self.fn(*self.args, **self.kwargs)

    class FakeExecutor:
        def __init__(self, max_workers, *args, **kwargs):
            attempts.append(max_workers)
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return FakeFuture(self.max_workers, fn, args, kwargs)

    monkeypatch.setattr(run_mesh_qc, "ProcessPoolExecutor", FakeExecutor)

    rows, exception_rows = run_mesh_qc._run_qc_detailed(records, args)

    assert attempts == [4, 2]
    assert len(rows) == 4
    assert exception_rows == []
    assert [row["subject"] for row in rows] == [f"sub-CC{idx}" for idx in range(4)]


def test_cli_skips_rendering_meshes_that_failed_qc_loading(tmp_path, monkeypatch, capsys):
    ok_mesh = tmp_path / "Runs" / "sub-CC1" / "repeat_01" / "m2m_sub-CC1" / "head.msh"
    bad_mesh = tmp_path / "Runs" / "sub-CC2" / "repeat_01" / "m2m_sub-CC2" / "head.msh"
    ok_mesh.parent.mkdir(parents=True)
    bad_mesh.parent.mkdir(parents=True)
    ok_mesh.write_text("$MeshFormat\n", encoding="utf-8")
    bad_mesh.write_text("$MeshFormat\n", encoding="utf-8")
    out_dir = tmp_path / "qc"
    rendered = []

    def fake_loader(path):
        if path == bad_mesh:
            raise RuntimeError("not a readable mesh")
        return SurfaceArrays(
            points=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            faces=np.array(
                [
                    [0, 2, 1],
                    [0, 1, 3],
                    [1, 2, 3],
                    [2, 0, 3],
                ]
            ),
        )

    def fake_render(path, out_png, *, label, image_size, renderer):
        rendered.append((path, out_png, label))
        out_png.parent.mkdir(parents=True)
        out_png.write_bytes(b"fake png")

    def fake_mosaic(image_paths, out_png, *, cols, tile_size):
        out_png.parent.mkdir(parents=True)
        out_png.write_bytes(b"fake mosaic")

    monkeypatch.setattr(run_mesh_qc, "load_surface_arrays", fake_loader)
    monkeypatch.setattr(run_mesh_qc, "render_mesh_png", fake_render)
    monkeypatch.setattr(run_mesh_qc, "make_mosaic", fake_mosaic)

    rc = run_mesh_qc.main(
        [
            "--root",
            str(tmp_path),
            "--out",
            str(out_dir),
            "--workers",
            "1",
            "--progress",
            "text",
            "--progress-every",
            "1",
        ]
    )

    captured = capsys.readouterr()

    assert rc == 0
    assert rendered == [(ok_mesh, rendered[0][1], "sub-CC1\nrepeat_01\nm2m_sub-CC1")]
    assert "[RENDER] Skipping 1 mesh(es) that failed QC loading" in captured.out
    assert "all_mesh_wall.png" in str(out_dir / "mosaics" / "all_mesh_wall.png")
    assert (out_dir / "mosaics" / "all_mesh_wall.png").exists()


def test_cli_renders_meshes_with_geometry_qc_flags(tmp_path, monkeypatch):
    flagged_mesh = tmp_path / "Runs" / "sub-CC1" / "repeat_01" / "m2m_sub-CC1" / "head.msh"
    flagged_mesh.parent.mkdir(parents=True)
    flagged_mesh.write_text("$MeshFormat\n", encoding="utf-8")
    out_dir = tmp_path / "qc"
    rendered = []

    monkeypatch.setattr(
        run_mesh_qc,
        "load_surface_arrays",
        lambda path: SurfaceArrays(
            points=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            faces=np.array([[0, 1, 2], [0, 2, 3]]),
        ),
    )

    def fake_render(path, out_png, *, label, image_size, renderer):
        rendered.append(path)
        out_png.parent.mkdir(parents=True)
        out_png.write_bytes(b"fake png")

    monkeypatch.setattr(run_mesh_qc, "render_mesh_png", fake_render)
    def fake_mosaic(image_paths, out_png, *, cols, tile_size):
        out_png.parent.mkdir(parents=True)
        out_png.write_bytes(b"fake mosaic")

    monkeypatch.setattr(run_mesh_qc, "make_mosaic", fake_mosaic)

    rc = run_mesh_qc.main(["--root", str(tmp_path), "--out", str(out_dir), "--workers", "1"])

    assert rc == 0
    assert rendered == [flagged_mesh]


def test_cli_qc_only_runs_qc_without_rendering(tmp_path, monkeypatch):
    mesh_path = tmp_path / "M1" / "sub-CC110056" / "repeat_01" / "m2m_sub-CC110056" / "head.msh"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text("$MeshFormat\n", encoding="utf-8")
    out_dir = tmp_path / "qc"

    monkeypatch.setattr(
        run_mesh_qc,
        "load_surface_arrays",
        lambda path: SurfaceArrays(
            points=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            faces=np.array(
                [
                    [0, 2, 1],
                    [0, 1, 3],
                    [1, 2, 3],
                    [2, 0, 3],
                ]
            ),
        ),
    )
    monkeypatch.setattr(run_mesh_qc, "render_mesh_png", lambda *args, **kwargs: pytest.fail("render should not run"))

    rc = run_mesh_qc.main(
        [
            "--root",
            str(tmp_path),
            "--out",
            str(out_dir),
            "--workers",
            "1",
            "--qc-only",
        ]
    )

    assert rc == 0
    assert (out_dir / "qc_summary.csv").exists()
    assert not (out_dir / "mosaics" / "all_mesh_wall.png").exists()


def test_cli_render_only_uses_existing_qc_outputs(tmp_path, monkeypatch):
    out_dir = tmp_path / "qc"
    out_dir.mkdir(parents=True)
    ok_mesh = tmp_path / "Runs" / "sub-CC1" / "repeat_01" / "m2m_sub-CC1" / "head.msh"
    bad_mesh = tmp_path / "Runs" / "sub-CC2" / "repeat_01" / "m2m_sub-CC2" / "head.msh"

    with (out_dir / "found_meshes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=run_mesh_qc.FOUND_FIELDS)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "mesh_id": "m2m_sub-CC1",
                    "subject": "sub-CC1",
                    "repeat": "repeat_01",
                    "roi": "unknown_roi",
                    "path": str(ok_mesh),
                },
                {
                    "mesh_id": "m2m_sub-CC2",
                    "subject": "sub-CC2",
                    "repeat": "repeat_01",
                    "roi": "unknown_roi",
                    "path": str(bad_mesh),
                },
            ]
        )

    with (out_dir / "qc_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=run_mesh_qc.SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "mesh_id": "m2m_sub-CC1",
                    "subject": "sub-CC1",
                    "repeat": "repeat_01",
                    "roi": "unknown_roi",
                    "path": str(ok_mesh),
                    "status": "OK",
                    "flags": "",
                    "n_points": 4,
                    "n_faces": 4,
                    "degenerate_faces": 0,
                    "boundary_edges": 0,
                    "nonmanifold_edges": 0,
                    "connected_components": -1,
                    "x_size": 1.0,
                    "y_size": 1.0,
                    "z_size": 1.0,
                },
                {
                    "mesh_id": "m2m_sub-CC2",
                    "subject": "sub-CC2",
                    "repeat": "repeat_01",
                    "roi": "unknown_roi",
                    "path": str(bad_mesh),
                    "status": "FAIL",
                    "flags": "READ_FAIL:not readable",
                    "n_points": 0,
                    "n_faces": 0,
                    "degenerate_faces": 0,
                    "boundary_edges": 0,
                    "nonmanifold_edges": 0,
                    "connected_components": 0,
                    "x_size": 0.0,
                    "y_size": 0.0,
                    "z_size": 0.0,
                },
            ]
        )

    monkeypatch.setattr(run_mesh_qc, "discover_meshes", lambda *args, **kwargs: pytest.fail("discovery should not run"))
    monkeypatch.setattr(run_mesh_qc, "_run_qc", lambda *args, **kwargs: pytest.fail("qc should not run"))

    rendered = []

    def fake_render(path, out_png, *, label, image_size, renderer):
        rendered.append((path, label))
        out_png.parent.mkdir(parents=True)
        out_png.write_bytes(b"fake png")

    def fake_mosaic(image_paths, out_png, *, cols, tile_size):
        out_png.parent.mkdir(parents=True)
        out_png.write_bytes(b"fake mosaic")

    monkeypatch.setattr(run_mesh_qc, "render_mesh_png", fake_render)
    monkeypatch.setattr(run_mesh_qc, "make_mosaic", fake_mosaic)

    rc = run_mesh_qc.main(
        [
            "--root",
            str(tmp_path),
            "--out",
            str(out_dir),
            "--render-only",
        ]
    )

    assert rc == 0
    assert rendered == [(ok_mesh, "sub-CC1\nrepeat_01\nm2m_sub-CC1")]
    assert (out_dir / "mosaics" / "all_mesh_wall.png").exists()


def test_render_only_refreshes_unknown_roi_metadata_from_paths(tmp_path, monkeypatch):
    mesh_path = (
        tmp_path
        / "Right_DLPC_Runs"
        / "Right_DLPC_Data_04"
        / "sub-CC1"
        / "anat"
        / "m2m_sub-CC1"
        / "sub-CC1.msh"
    )
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text("$MeshFormat\n", encoding="utf-8")
    out_dir = tmp_path / "qc"
    out_dir.mkdir()
    run_mesh_qc._write_csv(
        out_dir / "found_meshes.csv",
        [
            {
                "mesh_id": "m2m_sub-CC1",
                "subject": "sub-CC1",
                "repeat": "04",
                "roi": "unknown_roi",
                "path": str(mesh_path),
            }
        ],
        run_mesh_qc.FOUND_FIELDS,
    )
    run_mesh_qc._write_csv(
        out_dir / "qc_summary.csv",
        [
            {
                "mesh_id": "m2m_sub-CC1",
                "subject": "sub-CC1",
                "repeat": "04",
                "roi": "unknown_roi",
                "path": str(mesh_path),
                "status": "OK",
                "flags": "",
                "n_points": 4,
                "n_faces": 4,
                "degenerate_faces": 0,
                "boundary_edges": 0,
                "nonmanifold_edges": 0,
                "connected_components": -1,
                "x_size": 1.0,
                "y_size": 1.0,
                "z_size": 1.0,
            }
        ],
        run_mesh_qc.SUMMARY_FIELDS,
    )

    monkeypatch.setattr(run_mesh_qc, "discover_meshes", lambda *args, **kwargs: pytest.fail("discovery should not run"))
    monkeypatch.setattr(run_mesh_qc, "_run_qc", lambda *args, **kwargs: pytest.fail("qc should not run"))

    def fake_render(path, out_png, *, label, image_size, renderer):
        out_png.parent.mkdir(parents=True)
        out_png.write_bytes(b"fake png")
        return "gmsh"

    def fake_mosaic(image_paths, out_png, *, cols, tile_size):
        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_png.write_bytes(b"fake mosaic")

    monkeypatch.setattr(run_mesh_qc, "render_mesh_png", fake_render)
    monkeypatch.setattr(run_mesh_qc, "make_mosaic", fake_mosaic)

    rc = run_mesh_qc.main(
        [
            "--root",
            str(tmp_path),
            "--out",
            str(out_dir),
            "--render-only",
            "--roi-walls",
        ]
    )

    assert rc == 0
    with (out_dir / "found_meshes.csv").open(newline="", encoding="utf-8") as f:
        found_rows = list(csv.DictReader(f))
    with (out_dir / "qc_summary.csv").open(newline="", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))
    assert found_rows[0]["roi"] == "Right_DLPC_Runs"
    assert summary_rows[0]["roi"] == "Right_DLPC_Runs"
    assert (out_dir / "mosaics" / "Right_DLPC_Runs_wall.png").exists()


def test_render_outputs_resumes_from_existing_pngs(tmp_path, monkeypatch):
    out_dir = tmp_path / "qc"
    records = [
        run_mesh_qc.MeshRecord(
            path=tmp_path / "sub-CC1.msh",
            roi="unknown_roi",
            subject="sub-CC1",
            repeat="repeat_01",
            mesh_id="m2m_sub-CC1",
        ),
        run_mesh_qc.MeshRecord(
            path=tmp_path / "sub-CC2.msh",
            roi="unknown_roi",
            subject="sub-CC2",
            repeat="repeat_01",
            mesh_id="m2m_sub-CC2",
        ),
    ]
    summary_rows = [
        {
            "mesh_id": record.mesh_id,
            "subject": record.subject,
            "repeat": record.repeat,
            "roi": record.roi,
            "path": str(record.path),
            "status": "OK",
            "flags": "",
        }
        for record in records
    ]
    args = run_mesh_qc.build_parser().parse_args(
        ["--root", str(tmp_path), "--out", str(out_dir), "--workers", "1", "--progress", "none"]
    )
    existing_png = out_dir / "renders" / "meshes" / "00001__sub-CC1__repeat_01__m2m_sub-CC1__sub-CC1.png"
    existing_png.parent.mkdir(parents=True)
    existing_png.write_bytes(b"already rendered")
    rendered = []
    mosaic_inputs = []

    def fake_render(path, out_png, *, label, image_size, renderer):
        rendered.append(path)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_png.write_bytes(b"new render")
        return "gmsh"

    def fake_mosaic(image_paths, out_png, *, cols, tile_size):
        mosaic_inputs.extend(image_paths)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_png.write_bytes(b"fake mosaic")

    monkeypatch.setattr(run_mesh_qc, "render_mesh_png", fake_render)
    monkeypatch.setattr(run_mesh_qc, "make_mosaic", fake_mosaic)

    run_mesh_qc._render_outputs(records, summary_rows, out_dir, args)

    assert rendered == [records[1].path]
    assert mosaic_inputs[0] == existing_png
    assert mosaic_inputs[1].name.startswith("00002__sub-CC2__")
    with (out_dir / "render_manifest.csv").open(newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    assert [row["subject"] for row in manifest_rows] == ["sub-CC1", "sub-CC2"]
    assert [row["actual_renderer"] for row in manifest_rows] == ["unknown_existing", "gmsh"]
    assert [row["requested_renderer"] for row in manifest_rows] == ["auto", "auto"]
    assert [row["resumed"] for row in manifest_rows] == ["1", "0"]


def test_render_manifest_is_written_before_mosaic_failure(tmp_path, monkeypatch):
    out_dir = tmp_path / "qc"
    record = run_mesh_qc.MeshRecord(
        path=tmp_path / "sub-CC1.msh",
        roi="unknown_roi",
        subject="sub-CC1",
        repeat="repeat_01",
        mesh_id="m2m_sub-CC1",
    )
    summary_rows = [
        {
            "mesh_id": record.mesh_id,
            "subject": record.subject,
            "repeat": record.repeat,
            "roi": record.roi,
            "path": str(record.path),
            "status": "OK",
            "flags": "",
        }
    ]
    args = run_mesh_qc.build_parser().parse_args(
        ["--root", str(tmp_path), "--out", str(out_dir), "--workers", "1", "--progress", "none"]
    )

    def fake_render(path, out_png, *, label, image_size, renderer):
        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_png.write_bytes(b"new render")
        return "gmsh"

    def fake_mosaic(image_paths, out_png, *, cols, tile_size):
        raise RuntimeError("mosaic backend was killed")

    monkeypatch.setattr(run_mesh_qc, "render_mesh_png", fake_render)
    monkeypatch.setattr(run_mesh_qc, "make_mosaic", fake_mosaic)

    with pytest.raises(RuntimeError, match="mosaic assemblies failed"):
        run_mesh_qc._render_outputs([record], summary_rows, out_dir, args)

    with (out_dir / "render_manifest.csv").open(newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    assert manifest_rows[0]["subject"] == "sub-CC1"
    assert manifest_rows[0]["actual_renderer"] == "gmsh"

    with (out_dir / "mosaic_exception_details.csv").open(newline="", encoding="utf-8") as f:
        mosaic_rows = list(csv.DictReader(f))
    assert mosaic_rows[0]["stage"] == "mosaic"
    assert mosaic_rows[0]["roi"] == "all"
    assert "mosaic backend was killed" in mosaic_rows[0]["error_message"]


def test_render_completeness_reports_missing_tiles_per_roi(tmp_path):
    records = [
        run_mesh_qc.MeshRecord(
            path=tmp_path / "a.msh",
            roi="Left_M1_Intact",
            subject="sub-CCMe",
            repeat="01",
            mesh_id="m2m_sub-CCMe",
        ),
        run_mesh_qc.MeshRecord(
            path=tmp_path / "b.msh",
            roi="Left_M1_Intact",
            subject="sub-CCMe",
            repeat="02",
            mesh_id="m2m_sub-CCMe",
        ),
        run_mesh_qc.MeshRecord(
            path=tmp_path / "c.msh",
            roi="Left_M1_Intact",
            subject="sub-CCMe",
            repeat="03",
            mesh_id="m2m_sub-CCMe",
        ),
    ]
    render_records = records[:2]
    successful = [(records[0], tmp_path / "a.png")]
    failure_rows = [
        {
            "roi": "Left_M1_Intact",
            "subject": "sub-CCMe",
            "repeat": "02",
            "mesh_id": "m2m_sub-CCMe",
            "path": str(records[1].path),
            "stage": "render",
            "output_path": str(tmp_path / "b.png"),
            "error_type": "RuntimeError",
            "error_message": "timeout",
            "traceback": "timeout",
        }
    ]

    rows = run_mesh_qc._write_render_completeness_outputs(
        tmp_path,
        records=records,
        render_records=render_records,
        successful=successful,
        failure_rows=failure_rows,
    )

    assert rows == [
        {
            "roi": "Left_M1_Intact",
            "status": "INCOMPLETE",
            "discovered_meshes": 3,
            "qc_loadable_meshes": 2,
            "rendered_meshes": 1,
            "qc_read_failures": 1,
            "render_failures": 1,
            "missing_tiles": 2,
        }
    ]
    with (tmp_path / "render_completeness.csv").open(newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert csv_rows[0]["status"] == "INCOMPLETE"
    assert csv_rows[0]["missing_tiles"] == "2"


def test_render_display_context_prefers_managed_xvfb_for_gmsh(monkeypatch):
    args = run_mesh_qc.build_parser().parse_args(
        ["--root", "/tmp/in", "--out", "/tmp/out", "--renderer", "gmsh"]
    )
    calls = []
    monkeypatch.setenv("DISPLAY", ":stale")

    class FakeDisplaySession:
        def __init__(self, *, width, height):
            calls.append(("init", width, height))

        def start(self):
            calls.append(("start",))
            return ":123"

        def stop(self):
            calls.append(("stop",))

    monkeypatch.setattr(run_mesh_qc, "VirtualDisplaySession", FakeDisplaySession)

    with run_mesh_qc._render_display_context(args):
        assert os.environ["DISPLAY"] == ":123"

    assert calls == [("init", 1200, 1200), ("start",), ("stop",)]


def test_render_display_context_does_not_fallback_to_stale_display_for_forced_gmsh(monkeypatch):
    args = run_mesh_qc.build_parser().parse_args(
        ["--root", "/tmp/in", "--out", "/tmp/out", "--renderer", "gmsh"]
    )
    monkeypatch.setenv("DISPLAY", ":stale")

    class BrokenDisplaySession:
        def __init__(self, *, width, height):
            pass

        def start(self):
            raise RuntimeError("xvfb broken")

        def stop(self):
            pass

    monkeypatch.setattr(run_mesh_qc, "VirtualDisplaySession", BrokenDisplaySession)

    with pytest.raises(RuntimeError, match="Gmsh rendering requires"):
        with run_mesh_qc._render_display_context(args):
            pass


def test_virtual_display_start_times_out_waiting_for_displayfd(monkeypatch):
    monkeypatch.setattr(run_mesh_qc.shutil, "which", lambda name: "/fake/Xvfb")
    monkeypatch.setattr(run_mesh_qc.select, "select", lambda read, write, err, timeout: ([], [], []))

    class FakeProc:
        def communicate(self, timeout=None):
            return "", ""

        def kill(self):
            pass

        def wait(self, timeout=None):
            return 1

    monkeypatch.setattr(run_mesh_qc.subprocess, "Popen", lambda *args, **kwargs: FakeProc())

    with pytest.raises(RuntimeError, match="did not report a display"):
        run_mesh_qc.VirtualDisplaySession().start()


def test_forced_gmsh_render_aborts_after_preflight_failure(tmp_path, monkeypatch):
    out_dir = tmp_path / "qc"
    records = [
        run_mesh_qc.MeshRecord(
            path=tmp_path / "sub-CC1.msh",
            roi="unknown_roi",
            subject="sub-CC1",
            repeat="repeat_01",
            mesh_id="m2m_sub-CC1",
        ),
        run_mesh_qc.MeshRecord(
            path=tmp_path / "sub-CC2.msh",
            roi="unknown_roi",
            subject="sub-CC2",
            repeat="repeat_01",
            mesh_id="m2m_sub-CC2",
        ),
    ]
    summary_rows = [
        {
            "mesh_id": record.mesh_id,
            "subject": record.subject,
            "repeat": record.repeat,
            "roi": record.roi,
            "path": str(record.path),
            "status": "OK",
            "flags": "",
        }
        for record in records
    ]
    args = run_mesh_qc.build_parser().parse_args(
        [
            "--root",
            str(tmp_path),
            "--out",
            str(out_dir),
            "--renderer",
            "gmsh",
            "--workers",
            "2",
            "--progress",
            "none",
        ]
    )
    calls = []

    def fake_preflight(record, out_png, *, image_size, renderer):
        calls.append(record.subject)
        return (
            run_mesh_qc._exception_row(
                record,
                stage="render",
                exc=RuntimeError("gmsh exited with status 1"),
                traceback_text="gmsh exited with status 1",
                output_path=out_png,
            ),
            None,
            0,
        )

    monkeypatch.setattr(run_mesh_qc, "_render_display_context", lambda args: nullcontext())
    monkeypatch.setattr(run_mesh_qc, "_run_render_isolated_once", fake_preflight)
    monkeypatch.setattr(run_mesh_qc, "_run_render_parallel_attempt", lambda *args, **kwargs: pytest.fail("parallel render should not start"))
    monkeypatch.setattr(run_mesh_qc, "make_mosaic", lambda *args, **kwargs: pytest.fail("mosaic should not run"))

    with pytest.raises(RuntimeError, match="Forced Gmsh render preflight failed"):
        run_mesh_qc._render_outputs(records, summary_rows, out_dir, args)

    assert calls == ["sub-CC1"]
    with (out_dir / "render_exception_details.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["subject"] == "sub-CC1"
    assert "gmsh exited with status 1" in rows[0]["error_message"]


def test_render_outputs_uses_parallel_workers_when_requested(tmp_path, monkeypatch):
    out_dir = tmp_path / "qc"
    records = [
        run_mesh_qc.MeshRecord(
            path=tmp_path / "sub-CC1.msh",
            roi="unknown_roi",
            subject="sub-CC1",
            repeat="repeat_01",
            mesh_id="m2m_sub-CC1",
        ),
        run_mesh_qc.MeshRecord(
            path=tmp_path / "sub-CC2.msh",
            roi="unknown_roi",
            subject="sub-CC2",
            repeat="repeat_01",
            mesh_id="m2m_sub-CC2",
        ),
    ]
    summary_rows = [
        {
            "mesh_id": record.mesh_id,
            "subject": record.subject,
            "repeat": record.repeat,
            "roi": record.roi,
            "path": str(record.path),
            "status": "OK",
            "flags": "",
        }
        for record in records
    ]
    args = run_mesh_qc.build_parser().parse_args(
        ["--root", str(tmp_path), "--out", str(out_dir), "--workers", "2", "--progress", "none"]
    )

    used_parallel = {"value": False}

    class FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class FakeExecutor:
        def __init__(self, *args, **kwargs):
            used_parallel["value"] = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return FakeFuture(fn(*args, **kwargs))

    rendered = []

    def fake_render(path, out_png, *, label, image_size, renderer):
        rendered.append(path)
        out_png.parent.mkdir(parents=True)
        out_png.write_bytes(b"fake png")

    def fake_as_completed(futures):
        return list(futures)

    def fake_mosaic(image_paths, out_png, *, cols, tile_size):
        out_png.parent.mkdir(parents=True)
        out_png.write_bytes(b"fake mosaic")

    monkeypatch.setattr(run_mesh_qc, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(run_mesh_qc, "as_completed", fake_as_completed)
    monkeypatch.setattr(run_mesh_qc, "render_mesh_png", fake_render)
    monkeypatch.setattr(run_mesh_qc, "make_mosaic", fake_mosaic)

    run_mesh_qc._render_outputs(records, summary_rows, out_dir, args)

    assert used_parallel["value"] is True
    assert rendered == [record.path for record in records]


def test_cli_writes_qc_exception_details_and_run_context(tmp_path, monkeypatch):
    ok_mesh = tmp_path / "Runs" / "sub-CC1" / "repeat_01" / "m2m_sub-CC1" / "head.msh"
    bad_mesh = tmp_path / "Runs" / "sub-CC2" / "repeat_01" / "m2m_sub-CC2" / "head.msh"
    ok_mesh.parent.mkdir(parents=True)
    bad_mesh.parent.mkdir(parents=True)
    ok_mesh.write_text("$MeshFormat\n", encoding="utf-8")
    bad_mesh.write_text("$MeshFormat\n", encoding="utf-8")
    out_dir = tmp_path / "qc"

    def fake_loader(path):
        if path == bad_mesh:
            raise RuntimeError("not a readable mesh")
        return SurfaceArrays(
            points=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            faces=np.array(
                [
                    [0, 2, 1],
                    [0, 1, 3],
                    [1, 2, 3],
                    [2, 0, 3],
                ]
            ),
        )

    monkeypatch.setattr(run_mesh_qc, "load_surface_arrays", fake_loader)

    rc = run_mesh_qc.main(
        [
            "--root",
            str(tmp_path),
            "--out",
            str(out_dir),
            "--workers",
            "1",
            "--qc-only",
            "--progress",
            "none",
        ]
    )

    assert rc == 0
    with (out_dir / "qc_exception_details.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["stage"] == "qc"
    assert rows[0]["subject"] == "sub-CC2"
    assert rows[0]["error_type"] == "RuntimeError"
    assert "not a readable mesh" in rows[0]["error_message"]
    assert "RuntimeError: not a readable mesh" in rows[0]["traceback"]

    context = json.loads((out_dir / "logs" / "run_context.json").read_text(encoding="utf-8"))
    assert context["root"] == str(tmp_path.resolve())
    assert context["out"] == str(out_dir.resolve())


def test_cli_writes_render_exception_details(tmp_path, monkeypatch):
    mesh_path = tmp_path / "Runs" / "sub-CC1" / "repeat_01" / "m2m_sub-CC1" / "head.msh"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text("$MeshFormat\n", encoding="utf-8")
    out_dir = tmp_path / "qc"

    monkeypatch.setattr(
        run_mesh_qc,
        "load_surface_arrays",
        lambda path: SurfaceArrays(
            points=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            faces=np.array(
                [
                    [0, 2, 1],
                    [0, 1, 3],
                    [1, 2, 3],
                    [2, 0, 3],
                ]
            ),
        ),
    )
    monkeypatch.setattr(
        run_mesh_qc,
        "render_mesh_png",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("render exploded")),
    )

    rc = run_mesh_qc.main(
        [
            "--root",
            str(tmp_path),
            "--out",
            str(out_dir),
            "--workers",
            "1",
            "--progress",
            "none",
        ]
    )

    assert rc == 0
    with (out_dir / "render_exception_details.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["stage"] == "render"
    assert rows[0]["subject"] == "sub-CC1"
    assert rows[0]["error_type"] == "RuntimeError"
    assert rows[0]["error_message"] == "render exploded"
    assert "RuntimeError: render exploded" in rows[0]["traceback"]
    assert (out_dir / "render_failures.txt").exists()


def test_main_writes_fatal_error_log_and_returns_nonzero(tmp_path, monkeypatch):
    out_dir = tmp_path / "qc"
    monkeypatch.setattr(run_mesh_qc, "_run_full_or_qc_only", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad root config")))

    rc = run_mesh_qc.main(["--root", str(tmp_path), "--out", str(out_dir), "--progress", "none"])

    assert rc == 1
    fatal_text = (out_dir / "logs" / "fatal_error.txt").read_text(encoding="utf-8")
    assert "ValueError: bad root config" in fatal_text
    assert "Traceback" in fatal_text
    run_log = (out_dir / "logs" / "mesh_qc.log").read_text(encoding="utf-8")
    assert "Fatal pipeline error" in run_log
