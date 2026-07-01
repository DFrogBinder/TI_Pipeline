import csv
import os

import numpy as np
import pytest

from mesh_repeat_analysis.post.mesh_qc import run_mesh_qc
from mesh_repeat_analysis.post.mesh_qc.geometry_qc import SurfaceArrays


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


def test_parser_defaults_to_pillow_and_all_cpus():
    args = run_mesh_qc.build_parser().parse_args(["--root", "/tmp/in", "--out", "/tmp/out"])

    assert args.renderer == "pillow"
    assert args.workers == 0
    assert args.image_size == 1200
    assert args.qc_only is False
    assert args.render_only is False


def test_resolve_auto_workers_prefers_slurm_cpu_allocation(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "20")
    monkeypatch.setattr(run_mesh_qc.os, "cpu_count", lambda: 64)

    assert run_mesh_qc._resolve_worker_count(0) == 20


def test_resolve_auto_workers_uses_affinity_when_available(monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setattr(run_mesh_qc.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(run_mesh_qc.os, "sched_getaffinity", lambda pid: set(range(12)))

    assert run_mesh_qc._resolve_worker_count(0) == 12


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
