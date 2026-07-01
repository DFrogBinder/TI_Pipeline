import csv

import numpy as np

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
            "--skip-renders",
            "--progress",
            "tqdm",
        ]
    )

    assert rc == 0
    assert [bar.desc for bar in FakeTqdm.instances] == ["Discovery", "QC"]
    assert all(bar.closed for bar in FakeTqdm.instances)
    assert FakeTqdm.instances[-1].n == 1


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
