import pytest
import numpy as np
import sys
from pathlib import Path

from mesh_repeat_analysis.post.mesh_qc import rendering
from mesh_repeat_analysis.post.mesh_qc.geometry_qc import SurfaceArrays
from mesh_repeat_analysis.post.mesh_qc.rendering import make_mosaic, render_mesh_png


def test_make_mosaic_combines_png_tiles(tmp_path):
    try:
        from PIL import Image
    except Exception:
        pytest.skip("Pillow is not installed")

    img1 = tmp_path / "a.png"
    img2 = tmp_path / "b.png"
    Image.new("RGB", (20, 20), color="red").save(img1)
    Image.new("RGB", (20, 20), color="blue").save(img2)

    out = tmp_path / "wall.png"
    make_mosaic([img1, img2], out, cols=2, tile_size=20, label_height=10)

    assert out.exists()
    with Image.open(out) as mosaic:
        assert mosaic.size[0] > 40
        assert mosaic.size[1] > 20


def test_pillow_renderer_writes_png_without_pyvista(tmp_path, monkeypatch):
    try:
        from PIL import Image
    except Exception:
        pytest.skip("Pillow is not installed")

    mesh_path = tmp_path / "head.msh"
    mesh_path.write_text("$MeshFormat\n", encoding="utf-8")
    out = tmp_path / "render.png"
    surface = SurfaceArrays(
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
    monkeypatch.setattr(rendering, "load_surface_arrays", lambda path: surface)

    render_mesh_png(mesh_path, out, label="sub-CC1", image_size=160, renderer="pillow")

    assert out.exists()
    with Image.open(out) as image:
        assert image.size == (160, 160)


def test_auto_renderer_falls_back_to_pillow_when_pyvista_fails(tmp_path, monkeypatch):
    mesh_path = tmp_path / "head.msh"
    mesh_path.write_text("$MeshFormat\n", encoding="utf-8")
    out = tmp_path / "render.png"
    calls = []

    def fail_pyvista(*args, **kwargs):
        raise ImportError("no pyvista")

    def fake_pillow(path, out_png, *, label, image_size, max_faces):
        calls.append((path, out_png, label, image_size, max_faces))
        out_png.write_bytes(b"png")

    monkeypatch.setattr(rendering, "_render_with_pyvista", fail_pyvista)
    monkeypatch.setattr(rendering, "_render_with_pillow", fake_pillow)

    render_mesh_png(mesh_path, out, label="mesh", image_size=99, renderer="auto", max_faces=123)

    assert calls == [(mesh_path, out, "mesh", 99, 123)]


def test_auto_renderer_prefers_gmsh_before_other_renderers(tmp_path, monkeypatch):
    mesh_path = tmp_path / "head.msh"
    mesh_path.write_text("$MeshFormat\n", encoding="utf-8")
    out = tmp_path / "render.png"
    calls = []

    def fake_gmsh(path, out_png, *, label, image_size):
        calls.append((path, out_png, label, image_size))
        out_png.write_bytes(b"png")

    monkeypatch.setattr(rendering, "_render_with_gmsh", fake_gmsh)
    monkeypatch.setattr(
        rendering,
        "_render_with_pyvista",
        lambda *args, **kwargs: pytest.fail("pyvista fallback should not run when gmsh succeeds"),
    )
    monkeypatch.setattr(
        rendering,
        "_render_with_pillow",
        lambda *args, **kwargs: pytest.fail("pillow fallback should not run when gmsh succeeds"),
    )

    render_mesh_png(mesh_path, out, label="mesh", image_size=144, renderer="auto")

    assert calls == [(mesh_path, out, "mesh", 144)]


def test_auto_renderer_falls_back_from_gmsh_and_pyvista_to_pillow(tmp_path, monkeypatch):
    mesh_path = tmp_path / "head.msh"
    mesh_path.write_text("$MeshFormat\n", encoding="utf-8")
    out = tmp_path / "render.png"
    calls = []

    def fail_gmsh(*args, **kwargs):
        raise RuntimeError("gmsh unavailable")

    def fail_pyvista(*args, **kwargs):
        raise ImportError("no pyvista")

    def fake_pillow(path, out_png, *, label, image_size, max_faces):
        calls.append((path, out_png, label, image_size, max_faces))
        out_png.write_bytes(b"png")

    monkeypatch.setattr(rendering, "_render_with_gmsh", fail_gmsh)
    monkeypatch.setattr(rendering, "_render_with_pyvista", fail_pyvista)
    monkeypatch.setattr(rendering, "_render_with_pillow", fake_pillow)

    render_mesh_png(mesh_path, out, label="mesh", image_size=111, renderer="auto", max_faces=321)

    assert calls == [(mesh_path, out, "mesh", 111, 321)]


def test_write_surface_mesh_msh2_exports_triangles(tmp_path):
    surface = SurfaceArrays(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64),
    )
    out = tmp_path / "surface_only.msh"

    rendering._write_surface_mesh_msh2(surface, out)

    text = out.read_text(encoding="utf-8")
    assert "$MeshFormat" in text
    assert "2.2 0 8" in text
    assert "$Nodes" in text
    assert "$Elements" in text
    assert "\n4\n" in text
    assert "\n2\n1 2 0 1 2 3\n2 2 0 1 2 4\n" in text


def test_pillow_renderer_keeps_dense_surface_filled_in_front_view(tmp_path, monkeypatch):
    try:
        from PIL import Image
    except Exception:
        pytest.skip("Pillow is not installed")

    mesh_path = tmp_path / "head.msh"
    mesh_path.write_text("$MeshFormat\n", encoding="utf-8")
    out = tmp_path / "render.png"

    nx = 41
    nz = 61
    points = []
    faces = []
    for iz in range(nz):
        z = 1.8 * (iz / (nz - 1) - 0.5)
        for ix in range(nx):
            x = 1.0 * (ix / (nx - 1) - 0.5)
            points.append([x, 0.0, z])
    for iz in range(nz - 1):
        for ix in range(nx - 1):
            a = iz * nx + ix
            b = a + 1
            c = a + nx
            d = c + 1
            faces.append([a, c, b])
            faces.append([b, c, d])

    monkeypatch.setattr(
        rendering,
        "load_surface_arrays",
        lambda path: SurfaceArrays(points=np.asarray(points, dtype=float), faces=np.asarray(faces, dtype=np.int64)),
    )

    render_mesh_png(mesh_path, out, label="", image_size=200, renderer="pillow", max_faces=20)

    with Image.open(out) as image:
        arr = np.asarray(image.convert("L"))
    mask = arr < 250
    ys, xs = np.nonzero(mask)

    assert mask.sum() > 12000
    aspect = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
    assert 0.48 < aspect < 0.62


def test_gmsh_geo_script_hides_surface_edges(tmp_path):
    script = rendering._build_gmsh_geo_script(tmp_path / "head.msh", tmp_path / "render.png", image_size=320)

    assert "Mesh.SurfaceFaces = 1;" in script
    assert "Mesh.SurfaceEdges = 0;" in script


def test_pyvista_renderer_uses_shaded_nonfrontal_view(tmp_path, monkeypatch):
    mesh_path = tmp_path / "head.msh"
    mesh_path.write_text("$MeshFormat\n", encoding="utf-8")
    out = tmp_path / "render.png"
    surface = SurfaceArrays(
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
    monkeypatch.setattr(rendering, "load_surface_arrays", lambda path: surface)

    calls = {}

    class FakePolyData:
        def __init__(self, points, faces):
            self.points = np.asarray(points, dtype=float)
            self._faces = faces
            mins = self.points.min(axis=0)
            maxs = self.points.max(axis=0)
            self.center = tuple(((mins + maxs) * 0.5).tolist())
            self.bounds = (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])
            self.n_points = len(self.points)

        def triangulate(self):
            return self

    class FakeCamera:
        def __init__(self):
            self.parallel_projection = None
            self.focal_point = None
            self.position = None
            self.view_up = None
            self.parallel_scale = None

    class FakePlotter:
        def __init__(self, off_screen, window_size):
            calls["plotter"] = self
            calls["off_screen"] = off_screen
            calls["window_size"] = window_size
            self.camera = FakeCamera()

        def set_background(self, color):
            calls["background"] = color

        def add_mesh(self, mesh, **kwargs):
            calls["add_mesh_kwargs"] = kwargs

        def add_text(self, label, **kwargs):
            calls["label"] = label

        def reset_camera_clipping_range(self):
            calls["reset_camera"] = True

        def screenshot(self, path):
            calls["screenshot"] = path
            Path(path).write_bytes(b"png")

        def close(self):
            calls["closed"] = True

    class FakePyVista:
        PolyData = FakePolyData
        Plotter = FakePlotter

    monkeypatch.setitem(sys.modules, "pyvista", FakePyVista)

    rendering._render_with_pyvista(mesh_path, out, label="mesh", image_size=180)

    assert out.exists()
    assert calls["add_mesh_kwargs"]["show_edges"] is False
    assert calls["plotter"].camera.parallel_projection is False
    x, y, z = calls["plotter"].camera.position
    assert x != 0.0
    assert y != 0.0
    assert z != 0.0
