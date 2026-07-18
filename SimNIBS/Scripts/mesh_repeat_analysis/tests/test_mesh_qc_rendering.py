import numpy as np
import pytest
import sys
import subprocess
import builtins
from pathlib import Path

from mesh_repeat_analysis.post.mesh_qc import rendering
from mesh_repeat_analysis.post.mesh_qc.geometry_qc import SurfaceArrays
from mesh_repeat_analysis.post.mesh_qc.rendering import make_mosaic, render_mesh_png, render_surface_png


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


def test_make_mosaic_falls_back_to_imagemagick_when_pillow_is_missing(tmp_path, monkeypatch):
    img1 = tmp_path / "a.png"
    img2 = tmp_path / "b.png"
    img1.write_bytes(b"fake-png")
    img2.write_bytes(b"fake-png")
    out = tmp_path / "wall.png"

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ModuleNotFoundError("No module named 'PIL'")
        return real_import(name, *args, **kwargs)

    calls = []

    def fake_which(name):
        if name == "magick":
            return "/usr/bin/magick"
        return None

    def fake_run(cmd, *, check, capture_output, text):
        calls.append(cmd)
        out.write_bytes(b"mosaic-png")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(rendering.shutil, "which", fake_which)
    monkeypatch.setattr(rendering.subprocess, "run", fake_run)

    make_mosaic([img1, img2], out, cols=2, tile_size=20, label_height=10)

    assert out.exists()
    assert calls
    assert calls[0][:2] == ["/usr/bin/magick", "montage"]
    assert str(img1) in calls[0]
    assert str(img2) in calls[0]
    assert str(out) == calls[0][-1]


def test_make_mosaic_chunks_large_imagemagick_walls(tmp_path, monkeypatch):
    image_paths = []
    for idx in range(5):
        image_path = tmp_path / f"tile_{idx}.png"
        image_path.write_bytes(b"fake-png")
        image_paths.append(image_path)
    out = tmp_path / "wall.png"

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ModuleNotFoundError("No module named 'PIL'")
        return real_import(name, *args, **kwargs)

    calls = []

    def fake_which(name):
        if name == "montage":
            return "/usr/bin/montage"
        return None

    def fake_run(cmd, *, check, capture_output, text):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"mosaic-png")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(rendering.shutil, "which", fake_which)
    monkeypatch.setattr(rendering.subprocess, "run", fake_run)
    monkeypatch.setenv("MESH_QC_IMAGEMAGICK_MAX_INPUTS", "2")

    make_mosaic(image_paths, out, cols=2, tile_size=20)

    assert out.exists()
    assert len(calls) == 4
    assert all("-label" in call for call in calls[:3])
    assert "-label" not in calls[-1]
    assert calls[-1][:2] == ["/usr/bin/montage", "-background"]
    assert calls[-1][-1] == str(out)


def test_make_mosaic_reports_missing_mosaic_backends(tmp_path, monkeypatch):
    img = tmp_path / "a.png"
    img.write_bytes(b"fake-png")
    out = tmp_path / "wall.png"

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ModuleNotFoundError("No module named 'PIL'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(rendering.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="Pillow.*ImageMagick"):
        make_mosaic([img], out, cols=1, tile_size=20)


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


def test_surface_renderer_exports_temporary_surface_for_gmsh(tmp_path, monkeypatch):
    out = tmp_path / "tissue.png"
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
    seen = {}

    def fake_gmsh(path, out_png, *, label, image_size, orthographic):
        seen["mesh_text"] = path.read_text(encoding="utf-8")
        seen["label"] = label
        seen["image_size"] = image_size
        seen["orthographic"] = orthographic
        out_png.write_bytes(b"png")

    monkeypatch.setattr(rendering, "_render_with_gmsh", fake_gmsh)

    actual = render_surface_png(surface, out, label="Scalp", image_size=180, renderer="gmsh")

    assert actual == "gmsh"
    assert "$MeshFormat" in seen["mesh_text"]
    assert "$Elements\n1\n" in seen["mesh_text"]
    assert seen["label"] == "Scalp"
    assert seen["image_size"] == 180
    assert seen["orthographic"] is True
    assert out.exists()


@pytest.mark.parametrize(
    ("view", "expected_points"),
    [
        (
            "front",
            np.array(
                [
                    [-2.0, -1.5, -1.0],
                    [2.0, -1.5, -1.0],
                    [-2.0, 1.5, 1.0],
                ]
            ),
        ),
        (
            "back",
            np.array(
                [
                    [2.0, -1.5, 1.0],
                    [-2.0, -1.5, 1.0],
                    [2.0, 1.5, -1.0],
                ]
            ),
        ),
        (
            "top",
            np.array(
                [
                    [-2.0, -1.0, -1.5],
                    [2.0, -1.0, -1.5],
                    [-2.0, 1.0, 1.5],
                ]
            ),
        ),
    ],
)
def test_surface_renderer_maps_ras_to_anatomical_orthographic_view(
    tmp_path,
    monkeypatch,
    view,
    expected_points,
):
    out = tmp_path / f"{view}.png"
    surface = SurfaceArrays(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [0.0, 2.0, 3.0],
            ]
        ),
        faces=np.array([[0, 1, 2]]),
    )
    seen = {}

    def fake_pillow(
        surface_arg,
        out_png,
        *,
        label,
        image_size,
        max_faces,
        camera_aligned,
    ):
        seen["points"] = surface_arg.points.copy()
        seen["camera_aligned"] = camera_aligned
        out_png.write_bytes(b"png")

    monkeypatch.setattr(rendering, "_render_surface_with_pillow", fake_pillow)

    actual = render_surface_png(
        surface,
        out,
        label=view.title(),
        view=view,
        renderer="pillow",
    )

    assert actual == "pillow"
    assert seen["camera_aligned"] is True
    np.testing.assert_allclose(seen["points"], expected_points)


def test_surface_renderer_rejects_unknown_view(tmp_path):
    surface = SurfaceArrays(
        points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.array([[0, 1, 2]]),
    )

    with pytest.raises(ValueError, match="Unsupported surface view"):
        render_surface_png(surface, tmp_path / "bad.png", label="bad", view="side")


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


def test_gmsh_tissue_geo_script_forces_anatomical_orthographic_camera(tmp_path):
    script = rendering._build_gmsh_geo_script(
        tmp_path / "tissue.msh",
        tmp_path / "render.png",
        image_size=320,
        orthographic=True,
    )

    assert "General.Orthographic = 1;" in script
    assert "General.Trackball = 0;" in script
    assert "General.RotationX = 0;" in script
    assert "General.RotationY = 0;" in script
    assert "General.RotationZ = 0;" in script


def test_gmsh_renderer_reports_timeout_as_runtime_error(tmp_path, monkeypatch):
    mesh_path = tmp_path / "head.msh"
    mesh_path.write_text("$MeshFormat\n", encoding="utf-8")
    out = tmp_path / "render.png"

    monkeypatch.setattr(rendering, "_build_gmsh_command", lambda script_path: ["gmsh", str(script_path)])

    def fake_run(cmd, *, check, capture_output, text, env, timeout):
        assert timeout == 900
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(rendering.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="timed out after 900 seconds"):
        rendering._render_with_gmsh(mesh_path, out, label="mesh", image_size=320)


def test_find_gmsh_binary_skips_unusable_path_entries(tmp_path, monkeypatch):
    bad_dir = tmp_path / "bad"
    good_dir = tmp_path / "good"
    bad_dir.mkdir()
    good_dir.mkdir()
    bad = bad_dir / "gmsh"
    good = good_dir / "gmsh"
    bad.write_text(
        "#!/bin/sh\n"
        "echo \"gmsh: /lib64/libm.so.6: version GLIBC_2.23 not found\" >&2\n"
        "exit 1\n",
        encoding="utf-8",
    )
    good.write_text("#!/bin/sh\necho 4.11.1\n", encoding="utf-8")
    bad.chmod(0o755)
    good.chmod(0o755)

    monkeypatch.delenv("MESH_QC_GMSH_BIN", raising=False)
    monkeypatch.setenv("PATH", f"{bad_dir}:{good_dir}")
    monkeypatch.setattr(rendering, "_GMSH_BIN_CACHE", None)

    assert rendering._find_gmsh_binary() == str(good)


def test_find_gmsh_binary_uses_explicit_env_override(tmp_path, monkeypatch):
    gmsh = tmp_path / "gmsh-custom"
    gmsh.write_text("#!/bin/sh\necho 4.12.0\n", encoding="utf-8")
    gmsh.chmod(0o755)

    monkeypatch.setenv("MESH_QC_GMSH_BIN", str(gmsh))
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(rendering, "_GMSH_BIN_CACHE", None)

    assert rendering._find_gmsh_binary() == str(gmsh)


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
