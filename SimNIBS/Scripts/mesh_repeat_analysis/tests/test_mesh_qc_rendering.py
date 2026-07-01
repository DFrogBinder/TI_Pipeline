import pytest
import numpy as np

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
