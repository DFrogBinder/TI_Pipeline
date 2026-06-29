import pytest

from mesh_repeat_analysis.post.mesh_qc.rendering import make_mosaic


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

