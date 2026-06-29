from __future__ import annotations

import math
from pathlib import Path


def render_mesh_png(path: Path, out_png: Path, *, label: str, image_size: int = 600) -> None:
    import numpy as np
    import pyvista as pv

    from .loaders import load_surface_arrays

    surface = load_surface_arrays(path)
    faces = np.asarray(surface.faces, dtype=np.int64)
    if faces.size == 0:
        raise RuntimeError("Surface extraction produced an empty mesh.")
    pv_faces = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
    mesh = pv.PolyData(np.asarray(surface.points, dtype=float), pv_faces).triangulate()
    if mesh.n_points == 0:
        raise RuntimeError("Surface extraction produced an empty mesh.")

    center = np.asarray(mesh.center, dtype=float)
    mesh.points = mesh.points - center
    bounds = mesh.bounds
    span = max(
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
        1.0,
    )

    plotter = pv.Plotter(off_screen=True, window_size=(image_size, image_size))
    plotter.set_background("white")
    plotter.add_mesh(
        mesh,
        color="lightgray",
        show_edges=True,
        edge_color="black",
        smooth_shading=True,
    )
    plotter.add_text(label, position="upper_left", font_size=10, color="black")
    plotter.camera_position = "iso"
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = span * 0.65
    plotter.reset_camera()
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = span * 0.65
    plotter.reset_camera_clipping_range()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out_png))
    plotter.close()


def _font(font_size: int):
    from PIL import ImageFont

    for candidate in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def make_mosaic(
    image_paths: list[Path],
    out_png: Path,
    *,
    cols: int | None = None,
    tile_size: int = 220,
    padding: int = 10,
    margin: int = 16,
    label_height: int = 28,
) -> None:
    from PIL import Image, ImageDraw

    if not image_paths:
        raise ValueError("No images supplied for mosaic.")

    image_paths = [Path(p) for p in image_paths]
    cols = cols or math.ceil(math.sqrt(len(image_paths)))
    rows = math.ceil(len(image_paths) / cols)
    per_tile_h = tile_size + label_height
    width = cols * tile_size + (cols - 1) * padding + 2 * margin
    height = rows * per_tile_h + (rows - 1) * padding + 2 * margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(11)

    for idx, image_path in enumerate(image_paths):
        row = idx // cols
        col = idx % cols
        x = margin + col * (tile_size + padding)
        y = margin + row * (per_tile_h + padding)
        with Image.open(image_path) as im:
            tile = im.convert("RGB").resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        canvas.paste(tile, (x, y))
        label = image_path.stem
        if len(label) > 44:
            label = label[:20] + "..." + label[-20:]
        draw.text((x + 2, y + tile_size + 4), label, fill="black", font=font)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
