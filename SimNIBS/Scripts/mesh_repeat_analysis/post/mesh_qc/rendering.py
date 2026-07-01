from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .loaders import load_surface_arrays


def render_mesh_png(
    path: Path,
    out_png: Path,
    *,
    label: str,
    image_size: int = 600,
    renderer: str = "auto",
    max_faces: int = 12000,
) -> None:
    renderer = renderer.lower()
    if renderer not in {"auto", "pyvista", "pillow"}:
        raise ValueError(f"Unsupported renderer: {renderer}")
    if renderer in {"auto", "pyvista"}:
        try:
            _render_with_pyvista(path, out_png, label=label, image_size=image_size)
            return
        except Exception:
            if renderer == "pyvista":
                raise
    _render_with_pillow(path, out_png, label=label, image_size=image_size, max_faces=max_faces)


def _render_with_pyvista(path: Path, out_png: Path, *, label: str, image_size: int = 600) -> None:
    import pyvista as pv

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


def _render_with_pillow(
    path: Path,
    out_png: Path,
    *,
    label: str,
    image_size: int = 600,
    max_faces: int = 12000,
) -> None:
    from PIL import Image, ImageDraw

    surface = load_surface_arrays(path)
    points = np.asarray(surface.points, dtype=float)
    faces = np.asarray(surface.faces, dtype=np.int64)
    if points.size == 0 or faces.size == 0:
        raise RuntimeError("Surface extraction produced an empty mesh.")

    valid = np.all((faces >= 0) & (faces < len(points)), axis=1)
    faces = faces[valid]
    if len(faces) == 0:
        raise RuntimeError("Surface extraction produced no valid triangular faces.")
    if len(faces) > max_faces:
        take = np.linspace(0, len(faces) - 1, num=max_faces, dtype=np.int64)
        faces = faces[take]

    centered = points - ((points.min(axis=0) + points.max(axis=0)) * 0.5)
    theta = math.radians(45.0)
    phi = math.radians(35.264)
    rz = np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(phi), -math.sin(phi)],
            [0.0, math.sin(phi), math.cos(phi)],
        ]
    )
    rotated = centered @ rz.T @ rx.T
    xy = rotated[:, :2]
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    span = max(float((maxs - mins).max()), 1e-9)
    pad = max(18, int(image_size * 0.08))
    scale = (image_size - 2 * pad) / span
    pix = (xy - (mins + maxs) * 0.5) * scale
    pix[:, 0] += image_size * 0.5
    pix[:, 1] = image_size * 0.5 - pix[:, 1]

    tri_rot = rotated[faces]
    normals = np.cross(tri_rot[:, 1] - tri_rot[:, 0], tri_rot[:, 2] - tri_rot[:, 0])
    norm_len = np.linalg.norm(normals, axis=1)
    shade = np.divide(np.abs(normals[:, 2]), norm_len, out=np.zeros_like(norm_len), where=norm_len > 0)
    depth = tri_rot[:, :, 2].mean(axis=1)
    order = np.argsort(depth)

    image = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(image)
    outline = (80, 80, 80) if len(faces) <= 5000 else None
    for face_idx in order:
        polygon = [tuple(map(float, pix[vertex])) for vertex in faces[face_idx]]
        gray = int(150 + 80 * shade[face_idx])
        draw.polygon(polygon, fill=(gray, gray, gray), outline=outline)

    if label:
        font = _font(max(10, image_size // 42))
        lines = label.splitlines()
        line_height = max(12, image_size // 32)
        box_h = 8 + line_height * len(lines)
        draw.rectangle((0, 0, image_size, box_h), fill=(255, 255, 255))
        for idx, line in enumerate(lines):
            draw.text((6, 4 + idx * line_height), line, fill=(0, 0, 0), font=font)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_png)


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
