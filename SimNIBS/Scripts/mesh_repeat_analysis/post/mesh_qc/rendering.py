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
    plotter.camera.parallel_projection = True
    plotter.camera.focal_point = (0.0, 0.0, 0.0)
    plotter.camera.position = (0.0, span * 2.5, 0.0)
    plotter.camera.view_up = (0.0, 0.0, 1.0)
    plotter.camera.parallel_scale = span * 0.62
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
    from PIL import Image, ImageDraw, ImageFilter

    surface = load_surface_arrays(path)
    points = np.asarray(surface.points, dtype=float)
    faces = np.asarray(surface.faces, dtype=np.int64)
    if points.size == 0 or faces.size == 0:
        raise RuntimeError("Surface extraction produced an empty mesh.")

    valid = np.all((faces >= 0) & (faces < len(points)), axis=1)
    faces = faces[valid]
    if len(faces) == 0:
        raise RuntimeError("Surface extraction produced no valid triangular faces.")
    rotated = _front_view(points)
    pix = _fit_pixels(rotated[:, :2], image_size=image_size)

    if len(faces) <= max_faces:
        image = _draw_triangle_surface(rotated, pix, faces, image_size=image_size)
    else:
        image = _draw_dense_surface_preview(
            rotated,
            pix,
            faces,
            image_size=image_size,
            sample_face_limit=max_faces,
        )

    if label:
        draw = ImageDraw.Draw(image)
        font = _font(max(10, image_size // 42))
        lines = label.splitlines()
        line_height = max(12, image_size // 32)
        box_h = 8 + line_height * len(lines)
        draw.rectangle((0, 0, image_size, box_h), fill=(255, 255, 255))
        for idx, line in enumerate(lines):
            draw.text((6, 4 + idx * line_height), line, fill=(0, 0, 0), font=font)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_png)


def _front_view(points: np.ndarray) -> np.ndarray:
    centered = points - ((points.min(axis=0) + points.max(axis=0)) * 0.5)
    return np.column_stack((centered[:, 0], centered[:, 2], centered[:, 1]))


def _fit_pixels(xy: np.ndarray, *, image_size: int) -> np.ndarray:
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    span = max(float((maxs - mins).max()), 1e-9)
    pad = max(24, int(image_size * 0.08))
    scale = (image_size - 2 * pad) / span
    pix = (xy - (mins + maxs) * 0.5) * scale
    pix[:, 0] += image_size * 0.5
    pix[:, 1] = image_size * 0.5 - pix[:, 1]
    return pix


def _draw_triangle_surface(
    rotated: np.ndarray,
    pix: np.ndarray,
    faces: np.ndarray,
    *,
    image_size: int,
):
    from PIL import Image, ImageDraw

    tri_rot = rotated[faces]
    normals = np.cross(tri_rot[:, 1] - tri_rot[:, 0], tri_rot[:, 2] - tri_rot[:, 0])
    norm_len = np.linalg.norm(normals, axis=1)
    shade = np.divide(np.abs(normals[:, 2]), norm_len, out=np.zeros_like(norm_len), where=norm_len > 0)
    depth = tri_rot[:, :, 2].mean(axis=1)
    order = np.argsort(depth)

    image = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(image)
    outline = (92, 92, 92) if len(faces) <= 2500 else None
    for face_idx in order:
        polygon = [tuple(map(float, pix[vertex])) for vertex in faces[face_idx]]
        gray = int(150 + 75 * shade[face_idx])
        draw.polygon(polygon, fill=(gray, gray, gray), outline=outline)
    return image


def _draw_dense_surface_preview(
    rotated: np.ndarray,
    pix: np.ndarray,
    faces: np.ndarray,
    *,
    image_size: int,
    sample_face_limit: int,
):
    from PIL import Image, ImageFilter

    surface_vertex_ids = np.unique(faces.ravel())
    sample_xy = pix[surface_vertex_ids]
    sample_depth = rotated[surface_vertex_ids, 2]

    centroid_count = min(len(faces), max(sample_face_limit, 4000))
    if centroid_count > 0:
        if centroid_count == len(faces):
            centroid_faces = faces
        else:
            take = np.linspace(0, len(faces) - 1, num=centroid_count, dtype=np.int64)
            centroid_faces = faces[take]
        centroid_xy = pix[centroid_faces].mean(axis=1)
        centroid_depth = rotated[centroid_faces, 2].mean(axis=1)
        sample_xy = np.vstack((sample_xy, centroid_xy))
        sample_depth = np.concatenate((sample_depth, centroid_depth))

    depth_map = _splat_depth_map(sample_xy, sample_depth, image_size=image_size)
    valid = np.isfinite(depth_map)
    if not np.any(valid):
        raise RuntimeError("Surface preview projection produced an empty image.")

    finite = depth_map[valid]
    lo = float(finite.min())
    hi = float(finite.max())
    span = max(hi - lo, 1e-6)

    normalized = (depth_map - lo) / span
    normalized[~valid] = 0.0
    grad_y, grad_x = np.gradient(normalized)
    normal_z = 1.0 / np.sqrt(grad_x * grad_x + grad_y * grad_y + 1.0)
    normal_x = -grad_x * normal_z
    normal_y = -grad_y * normal_z
    light = np.array([0.35, -0.25, 0.90], dtype=np.float32)
    light /= np.linalg.norm(light)
    shade = np.clip(normal_x * light[0] + normal_y * light[1] + normal_z * light[2], 0.0, 1.0)

    gray = (145 + 95 * shade).astype(np.uint8)
    gray[~valid] = 255
    rgb = np.dstack((gray, gray, gray))

    mask = (valid.astype(np.uint8) * 255)
    eroded = np.asarray(Image.fromarray(mask, mode="L").filter(ImageFilter.MinFilter(3)))
    edge = (mask > 0) & (eroded == 0)
    rgb[edge] = np.array([88, 88, 88], dtype=np.uint8)

    return Image.fromarray(rgb, mode="RGB")


def _splat_depth_map(sample_xy: np.ndarray, sample_depth: np.ndarray, *, image_size: int) -> np.ndarray:
    projected_area = max(
        float(np.ptp(sample_xy[:, 0]) * np.ptp(sample_xy[:, 1])),
        1.0,
    )
    spacing = math.sqrt(projected_area / max(len(sample_xy), 1))
    radius = max(2, min(5, int(round(spacing * 0.7))))

    depth_map = np.full((image_size, image_size), -np.inf, dtype=np.float32)
    ix = np.rint(sample_xy[:, 0]).astype(np.int32)
    iy = np.rint(sample_xy[:, 1]).astype(np.int32)

    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                offsets.append((dx, dy))

    for dx, dy in offsets:
        x = ix + dx
        y = iy + dy
        valid = (x >= 0) & (x < image_size) & (y >= 0) & (y < image_size)
        if np.any(valid):
            np.maximum.at(depth_map, (y[valid], x[valid]), sample_depth[valid])

    valid = np.isfinite(depth_map)
    for _ in range(radius + 2):
        if np.all(valid):
            break
        padded = np.pad(depth_map, 1, mode="constant", constant_values=-np.inf)
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbors.append(padded[1 + dy : 1 + dy + image_size, 1 + dx : 1 + dx + image_size])
        neighbor_max = np.maximum.reduce(neighbors)
        grow = (~valid) & np.isfinite(neighbor_max)
        if not np.any(grow):
            break
        depth_map[grow] = neighbor_max[grow]
        valid = np.isfinite(depth_map)

    return depth_map


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
