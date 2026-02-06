#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render multiple .msh files into a single mosaic image for quick inspection.

Requires: pyvista (and meshio for .msh reading) and pillow.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np


def _load_mesh(path: Path):
    try:
        import pyvista as pv
    except Exception as exc:
        raise RuntimeError("pyvista is required. Install with: pip install pyvista") from exc

    try:
        mesh = pv.read(str(path))
    except Exception:
        try:
            import meshio
        except Exception as exc:
            raise RuntimeError(
                "Failed to read .msh with pyvista; install meshio: pip install meshio"
            ) from exc
        mesh = pv.wrap(meshio.read(str(path)))

    if not isinstance(mesh, pv.DataSet):
        raise RuntimeError(f"Unsupported mesh type for {path}")

    # Use a surface for consistent rendering
    if hasattr(mesh, "extract_surface"):
        mesh = mesh.extract_surface().triangulate()
    return mesh


def _center_mesh(mesh):
    center = np.array(mesh.center)
    if mesh.n_points:
        mesh.points = mesh.points - center
    return mesh


def _max_radius(mesh) -> float:
    if mesh.n_points == 0:
        return 1.0
    pts = np.asarray(mesh.points)
    return float(np.linalg.norm(pts, axis=1).max())


def _camera_from_view(view: str, distance: float):
    view = view.lower()
    if view == "xy":
        pos = (0.0, 0.0, distance)
        viewup = (0.0, 1.0, 0.0)
    elif view == "xz":
        pos = (0.0, distance, 0.0)
        viewup = (0.0, 0.0, 1.0)
    elif view == "yz":
        pos = (distance, 0.0, 0.0)
        viewup = (0.0, 0.0, 1.0)
    else:  # iso
        pos = (distance, distance, distance)
        viewup = (0.0, 0.0, 1.0)
    return [pos, (0.0, 0.0, 0.0), viewup]


def _render_mesh(mesh, *, size, bg, show_edges, camera_position):
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True, window_size=size)
    plotter.set_background(bg)
    plotter.add_mesh(
        mesh,
        color="lightgray",
        smooth_shading=True,
        show_edges=show_edges,
        edge_color="black",
    )
    plotter.camera_position = camera_position
    plotter.camera.SetParallelProjection(False)
    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img


def _tile_images(images, *, cols, margin, label_paths, label_size):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        raise RuntimeError("Pillow is required. Install with: pip install pillow") from exc

    if not images:
        raise RuntimeError("No images to tile.")

    h, w = images[0].shape[0], images[0].shape[1]
    rows = math.ceil(len(images) / cols)
    out_w = cols * w + margin * (cols + 1)
    out_h = rows * h + margin * (rows + 1)

    mosaic = Image.new("RGB", (out_w, out_h), color="white")
    draw = ImageDraw.Draw(mosaic)

    font = None
    if label_paths:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x0 = margin + col * (w + margin)
        y0 = margin + row * (h + margin)
        tile = Image.fromarray(img)
        mosaic.paste(tile, (x0, y0))

        if label_paths:
            label = label_paths[idx]
            label = label if len(label) <= 60 else "…" + label[-59:]
            if font:
                draw.rectangle(
                    [x0, y0, x0 + w, y0 + label_size + 4],
                    fill=(255, 255, 255),
                )
                draw.text((x0 + 4, y0 + 2), label, fill=(0, 0, 0), font=font)

    return mosaic


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Render multiple .msh files into a single mosaic image."
    )
    parser.add_argument("msh", nargs="*", help="Paths to .msh files.")
    parser.add_argument("--glob", dest="glob", default=None, help="Glob for .msh files.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--cols", type=int, default=None, help="Number of columns.")
    parser.add_argument("--img-size", default="400x400", help="Per-mesh render size, e.g. 400x400.")
    parser.add_argument("--margin", type=int, default=12, help="Margin between tiles.")
    parser.add_argument("--bg", default="white", help="Background color.")
    parser.add_argument("--edges", action="store_true", help="Show mesh edges.")
    parser.add_argument("--view", default="iso", choices=["iso", "xy", "xz", "yz"], help="Camera view.")
    parser.add_argument("--zoom", type=float, default=2.5, help="Camera distance multiplier.")
    parser.add_argument("--label", action="store_true", help="Overlay file names on tiles.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    msh_paths: list[Path] = []
    if args.glob:
        msh_paths.extend(sorted(Path().glob(args.glob)))
    msh_paths.extend(Path(p) for p in args.msh)
    msh_paths = [p for p in msh_paths if p.exists()]

    if not msh_paths:
        print("No .msh files found. Provide paths or --glob.")
        return 2

    try:
        w_str, h_str = args.img_size.lower().split("x")
        size = (int(w_str), int(h_str))
    except Exception:
        print("Invalid --img-size. Use format like 400x400.")
        return 2

    meshes = []
    radii = []
    for path in msh_paths:
        mesh = _center_mesh(_load_mesh(path))
        meshes.append(mesh)
        radii.append(_max_radius(mesh))

    max_radius = max(radii) if radii else 1.0
    distance = max(1.0, max_radius * args.zoom)
    camera_position = _camera_from_view(args.view, distance)

    images = []
    for mesh in meshes:
        img = _render_mesh(
            mesh,
            size=size,
            bg=args.bg,
            show_edges=args.edges,
            camera_position=camera_position,
        )
        images.append(img)

    cols = args.cols or math.ceil(math.sqrt(len(images)))
    labels = [p.name for p in msh_paths] if args.label else None
    mosaic = _tile_images(
        images,
        cols=cols,
        margin=args.margin,
        label_paths=labels,
        label_size=14,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(output_path)
    print(f"Saved mosaic to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
