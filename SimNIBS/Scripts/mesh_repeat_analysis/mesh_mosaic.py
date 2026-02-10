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
        description=(
            "Render a mosaic for meshes found under <root>/repeats/**/<mesh-name>."
        )
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Root folder that contains the repeats/ directory (default: current dir).",
    )
    parser.add_argument(
        "--mesh-name",
        required=True,
        help="Mesh filename to look for under repeats (e.g., TI.msh).",
    )
    parser.add_argument(
        "--output",
        default="mesh_mosaic.png",
        help="Output PNG path (default: mesh_mosaic.png).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    root = Path(args.root)
    repeats_dir = root / "repeats"
    if not repeats_dir.exists():
        print(f"Missing repeats directory: {repeats_dir}")
        return 2

    pattern = f"**/{args.mesh_name}"
    msh_paths = sorted(p for p in repeats_dir.glob(pattern) if p.is_file())

    if not msh_paths:
        print(f"No meshes found matching {args.mesh_name} under {repeats_dir}")
        return 2

    size = (400, 400)

    meshes = []
    radii = []
    for path in msh_paths:
        mesh = _center_mesh(_load_mesh(path))
        meshes.append(mesh)
        radii.append(_max_radius(mesh))

    max_radius = max(radii) if radii else 1.0
    distance = max(1.0, max_radius * 2.5)
    camera_position = _camera_from_view("iso", distance)

    images = []
    for mesh in meshes:
        img = _render_mesh(
            mesh,
            size=size,
            bg="white",
            show_edges=True,
            camera_position=camera_position,
        )
        images.append(img)

    cols = math.ceil(math.sqrt(len(images)))
    labels = [p.parent.parent.name for p in msh_paths]
    mosaic = _tile_images(
        images,
        cols=cols,
        margin=12,
        label_paths=labels,
        label_size=14,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(output_path)
    print(f"Saved mosaic to {output_path} ({len(images)} meshes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
