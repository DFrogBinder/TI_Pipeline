#!/usr/bin/env python3
"""
msh_qc_wall_subject_scan.py

Two modes:

1) Flat mode (legacy):
   --msh-dir /path/with/msh_files --pattern "*.msh"

2) Subject-root mode (requested):
   --subjects-root /path/to/dataset_root --target-filename "SAME_NAME_FOR_ALL.msh"
   For each immediate subject directory under subjects-root, recursively search
   for the target filename. If found, render it.

Outputs:
  <out_dir>/renders/*.png
  <out_dir>/mosaic.png
  <out_dir>/found_paths.csv   (subject -> found msh path)
  <out_dir>/failures.txt      (subjects missing file or render failures)
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import meshio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import pyvista as pv
import gmsh


# ----------------------------
# Mesh utilities
# ----------------------------

def _pick_scalar_name(mesh: pv.DataSet, preferred: Optional[str]) -> Optional[Tuple[str, str]]:
    """
    Returns (association, name) where association is 'cell' or 'point'.
    """
    if preferred:
        if preferred in mesh.cell_data:
            return ("cell", preferred)
        if preferred in mesh.point_data:
            return ("point", preferred)

    common = [
        "gmsh:physical",
        "gmsh:geometrical",
        "PhysicalIds",
        "GeometryIds",
        "tag",
        "labels",
        "region",
        "tissue",
        "ElementData",
    ]

    for name in common:
        if name in mesh.cell_data:
            return ("cell", name)
        if name in mesh.point_data:
            return ("point", name)

    if len(mesh.cell_data.keys()) > 0:
        return ("cell", list(mesh.cell_data.keys())[0])
    if len(mesh.point_data.keys()) > 0:
        return ("point", list(mesh.point_data.keys())[0])

    return None


def _read_msh_to_pyvista(path: Path) -> pv.DataSet:
    try:
        m = meshio.read(str(path))
        return pv.from_meshio(m)
    except Exception as exc:
        try:
            return _read_msh_to_pyvista_gmsh(path)
        except Exception as gmsh_exc:
            raise RuntimeError(
                f"meshio read failed: {exc}; gmsh fallback failed: {gmsh_exc}"
            ) from gmsh_exc


def _read_msh_to_pyvista_gmsh(path: Path) -> pv.PolyData:
    initialized_here = False
    if not gmsh.isInitialized():
        gmsh.initialize()
        initialized_here = True
        gmsh.option.setNumber("General.Terminal", 0)

    try:
        gmsh.clear()
        gmsh.open(str(path))

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        if len(node_tags) == 0:
            raise RuntimeError("No nodes found in mesh.")

        points = np.asarray(node_coords, dtype=float).reshape(-1, 3)
        tag_to_index = {int(tag): idx for idx, tag in enumerate(node_tags)}

        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        if not elem_types:
            raise RuntimeError("No surface (2D) elements found in mesh.")

        triangles = []
        for elem_type, nodes in zip(elem_types, elem_node_tags):
            name, dim, order, num_nodes, _, num_primary = gmsh.model.mesh.getElementProperties(elem_type)
            if dim != 2 or num_primary < 3:
                continue

            nodes = np.asarray(nodes, dtype=int).reshape(-1, num_nodes)
            if num_primary == 3:
                tri_nodes = nodes[:, :3]
            elif num_primary == 4:
                quad_nodes = nodes[:, :4]
                tri_nodes = np.vstack([quad_nodes[:, [0, 1, 2]], quad_nodes[:, [0, 2, 3]]])
            else:
                continue

            tri_idx = np.fromiter(
                (tag_to_index[int(t)] for t in tri_nodes.ravel()),
                dtype=np.int64,
                count=tri_nodes.size,
            ).reshape(tri_nodes.shape)
            triangles.append(tri_idx)

        if not triangles:
            raise RuntimeError("No usable surface elements found for rendering.")

        tri_idx = np.vstack(triangles)
        faces = np.hstack([np.full((tri_idx.shape[0], 1), 3, dtype=np.int64), tri_idx]).ravel()
        return pv.PolyData(points, faces)
    finally:
        gmsh.clear()
        if initialized_here:
            gmsh.finalize()


def _render_mesh(
    pv_mesh: pv.UnstructuredGrid,
    out_png: Path,
    *,
    scalar_preference: Optional[str] = None,
    show_edges: bool = True,
    edge_color: str = "black",
    background: str = "white",
    image_size: int = 800,
    parallel_projection: bool = True,
    overlay_text: Optional[str] = None,
    font_size: int = 18,
) -> None:
    surf = pv_mesh.extract_surface().triangulate()
    if surf.n_points == 0:
        raise RuntimeError("Surface extraction produced an empty mesh.")

    scalar_choice = _pick_scalar_name(surf, scalar_preference)

    plotter = pv.Plotter(off_screen=True, window_size=(image_size, image_size))
    plotter.set_background(background)

    # Consistent camera
    plotter.camera_position = "iso"
    if parallel_projection:
        plotter.camera.parallel_projection = True

    if scalar_choice is None:
        plotter.add_mesh(
            surf,
            color="lightgray",
            show_edges=show_edges,
            edge_color=edge_color,
            smooth_shading=True,
        )
    else:
        assoc, name = scalar_choice
        scalars = surf.cell_data[name] if assoc == "cell" else surf.point_data[name]
        plotter.add_mesh(
            surf,
            scalars=scalars,
            show_edges=show_edges,
            edge_color=edge_color,
            smooth_shading=True,
        )

    if overlay_text:
        plotter.add_text(overlay_text, position="upper_left", font_size=font_size, color="black")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plotter.show(auto_close=False)
    plotter.screenshot(str(out_png))
    plotter.close()


# ----------------------------
# Mosaic utilities
# ----------------------------

def _auto_grid(n: int) -> Tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return cols, rows


def _load_font(font_size: int) -> Optional[ImageFont.FreeTypeFont]:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except Exception:
            return None


def _make_mosaic(
    images: List[Path],
    out_path: Path,
    *,
    cols: Optional[int] = None,
    tile_size: int = 800,
    padding: int = 20,
    margin: int = 30,
    background: str = "white",
    label: bool = True,
    label_font_size: int = 18,
    label_height: int = 40,
) -> None:
    n = len(images)
    if n == 0:
        raise ValueError("No images provided for mosaic.")

    if cols is None:
        cols, rows = _auto_grid(n)
    else:
        rows = math.ceil(n / cols)

    font = _load_font(label_font_size)
    per_tile_h = tile_size + (label_height if label else 0)

    mosaic_w = (cols * tile_size) + ((cols - 1) * padding) + (2 * margin)
    mosaic_h = (rows * per_tile_h) + ((rows - 1) * padding) + (2 * margin)

    canvas = Image.new("RGB", (mosaic_w, mosaic_h), color=background)
    draw = ImageDraw.Draw(canvas)

    for idx, img_path in enumerate(images):
        r = idx // cols
        c = idx % cols

        x0 = margin + c * (tile_size + padding)
        y0 = margin + r * (per_tile_h + padding)

        img = Image.open(img_path).convert("RGB")
        img = img.resize((tile_size, tile_size), resample=Image.Resampling.LANCZOS)
        canvas.paste(img, (x0, y0))

        if label:
            name = img_path.stem
            if len(name) > 60:
                name = name[:57] + "..."
            text_y = y0 + tile_size + 6
            if font:
                draw.text((x0, text_y), name, fill="black", font=font)
            else:
                draw.text((x0, text_y), name, fill="black")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG", optimize=False)


# ----------------------------
# Scanning logic (new)
# ----------------------------

def _list_subject_dirs(subjects_root: Path) -> List[Path]:
    """
    Treat immediate subdirectories of subjects_root as subjects.
    (This matches typical datasets: root/sub-XXXX/...)
    """
    return sorted([p for p in subjects_root.iterdir() if p.is_dir()])


def _find_target_file_in_subject(subject_dir: Path, target_filename: str) -> Optional[Path]:
    """
    Recursively search for target_filename under subject_dir.
    Returns the first match (sorted for determinism) or None.
    """
    matches = sorted(subject_dir.rglob(target_filename))
    return matches[0] if matches else None


def _write_found_paths_csv(rows: List[Dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "msh_path"])
        w.writeheader()
        for row in rows:
            w.writerow(row)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Render per-subject .msh (found by recursive search) and stitch into a QC mosaic."
    )

    # Mode A: subject-root scanning (new)
    p.add_argument("--subjects-root", default=None, help="Root folder containing subject directories (immediate children).")
    p.add_argument("--target-filename", default=None, help="Exact filename to find within each subject tree (same for all).")

    # Mode B: flat directory mode (legacy)
    p.add_argument("--msh-dir", default=None, help="Folder containing .msh files directly (legacy mode).")
    p.add_argument("--pattern", default="*.msh", help="Glob pattern for legacy mode (default: *.msh).")

    # Common
    p.add_argument("--out", dest="out_dir", required=True, help="Output folder for renders and mosaic.")

    # Rendering
    p.add_argument("--image-size", type=int, default=800, help="Per-mesh screenshot size (px).")
    p.add_argument("--no-edges", action="store_true", help="Disable edge overlay.")
    p.add_argument("--scalar", default=None, help="Preferred scalar field for coloring (optional).")
    p.add_argument("--background", default="white", help="Background color.")
    p.add_argument("--no-parallel", action="store_true", help="Disable parallel projection.")

    # Mosaic
    p.add_argument("--cols", type=int, default=None, help="Columns in mosaic (default: auto).")
    p.add_argument("--tile-size", type=int, default=800, help="Tile size in mosaic (px).")
    p.add_argument("--padding", type=int, default=20, help="Padding between tiles.")
    p.add_argument("--margin", type=int, default=30, help="Outer margin.")
    p.add_argument("--no-labels", action="store_true", help="Disable filename labels under tiles.")
    p.add_argument("--label-font-size", type=int, default=18, help="Label font size.")
    p.add_argument("--label-height", type=int, default=40, help="Reserved label height under each tile.")
    p.add_argument("--subject-labels", action="store_true",
                   help="In subject-root mode, use subject name as the PNG name and overlay label.")

    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    renders_dir = out_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    # PyVista settings
    pv.global_theme.smooth_shading = True

    # Determine mode
    subject_mode = args.subjects_root is not None or args.target_filename is not None
    legacy_mode = args.msh_dir is not None

    if subject_mode and legacy_mode:
        raise ValueError("Choose either --subjects-root/--target-filename OR --msh-dir (not both).")

    rendered_pngs: List[Path] = []
    failures: List[str] = []
    found_rows: List[Dict[str, str]] = []

    if subject_mode:
        if args.subjects_root is None or args.target_filename is None:
            raise ValueError("Subject-root mode requires BOTH --subjects-root and --target-filename.")

        subjects_root = Path(args.subjects_root).expanduser().resolve()
        if not subjects_root.exists():
            raise FileNotFoundError(f"subjects-root does not exist: {subjects_root}")

        subject_dirs = _list_subject_dirs(subjects_root)
        if not subject_dirs:
            raise FileNotFoundError(f"No subject directories found under: {subjects_root}")

        print(f"[INFO] Subject-root mode: {len(subject_dirs)} subjects under {subjects_root}")
        print(f"[INFO] Target filename: {args.target_filename}")

        for i, subj_dir in enumerate(subject_dirs, start=1):
            subject = subj_dir.name
            print(f"[{i:04d}/{len(subject_dirs):04d}] Scanning {subject} ...")

            msh_path = _find_target_file_in_subject(subj_dir, args.target_filename)
            if msh_path is None:
                msg = f"[MISSING] {subject}: could not find {args.target_filename}"
                print(msg)
                failures.append(msg)
                continue

            found_rows.append({"subject": subject, "msh_path": str(msh_path)})

            # Output naming:
            # - default: include subject + mesh stem to avoid collisions
            # - if --subject-labels: use subject name as primary
            if args.subject_labels:
                png_name = f"{subject}.png"
                overlay = subject
            else:
                png_name = f"{subject}__{msh_path.stem}.png"
                overlay = f"{subject}\n{msh_path.name}"

            out_png = renders_dir / png_name

            try:
                pv_mesh = _read_msh_to_pyvista(msh_path)
                _render_mesh(
                    pv_mesh,
                    out_png,
                    scalar_preference=args.scalar,
                    show_edges=(not args.no_edges),
                    background=args.background,
                    image_size=args.image_size,
                    parallel_projection=(not args.no_parallel),
                    overlay_text=overlay,
                )
                rendered_pngs.append(out_png)
            except Exception as e:
                msg = f"[FAIL] {subject}: render failed for {msh_path} ({e})"
                print(msg)
                failures.append(msg)

        # Write a CSV mapping subject -> mesh path for traceability
        _write_found_paths_csv(found_rows, out_dir / "found_paths.csv")

    elif legacy_mode:
        msh_dir = Path(args.msh_dir).expanduser().resolve()
        if not msh_dir.exists():
            raise FileNotFoundError(f"msh-dir does not exist: {msh_dir}")

        msh_files = sorted(msh_dir.glob(args.pattern))
        if not msh_files:
            raise FileNotFoundError(f"No files matched {args.pattern} in {msh_dir}")

        print(f"[INFO] Legacy mode: Found {len(msh_files)} meshes in {msh_dir}")

        for i, msh_path in enumerate(msh_files, start=1):
            out_png = renders_dir / f"{msh_path.stem}.png"
            print(f"[{i:04d}/{len(msh_files):04d}] Rendering {msh_path.name}")

            try:
                pv_mesh = _read_msh_to_pyvista(msh_path)
                _render_mesh(
                    pv_mesh,
                    out_png,
                    scalar_preference=args.scalar,
                    show_edges=(not args.no_edges),
                    background=args.background,
                    image_size=args.image_size,
                    parallel_projection=(not args.no_parallel),
                    overlay_text=msh_path.stem,
                )
                rendered_pngs.append(out_png)
            except Exception as e:
                msg = f"[FAIL] {msh_path}: {e}"
                print(msg)
                failures.append(msg)

    else:
        raise ValueError("Specify either --subjects-root/--target-filename OR --msh-dir.")

    if not rendered_pngs:
        raise RuntimeError("No renders succeeded; cannot build mosaic.")

    mosaic_path = out_dir / "mosaic.png"
    print(f"[INFO] Building mosaic -> {mosaic_path}")

    _make_mosaic(
        rendered_pngs,
        mosaic_path,
        cols=args.cols,
        tile_size=args.tile_size,
        padding=args.padding,
        margin=args.margin,
        background=args.background,
        label=(not args.no_labels),
        label_font_size=args.label_font_size,
        label_height=args.label_height,
    )

    print(f"[DONE] Mosaic saved: {mosaic_path}")

    if failures:
        fail_log = out_dir / "failures.txt"
        with open(fail_log, "w", encoding="utf-8") as f:
            for line in failures:
                f.write(line + "\n")
        print(f"[WARN] Some subjects failed/missing. See: {fail_log}")


if __name__ == "__main__":
    main()
