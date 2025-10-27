#!/usr/bin/env python3
"""
Place custom STL plank electrodes in SimNIBS 4.5, auto-aligned along the scalp.
If a scalp surface is missing, extract it from a segmentation label map and
generate an STL on the fly.

New CLI:
  --seg SEG.nii.gz                # segmentation NIfTI (label map)
  --skin-labels 1,1000            # label IDs to treat as 'skin' (comma-separated)
  --skin-stl-out PATH.stl         # optional path to write the extracted skin STL
  --close-mm 2.0                  # morphological closing radius (mm) before surface
  --fill-holes                    # fill interior holes in the binary mask
  --smooth-iters 10               # Taubin/Laplacian smoothing iterations
  --decimate-frac 0.2             # target fraction to decimate (0.2 -> 80% triangles removed)

Core usage stays the same as before.
"""

from __future__ import annotations
import argparse
import os
import sys
import math
import numpy as np
from typing import Optional, Tuple, List

# Optional deps used in robust fallbacks
try:
    import nibabel as nib
except Exception:
    nib = None

try:
    from skimage import measure
except Exception:
    measure = None

try:
    import scipy.ndimage as ndi
except Exception:
    ndi = None

try:
    import trimesh
except Exception:
    trimesh = None

try:
    import meshio
except Exception:
    meshio = None

# SimNIBS
from simnibs import sim_struct


# ----------------- Small utilities -----------------

def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def _closest_vertex(vertices: np.ndarray, p: np.ndarray) -> int:
    diffs = vertices - p[None, :]
    i = np.argmin(np.einsum('ij,ij->i', diffs, diffs))
    return int(i)

def voxel_sizes_from_affine(aff: np.ndarray) -> np.ndarray:
    # Column vectors are voxel axes in world space
    return np.sqrt((aff[:3, :3] ** 2).sum(axis=0))

def ijk_to_xyz(aff: np.ndarray, ijk: np.ndarray) -> np.ndarray:
    """Convert Nx3 ijk to Nx3 xyz using affine."""
    ijk_h = np.c_[ijk, np.ones((ijk.shape[0], 1))]
    xyz_h = ijk_h @ aff.T
    return xyz_h[:, :3]

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ----------------- Skin surface: load/find/extract -----------------

def write_stl(vertices: np.ndarray, faces: np.ndarray, out_path: str):
    out_path = os.path.abspath(out_path)
    if trimesh is not None:
        m = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        m.export(out_path)
        return
    if meshio is not None:
        meshio.write(out_path, meshio.Mesh(points=vertices, cells=[('triangle', faces.astype(np.int32))]))
        return
    raise ImportError("Need either 'trimesh' or 'meshio' to write STL files.")

def smooth_and_decimate(vertices: np.ndarray, faces: np.ndarray,
                        smooth_iters: int = 10,
                        decimate_frac: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    if trimesh is None:
        return vertices, faces
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if smooth_iters > 0:
        try:
            trimesh.smoothing.filter_taubin(tm, iterations=smooth_iters)
        except Exception:
            try:
                trimesh.smoothing.filter_laplacian(tm, iterations=smooth_iters)
            except Exception:
                pass
    if decimate_frac and decimate_frac > 0:
        try:
            target_faces = max(1000, int(len(tm.faces) * (1.0 - float(decimate_frac))))
            tm = tm.simplify_quadratic_decimation(target_faces)
        except Exception:
            pass
    return tm.vertices.view(np.ndarray), tm.faces.view(np.ndarray)

def extract_skin_from_seg(seg_path: str,
                          skin_labels: List[int],
                          close_mm: float = 0.0,
                          fill_holes: bool = False,
                          smooth_iters: int = 10,
                          decimate_frac: float = 0.0,
                          out_stl: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Build a scalp/skin STL from a labeled segmentation (NIfTI).
    Returns (V, N, stl_path). Normals N are estimated from the surface.
    """
    if nib is None:
        raise ImportError("nibabel is required to read NIfTI segmentation (pip install nibabel)")
    if measure is None:
        raise ImportError("scikit-image is required for marching cubes (pip install scikit-image)")

    img = nib.load(seg_path)
    data = np.asanyarray(img.get_fdata())
    aff = img.affine

    # Binary mask for skin labels
    mask = np.isin(data, np.array(skin_labels, dtype=data.dtype))

    # Morphological cleanup (optional)
    if close_mm and close_mm > 0:
        if ndi is None:
            print("[warn] scipy not available; skipping closing", file=sys.stderr)
        else:
            # build a spherical struct elem roughly close_mm wide in voxels
            vox = voxel_sizes_from_affine(aff)
            rad_vox = np.maximum(1, (close_mm / np.mean(vox)))
            rad = int(round(float(rad_vox)))
            se = ndi.generate_binary_structure(3, 1)
            se = ndi.iterate_structure(se, rad)
            mask = ndi.binary_closing(mask, structure=se)

    if fill_holes:
        if ndi is None:
            print("[warn] scipy not available; skipping fill_holes", file=sys.stderr)
        else:
            mask = ndi.binary_fill_holes(mask)

    # marching cubes on binary volume -> vertices in voxel coords
    # Use level 0.5 to capture boundary of mask True
    verts_vox, faces, _, _ = measure.marching_cubes(mask.astype(np.uint8), level=0.5)

    # Convert to world coordinates
    verts_xyz = ijk_to_xyz(aff, verts_vox)

    # Optional smoothing/decimation
    verts_xyz, faces = smooth_and_decimate(verts_xyz, faces, smooth_iters=smooth_iters, decimate_frac=decimate_frac)

    # Estimate vertex normals (via trimesh if available)
    if trimesh is not None:
        tm = trimesh.Trimesh(vertices=verts_xyz, faces=faces, process=False)
        N = np.asarray(tm.vertex_normals, dtype=float)
    else:
        # crude normals via face-averaging
        N = np.zeros_like(verts_xyz)
        v0 = verts_xyz[faces[:, 0]]
        v1 = verts_xyz[faces[:, 1]]
        v2 = verts_xyz[faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)
        for f, (i, j, k) in enumerate(faces):
            N[i] += fn[f]; N[j] += fn[f]; N[k] += fn[f]
        N /= (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)

    # Write STL
    if out_stl is None:
        out_dir = os.path.abspath(os.path.join(os.getcwd(), "tdcs_custom_out"))
        ensure_dir(out_dir)
        out_stl = os.path.join(out_dir, "skin_extracted.stl")
    write_stl(verts_xyz, faces, out_stl)
    return verts_xyz, N, out_stl


# ----------------- Scalp vertices + normals loader -----------------

def load_scalp_vertices_normals(m2m_dir: str,
                                seg_path: Optional[str],
                                skin_labels: Optional[List[int]],
                                close_mm: float,
                                fill_holes: bool,
                                smooth_iters: int,
                                decimate_frac: float,
                                skin_stl_out: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Try to find an existing scalp surface under m2m_dir.
    If not found and seg_path is given, extract from segmentation and return that.
    """
    # 1) Search common places
    candidates = [
        os.path.join(m2m_dir, "skin", "skin.stl"),
        os.path.join(m2m_dir, "skin", "skin.surf.gii"),
        os.path.join(m2m_dir, "skin", "skin.gii"),
        os.path.join(m2m_dir, "head", "skin.stl"),
    ]
    surf_path = None
    for c in candidates:
        if os.path.isfile(c):
            surf_path = c
            break

    # 2) If still not found, glob anywhere under m2m_dir
    if surf_path is None:
        import glob
        globs = []
        globs += glob.glob(os.path.join(m2m_dir, "**", "skin*.stl"), recursive=True)
        globs += glob.glob(os.path.join(m2m_dir, "**", "*skin*.stl"), recursive=True)
        globs += glob.glob(os.path.join(m2m_dir, "**", "skin*.gii"), recursive=True)
        globs += glob.glob(os.path.join(m2m_dir, "**", "*skin*.surf.gii"), recursive=True)
        if globs:
            surf_path = os.path.abspath(globs[0])

    # 3) If still not found, and we have a segmentation, extract it
    if surf_path is None:
        if seg_path is None or skin_labels is None:
            raise FileNotFoundError(
                "No scalp surface found under m2m, and no segmentation parameters were provided. "
                "Pass --seg and --skin-labels to extract a scalp STL."
            )
        print("[info] No scalp surface found. Extracting from segmentation...")
        V, N, out_stl = extract_skin_from_seg(
            seg_path=seg_path,
            skin_labels=skin_labels,
            close_mm=close_mm,
            fill_holes=fill_holes,
            smooth_iters=smooth_iters,
            decimate_frac=decimate_frac,
            out_stl=skin_stl_out
        )
        print(f"[ok] Wrote extracted scalp STL: {out_stl}")
        return V, N

    # --------- Read found surface and estimate normals ----------
    path = surf_path
    if path.lower().endswith(".stl") and trimesh is not None:
        mesh = trimesh.load(path, process=True)
        V = mesh.vertices.view(np.ndarray)
        N = np.asarray(mesh.vertex_normals, dtype=float)
        if N is None or len(N) != len(V):
            mesh.fix_normals()
            N = np.asarray(mesh.vertex_normals, dtype=float)
        return V, N

    if meshio is None:
        raise ImportError("Need 'meshio' (or use STL + trimesh) to read the scalp surface.")

    m = meshio.read(path)
    V = np.asarray(m.points[:, :3], dtype=float)
    tri = None
    for cb in m.cells:
        if cb.type in ("triangle", "tri", "vtk_triangle"):
            tri = np.asarray(cb.data, dtype=int)
            break
    if tri is None:
        raise RuntimeError(f"No triangle cells in {path} to estimate normals.")
    N = np.zeros_like(V)
    v0, v1, v2 = V[tri[:,0]], V[tri[:,1]], V[tri[:,2]]
    face_n = np.cross(v1 - v0, v2 - v0)
    face_n /= (np.linalg.norm(face_n, axis=1, keepdims=True) + 1e-12)
    for f,(i,j,k) in enumerate(tri):
        N[i] += face_n[f]; N[j] += face_n[f]; N[k] += face_n[f]
    N /= (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
    return V, N


# ----------------- Geometry helpers -----------------

def project_vector_to_tangent(n: np.ndarray, v: np.ndarray) -> np.ndarray:
    n = _norm(n)
    return v - np.dot(v, n) * n

def local_rotation_degrees(center: np.ndarray, toward: np.ndarray,
                           V: np.ndarray, N: np.ndarray) -> float:
    idx = _closest_vertex(V, center)
    n = N[idx]
    d = project_vector_to_tangent(n, toward - center)
    d = _norm(d)
    if np.linalg.norm(d) == 0:
        return 0.0

    # Tangent basis
    global_x = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(global_x, n)) > 0.9:
        global_x = np.array([0.0, 1.0, 0.0])
    x_tan = _norm(project_vector_to_tangent(n, global_x))
    y_tan = _norm(np.cross(n, x_tan))

    x = float(np.dot(d, x_tan))
    y = float(np.dot(d, y_tan))
    ang_rad = math.atan2(y, x)
    return math.degrees(ang_rad)

def resolve_point(m2m_dir: str,
                  label: Optional[str],
                  xyz: Optional[List[float]]) -> np.ndarray:
    if xyz is not None:
        arr = np.asarray(xyz, dtype=float)
        if arr.shape != (3,):
            raise ValueError("xyz must be three numbers")
        return arr
    if label is None:
        raise ValueError("Need either an EEG label or explicit coordinates")
    try:
        from simnibs.utils.eeg_positions import eeg_pos_by_label  # type: ignore
        p = eeg_pos_by_label(m2m_dir, label)
        return np.asarray(p, dtype=float)
    except Exception as e:
        raise RuntimeError(
            f"Could not resolve EEG label '{label}'. Pass coordinates instead. Original error: {e}"
        )


# ----------------- Session builder -----------------

def build_session(args: argparse.Namespace) -> sim_struct.SESSION:
    # Load scalp vertices/normals; extract if necessary
    V, N = load_scalp_vertices_normals(
        m2m_dir=args.m2m,
        seg_path=args.seg,
        skin_labels=args.skin_labels,
        close_mm=args.close_mm,
        fill_holes=args.fill_holes,
        smooth_iters=args.smooth_iters,
        decimate_frac=args.decimate_frac,
        skin_stl_out=args.skin_stl_out
    )

    # Resolve points
    p_center = resolve_point(args.m2m, args.center, args.center_xyz)
    p_toward = resolve_point(args.m2m, args.toward, args.toward_xyz)

    # Rotation about local normal so long axis points toward p_toward
    rot_deg = local_rotation_degrees(p_center, p_toward, V, N)

    s = sim_struct.SESSION()
    s.subpath = args.m2m
    s.pathfem = args.out

    tdcs = s.add_tdcslist()

    # Main electrode: custom plank (gel + electrode)
    el = tdcs.add_electrode()
    el.name = args.name
    el.centre = p_center.tolist()
    el.shape = 'custom'
    el.electrode_surfaces = [os.path.abspath(args.el_stl)]
    el.gel_surfaces = [os.path.abspath(args.gel_stl)]
    el.gel = True
    el.gel_conductivity = float(args.gel_sigma)
    el.conductivity = float(args.el_sigma)
    el.rotation = float(rot_deg)
    el.current = float(args.current_ma)

    # Optional return electrode
    if args.return_center is not None or args.return_xyz is not None:
        ret = tdcs.add_electrode()
        ret.name = args.return_name
        if args.return_xyz is not None:
            ret.centre = np.asarray(args.return_xyz, dtype=float).tolist()
        else:
            ret.centre = resolve_point(args.m2m, args.return_center, None).tolist()

        if args.return_gel_stl and args.return_el_stl:
            ret.shape = 'custom'
            ret.electrode_surfaces = [os.path.abspath(args.return_el_stl)]
            ret.gel_surfaces = [os.path.abspath(args.return_gel_stl)]
            ret.gel = True
            ret.gel_conductivity = float(args.gel_sigma)
            ret.conductivity = float(args.el_sigma)
        else:
            ret.shape = 'ellipse'
            ret.dimensions = [35.0, 25.0]
            ret.gel = True
            ret.gel_thickness = 3.0
            ret.gel_conductivity = float(args.gel_sigma)
            ret.thickness = 1.5
            ret.conductivity = float(args.el_sigma)
            ret.rotation = 0.0

        ret.current = float(args.return_current_ma)

    return s


# ----------------- CLI -----------------

def parse_xyz(prefix: str, ns: argparse.Namespace) -> Optional[List[float]]:
    x = getattr(ns, f"{prefix}_x")
    y = getattr(ns, f"{prefix}_y")
    z = getattr(ns, f"{prefix}_z")
    if x is None and y is None and z is None:
        return None
    if None in (x, y, z):
        raise ValueError(f"Provide all three --{prefix}-x/--{prefix}-y/--{prefix}-z or none")
    return [float(x), float(y), float(z)]

def main():
    ap = argparse.ArgumentParser(description="Place custom plank electrodes in SimNIBS 4.5 (with optional skin extraction)")
    ap.add_argument("--m2m", required=True, help="Path to m2m_* subject folder")
    ap.add_argument("--out", required=True, help="Output FEM directory")
    ap.add_argument("--gel-stl", required=True, help="Gel STL (custom plank)")
    ap.add_argument("--el-stl", required=True, help="Electrode STL (custom plank)")
    ap.add_argument("--name", default="Plank_Main", help="Name for main electrode")

    # Center/toward (EEG or XYZ)
    ap.add_argument("--center", help="EEG label for center (e.g., AF4)")
    ap.add_argument("--toward", help="EEG label indicating longitudinal direction (e.g., PO4)")
    ap.add_argument("--center-x", type=float)
    ap.add_argument("--center-y", type=float)
    ap.add_argument("--center-z", type=float)
    ap.add_argument("--toward-x", type=float)
    ap.add_argument("--toward-y", type=float)
    ap.add_argument("--toward-z", type=float)

    ap.add_argument("--current-ma", type=float, default=2.0, help="Current at plank (mA)")
    ap.add_argument("--gel-sigma", type=float, default=1.4, help="Gel conductivity (S/m)")
    ap.add_argument("--el-sigma", type=float, default=0.1, help="Electrode conductivity (S/m)")

    # Optional return electrode
    ap.add_argument("--return-name", default="Return")
    ap.add_argument("--return-center", help="EEG label for return electrode center")
    ap.add_argument("--return-x", type=float)
    ap.add_argument("--return-y", type=float)
    ap.add_argument("--return-z", type=float)
    ap.add_argument("--return-gel-stl", help="Return gel STL (optional)")
    ap.add_argument("--return-el-stl", help="Return electrode STL (optional)")
    ap.add_argument("--return-current-ma", type=float, default=-2.0)

    # NEW: segmentation-based scalp extraction
    ap.add_argument("--seg", help="Segmentation NIfTI (label map) to extract skin if scalp surface is missing")
    ap.add_argument("--skin-labels", type=str, help="Comma-separated label integers treated as skin (e.g., '1' or '1,1000')")
    ap.add_argument("--skin-stl-out", help="Where to write extracted skin STL (defaults under --out)")

    ap.add_argument("--close-mm", type=float, default=0.0, help="Morphological closing radius (mm) before surface")
    ap.add_argument("--fill-holes", action="store_true", help="Fill interior holes in the mask before surface")
    ap.add_argument("--smooth-iters", type=int, default=10, help="Surface smoothing iterations (0=off)")
    ap.add_argument("--decimate-frac", type=float, default=0.0, help="Fraction to decimate faces (0.2 = remove ~80%)")

    args = ap.parse_args()

    # Parse XYZs
    args.center_xyz = parse_xyz("center", args)
    args.toward_xyz = parse_xyz("toward", args)
    args.return_xyz = parse_xyz("return", args)

    # Parse skin labels
    if args.skin_labels is not None:
        try:
            args.skin_labels = [int(t.strip()) for t in args.skin_labels.split(",") if t.strip() != ""]
            if not args.skin_labels:
                args.skin_labels = None
        except Exception:
            raise ValueError("Could not parse --skin-labels. Use comma-separated integers, e.g., '1' or '1,1000'.")
    else:
        args.skin_labels = None

    # Ensure output dir exists (also used for default skin STL path)
    ensure_dir(args.out)
    if args.skin_stl_out and not os.path.isabs(args.skin_stl_out):
        args.skin_stl_out = os.path.abspath(os.path.join(args.out, args.skin_stl_out))

    # Build & run
    sess = build_session(args)
    print("[info] Running SimNIBS session...")
    sess.run()
    print(f"[ok] Finished. Results in: {args.out}")

if __name__ == "__main__":
    main()
