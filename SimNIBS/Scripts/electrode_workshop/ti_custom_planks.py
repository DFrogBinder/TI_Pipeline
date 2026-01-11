#!/usr/bin/env python3
"""
Temporal Interference with custom plank electrodes in SimNIBS 4.5.

- Uses your plank placement helpers (rotation via scalp normals, EEG label resolution)
  from `place_plank_electrode.py`.  (AF4→PO4, AF3→PO3 by default)
- Builds two tdcs montages (4 planks total), runs SimNIBS, computes TImax, exports NIfTI,
  and generates overlay PNGs via `functions.py`.

Requirements: SimNIBS 4.5 env, your three files in PYTHONPATH or same folder:
  - place_plank_electrode.py  (reused helpers)  [projection/rotation, etc.]
  - functions.py              (overlay helpers) [png overlays]

Author: you+me :)
"""

from __future__ import annotations
import os
import argparse
import math
from copy import deepcopy

import numpy as np
import nibabel as nib

import simnibs as sim
from simnibs import sim_struct, mesh_io, ElementTags
from simnibs.utils import TI_utils as TI

from place_plank_electrode import (
    load_scalp_vertices_normals,
    resolve_point,
    local_rotation_degrees,
    ensure_dir,
)

# Overlay helpers for PNGs
from functions import make_overlay_png  # nice colourbar behavior
try:
    import trimesh
    try:
        _ = trimesh.util.bounds_tree  # triggers rtree import on demand
        import rtree  # noqa: F401
    except Exception:
        print("[hint] Fast nearest-point lookups need 'rtree'. Install with: pip install rtree")
        raise
except ImportError:
    raise ImportError("Please install 'trimesh' (and 'rtree' for speed): pip install trimesh rtree")


# ----------------------------- helpers -----------------------------

def add_custom_plank_electrode(
    tdcs,
    *,
    centre_xyz,                # [x,y,z] in subject space
    gel_stl: str,              # path to GEL surface STL
    el_stl: str,               # path to ELECTRODE surface STL
    gel_sigma: float,          # S/m
    el_sigma: float,           # S/m
    rotation_deg: float | None = None,
    name: str = "Plank",
    thickness_mm: float = 2.0,
    current_mA: float = 0.0,   # per-electrode current in mA (matches your pipeline)
):
    import os

    # --- sanity guards to catch swapped args early ---
    for pth, label in [(gel_stl, "gel_stl"), (el_stl, "el_stl")]:
        if not isinstance(pth, str) or not os.path.isfile(pth):
            raise ValueError(f"{label} must be an existing STL file. Got: {pth!r}")

    for val, label in [(gel_sigma, "gel_sigma"), (el_sigma, "el_sigma")]:
        if isinstance(val, str):
            raise TypeError(f"{label} must be a float (S/m), not a path/string. Got: {val!r}")

    el = tdcs.add_electrode()
    el.name = name
    el.centre = [float(x) for x in centre_xyz]

    el.shape = "custom"
    el.electrode_surfaces = [os.path.abspath(el_stl)]
    el.gel_surfaces       = [os.path.abspath(gel_stl)]
    el.gel = True
    el.gel_conductivity = float(gel_sigma)
    el.conductivity     = float(el_sigma)
    el.thickness        = float(thickness_mm)

    if rotation_deg is not None:
        el.rotation = float(rotation_deg)

    # per-electrode mode (mA to match your working scripts)
    el.current = float(current_mA)






def rotation_from_surface(
    m2m_dir: str,
    centre: np.ndarray,
    toward: np.ndarray,
    *,
    seg_path: str | None,
    skin_labels: list[int] | None,
    close_mm: float,
    fill_holes: bool,
    smooth_iters: int,
    decimate_frac: float,
    skin_stl_out: str | None,
) -> float:
    V, N = load_scalp_vertices_normals(
        m2m_dir=m2m_dir,
        seg_path=seg_path,
        skin_labels=skin_labels,
        close_mm=close_mm,
        fill_holes=fill_holes,
        smooth_iters=smooth_iters,
        decimate_frac=decimate_frac,
        skin_stl_out=skin_stl_out,
    )
    return float(local_rotation_degrees(centre, toward, V, N))



def resolve_label_or_xyz(m2m_dir: str, label: str | None, xyz: list[float] | None) -> np.ndarray:
    return resolve_point(m2m_dir, label, xyz)


def crop_to_brain_like(T: mesh_io.Msh) -> mesh_io.Msh:
    """Keep brain-ish volume+surface tags (like you did in TI_runner)."""
    tags_keep = np.hstack((
        np.arange(ElementTags.TH_START, ElementTags.SALINE_START - 1),
        np.arange(ElementTags.TH_SURFACE_START, ElementTags.SALINE_TH_SURFACE_START - 1),
    ))
    return T.crop_mesh(tags=tags_keep)


# ----------------------------- main -----------------------------

def build_and_run(args: argparse.Namespace) -> None:
    ensure_dir(args.out)
    
    def _project_vertices_to_mesh(mesh: "trimesh.Trimesh", V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        For each vertex in V, find closest point P on the target mesh and its triangle index.
        Returns (P, face_index).
        """
        import trimesh
        P, d, face_idx = trimesh.proximity.closest_point(mesh, V)
        return P, face_idx

    def _interpolate_normals(mesh: "trimesh.Trimesh", face_idx: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Get a robust normal at each projected point P:
        - Use per-face normals (fast & stable). If you prefer vertex interpolation, we can add it.
        """
        n = mesh.face_normals[face_idx]
        # ensure unit normals
        n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
        return n

    def conform_pad_to_scalp(src_stl: str, *,
                            scalp_mesh: "trimesh.Trimesh",
                            world_R: np.ndarray, world_t: np.ndarray,
                            inplane_deg: float,
                            offset_mm: float,
                            out_stl: str):
        """
        1) Load pad STL
        2) Re-center to centroid
        3) Apply world transform (R,t) + in-plane rotation
        4) Project each vertex to nearest point on scalp
        5) Offset along scalp normal by offset_mm
        6) Export conformed STL
        """
        import trimesh, math
        pad = trimesh.load(src_stl, process=False)
        V = np.asarray(pad.vertices); F = np.asarray(pad.faces)

        # re-center to centroid before transform
        centroid = V.mean(axis=0)
        V0 = V - centroid[None, :]

        # in-plane rotation around local Z
        a = math.radians(inplane_deg)
        Rz = np.array([[ math.cos(a), -math.sin(a), 0.0],
                    [ math.sin(a),  math.cos(a), 0.0],
                    [ 0.0,          0.0,         1.0]], float)

        R = world_R @ Rz  # 3x3
        Vw = (V0 @ R.T) + world_t[None, :]  # to world

        # project to scalp
        P, fidx = _project_vertices_to_mesh(scalp_mesh, Vw)
        Np = _interpolate_normals(scalp_mesh, fidx, P)
        Vc = P + (offset_mm * Np)

        out = trimesh.Trimesh(vertices=Vc, faces=F, process=False)
        out.export(os.path.abspath(out_stl))

    def _frame_from_xyzn(x_tan: np.ndarray, y_tan: np.ndarray, n: np.ndarray) -> np.ndarray:
        """Return 3x3 rotation with columns [x_tan, y_tan, n]."""
        return np.column_stack([x_tan, y_tan, n])    
    
    
    def _nearest_tangent_frame(point: np.ndarray, V: np.ndarray, N: np.ndarray):
        """Return orthonormal (x_tan, y_tan, n) at scalp vertex nearest to 'point'."""
        idx = int(np.argmin(((V - point[None, :])**2).sum(axis=1)))
        n = N[idx]; n = n / (np.linalg.norm(n) + 1e-12)
        # pick any global axis not parallel to n
        gx = np.array([1.,0.,0.])
        if abs(np.dot(gx, n)) > 0.9:
            gx = np.array([0.,1.,0.])
        # project to tangent plane
        def proj_tan(v): return v - np.dot(v, n)*n
        x_tan = proj_tan(gx); x_tan /= (np.linalg.norm(x_tan) + 1e-12)
        y_tan = np.cross(n, x_tan); y_tan /= (np.linalg.norm(y_tan) + 1e-12)
        return x_tan, y_tan, n

    def _place_pad_stl(src_stl: str, *, center: np.ndarray,
                    x_tan: np.ndarray, y_tan: np.ndarray, n: np.ndarray,
                    inplane_deg: float, out_stl: str):
        """
        Load an STL (any orientation), re-center it to its own centroid,
        rotate so local +Z -> n and +X/+Y -> x_tan/y_tan, then add an extra in-plane rotation,
        and translate to 'center'. Write to out_stl.
        """
        if trimesh is None:
            raise ImportError("trimesh is required for preview placement (pip install trimesh)")

        m = trimesh.load(src_stl, process=False)
        V = np.asarray(m.vertices)

        # Re-center model to its centroid so placement uses this as the "pad origin"
        centroid = V.mean(axis=0)
        V0 = V - centroid[None, :]

        # Base world-from-local rotation
        R_base = np.column_stack([x_tan, y_tan, n])  # local axes X,Y,Z -> world

        # Extra in-plane rotation (about local +Z, i.e., world 'n')
        a = math.radians(inplane_deg)
        Rz = np.array([[ math.cos(a), -math.sin(a), 0.0],
                    [ math.sin(a),  math.cos(a), 0.0],
                    [ 0.0,          0.0,         1.0]], float)

        R = R_base @ Rz
        Vp = (V0 @ R.T) + center[None, :]

        mp = trimesh.Trimesh(vertices=Vp, faces=m.faces, process=False)
        mp.export(os.path.abspath(out_stl))

    def export_preview_scene(*, out_dir: str,
                            gel_stl: str, el_stl: str,
                            r_center, r_toward, l_center, l_toward,
                            rot_r: float, rot_l: float,
                            m2m_dir: str, seg_path: str | None, skin_labels: list[int] | None,
                            close_mm: float, fill_holes: bool, smooth_iters: int,
                            decimate_frac: float, skin_stl_out: str | None):
        """
        Writes:
        - scalp STL (found or extracted)
        - eight placed pad STLs (R/L × Pos/Neg × gel/electrode)
        - merged preview PLY containing scalp + all pads
        """
        os.makedirs(out_dir, exist_ok=True)

        # Get scalp surface + normals
        V, N = load_scalp_vertices_normals(
            m2m_dir=m2m_dir, seg_path=seg_path, skin_labels=skin_labels,
            close_mm=close_mm, fill_holes=fill_holes,
            smooth_iters=smooth_iters, decimate_frac=decimate_frac,
            skin_stl_out=skin_stl_out if skin_stl_out else os.path.join(out_dir, "scalp_preview.stl")
        )
        scalp_stl = skin_stl_out if skin_stl_out else os.path.join(out_dir, "scalp_preview.stl")

        # Frames per placement (compute at EACH point)
        xr_c, yr_c, nr_c = _nearest_tangent_frame(r_center,  V, N)
        xr_t, yr_t, nr_t = _nearest_tangent_frame(r_toward, V, N)
        xl_c, yl_c, nl_c = _nearest_tangent_frame(l_center,  V, N)
        xl_t, yl_t, nl_t = _nearest_tangent_frame(l_toward, V, N)

        # Right (+) at r_center, (-) at r_toward (flip = +180)
        _place_pad_stl(gel_stl, center=r_center,  x_tan=xr_c, y_tan=yr_c, n=nr_c, inplane_deg=rot_r,           out_stl=os.path.join(out_dir, "R_Pos_gel.stl"))
        _place_pad_stl(el_stl,  center=r_center,  x_tan=xr_c, y_tan=yr_c, n=nr_c, inplane_deg=rot_r,           out_stl=os.path.join(out_dir, "R_Pos_el.stl"))
        _place_pad_stl(gel_stl, center=r_toward,  x_tan=xr_t, y_tan=yr_t, n=nr_t, inplane_deg=rot_r + 180.0,   out_stl=os.path.join(out_dir, "R_Neg_gel.stl"))
        _place_pad_stl(el_stl,  center=r_toward,  x_tan=xr_t, y_tan=yr_t, n=nr_t, inplane_deg=rot_r + 180.0,   out_stl=os.path.join(out_dir, "R_Neg_el.stl"))

        # Left (+) at l_center, (-) at l_toward
        _place_pad_stl(gel_stl, center=l_center,  x_tan=xl_c, y_tan=yl_c, n=nl_c, inplane_deg=rot_l,           out_stl=os.path.join(out_dir, "L_Pos_gel.stl"))
        _place_pad_stl(el_stl,  center=l_center,  x_tan=xl_c, y_tan=yl_c, n=nl_c, inplane_deg=rot_l,           out_stl=os.path.join(out_dir, "L_Pos_el.stl"))
        _place_pad_stl(gel_stl, center=l_toward,  x_tan=xl_t, y_tan=yl_t, n=nl_t, inplane_deg=rot_l + 180.0,   out_stl=os.path.join(out_dir, "L_Neg_gel.stl"))
        _place_pad_stl(el_stl,  center=l_toward,  x_tan=xl_t, y_tan=yl_t, n=nl_t, inplane_deg=rot_l + 180.0,   out_stl=os.path.join(out_dir, "L_Neg_el.stl"))

        # Merged preview
        if trimesh is not None:
            parts = []
            for name in ["R_Pos_gel","R_Pos_el","R_Neg_gel","R_Neg_el","L_Pos_gel","L_Pos_el","L_Neg_gel","L_Neg_el"]:
                parts.append(trimesh.load(os.path.join(out_dir, f"{name}.stl"), process=False))
            try:
                parts.append(trimesh.load(scalp_stl, process=False))
            except Exception:
                pass
            scene = trimesh.util.concatenate(parts)
            scene.export(os.path.join(out_dir, "preview_scene.ply"))
        print(f"[ok] Preview scene exported to: {out_dir}/preview_scene.ply")

    # ---------- Resolve target points ----------
    r_center = resolve_label_or_xyz(args.m2m, args.right_center, args.right_center_xyz)
    r_toward = resolve_label_or_xyz(args.m2m, args.right_toward, args.right_toward_xyz)
    l_center = resolve_label_or_xyz(args.m2m, args.left_center, args.left_center_xyz)
    l_toward = resolve_label_or_xyz(args.m2m, args.left_toward, args.left_toward_xyz)

    # ---------- Rotations from scalp surface (with segmentation fallback) ----------
    rot_r = rotation_from_surface(
        args.m2m, r_center, r_toward,
        seg_path=args.seg,
        skin_labels=args.skin_labels,
        close_mm=args.close_mm,
        fill_holes=args.fill_holes,
        smooth_iters=args.smooth_iters,
        decimate_frac=args.decimate_frac,
        skin_stl_out=args.skin_stl_out,
    )
    rot_l = rotation_from_surface(
        args.m2m, l_center, l_toward,
        seg_path=args.seg,
        skin_labels=args.skin_labels,
        close_mm=args.close_mm,
        fill_holes=args.fill_holes,
        smooth_iters=args.smooth_iters,
        decimate_frac=args.decimate_frac,
        skin_stl_out=args.skin_stl_out,
    )

    # ---------- Conform pads to scalp (optional) ----------
    conformed_dir = os.path.join(os.path.abspath(args.out), "conformed")
    os.makedirs(conformed_dir, exist_ok=True)

    # Load/ensure scalp STL and also as trimesh for proximity
    scalp_stl_path = args.skin_stl_out if args.skin_stl_out else os.path.join(args.out, "preview", "scalp_preview.stl")
    try:
        import trimesh
        scalp_tm = trimesh.load(scalp_stl_path, process=False)
    except Exception as e:
        # if the file isn't there yet, force an extraction now to disk and retry
        Vtmp, Ntmp = load_scalp_vertices_normals(
            m2m_dir=args.m2m, seg_path=args.seg, skin_labels=args.skin_labels,
            close_mm=args.close_mm, fill_holes=args.fill_holes,
            smooth_iters=args.smooth_iters, decimate_frac=args.decimate_frac,
            skin_stl_out=scalp_stl_path
        )
        import trimesh
        scalp_tm = trimesh.load(scalp_stl_path, process=False)

    # Get tangent frames per placement (reusing nearest-tangent logic)
    V_s, N_s = load_scalp_vertices_normals(
        m2m_dir=args.m2m, seg_path=args.seg, skin_labels=args.skin_labels,
        close_mm=args.close_mm, fill_holes=args.fill_holes,
        smooth_iters=args.smooth_iters, decimate_frac=args.decimate_frac,
        skin_stl_out=scalp_stl_path
    )

    xr_c, yr_c, nr_c = _nearest_tangent_frame(r_center,  V_s, N_s)
    xr_t, yr_t, nr_t = _nearest_tangent_frame(r_toward, V_s, N_s)
    xl_c, yl_c, nl_c = _nearest_tangent_frame(l_center,  V_s, N_s)
    xl_t, yl_t, nl_t = _nearest_tangent_frame(l_toward, V_s, N_s)

    # World frames (columns are the world axes for pad local X,Y,Z)
    Rr_c = _frame_from_xyzn(xr_c, yr_c, nr_c)
    Rr_t = _frame_from_xyzn(xr_t, yr_t, nr_t)
    Rl_c = _frame_from_xyzn(xl_c, yl_c, nl_c)
    Rl_t = _frame_from_xyzn(xl_t, yl_t, nl_t)

    # Conform and write out: gel at gel_offset, electrode at electrode_offset
    gel_off = float(args.gel_offset_mm)
    el_off  = float(args.electrode_offset_mm)

    paths = {}  # will store which STL to use (conformed vs original) for each electrode

    # Right +
    rp_gel = os.path.join(conformed_dir, "R_Pos_gel_conf.stl")
    rp_el  = os.path.join(conformed_dir, "R_Pos_el_conf.stl")
    conform_pad_to_scalp(args.gel_stl, scalp_mesh=scalp_tm, world_R=Rr_c, world_t=r_center,
                         inplane_deg=rot_r, offset_mm=gel_off, out_stl=rp_gel)
    conform_pad_to_scalp(args.el_stl,  scalp_mesh=scalp_tm, world_R=Rr_c, world_t=r_center,
                         inplane_deg=rot_r, offset_mm=el_off,  out_stl=rp_el)
    paths["R_Pos_gel"] = rp_gel; paths["R_Pos_el"] = rp_el

    # Right -
    rn_gel = os.path.join(conformed_dir, "R_Neg_gel_conf.stl")
    rn_el  = os.path.join(conformed_dir, "R_Neg_el_conf.stl")
    conform_pad_to_scalp(args.gel_stl, scalp_mesh=scalp_tm, world_R=Rr_t, world_t=r_toward,
                         inplane_deg=rot_r + 180.0, offset_mm=gel_off, out_stl=rn_gel)
    conform_pad_to_scalp(args.el_stl,  scalp_mesh=scalp_tm, world_R=Rr_t, world_t=r_toward,
                         inplane_deg=rot_r + 180.0, offset_mm=el_off,  out_stl=rn_el)
    paths["R_Neg_gel"] = rn_gel; paths["R_Neg_el"] = rn_el

    # Left +
    lp_gel = os.path.join(conformed_dir, "L_Pos_gel_conf.stl")
    lp_el  = os.path.join(conformed_dir, "L_Pos_el_conf.stl")
    conform_pad_to_scalp(args.gel_stl, scalp_mesh=scalp_tm, world_R=Rl_c, world_t=l_center,
                         inplane_deg=rot_l, offset_mm=gel_off, out_stl=lp_gel)
    conform_pad_to_scalp(args.el_stl,  scalp_mesh=scalp_tm, world_R=Rl_c, world_t=l_center,
                         inplane_deg=rot_l, offset_mm=el_off,  out_stl=lp_el)
    paths["L_Pos_gel"] = lp_gel; paths["L_Pos_el"] = lp_el

    # Left -
    ln_gel = os.path.join(conformed_dir, "L_Neg_gel_conf.stl")
    ln_el  = os.path.join(conformed_dir, "L_Neg_el_conf.stl")
    conform_pad_to_scalp(args.gel_stl, scalp_mesh=scalp_tm, world_R=Rl_t, world_t=l_toward,
                         inplane_deg=rot_l + 180.0, offset_mm=gel_off, out_stl=ln_gel)
    conform_pad_to_scalp(args.el_stl,  scalp_mesh=scalp_tm, world_R=Rl_t, world_t=l_toward,
                         inplane_deg=rot_l + 180.0, offset_mm=el_off,  out_stl=ln_el)
    paths["L_Neg_gel"] = ln_gel; paths["L_Neg_el"] = ln_el

    # If conform-to-scalp, swap the paths used later for SimNIBS electrode_surfaces/gel_surfaces
    use_gel_stl = args.gel_stl
    use_el_stl  = args.el_stl
    if args.conform_to_scalp:
        # We'll pass these path pairs to add_custom_plank_electrode
        RPOS = (paths["R_Pos_gel"], paths["R_Pos_el"])
        RNEG = (paths["R_Neg_gel"], paths["R_Neg_el"])
        LPOS = (paths["L_Pos_gel"], paths["L_Pos_el"])
        LNEG = (paths["L_Neg_gel"], paths["L_Neg_el"])
    else:
        RPOS = (args.gel_stl, args.el_stl)
        RNEG = (args.gel_stl, args.el_stl)
        LPOS = (args.gel_stl, args.el_stl)
        LNEG = (args.gel_stl, args.el_stl)

    
    # ---------- SimNIBS session ----------
    S = sim_struct.SESSION()
    if args.fnamehead:
        S.fnamehead = os.path.abspath(args.fnamehead)
    else:
        S.subpath = os.path.abspath(args.m2m)
    S.pathfem = os.path.abspath(args.out)
    S.map_to_vol = True
    S.element_size = float(args.element_size_mm)

    # Per-electrode current magnitude (match your place_plank_electrode convention: mA)
    I_mA = float(args.current_ma)
    thick = float(getattr(args, "electrode_thickness_mm", 2.0))

    # ---------- Montage 1: Right pair (+I at r_center, -I at r_toward) ----------
    tdcs1 = S.add_tdcslist()
    tdcs1.currents = None  # force per-electrode mode
    tdcs1.cond[2].value = float(args.el_sigma)

    add_custom_plank_electrode(
        tdcs1,
        centre_xyz=r_center.tolist(),
        gel_stl=RPOS[0], el_stl=RPOS[1],
        gel_sigma=args.gel_sigma, el_sigma=args.el_sigma,
        rotation_deg=rot_r, name=args.right_name_pos,
        thickness_mm=thick,
        current_mA=+I_mA,   # NOTE: helper writes el.current; your env expects mA
    )
    add_custom_plank_electrode(
        tdcs1,
        centre_xyz=r_toward.tolist(),
        gel_stl=RNEG[0], el_stl=RNEG[1],
        gel_sigma=args.gel_sigma, el_sigma=args.el_sigma,
        rotation_deg=(rot_r + 180.0), name=args.right_name_neg,
        thickness_mm=thick,
        current_mA=-I_mA,
    )

    # ---------- Montage 2: Left pair (+I at l_center, -I at l_toward) ----------
    tdcs2 = S.add_tdcslist()
    tdcs2.currents = None  # force per-electrode mode
    tdcs2.cond[2].value = float(args.el_sigma)

    add_custom_plank_electrode(
        tdcs2,
        centre_xyz=l_center.tolist(),
        gel_stl=LPOS[0], el_stl=LPOS[1],
        gel_sigma=args.gel_sigma, el_sigma=args.el_sigma,
        rotation_deg=rot_l, name=args.left_name_pos,
        thickness_mm=thick,
        current_mA=+I_mA,
    )
    add_custom_plank_electrode(
        tdcs2,
        centre_xyz=l_toward.tolist(),
        gel_stl=LNEG[0], el_stl=LNEG[1],
        gel_sigma=args.gel_sigma, el_sigma=args.el_sigma,
        rotation_deg=(rot_l + 180.0), name=args.left_name_neg,
        thickness_mm=thick,
        current_mA=-I_mA,
    )

    # ---------- Debug ----------
    for i, td in enumerate(S.poslists, 1):   # <- poslist (singular), not poslists
        print(f"[dbg] Montage {i}: tdcs.currents={getattr(td, 'currents', None)}")
        for e in td.electrode:
            print(f"[dbg]  - {e.name}: centre={e.centre}, rot={getattr(e,'rotation',None)}, "
                  f"I={getattr(e,'current',None)} (mA), shape={e.shape}, thick={getattr(e,'thickness',None)}")

    # ---------- Preview export (BEFORE running FEM) ----------
    preview_dir = os.path.join(S.pathfem, "preview")
    try:
        export_preview_scene(
            out_dir=preview_dir,
            gel_stl=args.gel_stl, el_stl=args.el_stl,
            r_center=r_center, r_toward=r_toward,
            l_center=l_center, l_toward=l_toward,
            rot_r=rot_r, rot_l=rot_l,
            m2m_dir=args.m2m, seg_path=args.seg, skin_labels=args.skin_labels,
            close_mm=args.close_mm, fill_holes=args.fill_holes,
            smooth_iters=args.smooth_iters, decimate_frac=args.decimate_frac,
            skin_stl_out=args.skin_stl_out,
        )
        print(f"[ok] Preview written in: {preview_dir}")
    except Exception as e:
        print(f"[warn] Could not export preview scene: {e}")

    
    
    # ---------- Run ----------
    print("[run] SimNIBS…")
    sim.run_simnibs(S)
    print("[ok] Done FEM. Computing TI…")

    # ---------- TI post-proc ----------
    run_base = os.path.join(S.pathfem)

    def find_first_suffix(suffix: str) -> str:
        for f in os.listdir(run_base):
            if f.endswith(suffix):
                return os.path.join(run_base, f)
        raise FileNotFoundError(f"Could not find *{suffix} under {run_base}")

    m1 = mesh_io.read_msh(find_first_suffix("_TDCS_1_scalar.msh"))
    m2 = mesh_io.read_msh(find_first_suffix("_TDCS_2_scalar.msh"))

    m1 = crop_to_brain_like(m1)
    m2 = crop_to_brain_like(m2)

    E1 = m1.field["E"].value
    E2 = m2.field["E"].value

    TImax = TI.get_maxTI(E1, E2)
    mout = deepcopy(m1)
    mout.elmdata = []
    mout.add_element_field(TImax, "TImax")

    ti_msh = os.path.join(run_base, "TI.msh")
    mesh_io.write_msh(mout, ti_msh)
    print(f"[ok] TI mesh: {ti_msh}")

    # ---------- Export NIfTI + overlays ----------
    t1_candidates = [
        os.path.join(os.path.dirname(getattr(S, "fnamehead", "")), "T1.nii.gz"),
        os.path.join(args.t1_fallback) if args.t1_fallback else "",
    ]
    t1_ref = next((p for p in t1_candidates if p and os.path.isfile(p)), None)

    vol_dir = os.path.join(run_base, "TI_volume")
    ensure_dir(vol_dir)

    if t1_ref is None:
        print("[warn] No T1 reference found; msh2nii will still write a TI volume in mesh space.")
        t1_ref = ti_msh

    try:
        import subprocess
        subprocess.run(["msh2nii", ti_msh, t1_ref, vol_dir], check=True)
    except Exception as e:
        print(f"[warn] msh2nii failed: {e}")

    ti_nii = None
    for f in os.listdir(vol_dir):
        if f.lower().endswith(".nii") or f.lower().endswith(".nii.gz"):
            ti_nii = os.path.join(vol_dir, f)
            break

    if ti_nii is None:
        print("[warn] Could not locate TI NIfTI; skipping overlay PNGs.")
        return

    ti_img = nib.load(ti_nii)

    bg_img = None
    if os.path.isfile(t1_ref) and t1_ref.endswith(".nii.gz"):
        try:
            bg_img = nib.load(t1_ref)
        except Exception:
            bg_img = None

    png_dir = os.path.join(run_base, "overlays")
    ensure_dir(png_dir)

    make_overlay_png(
        out_png=os.path.join(png_dir, "TI_overlay.png"),
        overlay_img=ti_img,
        bg_img=bg_img,
        title="TImax (custom plank electrodes)",
        roi_mask_img=None,
        abs_colour=True,
    )
    print(f"[ok] Overlay written to {png_dir}/TI_overlay.png")



def parse_xyz_triplet(prefix: str, ns: argparse.Namespace) -> list[float] | None:
    x = getattr(ns, f"{prefix}_x")
    y = getattr(ns, f"{prefix}_y")
    z = getattr(ns, f"{prefix}_z")
    if x is None and y is None and z is None:
        return None
    if None in (x, y, z):
        raise ValueError(f"Provide all three --{prefix}-x/--{prefix}-y/--{prefix}-z or none")
    return [float(x), float(y), float(z)]


def cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run TI with custom plank electrodes (4 total) in SimNIBS 4.5")

    ap.add_argument("--m2m", required=True, help="Path to subject m2m_* directory (for EEG labels, scalp surface, etc.)")
    ap.add_argument("--fnamehead", help="Optional explicit head mesh (.msh). If omitted, SimNIBS uses --m2m subpath.")
    ap.add_argument("--out", required=True, help="Output FEM directory")

    ap.add_argument("--gel-stl", required=True, help="Gel STL (custom plank)")
    ap.add_argument("--el-stl", required=True, help="Electrode STL (custom plank)")
    ap.add_argument("--gel-sigma", type=float, default=1.4, help="Gel conductivity (S/m)")
    ap.add_argument("--el-sigma", type=float, default=0.1, help="Electrode conductivity (S/m)")

    ap.add_argument("--current-ma", type=float, default=2.0, help="Current magnitude for each pair (mA)")
    ap.add_argument("--element-size-mm", type=float, default=0.1)

    # Right pair (defaults AF4 -> PO4)
    ap.add_argument("--right-center", default="AF4", help="EEG label for right pair center")
    ap.add_argument("--right-toward", default="PO4", help="EEG label where the opposing plank is placed")
    ap.add_argument("--right-center-x", type=float)
    ap.add_argument("--right-center-y", type=float)
    ap.add_argument("--right-center-z", type=float)
    ap.add_argument("--right-toward-x", type=float)
    ap.add_argument("--right-toward-y", type=float)
    ap.add_argument("--right-toward-z", type=float)
    ap.add_argument("--right-name-pos", default="R_Pos")
    ap.add_argument("--right-name-neg", default="R_Neg")

    # Left pair (defaults AF3 -> PO3)
    ap.add_argument("--left-center", default="AF3", help="EEG label for left pair center")
    ap.add_argument("--left-toward", default="PO3", help="EEG label where the opposing plank is placed")
    ap.add_argument("--left-center-x", type=float)
    ap.add_argument("--left-center-y", type=float)
    ap.add_argument("--left-center-z", type=float)
    ap.add_argument("--left-toward-x", type=float)
    ap.add_argument("--left-toward-y", type=float)
    ap.add_argument("--left-toward-z", type=float)
    ap.add_argument("--left-name-pos", default="L_Pos")
    ap.add_argument("--left-name-neg", default="L_Neg")
    
    ap.add_argument("--seg", help="Segmentation NIfTI (label map) to extract scalp if no STL exists")
    ap.add_argument("--skin-labels", type=lambda s: [int(x) for x in s.split(",")], default=[5],
                    help="Comma-separated label IDs considered 'skin' (default: 5)")
    ap.add_argument("--close-mm", type=float, default=2.0, help="Morphological closing radius (mm)")
    ap.add_argument("--fill-holes", action="store_true", help="Fill interior holes in the mask")
    ap.add_argument("--smooth-iters", type=int, default=10, help="Surface smoothing iterations")
    ap.add_argument("--decimate-frac", type=float, default=0.2,
                    help="Target decimation fraction (0.2 → ~80% triangles removed)")
    ap.add_argument("--skin-stl-out", help="Optional path to write the extracted scalp STL")
    ap.add_argument("--conform-to-scalp", action="store_true",
                help="Shrink-wrap gel/electrode STLs to scalp surface before running.")
    ap.add_argument("--gel-offset-mm", type=float, default=0.0,
                    help="Offset for GEL surface after projection (0.0 = contact).")
    ap.add_argument("--electrode-offset-mm", type=float, default=2.0,
                    help="Offset for ELECTRODE surface after projection (typically gel thickness).")



    # Optional T1 fallback for msh2nii
    ap.add_argument("--t1-fallback", help="Path to T1.nii.gz if not colocated with head mesh")

    args = ap.parse_args()

    # Pack XYZ triplets if provided (labels take precedence unless you clear them)
    args.right_center_xyz = parse_xyz_triplet("right_center", args)
    args.right_toward_xyz = parse_xyz_triplet("right_toward", args)
    args.left_center_xyz  = parse_xyz_triplet("left_center", args)
    args.left_toward_xyz  = parse_xyz_triplet("left_toward", args)

    return args


if __name__ == "__main__":
    build_and_run(cli())
'''
simnibs_python ti_custom_planks.py \
  --m2m /home/boyan/sandbox/Jake_Data/camcan_test_run/sub-CC110056/anat/m2m_sub-CC110056 \
  --fnamehead /home/boyan/sandbox/Jake_Data/camcan_test_run/sub-CC110056/anat/m2m_sub-CC110056/sub-CC110056.msh \
  --out ./TI_CustomPlanks \
  --gel-stl /home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/electrode_workshop/gel_plank.stl \
  --el-stl /home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/electrode_workshop/electrode_plank.stl \
  --gel-sigma 1.4 \
  --el-sigma 0.1 \
  --current-ma 2.0 \
  --element-size-mm 0.5 \
  --right-center AF4 \
  --right-toward PO4 \
  --left-center AF3 \
  --left-toward PO3 \
  --t1-fallback /home/boyan/sandbox/Jake_Data/camcan_test_run/sub-CC110056/anat/sub-CC110056_T1w.nii.gz
  '''
