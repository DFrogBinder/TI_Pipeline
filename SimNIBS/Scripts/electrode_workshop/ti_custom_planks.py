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
from copy import deepcopy
import numpy as np
import nibabel as nib

from place_plank_electrode import (
    load_scalp_vertices_normals,
    resolve_point,
    local_rotation_degrees,
    ensure_dir,
)


import simnibs as sim
from simnibs import sim_struct, mesh_io, ElementTags
from simnibs.utils import TI_utils as TI

# Reuse your helpers (projection, normals, EEG label lookup) from the plank placer
from place_plank_electrode import (
    load_scalp_vertices_normals,
    resolve_point,
    local_rotation_degrees,
    ensure_dir,
)

# Overlay helpers for PNGs
from functions import make_overlay_png  # nice colourbar behavior


# ----------------------------- helpers -----------------------------

def add_custom_plank_electrode(
    tdcs,
    *,
    centre_xyz,
    gel_stl,
    el_stl,
    gel_sigma,
    el_sigma,
    rotation_deg=None,
    name="Plank",
    thickness_mm=2.0,
    current_A=0.0,   # <- keep name, we’ll pass mA just like your working script
):
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

    # Per-electrode mode: set current on the electrode and DO NOT set channelnr
    el.current = float(current_A)     # << matches place_plank_electrode.py (mA)





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
        gel_stl=args.gel_stl, el_stl=args.el_stl,
        gel_sigma=args.gel_sigma, el_sigma=args.el_sigma,
        rotation_deg=rot_r, name=args.right_name_pos,
        thickness_mm=thick,
        current_A=+I_mA,   # NOTE: helper writes el.current; your env expects mA
    )
    add_custom_plank_electrode(
        tdcs1,
        centre_xyz=r_toward.tolist(),
        gel_stl=args.gel_stl, el_stl=args.el_stl,
        gel_sigma=args.gel_sigma, el_sigma=args.el_sigma,
        rotation_deg=(rot_r + 180.0), name=args.right_name_neg,
        thickness_mm=thick,
        current_A=-I_mA,
    )

    # ---------- Montage 2: Left pair (+I at l_center, -I at l_toward) ----------
    tdcs2 = S.add_tdcslist()
    tdcs2.currents = None  # force per-electrode mode
    tdcs2.cond[2].value = float(args.el_sigma)

    add_custom_plank_electrode(
        tdcs2,
        centre_xyz=l_center.tolist(),
        gel_stl=args.gel_stl, el_stl=args.el_stl,
        gel_sigma=args.gel_sigma, el_sigma=args.el_sigma,
        rotation_deg=rot_l, name=args.left_name_pos,
        thickness_mm=thick,
        current_A=+I_mA,
    )
    add_custom_plank_electrode(
        tdcs2,
        centre_xyz=l_toward.tolist(),
        gel_stl=args.gel_stl, el_stl=args.el_stl,
        gel_sigma=args.gel_sigma, el_sigma=args.el_sigma,
        rotation_deg=(rot_l + 180.0), name=args.left_name_neg,
        thickness_mm=thick,
        current_A=-I_mA,
    )

    # ---------- Debug ----------
    for i, td in enumerate(S.poslists, 1):   # <- poslist (singular), not poslists
        print(f"[dbg] Montage {i}: tdcs.currents={getattr(td, 'currents', None)}")
        for e in td.electrode:
            print(f"[dbg]  - {e.name}: centre={e.centre}, rot={getattr(e,'rotation',None)}, "
                  f"I={getattr(e,'current',None)} (mA), shape={e.shape}, thick={getattr(e,'thickness',None)}")

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