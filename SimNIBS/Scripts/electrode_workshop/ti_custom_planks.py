#!/usr/bin/env python3
"""
Minimal TI pipeline using SimNIBS-native rectangular electrodes.

Creates two tdcs montages (right pair and left pair) and computes TImax.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import List, Optional, Dict, Any

import numpy as np
import simnibs as sim
from simnibs import mesh_io, sim_struct
from simnibs.utils import TI_utils as TI


def resolve_point(m2m_dir: str, label: Optional[str], xyz: Optional[List[float]]) -> np.ndarray:
    if xyz is not None:
        return np.asarray(xyz, dtype=float)
    if not label:
        raise RuntimeError("Provide either an EEG label or explicit XYZ coordinates.")

    eeg_by_label_funcs = []
    try:
        from simnibs.utils.eeg_positions import eeg_pos_by_label  # type: ignore
        eeg_by_label_funcs.append(eeg_pos_by_label)
    except Exception:
        pass
    try:
        from simnibs.utils.electrode_positions import eeg_pos_by_label  # type: ignore
        eeg_by_label_funcs.append(eeg_pos_by_label)
    except Exception:
        pass
    try:
        from simnibs.utils.electrodes import eeg_pos_by_label  # type: ignore
        eeg_by_label_funcs.append(eeg_pos_by_label)
    except Exception:
        pass

    for fn in eeg_by_label_funcs:
        try:
            p = fn(label, m2m_dir)
            return np.asarray(p, dtype=float)
        except Exception:
            continue

    raise RuntimeError(
        f"Could not resolve EEG label '{label}'. "
        "Provide explicit --*-x/--*-y/--*-z coordinates instead."
    )


def add_rect_electrode(
    tdcs,
    *,
    centre_xyz: np.ndarray,
    name: str,
    current_mA: float,
    dimensions_mm: List[float],
    thickness_mm: float,
    gel_thickness_mm: float,
    gel_sigma: float,
    el_sigma: float,
    rotation_deg: Optional[float],
):
    el = tdcs.add_electrode()
    el.name = name
    el.centre = [float(x) for x in centre_xyz]
    el.shape = "rect"
    el.dimensions = [float(dimensions_mm[0]), float(dimensions_mm[1])]
    el.thickness = float(thickness_mm)
    el.conductivity = float(el_sigma)
    el.gel = True
    el.gel_thickness = float(gel_thickness_mm)
    el.gel_conductivity = float(gel_sigma)
    if rotation_deg is not None:
        el.rotation = float(rotation_deg)
    el.current = float(current_mA)


def _config(config: Dict[str, Any], key: str, default: Any) -> Any:
    return config[key] if key in config else default


def run_ti(config: Dict[str, Any]) -> str:
    """
    Run a minimal 4-rectangle TI setup and return the TI.msh path.

    Required keys: m2m, out
    Optional keys: fnamehead, right/left labels or xyz, and parameters below.
    """
    m2m = config["m2m"]
    out = config["out"]
    os.makedirs(out, exist_ok=True)

    r_center = resolve_point(m2m, _config(config, "right_center", "AF4"), config.get("right_center_xyz"))
    r_toward = resolve_point(m2m, _config(config, "right_toward", "PO4"), config.get("right_toward_xyz"))
    l_center = resolve_point(m2m, _config(config, "left_center", "AF3"), config.get("left_center_xyz"))
    l_toward = resolve_point(m2m, _config(config, "left_toward", "PO3"), config.get("left_toward_xyz"))

    S = sim_struct.SESSION()
    fnamehead = config.get("fnamehead")
    if fnamehead:
        S.fnamehead = os.path.abspath(fnamehead)
    else:
        S.subpath = os.path.abspath(m2m)
    S.pathfem = os.path.abspath(out)
    S.map_to_vol = True
    S.element_size = float(_config(config, "element_size_mm", 0.1))

    dims = [float(_config(config, "length_mm", 35.0)), float(_config(config, "width_mm", 25.0))]
    thick = float(_config(config, "electrode_thickness_mm", 1.5))
    gel_thick = float(_config(config, "gel_thickness_mm", 3.0))
    I_mA = float(_config(config, "current_ma", 2.0))
    gel_sigma = float(_config(config, "gel_sigma", 1.4))
    el_sigma = float(_config(config, "el_sigma", 0.1))
    rotation_deg = config.get("rotation_deg")

    tdcs1 = S.add_tdcslist()
    tdcs1.currents = None
    tdcs1.cond[2].value = el_sigma

    add_rect_electrode(
        tdcs1,
        centre_xyz=r_center,
        name=_config(config, "right_name_pos", "R_Pos"),
        current_mA=+I_mA,
        dimensions_mm=dims,
        thickness_mm=thick,
        gel_thickness_mm=gel_thick,
        gel_sigma=gel_sigma,
        el_sigma=el_sigma,
        rotation_deg=rotation_deg,
    )
    add_rect_electrode(
        tdcs1,
        centre_xyz=r_toward,
        name=_config(config, "right_name_neg", "R_Neg"),
        current_mA=-I_mA,
        dimensions_mm=dims,
        thickness_mm=thick,
        gel_thickness_mm=gel_thick,
        gel_sigma=gel_sigma,
        el_sigma=el_sigma,
        rotation_deg=rotation_deg,
    )

    tdcs2 = S.add_tdcslist()
    tdcs2.currents = None
    tdcs2.cond[2].value = el_sigma

    add_rect_electrode(
        tdcs2,
        centre_xyz=l_center,
        name=_config(config, "left_name_pos", "L_Pos"),
        current_mA=+I_mA,
        dimensions_mm=dims,
        thickness_mm=thick,
        gel_thickness_mm=gel_thick,
        gel_sigma=gel_sigma,
        el_sigma=el_sigma,
        rotation_deg=rotation_deg,
    )
    add_rect_electrode(
        tdcs2,
        centre_xyz=l_toward,
        name=_config(config, "left_name_neg", "L_Neg"),
        current_mA=-I_mA,
        dimensions_mm=dims,
        thickness_mm=thick,
        gel_thickness_mm=gel_thick,
        gel_sigma=gel_sigma,
        el_sigma=el_sigma,
        rotation_deg=rotation_deg,
    )

    print("[run] SimNIBS…")
    sim.run_simnibs(S)
    print("[ok] Done FEM. Computing TI…")

    run_base = S.pathfem

    def find_first_suffix(suffix: str) -> str:
        for f in os.listdir(run_base):
            if f.endswith(suffix):
                return os.path.join(run_base, f)
        raise FileNotFoundError(f"Could not find *{suffix} under {run_base}")

    m1 = mesh_io.read_msh(find_first_suffix("_TDCS_1_scalar.msh"))
    m2 = mesh_io.read_msh(find_first_suffix("_TDCS_2_scalar.msh"))

    E1 = m1.field["E"].value
    E2 = m2.field["E"].value
    TImax = TI.get_maxTI(E1, E2)

    mout = deepcopy(m1)
    mout.elmdata = []
    mout.add_element_field(TImax, "TImax")
    ti_msh = os.path.join(run_base, "TI.msh")
    mesh_io.write_msh(mout, ti_msh)
    print(f"[ok] TI mesh: {ti_msh}")
    return ti_msh
