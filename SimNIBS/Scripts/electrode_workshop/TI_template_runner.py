#!/home/boyan/SimNIBS-4.5/bin/simnibs_python
# -*- coding: utf-8 -*-
"""
Run TI simulations on the MNI152 head model only (no meshing).
Adjust paths and sweep parameters below.
"""

import os
import json
import shutil
import logging
import numpy as np
import simnibs as sim
from copy import deepcopy
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TaskProgressColumn
from simnibs import sim_struct, mesh_io, ElementTags
from simnibs.utils import TI_utils as TI


# ----- MNI152 paths -----
SADNBOX_DIR = "/home/boyan/sandbox"
MNI152_M2M = os.path.join(SADNBOX_DIR, "simnibs4_exmaples", "m2m_MNI152")
FNAMEHEAD = os.path.join(MNI152_M2M, "MNI152.msh")
OUTPUT_ROOT = os.path.join(SADNBOX_DIR, "Jake_Data", "camcan_test_run", "main_data", "MNI152", "anat", "SimNIBS")


# ----- Sweep parameters -----
electrode_shape = "rect"
electrode_thickness = 5.0  # mm
electrode_conductivity = 0.85
element_size = 0.1

# Sweep only electrode length; keep width fixed.
length_values = np.arange(10.0, 20.0 + 1e-6, 5.0)
width_fixed = 25.0

pair1_current_mA = 2.0
pair2_current_mA = 2.0

# Hippocampus montage (two electrode pairs)
montage_right = ("T7", "C5")
montage_left = ("T8", "C6")


def run_sweep():
    subject = "MNI152"
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    log_root = os.path.join(OUTPUT_ROOT, "logs")
    os.makedirs(log_root, exist_ok=True)

    total_runs = len(length_values)
    console = Console()
    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        refresh_per_second=2,
    )

    simnibs_logger = logging.getLogger("simnibs")
    simnibs_logger.handlers = []
    simnibs_logger.setLevel(logging.INFO)
    simnibs_logger.propagate = False
    simnibs_logger.addHandler(
        RichHandler(
            console=console,
            show_time=True,
            show_level=True,
            show_path=False,
            markup=False,
        )
    )

    with progress:
        task_id = progress.add_task("SimNIBS runs", total=total_runs)
        for length in length_values:
            electrode_dims = [float(length), float(width_fixed)]
            length_tag = f"{length:.1f}".replace(".", "p")
            width_tag = f"{width_fixed:.1f}".replace(".", "p")

            scenario_dir = os.path.join(
                OUTPUT_ROOT,
                "Output",
                subject,
                "ratio_1to1_2mA",
                f"len{length_tag}_w{width_tag}",
            )
            if os.path.isdir(scenario_dir):
                shutil.rmtree(scenario_dir)
            os.makedirs(scenario_dir, exist_ok=True)

            S = sim_struct.SESSION()
            S.open_in_gmsh = False
            S.fnamehead = FNAMEHEAD
            S.pathfem = scenario_dir
            S.element_size = element_size
            S.map_to_vol = True

            tdcs1 = S.add_tdcslist()
            tdcs1.cond[2].value = electrode_conductivity
            pair1_current_A = pair1_current_mA * 1e-3
            tdcs1.currents = [pair1_current_A, -pair1_current_A]

            el1 = tdcs1.add_electrode()
            el1.channelnr = 1
            el1.centre = montage_right[0]
            el1.shape = electrode_shape
            el1.dimensions = electrode_dims
            el1.thickness = electrode_thickness

            el2 = tdcs1.add_electrode()
            el2.channelnr = 2
            el2.centre = montage_right[1]
            el2.shape = electrode_shape
            el2.dimensions = electrode_dims
            el2.thickness = electrode_thickness

            tdcs2 = S.add_tdcslist(deepcopy(tdcs1))
            pair2_current_A = pair2_current_mA * 1e-3
            tdcs2.currents = [pair2_current_A, -pair2_current_A]
            tdcs2.electrode[0].centre = montage_left[0]
            tdcs2.electrode[1].centre = montage_left[1]
            tdcs2.electrode[0].mesh_element_size = element_size
            tdcs2.electrode[1].mesh_element_size = element_size

            metadata = {
                "scenario": "ratio_1to1_2mA",
                "current_ratio": "1:1",
                "pair1_current_mA": pair1_current_mA,
                "pair2_current_mA": pair2_current_mA,
                "electrode_dimensions_mm": {
                    "length": electrode_dims[0],
                    "width": electrode_dims[1],
                    "thickness": electrode_thickness,
                },
                "montage": {
                    "pair1": {"anode": montage_right[0], "cathode": montage_right[1]},
                    "pair2": {"anode": montage_left[0], "cathode": montage_left[1]},
                },
            }
            with open(os.path.join(scenario_dir, "sweep_metadata.json"), "w") as fh:
                json.dump(metadata, fh, indent=2)

            progress.update(
                task_id,
                description=f"len={length:.1f} w={width_fixed:.1f}",
            )
            log_path = os.path.join(log_root, f"simnibs_len{length_tag}_w{width_tag}_ratio_1to1_2mA.log")
            for handler in list(simnibs_logger.handlers):
                if isinstance(handler, logging.FileHandler):
                    simnibs_logger.removeHandler(handler)
                    handler.close()
            file_handler = logging.FileHandler(log_path)
            formatter = logging.Formatter("[ %(name)s ] %(levelname)s: %(message)s")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            simnibs_logger.addHandler(file_handler)

            sim.run_simnibs(S)

            m1 = mesh_io.read_msh(os.path.join(S.pathfem, f"{subject}_TDCS_1_scalar.msh"))
            m2 = mesh_io.read_msh(os.path.join(S.pathfem, f"{subject}_TDCS_2_scalar.msh"))

            tags_keep = np.hstack(
                (
                    np.arange(ElementTags.TH_START, ElementTags.SALINE_START - 1),
                    np.arange(ElementTags.TH_SURFACE_START, ElementTags.SALINE_TH_SURFACE_START - 1),
                )
            )

            m1 = m1.crop_mesh(tags=tags_keep)
            m2 = m2.crop_mesh(tags=tags_keep)

            E1_vec = m1.field["E"]
            E2_vec = m2.field["E"]

            TImax = TI.get_maxTI(E1_vec.value, E2_vec.value)

            mout = deepcopy(m1)
            mout.elmdata = []
            mout.add_element_field(TImax, "TImax")

            out_path = os.path.join(S.pathfem, "TI.msh")
            mesh_io.write_msh(mout, out_path)

            progress.advance(task_id, 1)


if __name__ == "__main__":
    run_sweep()
