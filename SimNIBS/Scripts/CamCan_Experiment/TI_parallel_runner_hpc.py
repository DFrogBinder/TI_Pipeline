#!/home/boyan/SimNIBS-4.5/bin/simnibs_python
import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from typing import Dict, Iterable, List, Tuple

import nibabel as nib
import numpy as np
import simnibs as sim
from simnibs import mesh_io, sim_struct
from simnibs.utils import TI_utils as TI


def _log(msg: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def prepare_subject_environment(root_dir: str,
                                run_mni152: bool = True,
                                mesh_present: bool = True) -> Dict[str, str]:
    """
    Prepares subject-specific paths (only once) and optionally re-runs meshing.
    Returns a dict with paths used by individual simulations.
    """
    subject = os.listdir(root_dir)[1]

    if run_mni152:
        subject = 'MNI152'
        sandbox_dir = root_dir.split('base_data')[0]
        fnamehead = os.path.join(sandbox_dir, 'simnibs4_exmaples', 'm2m_MNI152', 'MNI152.msh')
        output_root = os.path.join(root_dir, subject, 'anat', 'SimNIBS')
        subject_dir = os.path.join(root_dir, subject, 'anat')
    else:
        fnamehead = os.path.join(root_dir, subject, 'anat', f'm2m_{subject}', f'{subject}.msh')
        output_root = os.path.join(root_dir, subject, 'anat', 'SimNIBS')
        subject_dir = os.path.join(root_dir, subject, 'anat')

    if mesh_present:
        _log("Mesh present, skipping meshing step.")
    else:
        _run_meshing_pipeline(subject, subject_dir)

    os.makedirs(output_root, exist_ok=True)

    return {
        "subject": subject,
        "fnamehead": fnamehead,
        "output_root": output_root,
        "subject_dir": subject_dir,
        "run_mni152": run_mni152,
    }


def _run_meshing_pipeline(subject: str, subject_dir: str) -> None:
    """Runs the CHARM meshing pipeline when needed."""
    cmd = [
        "charm",
        subject,
        os.path.join(subject_dir, f"{subject}_T1w.nii.gz"),
    ]
    _log(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=str(subject_dir), check=True)
    except Exception as exc:
        _log(f"[ERROR] Meshing failed: {exc}")
        raise


def create_parameter_space(
    width_range: Iterable[float],
    height_range: Iterable[float],
    current_scenarios: Iterable[Dict[str, float]]
) -> List[Tuple[float, float, Dict[str, float]]]:
    """
    Generates all combinations (width, height, scenario) to be simulated.
    """
    combos: List[Tuple[float, float, Dict[str, float]]] = []
    for width, height, scenario in itertools.product(width_range, height_range, current_scenarios):
        combos.append((float(width), float(height), dict(scenario)))
    return combos


def _format_dimension_tag(value: float) -> str:
    return f"{value:.1f}".replace('.', 'p')


def _ensure_clean_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _save_metadata(path: str, metadata: Dict) -> None:
    with open(path, "w") as fh:
        json.dump(metadata, fh, indent=2)


def _export_volumes(pathfem: str, fnamehead: str, subject_dir: str, run_mni152: bool, subject: str) -> None:
    volume_masks_path = os.path.join(pathfem, 'Volume_Maks')
    volume_base_path = os.path.join(pathfem, 'Volume_Base')
    volume_labels_path = os.path.join(pathfem, 'Volume_Labels')
    os.makedirs(volume_masks_path, exist_ok=True)
    os.makedirs(volume_base_path, exist_ok=True)
    os.makedirs(volume_labels_path, exist_ok=True)

    labels_path = os.path.join(volume_labels_path, "TI_Volumetric_Labels")
    masks_path = os.path.join(volume_masks_path, "TI_Volumetric_Masks")
    ti_volume_path = os.path.join(volume_base_path, "TI_Volumetric_Base")

    if run_mni152:
        t1_path = os.path.join(os.path.dirname(fnamehead), 'T1.nii.gz')
    else:
        t1_path = os.path.join(subject_dir, f"{subject}_T1w.nii.gz")

    def _call_msh2nii(args: List[str]) -> None:
        try:
            subprocess.run(args, check=True)
        except Exception as exc:
            _log(f"Error running {' '.join(args)}: {exc}")

    _call_msh2nii(["msh2nii", os.path.join(pathfem, 'TI.msh'), t1_path, labels_path, "--create_label"])
    _call_msh2nii(["msh2nii", os.path.join(pathfem, 'TI.msh'), t1_path, masks_path, "--create_masks"])
    _call_msh2nii(["msh2nii", os.path.join(pathfem, 'TI.msh'), t1_path, ti_volume_path])


def _postprocess_volume_outputs(volume_labels_path: str, volume_base_path: str, output_path: str) -> None:
    label_files = sorted(f for f in os.listdir(volume_labels_path) if f.endswith(('.nii', '.nii.gz')))
    if not label_files:
        raise FileNotFoundError(f"No label NIfTI found in {volume_labels_path}")
    label_file_path = label_files[0]

    ti_files = sorted(f for f in os.listdir(volume_base_path) if f.endswith(('.nii', '.nii.gz')))
    if not ti_files:
        raise FileNotFoundError(f"No TI NIfTI found in {volume_base_path}")
    ti_volume_file = ti_files[0]

    label_img = nib.load(os.path.join(volume_labels_path, label_file_path))
    ti_img = nib.load(os.path.join(volume_base_path, ti_volume_file))

    labels = np.asarray(label_img.dataobj).astype(np.int32, copy=False)
    GM_LABELS = {2}
    WM_LABELS = {1}
    brain_mask = np.isin(labels, list(GM_LABELS | WM_LABELS))

    ti_data = ti_img.get_fdata(dtype=np.float32)
    masked = np.where(brain_mask, ti_data, np.nan).astype(np.float32)
    masked_img = nib.Nifti1Image(masked, label_img.affine, label_img.header)
    masked_img.header.set_data_dtype(np.float32)
    nib.save(masked_img, os.path.join(output_path, "ti_brain_only.nii.gz"))


def run_single_combination(width: float,
                           height: float,
                           scenario: Dict[str, float],
                           config: Dict[str, str]) -> str:
    """
    Executes one parameter combination.
    Returns the path of the simulation output directory on success.
    """
    subject = config['subject']
    fnamehead = config['fnamehead']
    output_root = config['output_root']
    subject_dir = config['subject_dir']
    run_mni152 = config['run_mni152']

    electrode_shape = 'rect'
    electrode_thickness = 5.0  # mm
    electrode_conductivity = 0.85

    montage_right = ('T7', 'C5')
    montage_left = ('T8', 'C6')

    width_tag = _format_dimension_tag(width)
    height_tag = _format_dimension_tag(height)

    scenario_dir = os.path.join(
        output_root,
        'Output',
        subject,
        scenario['name'],
        f"w{width_tag}_h{height_tag}"
    )
    _ensure_clean_dir(scenario_dir)

    S = sim_struct.SESSION()
    S.open_in_gmsh = False
    S.fnamehead = fnamehead
    S.pathfem = scenario_dir
    S.element_size = 0.1
    S.map_to_vol = True

    tdcs1 = S.add_tdcslist()
    tdcs1.cond[2].value = electrode_conductivity
    pair1_current_A = scenario['pair1_mA'] * 1e-3
    tdcs1.currents = [pair1_current_A, -pair1_current_A]

    electrode_dims = [width, height]

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
    pair2_current_A = scenario['pair2_mA'] * 1e-3
    tdcs2.currents = [pair2_current_A, -pair2_current_A]
    tdcs2.electrode[0].centre = montage_left[0]
    tdcs2.electrode[1].centre = montage_left[1]
    tdcs2.electrode[0].mesh_element_size = 0.1
    tdcs2.electrode[1].mesh_element_size = 0.1

    metadata = {
        "scenario": scenario['name'],
        "current_ratio": scenario['ratio'],
        "pair1_current_mA": scenario['pair1_mA'],
        "pair2_current_mA": scenario['pair2_mA'],
        "electrode_dimensions_mm": {
            "width": width,
            "height": height,
            "thickness": electrode_thickness
        },
        "montage": {
            "pair1": {"anode": montage_right[0], "cathode": montage_right[1]},
            "pair2": {"anode": montage_left[0], "cathode": montage_left[1]}
        }
    }
    _save_metadata(os.path.join(scenario_dir, "sweep_metadata.json"), metadata)

    _log(
        f"[RUN] Scenario {scenario['name']} | dims {width:.1f}×{height:.1f} mm | "
        f"currents {scenario['pair1_mA']:.1f} mA vs {scenario['pair2_mA']:.1f} mA"
    )
    sim.run_simnibs(S)

    m1 = mesh_io.read_msh(os.path.join(S.pathfem, f'{subject}_TDCS_1_scalar.msh'))
    m2 = mesh_io.read_msh(os.path.join(S.pathfem, f'{subject}_TDCS_2_scalar.msh'))

    tags_keep = np.hstack((
        np.arange(0, 499),     # 0–498 inclusive
        np.arange(1000, 1499)  # 1000–1498 inclusive
        ))

    m1 = m1.crop_mesh(tags=tags_keep)
    m2 = m2.crop_mesh(tags=tags_keep)

    E1_vec = m1.field['E']
    E2_vec = m2.field['E']
    TImax = TI.get_maxTI(E1_vec.value, E2_vec.value)

    mout = deepcopy(m1)
    mout.elmdata = []
    mout.add_element_field(TImax, 'TImax')

    mesh_io.write_msh(mout, os.path.join(S.pathfem, 'TI.msh'))
    _log(f"Saved gray+white TI mesh to: {os.path.join(S.pathfem, 'TI.msh')}")

    _export_volumes(S.pathfem, fnamehead, subject_dir, run_mni152, subject)
    _postprocess_volume_outputs(
        os.path.join(S.pathfem, 'Volume_Labels'),
        os.path.join(S.pathfem, 'Volume_Base'),
        S.pathfem
    )

    return scenario_dir


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel parameter sweep runner for TI simulations on MNI152."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel worker processes (default: 2)."
    )
    parser.add_argument(
        "--width-min",
        type=float,
        default=5.0,
        help="Minimum electrode width in mm (default: 5.0)."
    )
    parser.add_argument(
        "--width-max",
        type=float,
        default=10.0,
        help="Maximum electrode width in mm (default: 10.0)."
    )
    parser.add_argument(
        "--width-step",
        type=float,
        default=2.5,
        help="Width increment in mm (default: 2.5)."
    )
    parser.add_argument(
        "--height-min",
        type=float,
        default=30.0,
        help="Minimum electrode height in mm (default: 30.0)."
    )
    parser.add_argument(
        "--height-max",
        type=float,
        default=80.0,
        help="Maximum electrode height in mm (default: 80.0)."
    )
    parser.add_argument(
        "--height-step",
        type=float,
        default=2.5,
        help="Height increment in mm (default: 2.5)."
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default='~/Data/custom_electrodes/base_data/',
        help="Root directory of the dataset (default: current setting from TI_runner)."
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    width_values = np.arange(args.width_min, args.width_max + 1e-6, args.width_step)
    height_values = np.arange(args.height_min, args.height_max + 1e-6, args.height_step)

    current_scenarios = [
        {"name": "ratio_1to1", "ratio": "1:1", "pair1_mA": 1.5, "pair2_mA": 1.5},
        {"name": "ratio_1to2", "ratio": "1:2", "pair1_mA": 0.8, "pair2_mA": 1.6},
        {"name": "ratio_1to3", "ratio": "1:3", "pair1_mA": 0.6, "pair2_mA": 1.8},
    ]

    config = prepare_subject_environment(args.root_dir, run_mni152=True, mesh_present=True)

    combos = create_parameter_space(width_values, height_values, current_scenarios)
    total_jobs = len(combos)
    _log(f"Launching {total_jobs} simulations using {args.workers} workers.")

    results = []
    failures = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_single_combination, width, height, scenario, config): (width, height, scenario)
            for width, height, scenario in combos
        }

        for future in as_completed(futures):
            width, height, scenario = futures[future]
            try:
                result_path = future.result()
                results.append(result_path)
                _log(f"[DONE] {scenario['name']} w={width:.1f} h={height:.1f} -> {result_path}")
            except Exception as exc:
                failures.append((width, height, scenario, exc))
                _log(f"[FAIL] {scenario['name']} w={width:.1f} h={height:.1f}: {exc}")

    _log(f"Completed {len(results)} simulations with {len(failures)} failures.")

    if failures:
        for width, height, scenario, exc in failures:
            _log(f"  Failed combo {scenario['name']} w={width:.1f} h={height:.1f}: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
