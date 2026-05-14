#!/home/boyan/SimNIBS-4.5/bin/simnibs_python
# -*- coding: utf-8 -*-
import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import simnibs as sim
from nibabel.processing import resample_from_to
from simnibs import mesh_io, sim_struct
from simnibs.utils import TI_utils as TI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paths import sim_output_dir, simnibs_root
from utils.sim_utils import format_output_dir


SUBJECT = "MNI152"
DEFAULT_ROOT_DIR = "/home/boyan/sandbox/Jake_Data/MNI152-data"
DEFAULT_MNI_MESH_PATH = "/home/boyan/sandbox/simnibs4_exmaples/m2m_MNI152/MNI152.msh"
DEFAULT_REFERENCE_T1_PATH = "/home/boyan/sandbox/simnibs4_exmaples/m2m_MNI152/T1.nii.gz"
DEFAULT_ELEMENT_SIZE = 0.1


@dataclass(frozen=True)
class PairSpec:
    anode: str
    cathode: str
    current_amp: float


@dataclass(frozen=True)
class MontageSpec:
    name: str
    description: str
    pair1: PairSpec
    pair2: PairSpec
    electrode_radius_mm: float = 10.0
    electrode_thickness_mm: float = 1.0
    electrode_shape: str = "ellipse"
    electrode_conductivity: float = 0.85


MONTAGE_PRESETS: dict[str, MontageSpec] = {
    "right-thalamus": MontageSpec(
        name="right-thalamus",
        description="Right-thalamus preset from the existing commented MNI152 montage block.",
        pair1=PairSpec("AF7", "TP7", 2e-3),
        pair2=PairSpec("T8", "PO8", 2e-3),
    ),
    "left-hippocampus": MontageSpec(
        name="left-hippocampus",
        description="Left-hippocampus preset from the existing MNI152 runner.",
        pair1=PairSpec("F10", "P8", 2e-3),
        pair2=PairSpec("T7", "P7", 1.588656e-3),
    ),
    "left-m1": MontageSpec(
        name="left-m1",
        description="Primary motor cortex preset from the existing MNI152 runner.",
        pair1=PairSpec("FC1", "FCz", 1.34e-3),
        pair2=PairSpec("C3", "P5", 2.66e-3),
    ),
    "right-dlpc": MontageSpec(
        name="right-dlpc",
        description="Right-dlpc preset from the existing MNI152 runner.",
        pair1=PairSpec("AF4", "F4", 0.796214e-3),
        pair2=PairSpec("C2", "CP1", 2e-3),
    ),
    "right-m1": MontageSpec(
        name="right-m1",
        description="Right M1 montage that was previously active in this runner.",
        pair1=PairSpec("FC6", "FT8", 2e-3),
        pair2=PairSpec("C2", "C4", 0.796214e-3),
    )
}


def log_event(event: str, **fields) -> None:
    payload = {"event": event, **fields}
    print(json.dumps(payload, default=str))


def log_file_info(label: str, path: str | Path) -> None:
    candidate = Path(path)
    log_event(
        "file_info",
        label=label,
        path=str(candidate),
        exists=candidate.exists(),
        size_bytes=candidate.stat().st_size if candidate.exists() else None,
    )


def run_cmd(cmd: list[str], *, cwd: str | None = None, label: str = "cmd") -> None:
    log_event("run_cmd", label=label, cmd=cmd, cwd=cwd)
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    log_event(
        "cmd_result",
        label=label,
        returncode=result.returncode,
        stdout_tail=result.stdout[-2000:] if result.stdout else "",
        stderr_tail=result.stderr[-2000:] if result.stderr else "",
    )
    result.check_returncode()


def select_output_file(directory: str | Path, prefix: str) -> tuple[str, list[str]]:
    directory = str(directory)
    candidates = sorted(
        [name for name in os.listdir(directory) if name.startswith(prefix)]
        or os.listdir(directory)
    )
    if not candidates:
        raise FileNotFoundError(f"No output files found in '{directory}'.")

    for candidate in candidates:
        if candidate.endswith((".nii", ".nii.gz")):
            return candidate, candidates

    raise ValueError(f"No NIfTI outputs found in '{directory}': {candidates}")


def list_presets() -> None:
    print("Available MNI152 montage presets:")
    for name in sorted(MONTAGE_PRESETS):
        preset = MONTAGE_PRESETS[name]
        print(
            f"- {name}: "
            f"pair1={preset.pair1.anode}->{preset.pair1.cathode} ({preset.pair1.current_amp:.6g} A), "
            f"pair2={preset.pair2.anode}->{preset.pair2.cathode} ({preset.pair2.current_amp:.6g} A), "
            f"electrode radius={preset.electrode_radius_mm:.1f} mm, "
            f"thickness={preset.electrode_thickness_mm:.1f} mm, "
            f"conductivity={preset.electrode_conductivity:.3g} S/m"
        )
        print(f"  {preset.description}")


def positive_current(value: float, label: str) -> float:
    if value <= 0:
        raise ValueError(f"{label} must be > 0 A, got {value}.")
    return float(value)


def positive_scalar(value: float, label: str) -> float:
    if value <= 0:
        raise ValueError(f"{label} must be > 0, got {value}.")
    return float(value)


def build_montage(args: argparse.Namespace) -> MontageSpec:
    preset = MONTAGE_PRESETS[args.preset]
    pair1 = PairSpec(
        anode=args.pair1_anode or preset.pair1.anode,
        cathode=args.pair1_cathode or preset.pair1.cathode,
        current_amp=positive_current(
            args.pair1_current_a if args.pair1_current_a is not None else preset.pair1.current_amp,
            "pair1 current",
        ),
    )
    pair2 = PairSpec(
        anode=args.pair2_anode or preset.pair2.anode,
        cathode=args.pair2_cathode or preset.pair2.cathode,
        current_amp=positive_current(
            args.pair2_current_a if args.pair2_current_a is not None else preset.pair2.current_amp,
            "pair2 current",
        ),
    )
    return MontageSpec(
        name=preset.name,
        description=preset.description,
        pair1=pair1,
        pair2=pair2,
        electrode_radius_mm=positive_scalar(
            args.electrode_radius_mm
            if args.electrode_radius_mm is not None else preset.electrode_radius_mm,
            "electrode radius",
        ),
        electrode_thickness_mm=positive_scalar(
            args.electrode_thickness_mm
            if args.electrode_thickness_mm is not None else preset.electrode_thickness_mm,
            "electrode thickness",
        ),
        electrode_shape=args.electrode_shape or preset.electrode_shape,
        electrode_conductivity=positive_scalar(
            args.electrode_conductivity
            if args.electrode_conductivity is not None else preset.electrode_conductivity,
            "electrode conductivity",
        ),
    )


def add_pair_to_session(
    tdcs,
    pair: PairSpec,
    *,
    electrode_radius_mm: float,
    electrode_thickness_mm: float,
    electrode_shape: str,
) -> None:
    tdcs.currents = [pair.current_amp, -pair.current_amp]

    anode = tdcs.add_electrode()
    anode.channelnr = 1
    anode.centre = pair.anode
    anode.shape = electrode_shape
    anode.dimensions = [electrode_radius_mm * 2.0, electrode_radius_mm * 2.0]
    anode.thickness = electrode_thickness_mm

    cathode = tdcs.add_electrode()
    cathode.channelnr = 2
    cathode.centre = pair.cathode
    cathode.shape = electrode_shape
    cathode.dimensions = [electrode_radius_mm * 2.0, electrode_radius_mm * 2.0]
    cathode.thickness = electrode_thickness_mm


def prepare_output_dirs(root_dir: str) -> tuple[Path, Path]:
    output_root = simnibs_root(root_dir, SUBJECT)
    pathfem = sim_output_dir(root_dir, SUBJECT)
    output_root.mkdir(parents=True, exist_ok=True)
    pathfem.mkdir(parents=True, exist_ok=True)
    format_output_dir(str(pathfem))
    return output_root, pathfem


def export_volumes(out_path: str, reference_t1_path: str, pathfem: Path) -> tuple[Path, Path]:
    volume_masks_path = pathfem / "Volume_Maks"
    volume_base_path = pathfem / "Volume_Base"
    volume_labels_path = pathfem / "Volume_Labels"
    for directory in (volume_masks_path, volume_base_path, volume_labels_path):
        directory.mkdir(parents=True, exist_ok=True)
        format_output_dir(str(directory))

    labels_path = str(volume_labels_path / "TI_Volumetric_Labels")
    masks_path = str(volume_masks_path / "TI_Volumetric_Masks")
    ti_volume_prefix = str(volume_base_path / "TI_Volumetric_Base")

    print("Exporting volumetric meshes...")
    run_cmd(
        ["msh2nii", out_path, reference_t1_path, labels_path, "--create_label"],
        label="msh2nii_labels",
    )
    run_cmd(
        ["msh2nii", out_path, reference_t1_path, masks_path, "--create_masks"],
        label="msh2nii_masks",
    )
    run_cmd(
        ["msh2nii", out_path, reference_t1_path, ti_volume_prefix],
        label="msh2nii_volume",
    )

    label_file_name, label_candidates = select_output_file(volume_labels_path, "TI_Volumetric_")
    ti_volume_file_name, volume_candidates = select_output_file(volume_base_path, "TI_Volumetric_")
    log_event(
        "volume_selection",
        label_file=label_file_name,
        ti_volume_file=ti_volume_file_name,
        label_candidates=label_candidates,
        volume_candidates=volume_candidates,
    )

    label_file_path = volume_labels_path / label_file_name
    ti_volume_path = volume_base_path / ti_volume_file_name
    log_file_info("label_volume", label_file_path)
    log_file_info("ti_volume", ti_volume_path)
    return label_file_path, ti_volume_path


def save_brain_only_ti(
    label_file_path: Path,
    ti_volume_path: Path,
    output_root: Path,
) -> Path:
    label_img = nib.load(str(label_file_path))
    ti_img = nib.load(str(ti_volume_path))

    same_shape = ti_img.shape == label_img.shape
    same_affine = np.allclose(ti_img.affine, label_img.affine, atol=1e-3)
    log_event(
        "grid_check",
        same_shape=same_shape,
        same_affine=same_affine,
        ti_shape=ti_img.shape,
        label_shape=label_img.shape,
    )
    if not (same_shape and same_affine):
        label_img = resample_from_to(label_img, ti_img, order=0)

    labels = np.rint(np.asarray(label_img.dataobj)).astype(np.int32, copy=False)
    brain_mask = np.isin(labels, [1, 2])

    ti_data = ti_img.get_fdata(dtype=np.float32)
    masked = np.where(brain_mask, ti_data, np.nan).astype(np.float32)
    finite = np.isfinite(masked)
    log_event(
        "ti_stats",
        ti_min=float(np.nanmin(ti_data)),
        ti_max=float(np.nanmax(ti_data)),
        ti_mean=float(np.nanmean(ti_data)),
        brain_mask_voxels=int(brain_mask.sum()),
        masked_finite_voxels=int(finite.sum()),
    )

    masked_output_path = output_root / "ti_brain_only.nii.gz"
    masked_img = nib.Nifti1Image(masked, ti_img.affine, ti_img.header)
    masked_img.header.set_data_dtype(np.float32)
    nib.save(masked_img, str(masked_output_path))
    log_file_info("ti_brain_only", masked_output_path)
    return masked_output_path


def run_mni152(args: argparse.Namespace) -> None:
    montage = build_montage(args)
    element_size = positive_scalar(args.element_size, "element size")
    mesh_path = Path(args.mni_mesh_path).expanduser().resolve()
    reference_t1_path = Path(args.reference_t1_path).expanduser().resolve()
    output_root, pathfem = prepare_output_dirs(args.root_dir)

    print(f"[INFO] Starting dedicated MNI152 TI pipeline with preset '{args.preset}'.")
    log_file_info("mni_mesh", mesh_path)
    log_file_info("reference_t1", reference_t1_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"MNI152 mesh not found: {mesh_path}")
    if not reference_t1_path.exists():
        raise FileNotFoundError(f"MNI152 reference T1 not found: {reference_t1_path}")

    log_event(
        "montage_config",
        preset=montage.name,
        description=montage.description,
        pair1_anode=montage.pair1.anode,
        pair1_cathode=montage.pair1.cathode,
        pair1_current_a=montage.pair1.current_amp,
        pair2_anode=montage.pair2.anode,
        pair2_cathode=montage.pair2.cathode,
        pair2_current_a=montage.pair2.current_amp,
        electrode_radius_mm=montage.electrode_radius_mm,
        electrode_thickness_mm=montage.electrode_thickness_mm,
        electrode_shape=montage.electrode_shape,
        electrode_conductivity=montage.electrode_conductivity,
        output_root=str(output_root),
    )

    session = sim_struct.SESSION()
    session.fnamehead = str(mesh_path)
    session.pathfem = str(pathfem)
    session.element_size = element_size
    session.map_to_vol = True

    tdcs1 = session.add_tdcslist()
    custom_conductivities = {
        "WM": 0.126,
        "GM": 0.276,
        "CSF": 1.65,
        "Skull": 0.01,
        "Scalp": 0.465,
        "Eye": 0.5,
        "Muscle": 0.16,
        "Saline": montage.electrode_conductivity,
    }
    for conductivity in tdcs1.cond:
        if conductivity.name in custom_conductivities:
            conductivity.value = float(custom_conductivities[conductivity.name])
            print(f"[COND] {conductivity.name} set to {conductivity.value} S/m")

    add_pair_to_session(
        tdcs1,
        montage.pair1,
        electrode_radius_mm=montage.electrode_radius_mm,
        electrode_thickness_mm=montage.electrode_thickness_mm,
        electrode_shape=montage.electrode_shape,
    )

    tdcs2 = session.add_tdcslist(deepcopy(tdcs1))
    tdcs2.currents = [montage.pair2.current_amp, -montage.pair2.current_amp]
    tdcs2.electrode[0].centre = montage.pair2.anode
    tdcs2.electrode[1].centre = montage.pair2.cathode
    tdcs2.electrode[0].shape = montage.electrode_shape
    tdcs2.electrode[1].shape = montage.electrode_shape
    tdcs2.electrode[0].dimensions = [montage.electrode_radius_mm * 2.0, montage.electrode_radius_mm * 2.0]
    tdcs2.electrode[1].dimensions = [montage.electrode_radius_mm * 2.0, montage.electrode_radius_mm * 2.0]
    tdcs2.electrode[0].thickness = montage.electrode_thickness_mm
    tdcs2.electrode[1].thickness = montage.electrode_thickness_mm
    tdcs2.electrode[0].mesh_element_size = element_size
    tdcs2.electrode[1].mesh_element_size = element_size

    print("[INFO] Running SimNIBS for MNI152 TI mesh.")
    log_event("simnibs_start", subject=SUBJECT)
    sim.run_simnibs(session)
    log_event("simnibs_done", subject=SUBJECT)

    mesh_1_path = pathfem / f"{SUBJECT}_TDCS_1_scalar.msh"
    mesh_2_path = pathfem / f"{SUBJECT}_TDCS_2_scalar.msh"
    m1 = mesh_io.read_msh(str(mesh_1_path))
    m2 = mesh_io.read_msh(str(mesh_2_path))

    tags_keep = np.hstack((np.arange(0, 499), np.arange(1000, 1499)))
    m1 = m1.crop_mesh(tags=tags_keep)
    m2 = m2.crop_mesh(tags=tags_keep)

    e1_vec = m1.field["E"]
    e2_vec = m2.field["E"]
    timax = TI.get_maxTI(e1_vec.value, e2_vec.value)

    mout = deepcopy(m1)
    mout.elmdata = []
    mout.add_element_field(timax, "TImax")

    out_path = pathfem / "TI.msh"
    mesh_io.write_msh(mout, str(out_path))
    log_file_info("ti_mesh", out_path)
    print(f"Saved gray+white TI mesh to: {out_path}")

    label_file_path, ti_volume_path = export_volumes(
        str(out_path),
        str(reference_t1_path),
        pathfem,
    )
    save_brain_only_ti(label_file_path, ti_volume_path, output_root)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dedicated MNI152 Temporal Interference runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root-dir",
        default=DEFAULT_ROOT_DIR,
        help=(
            "Experiment root. Outputs are written under "
            "<root-dir>/MNI152/anat/SimNIBS/."
        ),
    )
    parser.add_argument(
        "--mni-mesh-path",
        default=DEFAULT_MNI_MESH_PATH,
        help="Path to the SimNIBS MNI152 head mesh.",
    )
    parser.add_argument(
        "--reference-t1-path",
        default=DEFAULT_REFERENCE_T1_PATH,
        help="Reference T1 used for msh2nii export.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(MONTAGE_PRESETS),
        default="left-thalamus",
        help="Named montage preset to run on the MNI152 template.",
    )
    parser.add_argument(
        "--pair1-anode",
        help="Override preset anode electrode centre for the first montage pair.",
    )
    parser.add_argument(
        "--pair1-cathode",
        help="Override preset cathode electrode centre for the first montage pair.",
    )
    parser.add_argument(
        "--pair1-current-a",
        type=float,
        help="Override preset current magnitude for the first montage pair.",
    )
    parser.add_argument(
        "--pair2-anode",
        help="Override preset anode electrode centre for the second montage pair.",
    )
    parser.add_argument(
        "--pair2-cathode",
        help="Override preset cathode electrode centre for the second montage pair.",
    )
    parser.add_argument(
        "--pair2-current-a",
        type=float,
        help="Override preset current magnitude for the second montage pair.",
    )
    parser.add_argument(
        "--electrode-radius-mm",
        type=float,
        help="Override preset electrode radius in mm.",
    )
    parser.add_argument(
        "--electrode-thickness-mm",
        type=float,
        help="Override preset electrode thickness in mm.",
    )
    parser.add_argument(
        "--electrode-shape",
        help="Override preset electrode shape, e.g. ellipse.",
    )
    parser.add_argument(
        "--electrode-conductivity",
        type=float,
        help="Override preset electrode saline conductivity in S/m.",
    )
    parser.add_argument(
        "--element-size",
        type=float,
        default=DEFAULT_ELEMENT_SIZE,
        help="SimNIBS element size used for the session and electrode refinement.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print available montage presets and exit.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_presets:
        list_presets()
        return

    start = time.time()
    run_mni152(args)
    total_runtime = time.time() - start
    print("Done.")
    print(f"[INFO] Total execution time: {total_runtime:.2f} seconds.")


if __name__ == "__main__":
    main()
