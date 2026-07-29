#!/home/boyan/SimNIBS-4.5/bin/simnibs_python
# -*- coding: utf-8 -*-
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
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

from utils.paths import simnibs_root
from utils.sim_utils import format_output_dir
from target_montages import (
    MONTAGE_CHOICES,
    MONTAGE_PRESETS,
    PairSpec,
    MontageSpec,
    resolve_montage_preset,
    targets_csv_sha256,
)


SUBJECT = "MNI152"
DEFAULT_ROOT_DIR = "/home/boyan/sandbox/Jake_Data/MNI152-data"
DEFAULT_MNI_HEAD_MODEL_ROOT = (
    "/home/boyan/sandbox/Jake_Data/MNI152-data/m2m_MNI152"
)
DEFAULT_MNI_MESH_PATH = f"{DEFAULT_MNI_HEAD_MODEL_ROOT}/MNI152.msh"
DEFAULT_REFERENCE_T1_PATH = f"{DEFAULT_MNI_HEAD_MODEL_ROOT}/T1.nii.gz"
DEFAULT_EEG_CAP_NAME = "EEG10-10_UI_Jurak_2007.csv"
DEFAULT_ELEMENT_SIZE = 0.1
OUTPUT_SUBJECT_PATTERN = re.compile(r"^MNI152(?:-[a-z0-9-]+)?$")
CUSTOM_CONDUCTIVITIES = {
    "WM": 0.126,
    "GM": 0.276,
    "CSF": 1.65,
    "Skull": 0.01,
    "Scalp": 0.465,
    "Eye": 0.5,
    "Muscle": 0.16,
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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
            f"conductivity={preset.electrode_conductivity:.3g} S/m, "
            f"roi={preset.roi}, configuration={preset.configuration}"
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
    preset = resolve_montage_preset(args.preset)
    pair1 = PairSpec(
        anode=args.pair1_anode or preset.pair1.anode,
        cathode=args.pair1_cathode or preset.pair1.cathode,
        current_a=positive_current(
            args.pair1_current_a if args.pair1_current_a is not None else preset.pair1.current_amp,
            "pair1 current",
        ),
    )
    pair2 = PairSpec(
        anode=args.pair2_anode or preset.pair2.anode,
        cathode=args.pair2_cathode or preset.pair2.cathode,
        current_a=positive_current(
            args.pair2_current_a if args.pair2_current_a is not None else preset.pair2.current_amp,
            "pair2 current",
        ),
    )
    return MontageSpec(
        name=preset.name,
        description=preset.description,
        roi=preset.roi,
        e_target=preset.e_target,
        stimulated_volume=preset.stimulated_volume,
        configuration=preset.configuration,
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


def prepare_output_dirs(root_dir: str, output_subject: str) -> tuple[Path, Path]:
    if not OUTPUT_SUBJECT_PATTERN.fullmatch(output_subject):
        raise ValueError(
            "Output subject must be MNI152 or an ROI-specific MNI152-* name; "
            f"got {output_subject!r}."
        )
    output_root = simnibs_root(root_dir, output_subject)
    # Keep the established baseline layout:
    # MNI152-<roi>/anat/SimNIBS/Output/MNI152/.
    pathfem = output_root / "Output" / SUBJECT
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


def write_provenance(
    *,
    args: argparse.Namespace,
    montage: MontageSpec,
    mesh_path: Path,
    reference_t1_path: Path,
    eeg_cap_path: Path,
    output_root: Path,
    pathfem: Path,
    brain_only_path: Path,
    started_at: str,
) -> Path:
    mesh_1_path = pathfem / f"{SUBJECT}_TDCS_1_scalar.msh"
    mesh_2_path = pathfem / f"{SUBJECT}_TDCS_2_scalar.msh"
    ti_mesh_path = pathfem / "TI.msh"
    finished_at = datetime.now(timezone.utc).isoformat()
    simnibs_version = str(getattr(sim, "__version__", "unknown"))
    provenance = {
        "schema_version": 3,
        "status": "complete",
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "subject": SUBJECT,
        "output_subject": args.output_subject,
        "preset": montage.name,
        "roi": montage.roi,
        "configuration": montage.configuration,
        "software": {
            "simnibs_version": simnibs_version,
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "ti_method": "simnibs.utils.TI_utils.get_maxTI",
        },
        "inputs": {
            "mni_mesh": str(mesh_path),
            "mni_mesh_sha256": sha256_file(mesh_path),
            "reference_t1": str(reference_t1_path),
            "reference_t1_sha256": sha256_file(reference_t1_path),
            "eeg_cap": str(eeg_cap_path),
            "eeg_cap_sha256": sha256_file(eeg_cap_path),
            "head_model_manifest_sha256": args.head_model_manifest_sha256,
            "targets_csv_sha256": targets_csv_sha256(),
        },
        "stimulation": {
            "pair1": {
                "anode": montage.pair1.anode,
                "cathode": montage.pair1.cathode,
                "current_a": montage.pair1.current_amp,
            },
            "pair2": {
                "anode": montage.pair2.anode,
                "cathode": montage.pair2.cathode,
                "current_a": montage.pair2.current_amp,
            },
            "electrode_radius_mm": montage.electrode_radius_mm,
            "electrode_thickness_mm": montage.electrode_thickness_mm,
            "electrode_shape": montage.electrode_shape,
            "electrode_conductivity_s_per_m": montage.electrode_conductivity,
            "element_size": args.element_size,
            "conductivities_s_per_m": {
                **CUSTOM_CONDUCTIVITIES,
                "Saline": montage.electrode_conductivity,
            },
        },
        "outputs": {
            "ti_brain_only": {
                "path": str(brain_only_path),
                "size_bytes": brain_only_path.stat().st_size,
                "sha256": sha256_file(brain_only_path),
            },
            "ti_mesh": {
                "path": str(ti_mesh_path),
                "size_bytes": ti_mesh_path.stat().st_size,
            },
            "tdcs_scalar_meshes": [
                {"path": str(path), "size_bytes": path.stat().st_size}
                for path in (mesh_1_path, mesh_2_path)
            ],
        },
    }
    provenance_path = output_root / "mni_baseline_provenance.json"
    temporary_path = provenance_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(provenance_path)
    log_file_info("provenance", provenance_path)
    return provenance_path


def run_mni152(args: argparse.Namespace) -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    simnibs_version = str(getattr(sim, "__version__", "unknown"))
    if (
        args.expected_simnibs_version
        and simnibs_version != args.expected_simnibs_version
    ):
        raise RuntimeError(
            "Wrong SimNIBS runtime: "
            f"{simnibs_version!r} != {args.expected_simnibs_version!r}."
        )
    montage = build_montage(args)
    element_size = positive_scalar(args.element_size, "element size")
    mesh_path = Path(args.mni_mesh_path).expanduser().resolve()
    reference_t1_path = Path(args.reference_t1_path).expanduser().resolve()
    eeg_cap_path = Path(
        args.eeg_cap_path
        or mesh_path.parent / "eeg_positions" / DEFAULT_EEG_CAP_NAME
    ).expanduser().resolve()
    output_root, pathfem = prepare_output_dirs(
        args.root_dir,
        args.output_subject,
    )

    print(f"[INFO] Starting dedicated MNI152 TI pipeline with preset '{args.preset}'.")
    log_event(
        "software",
        simnibs_version=simnibs_version,
        expected_simnibs_version=args.expected_simnibs_version,
        python_version=sys.version,
        numpy_version=np.__version__,
    )
    log_file_info("mni_mesh", mesh_path)
    log_file_info("reference_t1", reference_t1_path)
    log_file_info("eeg_cap", eeg_cap_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"MNI152 mesh not found: {mesh_path}")
    if not reference_t1_path.exists():
        raise FileNotFoundError(f"MNI152 reference T1 not found: {reference_t1_path}")
    if not eeg_cap_path.exists():
        raise FileNotFoundError(f"MNI152 EEG cap not found: {eeg_cap_path}")

    log_event(
        "montage_config",
        preset=montage.name,
        description=montage.description,
        roi=montage.roi,
        e_target=montage.e_target,
        stimulated_volume=montage.stimulated_volume,
        configuration=montage.configuration,
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
        output_subject=args.output_subject,
        output_root=str(output_root),
    )

    session = sim_struct.SESSION()
    session.fnamehead = str(mesh_path)
    session.eeg_cap = str(eeg_cap_path)
    session.pathfem = str(pathfem)
    session.element_size = element_size
    session.map_to_vol = True
    session.open_in_gmsh = False

    tdcs1 = session.add_tdcslist()
    custom_conductivities = {
        **CUSTOM_CONDUCTIVITIES,
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
    brain_only_path = save_brain_only_ti(
        label_file_path,
        ti_volume_path,
        output_root,
    )
    write_provenance(
        args=args,
        montage=montage,
        mesh_path=mesh_path,
        reference_t1_path=reference_t1_path,
        eeg_cap_path=eeg_cap_path,
        output_root=output_root,
        pathfem=pathfem,
        brain_only_path=brain_only_path,
        started_at=started_at,
    )


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
            "<root-dir>/<output-subject>/anat/SimNIBS/."
        ),
    )
    parser.add_argument(
        "--output-subject",
        default=SUBJECT,
        help=(
            "Output directory name. The head model remains MNI152; this only "
            "allows isolated ROI-specific baseline directories."
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
        "--eeg-cap-path",
        help=(
            "Exact EEG coordinate file used to resolve named electrode "
            "centres. Defaults to the standard Jurak 10-10 cap next to the "
            "selected MNI152 mesh."
        ),
    )
    parser.add_argument(
        "--head-model-manifest-sha256",
        help=(
            "SHA-256 of the external manifest used to verify the complete "
            "staged MNI152 head-model bundle."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=MONTAGE_CHOICES,
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
        "--expected-simnibs-version",
        help="Fail before simulation unless the imported SimNIBS version matches.",
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
