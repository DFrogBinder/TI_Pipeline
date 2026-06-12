#!/home/boyan/SimNIBS-4.5/bin/simnibs_python
# -*- coding: utf-8 -*-
import argparse
import json
import os
import shutil
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
from scipy import ndimage as ndi
from simnibs import mesh_io, sim_struct
from simnibs.utils import TI_utils as TI


def _find_camcan_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "CamCan_Experiment",
        here.parents[2] / "CamCan_Experiment",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise ModuleNotFoundError(
        "Could not resolve CamCan_Experiment relative to "
        f"{here}. Checked: {', '.join(str(c) for c in candidates)}"
    )


ROOT = _find_camcan_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.sim_utils import atomic_replace  # noqa: E402


DEFAULT_ROOTDIR = "/mnt/parscratch/users/cop23bi/repeatability-ti-dataset"
REPEAT_PREFIX = "repeat_"
DEFAULT_REPEAT_COUNT = 40

# Deterministic meshing settings
USE_CUSTOM_LABELS_ONLY = True
SMOOTH_SCALP = True
SCALP_LABEL = 5
MORPH_KERNEL = (3, 3, 3)
CLOSE_ITERS = 1
OPEN_ITERS = 0
DILATE_ITERS = 0
ERODE_ITERS = 0


@dataclass(frozen=True)
class SubjectPaths:
    subject: str
    subject_root: Path
    anat_dir: Path
    t1_path: Path
    t2_path: Path
    seg_path: Path
    m2m_dir: Path
    mesh_path: Path


def log_event(event: str, **fields) -> None:
    payload = {"event": event, **fields}
    print(json.dumps(payload, default=str))


def log_file_info(label: str, path: Path) -> None:
    log_event(
        "file_info",
        label=label,
        path=str(path),
        exists=path.exists(),
        size_bytes=path.stat().st_size if path.exists() else None,
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


def _smooth_scalp_labels(label_data: np.ndarray) -> np.ndarray:
    mask = label_data == SCALP_LABEL
    structure = np.ones(MORPH_KERNEL, dtype=bool)
    if CLOSE_ITERS > 0:
        mask = ndi.binary_closing(mask, structure=structure, iterations=CLOSE_ITERS)
    if OPEN_ITERS > 0:
        mask = ndi.binary_opening(mask, structure=structure, iterations=OPEN_ITERS)
    if DILATE_ITERS > 0:
        mask = ndi.binary_dilation(mask, structure=structure, iterations=DILATE_ITERS)
    if ERODE_ITERS > 0:
        mask = ndi.binary_erosion(mask, structure=structure, iterations=ERODE_ITERS)

    out = label_data.copy()
    out[out == SCALP_LABEL] = 0
    out[mask] = SCALP_LABEL
    return out


def _repeat_tag(index: int, width: int = 3) -> str:
    return f"{REPEAT_PREFIX}{index:0{width}d}"


def _subject_paths(root_dir: Path, subject: str) -> SubjectPaths:
    anat_dir = root_dir / subject / "anat"
    m2m_dir = anat_dir / f"m2m_{subject}"
    return SubjectPaths(
        subject=subject,
        subject_root=root_dir / subject,
        anat_dir=anat_dir,
        t1_path=anat_dir / f"{subject}_T1w.nii",
        t2_path=anat_dir / f"{subject}_T2w.nii",
        seg_path=anat_dir / f"{subject}_T1w_ras_1mm_T1andT2_masks.nii",
        m2m_dir=m2m_dir,
        mesh_path=m2m_dir / f"{subject}.msh",
    )


def _discover_subjects(root_dir: Path) -> list[str]:
    subjects: list[str] = []
    for entry in sorted(root_dir.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "anat").is_dir():
            subjects.append(entry.name)
    return subjects


def _validate_subject_inputs(paths: SubjectPaths) -> None:
    missing = [
        str(path)
        for path in (paths.anat_dir, paths.t1_path, paths.t2_path, paths.seg_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Subject '{paths.subject}' is missing required inputs: {', '.join(missing)}"
        )


def _ensure_link(dest: Path, src: Path) -> None:
    if dest.is_symlink():
        if dest.exists():
            return
        dest.unlink()
    elif dest.exists():
        return
    if not src.exists():
        raise FileNotFoundError(f"Cannot create link, source is missing: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dest)


def _reset_dir_contents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_symlink():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _prepare_repeat_workspace(
    paths: SubjectPaths,
    repeat_tag: str,
    repeat_base: Path,
) -> tuple[Path, Path, Path, Path]:
    repeat_root = repeat_base / repeat_tag / paths.subject
    repeat_anat = repeat_root / "anat"
    output_root = repeat_anat / "SimNIBS"
    repeat_mesh_dir = repeat_anat / paths.m2m_dir.name
    repeat_mesh_path = repeat_mesh_dir / paths.mesh_path.name

    repeat_anat.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    _ensure_link(repeat_anat / paths.t1_path.name, paths.t1_path)
    _ensure_link(repeat_anat / paths.t2_path.name, paths.t2_path)
    _ensure_link(repeat_anat / paths.seg_path.name, paths.seg_path)
    _ensure_link(repeat_mesh_dir, paths.m2m_dir)

    return repeat_root, repeat_anat, output_root, repeat_mesh_path


def _ensure_subject_mesh(paths: SubjectPaths, *, force_remesh: bool = False) -> None:
    _validate_subject_inputs(paths)
    log_file_info("t1", paths.t1_path)
    log_file_info("t2", paths.t2_path)
    log_file_info("custom_seg_map", paths.seg_path)
    log_file_info("mesh", paths.mesh_path)

    if paths.mesh_path.exists() and not force_remesh:
        print(f"[INFO] Reusing existing mesh for {paths.subject}: {paths.mesh_path}")
        log_event("mesh_reuse", subject=paths.subject, mesh_path=str(paths.mesh_path))
        return

    print(f"[INFO] Generating mesh once for {paths.subject}")
    run_cmd(
        [
            "charm",
            paths.subject,
            str(paths.t1_path),
            str(paths.t2_path),
            "--forcerun",
            "--forceqform",
        ],
        cwd=str(paths.anat_dir),
        label="charm_init",
    )

    custom_seg_map = nib.load(str(paths.seg_path))
    charm_seg_map_path = (
        paths.anat_dir
        / f"m2m_sub-{paths.subject.split('-')[-1].upper()}"
        / "label_prep"
        / "tissue_labeling_upsampled.nii.gz"
    )
    log_file_info("charm_seg_map", charm_seg_map_path)

    def to_int_img(img: nib.Nifti1Image) -> nib.Nifti1Image:
        data = img.get_fdata(dtype=np.float32)
        if not np.allclose(data, np.round(data)):
            print(
                "[WARN] Custom segmentation contains non-integer values; "
                "rounding to nearest integers."
            )
        data = np.rint(data).astype(np.int16)
        return nib.Nifti1Image(data, img.affine, img.header)

    custom_int = to_int_img(custom_seg_map)
    label_data = custom_int.get_fdata(dtype=np.float32).astype(np.int16)

    if USE_CUSTOM_LABELS_ONLY and SMOOTH_SCALP:
        label_data = _smooth_scalp_labels(label_data)

    merged_seg_img_path = paths.anat_dir / f"{paths.subject}_T1w_ras_1mm_T1andT2_masks_merged.nii"
    out_img = nib.Nifti1Image(
        label_data.astype(np.uint16),
        custom_int.affine,
        custom_int.header,
    )
    nib.save(out_img, str(merged_seg_img_path))

    atomic_replace(
        str(merged_seg_img_path),
        str(charm_seg_map_path),
        force_int=True,
        int_dtype="uint16",
    )

    run_cmd(
        ["charm", paths.subject, "--mesh"],
        cwd=str(paths.anat_dir),
        label="charm_remesh",
    )

    if not paths.mesh_path.exists():
        raise FileNotFoundError(f"Expected mesh was not created: {paths.mesh_path}")


def _run_ti_pipeline(
    *,
    subject: str,
    subject_dir: Path,
    output_root: Path,
    fnamehead: Path,
    repeat_tag: str,
) -> float:
    subject_start = time.time()
    log_event("subject_start", subject=subject, repeat_tag=repeat_tag)
    _reset_dir_contents(output_root)

    electrode_size = [10, 2]
    electrode_shape = "ellipse"
    electrode_conductivity = 1.4
    custom_conductivities = {
        "WM": 0.126,
        "GM": 0.276,
        "CSF": 1.65,
        "Skull": 0.01,
        "Scalp": 0.465,
        "Eye": 0.5,
        "Muscle": 0.16,
        "Saline": electrode_conductivity,
    }

    montage_right = ("F10", 2e-3, "P8", -2e-3)
    montage_left = ("T7", 1.588656e-3, "P7", -1.588656e-3)

    S = sim_struct.SESSION()
    S.fnamehead = str(fnamehead)
    S.pathfem = str(output_root / "Output" / subject)
    Path(S.pathfem).mkdir(parents=True, exist_ok=True)
    S.element_size = 0.1
    S.map_to_vol = True

    tdcs1 = S.add_tdcslist()
    for conductivity in tdcs1.cond:
        if conductivity.name in custom_conductivities:
            conductivity.value = float(custom_conductivities[conductivity.name])
    tdcs1.currents = [montage_right[1], montage_right[3]]

    el1 = tdcs1.add_electrode()
    el1.channelnr = 1
    el1.centre = montage_right[0]
    el1.shape = electrode_shape
    el1.dimensions = [electrode_size[0] * 2, electrode_size[0] * 2]
    el1.thickness = electrode_size[1]

    el2 = tdcs1.add_electrode()
    el2.channelnr = 2
    el2.centre = montage_right[2]
    el2.shape = electrode_shape
    el2.dimensions = [electrode_size[0] * 2, electrode_size[0] * 2]
    el2.thickness = electrode_size[1]

    tdcs2 = S.add_tdcslist(deepcopy(tdcs1))
    tdcs2.currents = [montage_left[1], montage_left[3]]
    tdcs2.electrode[0].centre = montage_left[0]
    tdcs2.electrode[1].centre = montage_left[2]
    tdcs2.electrode[0].mesh_element_size = 0.1
    tdcs2.electrode[1].mesh_element_size = 0.1

    print(f"[INFO] Running SimNIBS for {subject} ({repeat_tag}) using mesh: {fnamehead}")
    log_event("simnibs_start", subject=subject, repeat_tag=repeat_tag)
    sim.run_simnibs(S)
    log_event("simnibs_done", subject=subject, repeat_tag=repeat_tag)

    m1 = mesh_io.read_msh(os.path.join(S.pathfem, f"{subject}_TDCS_1_scalar.msh"))
    m2 = mesh_io.read_msh(os.path.join(S.pathfem, f"{subject}_TDCS_2_scalar.msh"))

    tags_keep = np.hstack((np.arange(0, 499), np.arange(1000, 1499)))
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
    print(f"[INFO] Saved TI mesh to: {out_path}")

    volume_masks_path = Path(S.pathfem) / "Volume_Maks"
    volume_base_path = Path(S.pathfem) / "Volume_Base"
    volume_labels_path = Path(S.pathfem) / "Volume_Labels"
    volume_masks_path.mkdir(exist_ok=True)
    volume_base_path.mkdir(exist_ok=True)
    volume_labels_path.mkdir(exist_ok=True)

    labels_path = volume_labels_path / "TI_Volumetric_Labels"
    masks_path = volume_masks_path / "TI_Volumetric_Masks"
    ti_volume_path = volume_base_path / "TI_Volumetric_Base"
    t1_path = subject_dir / f"{subject}_T1w.nii"

    print(f"[INFO] Exporting volumetric meshes for {subject} ({repeat_tag})")
    run_cmd(
        ["msh2nii", os.path.join(S.pathfem, "TI.msh"), str(t1_path), str(labels_path), "--create_label"],
        label="msh2nii_labels",
    )
    run_cmd(
        ["msh2nii", os.path.join(S.pathfem, "TI.msh"), str(t1_path), str(masks_path), "--create_masks"],
        label="msh2nii_masks",
    )
    run_cmd(
        ["msh2nii", os.path.join(S.pathfem, "TI.msh"), str(t1_path), str(ti_volume_path)],
        label="msh2nii_volume",
    )

    label_candidates = sorted(
        [f for f in os.listdir(volume_labels_path) if f.startswith("TI_Volumetric_")]
        or os.listdir(volume_labels_path)
    )
    volume_candidates = sorted(
        [f for f in os.listdir(volume_base_path) if f.startswith("TI_Volumetric_")]
        or os.listdir(volume_base_path)
    )
    label_file_path = label_candidates[0]
    ti_volume_file = volume_candidates[0]
    log_event(
        "volume_selection",
        label_file=label_file_path,
        ti_volume_file=ti_volume_file,
        label_candidates=label_candidates,
        volume_candidates=volume_candidates,
    )

    if not label_file_path.endswith(".nii") and not label_file_path.endswith(".nii.gz"):
        raise ValueError("The label file is not a NIfTI file.")

    label_img = nib.load(str(volume_labels_path / label_file_path))
    ti_img = nib.load(str(volume_base_path / ti_volume_file))

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

    labels = np.asarray(label_img.dataobj).astype(np.int32, copy=False)
    GM_LABELS = {2}
    WM_LABELS = {1}
    brain_mask = np.isin(labels, list(GM_LABELS | WM_LABELS))

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

    masked_img = nib.Nifti1Image(masked, ti_img.affine, ti_img.header)
    masked_img.header.set_data_dtype(np.float32)
    ti_brain_only_path = output_root / "ti_brain_only.nii.gz"
    nib.save(masked_img, str(ti_brain_only_path))
    log_file_info("ti_brain_only", ti_brain_only_path)

    elapsed = time.time() - subject_start
    print(f"[INFO] Completed TI pipeline for {subject} ({repeat_tag}) in {elapsed:.2f} seconds.")
    log_event("subject_done", subject=subject, repeat_tag=repeat_tag, elapsed_sec=elapsed)
    return elapsed


def process_subject_repeats(
    subject: str,
    *,
    root_dir: Path,
    repeat_base: Path,
    repeat_count: int,
    force_remesh: bool = False,
) -> dict[str, object]:
    paths = _subject_paths(root_dir, subject)
    batch_start = time.time()
    log_event(
        "subject_batch_start",
        subject=subject,
        repeat_count=repeat_count,
        root_dir=str(root_dir),
        repeat_base=str(repeat_base),
    )

    _ensure_subject_mesh(paths, force_remesh=force_remesh)

    completed = 0
    skipped = 0
    total_repeat_elapsed = 0.0
    for repeat_index in range(1, repeat_count + 1):
        repeat_tag = _repeat_tag(repeat_index)
        _, repeat_anat, output_root, repeat_mesh_path = _prepare_repeat_workspace(
            paths,
            repeat_tag,
            repeat_base,
        )
        done_marker = output_root / "ti_brain_only.nii.gz"
        if done_marker.exists():
            print(f"[INFO] ({subject}) Repeat {repeat_tag} already complete, skipping.")
            skipped += 1
            continue
        if not repeat_mesh_path.exists():
            raise FileNotFoundError(
                f"Repeat workspace mesh link is missing for {subject} {repeat_tag}: "
                f"{repeat_mesh_path}"
            )
        total_repeat_elapsed += _run_ti_pipeline(
            subject=subject,
            subject_dir=repeat_anat,
            output_root=output_root,
            fnamehead=repeat_mesh_path,
            repeat_tag=repeat_tag,
        )
        completed += 1

    elapsed = time.time() - batch_start
    result = {
        "subject": subject,
        "completed_repeats": completed,
        "skipped_repeats": skipped,
        "total_repeat_runtime_sec": total_repeat_elapsed,
        "wallclock_runtime_sec": elapsed,
    }
    log_event("subject_batch_done", **result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch Temporal Interference runner that scans subjects under a root "
            "directory, generates each subject mesh once, and reuses that mesh "
            "across repeat simulations."
        )
    )
    parser.add_argument(
        "--rootdir",
        default=DEFAULT_ROOTDIR,
        help="Dataset root containing subject folders with an anat/ subdirectory.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Optional single subject ID. Omit to scan all subjects under --rootdir.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEAT_COUNT,
        help=f"Number of repeat simulations per subject. Defaults to {DEFAULT_REPEAT_COUNT}.",
    )
    parser.add_argument(
        "--repeat-base",
        default=None,
        help="Optional output root for repeat folders. Defaults to <rootdir>/repeats.",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Optional cap when scanning all subjects, useful for smoke tests.",
    )
    parser.add_argument(
        "--force-remesh",
        action="store_true",
        help="Rebuild the subject mesh once before running repeats, even if it already exists.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first subject failure instead of continuing to the next subject.",
    )
    args = parser.parse_args()

    root_dir = Path(args.rootdir).expanduser().resolve()
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    repeat_base = (
        Path(args.repeat_base).expanduser().resolve()
        if args.repeat_base
        else root_dir / "repeats"
    )
    repeat_base.mkdir(parents=True, exist_ok=True)

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    if args.subject:
        subjects = [args.subject.strip()]
    else:
        subjects = _discover_subjects(root_dir)
        if args.max_subjects is not None:
            subjects = subjects[: args.max_subjects]

    if not subjects:
        raise RuntimeError(f"No subject folders with anat/ found under {root_dir}")

    print(
        f"[INFO] Running mesh-reuse TI batch for {len(subjects)} subject(s) "
        f"with {args.repeats} repeat(s) each."
    )
    print(f"[INFO] Dataset root: {root_dir}")
    print(f"[INFO] Repeat output root: {repeat_base}")

    started = time.time()
    results: list[dict[str, object]] = []
    failures: dict[str, str] = {}
    for subject in subjects:
        try:
            results.append(
                process_subject_repeats(
                    subject,
                    root_dir=root_dir,
                    repeat_base=repeat_base,
                    repeat_count=args.repeats,
                    force_remesh=args.force_remesh,
                )
            )
        except Exception as exc:
            failures[subject] = str(exc)
            log_event("subject_batch_error", subject=subject, error=str(exc))
            print(f"[ERROR] Subject failed: {subject}: {exc}")
            if args.fail_fast:
                raise

    total_runtime = time.time() - started
    print("Done.")
    print(f"[INFO] Total execution time: {total_runtime:.2f} seconds.")
    print(f"[INFO] Subjects processed: {len(results)}")
    if failures:
        print(f"[WARN] Subjects failed: {len(failures)}")
        for subject, error in sorted(failures.items()):
            print(f"[WARN] {subject}: {error}")


if __name__ == "__main__":
    main()
