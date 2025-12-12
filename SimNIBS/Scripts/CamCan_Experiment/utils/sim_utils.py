"""
Simulation-side utilities for TI pipeline (meshing, segmentation merging, misc helpers).
"""
from __future__ import annotations

import inspect
import os
import tempfile
import types
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cloudpickle
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


def save_locals(dst, *, exclude=("__builtins__",), frame=None):
    """Serialize the caller’s locals into `dst`, skipping unpicklable entries."""
    if frame is None:
        frame = inspect.currentframe().f_back

    keep = {}
    skipped = []
    for name, value in frame.f_locals.items():
        if name in exclude or isinstance(value, types.FrameType):
            skipped.append(name)
            continue
        try:
            cloudpickle.dumps(value)
        except Exception:
            skipped.append(name)
            continue
        keep[name] = value

    path = Path(dst).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cloudpickle.dumps(keep))

    if skipped:
        print(f"[INFO] save_locals skipped {len(skipped)} names: {', '.join(skipped)}")
    print(f"[INFO] save_locals wrote {len(keep)} objects to {path}")


def img_info(label, img):
    zooms = img.header.get_zooms()[:3]
    shape = img.shape
    center_vox = (np.array(shape) - 1) / 2.0
    center_mm = img.affine @ np.append(center_vox, 1)
    print(f"{label}: shape={shape}, zooms={zooms}, axcodes={nib.aff2axcodes(img.affine)}")
    print(f"{label}: center_vox={center_vox}, center_mm={center_mm[:3]}")
    print(f"{label}: affine=\n{img.affine}\n")


def close_gmsh_windows():
    """Kill any lingering Gmsh GUI windows (best-effort)."""
    import subprocess

    stop_flag = True
    while stop_flag:
        try:
            result = subprocess.run(['bash', '-c', 'xdotool search --name "Gmsh" windowkill'])
            if result.returncode == 1:
                stop_flag = False
        except Exception:
            stop_flag = False


def format_output_dir(directory_path: str) -> None:
    """Delete all files in a folder (leave subfolders intact)."""
    if not os.path.isdir(directory_path):
        print(f"Not a directory: {directory_path}")
        return
    for fname in os.listdir(directory_path):
        fpath = os.path.join(directory_path, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)
            print(f"Deleted {fpath}")


def atomic_replace(src_path: str, dst_path: str, force_int: bool = False, int_dtype: str = "uint16"):
    """
    Atomically replace dst_path with src_path's NIfTI contents.
    Handles .nii <-> .nii.gz re-encoding and preserves the source file.
    Optionally force integer dtype for segmentation maps.
    """
    src = Path(src_path)
    dst = Path(dst_path)
    dst_dir = dst.parent
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst_suffix = "".join(dst.suffixes)  # handles '.nii.gz'
    fd, tmp_name = tempfile.mkdtemp(dir=dst_dir), None
    tmp_name = str(Path(fd) / f"tmp{dst_suffix}")

    try:
        img = nib.load(str(src))
        if force_int:
            data = np.asarray(img.dataobj).astype(int_dtype, copy=False)
            out = nib.Nifti1Image(data, img.affine, img.header)
            out.header.set_data_dtype(int_dtype)
            nib.save(out, tmp_name)
        else:
            nib.save(img, tmp_name)

        with open(tmp_name, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp_name, str(dst))
    except Exception:
        try:
            if tmp_name and os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except Exception:
            pass
        raise


def merge_segmentation_maps(
    manual_seg: Union[str, nib.Nifti1Image],
    charm_seg: Union[str, nib.Nifti1Image],
    *,
    output_path: Optional[str] = None,
    manual_skin_id: int = 5,
    background_label: int = 0,
    dilate_envelope_voxels: int = 0,
) -> tuple[nib.Nifti1Image, Dict[str, Any]]:
    """
    Clip manual scalp to CHARM's head envelope (>0):
      1) envelope = (CHARM > 0)
      2) manual scalp outside envelope -> background_label
      3) keep other manual labels as-is
    If grids differ, manual is resampled to CHARM grid (NN).
    """
    def _resample_nn(src_img, tgt_img):
        return resample_from_to(src_img, tgt_img, order=0)

    # --- Load ---
    man_img = nib.load(manual_seg) if not isinstance(manual_seg, nib.Nifti1Image) else manual_seg
    cha_img = nib.load(charm_seg) if not isinstance(charm_seg, nib.Nifti1Image) else charm_seg

    # Reorient to canonical for safer comparison
    man_img = nib.as_closest_canonical(man_img)
    cha_img = nib.as_closest_canonical(cha_img)

    # Align manual -> CHARM grid if needed
    if (man_img.shape != cha_img.shape) or (not np.allclose(man_img.affine, cha_img.affine, atol=1e-5)):
        man_img = _resample_nn(man_img, cha_img)

    man = np.asarray(man_img.get_fdata(), dtype=np.int32)
    cha = np.asarray(cha_img.get_fdata(), dtype=np.int32)

    envelope = cha > 0

    # Optional dilation
    if dilate_envelope_voxels > 0:
        from scipy.ndimage import binary_dilation
        env = envelope
        for _ in range(int(dilate_envelope_voxels)):
            env = binary_dilation(env)
        envelope = env

    man_scalp = (man == manual_skin_id)
    outside = man_scalp & (~envelope)

    removed_voxels = int(outside.sum())
    total_scalp_voxels = int(man_scalp.sum())

    out_arr = man.copy()
    out_arr[outside] = background_label

    out_img = nib.Nifti1Image(out_arr.astype(np.int16, copy=False), cha_img.affine, cha_img.header)

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(out_img, str(out_path))

    debug = {
        "removed_outside_voxels": removed_voxels,
        "manual_scalp_voxels": total_scalp_voxels,
        "dilate_envelope_voxels": int(dilate_envelope_voxels),
        "background_label": int(background_label),
    }
    return out_img, debug
