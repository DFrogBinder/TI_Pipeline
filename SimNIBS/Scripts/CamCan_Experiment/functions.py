import os
import shutil
import tempfile
import json
import numpy as np
import nibabel as nib
import subprocess

import csv, numpy as np, nibabel as nib

from pathlib import Path
from scipy.ndimage import binary_dilation  
from typing import Tuple, Union, Optional, Sequence
from typing import Optional, Sequence, Tuple, Dict, Any, Union

from nilearn import datasets, image as nli, plotting
from nibabel.processing import resample_from_to
from nibabel.affines import apply_affine

NiftiLike = Union[str, os.PathLike, nib.Nifti1Image]

# Which ROIs to extract. The queries are matched case-insensitively against labels.
# We fetch the appropriate HO atlas (cortical OR subcortical) per-ROI under the hood.
ROI_QUERIES = {
    "M1":          {"atlas": "cort-maxprob-thr25-2mm", "query": "precentral gyrus"},
    "Hippocampus": {"atlas": "sub-maxprob-thr25-2mm",  "query": "hippocampus"},
}

EFIELD_PERCENTILE     = 95   # top percentile (e.g., 90 or 95)
WRITE_PER_VOXEL_CSV   = True # CSV per voxel; set False if files get too big
# ------------------------------------------------
def close_gmsh_windows():
    """Kill any lingering Gmsh GUI windows."""
    stop_flag = True
    while stop_flag:
        try:
            result = subprocess.run(
                ['bash', '-c', 'xdotool search --name "Gmsh" windowkill']
            )
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
def generate_mesh_from_nii( output_path: str, T1_path: str, T2_path: str = None) -> str:
    try:
        subprocess.run(["charm", T1_path, T2_path, output_path])
        return True
    except Exception as e:
        print(f"Error creating volumetric mesh: {e}")
        return False

def atomic_replace(src_path: str, dst_path: str, force_int: bool = False, int_dtype: str = "uint16"):
    """
    Atomically replace dst_path with src_path's NIfTI contents.
    Handles .nii <-> .nii.gz re-encoding and preserves the source file.
    Optionally force integer dtype for segmentation maps.

    Args:
        src_path: path to the source NIfTI (.nii or .nii.gz)
        dst_path: path to the target NIfTI (.nii or .nii.gz)
        force_int: if True, cast data to integer (use for label maps)
        int_dtype: integer dtype to cast to when force_int=True (e.g., 'uint16' or 'int16')
    """
    src = Path(src_path)
    dst = Path(dst_path)
    dst_dir = dst.parent
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Use a temp file in the destination directory so the final os.replace is atomic
    dst_suffix = "".join(dst.suffixes)  # handles '.nii.gz'
    fd, tmp_name = tempfile.mkstemp(dir=dst_dir, suffix=dst_suffix)
    os.close(fd)

    try:
        img = nib.load(str(src))
        if force_int:
            data = np.asarray(img.dataobj).astype(int_dtype, copy=False)
            out = nib.Nifti1Image(data, img.affine, img.header)
            out.header.set_data_dtype(int_dtype)
            nib.save(out, tmp_name)
        else:
            # Re-save in the destination's format (nibabel picks compression from extension)
            nib.save(img, tmp_name)

        # Ensure file contents are flushed to disk
        with open(tmp_name, "rb") as f:
            os.fsync(f.fileno())

        # Fsync the directory metadata (best-effort)
        try:
            dfd = os.open(str(dst_dir), os.O_DIRECTORY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except Exception:
            pass

        # Atomic swap
        os.replace(tmp_name, str(dst))
    except Exception:
        try:
            os.unlink(tmp_name)
        except Exception:
            pass
        raise

def merge_segmentation_maps(
    manual_seg: NiftiLike,
    charm_seg: NiftiLike,
    *,
    output_path: Optional[str] = None,
    manual_skin_id: int = 5,
    background_label: int = 0,
    dilate_envelope_voxels: int = 0,
    # optional artifacts
    save_envelope_path: Optional[str] = None,
    save_outside_mask_path: Optional[str] = None,
    qc_dir: Optional[str] = None,
    make_summary_json: bool = False,
) -> Tuple[nib.Nifti1Image, Dict[str, Any]]:
    """
    Clip manual scalp to CHARM's head envelope (>0), as suggested:
      1) envelope = (CHARM > 0)
      2) manual scalp outside envelope -> set to background_label
      3) keep all other manual labels as-is
    If grids differ, manual is resampled to CHARM grid using nearest-neighbor.

    Parameters
    ----------
    manual_seg : path or NIfTI
        Manual/Jake segmentation (integer labels).
    charm_seg : path or NIfTI
        CHARM segmentation (integer labels). Defines target grid & envelope.
    output_path : str or None
        If given, save the clipped segmentation here (.nii or .nii.gz).
    manual_skin_id : int
        Label id used for scalp/skin in the manual segmentation.
    background_label : int
        Label id to assign where manual scalp lies outside the envelope.
    dilate_envelope_voxels : int
        Optional morphological dilation steps for the envelope (forgiving on tight areas).
    save_envelope_path : str or None
        If set, save the (possibly dilated) envelope as a NIfTI (uint8).
    save_outside_mask_path : str or None
        If set, save the mask of manual scalp voxels removed (uint8).
    qc_dir : str or None
        If set, save a few axial QC PNGs (requires matplotlib).
    make_summary_json : bool
        If True and qc_dir is set, write a summary.json there.

    Returns
    -------
    out_img : nib.Nifti1Image
        The clipped segmentation on CHARM's grid.
    debug : dict
        Stats and parameter echo.
    """
    def _resample_nn(src_img, tgt_img):
        from nibabel.processing import resample_from_to
        return resample_from_to(src_img, tgt_img, order=0)

    def _maybe_dilate(mask: np.ndarray, steps: int) -> np.ndarray:
        if steps <= 0:
            return mask
        try:
            from scipy.ndimage import binary_dilation
        except Exception:
            # SciPy not available -> skip dilation
            return mask
        out = mask.copy()
        for _ in range(int(steps)):
            out = binary_dilation(out)
        return out

    # --- Load ---
    man_img = nib.load(manual_seg) if not isinstance(manual_seg, nib.Nifti1Image) else manual_seg
    cha_img = nib.load(charm_seg)  if not isinstance(charm_seg,  nib.Nifti1Image)  else charm_seg

    # --- Align manual -> CHARM grid if needed (NN preserves labels) ---
    if (man_img.shape != cha_img.shape) or (not np.allclose(man_img.affine, cha_img.affine, atol=1e-5)):
        man_img = _resample_nn(man_img, cha_img)

    man = np.asarray(man_img.get_fdata(), dtype=np.int32)
    cha = np.asarray(cha_img.get_fdata(), dtype=np.int32)

    # --- Build envelope (>0), optional dilation ---
    envelope = cha > 0
    envelope = _maybe_dilate(envelope, dilate_envelope_voxels)

    # --- Manual scalp & outside mask ---
    man_scalp = (man == manual_skin_id)
    outside = man_scalp & (~envelope)

    removed_voxels = int(outside.sum())
    total_scalp_voxels = int(man_scalp.sum())

    # --- Apply clipping ---
    out_arr = man.copy()
    out_arr[outside] = background_label

    out_img = nib.Nifti1Image(out_arr.astype(np.int16, copy=False), cha_img.affine, cha_img.header)

    # --- Save main output (optional) ---
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        nib.save(out_img, output_path)

    # --- Save side artifacts (optional) ---
    if save_envelope_path:
        env_img = nib.Nifti1Image(envelope.astype(np.uint8, copy=False), cha_img.affine, cha_img.header)
        nib.save(env_img, save_envelope_path)

    if save_outside_mask_path:
        outside_img = nib.Nifti1Image(outside.astype(np.uint8, copy=False), cha_img.affine, cha_img.header)
        nib.save(outside_img, save_outside_mask_path)

    # --- QC PNGs (optional) ---
    if qc_dir:
        try:
            import matplotlib.pyplot as plt
            os.makedirs(qc_dir, exist_ok=True)
            Z = out_arr.shape[2]
            zs = sorted(set([Z // 2, Z // 3, (2 * Z) // 3]))
            for i, z in enumerate(zs, 1):
                base = envelope[:, :, z].astype(float)
                pre_scalp = (man[:, :, z] == manual_skin_id)
                post_scalp = (out_arr[:, :, z] == manual_skin_id)
                removed = outside[:, :, z]

                fig = plt.figure(figsize=(6, 6), dpi=120)
                plt.imshow(base, vmin=0, vmax=1)  # grayscale envelope
                # simple scatter overlays (fast & readable)
                ys, xs = np.where(pre_scalp)
                plt.scatter(xs, ys, s=0.15, label="manual scalp (pre)")
                ys, xs = np.where(removed)
                if ys.size:
                    plt.scatter(xs, ys, s=0.15, label="removed (outside)")
                ys, xs = np.where(post_scalp)
                plt.scatter(xs, ys, s=0.15, label="manual scalp (post)")
                plt.title(f"QC: z={z}")
                plt.legend(markerscale=10, loc="lower right", fontsize=7)
                plt.axis("off")
                out_png = os.path.join(qc_dir, f"qc_axial_{i}_z{z}.png")
                plt.savefig(out_png, bbox_inches="tight")
                plt.close(fig)
        except Exception:
            pass  # QC is optional; ignore plotting errors

        if make_summary_json:
            summary = {
                "manual_skin_id": int(manual_skin_id),
                "background_label": int(background_label),
                "dilate_envelope_voxels": int(dilate_envelope_voxels),
                "stats": {
                    "manual_scalp_voxels": total_scalp_voxels,
                    "removed_outside_voxels": removed_voxels,
                },
                "shapes": {
                    "output_shape": list(map(int, out_arr.shape)),
                },
                "paths": {
                    "output_path": output_path,
                    "envelope_path": save_envelope_path,
                    "outside_mask_path": save_outside_mask_path,
                },
            }
            with open(os.path.join(qc_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)

    debug = {
        "removed_outside_voxels": removed_voxels,
        "manual_scalp_voxels": total_scalp_voxels,
        "dilate_envelope_voxels": int(dilate_envelope_voxels),
        "background_label": int(background_label),
    }
    return out_img, debug