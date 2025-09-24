#!/home/boyan/anaconda3/envs/simnibs_post/bin/python3
import os, numpy as np, nibabel as nib
from nilearn import datasets, image as nli
from nibabel.processing import resample_from_to

output_root = '/home/boyan/sandbox/TI_Pipeline/SimNIBS/MNI152_TI_Output'

# --- Atlas (Harvard–Oxford, max-prob 25%) ---
ho = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
ho_img = nli.load_img(ho.maps)
labels = [str(l, 'utf-8').lower() if isinstance(l, bytes) else str(l).lower()
          for l in ho.labels]
m1_ids = [i for i, name in enumerate(labels) if "precentral gyrus" in name]
if not m1_ids:
    raise RuntimeError("Precentral Gyrus not found in Harvard–Oxford labels.")

# --- FEM volume ---
ti_img = nib.load(os.path.join(output_root, "ti_brain_only.nii.gz"))
ti_data = ti_img.get_fdata(dtype=np.float32)

# If vector field, convert to magnitude
if ti_data.ndim == 4 and ti_data.shape[-1] in (3,):
    ti_data = np.linalg.norm(ti_data, axis=-1).astype(np.float32)

# --- Resample atlas to FEM grid (labels -> nearest) ---
ho_res = resample_from_to(ho_img, ti_img, order=0)
ho_data = ho_res.get_fdata().astype(np.int32, copy=False)

m1_mask = np.isin(ho_data, m1_ids)

# --- Threshold E-field (brain-only recommended) ---
finite = np.isfinite(ti_data)
if not np.any(finite):
    raise RuntimeError("No finite values in ti_data; check your masking.")
EFIELD_PERCENTILE = 95
thr = np.nanpercentile(ti_data[finite], EFIELD_PERCENTILE)
stim_mask = finite & (ti_data >= thr)

overlap = stim_mask & m1_mask

# --- Quick numbers & optional saves ---
def vol_mm3(img): 
    z = img.header.get_zooms()[:3]; return float(z[0]*z[1]*z[2])

vox_vol = vol_mm3(ti_img)
print(f"Threshold @ {EFIELD_PERCENTILE}th pct = {thr:.4g}")
print(f"M1 voxels: {m1_mask.sum():,} ({m1_mask.sum()*vox_vol/1e3:.1f} mL)")
print(f"High-E voxels: {stim_mask.sum():,} ({stim_mask.sum()*vox_vol/1e3:.1f} mL)")
print(f"Overlap voxels: {overlap.sum():,} ({overlap.sum()*vox_vol/1e3:.1f} mL)")
print(f"Overlap within high-E: {overlap.sum()/max(1, stim_mask.sum()):.2%}")
print(f"Overlap within M1:     {overlap.sum()/max(1, m1_mask.sum()):.2%}")

# Save masks for external inspection if you want
nib.save(nib.Nifti1Image(m1_mask.astype(np.uint8), ti_img.affine), os.path.join(output_root, "atlas_M1_mask.nii.gz"))
nib.save(nib.Nifti1Image(stim_mask.astype(np.uint8), ti_img.affine), os.path.join(output_root, "efield_topP_mask.nii.gz"))
nib.save(nib.Nifti1Image(overlap.astype(np.uint8),  ti_img.affine), os.path.join(output_root, "M1_overlap_mask.nii.gz"))
