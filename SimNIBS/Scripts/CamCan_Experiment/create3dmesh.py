import nibabel as nib
import os
import numpy as np
from skimage.measure import marching_cubes
from scipy.ndimage import map_coordinates, binary_closing, binary_fill_holes
import meshio


def load_canonical(path):
    img = nib.load(path)
    return nib.as_closest_canonical(img)

def make_surface_from_two_label_masks(
    ti_path: str,
    lbl1_path: str,
    lbl2_path: str,
    out_vtk_path: str,
    labels_to_include: set[int] | None = None,  # e.g., {1002, 1003} for GM/WM if you have those labels
    iso_level: float = 0.5
):
    # 1) Load as images (keep affines/orientation)
    ti_img   = load_canonical(ti_path)
    lbl1_img = load_canonical(lbl1_path)
    lbl2_img = load_canonical(lbl2_path)

    # 2) Sanity check: shapes + affines should match (they should, since both are from msh2nii)
    if ti_img.shape != lbl1_img.shape or ti_img.shape != lbl2_img.shape \
       or not np.allclose(ti_img.affine, lbl1_img.affine) \
       or not np.allclose(ti_img.affine, lbl2_img.affine):
        raise ValueError("TI and label masks are not on the same grid/affine. (They should be if all came from msh2nii.)")

    # 3) Build merged **binary** mask from label maps
    lbl1 = np.asarray(lbl1_img.dataobj)
    lbl2 = np.asarray(lbl2_img.dataobj)

    if labels_to_include:
        m1 = np.isin(lbl1, list(labels_to_include))
        m2 = np.isin(lbl2, list(labels_to_include))
    else:
        # default: anything > 0 is included
        m1 = lbl1 > 0
        m2 = lbl2 > 0

    merged_mask = (m1 | m2).astype(np.uint8)

    # 4) Surface extraction on the mask (complete geometry)
    # marching_cubes expects array shaped (Z, Y, X)
    verts_vox, faces, normals, _ = marching_cubes(merged_mask, level=iso_level)

    # 5) World coordinates for the surface vertices (use TI affine)
    aff = ti_img.affine
    ones = np.ones((verts_vox.shape[0], 1), dtype=np.float64)
    ijk = np.column_stack([verts_vox[:, 2], verts_vox[:, 1], verts_vox[:, 0], ones])  # (x,y,z,1)
    verts_world = (aff @ ijk.T).T[:, :3].astype(np.float32)

    # 6) Color the surface by the TI field (trilinear sampling at vertex voxel coords)
    TI = np.asarray(ti_img.dataobj)
    TI = np.nan_to_num(TI, nan=0.0, posinf=0.0, neginf=0.0)
    coords_zyx = np.column_stack([verts_vox[:, 0], verts_vox[:, 1], verts_vox[:, 2]]).T
    TI_vals = map_coordinates(TI, coords_zyx, order=1, mode="nearest").astype(np.float32)

    # 7) Save as VTK (triangles + per-vertex scalar)
    mesh = meshio.Mesh(
        points=verts_world,
        cells=[("triangle", faces.astype(np.int32))],
        point_data={"TImax": TI_vals}
    )
    meshio.write(out_vtk_path, mesh)
    print(f"Wrote: {out_vtk_path}")

subject    = 'sub-CC110062'  # change to your subject ID
rootDIR     = '/home/boyan/sandbox/Jake_Data/camcan_test_run'
output_root  = os.path.join(rootDIR,subject, 'anat','SimNIBS')

# --- Example usage ---
make_surface_from_two_label_masks(
    ti_path   = os.path.join(output_root,"ti_brain_only.nii.gz"),
    lbl1_path = os.path.join(output_root,'Output',subject,'Volume_Maks','TI_Volumetric_Masks_mask_2.nii.gz'),
    lbl2_path = os.path.join(output_root,'Output',subject,'Volume_Maks','TI_Volumetric_Masks_mask_1.nii.gz'),
    out_vtk_path = os.path.join(output_root,'BrainSurface_TImap.vtk'),
    labels_to_include=None  # or e.g. {1002, 1003} if your label scheme uses those for GM/WM
)
