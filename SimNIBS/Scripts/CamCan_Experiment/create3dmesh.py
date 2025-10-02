import os
import meshio
import numpy as np
import nibabel as nib
from typing import Iterable, Optional

def load_canonical(path: str) -> nib.Nifti1Image:
    img = nib.load(path)
    return nib.as_closest_canonical(img)

def make_hex_volume_from_nifti(
    ti_path: str,
    lbl1_path: str,
    lbl2_path: str,
    out_vtk_path: str,
    labels_to_include: Optional[Iterable[int]] = None,
    value_name: str = "TImax",
):
    """
    Build a HE(X)ahedral unstructured mesh from a TI volume, restricted to voxels
    where (lbl1 ∪ lbl2) is true (or matches labels_to_include if provided).
    Stores the TI as CELL DATA, one value per hexahedron.

    Inputs are assumed to come from msh2nii for the SAME subject (same grid+affine).
    """

    # --- 1) Load as images, keep affines/orientation
    ti_img   = load_canonical(ti_path)
    lbl1_img = load_canonical(lbl1_path)
    lbl2_img = load_canonical(lbl2_path)

    # Sanity checks: identical grids
    if ti_img.shape != lbl1_img.shape or ti_img.shape != lbl2_img.shape:
        raise ValueError(f"Shape mismatch: TI {ti_img.shape}, lbl1 {lbl1_img.shape}, lbl2 {lbl2_img.shape}")
    if not (np.allclose(ti_img.affine, lbl1_img.affine) and np.allclose(ti_img.affine, lbl2_img.affine)):
        raise ValueError("Affine mismatch between TI and label masks.")

    # --- 2) Arrays
    # Nib canonical storage is (X, Y, Z) in the header, but data arrays are read as (Z, Y, X).
    # nibabel returns array data as (Z, Y, X); keep that convention consistently.
    TI   = np.asarray(ti_img.dataobj)           # (Z, Y, X)
    L1   = np.asarray(lbl1_img.dataobj)         # (Z, Y, X)
    L2   = np.asarray(lbl2_img.dataobj)         # (Z, Y, X)
    TI   = np.nan_to_num(TI, nan=0.0)

    # --- 3) Build merged mask
    if labels_to_include is not None:
        labels_to_include = set(labels_to_include)
        M = np.isin(L1, list(labels_to_include)) | np.isin(L2, list(labels_to_include))
    else:
        M = (L1 > 0) | (L2 > 0)
    M = M.astype(bool)

    Z, Y, X = TI.shape
    if not M.any():
        raise ValueError("Merged mask is empty; nothing to mesh.")

    # We will create a compact set of grid points (voxel corners) and
    # hexahedral cells for voxels where M[z,y,x] is True.
    # Voxel corner indices are (i,j,k) with i in [0..X], j in [0..Y], k in [0..Z],
    # where voxel (z,y,x) spans corners:
    # (x, y, z) corners: (i,j,k) in {(x,y,z), (x+1,y,z), (x+1,y+1,z), (x,y+1,z),
    #                               (x,y,z+1), (x+1,y,z+1), (x+1,y+1,z+1), (x,y+1,z+1)}

    # --- 4) Collect all needed corner lattice points for selected voxels
    # Indices of selected voxels
    sel = np.argwhere(M)  # rows: [z, y, x]

    # For point compaction, map lattice corner (i,j,k) -> point id
    point_id = {}
    points_ijk = []

    # To build cells after we have point ids
    hex_cells = []

    def pid(i, j, k):
        key = (i, j, k)
        if key in point_id:
            return point_id[key]
        idx = len(points_ijk)
        point_id[key] = idx
        points_ijk.append(key)
        return idx

    # Build cells
    for z, y, x in sel:
        # corners in (i,j,k) = (x,y,z) lattice
        n000 = pid(x,   y,   z)
        n100 = pid(x+1, y,   z)
        n110 = pid(x+1, y+1, z)
        n010 = pid(x,   y+1, z)
        n001 = pid(x,   y,   z+1)
        n101 = pid(x+1, y,   z+1)
        n111 = pid(x+1, y+1, z+1)
        n011 = pid(x,   y+1, z+1)

        # VTK hexahedron node ordering (VTK_HEXAHEDRON=12) is:
        # (0,1,2,3,4,5,6,7) = (x0y0z0, x1y0z0, x1y1z0, x0y1z0, x0y0z1, x1y0z1, x1y1z1, x0y1z1)
        hex_cells.append([n000, n100, n110, n010, n001, n101, n111, n011])

    # --- 5) Convert lattice (i,j,k) to world coordinates via affine
    # NIfTI affine expects (x, y, z) in voxel index space. Our lattice is:
    # i -> x, j -> y, k -> z. So form (x,y,z,1) = (i, j, k, 1).
    pts = np.array(points_ijk, dtype=np.float64)  # (N, 3) in (i,j,k) = (x,y,z)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    ijk1 = np.hstack([pts[:, [0, 1, 2]], ones])  # (x, y, z, 1)
    xyz  = (ti_img.affine @ ijk1.T).T[:, :3].astype(np.float32)  # world mm

    # --- 6) Cell data: TI per voxel (use center value)
    # TI array is indexed as TI[z,y,x]. This matches our 'sel' list order.
    # We will store TI at the voxel as the CELL scalar.
    cell_ti = TI[sel[:, 0], sel[:, 1], sel[:, 2]].astype(np.float32)

    # --- 7) Write VTK UnstructuredGrid with HEX cells and cell_data
    cells = [("hexahedron", np.asarray(hex_cells, dtype=np.int32))]
    cell_data = {value_name: [cell_ti]}  # list aligned with 'cells' list

    mesh = meshio.Mesh(points=xyz, cells=cells, cell_data=cell_data)
    meshio.write(out_vtk_path, mesh)
    print(f"Wrote volumetric HEX mesh: {out_vtk_path}\n"
          f"  points: {xyz.shape[0]}, cells: {cells[0][1].shape[0]} (hexahedra)")

subject    = 'sub-CC110062'  # change to your subject ID
rootDIR     = '/home/boyan/sandbox/Jake_Data/camcan_test_run'
output_root  = os.path.join(rootDIR,subject, 'anat','SimNIBS')

# --- Example usage ---
make_hex_volume_from_nifti(
    ti_path   = os.path.join(output_root,"ti_brain_only.nii.gz"),
    lbl1_path = os.path.join(output_root,'Output',subject,'Volume_Maks','TI_Volumetric_Masks_mask_2.nii.gz'),
    lbl2_path = os.path.join(output_root,'Output',subject,'Volume_Maks','TI_Volumetric_Masks_mask_1.nii.gz'),
    out_vtk_path = os.path.join(output_root,'BrainSurface_TImap.vtk'),
    labels_to_include=None  # or e.g. {1002, 1003} if your label scheme uses those for GM/WM
)
