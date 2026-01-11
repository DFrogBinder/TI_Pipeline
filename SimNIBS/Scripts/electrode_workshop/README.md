# Electrode Workshop (SimNIBS)

Helpers and scripts for placing custom plank electrodes and running
temporal interference (TI) simulations in SimNIBS 4.5.

## Contents
- `place_plank_electrode.py`: place a custom plank (and optional return) electrode
  aligned to scalp normals, with optional scalp extraction from a segmentation.
- `ti_custom_planks.py`: run a 4-plank TI setup, export TI volume + overlay PNGs,
  and generate preview geometry.
- `functions.py`: post-processing utilities (overlays, ROI masks, CSV export, seg merge).
- `electrode_plank.stl`, `gel_plank.stl`: example plank meshes.

## Dependencies
You need a SimNIBS 4.5 environment plus:
- numpy, nibabel, scipy, nilearn, scikit-image
- trimesh and rtree (fast closest-point queries)
- meshio (if reading non-STL surfaces)

## Typical usage

Place a single custom plank electrode:
```bash
simnibs_python place_plank_electrode.py \
  --m2m /path/to/m2m_subject \
  --out ./tdcs_custom_out \
  --gel-stl /path/to/gel_plank.stl \
  --el-stl /path/to/electrode_plank.stl \
  --center AF4 \
  --toward PO4
```

Run a 4-plank TI setup (AF4/PO4 and AF3/PO3 by default):
```bash
simnibs_python ti_custom_planks.py \
  --m2m /path/to/m2m_subject \
  --fnamehead /path/to/m2m_subject/subject.msh \
  --out ./TI_CustomPlanks \
  --gel-stl /path/to/gel_plank.stl \
  --el-stl /path/to/electrode_plank.stl \
  --current-ma 2.0 \
  --element-size-mm 0.5
```

## Notes
- If no scalp surface is found in `m2m`, pass `--seg` and `--skin-labels` to
  extract a scalp STL from a segmentation.
- `ti_custom_planks.py` can conform the pad STLs to the scalp surface
  via `--conform-to-scalp` (writes conformed STLs under `out/conformed/`).
