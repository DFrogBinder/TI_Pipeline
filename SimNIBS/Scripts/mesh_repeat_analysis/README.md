# Mesh Repeatability Analysis

This folder contains the Slurm runner and analysis tooling for the SimNIBS
repeatability experiment. Each repeat builds a fresh mesh and runs TI, and the
analysis compares those outputs on a common T1 grid.

## Key Files

- `my_jobArray.slurm`: Slurm array job for per-repeat runs.
- `TI_runner_multi-core_repeat.py`: Runs the full pipeline per repeat, writing
  outputs under `.../repeats/repeat_###/sub-*/anat`.
- `mesh_repeat_report.py`: Repeatability analysis and reporting.
- `mesh_repeat_flowchart.drawio`: Flowchart of the pipeline and analysis.

## Output Layout (Per Repeat)

Each repeat writes to:

```
/mnt/parscratch/users/cop23bi/repeatability-ti-dataset/
  repeats/repeat_###/sub-CC110056/anat/
    sub-CC110056_T1w.nii  (symlink)
    sub-CC110056_T2w.nii  (symlink)
    sub-CC110056_T1w_ras_1mm_T1andT2_masks.nii (symlink)
    m2m_sub-CC110056/...
    SimNIBS/Output/sub-CC110056/...
```

## Analysis Pipeline (mesh_repeat_report.py)

1) Discover repeat folders under `rootdir/repeats/repeat_###/sub-*/anat`.
2) Load or generate `TI_Volumetric_*` label/base volumes from `TI.msh`.
3) Resample all volumes to the subject T1 grid.
4) Build M1 mask from the FreeSurfer atlas (`ctx-lh-precentral`, label 1024).
5) Compute:
   - Label difference fraction vs a reference repeat.
   - M1 label difference fraction.
   - Peak and mean TI in M1.
6) Save summary CSV/JSON and difference frequency map.

## Metrics Definitions

- **diff_fraction**: Fraction of voxels whose tissue label differs from the reference repeat, over the full T1 grid.
- **diff_fraction_m1**: Same as `diff_fraction`, but restricted to the M1 mask from the FreeSurfer atlas.
- **peak_m1**: Maximum TI value within the M1 mask.
- **mean_m1**: Mean TI value within the M1 mask.
- **label_diff_frequency.nii.gz**: Per-voxel fraction (0–1) of repeats that differ from the reference label at that voxel.

## Run Analysis

```
python mesh_repeat_report.py \
  --subject sub-CC110056 \
  --atlas /home/boyan/sandbox/Jake_Data/atlases/sub-CC110056.nii.gz \
  --m1-labels 1024 \
  --rootdir /media/boyan/main/PhD/CamCan-SimNIBS_Repeatability/simulation-data
```

If your layout is:
```
/.../simulation-data/sub-CC110056_repeatability/
  repeats/
  sub-CC110056/
```
then use the `simulation-data` directory as `--rootdir` (this is now the
default), and the script will resolve repeats and T1 paths from there.

Outputs are written to:
```
/mnt/parscratch/users/cop23bi/repeatability-ti-dataset/repeats/_analysis/sub-CC110056/
```

## Notes

- Use T1 as the reference grid to keep all repeats comparable.
- Label resampling uses nearest-neighbor; TI volumes use linear interpolation.
- If node counts differ across repeats, voxel comparison avoids topology mismatch.
