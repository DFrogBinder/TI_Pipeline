# Pipeline Walkthrough

Use this as a quick map from raw data to population summaries and where each step’s code/output lives.

## 1) Atlases (optional but preferred)
- **Script**: `atlas/make_atlas.sh` (Docker) or `atlas/run_atlasMaker.py` (parallel launcher).
- **Input**: `<DATA_ROOT>/<sub>/anat/<sub>_T1w.nii.gz`.
- **Output**: `<DATA_ROOT>/FastSurfer_out/<sub>/mri/aparc.DKTatlas+aseg.deep.nii.gz`.
- **When**: Run once per subject to enable FastSurfer-based ROIs.

## 2) Meshing & TI simulation
- **Scripts**: `simulation/TI_runner_multi-core.py` (Slurm/parallel) or `simulation/TI_runner_single-core.py` (sequential).
- **Inputs**: T1/T2 + manual seg (`<sub>_T1w_ras_1mm_T1andT2_masks.nii`), optional atlas from step 1.
- **Process**: CHARM mesh, merge manual seg with CHARM labels, remesh, set TDCS montages, run SimNIBS, compute TImax.
- **Outputs** (per subject):
  - Mesh outputs: `<root>/<sub>/anat/SimNIBS/Output/<sub>/TI.msh`
  - Volumes via `msh2nii`: `<root>/<sub>/anat/SimNIBS/Output/<sub>/Volume_Base/TI_Volumetric_Base*.nii*`
  - Brain-masked TI: `<root>/<sub>/anat/SimNIBS/ti_brain_only.nii.gz`
- **Launch examples**:
  - Single: `python simulation/TI_runner_multi-core.py --subject sub-CCXXXXX`
  - Batch (Slurm array): `sbatch my_jobArray.slurm`

## 3) Subject post-processing
- **Script**: `post_process.py` (configure `PostProcessConfig` at bottom or import) with helpers in `post_functions.py` / `ti_utils.py`.
- **Inputs**: `ti_brain_only.nii.gz`, T1, atlas (FastSurfer if present, else Harvard–Oxford).
- **Outputs** (in `<root>/<sub>/anat/post/`):
  - ROI masks/overlaps: `atlas_<ROI>_mask.nii.gz`, `<ROI>_overlap_topXXpct_mask.nii.gz`
  - TI masked volumes: `TI_in_<ROI>.nii.gz`, `TI_in_TopXX.nii.gz`
  - CSVs: `<ROI>_values.csv`, `TopXX_values.csv`, `<ROI>_TopXX_overlap_values.csv`
  - Region stats: `region_stats_fastsurfer.csv`
  - Metrics: `subject_metrics.json`
  - Overlays: `<ROI>_TI_overlay_topXX.png`, `<ROI>_TI_overlay_aboveHHH.png`

## 4) Population aggregation
- **Script**: `post_population.py`
- **Inputs**: All subjects’ `region_stats_fastsurfer.csv` + `subject_metrics.json`
- **Command**:
```bash
  python post/post_population.py \
    --root <root> \
    --peak-threshold 0.2 \
    --target-roi Hippocampus \
    --template-region-csv <MNI>/region_stats_fastsurfer.csv
  ```
- **Outputs** (in `<root>/population_analysis/`):
  - `all_region_values.csv`
  - `population_region_summary.csv` (IQR, CV, peak fraction > threshold)
  - `subject_robustness.csv` (target ROI peaks/overlaps)
  - `volume_intensity_correlation.csv`

## 5) Optional geometry exports
- **Script**: `create3dmesh.py`
- **Use**: Convert TI volumes + masks to VTK/PLY/STL surfaces for visualization.
