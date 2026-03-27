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
4) Build the ROI mask from the subject atlas (for left precentral / M1, use `ctx-lh-precentral`, label `1022`).
5) Compute (using `anat/SimNIBS/ti_brain_only.nii.gz` for TI):
   - Label difference fraction vs a reference repeat.
   - ROI label difference fraction.
   - Mean, median, and peak TI in the ROI.
   - Mean, median, and peak TI across the whole head.
   - Hotspot distance and high-field overlap vs the reference repeat.
6) Summarise repeatability across runs:
   - Mean, standard deviation, minimum, maximum, and coefficient of variation (CV) for the key metrics.
   - Parameter-consistency checks (T1/T2/segmentation/settings and simulation signatures).
7) Optionally compare within-subject repeatability against the cohort spread from `region_stats_fastsurfer.csv`.
8) Save summary CSV/JSON, repeatability tables, report Markdown, and difference frequency map.

## Metrics Definitions

- **diff_fraction**: Fraction of voxels whose tissue label differs from the reference repeat, over the full T1 grid.
- **diff_fraction_roi**: Same as `diff_fraction`, but restricted to the ROI mask from the atlas.
- **mean_roi / median_roi / peak_roi**: TI summary values within the ROI.
- **mean_head / median_head / peak_head**: TI summary values over the whole `ti_brain_only.nii.gz` volume.
- **high_field_dice_head**: Dice overlap of the top percentile high-field mask vs the reference repeat.
- **hotspot_distance_head_mm**: Distance between the repeat and reference whole-head hotspot locations.
- **Repeatability stats**: `repeatability_stats.csv/json` reports mean, SD, min, max, and CV across repeats for each metric.
- **Parameter consistency**: `parameter_consistency.json` confirms whether the input images, CHARM settings, and extracted simulation signatures stayed identical across repeats.
- **Plot scaling**: TI values in `ti_brain_only.nii.gz` are already in V/m. No extra scale factor is applied.
- **label_diff_frequency.nii.gz**: Per-voxel fraction (0–1) of repeats that differ from the reference label at that voxel.
- **label_diff_overlay.png**: Orthogonal slice overlay of label difference frequency on top of the T1 image.
- **roi_mask_on_t1.nii.gz**: ROI mask on the reference T1 grid used for analysis.
- **roi_outline_on_t1.png**: Orthogonal T1 slices through the ROI center with the analysed ROI outlined.
- **roi_outline_on_mean_ti.png**: The same ROI outline overlaid on the mean TI field and T1 anatomy for QC.
- **repeat_qc/repeat_###_roi_outline_on_ti.png**: Per-repeat TI + T1 QC overlays with the analysed ROI outlined, so you can confirm the correct ROI is being sampled for that dataset.
- **PNG titles**: Each plot includes a title describing the source file(s) used.
- **batch_summary.csv/json**: Written in batch mode, with one summary row per subject.
- **batch_failures.json**: Written in batch mode if any subject fails.

## Run Analysis

Minimal invocation on this machine:

```
python mesh_repeat_report.py --subject sub-CC721888
```

Batch mode for all repeatability studies in the same folder:

```bash
python mesh_repeat_report.py \
  --all-subjects \
  --rootdir /media/boyan/main/PhD/CamCan-SimNIBS_Repeatability/new_params
```

The script will auto-resolve, when possible:
- the repeatability root from known dataset locations
- the atlas from `~/sandbox/Jake_Data/atlases/<subject>.nii.gz`
- the ROI labels from common ROI names such as `ctx-lh-precentral` (`1022`), `Left-Hippocampus` (`17`), and `Right-Hippocampus` (`53`)
- the output directory as `repeats/_analysis/<subject>/`

In batch mode, the same ROI settings are reused for every subject and outputs go under:

```text
<rootdir>/_analysis/<subject>/
```

Useful batch options:
- `--roi-name ...` and optionally `--roi-labels ...` to apply the same ROI to every study
- `--output-dir /path/to/out` to choose a different batch output root
- `--max-subjects N` to test batch mode on only the first `N` discovered subjects

Explicit invocation:

```
python mesh_repeat_report.py \
  --subject sub-CC110056 \
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

Optional logging (JSONL file; stdout is human-readable with timestamps):
```
python mesh_repeat_report.py \
  --subject sub-CC110056 \
  --roi-name ctx-lh-precentral \
  --log-file /path/to/mesh_repeat_report.log
```

With cohort comparison:
```
python mesh_repeat_report.py \
  --subject sub-CC721888 \
  --rootdir /media/boyan/main/PhD/CamCan-SimNIBS_Repeatability/new_params \
  --roi-name ctx-lh-precentral \
  --compare-cohort-root /media/boyan/main/PhD/Left_Hippocampus_Data \
  --cohort-region-name ctx-lh-precentral \
  --cohort-region-label 1022 \
  --output-dir /tmp/mesh_repeat_report/sub-CC721888
```

Outputs are written to:
```
/mnt/parscratch/users/cop23bi/repeatability-ti-dataset/repeats/_analysis/sub-CC110056/
```

Key outputs now include:
- `summary.csv/json`: per-repeat metrics.
- `repeatability_stats.csv/json`: across-repeat summary statistics.
- `repeatability_report.md`: compact text report for thesis/paper use.
- `parameter_consistency.json`: verification that non-mesh inputs/settings stayed fixed.
- `cohort_comparison.json`: within-subject vs between-subject comparison when a cohort root is provided.
- `repeat_qc/`: per-repeat ROI-on-TI QC PNGs.
- `batch_summary.csv/json`: batch-mode rollup across subjects.

## Notes

- Use T1 as the reference grid to keep all repeats comparable.
- Label resampling uses nearest-neighbor; TI volumes use linear interpolation.
- If node counts differ across repeats, voxel comparison avoids topology mismatch.
- `--atlas`, `--atlas-dir`, and `--roi-labels` remain available as overrides when auto-resolution is not sufficient.
- `--atlas` is single-subject only. In batch mode, use `--atlas-dir` or rely on atlas auto-resolution.
- `--m1-labels` is still accepted for backward compatibility, but `--roi-labels` is the preferred argument.
