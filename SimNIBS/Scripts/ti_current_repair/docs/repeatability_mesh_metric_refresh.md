# Repeatability mesh-metric refresh

This analysis-only workflow extracts tetrahedral element counts and physical
tetrahedral volumes by SimNIBS tissue tag from the retained repeatability
`TI.msh` files. It does not remesh, rerun FEM, or modify the existing
simulation and analysis outputs.

## Fixed scope

- Left hippocampus: 10 subjects × 2 conditions × 40 repeats = 800 meshes.
- Right M1: 10 subjects × 2 conditions × 40 repeats = 800 meshes.
- Full total: 1,600 retained meshes, 20 subject extraction tasks, and two
  dependent collectors.
- Arrays: two `0-9%10` arrays.
- Expected combined metric rows: 1,600.
- Output policy: isolated `repeatability_mesh_metrics_v1` directories.

## Stanage preflight and submission

```bash
cd ~/Repos/TI_Pipeline/SimNIBS/Scripts
git pull

bash ti_current_repair/hpc_scripts/submit_repeatability_mesh_metric_refresh.sh all --preflight
bash ti_current_repair/hpc_scripts/submit_repeatability_mesh_metric_refresh.sh all
```

The submitter prints the full scope before either preflight or submission. It
uses explicit Slurm environment exports and preserves the established
SimNIBS 4.0.1, Sheffield partition, 8-CPU, 32-GB, and 8-hour resource profile.

## Expected archives

```text
/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10/_post_processing/repeatability_mesh_metrics_v1/left_hippocampus_repeatability_mesh_metrics_v1.tar.gz
/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_right_m1/_post_processing/repeatability_mesh_metrics_v1/right_m1_repeatability_mesh_metrics_v1.tar.gz
```

Each archive has a neighbouring `.sha256` file. The collector fails rather
than packaging partial data unless it validates all 800 expected rows for its
ROI, all subject/condition/repeat keys, positive tetrahedral counts, and
tissue-count sums equal to each total element count.

## Rendering after download

After extracting each archive locally, pass its `mesh_metrics.csv` to the
repeatability renderer:

```bash
python ti_current_repair/post/make_presentation_figures.py \
    --experiment-root /path/to/downloaded/repeatability_analysis \
    --mesh-metrics-csv /path/to/extracted/mesh_metrics.csv \
    --output-dir /path/to/final_figures
```

This produces the existing median target-field figure plus the identically
styled tetrahedral-element figure. Tissue-specific element and volume
diagnostic figures are also generated when the extracted mappings are
available.
