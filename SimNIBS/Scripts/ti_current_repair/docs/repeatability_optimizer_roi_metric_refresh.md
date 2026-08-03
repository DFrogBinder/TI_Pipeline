# Repeatability optimizer-ROI metric refresh

This refresh corrects the target-region definition used by the repeatability
paper. The completed simulations are read only. No mesh generation or FEM
solve is submitted.

The analysis reconstructs the same parcel-clipped spherical targets used for
montage optimization:

- left hippocampus: label 17, 200 mm3
- right M1: label 12129, 100 mm3

Each sphere is centred on the subject parcel centroid and grown from 3 mm in
0.01 mm increments until it reaches the requested volume. Metrics are
calculated from all existing runs in both conditions. Outputs are written to
`_post_processing/repeatability_optimizer_roi_metrics_v1` under each completed
experiment.

## Stanage commands

```bash
cd ~/Repos/TI_Pipeline/SimNIBS/Scripts
bash ti_current_repair/hpc_scripts/submit_repeatability_optimizer_roi_metric_refresh.sh all --preflight
bash ti_current_repair/hpc_scripts/submit_repeatability_optimizer_roi_metric_refresh.sh all
```

The full scope is 1,600 existing TI NIfTIs, 20 subject extraction tasks, and
two dependent collectors. The collectors create one archive per target plus a
SHA-256 sidecar.
