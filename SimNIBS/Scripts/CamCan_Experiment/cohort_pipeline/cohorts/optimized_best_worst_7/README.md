# Individualized best/worst seven-subject cohort

This cohort contains the seven unique CamCan subjects selected as the highest
or lowest `roi_median_v_per_m` case in at least one of the four manuscript
ROIs. `sub-CC420100` is the worst case for both the left hippocampus and right
thalamus, so eight best/worst selections collapse to seven unique subjects.

Every subject has four individualized optimization files. The executable
table in `individualized_targets.csv` uses `ParetoBest.TI_free.Emin`, the
Pareto solution that reaches approximately 0.2 V/m while minimizing stimulated
volume. `TI_free.Emax` is intentionally excluded because it maximizes field
amplitude and produces broad stimulation inconsistent with the declared
selection objective.

The CSV was audited row-for-row against all 28 supervisor MATLAB files on
2026-07-28. It records the optimized configuration ID, electrode pairs,
currents in mA, source filename, and source SHA-256.

Full execution scope:

- 7 subjects
- 4 ROIs
- 10 independent corrected-v4 remesh repeats
- 280 independent meshes
- 280 validated FEM simulations
- existing corrected-v4 scaffold reuse; no new segmentation expected

Run the HPC preflight from `SimNIBS/Scripts`:

```bash
bash CamCan_Experiment/cohort_pipeline/submit_optimized_best_worst_pipeline.sh --preflight
```

After confirming that the preflight reports 7 scaffold reuses, 280 ready mesh
tasks, 280 ready FEM tasks, and montage mode `subject_roi_individualized`,
submit the identical full scope:

```bash
bash CamCan_Experiment/cohort_pipeline/submit_optimized_best_worst_pipeline.sh
```
