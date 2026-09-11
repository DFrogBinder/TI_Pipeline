# Spherical-fixed and nested repeatability paper analysis

## Scientific units

The corrected fixed-mesh analysis contains 10 participants per target, with 40
technical repeats for each condition. Participant is the population-level unit.
The repeats characterize computational variation within a participant and must
not be treated as 400 independent participants per condition.

The nested analysis contains 1,600 measurements from one randomly selected
participant. Its design is 40 independently generated outer meshes by 40
solver/pipeline repeats conditional on each mesh. The 40 meshes are the outer
technical realizations. This analysis separates two computational variance
components for one participant and is not a population estimate.

## Recommended complete run

The complete analysis reads 1,600 corrected cohort rows and 1,600 nested rows,
creates eight figure files, and runs no meshing or FEM simulations. All source
experiment directories are read only. Outputs are written to the isolated
`final_132_repeatability_paper_update_v1` directory.

From the HPC repository root, run:

```bash
cd /users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts

/users/cop23bi/.conda/envs/ti-post/bin/python mesh_repeat_analysis/post/run_spherical_fixed_nested_paper_analysis.py
```

The completed driver prints a top-level `"status": "complete"` and creates the
download bundle and matching SHA-256 file at:

```text
/mnt/parscratch/users/cop23bi/final_132_repeatability_paper_update_v1/repeatability_paper_analysis_v1.tar.gz
/mnt/parscratch/users/cop23bi/final_132_repeatability_paper_update_v1/repeatability_paper_analysis_v1.tar.gz.sha256
```

The bundle contains the corrected input tables, complete statistical outputs,
four corrected fixed-mesh PNG figures, the main nested figure in PNG and SVG
formats, and a supplementary residual matrix in PNG and SVG formats.

## Corrected fixed-mesh inputs

The historical metric tables contain both remesh and obsolete fixed-mesh rows.
The correction tables contain the new fixed-mesh rows only. The adapter keeps
the historical remesh rows and replaces the obsolete fixed rows without
changing any source file.

Full analysis scope:

- left hippocampus: 10 participants × 2 conditions × 40 repeats = 800 rows
- right M1: 10 participants × 2 conditions × 40 repeats = 800 rows
- combined corrected paper inputs: 1,600 rows
- execution is analysis-only with no meshing or FEM simulation

From the HPC repository root, run:

```bash
cd /users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts

/users/cop23bi/.conda/envs/ti-post/bin/python mesh_repeat_analysis/post/prepare_spherical_fixed_paper_inputs.py \
  --left-historical-csv /mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10/_post_processing/repeatability_optimizer_roi_metrics_v1/optimizer_roi_metrics.csv \
  --left-corrected-fixed-csv /mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_spherical_fixed_v1/_post_processing/optimizer_roi_metrics_v1/optimizer_roi_metrics.csv \
  --right-historical-csv /mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_right_m1/_post_processing/repeatability_optimizer_roi_metrics_v1/optimizer_roi_metrics.csv \
  --right-corrected-fixed-csv /mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_right_m1_spherical_fixed_v1/_post_processing/optimizer_roi_metrics_v1/optimizer_roi_metrics.csv \
  --output-dir /mnt/parscratch/users/cop23bi/final_132_repeatability_paper_update_v1/corrected_fixed_inputs
```

Pass the two resulting 800-row CSV files to the established repeatability
figure generator. This preserves the established analysis and changes only its
fixed-mesh inputs.

## Nested figure

The paper-facing nested figure has two panels, with one panel assigned to each
level of the experimental hierarchy:

1. the 40 outer-mesh means, where each point summarizes the 40 repeats on that
   mesh
2. the distribution of all 1,600 within-mesh deviations, expressed in µV/m so
   the much smaller conditional solver/pipeline variation remains visible

The variance comparison is reported directly in the second panel rather than
through a separate logarithmic plot. The complete 40 × 40 residual matrix is
retained as a supplementary diagnostic figure.

Full analysis scope:

- participant: the persisted random selection `sub-CC320616`
- target: left hippocampus
- outer meshes: 40
- repeats per mesh: 40
- measurements: 1,600
- execution is analysis-only with no meshing or FEM simulation

Run:

```bash
/users/cop23bi/.conda/envs/ti-post/bin/python mesh_repeat_analysis/post/plot_nested_repeatability.py \
  --metrics-csv /mnt/parscratch/users/cop23bi/final_132_repeatability_nested_40x40_v1/_post_processing/optimizer_roi_metrics_v1/optimizer_roi_metrics.csv \
  --variance-json /mnt/parscratch/users/cop23bi/final_132_repeatability_nested_40x40_v1/_analysis/nested_variance/nested_variance_components.json \
  --output-dir /mnt/parscratch/users/cop23bi/final_132_repeatability_paper_update_v1/nested_figure
```

The renderer independently recomputes the balanced one-way random-effects
decomposition and refuses to produce a figure if it differs from the completed
nested analysis. PNG and SVG versions, plotted values, a caption draft, input
hashes, and a manifest are written into the isolated output directory.
