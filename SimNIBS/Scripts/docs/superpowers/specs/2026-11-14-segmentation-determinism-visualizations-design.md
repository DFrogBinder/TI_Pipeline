# Segmentation Determinism Visualizations Design

## Intent

Create a quick, well-formatted scientific visualization package for the completed
single-subject segmentation-determinism experiment. The package must make the
central result immediately legible: SimNIBS 3.2.6 headreco was exactly repeatable
for the retained outputs, while SimNIBS 4.0.1 CHARM preserved image geometry but
changed tissue labels across repeats.

The figures are for scientific review and later dataset archiving. They are not a
population-level inference and must not claim that segmentation variability caused
the previously observed electric-field differences.

## Approved approach

The selected approach is a standalone, read-only figure exporter within
`simnibs_segmentation_determinism`. It consumes the completed manifest, TSV tables,
and ten retained CHARM label maps. It writes a separate, versioned package at
`<staging-root>/visualizations_v1` and never rewrites the experiment's existing
analysis files or terminal receipt.

The alternatives considered were a table-only exporter and an exploratory
notebook. The table-only option was rejected because it cannot show where labels
change. The notebook was rejected because it weakens repeatability and provenance.

## Outputs

The exporter creates PNG and PDF versions of six figures:

1. `01_determinism_summary`: plain-language comparison of operational, geometric,
   and voxel-label determinism for both implementations.
2. `02_pairwise_disagreement_heatmap`: ten-by-ten repeat matrix of changed-label
   percentages, with an exact-zero headreco reference and the CHARM pairwise matrix.
3. `03_changed_voxel_distribution`: all 45 CHARM pairwise changed-label percentages
   shown as individual observations with median and range.
4. `04_label_dice_distributions`: all pairwise Dice scores by numeric tissue label;
   labels remain numeric unless an authoritative LUT is available.
5. `05_tissue_volume_variability`: CHARM tissue-volume deviations from each label's
   across-repeat mean, with all ten repeats visible.
6. `06_spatial_disagreement_frequency`: orthogonal slices through the strongest
   disagreement region, showing the modal tissue label beneath the fraction of runs
   that disagree with that mode.

It also creates `summary_statistics.json`, `figure_captions.md`,
`figure_manifest.json`, and `SHA256SUMS`. The manifest binds every generated file to
the source analysis manifest and terminal-receipt hashes.

## Scientific and visual rules

- Comparisons remain within implementation; v3 and v4 tissue taxonomies are not
  treated as voxel-equivalent.
- CHARM pairwise values come from all 45 repeat pairs. A missing or duplicate pair
  is an error.
- The spatial map uses `1 - modal_label_count / 10` at each voxel. It is a
  repeat-disagreement frequency, not uncertainty or error probability.
- Geometry, shapes, affines, and repeat indices must validate before rendering.
- V3 exact repeatability is represented as zero disagreement, based on the validated
  manifest; the exporter does not fabricate a cross-version voxel comparison.
- Axes use reader-facing language such as `Changed tissue labels (%)` and
  `Dice similarity`, not internal pipeline terminology.
- Use a colorblind-safe blue/orange palette, readable typography, restrained grids,
  and captions that define every plotted unit.
- Every figure is written at publication-quality raster resolution and as vector
  PDF. No seaborn or pandas dependency is introduced.

## Command and lifecycle

Add a `figures` CLI command and matching `hpc/submit.sh figures` mode. It submits no
Slurm jobs and runs through `ANALYSIS_PYTHON`, which must provide NumPy, NiBabel and
Matplotlib.

The command fails closed unless the experiment status is complete and the terminal
receipt validates. A completed visualization package with the same source hashes is
reused after checksum validation. A foreign, partial, or stale output directory is
rejected rather than overwritten.

## Testing and non-goals

Tests use synthetic label volumes and completed-analysis fixtures to validate
statistics, spatial disagreement, all required files, source/hash binding,
idempotent reuse, CLI behavior and the no-`sbatch` launcher boundary.

This feature does not rerun segmentation, alter the completed analysis, install the
extension into the combined dataset, compare electric fields, or infer a causal
source for CHARM's variability.
