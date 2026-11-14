# CHARM Journal Spatial Figures Design

## Purpose

Create a versioned, journal-ready visualization package for the three-dimensional
voxel-label differences observed across the ten completed SimNIBS 4.0.1 CHARM
segmentations of `sub-CC110174`.

The package is a read-only derivative of the completed segmentation-determinism
experiment. It must not rerun segmentation, surfaces, meshing, or FEM; alter the
twenty completed repeats; or replace the validated `visualizations_v1` package.

## Output boundary

The exporter writes only to:

`/mnt/parscratch/users/cop23bi/segmentation_determinism_staging_v1/journal_spatial_figures_v1`

An existing valid package is reused. An existing foreign, partial, stale, or
checksum-invalid destination is rejected. New output is built in a temporary
sibling directory and atomically published.

## Inputs and provenance

The exporter requires:

- the valid experiment terminal receipt and sidecar;
- all ten valid CHARM repeat receipts and retained
  `tissue_labeling_upsampled.nii.gz` maps;
- the complete determinism analysis manifest and tables; and
- the staged, immutable T1-weighted input recorded by the CHARM repeat receipts.

The output manifest records SHA-256 hashes for the terminal receipt, analysis
manifest, staged T1, and every generated artifact. Source outputs are read-only.

## Spatial data model

All ten CHARM label maps must have identical integer dtype, shape, affine, and
three-dimensional geometry. For each voxel, the exporter computes:

- the modal tissue label across ten runs; and
- an integer disagreement count from 0 to 6, defined as the number of runs not
  matching that modal label.

Ties retain the smallest numeric label, matching the existing deterministic
mode calculation. The arrays and affine are reoriented without interpolation to
closest-canonical RAS orientation. The staged T1 is linearly resampled to that
grid for display only and robustly normalized using finite positive intensities.

The display crop is the non-background consensus-label bounding box with a
fixed physical margin. Slice selection is deterministic:

- each main orthogonal slice maximizes the summed disagreement count for its
  plane; and
- atlas slices are selected at disagreement-weighted quantiles, with a
  deterministic evenly spaced fallback for zero-disagreement data.

## Figures

### Figure 1: global spatial overview and focal slices

A two-row, three-column composite:

- top row: sagittal, coronal, and axial maximum-intensity projections of the
  disagreement-count volume through the cropped head;
- bottom row: the most informative sagittal, coronal, and axial slices on a
  grayscale T1 background with thin consensus-label boundaries; and
- one discrete colour bar labelled `Runs not matching the modal tissue label
  (out of 10)` with integer ticks 1 through 6.

Panels use anatomical direction labels, physical RAS coordinates in millimetres,
plain-language titles, a white background, and no raw voxel-index titles.

### Figure 2: whole-head multi-slice atlas

A three-row, four-column atlas with four sagittal, four coronal, and four axial
slices selected from disagreement-weighted locations. It uses the same T1
background, boundary overlay, discrete scale, orientation convention, and
physical-coordinate labelling as Figure 1.

## Formats and metadata

Both figures are exported as:

- 600-dpi PNG;
- PDF; and
- SVG.

The package also contains:

- `figure_captions.md` with manuscript-ready captions and interpretation limits;
- `spatial_summary.json` containing the disagreement-count distribution,
  changed volume, voxel volume, crop, and selected slice coordinates;
- `figure_manifest.json`; and
- `SHA256SUMS`.

The captions must state that disagreement frequency is not segmentation error
probability and that the evidence is one subject under one SimNIBS 4.0.1/HPC
configuration.

## HPC execution

The existing working rendering profile is preserved:

- module: `SimNIBS/4.0.1-foss-2023a`;
- partition: `sheffield`;
- 4 CPU;
- 16G memory;
- 02:00:00 wall time; and
- one rendering job.

The only operational delta is the new explicit CLI/launcher mode and new output
directory. It reads 10 completed CHARM maps and one staged T1, creates two
figures in three formats plus metadata, and submits no segmentation, meshing, or
FEM tasks.
