# Electrode Dataset

This directory contains electrode-center inputs for post-processing electrode
distance metrics.

Required columns in every electrode CSV:

- `subject`: subject folder name, for example `sub-CC110056`
- `electrode`: electrode label used in the simulation montage
- `x`, `y`, `z`: electrode center coordinates in millimetres in the same world
  coordinate frame used for distance calculation

The generated files also include audit columns:

- `roi_alias`
- `montage_preset`
- `coordinate_source`
- `source_file`

Directory layout:

```text
<roi_alias>/
  electrode_centers.csv
  <subject>/
    electrodes.csv
manifest.csv
verification_report.csv
verification_summary.json
```

For the current generated dataset, coordinates come from the SimNIBS MNI152
`EEG10-10_UI_Jurak_2007.csv` cap because the collected post-data tree does not
include per-subject `m2m_*` EEG cap files. If subject-space cap files are
restored, rerun the build script after updating its coordinate source logic.

The prepared ROI aliases mirror the targets listed in
`/home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/utils/targets.csv`:

```text
left-m1
left-dlpc
left-hippocampus
left-thalamus
left-pallidum
right-m1
right-dlpc
right-hippocampus
right-thalamus
right-pallidum
```

To use this dataset in post-processing:

```bash
PIPELINE_ELECTRODE_DATASET_DIR=/path/to/electrode_dataset
PIPELINE_ELECTRODE_CSV=
PIPELINE_ELECTRODE_NAMES=
```

Scripts:

- `scripts/build_electrode_dataset.py`: regenerates the dataset from the v8
  post-data subject list, ROI electrode sets, and SimNIBS cap coordinates.
- `scripts/verify_electrode_dataset.py`: validates schema, finite coordinates,
  expected electrode names, and subject-file coverage.
