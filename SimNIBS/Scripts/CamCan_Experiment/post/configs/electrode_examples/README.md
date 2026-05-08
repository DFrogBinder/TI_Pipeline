# Electrode distance input examples

These files show the two supported ways to provide electrode centers for
post-processing electrode-to-ROI distance metrics. Coordinates in all examples
are placeholder millimetre world coordinates; replace them with subject-specific
coordinates before analysis.

## Option 1: one study-level electrode CSV

Use `electrode_centers_by_subject_example.csv` as the model for
`PIPELINE_ELECTRODE_CSV` or `--electrode-csv`.

Required columns:

- `subject`: subject folder name exactly as it appears under the dataset root
- `electrode`: electrode label written into downstream outputs
- `x`, `y`, `z`: electrode center in millimetres in the same world coordinate
  frame as the subject TI image

The example contains the right-DLPC montage electrodes for two subjects:

```bash
PIPELINE_ELECTRODE_CSV=/path/to/electrode_centers.csv
PIPELINE_ELECTRODE_NAMES=
PIPELINE_EEG_POSITIONS_PATH_TEMPLATE=
```

## Option 2: electrode names plus per-subject EEG position files

Use `roi_electrode_sets.csv` to select the names for the current ROI/montage,
then point the pipeline at subject-specific `eeg_positions.csv` files. The
example tree mirrors the default lookup path:

```text
subjects/
└── sub-CC110056/
    └── anat/
        └── m2m_sub-CC110056/
            └── eeg_positions.csv
```

The default pipeline lookup is:

```text
<root>/<subject>/anat/m2m_<subject>/eeg_positions.csv
```

If the files live somewhere else, use a template:

```bash
PIPELINE_ELECTRODE_NAMES="AF4 F4 C2 CP1"
PIPELINE_EEG_POSITIONS_PATH_TEMPLATE="/path/to/subjects/{subject}/anat/m2m_{subject}/eeg_positions.csv"
PIPELINE_ELECTRODE_CSV=
```

The CSV format accepted by the parser is:

```csv
name,x,y,z
AF4,28.4,73.2,61.8
F4,44.9,65.1,58.3
```

Whitespace-delimited rows are also accepted:

```text
AF4 28.4 73.2 61.8
F4 44.9 65.1 58.3
```

## Automation logic

To generate `electrode_centers.csv` automatically for all subjects:

1. Choose the ROI/montage row in `roi_electrode_sets.csv`.
2. For each subject, read that subject's `eeg_positions.csv`.
3. Extract only the configured electrode names.
4. Write one output row per subject and electrode with columns
   `subject,electrode,x,y,z`.
5. Fail or flag the subject if any required electrode name is missing.

For `Right_DLPC_Runs`, use the `right_dlpc` row:

```text
AF4 F4 C2 CP1
```

The included example generator performs this extraction:

```bash
python post/configs/electrode_examples/build_electrode_centers_from_eeg_positions.py \
  --subjects-root post/configs/electrode_examples/subjects \
  --subjects sub-CC110056 sub-CC110087 \
  --electrode-names AF4 F4 C2 CP1 \
  --out /tmp/right_dlpc_electrode_centers.csv
```
