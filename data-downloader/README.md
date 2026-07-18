# CamCAN T1/T2 copier

`copy_camcan_t1_t2.py` copies only the two exact original image names from
each subject:

- `<subjectID>_T1w.nii`
- `<subjectID>_T2w.nii`

It ignores `.nii.gz` duplicates, JSON sidecars, masks, resampled images, and
all other files. The output keeps the original subject organization:
`<destination>/<subjectID>/anat/<image>`.

First, preview the complete operation without writing anything:

```bash
./copy_camcan_t1_t2.py \
  /mnt/xdrive/arvaneh_group/Shared/stimsim/mridata/CamCAN \
  /path/to/local/camcan-images \
  --dry-run
```

Then perform the copy:

```bash
./copy_camcan_t1_t2.py \
  /mnt/xdrive/arvaneh_group/Shared/stimsim/mridata/CamCAN \
  /path/to/local/camcan-images
```

During an interactive terminal run, a `tqdm` bar shows completed images,
copied/skipped/failed counts, elapsed time, estimated time remaining, and data
processed versus total data. Install the dependency if needed:

```bash
python3 -m pip install -r requirements.txt
```

Existing destination files with the expected size are skipped, so an
interrupted copy can be rerun. Existing files with the wrong size are reported
as failures and left unchanged; pass `--overwrite` to replace them. Copies are
written to temporary files and moved into place only after their sizes have
been checked.

Use `--workers 1` for sequential copying, or another value to adjust the
default of four simultaneous copies. Discovery uses 32 simultaneous metadata
checks by default because individual checks can be slow across a VPN; adjust
that with `--scan-workers`. Use `--summary-only` to suppress the per-image
lines. The progress bar is automatically hidden when output is redirected;
`--no-progress` disables it explicitly. `--require-both` makes the command fail
if any subject is missing either expected image.

## Remove the original cohort from a full CamCAN dataset

`remove_camcan_subjects.py` uses the direct `sub-CC######` directories in an
extracted reference cohort as the removal list. It removes only same-named,
direct child directories from the full dataset. It does not inspect or modify
segmentation output directories.

First run the read-only audit. Every valid subject ID found directly inside
Folder A must exist directly inside Folder B, but there is no required cohort
size:

```bash
./remove_camcan_subjects.py \
  /path/to/extracted-original-175 \
  /path/to/full-camcan-dataset
```

The audit prints the reference, full, matching, missing, and projected remaining
subject counts followed by every directory it would delete. If the exact size
of either input is known, add `--expected-subjects N` and/or
`--expected-full-subjects N` as optional guards.

After reviewing the complete audit, perform the permanent deletion:

```bash
./remove_camcan_subjects.py \
  /path/to/extracted-original-175 \
  /path/to/full-camcan-dataset \
  --apply
```

Apply mode requires a typed confirmation and writes a timestamped TSV event
manifest beside the full dataset before deleting anything. `--yes` is available
for an intentional non-interactive run. A rerun after an interrupted/partial
cleanup will normally stop because some reference IDs are absent; only then use
`--allow-missing-targets` to resume. Deletion is permanent, so keep the original
archive or another verified copy until the new segmentation run is complete.
