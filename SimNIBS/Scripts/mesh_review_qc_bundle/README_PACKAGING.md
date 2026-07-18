# Packaging And Handoff

The `mesh_review_qc_bundle` directory is self-contained. It does not import code
from TI Pipeline or SimNIBS.

## Prepare A Supervisor Archive

1. Copy the desired render directory tree into `images/`. Preserve filenames and
   subdirectories.
2. Remove any previous review data from `review_state/`, except
   `ABOUT_THIS_FOLDER.txt`, when beginning a new independent review.
3. From the parent directory, create the archive:

```bash
zip -r Mesh_QC_Review_Tool.zip mesh_review_qc_bundle \
  -x '*/__pycache__/*' '*.pyc' '*/.DS_Store'
```

On macOS, Finder's **Compress "mesh_review_qc_bundle"** command is also valid.

Send the ZIP archive. The recipient should begin with
`README_FOR_SUPERVISOR.md`.

## Executable Option

The source bundle is the most portable option because it has no third-party
dependencies. A native macOS `.app` or PyInstaller binary must be built and
tested on macOS; binaries produced on Linux are not macOS executables. The
included `.command` launcher provides double-click startup without introducing
that platform-specific build step.

## Integrity Check

Before handoff, run the bundled self-test from inside this directory:

```bash
python3 self_test.py
```

The test uses temporary images and state and does not modify the supplied image
or review folders.
