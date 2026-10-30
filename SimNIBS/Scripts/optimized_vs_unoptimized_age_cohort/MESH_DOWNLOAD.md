# Download one verified mesh per selected subject

`download_hpc_meshes.py` retrieves exactly one corrected-v4 head mesh for each
of the 14 participants in `optimized_vs_unoptimized_age_balanced_14`.

## Fixed default source

- SSH destination: `cop23bi@stanage.shef.ac.uk`
- Study root: `/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Four_ROI`
- Dataset: `Left_Hippocampus_Data_01`
- Requested scope: 14 subjects and 14 `.msh` files

The final-132 campaign generated independent meshes for every ROI and repeat.
The default selects Left Hippocampus repeat 01 deterministically; the head mesh
itself is not target-specific. The canonical scaffold is not used because its
temporary bootstrap mesh was deliberately removed after producing the EEG-cap
coordinates.

## Run locally

To audit and download the complete set with a single Stanage two-factor
approval, run:

```bash
python3 optimized_vs_unoptimized_age_cohort/download_hpc_meshes.py --apply
```

The script first opens one temporary SSH ControlMaster connection. After that
single approval, the remote audit and every `rsync` transfer reuse the same
authenticated connection. Child connections run in batch mode, so if the
shared session is lost the script fails rather than asking for two-factor
authentication again. The control connection is closed when the script exits
normally and otherwise expires after four hours.

The audit checks all 14 remote `.msh` files against their completed mesh-result
JSON records, including subject, ROI, repeat, byte count, independent-remesh
flag, physical-scaffold flag, and SHA-256. Files are downloaded only after all
14 records pass.

An audit-only run is still available, but it is optional and constitutes a
separate SSH login:

```bash
python3 optimized_vs_unoptimized_age_cohort/download_hpc_meshes.py
```

The default local destination is:

```text
optimized_vs_unoptimized_age_cohort/downloads/optimized_vs_unoptimized_age_balanced_14/
```

The meshes are placed flat under `meshes/` as `<subject>.msh`. The downloader
then creates `download_manifest.tsv`, `download_receipt.json`, `subjects.txt`,
and `SHA256SUMS`. An interrupted transfer is resumable. A matching local mesh is
reused; a differing local mesh is never overwritten unless
`--replace-mismatch` is explicitly supplied, in which case it is retained as a
timestamped backup.

If Stanage is accessed through an SSH alias or gateway, specify that configured
destination explicitly:

```bash
python3 optimized_vs_unoptimized_age_cohort/download_hpc_meshes.py \
  --ssh-target stanage \
  --apply
```

To use a different completed ROI/repeat mesh set, pass `--roi` and `--repeat`.
The script prints the source delta and records it in the receipt.
