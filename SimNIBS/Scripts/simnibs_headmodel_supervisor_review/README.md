# SimNIBS head-model supervisor review package

This workflow creates a small, auditable supervisor-review package from one
randomly selected member of the retained nine-subject cohort. Selection occurs
only after all nine subjects pass the same receipt and checksum audit. The
selection is written once and then reused; it is not redrawn on reruns.

The source roots are read-only. The workflow writes only beneath:

`/mnt/parscratch/users/cop23bi/simnibs_headmodel_supervisor_review_package_v1`

## Scientific contents

The package contains exactly eight scientific artifacts from the same subject:

1. original CHARM tissue-label map;
2. corrected CHARM tissue-label map;
3. harmonised tissue-label map on the SimNIBS 3.2.6 T1fs grid;
4. original headreco tissue-label map derived from v3 mesh tissue tags;
5. subject-specific atlas;
6. SimNIBS 3.2.6 mesh made from the original headreco segmentation;
7. SimNIBS 3.2.6 mesh made from the new harmonised map; and
8. corrected-CHARM SimNIBS 4 median representative mesh from the completed
   left-hippocampus remesh campaign.

Item 4 is deliberately derived with SimNIBS 3.2.6 by assigning tetrahedral
tissue tags from item 6 onto `T1fs_conform.nii.gz`. It is not a raw CAT12 map.
The manifest records that derivation and the reference-image checksum. Items 6
and 7 preserve the original-headreco versus harmonised-v3 provenance boundary;
they must not be treated as interchangeable meshes.

Alongside the eight files, the package includes `README.md`, `manifest.json`,
and `checksums.sha256`. A sibling `.tar.gz` archive and `complete.json` receipt
bind the selection, manifest, checksum list, and archive. No package is accepted
merely because the Slurm job ended successfully.

## HPC contract

The build is one non-array job on `sheffield`, pinned to
`SimNIBS/3.2.6-foss-2023a`, 8 CPUs, 32 GB, and 08:00:00. Only the original
headreco tissue-label export requires the v3 runtime; all source discovery is
dependency-light and can run on the login node.

From `SimNIBS/Scripts`:

```bash
REVIEW=simnibs_headmodel_supervisor_review/hpc/submit.sh
OUTPUT_ROOT=/mnt/parscratch/users/cop23bi/simnibs_headmodel_supervisor_review_package_v1

bash "$REVIEW" preflight --output-root="$OUTPUT_ROOT"

bash "$REVIEW" submit \
    --output-root="$OUTPUT_ROOT" \
    --confirm-output-root="$OUTPUT_ROOT"
```

Scheduler submission is not completion. After the job leaves the queue, run:

```bash
bash "$REVIEW" status --output-root="$OUTPUT_ROOT"
bash "$REVIEW" validate --output-root="$OUTPUT_ROOT"
```

`validate` fails closed on a changed selection, missing or extra package file,
subject mismatch, checksum mismatch, invalid archive member set, or stale
completion receipt. The package is for methodological/supervisor review, not
clinical interpretation.
