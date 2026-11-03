# Final-132 combined repeatability analysis

This workflow creates the first source-bound 9+1 analysis across SimNIBS
3.2.6 and SimNIBS 4, left hippocampus and right M1, and remesh and
spherical-median fixed-mesh conditions. It analyses 10 participants, 80
participant cells, and 3,200 already completed simulation records. It submits
no meshing or FEM work.

The nine retained balanced-cohort participants are combined with
`sub-CC510415`; `sub-CC520083` is excluded. Each participant cell contains 40
technical repeats. Participant is the inferential unit.

## Safety boundary

- Completed simulation campaigns are read-only.
- The active combined-dataset copy is neither an input nor a destination.
- All outputs go to the code-owned root:
  `/mnt/parscratch/users/cop23bi/final_132_repeatability_combined_analysis_v1`.
- `scope`, `preflight`, `status`, and `rebind-preflight` are read-only.
- `run` prepares the exact owned root and submits exactly one Sheffield job.
- This workflow never deletes source data.

The run uses 4 CPU, 16 GB, and a 2-hour limit. The analysis Python must import
NumPy, pandas, SciPy, and Matplotlib. User site-packages are disabled in the
job, and Matplotlib writes only below the analysis workflow root.

## Inputs

Preflight verifies ten source metric tables and ten terminal receipts. It
assembles four canonical 800-row schema-v2 optimizer-ROI tables: v3 and v4 for
left hippocampus and right M1. SimNIBS 4 retained-participant remesh rows come
from the established corrected-label campaigns, fixed rows come from the
spherical-median fixed campaigns, and the replacement contributes both
conditions. The v3 tables combine the completed base-nine and replacement
arms.

If either replacement SimNIBS 4 schema-v2 metric table is absent, preflight
stops. It reports that a separate read-only refresh of 160 existing NIfTI
metrics is required; this workflow does not silently perform extraction.

## Inspect and run

From `SimNIBS/Scripts` on the login node:

```bash
ANALYSIS=repeatability_combined_analysis/hpc/submit.sh

bash "$ANALYSIS" scope
bash "$ANALYSIS" preflight
bash "$ANALYSIS" run
```

`run` performs the same preflight, prepares the exact code-owned root, records
the input-plan hash and Git commit, then submits one job. It does not submit by
default when the launcher is called without `run`.

## Progress and validation

```bash
bash "$ANALYSIS" status

ROOT=/mnt/parscratch/users/cop23bi/final_132_repeatability_combined_analysis_v1
cat "$ROOT/_workflow/submission.json"
cat "$ROOT/analysis_manifest.json"
(cd "$ROOT" && sha256sum -c SHA256SUMS)
(cd "$ROOT" && sha256sum -c "${ROOT##*/}.tar.gz.sha256")
```

`status` combines Slurm accounting, when a submission receipt exists, with an
independent package validation. A valid complete package contains the four
canonical inputs, established within-version and cross-version outputs, the
condition-inclusive participant analysis, figures, provenance, `SHA256SUMS`,
and a portable tar archive.

## Download

Run this from the local computer after `status` reports `complete`:

```bash
scp cop23bi@stanage.shef.ac.uk:/mnt/parscratch/users/cop23bi/final_132_repeatability_combined_analysis_v1/final_132_repeatability_combined_analysis_v1.tar.gz .
scp cop23bi@stanage.shef.ac.uk:/mnt/parscratch/users/cop23bi/final_132_repeatability_combined_analysis_v1/final_132_repeatability_combined_analysis_v1.tar.gz.sha256 .
sha256sum -c final_132_repeatability_combined_analysis_v1.tar.gz.sha256
```

## Rebind after dataset completion

After the separate physical combined dataset has a valid
`_consolidation/complete.json`, run the read-only binding audit:

```bash
DATASET_ROOT=/mnt/parscratch/users/cop23bi/final_132_repeatability_combined_dataset_v1
bash "$ANALYSIS" rebind-preflight "--dataset-root=$DATASET_ROOT"
```

The audit requires exact entries in `provenance/source_mapping.tsv` and proves
that every mapped destination has the source hash snapshotted by this analysis.
It writes nothing to either root. Source retirement remains a separate guarded
workflow after both the dataset and this analysis have been independently
validated.
