# Final-132 Repeatability Balanced-10 Cohort

This package defines the new 10-subject repeatability cohort selected only from
the supervisor-approved `final_132` CamCAN cohort.

## Outcome

- Subjects: `10`
- Sex balance: `5 FEMALE / 5 MALE`
- Age balance: `2 subjects in each of 5 equal-width age bands`
- Approved-pool age range: `19.00-85.17` years
- Selected age range: `26.00-79.17` years
- Selected mean age: `52.31` years
- Selected median age: `52.58` years
- Maximum female/male within-band age gap: `1.59` years
- Overlap with the previous 10-subject repeatability set: `0/10`

| Band | Female | Female age | Male | Male age | Pair gap |
| --- | --- | ---: | --- | ---: | ---: |
| 19.00-32.23 | `sub-CC110174` | 26.00 | `sub-CC121144` | 26.00 | 0.00 |
| 32.23-45.47 | `sub-CC310407` | 39.00 | `sub-CC320616` | 39.08 | 0.08 |
| 45.47-58.70 | `sub-CC420071` | 52.58 | `sub-CC410432` | 52.58 | 0.00 |
| 58.70-71.94 | `sub-CC520083` | 65.08 | `sub-CC520127` | 66.00 | 0.92 |
| 71.94-85.17 | `sub-CC610631` | 77.58 | `sub-CC720941` | 79.17 | 1.59 |

## Selection Rule

1. Restrict eligibility to the `132` IDs in the authoritative
   `final_132/subjects.txt` file.
2. Verify every approved ID has age and sex in the supplied workbook.
3. Use the more precise decimal age in `standard_data.csv` for ranking; the Excel
   workbook stores age at whole-year precision. Sex agrees for all 132 subjects,
   and the maximum age-source difference is one year.
4. Divide the final-132 age range into five equal-width bands:
   - `19.00-32.23` (center `25.617`)
   - `32.23-45.47` (center `38.851`)
   - `45.47-58.70` (center `52.085`)
   - `58.70-71.94` (center `65.319`)
   - `71.94-85.17` (center `78.553`)
5. In each band, select one female and one male by:
   - smallest absolute distance to the band center
   - lower age
   - lexical subject ID

This is the same age/sex balancing principle used for the old 175-subject pool,
recomputed from the new approved cohort rather than carrying forward the old
subjects or old band boundaries.

## Cohort Delta

- Eligible subjects: `175 -> 132`
- Old eligible sex counts: computed in the prior selection package
- New eligible sex counts: `45 FEMALE / 87 MALE`
- Old selected subjects retained by the new deterministic rule: `0`

## Files

- `subjects.txt`: canonical 10-subject list for staging and runner configs
- `selection_manifest.csv`: selected demographics and ranking evidence
- `eligible_132_demographics.csv`: full approved pool with selected flags
- `selection_audit.xlsx`: formatted audit workbook with formulas and top-three alternates
- `cohort.json`: machine-readable provenance and validation record
- `paired_repeatability_experiment.candidate.json`: candidate 40-remesh + 40-fixed config
- `build_balanced_repeatability_cohort.py`: exact reproducible cohort builder
- `stage_repeatability_dataset.py`: safe audit/staging utility for the 30 required NIfTI inputs
- `SHA256SUMS`: package checksums

## Imaging Dataset Staging

The supplied demographics directory does not contain the T1, T2, and corrected
segmentation NIfTI payload required by the repeatability runner. On HPC, use the
included staging utility against the canonical corrected-v4 scaffold tree. This
mode requires a completed per-subject provenance record, verifies all three
recorded source hashes, and decompresses `.nii.gz` inputs to the exact uncompressed
filenames required by the repeatability runner. It never modifies the scaffold.

Audit first:

```bash
python stage_repeatability_dataset.py \
  --source-layout final132-scaffold \
  --source-root /mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Scaffolds \
  --output-root /mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_corrected \
  --subjects-file subjects.txt
```

Create the physical 10-subject dataset only after the audit reports
`subjects_ready=10` and `required_files_ready=30`:

```bash
python stage_repeatability_dataset.py \
  --source-layout final132-scaffold \
  --source-root /mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Scaffolds \
  --output-root /mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_corrected \
  --subjects-file subjects.txt \
  --apply
```

The candidate config assumes the staged directory will be uploaded to:

`/mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_corrected`

That HPC path is a candidate, not a known-good live path. Confirm it after upload
before launching the runner.

## Requested Full Execution Scope

- Dataset: final-132 balanced repeatability cohort
- ROI: left hippocampus
- Subjects: `10`
- Conditions: `2` (`remesh`, `fixed_mesh`)
- Repeats per condition: `40`
- Total task count: `10 x 2 x 40 = 800`
- Expected repeat outputs: `800`
- Scope: full requested 10-subject repeatability study

No Slurm job is submitted by this package.

## Automated Two-Condition Execution

After the dataset is staged and the updated repository is available on HPC,
initialize the current-repair pipeline:

```bash
export CURRENT_REPAIR_DIR=/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/ti_current_repair
export EXPERIMENT_ROOT=/mnt/parscratch/users/cop23bi/current-repair/final_132_balanced_10

python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" init \
  --source-root /mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_corrected \
  --experiment-root "$EXPERIMENT_ROOT" \
  --subjects sub-CC110174,sub-CC121144,sub-CC310407,sub-CC320616,sub-CC420071,sub-CC410432,sub-CC520083,sub-CC520127,sub-CC610631,sub-CC720941 \
  --repeat-count 40 \
  --atlas-dir /mnt/parscratch/users/cop23bi/ZIPs/atlases \
  --roi-preset left-hippocampus
```

Run the read-only full-scope submission preflight:

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" submit-all \
  --experiment-root "$EXPERIMENT_ROOT" \
  --max-concurrent 50 \
  --analysis-max-concurrent 10 \
  --dry-run
```

It must report 400 remesh tasks (`0-399%50`), 400 fixed-mesh tasks
(`0-399%50`), and 800 expected `TI.msh` outputs. The production command is the
same without `--dry-run`.

The command initially submits only the remesh array and an `afterok` controller.
The controller validates all remesh outputs, analyzes them, selects each
subject's median representative mesh, physically seeds and validates the fixed
workspaces, then releases the fixed array, paired analysis, and final figures.
Any failed job or validation gate stops the chain before downstream submission.

## Provenance

- Approved-list SHA-256: `522e5085f4eee25cd4a2c01ad9ca885b5a214333c8da38dec9f918c965d4f607`
- Workbook SHA-256: `3db7da48f56ad96e45d43e86fca15a5258a57bae23415b73778f9faeb4b02d8a`
- Precise-demographics SHA-256: `2c72303f82617349bd64c1a44ade6a62cc61e8a593215bf0cc5c63802af4b510`
- Old-config SHA-256: `3e2af42f1d94ff3fcbd8357866f42a72e38fed50ea85e04c00f19414d447d2ed`
- Generated: `2026-07-27T01:02:52+03:00`

Source paths are also recorded in `cohort.json` and the workbook `Sources` sheet.
