# Required information and local-data inventory

## Analysis question

Identify whether simulated temporal-interference (TI) dosimetry differs by
recorded sex, age stratum, or their combination in the authoritative
`final_132` CamCAN cohort. The analysis concerns favorable simulated exposure,
not observed clinical benefit.

## Information required before analysis

1. An authoritative participant list for the 132-person variability cohort.
2. One participant identifier shared by the cohort, demographics, and
   simulation outputs.
3. Recorded sex and age for every participant, with the meaning and coding of
   the demographic fields documented.
4. A prespecified age-stratification rule.
5. The complete repeat structure: participant, target, and independent mesh
   repeat identifiers.
6. A target-specific TI endpoint and supporting focality/selectivity endpoints.
7. Metric definitions, units, and the threshold convention.
8. Completion/QC status for every expected simulation-derived record.
9. The intended inferential unit and a strategy for technical repeats.
10. A cautious interpretation boundary: simulated electric-field delivery is
    not evidence of clinical response.

## Operational choices

- Demographic variable: the available field is recorded biological sex,
  encoded `FEMALE`/`MALE`; no gender-identity variable is present. Results are
  therefore labelled by recorded sex even though the request used “gender.”
- Age: precise decimal age in years.
- Age strata: the five equal-width, previously documented final-132 bands:
  `19.00-32.23`, `32.23-45.47`, `45.47-58.70`, `58.70-71.94`, and
  `71.94-85.17` years.
- Primary endpoint: mean TI field in the target ROI (`roi_mean`, V/m).
- Supporting endpoints: target ROI P95 (V/m), fraction of the target ROI at or
  above 0.2 V/m, target-to-neighbour mean-field ratio, and off-target
  whole-brain fraction at or above 0.2 V/m.
- Inferential unit: participant. The ten independent remeshing repeats are
  averaged within participant and target; repeat SD/CV are retained as
  computational-variability descriptors.
- Inference: heteroskedasticity-robust OLS on participant means, with
  Benjamini-Hochberg FDR correction across the four targets within each
  outcome/effect family. Stratified cells are also reported descriptively.

## Local inventory result

All information needed for the requested analysis is available locally.

| Requirement | Local source | Verified status |
| --- | --- | --- |
| Authoritative cohort | `CamCan_Experiment/cohort_pipeline/cohorts/final_132/subjects.txt` | 132 unique IDs |
| Cohort protocol | `CamCan_Experiment/cohort_pipeline/cohorts/final_132/cohort.json` | status `final` |
| Demographics | `output/spreadsheet/final_132_repeatability_balanced_10/eligible_132_demographics.csv` | 132/132 matched; no missing age or sex |
| Simulation/post-processing metrics | `/home/boyan/sandbox/Jake_Data/CamCan-PostData/Anatomical-Parcelation/final_132_post_processing_analysis_results/runs` | 5,280/5,280 metric files present |
| Targets | Same post-processing tree | 4 targets, 1,320 records each |
| Repeats | Same post-processing tree | 10 repeats, 528 records each |
| Completion/QC | `subject_metrics.json` metadata | all 5,280 marked complete |
| Metric definitions | `CamCan_Experiment/docs/METRIC_DICTIONARY.md` | definitions and units available |

The demographic composition is 45 female and 87 male participants. Cell sizes
by age band and recorded sex range from 5 to 24, so the combined-stratum plots
show uncertainty and the report avoids treating a visually highest cell as
proof of differential response.

## Computation decision

No HPC-side computation is required. The full participant-level post-processing
outputs and demographics are local, so the analysis is executed locally in
this separate directory. No paper directory or paper-generation workflow is
used or modified.
