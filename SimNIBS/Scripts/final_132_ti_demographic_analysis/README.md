# Final-132 TI demographic analysis

This is a standalone analysis of simulated TI dosimetry in the 132-person
CamCAN variability cohort. It is separate from the paper and defacing tasks.

The analysis answers the request in three views:

1. recorded-sex stratification;
2. age stratification;
3. combined age-by-recorded-sex plots and models.

The primary endpoint is participant-level mean TI field in the target ROI,
after averaging the ten independent remeshing repeats. Supporting endpoints
describe target high-field exposure and focality/selectivity. The analysis does
not claim clinical benefit because the source data contain simulations rather
than treatment outcomes.

## Run

Use the locally compatible analysis environment:

```bash
export MPLCONFIGDIR=/tmp/final132-ti-demographic-mpl
/home/boyan/anaconda3/envs/simnibs_post/bin/python \
  final_132_ti_demographic_analysis/analyze_final132_demographics.py
```

The script validates the expected 132 participants, four targets, ten repeats,
and 5,280 complete metric files before producing any inferential results.

## Outputs

- `DATA_REQUIREMENTS_AND_INVENTORY.md`: required inputs, choices, and local
  availability decision.
- `outputs/ANALYSIS_REPORT.md`: plain-language findings and interpretation.
- `outputs/analysis_tables.xlsx`: formatted workbook of cohort summaries,
  stratified results, model results, and rankings.
- `outputs/run_level_metrics.csv`: one row per simulation-derived metric file.
- `outputs/subject_target_metrics.csv`: one row per participant and target;
  this is the inferential dataset.
- `outputs/sex_stratified_summary.csv`, `age_stratified_summary.csv`, and
  `sex_age_stratified_summary.csv`: separate and combined descriptive results.
- `outputs/model_results.csv`: robust regression estimates and FDR-adjusted
  p-values.
- `outputs/figures/`: PNG and PDF plots.

## Interpretation boundary

In this directory, “more favorable” means higher simulated target exposure
and/or selectivity with lower off-target coverage. It must not be rewritten as
evidence that a demographic group clinically benefits more from TI.
