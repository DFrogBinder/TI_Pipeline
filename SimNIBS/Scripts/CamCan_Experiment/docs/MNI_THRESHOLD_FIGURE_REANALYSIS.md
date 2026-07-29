# MNI-threshold manuscript figure reanalysis

This is an isolated, metric-only revision of the CamCan manuscript analysis.
It does not rerun FEM simulations and it does not replace the schema-3,
schema-4, v3, or v4 outputs.

## Scientific definition

ROI order is:

1. Left M1
2. Right DLPFC
3. Left hippocampus
4. Right thalamus

Each ROI is evaluated at its SimNIBS 4.0.1 MNI152 mean target field:

| ROI | Threshold (V/m) |
|---|---:|
| Left M1 | 0.18439651553583616 |
| Right DLPFC | 0.17202271761918309 |
| Left hippocampus | 0.20767908498193277 |
| Right thalamus | 0.19613854227394895 |

The versioned source is
`post/mni152_simnibs401_roi_thresholds.csv`. Coverage must be recalculated
from the source NIfTI fields at these exact thresholds. Interpolation between
the previously exported 0.20, 0.18, and 0.15 V/m summaries is not accepted.

## HPC submission

From `~/Repos/TI_Pipeline/SimNIBS/Scripts`:

```bash
bash CamCan_Experiment/cohort_pipeline/submit_mni_threshold_manuscript_reanalysis.sh all --preflight
bash CamCan_Experiment/cohort_pipeline/submit_mni_threshold_manuscript_reanalysis.sh all
```

The two analyses may also be submitted independently by replacing `all` with
`cohort` or `personalized`.

The isolated HPC output directories are:

- cohort:
  `.../campaigns/final_132/post_processing/optimizer_matched_analysis_mni_thresholds_v1`
- personalized:
  `.../post_processing/optimizer_matched_personalized_vs_generic_mni_thresholds_v1`

The cohort collector reads the fixed SimNIBS 4.0.1 MNI152 results from:

`/mnt/parscratch/users/cop23bi/MNI152_SimNIBS401_validation`

## Final figure contract

The renderer creates eight figures in PNG (400 dpi) and vector PDF:

1. absolute MNI152 minimum, mean, and maximum (P99.9);
2. one three-panel generic-to-personalized figure per ROI;
3. a combined population mean-field and target/off-target-ratio violin figure;
4. the MNI-relative mean-field/off-target relationship;
5. the MNI-relative target/off-target-coverage relationship.

It does not create percentile-context, effectiveness/spread, repeat
distribution, minimum-relationship, or maximum-relationship figures.

All axes are linear. Personalized markers are condition means across ten
repeat-level measurements, joined by generic-to-personalized arrows without
repeat error bars.

The selectivity ratio is:

`target coverage / off-target coverage`

Positive target coverage divided by zero off-target coverage is represented as
censored positive infinity at the finite plotting boundary. Zero divided by
zero is undefined and is counted separately. Every rendered result includes
`SUPERVISOR_NOTE_RATIO_ZERO_DENOMINATORS.md`, which requests confirmation of
this convention. If the MNI152 off-target denominator itself is zero, an
MNI-centred ratio is undefined; the renderer fails closed and requests a
separate decision rather than introducing an arbitrary epsilon.
