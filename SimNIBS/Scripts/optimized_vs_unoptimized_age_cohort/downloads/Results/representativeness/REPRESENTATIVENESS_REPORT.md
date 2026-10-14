# MNI-based field representativeness of the selected 14

## Bottom line

The 14 participants were selected using age and recorded sex, without using an
electric-field outcome. Their existing MNI152-derived generic-montage results
were then compared with both the corresponding MNI152 template value and the
field distribution in the final-132 cohort.

Across the four targets, the age/recorded-sex-adjusted standardized mean
differences for the primary target-ROI mean field ranged from -0.28 to 0.00.
The observed ranges of the selected participants covered 74.8 to 94.6% of
the adjusted residual distributions in the 111 eligible participants who were
neither selected nor part of the outcome-selected optimized seven. Across all
56 selected participant-target records, 1 had an absolute
adjusted residual above 2 reference standard deviations, and the maximum was
2.31. In the four-target multivariate field profile, selected-subject
distance percentiles ranged from 7.2 to 89.2% relative to the same
reference group.

For the supporting median-field and 0.20 V/m coverage outcomes, selected
residual ranges covered 71.2 to 98.2% of the reference
distributions. The largest supporting standardized mean difference was
+0.51 for
target coverage at 0.20 v/m in the
right dlpfc. This threshold-based
outcome was more heterogeneous than the continuous mean-field outcome and is
therefore retained explicitly in the supporting tables rather than being used
to redefine the primary cohort.

These diagnostics characterize how the selected participants sit within the
available generic-montage field variation. They do not establish formal
equivalence, and the final-132 cohort itself is a QC-selected modelling cohort
rather than a population-random sample.

## Primary target-field results

- Left hippocampus: selected MNI-relative mean-field range -4.6% to 34.9%, age/sex-adjusted SMD -0.10, and selected residual span covering 74.8% of the 111-person reference distribution.
- Left M1: selected MNI-relative mean-field range -35.6% to 25.0%, age/sex-adjusted SMD +0.00, and selected residual span covering 87.4% of the 111-person reference distribution.
- Right DLPFC: selected MNI-relative mean-field range -50.1% to 38.5%, age/sex-adjusted SMD -0.04, and selected residual span covering 94.6% of the 111-person reference distribution.
- Right thalamus: selected MNI-relative mean-field range -4.3% to 40.7%, age/sex-adjusted SMD -0.28, and selected residual span covering 80.2% of the 111-person reference distribution.

## Methods

- Source: complete schema-4 optimizer-matched analysis of 132 participants,
  four targets, and ten independently remeshed simulations per participant and
  target.
- Generic configuration: the MNI152-derived montage was simulated on every
  participant head. The reference value was the same montage simulated on the
  MNI152 head model.
- Primary quantity: percentage difference in the ten-repeat mean target-ROI
  field from the target-specific MNI152 value.
- Supporting quantities: target-ROI median field, target coverage at or above
  0.20 V/m, and off-target coverage at or above 0.20 V/m. Coverage differences
  are expressed in percentage points.
- ROI boundary: the same parcel-clipped spherical construction logic was used
  for participant and MNI152 analyses. The voxelized realizations are aligned
  definitions, not identical masks.
- Reference group: 111 final-132 participants after excluding the selected 14
  and the seven previously optimized outcome-selected participants.
- Adjustment: for each target and metric, the reference data were modelled
  using continuous age, recorded sex, and their interaction. Reference
  prediction errors were leave-one-out PRESS residuals. Selected participants
  were predicted from the model fitted only to the 111-person reference group.
- The standardized mean difference, residual range coverage, two-sample
  Kolmogorov-Smirnov statistic, within-selection-stratum percentile, and
  four-target Mahalanobis distance are descriptive diagnostics. No arbitrary
  pass/fail threshold was imposed.

## Figure captions

### Mean-field figure

**Generic-montage target fields in the selected 14 and final-132 cohort.** Each
point shows the ten-repeat mean target-ROI field for the MNI152-derived generic
montage on one participant head, expressed as a percentage difference from the
corresponding MNI152-head result. The selected age-balanced participants are
highlighted by recorded sex, and the seven previously optimized participants
are marked separately because they were selected using field outcomes. SMD is
the difference in mean age/recorded-sex-adjusted residual between the selected
14 and the 111-person reference group, divided by the reference residual
standard deviation. Selected span is the percentage of reference residuals
falling between the minimum and maximum selected residuals. The horizontal
dashed line denotes the MNI152 value.

### Target-coverage figure

**Target coverage at 0.20 V/m in the selected 14 and final-132 cohort.** Each
point shows the percentage of the optimizer-matched target ROI with a simulated
field at or above 0.20 V/m, averaged across ten remeshing repeats and expressed
as a percentage-point difference from the corresponding MNI152-head result.
The selected age-balanced participants are highlighted by recorded sex, and
the seven previously optimized participants are marked separately. SMD and
selected span use the same age/recorded-sex-adjusted 111-person reference as
the mean-field figure. The dashed line denotes the MNI152 coverage. The 0.20
V/m cutoff is an analysis convention, not a validated therapeutic threshold.

## Suggested concise email statement

Before proceeding, I assessed the 14 participants against the complete
final-132 generic-montage distribution using the optimizer-matched target ROIs.
Their mean target fields were close to the eligible reference distribution
after accounting for age and recorded sex, and the selected ranges covered
75-95% of its residual variation across the four targets (Figure attached).
Threshold-based coverage was more heterogeneous, particularly for right DLPFC,
so I would retain the outcome-independent cohort and include baseline field
position as a prespecified covariate or sensitivity analysis rather than
reselecting participants on the outcome.
