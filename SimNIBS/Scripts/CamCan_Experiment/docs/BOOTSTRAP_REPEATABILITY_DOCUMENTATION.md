# Bootstrap Repeatability Analysis for Choosing Repeats per Subject

## Executive Summary

This document explains the bootstrap methodology used to estimate how many repeated simulations are needed per subject for the large-scale TI simulation study. The goal was to use the repeatability dataset to answer a practical design question:

> How many repeats per subject are needed before the estimated E-field metric becomes sufficiently stable?

For the final analysis requested here, I used the ROI-level `mean_roi` and `median_roi` E-field metrics from the repeatability dataset and applied a bootstrap precision analysis separately to each metric.

Under the current design criterion:

- metric estimated across repeats: `mean`
- tolerance: `5%` relative precision
- coverage: `95%`
- decision rule: bootstrap central interval half-width <= `5%` of the full-repeat reference value

the results are:

| Metric | Cohort recommendation |
| --- | ---: |
| `mean_roi` | 2 repeats per subject |
| `median_roi` | 3 repeats per subject |

Therefore, if one final single number is needed for the large-scale simulation and both `mean_roi` and `median_roi` must be adequately stable, the recommended value is:

## Final Recommendation

**Run 3 repeats per subject.**

This is the smallest number that satisfies the current bootstrap precision target for both the ROI mean E-field and the ROI median E-field across all four repeatability subjects.

## Scope of This Analysis

This document is specifically about repeat-level numerical stability within subjects. It does **not** estimate biological variability between subjects. It answers:

- how stable the per-subject metric becomes as the number of repeats increases

It does **not** answer:

- how many subjects are needed for a population study
- whether the current four repeatability subjects fully represent the whole CamCan cohort
- whether a different montage, ROI, segmentation pipeline, or output metric would need a different repeat count

Also, per your instruction, the final recommendation here is based on `mean_roi` and `median_roi`, not on peak E-field metrics.

## Dataset Used

The repeatability data were read from:

```text
/media/boyan/main/PhD/CamCan-SimNIBS_Repeatability/new_params/_analysis
```

For each subject, the input file was:

```text
sub-*/summary.csv
```

These files contain one row per repeat and several repeat-level summary metrics, including:

- `mean_roi`
- `median_roi`
- `peak_roi`
- other head and hotspot metrics

The four analyzed subjects were:

- `sub-CC120120`
- `sub-CC210124`
- `sub-CC410243`
- `sub-CC721888`

Available valid repeats:

| Subject | Valid repeats used |
| --- | ---: |
| `sub-CC120120` | 40 |
| `sub-CC210124` | 40 |
| `sub-CC410243` | 38 |
| `sub-CC721888` | 40 |

Because `sub-CC410243` had only 38 valid repeats, the common maximum repeat count that could be evaluated uniformly across subjects was 38.

## What Bootstrapping Means in This Context

Bootstrapping is a resampling method. Instead of assuming a specific probability distribution for the data, it uses the observed data themselves as an empirical distribution and repeatedly resamples from them with replacement.

In this project, each subject has a set of repeat-level E-field values. For example, if a subject has 40 repeats, then that subject contributes 40 observed values of `mean_roi` and 40 observed values of `median_roi`.

The bootstrap question is:

> If in the future we only ran `n` repeats for a subject instead of all available repeats, how variable would the estimated subject-level metric be?

To answer that, for each candidate repeat count `n`, we repeatedly:

1. sample `n` repeats from the subject's observed repeat values, with replacement
2. compute the chosen subject-level estimator from those sampled values
3. examine the resulting distribution of the estimator across many bootstrap resamples

This gives an empirical approximation to the uncertainty we would expect if we only had `n` repeats for that subject.

## Why Bootstrapping Is Appropriate Here

The bootstrap is appropriate here for several reasons:

1. We have repeated measurements per subject, which is exactly the structure needed for within-subject resampling.
2. We do not need to assume that repeat-level values are normally distributed.
3. We want a design answer tied directly to the observed numerical stability of the simulation outputs.
4. The main question is precision, not hypothesis testing.

Conceptually, the repeatability dataset acts as the best currently available empirical approximation to the true within-subject repeat distribution.

## Statistical Target Used for the Recommendation

The analysis needs a formal definition of "stable enough." I used the following target:

- central bootstrap coverage: `95%`
- relative tolerance: `5%`
- primary decision rule: the central bootstrap interval half-width must be <= `5%` of the full-repeat reference estimate

In plain language:

> A repeat count `n` is considered adequate for a subject if the uncertainty around the bootstrapped estimate is narrow enough that the half-width of the central 95% interval is at most 5% of the subject's full-repeat reference value.

This criterion was evaluated for every subject, and the cohort-level recommendation is the smallest `n` for which **all** subjects pass.

## Exact Definition of the Quantities

For a subject `s`, let the observed repeat-level values be:

```text
x_s = {x_s1, x_s2, ..., x_sm}
```

where `m` is the number of valid repeats for that subject.

For the current reported results:

- when analyzing `mean_roi`, each `x_si` is the ROI mean E-field from repeat `i`
- when analyzing `median_roi`, each `x_si` is the ROI median E-field from repeat `i`

### Reference Value

The script uses a reference value computed from all available repeats for that subject:

```text
theta_s = mean(x_s)
```

This means:

- for `mean_roi`, the reference is the mean of the per-repeat ROI means
- for `median_roi`, the reference is the mean of the per-repeat ROI medians

This is important: the current reported results use the **mean across repeats** as the estimator, even for the `median_roi` metric. In other words, `median_roi` refers to the within-repeat summary metric, not to the across-repeat estimator.

### Bootstrap Estimator for a Candidate Repeat Count

For each candidate repeat count `n`, and for each bootstrap replicate `b`:

1. sample `n` values from `x_s` with replacement
2. compute

```text
theta_hat_s,n,b = mean(bootstrap sample)
```

This was repeated `20,000` times for each subject and each value of `n`.

### Precision Metrics

For each subject and candidate `n`, two precision quantities were computed from the bootstrap distribution:

#### 1. Relative central interval half-width

Let:

```text
L = 2.5th percentile of bootstrap estimates
U = 97.5th percentile of bootstrap estimates
```

Then the relative half-width is:

```text
(U - L) / (2 * |theta_s|)
```

This was the primary criterion used for the reported recommendation.

#### 2. 95th percentile of absolute relative error

For each bootstrap estimate:

```text
error = |theta_hat_s,n,b - theta_s| / |theta_s|
```

Then the alternative criterion is the 95th percentile of this error distribution.

This second rule is slightly more direct, because it asks:

> In 95% of bootstrap resamples, how close is the estimated value to the full-repeat reference?

In the present `mean_roi` and `median_roi` analyses, this alternative rule gave the same final cohort recommendations as the primary rule.

## How the Implementation Works

The implementation is in:

```text
simulation/bootstrap_repeatability.py
```

### Input Discovery

The script accepts either:

- the batch root containing `_analysis/`
- or the `_analysis/` directory directly

It then searches for:

```text
sub-*/summary.csv
```

### Main Parameters Used

For the final reported runs, the parameters were:

- `--metric mean_roi` or `--metric median_roi`
- `--estimator mean`
- `--criterion ci_half_width`
- `--tolerance 0.05`
- `--coverage 0.95`
- `--bootstrap-iterations 20000`
- `--seed 0`

### Algorithm

The script performs the following steps:

```text
for each subject:
    load the chosen metric column from summary.csv
    remove non-finite values
    compute the full-repeat reference value

for n from min_repeats to max_repeats:
    for each subject:
        draw 20,000 bootstrap samples of size n with replacement
        compute the estimator for each resample
        compute:
            - central 95% interval
            - relative half-width
            - 95th percentile absolute relative error
        mark whether the subject passes the tolerance rule

aggregate across subjects:
    for each n:
        determine whether all subjects pass

recommendation:
    choose the smallest n for which all subjects pass
```

### Output Files Written by the Script

For each run, the script writes:

- `per_subject_curve.csv`
- `overall_curve.csv`
- `recommendation.json`
- `precision_curves.png`

These contain:

- subject-level precision curves
- worst-subject cohort curves
- the final recommended repeat count
- a visual summary of how uncertainty decreases as repeats increase

## Commands Used

The two final runs used for the current recommendation were:

```bash
python simulation/bootstrap_repeatability.py \
  --input-root /media/boyan/main/PhD/CamCan-SimNIBS_Repeatability/new_params \
  --metric mean_roi \
  --estimator mean \
  --criterion ci_half_width \
  --tolerance 0.05 \
  --coverage 0.95 \
  --bootstrap-iterations 20000 \
  --out-dir simulation/bootstrap_repeatability_outputs/mean_roi
```

```bash
python simulation/bootstrap_repeatability.py \
  --input-root /media/boyan/main/PhD/CamCan-SimNIBS_Repeatability/new_params \
  --metric median_roi \
  --estimator mean \
  --criterion ci_half_width \
  --tolerance 0.05 \
  --coverage 0.95 \
  --bootstrap-iterations 20000 \
  --out-dir simulation/bootstrap_repeatability_outputs/median_roi
```

## Results

### Cohort-Level Results

| Metric | Estimator across repeats | Criterion | Recommendation |
| --- | --- | --- | ---: |
| `mean_roi` | mean | 95% interval half-width <= 5% | 2 |
| `median_roi` | mean | 95% interval half-width <= 5% | 3 |

Using the alternative absolute-error criterion:

| Metric | Alternative criterion | Recommendation |
| --- | --- | ---: |
| `mean_roi` | 95th percentile absolute relative error <= 5% | 2 |
| `median_roi` | 95th percentile absolute relative error <= 5% | 3 |

### Per-Subject Results

#### `mean_roi`

| Subject | Available repeats | Reference value | Repeats needed |
| --- | ---: | ---: | ---: |
| `sub-CC120120` | 40 | 0.2702635644 | 2 |
| `sub-CC210124` | 40 | 0.3252193622 | 2 |
| `sub-CC410243` | 38 | 0.2720391158 | 2 |
| `sub-CC721888` | 40 | 0.2203635282 | 2 |

#### `median_roi`

| Subject | Available repeats | Reference value | Repeats needed |
| --- | ---: | ---: | ---: |
| `sub-CC120120` | 40 | 0.2751025464 | 2 |
| `sub-CC210124` | 40 | 0.3426276546 | 2 |
| `sub-CC410243` | 38 | 0.2750497437 | 3 |
| `sub-CC721888` | 40 | 0.2201240057 | 2 |

### Interpretation of the Results

The most important observation is that the ROI mean and ROI median E-field metrics are substantially more stable across repeats than the ROI peak metric that had been explored earlier. For the current final scope:

- `mean_roi` stabilizes extremely quickly
- `median_roi` is also very stable, but one subject (`sub-CC410243`) requires 3 repeats instead of 2

That means the study does not need anywhere near 40 repeats per subject if the goal is to estimate stable subject-level ROI mean and ROI median E-field values under the current precision target.

Because a single operational repeat count is usually easier to manage in large-scale simulation, the conservative combined choice is:

**3 repeats per subject**

This covers both final metrics at once.

## Figures

### `mean_roi` precision curves

![Mean ROI precision curves](../simulation/bootstrap_repeatability_outputs/mean_roi/precision_curves.png)

### `median_roi` precision curves

![Median ROI precision curves](../simulation/bootstrap_repeatability_outputs/median_roi/precision_curves.png)

## Practical Interpretation for the Large-Scale Simulation

If the large-scale study will report subject-level ROI mean and ROI median E-field values, then:

- `2` repeats is enough for `mean_roi`
- `3` repeats is enough for `median_roi`
- therefore `3` repeats per subject is the recommended single design choice

This recommendation is efficient because it reduces compute substantially relative to 40 repeats while still keeping the estimated per-subject values within the current precision target.

## Assumptions and Limitations

The analysis depends on the following assumptions:

1. The observed repeatability dataset is representative of future repeats under the same simulation pipeline.
2. The four repeatability subjects are adequate proxies for the within-subject repeat behavior expected in the large-scale run.
3. The relevant final outputs are `mean_roi` and `median_roi`, not peak metrics.
4. The chosen tolerance (`5%`) and coverage (`95%`) reflect the desired study precision.

Limitations:

1. Only four subjects were available for this repeatability calibration.
2. The result is metric-specific. If the final study focuses on a different metric, the repeat count may change.
3. The result is pipeline-specific. Changes to the mesh generation, ROI definition, atlas handling, or stimulation setup could alter repeat variability.
4. This bootstrap treats the observed repeat distribution as the empirical sampling distribution; if future repeats behave differently, the true required repeat count may differ.

## Reproducibility

Relevant files:

- script: `simulation/bootstrap_repeatability.py`
- mean result JSON: `simulation/bootstrap_repeatability_outputs/mean_roi/recommendation.json`
- median result JSON: `simulation/bootstrap_repeatability_outputs/median_roi/recommendation.json`
- mean plot: `simulation/bootstrap_repeatability_outputs/mean_roi/precision_curves.png`
- median plot: `simulation/bootstrap_repeatability_outputs/median_roi/precision_curves.png`

These files contain everything needed to rerun or audit the analysis.

## Closing Statement

Using bootstrap resampling of the repeatability dataset, the large-scale simulation can be reduced to **3 repeats per subject** if the target outputs are the ROI mean E-field and ROI median E-field and the accepted precision target is a 95% bootstrap interval half-width of at most 5% of the full-repeat reference estimate.
