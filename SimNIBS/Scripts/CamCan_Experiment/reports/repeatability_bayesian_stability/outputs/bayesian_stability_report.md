# Two-stage Bayesian stability analysis

## Question addressed

Would the participant-level remeshing-variability summaries plausibly change if
new anatomies were added to the present 10-participant cohort?

This analysis does **not** prove that 10 participants are universally
representative. It tests internal predictive calibration and then quantifies
model-conditional stability under explicit cohort expansion.

## Inferential unit and outcome

- Inferential unit: participant, not technical repeat.
- Outcome: paired participant-level remesh CV (%) for left hippocampus and right
  M1. Each CV was estimated from 40 remeshing repeats.
- The pairing is preserved because both targets come from the same participant.
- Input: `/home/boyan/sandbox/repeatability_paper_analysis_v1/corrected_fixed_analysis/subject_condition_summary.csv`
- Participants: 10; remesh rows: 20; technical repeats
  per participant-target: 40.
- Fixed mesh is retained as a descriptive numerical-control arm. Of its 20
  participant-target CVs, 17 are below 0.000001%, and the maximum is
  0.01603%. A positive continuous log-CV population model is therefore
  inappropriate for that near-degenerate arm.

## Model

For participant *i*, let
`z_i = [log(CV_hippocampus), log(CV_M1)]`. The sampling model is

`z_i ~ MultivariateNormal(mu, Sigma)`.

`mu` describes the population-average paired log-CVs and `Sigma` describes
between-participant variation and cross-target correlation. A proper
Normal-Inverse-Wishart prior gives an exact posterior, so the analysis does not
depend on MCMC convergence. Posterior-predictive draws are exponentiated back to
CV percentage units.

Priors assessed:

- `reference_weak`: Weak proper prior: 0.05 participant-equivalents on the population mean, centred at 2.5% CV; broad log-SD 0.75.
- `diffuse_weak`: More diffuse weak prior: 0.005 participant-equivalents on the population mean and log-SD 1.50.
- `low_variability_challenge`: Deliberately more informative sensitivity challenge: 0.5 participant-equivalents centred at 1% CV; log-SD 0.50.

The reference results below use `reference_weak`. Full CSV outputs contain
all three priors.

## Descriptive, assumption-light stability

| Target | Observed median | Leave-one-out median range | Participant bootstrap 95% interval |
| --- | --- | --- | --- |
| Left hippocampus | 2.40% | 2.31–2.48% | 2.00–2.76% |
| Right M1 | 2.54% | 2.52–2.57% | 2.28–2.67% |

The bootstrap resamples whole paired participants. It describes uncertainty in
the current cohort median without assuming the Bayesian population model.

## Stage 1: leave-one-participant-out posterior prediction

The model was refitted 10 times. Each time, both targets and all technical
repeats belonging to one participant were withheld. A full predictive
distribution for that participant was generated from the remaining nine.

| Target | Inside 80% interval | Inside 95% interval | Median absolute error | Median 80% width |
| --- | --- | --- | --- | --- |
| Left hippocampus | 9/10 | 10/10 | 0.43 pp | 2.45 pp |
| Right M1 | 9/10 | 10/10 | 0.28 pp | 2.27 pp |

Coverage is necessarily coarse: one participant changes the rate by 10
percentage points. The observed coverage is compatible with internal
calibration, but the predictive intervals are broad and can over-cover. It does
not demonstrate coverage of rare anatomies absent from this cohort.

## Stage 2: posterior-predictive cohort expansion

The model was fitted to all 10 participants. For every posterior draw, up to
40 additional paired participants were sampled and appended to
the observed cohort. The statistic recomputed after expansion was the cohort
median remesh CV.

| Target | Total n | Predictive median | 80% interval | 95% interval |
| --- | --- | --- | --- | --- |
| Left hippocampus | 15 | 2.40% | 2.15–2.56% | 2.02–2.58% |
| Left hippocampus | 20 | 2.40% | 2.13–2.60% | 2.02–2.74% |
| Left hippocampus | 30 | 2.40% | 2.09–2.66% | 2.01–2.83% |
| Left hippocampus | 40 | 2.40% | 2.09–2.69% | 2.00–2.88% |
| Left hippocampus | 50 | 2.40% | 2.09–2.71% | 2.00–2.91% |
| Right M1 | 15 | 2.52% | 2.39–2.66% | 2.35–2.66% |
| Right M1 | 20 | 2.51% | 2.37–2.65% | 2.26–2.68% |
| Right M1 | 30 | 2.49% | 2.28–2.66% | 2.14–2.71% |
| Right M1 | 40 | 2.47% | 2.23–2.66% | 2.09–2.76% |
| Right M1 | 50 | 2.46% | 2.20–2.67% | 2.06–2.79% |

The intervals widen away from n=10 because the observed n=10 median is fixed,
whereas the expanded median depends increasingly on unseen participants and on
uncertainty about the population distribution. This is **not** evidence that a
larger completed cohort would be less precise. It is uncertainty about how far
the present n=10 result could move while that cohort is being expanded.

The next table shows the posterior probability that the expanded-cohort median
differs from the observed n=10 median by more than a candidate tolerance
`Delta`. Units are CV percentage points, not relative percent.

| Target | Total n | P(shift > 0.10 pp) | P(shift > 0.25 pp) | P(shift > 0.50 pp) |
| --- | --- | --- | --- | --- |
| Left hippocampus | 20 | 59.4% | 18.3% | 0.3% |
| Left hippocampus | 30 | 62.8% | 24.3% | 1.7% |
| Left hippocampus | 40 | 63.9% | 26.7% | 2.7% |
| Left hippocampus | 50 | 64.6% | 28.1% | 3.5% |
| Right M1 | 20 | 39.3% | 4.8% | 0.1% |
| Right M1 | 30 | 54.1% | 11.9% | 0.9% |
| Right M1 | 40 | 60.0% | 17.0% | 1.8% |
| Right M1 | 50 | 63.2% | 20.4% | 2.6% |

No single tolerance is declared acceptable here. That threshold should be set
from the scientific or practical consequence of a change before using this
analysis to make an adequacy claim. The complete probability surface is in
`material_change_probability.csv` for `Delta = 0.05` to `1.00` percentage
points.

The prior-sensitivity output is important with only 10 participants. The
diffuse prior produces wider predictive ranges than the reference prior; the
more informative low-variability challenge can shift the predicted population
centre. Agreement of a substantive conclusion across these columns is more
defensible than reliance on one prior alone.

## Defensible interpretation

If held-out observations are reasonably calibrated and the probability of
exceeding a prespecified meaningful `Delta` is low at relevant expanded cohort
sizes, the results support the narrower claim that the **central cohort median
is unlikely to change materially under this fitted population model**.

They do not establish the population tails, rare anatomies, subgroup effects,
or universal representativeness. An independent extension cohort remains the
strongest validation. This analysis also addresses participant-level remesh CV;
it does not re-estimate the 501x nested between-mesh/residual SD ratio, which was
obtained from one fully nested participant-target case.

## Reproducibility

- Posterior draws per fit: 40,000
- Random seed: 20261005
- Maximum simulated total n: 50
- Exact leave-one-out refits: 10 per prior
- Input SHA-256: `da22242a0b1a9659016cb4ee2f75dd88e543ebc1f6f753ac39bc9fdb87c0e0d0`

Run:

```bash
/home/boyan/anaconda3/envs/simnibs_post/bin/python \
  /home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/reports/repeatability_bayesian_stability/run_bayesian_stability.py
```
