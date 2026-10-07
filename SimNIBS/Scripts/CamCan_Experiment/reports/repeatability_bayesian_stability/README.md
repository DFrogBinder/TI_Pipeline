# Bayesian cohort-stability analysis

This directory implements the two-stage analysis proposed for the 10-participant
repeatability study:

1. exact leave-one-participant-out posterior-predictive validation; and
2. posterior-predictive cohort expansion after fitting all 10 participants.

The statistical unit is the participant. The left-hippocampus and right-M1
remesh CVs are modelled jointly so that their within-participant pairing is
preserved. The 40 technical repeats estimate each participant-target CV; they
do not increase the population sample size.

Run the verified default analysis with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-repeatability \
/home/boyan/anaconda3/envs/simnibs_post/bin/python \
  CamCan_Experiment/reports/repeatability_bayesian_stability/run_bayesian_stability.py
```

Run tests with:

```bash
cd CamCan_Experiment/reports/repeatability_bayesian_stability
/home/boyan/anaconda3/envs/simnibs_post/bin/python -m unittest -v
```

The main human-readable result is
`outputs/bayesian_stability_report.md`. Three presentation-ready PNGs and all
underlying CSV tables are written beside it.

Important scope boundary: the model addresses participant-level remesh-CV
stability. It does not turn the one fully nested participant into a population
sample and does not re-estimate the nested 501x between-mesh/residual SD ratio.
