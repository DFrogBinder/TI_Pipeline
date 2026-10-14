# MNI-relative field representativeness analysis

`analyze_mni_field_representativeness.py` checks where the selected 14
participants sit within the complete final-132 generic-montage field
distribution before personalised optimisation is performed.

The primary analysis uses the ten-repeat mean target-ROI field from the
schema-4 optimizer-matched ROI package. Each participant is expressed relative
to the same montage simulated on MNI152. Supporting tables retain target
median field, target coverage at 0.20 V/m, and off-target coverage at 0.20 V/m.

The comparison group contains the 111 final-132 participants who are neither
in the selected 14 nor in the seven previously optimised outcome-selected
cases. Diagnostics are adjusted for continuous age, recorded sex, and their
interaction. They are descriptive and do not constitute an equivalence test.

Run from the repository root with the local post-processing environment:

```bash
export MPLCONFIGDIR=/tmp/mni-field-representativeness-mpl
/home/boyan/anaconda3/envs/simnibs_post/bin/python \
  optimized_vs_unoptimized_age_cohort/analyze_mni_field_representativeness.py
```

The default output folder is:

```text
optimized_vs_unoptimized_age_cohort/downloads/Results/representativeness/
```

The script refuses to overwrite an existing result. Add `--force` only when
intentionally regenerating the package. Input paths can be replaced with the
corresponding command-line options when the manuscript source package is
moved. Use `--help` to list them.

Principal outputs include separate PNG/PDF figures for mean target field and
target coverage at or above 0.20 V/m, an analysis report, selected-subject
profiles, target-level diagnostic summaries, a four-target multivariate audit,
an input/output manifest, and SHA-256 checksums.
