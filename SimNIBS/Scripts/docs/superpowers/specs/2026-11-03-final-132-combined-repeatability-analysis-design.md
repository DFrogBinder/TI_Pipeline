# Final-132 combined repeatability analysis design

## Status

The comprehensive 9+1 analysis approach was selected on 2026-11-03. This
document fixes the detailed scientific and operational contract before
implementation. It does not by itself authorize a scheduler submission.

## Purpose

Create the first unified analysis of the completed replacement cohort across:

- SimNIBS 3.2.6 harmonized corrected-label simulations;
- SimNIBS 4 corrected-label simulations;
- left hippocampus and right M1 targets;
- remesh and spherical-median fixed-mesh conditions;
- 40 technical repeats per participant, version, target, and condition.

The canonical cohort contains the nine retained balanced-cohort participants
plus `sub-CC510415`. The faulty `sub-CC520083` is excluded. The analysis uses
the completed source campaigns read-only while the physical dataset copy is
running, and writes only to a separate analysis root.

## Scientific questions

The analysis addresses four distinct questions:

1. What central target-field value and technical variability does each
   participant show within each version, target, and condition?
2. How do SimNIBS 3.2.6 and SimNIBS 4 differ within matched participants?
3. How does fixing the mesh change central field values and technical
   variability within each version?
4. Does the version difference itself change between remesh and fixed-mesh
   conditions?

These questions are separated in the outputs. A fixed-mesh result is a
mechanistic control for identifying sources of computational variation, not a
general recommendation to use fixed meshes in applied modelling.

## Units and interpretation

- Participant is the inferential unit (`n = 10`).
- The 40 runs within a participant-version-target-condition cell are
  technical replicates.
- Technical runs are retained when estimating the participant cell centre and
  within-cell variability; they are never treated as 40 independent people.
- The primary field outcome is `roi_median_v_per_m`.
- `roi_mean_v_per_m` and `roi_p95_v_per_m` are prespecified sensitivity
  outcomes.
- The target masks use aligned optimizer-matched, parcel-clipped spherical ROI
  construction: 200 mm3 left hippocampus and 100 mm3 right M1. Version-specific
  voxel realizations are not described as identical masks.
- Continuous simulated fields are analysed without assigning a therapeutic or
  biological interpretation to 0.2 V/m or any other cutoff.
- The v3 remesh arm is interpreted as a repeated-pipeline/reproducibility
  control when mesh hashes are identical, not as evidence of stochastic mesh
  variation.

## Exact scope

The required matrix is:

- subjects: 10;
- versions: 2 (`simnibs326`, `simnibs4`);
- targets: 2 (`left_hippocampus`, `right_m1`);
- conditions: 2 (`remesh`, `fixed_mesh`);
- repeats per cell: 40;
- complete participant cells: 80;
- run-level metric rows: 3,200.

The canonical subject order is:

1. `sub-CC110174`
2. `sub-CC121144`
3. `sub-CC310407`
4. `sub-CC320616`
5. `sub-CC420071`
6. `sub-CC410432`
7. `sub-CC510415`
8. `sub-CC520127`
9. `sub-CC610631`
10. `sub-CC720941`

`sub-CC520083` must occur zero times in every assembled table and result.

## Source boundary

The active physical copy at
`/mnt/parscratch/users/cop23bi/final_132_repeatability_combined_dataset_v1`
is not an input and is never modified by this workflow.

The analysis reads completed source metric tables and terminal receipts. It
writes to:

```text
/mnt/parscratch/users/cop23bi/final_132_repeatability_combined_analysis_v1
```

The source-bound analysis package retains its four complete canonical input
tables, source paths, source hashes, selected-row provenance, code hash, and Git
commit. It therefore remains interpretable after source retirement. Once the
combined dataset has a completion receipt, a separate read-only rebind check
can verify that the same canonical inputs reconstruct from the final dataset.

The workflow performs:

- zero meshing tasks;
- zero FEM simulations;
- zero writes to completed source campaigns;
- zero writes to the in-progress combined dataset.

It does not silently extract NIfTI data. All four version-target inputs must
already be available as canonical schema-v2 optimizer-ROI metric rows. If a
replacement v4 metric table is absent, preflight stops and reports the exact
160-run metric-refresh scope needed; extraction requires a separate explicit
decision.

## Input assembly

The workflow builds four canonical 800-row tables:

1. SimNIBS 3.2.6 left hippocampus;
2. SimNIBS 3.2.6 right M1;
3. SimNIBS 4 left hippocampus;
4. SimNIBS 4 right M1.

### SimNIBS 3.2.6

For each target:

- select the 720 paired schema-v2 rows from the completed harmonized base-nine
  campaign;
- append the 80 paired schema-v2 rows for `sub-CC510415`;
- require both conditions and all repeats in both sources.

### SimNIBS 4

For each target and the retained nine participants:

- select 360 remesh rows from the established corrected-label campaign;
- select 360 fixed-mesh rows from the spherical-median corrected fixed campaign;
- discard all rows for `sub-CC520083` before assembly.

Append the 80 schema-v2 replacement rows for `sub-CC510415`. Replacement rows
must already represent its completed remesh and spherical-median fixed-mesh
conditions.

### Assembly gates

Every source and assembled table must pass all of the following:

- schema version exactly 2;
- required optimizer-ROI geometry and support columns present;
- expected target name and requested physical volume;
- finite, non-negative median, mean, and p95 field values;
- internally consistent finite/non-finite support counts;
- at least one finite target voxel in every run;
- one invariant target support geometry within each participant;
- exact `repeat_001` through `repeat_040` coverage in every cell;
- no duplicate participant-condition-repeat key;
- exact canonical subject set after assembly;
- exact 800 rows per version-target table and 3,200 rows overall;
- terminal source receipts present and complete;
- `sub-CC510415` present in all eight version-target-condition cells;
- `sub-CC520083` absent.

The assembly manifest records every source file's absolute path, size, SHA-256,
role, observed rows, selected rows, rejected rows, and selected subject set.

## Analysis layers

### Layer 1: established analyses

Run the existing, unchanged canonical analyzers against the four newly
assembled inputs:

- within-version repeatability analysis for SimNIBS 3.2.6;
- within-version repeatability analysis for SimNIBS 4;
- official v3-versus-v4 remesh-distribution comparison.

This layer preserves the established manuscript protocol and code hashes. It
produces subject-condition summaries, finite-support audits, within-version
rank uncertainty, remesh version comparisons, primary figures, sensitivity
outcomes, and exact input/code provenance.

The orchestration code adapts inputs externally. It does not edit the canonical
analysis implementations.

### Layer 2: complete condition-inclusive comparison

A new condition-inclusive analyzer summarizes each participant cell for all
three metrics using:

- mean and median across its 40 runs;
- standard deviation, coefficient of variation, range, and interquartile range;
- minimum finite-support fraction and count of affected runs.

The primary participant cell centre is the median across the 40 run-level
`roi_median_v_per_m` values. Mean and p95 ROI metrics remain sensitivities.

For central field values, calculate these participant-level contrasts:

- v3 minus v4 within each target and condition;
- fixed mesh minus remesh within each target and version;
- the difference-in-differences interaction:
  `(v3 - v4)_fixed - (v3 - v4)_remesh` within each target.

For technical variability, repeat the same contrast structure using the
within-cell run standard deviation as the outcome. CV is reported
descriptively but is not the primary variability contrast because it can be
unstable when means or standard deviations approach zero.

Each contrast uses an exact two-sided paired sign-flip test over the ten
participant contrasts (1,024 sign assignments). The analysis reports the raw
participant contrasts, mean and median effect, SD, Cohen's dz when defined,
exact p-value, and multiplicity-adjusted p-value.

Multiplicity is controlled with Holm adjustment within separate, named
families:

- central version contrasts: 2 targets x 2 conditions x 3 metrics = 12;
- central condition contrasts: 2 targets x 2 versions x 3 metrics = 12;
- central interactions: 2 targets x 3 metrics = 6;
- variability version contrasts: 12;
- variability condition contrasts: 12;
- variability interactions: 6.

The families are not pooled because central exposure and technical variability
answer different scientific questions. All raw values remain available.

### Layer 3: support and provenance audit

Summarize finite target support by version, target, condition, and participant:

- ROI voxel count and achieved volume;
- minimum and median finite fraction;
- runs with incomplete finite support;
- maximum number of non-finite target voxels.

Support differences are treated as a sensitivity/provenance boundary. Any
association between support loss and field values is descriptive and does not
establish causation.

## Outputs

```text
final_132_repeatability_combined_analysis_v1/
├── README.md
├── subjects.txt
├── inputs/
│   ├── simnibs326_left_hippocampus.csv
│   ├── simnibs326_right_m1.csv
│   ├── simnibs4_left_hippocampus.csv
│   ├── simnibs4_right_m1.csv
│   └── assembly_manifest.json
├── established/
│   ├── within_simnibs326/
│   ├── within_simnibs4/
│   └── cross_version_remesh/
├── comprehensive/
│   ├── participant_cell_summary.csv
│   ├── central_value_contrasts.csv
│   ├── technical_variability_contrasts.csv
│   ├── finite_support_audit.csv
│   ├── finite_support_summary.json
│   ├── figures/
│   ├── analysis_report.md
│   └── figure_captions.md
├── provenance/
│   ├── owner.json
│   ├── source_receipts.json
│   ├── analysis_code.json
│   └── run_environment.json
├── analysis_manifest.json
├── SHA256SUMS
├── final_132_repeatability_combined_analysis_v1.tar.gz
└── final_132_repeatability_combined_analysis_v1.tar.gz.sha256
```

Figures include, for each target:

- participant-level central field values across all four version-condition
  cells;
- participant-level technical SD across all four cells;
- paired version contrasts by condition;
- the version-by-condition interaction.

Figures show participant-level summaries. Dense run-level distributions remain
in the established analysis outputs and input tables.

## Workflow interface

Implement a new package:

```text
repeatability_combined_analysis/
```

with a dependency-light Python core, tests, and a thin Slurm launcher.

The shell interface exposes:

- `scope`: print exact rows, cells, jobs, resources, and zero-simulation scope;
- `preflight`: read-only validation and assembly planning; create no output;
- `run`: submit one lightweight analysis job after a successful preflight;
- `status`: report scheduler state and independently validate result receipts;
- `rebind-preflight`: after dataset completion, verify canonical dataset paths
  and reconstructed input hashes without changing either package.

Default analysis resources are one Sheffield job, 4 CPU, 16 GB, and 2 hours.
The launcher must not submit unless the caller explicitly selects `run`.

The run writes to a temporary attempt directory and atomically promotes it only
after all expected artifacts and hashes validate. It refuses to overwrite an
existing incomplete or unowned result root. A complete matching result is
idempotent.

## Failure behaviour

- Missing schema-v2 replacement metrics stop preflight; no fallback to older
  full-parcel summaries or implicit NIfTI extraction is allowed.
- Cohort, condition, repeat, ROI, support, or receipt mismatch stops before
  output creation or submission.
- Existing legacy v3 defaults are never accepted as corrected harmonized v3
  inputs.
- A source change between preflight and execution invalidates the run.
- Partial output stays in an attempt directory and is never reported complete.
- The active combined-dataset copy is never inspected as a scientific input.
- Source retirement remains blocked by the separate consolidation workflow;
  this analysis neither deletes nor moves any source.

## Verification

Tests cover:

- exact 9+1 assembly for each source pattern;
- exclusion and replacement gates;
- incomplete and duplicate repeat rejection;
- ROI geometry and finite-support validation;
- source hash changes between preflight and run;
- all contrast signs and difference-in-differences arithmetic;
- exact sign-flip p-values and Holm adjustment;
- zero-variance and undefined-CV handling;
- canonical analyzer orchestration without modifying canonical code;
- atomic output promotion and overwrite refusal;
- launcher scope, preflight non-mutation, and no implicit submission;
- manifest, checksum, archive, and completion validation.

Before HPC use, the complete local test suite must pass and the launcher must
print the exact 3,200-row, 80-cell, zero-simulation scope.
