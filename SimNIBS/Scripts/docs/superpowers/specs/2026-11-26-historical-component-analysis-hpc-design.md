# Historical Component-Field Analysis HPC Design

## Purpose

Prepare a read-only HPC workflow that extracts the already simulated pair-1
component E field alongside TI from the consolidated SimNIBS 3.2.6 and 4.0.1
repeatability datasets. This is post-processing of the variability study, not a
tDCS replay and not a new FEM campaign.

## Locked scope

- Source: `final_132_repeatability_combined_dataset_v1` after its consolidation
  completion receipt validates.
- Cohort: the canonical 9+1 participants, including `sub-CC510415` and excluding
  `sub-CC520083`.
- Versions: SimNIBS 3.2.6 and 4.0.1.
- Targets: left hippocampus and right M1.
- Conditions: remesh and fixed mesh.
- Repetitions: 40 per participant/version/target/condition.
- Existing simulations: 3,200.
- Outcomes per simulation: TI target, pair-1 matched target, and pair-1 under
  anode, producing 9,600 metric rows.
- Authoritative source: native float64 `.msh`; NIfTI is never a metric source.
- Meshing and FEM tasks: zero.

## Provenance and ROI contracts

Historical task manifests must record the confirmed stimulation configuration.
The pair-1 anode must be F8 for left hippocampus and Fp2 for right M1. The
historical simulation runner assigned pair-1's positive electrode to channel 1;
SimNIBS constructs that channel's stimulation surface as tag 2101. Extraction
therefore requires tag 2101 to exist in the pair-1 field mesh and aborts if the
manifest, expected anode, or surface geometry disagrees.

The target ROI is reconstructed directly on each native tetrahedral mesh from
the version-appropriate integer atlas. The exact target ROI realization is
mapped to the pair-1 biological support. The under-anode ROI uses the
area-weighted centre of the actual tag-2101 surface, projects to cortical GM,
and grows a 100 mm3 GM-only sphere.

## Execution design

A dedicated historical CLI is separate from the SimNIBS 4.6 production
launcher. Its commands are:

1. `preflight`: validate the completed consolidation receipt, exact source
   matrix, manifests, native mesh paths, atlas paths, and output isolation. It
   performs no writes.
2. `extract-task`: process one version/target/subject task containing both
   conditions and all 40 repetitions (80 simulations, 240 metric rows).
3. `collect-version`: require all 20 task receipts for one version and combine
   them into 12 schema-compatible tables.
4. `analyse-preliminary`: require both version collectors and create a clearly
   preliminary 3.2.6-versus-4.0.1 analysis package.

The scheduler uses two independent 20-task arrays, one per SimNIBS version,
followed by two collectors and one combined analysis job. Candidate resources
follow the previously validated read-only extraction profile: Sheffield, four
CPUs, 16 GB, two hours, with at most ten concurrent array tasks. Preflight is
the default launcher action; submission requires explicit `--submit`.

Every task writes only beneath a new isolated output root. A complete task is
reused only when its receipt, source bindings, table hashes, row counts, and
code version all validate. No source file is modified or deleted.

## Preliminary-analysis boundary

The existing final analysis remains unchanged and continues to accept only
SimNIBS 3.2.6 and 4.6.0. A separate validator and workflow accept exactly
SimNIBS 3.2.6 plus preliminary SimNIBS 4.0.1, require 9,600 rows and 240
participant cells, and stamp all outputs as preliminary. Those outputs may be
used for abstract pre-screening but cannot enter the final paper package.

## Operational boundary

The user alone executes HPC commands. Repository work may prepare and test the
code locally, but it does not connect to the cluster, query Slurm, submit jobs,
or remove data. If an HPC path differs from the documented defaults, the user
will run one exact preflight/path command and return its output.
