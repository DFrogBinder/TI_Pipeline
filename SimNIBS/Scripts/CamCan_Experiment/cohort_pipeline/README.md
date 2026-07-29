# Corrected-v4 four-ROI cohort pipeline

This workflow runs a selected CamCan subject cohort through all four study
ROIs and all ten remeshing repeats without tying the implementation to one
subject count. Cohorts may contain 1–200 subjects.

The established scientific constraints are enforced:

- the segmentation source is the supervisor-reviewed corrected-v4 collection;
- original-map meshes and simulations are not reused;
- every subject/ROI/repeat receives an independently generated tetrahedral
  mesh;
- FEM starts only after every mesh in the selected cohort validates;
- ROAST is never invoked;
- each subject has one reusable canonical m2m scaffold;
- a valid legacy scaffold is imported without rerunning segmentation;
- subjects without a reusable scaffold receive one CHARM segmentation
  bootstrap followed by a temporary `charm <subject> --mesh` run, which is
  required to create the subject-space EEG cap;
- the corrected-v4 map replaces any label produced or imported during
  scaffold creation before the temporary mesh and all study remeshes;
- the temporary scaffold mesh is verified and deleted; it is never counted as
  or reused for any independent study mesh.

## Storage layout

The study and scaffold roots are stable across cohorts:

```text
/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Scaffolds/
└── subjects/<subject>/anat/m2m_<subject>/

/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Four_ROI/
├── campaigns/<cohort-id>/
├── results/
│   ├── meshes/<roi>/<repeat>/<subject>.json
│   └── simulations/<roi>/<repeat>/<subject>.json
└── runs/<roi>_Runs/<roi>_Data_<repeat>/<subject>/anat/
```

The stable paths are intentional. If the final cohort expands the interim
cohort, valid scaffold, mesh, and simulation results are reused by subject,
ROI, repeat, input-map hash, and mesh hash. Removed subjects remain on disk but
are absent from the new cohort manifest.

Each repeat has a physical m2m scaffold copy and its own `.msh` file. The T1
and T2 files in a repeat anatomy directory are relative links to the canonical
scaffold copies; the m2m directory itself is not symlinked.

## Execution architecture

The launcher submits one automatically released dependency chain:

1. one scaffold array;
2. sequential packed direct-meshing arrays, two mesh workers per eight-CPU
   array element, four threads per worker, with node-local staging;
3. sequential FEM arrays using `--reuse-existing-mesh`.

For a new bootstrap subject, the scaffold task has two CHARM phases:

1. `--registerT2 --initatlas --segment --forceqform` creates the complete m2m
   support directory;
2. after installing the exact corrected-v4 label, `--mesh` creates the
   transformed EEG-cap coordinates required by the four montages.

The resulting temporary `.msh` is removed immediately. If phase 1 completed
but a retry occurred before the cap was available, the exact installed-label
and source-image hashes act as a recovery checkpoint: the retry resumes at
phase 2 without repeating segmentation.

Only one large array is submitted at a time. A one-CPU release job with an
`afterok` dependency submits the next chunk when the preceding chunk finishes,
then attaches the next release job. The 875-element chunk ceiling and one
pending releaser keep this campaign at or below 877 submitted jobs, avoiding
Stanage's per-user QOS submission ceiling. The release job retries only
`QOSMaxSubmitJobPerUserLimit` submission failures; scientific task retries
remain unlimited and separate.

For the final 132-subject cohort the full scope is:

- 132 scaffold tasks;
- 5,280 independent meshes (132 × 4 ROIs × 10 repeats);
- 2,640 packed mesh array elements, split into four sequential chunks;
- 5,280 validated FEM simulations, split into seven sequential chunks;
- twelve scientific arrays in total, connected by small release jobs.

For 200 subjects the same code creates 8,000 mesh tasks and 8,000 FEM tasks,
split automatically. No script change is required.

Task retries default to `unlimited`. Completed siblings in a requeued packed
mesh element are reused rather than recalculated. A deterministic failure can
therefore remain active until it is diagnosed or manually cancelled.

## Final supervisor-approved cohort

`cohorts/final_132/` contains the authoritative 132 subjects from the
supervisor's completed corrected-v4 review export. Its `cohort.json` records
the review-database and accepted-export hashes, final review counts, and its
relationship to the earlier provisional cohorts.

The folder also preserves a history-derived candidate for the supervisor's
strict 39-subject checkpoint. This is an optional downstream sensitivity tier:
all 132 subjects are simulated once, and the 39-subject subset can be selected
during analysis without duplicating any meshing or FEM work.

After pulling the committed code on Stanage, run the non-submitting gate first:

```bash
cd /users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts
conda deactivate

bash CamCan_Experiment/cohort_pipeline/submit_cohort_pipeline.sh \
    final_132 \
    --preflight
```

Review the reported subject, scaffold-import/bootstrap, mesh, and FEM counts.
Then submit the complete dependency chain:

```bash
bash CamCan_Experiment/cohort_pipeline/submit_cohort_pipeline.sh \
    final_132
```

The launcher refuses a duplicate submission while job IDs recorded for that
cohort are still active. The job-ID file grows as later chunks are released.

Monitor the chain with the bounded, read-only progress reporter:

```bash
bash CamCan_Experiment/cohort_pipeline/check_cohort_progress.bash final_132
```

The report summarizes manifest scope, completed scaffold/mesh/simulation
markers, per-ROI/repeat output counts, release receipts, live Slurm state,
accounting failures, retry markers, and recent releaser messages. Slurm calls
and filesystem scans are individually time-bounded; the report performs no
hashing and does not load meshes or NIfTI outputs.

The lower-level queue view remains available with:

```bash
CAMPAIGN=/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Four_ROI/campaigns/final_132

squeue -j "$(paste -sd, "$CAMPAIGN/submitted_job_ids.txt")" \
    -o '%.18i %.2t %.10M %.30R'
```

## Final-cohort post-processing

After all 5,280 FEM simulations validate, run the post-processing preflight:

```bash
bash CamCan_Experiment/cohort_pipeline/submit_cohort_post_processing.sh \
    final_132 \
    --preflight
```

The gate requires:

- the completed FEM release-chain receipt;
- 5,280 current simulation outputs;
- all 132 flat subject-space FastSurfer atlases;
- the fixed MNI atlas;
- one ROI-specific MNI baseline for each of the four study ROIs;
- the confirmed `targets.csv` hash;
- the explicit `ti-post` Python environment and its dependencies.

Submit after the preflight passes:

```bash
bash CamCan_Experiment/cohort_pipeline/submit_cohort_post_processing.sh \
    final_132
```

The post launcher submits 40 independent dataset jobs (four ROIs times ten
repeats) with a maximum of 20 active dataset jobs. Each dataset job uses 12
single-threaded Python workers to produce the 132 subject-level metric sets.
Four ROI collector jobs start only after all 40 dataset jobs succeed. The
collectors fingerprint-check and skip completed subject outputs, then build:

- 40 complete-case within-run population summaries;
- four across-repeat repeatability analyses;
- four static post-processing figure collections;
- one auditable batch summary and complete-repeat subject manifest per ROI.

Subject outputs are resumable. If the one-shot post array needs to be
resubmitted after a failure, complete metrics with the same configuration
fingerprint are reused; `PIPELINE_FORCE` remains disabled by default.

## Optimizer-matched manuscript analyses

The manuscript analysis is a compact, visualization-free extension of the
completed simulation campaigns. It reads the existing whole-brain TI NIfTIs
and subject-space atlases without modifying either. For each subject and ROI,
it reproduces the target construction in the optimization `MakeROIs.m`:

- calculate the world-space volume centroid of the anatomical parcel;
- grow a sphere from 3.00 mm in 0.01 mm increments (strict distance `< radius`);
- clip the sphere to the anatomical parcel;
- stop at 100 mm³ for cortical M1/DLPFC targets or 200 mm³ for subcortical
  hippocampus/thalamus targets, with a radius cap below 10 mm.

The unprefixed metrics use this optimizer-matched target. The same metrics are
also calculated over the complete anatomical parcel with an `anatomical_`
prefix as a secondary analysis. Metrics are calculated independently in every
repeat and then arithmetic-mean aggregated across the ten repeats. Run:

```bash
bash CamCan_Experiment/cohort_pipeline/submit_cohort_manuscript_analysis.sh \
    final_132 \
    --preflight

bash CamCan_Experiment/cohort_pipeline/submit_cohort_manuscript_analysis.sh \
    final_132
```

The cohort launcher submits 40 resumable ROI/repeat jobs at up to 40-way concurrency,
followed by one `afterok` collector. It uses 12 CPU, 24 GB, and a one-hour
time limit for both metric extraction and collection. The analysis calculates:

- target, off-target, and whole-brain coverage at 0.20, 0.18, and 0.15 V/m;
- localization of suprathreshold and whole-brain top-5% voxels in the target;
- mean and median target field as primary summaries, with the minimum and
  P99.9 robust maximum retained for validation/QC in the target, whole brain,
  and off-target compartment;
- the median of the upper 1% as a robust-maximum sensitivity analysis;
- the same metrics for each ROI-specific MNI152 baseline.

Target coverage uses the complete optimizer-matched target as its denominator,
so non-finite target voxels count as unstimulated. The collector writes
repeat-level and ten-repeat subject-mean CSVs, an ROI construction audit table,
a primary table, a supplementary descriptive table (mean, SD, median,
quartiles, IQR, range), effectiveness-versus-spread figures, an audit manifest,
and a compact download archive under:

```text
campaigns/final_132/post_processing/optimizer_matched_analysis/
```

The completed individualized campaign contains the correct personalized
montage for all seven subjects and all four ROIs. Analyse its complete 28
subject/ROI grid against the generic montage on the same heads with:

```bash
bash CamCan_Experiment/cohort_pipeline/submit_personalized_vs_generic_analysis.sh \
    --preflight

bash CamCan_Experiment/cohort_pipeline/submit_personalized_vs_generic_analysis.sh
```

This submits 28 resumable configuration jobs at the preserved eight-way
concurrency, followed by one collector. It reads 560 existing inputs
(28 configurations × two conditions × ten repeats), includes all 280
personalized simulations, performs no new FEM simulation, and writes to:

```text
campaigns/optimized_best_worst_7/post_processing/
    optimizer_matched_personalized_vs_generic/
```

Because the seven subjects were selected as outcome extremes, this comparison
is descriptive and is not used for population inference.

The exact MATLAB-to-Python mapping and metric-scope rules are recorded in
[`OPTIMIZER_MATCHED_ROI_ANALYSIS.md`](../docs/OPTIMIZER_MATCHED_ROI_ANALYSIS.md).

If the post-processing preflight reports missing subject-space atlases, first
prepare and submit the missing-only Destrieux repair stage:

```bash
bash CamCan_Experiment/cohort_pipeline/submit_cohort_atlas_repair.sh \
    final_132 \
    --preflight

bash CamCan_Experiment/cohort_pipeline/submit_cohort_atlas_repair.sh \
    final_132
```

The repair preflight preserves existing flat atlases, searches known nested
FreeSurfer output roots for reusable `aparc.a2009s+aseg` NIfTI or MGZ files,
and runs native FreeSurfer reconstruction only where no reusable atlas exists.
It uses the canonical T1 files in `CamCan_Corrected_v4_Scaffolds`; it does not
modify or substitute the corrected CHARM tissue maps. The after-any collector
writes `atlas_repair/validation.json`. Rerun the post-processing preflight only
after that file reports all cohort subjects complete. By default, each
16-CPU/64-GB array element runs two independent eight-thread reconstructions.
If either fails, the element requeues safely: a completed partner atlas is
detected and preserved while the incomplete subject resumes its FreeSurfer
state.

## Archived interim cohort

`cohorts/interim_53/` contains the provisional 53-subject list selected from
the second review database. Its `cohort.json` records the selection rule,
database hash, and subject-list hash. It is retained for provenance and should
not be submitted now that `final_132` exists. Of those 53 provisional subjects,
52 are in the final cohort; `sub-CC410390` is not.

## Validation

After the chain is absent from `squeue`, validate the three layers:

```bash
CAMPAIGN=/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Four_ROI/campaigns/final_132
WORKFLOW=CamCan_Experiment/cohort_pipeline/workflow.py

python3 "$WORKFLOW" validate \
    --stage scaffolds \
    --manifest "$CAMPAIGN/scaffold_tasks.tsv" \
    --summary "$CAMPAIGN/scaffold_validation.tsv"

python3 "$WORKFLOW" validate \
    --stage meshes \
    --manifest "$CAMPAIGN/mesh_tasks.tsv" \
    --summary "$CAMPAIGN/mesh_validation.tsv"

python3 "$WORKFLOW" validate \
    --stage simulations \
    --manifest "$CAMPAIGN/simulation_tasks.tsv" \
    --summary "$CAMPAIGN/simulation_validation.tsv"
```

The simulation-stage validation checks the result marker, exact mesh hash, and
loadable final NIfTI outputs. A second independent output-only validation is:

```bash
python3 CamCan_Experiment/simulation/prepare_inplace_rerun.py validate \
    --manifest "$CAMPAIGN/simulation_tasks.tsv" \
    --summary "$CAMPAIGN/simulation_output_validation.tsv"
```

## Registering the final cohort

Prepare a text file containing one `sub-CC...` identifier per line. Register
it without editing pipeline code:

```bash
python3 CamCan_Experiment/cohort_pipeline/workflow.py create-cohort \
    --cohort-id final_v1 \
    --subjects-file /absolute/path/to/final_subjects.txt \
    --output-root CamCan_Experiment/cohort_pipeline/cohorts \
    --status final \
    --note "Final supervisor-approved corrected-v4 cohort"
```

Commit and pull the new `cohorts/final_v1/` directory, then use the same
preflight and submission commands with `final_v1`. Subjects already completed
under earlier cohorts are validated and reused; only missing or stale work is
performed.
