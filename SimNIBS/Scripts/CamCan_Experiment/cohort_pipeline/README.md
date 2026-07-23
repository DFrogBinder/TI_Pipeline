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
- subjects without a reusable scaffold receive exactly one CHARM bootstrap;
- the corrected-v4 map replaces any label produced or imported during
  scaffold creation before meshing starts.

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

The launcher submits one dependency chain:

1. one scaffold array;
2. packed direct-meshing arrays, two mesh workers per eight-CPU array element,
   four threads per worker, with node-local staging;
3. FEM arrays using `--reuse-existing-mesh`.

Only one array chunk is eligible at a time. This keeps the established
50-array-task concurrency within Stanage's user limits. Array chunks are
calculated from the live `MaxArraySize` and the cohort size.

For the current 53-subject cohort the full scope is:

- 53 scaffold tasks;
- 2,120 independent meshes (53 × 4 ROIs × 10 repeats);
- 1,060 packed mesh array elements, normally split into two chunks;
- 2,120 validated FEM simulations, normally split into three chunks;
- six dependency-linked arrays in total.

For 200 subjects the same code creates 8,000 mesh tasks and 8,000 FEM tasks,
split automatically. No script change is required.

Task retries default to `unlimited`. Completed siblings in a requeued packed
mesh element are reused rather than recalculated. A deterministic failure can
therefore remain active until it is diagnosed or manually cancelled.

## Current interim cohort

`cohorts/interim_53/` contains the provisional 53-subject list selected from
the second review database. Its `cohort.json` records the selection rule,
database hash, and subject-list hash.

After pulling the committed code on Stanage, run the non-submitting gate first:

```bash
cd /users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts
conda deactivate

bash CamCan_Experiment/cohort_pipeline/submit_cohort_pipeline.sh \
    interim_53 \
    --preflight
```

Review the reported subject, scaffold-import/bootstrap, mesh, and FEM counts.
Then submit the complete dependency chain:

```bash
bash CamCan_Experiment/cohort_pipeline/submit_cohort_pipeline.sh \
    interim_53
```

The launcher refuses a duplicate submission while job IDs recorded for that
cohort are still active.

Monitor the chain with:

```bash
CAMPAIGN=/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Four_ROI/campaigns/interim_53

squeue -j "$(paste -sd, "$CAMPAIGN/submitted_job_ids.txt")" \
    -o '%.18i %.2t %.10M %.30R'
```

## Validation

After the chain is absent from `squeue`, validate the three layers:

```bash
CAMPAIGN=/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Four_ROI/campaigns/interim_53
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
under `interim_53` are validated and reused; only missing or stale work is
performed.

