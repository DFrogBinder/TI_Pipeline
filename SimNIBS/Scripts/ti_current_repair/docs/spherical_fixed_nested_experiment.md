# Spherical-median fixed-mesh correction and nested repeatability experiment

## Confirmed inconsistency

The completed fixed-mesh runs were seeded by
`post/select_median_remesh_repeats.py`. That selector reads
`_analysis/<subject>/remesh/summary.csv` and uses `median_roi`. The source
`median_roi` is calculated over the complete anatomical atlas parcel by
`post/mesh_repeat_report.py`.

The later optimizer-ROI refresh is separate and read only. Its
`roi_median_v_per_m` field uses the parcel-clipped spherical ROI, but the
completed fixed-mesh selection did not consume that field. The current paper
therefore correctly describes the historical mismatch, but the fixed-mesh
geometry must be regenerated for a methodologically consistent final analysis.

Using the validated schema-2 spherical metric exports downloaded on 2026-08-03,
18 of the 20 participant-target representative meshes change:

| Target | Participant | Anatomical selection | Spherical selection |
| --- | --- | --- | --- |
| Left hippocampus | `sub-CC110174` | `repeat_023` | `repeat_011` |
| Left hippocampus | `sub-CC121144` | `repeat_002` | `repeat_013` |
| Left hippocampus | `sub-CC310407` | `repeat_002` | `repeat_002` |
| Left hippocampus | `sub-CC320616` | `repeat_018` | `repeat_018` |
| Left hippocampus | `sub-CC420071` | `repeat_007` | `repeat_019` |
| Left hippocampus | `sub-CC410432` | `repeat_003` | `repeat_004` |
| Left hippocampus | `sub-CC520083` | `repeat_010` | `repeat_001` |
| Left hippocampus | `sub-CC520127` | `repeat_025` | `repeat_012` |
| Left hippocampus | `sub-CC610631` | `repeat_005` | `repeat_027` |
| Left hippocampus | `sub-CC720941` | `repeat_022` | `repeat_008` |
| Right M1 | `sub-CC110174` | `repeat_013` | `repeat_016` |
| Right M1 | `sub-CC121144` | `repeat_007` | `repeat_006` |
| Right M1 | `sub-CC310407` | `repeat_026` | `repeat_022` |
| Right M1 | `sub-CC320616` | `repeat_007` | `repeat_001` |
| Right M1 | `sub-CC420071` | `repeat_005` | `repeat_035` |
| Right M1 | `sub-CC410432` | `repeat_007` | `repeat_024` |
| Right M1 | `sub-CC520083` | `repeat_008` | `repeat_023` |
| Right M1 | `sub-CC520127` | `repeat_010` | `repeat_007` |
| Right M1 | `sub-CC610631` | `repeat_006` | `repeat_004` |
| Right M1 | `sub-CC720941` | `repeat_003` | `repeat_005` |

The HPC preflight repeats this calculation from the authoritative cluster
files and also requires every selected `.msh` file to exist.

## Exact scope

The correction reuses all completed remesh simulations as immutable sources.
It creates only new fixed-mesh runs in isolated roots:

- 2 targets;
- 10 participants per target;
- 40 fixed-mesh repeats per participant-target case;
- 800 new simulations and 800 expected `TI.msh`/`ti_brain_only.nii.gz` outputs.

The nested experiment uses one prespecified target and one randomly selected
participant:

- default target: left hippocampus;
- 10 eligible participants;
- 40 completed remesh geometries for the selected participant;
- 40 repeated simulations conditional on each geometry;
- 1,600 new simulations and 1,600 expected outputs.

The default random seed is `20260831`. For the current sorted ten-participant
pool, it selects `sub-CC320616`. The first preparation writes
`_pipeline/nested_case_selection.json` before creating caches. Every preflight,
retry, or resumable relaunch reuses that file and refuses to select another
participant if the candidate pool changes. The fixed seed also reproduces the
same choice if the output root must be reconstructed from scratch.

The nested target is deliberately not randomized: the user requested a random
participant, and keeping one target gives the specified 40 x 40 = 1,600 scope.
Set `NESTED_TARGET=right-m1` only before the first preparation if right M1 is
the intended demonstration. Once a selection manifest exists, changing the
target is rejected.

## Output roots

Defaults are isolated from the completed studies:

```text
/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_spherical_fixed_v1
/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_right_m1_spherical_fixed_v1
/mnt/parscratch/users/cop23bi/final_132_repeatability_nested_40x40_v1
```

Each mesh cache contains one physical, checksum-locked copy of its source
remesh geometry. Repeat workspaces use the established fixed-mesh cache logic,
so the completed remesh tree remains read only without creating 40 redundant
geometry copies per cache. No existing remesh or historical fixed-mesh output
is overwritten.

## Stanage workflow

Start from the repository root:

```bash
cd /users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts
```

Run the full read-only preflight. It checks both metric CSVs, all 20 spherical
selections, all candidate nested meshes, the exact task counts, and the
preserved scheduler settings. It creates no output and submits no jobs.

```bash
bash ti_current_repair/hpc_scripts/submit_spherical_fixed_nested_experiment.sh all --preflight
```

Optionally prepare the configs, immutable selection records, and cache links
without submitting:

```bash
bash ti_current_repair/hpc_scripts/submit_spherical_fixed_nested_experiment.sh all --prepare-only
```

Stanage enforces two separate scheduler limits: an array may contain at most
1,000 elements, and all submitted array elements count against the per-user
submitted-job QOS. A 1,000-element array is therefore not a safe chunk size
for a multi-stage workflow because it leaves no QOS slot for a dependent
continuation job. This launcher uses 875-element production chunks, matching
the previously validated QOS-safe Stanage pattern. Production submission must
also keep the 800-task spherical fixed correction and the 1,600-task nested
experiment operationally separate. Mode `all` is therefore restricted to
`--preflight` and `--prepare-only`.

Submit the spherical fixed correction first:

```bash
bash ti_current_repair/hpc_scripts/submit_spherical_fixed_nested_experiment.sh fixed
```

Submit the nested experiment separately:

```bash
bash ti_current_repair/hpc_scripts/submit_spherical_fixed_nested_experiment.sh nested
```

The commands submit resumable simulation arrays. Arrays are split at 875 local
tasks and sequentially released so the global concurrency limit remains 50.
Only the active array and one small `afterok` release controller are submitted
at the first step; the controller submits the next chunk and downstream jobs
after its predecessor succeeds:

- left-hippocampus spherical fixed correction: `0-399%50`;
- right-M1 spherical fixed correction: `0-399%50`;
- persisted nested case chunk 1: local `0-874%50`, global tasks `0-874`;
- persisted nested case chunk 2: local `0-724%50`, global tasks `875-1599`,
  after successful completion of chunk 1.

The array runner adds `TASK_OFFSET` to the local Slurm index before resolving
the experiment task. This avoids Stanage's array-index limit without changing
the 1,600-task nested design.

Each array uses the established SimNIBS 4.0.1 launcher with 8 CPUs, 32 GB,
8 hours, 4-hour per-attempt timeout, and unlimited task requeue. Each is
dependency-gated into completion validation and optimizer-sphere metric
extraction. The nested chain additionally runs the variance decomposition.

## Failure and resume semantics

Do not delete the output root or selection manifests after a failure. Inspect
Slurm accounting and the relevant `.err` file first. The array launcher already
requeues incomplete tasks. If an array must be deliberately resubmitted after
it has left the queue, use:

```bash
RESUBMIT=1 bash ti_current_repair/hpc_scripts/submit_spherical_fixed_nested_experiment.sh fixed
```

or:

```bash
RESUBMIT=1 bash ti_current_repair/hpc_scripts/submit_spherical_fixed_nested_experiment.sh nested
```

The workflow archives the previous job receipt, reuses the same selection and
mesh caches, and skips tasks whose complete outputs and stimulation provenance
already validate. `RESUBMIT=1` must not be used while the prior jobs are still
active.

### Recovery from the accepted 1,000-task nested chunk

Commit `32518ab` submitted global tasks `0-999` as job `11420510`, but Stanage
rejected the simultaneous 600-task chunk with
`QOSMaxSubmitJobPerUserLimit`. Keep job `11420510`; its outputs are valid and
resumable. Once at least one element has completed and released one submitted-
job QOS slot, attach the QOS-safe continuation without resubmitting tasks
`0-999`:

```bash
bash ti_current_repair/hpc_scripts/submit_spherical_fixed_nested_experiment.sh nested --attach-continuation --continue-offset=1000 --previous-job=11420510
```

The one-task controller waits on `afterok:11420510`. After all first-chunk
elements succeed, it submits local `0-599%50` with `TASK_OFFSET=1000`, followed
by finalization, spherical ROI extraction, collection, and nested variance
analysis. If the attachment command itself reports
`QOSMaxSubmitJobPerUserLimit`, no controller was recorded and the same command
is safe to retry after another first-chunk element completes.

## Analysis products

Each correction root receives an isolated spherical metric table at:

```text
_post_processing/optimizer_roi_metrics_v1/optimizer_roi_metrics.csv
```

These tables contain the corrected fixed condition only. The historical remesh
rows remain in the original schema-2 optimizer metric exports and should be
combined with the new fixed rows for the revised paper figures.

The nested root receives:

```text
_analysis/nested_variance/nested_mesh_summary.csv
_analysis/nested_variance/nested_variance_components.csv
_analysis/nested_variance/nested_variance_components.json
```

The balanced one-way random-effects decomposition reports between-mesh and
within-mesh solver/pipeline variance for the spherical `roi_median_v_per_m`
metric. It is a sensitivity demonstration for one persisted participant, not a
population estimate.
