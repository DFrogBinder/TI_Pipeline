# Current-Repair Staged Median-Fixed Pipeline

This directory owns the current-correction repeatability experiment. The staged
pipeline here is self-contained: it uses the local `experiment_config.py`,
`simulation_runners/`, `hpc_scripts/`, `post/`, and `pipeline/` modules under
`SimNIBS/Scripts/ti_current_repair`.

For the full HPC launch procedure covering both this staged current-repair
pipeline and the restored original mesh-repeat rerun, see
`docs/current_repair_and_mesh_repeat_hpc_runbook.md`.

Run commands from the repository root on the HPC:

```bash
cd /users/cop23bi/Repos/TI_Pipeline
export REPO_ROOT=/users/cop23bi/Repos/TI_Pipeline
export CURRENT_REPAIR_DIR="${REPO_ROOT}/SimNIBS/Scripts/ti_current_repair"
export TARGETS_CSV="${REPO_ROOT}/SimNIBS/Scripts/utils/targets.csv"
```

## End-to-End Run

The preferred production path is one dependency-aware submission. Initialize
the experiment once:

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" init \
  --source-root /mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_corrected \
  --experiment-root /mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10 \
  --subjects sub-CC110174,sub-CC121144,sub-CC310407,sub-CC320616,sub-CC420071,sub-CC410432,sub-CC520083,sub-CC520127,sub-CC610631,sub-CC720941 \
  --repeat-count 40 \
  --atlas-dir /mnt/parscratch/users/cop23bi/ZIPs/atlases \
  --roi-preset left-hippocampus \
  --montage-preset left-hippocampus \
  --targets-csv "$TARGETS_CSV"
```

Run the read-only full-scope submission preflight:

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" submit-all \
  --experiment-root /mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10 \
  --max-concurrent 50 \
  --analysis-max-concurrent 10 \
  --dry-run
```

The preflight must report 10 subjects, 400 remesh tasks, 400 fixed-mesh tasks,
800 expected `TI.msh` outputs, and full requested scope. Then perform the single
production submission:

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" submit-all \
  --experiment-root /mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10 \
  --max-concurrent 50 \
  --analysis-max-concurrent 10
```

This submits the remesh array plus one `afterok` controller. Later stages are
released automatically only after their validation gates pass:

1. 400 remesh simulations.
2. Ten-subject remesh analysis.
3. Median-repeat selection using `median_roi`.
4. Checksum-validated physical fixed-mesh seeding with no symlinks.
5. 400 fixed-mesh simulations.
6. Ten-subject paired analysis.
7. Final figures and completion receipt.

If any gate or job fails, `afterok` prevents downstream release. The workflow
does not silently continue with incomplete subjects.

Monitor at any time:

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" status \
  --experiment-root /mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10
```

Workflow submission, receipts, controller logs, job IDs, and final completion
state are written under `<experiment_root>/_pipeline/workflow/`.

## Final-132 Right-M1 Companion Study

The right-M1 repeatability study uses the same balanced 10 subjects, corrected
staged inputs, 40 remesh repeats, 40 fixed-mesh repeats, resource profile, and
dependency-gated workflow as the left-hippocampus study. It writes to the
separate experiment root:

`/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_right_m1`

The launcher fails closed on the full scope and the confirmed `targets.csv`
row for `ctx_rh_G_precentral`: `Fp2-F6` at 2.0 mA and `C4-CP2` at
0.6324555320336759 mA. Analysis uses the Destrieux right precentral-gyrus
label `12129`.

From `SimNIBS/Scripts` on Stanage, initialize and validate without submitting:

```bash
bash ti_current_repair/hpc_scripts/submit_final132_right_m1_repeatability.sh --preflight
```

The preflight must report:

```text
subjects: 10
repeats per condition: 40
remesh tasks: 400 (0-399%50)
fixed-mesh tasks: 400 (0-399%50)
total simulation tasks: 800
analysis array: 0-9%10
expected TI.msh outputs: 800
execution: full requested experiment; not a smoke or subset
```

Submit the validated full study:

```bash
bash ti_current_repair/hpc_scripts/submit_final132_right_m1_repeatability.sh
```

This initially submits the 400-element remesh array and its `afterok`
controller. It does not submit a reduced pilot and does not read from or write
to the left-hippocampus experiment root. Monitor it with:

```bash
python ti_current_repair/pipeline/staged_median_fixed_experiment.py status \
  --experiment-root /mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_right_m1
```

## Pending right-M1 tDCS repeatability extension

The former provisional left-M1/C3-Fp2 variability launcher has been removed.
The replacement workflow targets the optimizer-matched right-M1 ROI and
deliberately contains no inferred montage, current, or electrode values. The
runner, preflight, Slurm chain, retry validation, and completion receipts are
implemented, but the repository does not ship a scientific plan. A separately
reviewed plan must be supplied explicitly, keeping execution and submission
fail-closed.

Review the machine-readable study plan locally:

```bash
bash ti_current_repair/hpc_scripts/review_final132_right_m1_tdcs_repeatability.sh \
  --plan /path/to/reviewed-right-m1-tdcs-plan.json
```

The full scope is 1,200 tDCS solves: 400 replays on the existing v4
remesh heads, 400 repeated v4 fixed-head solves, and 400 repeated v3.2.6
deterministic-head solves. The v3 arm is a solver/reproducibility control, not a
v3 remeshing-variability arm. Existing v3 and v4 roots are read-only inputs;
new outputs must use the isolated roots recorded in the reviewed plan.

Downloaded manifests and analysis exports may establish experiment identities
and completed TI task counts, but contain no `.msh` heads. When logged into
Stanage, run the content audit directly there; it writes nothing to either
experiment root and emits its receipt to stdout:

```bash
python ti_current_repair/pipeline/audit_existing_right_m1_meshes.py \
  > right_m1_mesh_audit.json
```

The redirection target should be a user-chosen receipt location. The audit must
pass for all ten subjects before the replay preflight accepts the existing
heads. Exact montage, current, electrode geometry, shape, and conductivity must
also be approved and entered explicitly. Generic tDCS initialization supplies
no defaults for these fields.

Once those two inputs exist, the full read-only source preflight is:

```bash
python ti_current_repair/pipeline/tdcs_replay_workflow.py preflight \
  --plan /path/to/reviewed-right-m1-tdcs-plan.json \
  --audit /path/to/right_m1_mesh_audit.json
```

It verifies all 420 unique source meshes used by the 1,200 tasks, all ten atlas
files, source/output isolation, exact task counts, and any resumable outputs.
`prepare` writes version-specific analysis configs and provenance receipts only
under the new output roots. The candidate Slurm chain preserves the established
Stanage settings: `sheffield`, 8 CPUs, 32 GB, 8 hours, `%50` concurrency,
SimNIBS 4.0.1 then 3.2.6, and one exact-array-element retry. It runs v4
`0-799%50`, validates it, then v3 `0-399%50`, and writes version and combined
completion receipts.

The submission helper prints scope without submitting unless `--submit` is
provided. Even with that flag it refuses to call `sbatch` unless the plan has a
passing audit, no pending stimulation values, `status=ready_for_submission`,
and `submission_allowed=true`:

```bash
bash ti_current_repair/hpc_scripts/submit_final132_right_m1_tdcs_replay.sh \
  --plan /path/to/reviewed-right-m1-tdcs-plan.json \
  --audit /path/to/right_m1_mesh_audit.json
```

### Manual stage-by-stage fallback

The original commands remain available for deliberate recovery or inspection.
They should not be mixed with an active automated chain.

Submit remesh:

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" submit-remesh \
  --experiment-root /mnt/parscratch/users/cop23bi/<run_name> \
  --max-concurrent 50
```

Then run `analyze-remesh`, `select-medians`, and `seed-fixed` in order. Submit
fixed simulations only after all ten seed rows validate:

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" submit-fixed \
  --experiment-root /mnt/parscratch/users/cop23bi/<run_name> \
  --max-concurrent 50
```

After fixed simulations finish, run `analyze-paired` and then build figures:

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" make-figures \
  --experiment-root /mnt/parscratch/users/cop23bi/<run_name>
```

## Outputs

The pipeline writes operational state under:

- `<experiment_root>/_pipeline/`: configs, events, submitted job records,
  status snapshots, median-selection CSV, and fixed seeding manifest.
- `<experiment_root>/_pipeline/workflow/`: automatic-chain submission record,
  job-ID ledger, step receipts, controller logs, and final completion receipt.
- `<experiment_root>/_analysis/`: per-subject report outputs plus paired
  summary CSV/JSON.
- `<experiment_root>/_figures/presentation/`: generated figures and tables.

Fixed-mesh seeding uses physical copies only. The seeder excludes previous
`SimNIBS/` outputs, mesh locks, temp files, and done markers; it writes
`.mesh_ready.json` and fails if symlinks remain in seeded anatomy workspaces.

Initialization requires one atlas per subject at
`<atlas-dir>/<subject>.nii.gz`. It validates every exact filename before
writing configs or allowing Slurm submissions.

## Pre-HPC Smoke Checks

```bash
PYTHONDONTWRITEBYTECODE=1 pytest "$CURRENT_REPAIR_DIR/tests/test_staged_current_repair_pipeline.py" -q -p no:cacheprovider
mkdir -p /tmp/atlases
touch /tmp/atlases/sub-01.nii.gz
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" init --dry-run \
  --source-root /tmp/source \
  --experiment-root /tmp/current-repair-smoke \
  --subjects sub-01 \
  --repeat-count 2 \
  --atlas-dir /tmp/atlases \
  --roi-preset left-hippocampus \
  --montage-preset left-hippocampus \
  --targets-csv "$TARGETS_CSV"
```
