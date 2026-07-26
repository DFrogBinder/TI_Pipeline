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
