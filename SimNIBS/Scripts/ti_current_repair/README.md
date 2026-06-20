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
```

## End-to-End Run

1. Initialize the experiment.

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" init \
  --source-root /mnt/parscratch/users/cop23bi/ti_dataset_balanced_10_corrected \
  --experiment-root /mnt/parscratch/users/cop23bi/current-repair/<run_name> \
  --subjects sub-CC122620,sub-CC222496,sub-CC120120,sub-CC321506,sub-CC410182,sub-CC420075,sub-CC510534,sub-CC520209,sub-CC711128,sub-CC721418 \
  --repeat-count 40 \
  --atlas-dir /mnt/parscratch/users/cop23bi/ZIPs/atlases \
  --roi-preset left-hippocampus
```

2. Submit remesh simulations.

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" submit-remesh \
  --experiment-root /mnt/parscratch/users/cop23bi/current-repair/<run_name> \
  --max-concurrent 50
```

3. After the remesh Slurm array finishes, analyze remesh outputs.

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" analyze-remesh \
  --experiment-root /mnt/parscratch/users/cop23bi/current-repair/<run_name> \
  --max-concurrent 10
```

4. Select the representative median remesh repeat per subject.

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" select-medians \
  --experiment-root /mnt/parscratch/users/cop23bi/current-repair/<run_name> \
  --metric median_roi
```

5. Seed fixed-mesh workspaces using physical copies of the selected remesh
   anatomy and mesh.

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" seed-fixed \
  --experiment-root /mnt/parscratch/users/cop23bi/current-repair/<run_name>
```

6. Submit fixed-mesh simulations.

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" submit-fixed \
  --experiment-root /mnt/parscratch/users/cop23bi/current-repair/<run_name> \
  --max-concurrent 50
```

7. After the fixed Slurm array finishes, run paired analysis.

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" analyze-paired \
  --experiment-root /mnt/parscratch/users/cop23bi/current-repair/<run_name> \
  --max-concurrent 10
```

8. Build presentation figures and summary tables.

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" make-figures \
  --experiment-root /mnt/parscratch/users/cop23bi/current-repair/<run_name>
```

9. Check final status at any point.

```bash
python "$CURRENT_REPAIR_DIR/pipeline/staged_median_fixed_experiment.py" status \
  --experiment-root /mnt/parscratch/users/cop23bi/current-repair/<run_name>
```

## Outputs

The pipeline writes operational state under:

- `<experiment_root>/_pipeline/`: configs, events, submitted job records,
  status snapshots, median-selection CSV, and fixed seeding manifest.
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
  --roi-preset left-hippocampus
```
