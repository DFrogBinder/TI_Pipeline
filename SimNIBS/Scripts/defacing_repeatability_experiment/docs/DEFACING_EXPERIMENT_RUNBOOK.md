# Defacing Repeatability Experiment Runbook

This pipeline compares SimNIBS outputs from one subject scanned in two
conditions:

- `full_face`: original T1/T2 inputs.
- `defaced`: T1/T2 anonymized with `fsl_deface`.

Each condition regenerates CHARM/SimNIBS outputs independently for every repeat.
Operational state is written under `<experiment-root>/_pipeline/`.

## HPC Usage

Run from the repository root on the cluster:

```bash
export REPO_ROOT=/users/cop23bi/Repos/TI_Pipeline
export DEFACING_DIR="${REPO_ROOT}/SimNIBS/Scripts/defacing_repeatability_experiment"

python "$DEFACING_DIR/pipeline/staged_defacing_experiment.py" init \
  --source-root /path/to/source_subject_root \
  --experiment-root /mnt/parscratch/users/cop23bi/defacing/<run_name> \
  --subject sub-<your_subject_id> \
  --repeat-count-full-face 40 \
  --repeat-count-defaced 40 \
  --roi-preset left-hippocampus

python "$DEFACING_DIR/pipeline/staged_defacing_experiment.py" prepare-inputs \
  --experiment-root /mnt/parscratch/users/cop23bi/defacing/<run_name>

python "$DEFACING_DIR/pipeline/staged_defacing_experiment.py" deface-inputs \
  --experiment-root /mnt/parscratch/users/cop23bi/defacing/<run_name>

python "$DEFACING_DIR/pipeline/staged_defacing_experiment.py" submit-simulations \
  --experiment-root /mnt/parscratch/users/cop23bi/defacing/<run_name> \
  --max-concurrent 50

python "$DEFACING_DIR/pipeline/staged_defacing_experiment.py" analyze \
  --experiment-root /mnt/parscratch/users/cop23bi/defacing/<run_name> \
  --max-concurrent 10

python "$DEFACING_DIR/pipeline/staged_defacing_experiment.py" make-figures \
  --experiment-root /mnt/parscratch/users/cop23bi/defacing/<run_name>

python "$DEFACING_DIR/pipeline/staged_defacing_experiment.py" status \
  --experiment-root /mnt/parscratch/users/cop23bi/defacing/<run_name>
```

## Simulation Runner Hook

The simulation task wrapper always writes:

- `<run-root>/simulation_spec.json`
- `<run-root>/run_simulation_commands.sh`

It regenerates CHARM from the condition-specific T1/T2. The final SimNIBS/TI
command is supplied through either:

- `DEFACING_SIMULATION_RUNNER`
- `simulation.runner_command` in `_pipeline/config.json`

The template can use these variables:

- `${config}`
- `${task_manifest}`
- `${task_index}`
- `${simulation_spec}`
- `${run_root}`
- `${output_dir}`
- `${ti_mesh}`

If no runner is configured, the wrapper exits with status `64` after writing the
spec and command script. This prevents accidentally recording a successful HPC
simulation that never ran.

## State Files

Key files under `<experiment-root>/_pipeline/`:

- `config.json`
- `events.jsonl`
- `stage_status.json`
- `input_manifest.json`
- `defacing_manifest.json`
- `simulation_tasks.json`
- `report_tasks.json`
- `submissions.jsonl`

Analysis and figures are written under:

- `<experiment-root>/_analysis/`
- `<experiment-root>/figures/`

## Smoke Checks

```bash
python -m pytest SimNIBS/Scripts/defacing_repeatability_experiment/tests -q
python "$DEFACING_DIR/pipeline/staged_defacing_experiment.py" init --dry-run \
  --source-root /tmp/source \
  --experiment-root /tmp/defacing-smoke \
  --subject sub-smoke
python "$DEFACING_DIR/pipeline/staged_defacing_experiment.py" status \
  --experiment-root /mnt/parscratch/users/cop23bi/defacing/<run_name>
```
