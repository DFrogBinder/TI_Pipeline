# Mesh Repeatability Analysis

This folder now supports two repeatability experiment styles:

1. `remesh`: every repeat generates a fresh mesh before running TI.
2. `fixed_mesh`: one mesh is generated once per subject and then reused across repeats, so the remaining variation is primarily FEM / solver variation.

The new paired experiment path is designed for your `40 + 40` study:

- `40` remesh repeats per subject
- `40` fixed-mesh repeats per subject
- the same subject list for both conditions
- one paired analysis step that compares the two conditions directly

## What Changed

The old codebase had two separate execution paths:

- `simulation_runners/TI_runner_multi-core_repeat.py`
- `simulation_runners/TI_runner_batch_reuse_mesh.py`

They solved related problems, but they were not a single coherent experiment framework. The new implementation adds a shared configuration model and a unified runner:

- `experiment_config.py`
- `simulation_runners/repeatability_experiment.py`
- `hpc_scripts/repeatability_experiment_array.slurm`
- `hpc_scripts/submit_repeatability_experiment.sh`
- `post/repeatability_experiment_report.py`
- `paired_repeatability_experiment.example.json`

The legacy runners are still present, but the new paired experiment should use the files above.

## Core Idea

For each configured subject, the new workflow creates two condition folders:

- `remesh`
- `fixed_mesh`

Each condition contains its own `repeats/` directory, so the two conditions do not overwrite each other and can be analyzed independently or compared directly.

The condition-specific behavior is:

- `remesh`: each repeat workspace is meshed locally and simulated once.
- `fixed_mesh`: a shared mesh cache is created once for that subject and condition, and every repeat workspace symlinks that same `m2m_<subject>` mesh directory before running the simulation.

That means the only intended difference between the two conditions is the mesh strategy.

Implementation note:

- `fixed_mesh` now uses a filesystem lock around shared mesh-cache creation, so concurrent array tasks for the same subject cannot try to build the same mesh at the same time.
- A mesh is only reused when both the mesh file and the `.mesh_ready.json` marker exist. A stray mesh without that ready marker is treated as incomplete and rebuilt.
- Repeat outputs are still isolated per repeat, so missing fixed-mesh repeats indicate task failures or incomplete runs, not cross-repeat overwrites.
- Recovery submissions skip repeats only when the required analysis inputs already exist (`TI.msh`, base volume, label volume, and `ti_brain_only.nii.gz`), which avoids remeshing completed repeats during resubmission.

## New Experiment Layout

With the new paired runner, outputs are written like this:

```text
<experiment_root>/
  sub-CC110056_repeatability/
    remesh/
      condition_manifest.json
      repeats/
        repeat_001/
          sub-CC110056/
            task_manifest.json
            anat/
              sub-CC110056_T1w.nii
              sub-CC110056_T2w.nii
              sub-CC110056_T1w_ras_1mm_T1andT2_masks.nii
              m2m_sub-CC110056/
              SimNIBS/
                Output/sub-CC110056/TI.msh
                ti_brain_only.nii.gz
        ...
    fixed_mesh/
      condition_manifest.json
      mesh_cache/
        sub-CC110056/
          anat/
            m2m_sub-CC110056/
            sub-CC110056.msh
      repeats/
        repeat_001/
          sub-CC110056/
            task_manifest.json
            anat/
              m2m_sub-CC110056 -> symlink to fixed mesh cache
              SimNIBS/
                Output/sub-CC110056/TI.msh
                ti_brain_only.nii.gz
        ...
  _analysis/
    sub-CC110056/
      remesh/
      fixed_mesh/
      condition_comparison.json
      condition_comparison.md
      condition_metric_std_comparison.png
      condition_metric_cv_comparison.png
      median_roi_by_condition.png
      mesh_nodes_by_condition.png
```

## Configuration

The entire paired experiment is controlled by one JSON file.

Starter file:

- `paired_repeatability_experiment.example.json`

You can also generate a fresh template with:

```bash
python simulation_runners/repeatability_experiment.py \
  write-template-config \
  --output /path/to/my_experiment.json
```

### Required Config Fields

- `source_root`
  - Root containing the original subject folders, for example `/mnt/parscratch/users/cop23bi/repeatability-ti-dataset`
- `experiment_root`
  - Root where the new paired repeatability outputs will be written
- `subjects`
  - Explicit list of subject IDs
- `conditions`
  - List of condition objects

### Condition Fields

Each condition object must include:

- `name`
  - Example: `remesh` or `fixed_mesh`
- `mesh_mode`
  - Must be either `remesh` or `fixed_mesh`
- `repeat_count`
  - Number of repeats for that condition
- `description`
  - Optional but recommended

### Analysis Fields

The optional `analysis` section lets the simulation config also drive the report step:

- `roi_preset`
- `roi_name`
- `roi_labels`
- `atlas_dir`
- `compare_cohort_root`
- `cohort_region_name`
- `cohort_region_label`
- `compare_metric`

For your hippocampus study, the most natural defaults are:

```json
"analysis": {
  "roi_preset": "left-hippocampus",
  "atlas_dir": "/home/boyan/sandbox/Jake_Data/atlases",
  "compare_cohort_root": "/media/boyan/main/PhD/Left_Hippocampus_Data",
  "cohort_region_name": "Left-Hippocampus",
  "cohort_region_label": 17,
  "compare_metric": "median_roi"
}
```

## Recommended Config For The 40 + 40 Experiment

```json
{
  "source_root": "/mnt/parscratch/users/cop23bi/repeatability-ti-dataset",
  "experiment_root": "/mnt/parscratch/users/cop23bi/repeatability-ti-experiment",
  "subjects": [
    "sub-CC110056",
    "sub-CC120120",
    "sub-CC210124",
    "sub-CC310086",
    "sub-CC410243",
    "sub-CC510259",
    "sub-CC610052",
    "sub-CC710214",
    "sub-CC721888",
    "sub-CC810469"
  ],
  "conditions": [
    {
      "name": "remesh",
      "mesh_mode": "remesh",
      "repeat_count": 40,
      "description": "Fresh mesh per repeat."
    },
    {
      "name": "fixed_mesh",
      "mesh_mode": "fixed_mesh",
      "repeat_count": 40,
      "description": "One shared mesh reused across repeats."
    }
  ],
  "analysis": {
    "roi_preset": "left-hippocampus",
    "atlas_dir": "/home/boyan/sandbox/Jake_Data/atlases",
    "compare_cohort_root": "/media/boyan/main/PhD/Left_Hippocampus_Data",
    "cohort_region_name": "Left-Hippocampus",
    "cohort_region_label": 17,
    "compare_metric": "median_roi"
  }
}
```

## Preflight Checks

Before submitting the cluster jobs, inspect the planned task list:

```bash
python simulation_runners/repeatability_experiment.py \
  show-plan \
  --config /path/to/my_experiment.json
```

To print only the total number of tasks:

```bash
python simulation_runners/repeatability_experiment.py \
  show-plan \
  --config /path/to/my_experiment.json \
  --count-only
```

For the `10 subjects x (40 remesh + 40 fixed_mesh)` design, the total should be:

```text
800
```

You can also validate one task without running SimNIBS:

```bash
python simulation_runners/repeatability_experiment.py \
  run-task \
  --config /path/to/my_experiment.json \
  --task-index 0 \
  --dry-run
```

## Running The Experiment

### Recommended Cluster Submission

Use the submit helper:

```bash
bash hpc_scripts/submit_repeatability_experiment.sh \
  /path/to/my_experiment.json \
  20
```

The second argument is the maximum number of concurrent array tasks.

What the helper does:

- reads the JSON config
- computes the task count automatically
- submits `repeatability_experiment_array.slurm`
- passes `EXPERIMENT_CONFIG` and the correct `--array=0-(N-1)%K` value to Slurm

### Direct Slurm Submission

If you prefer manual submission:

```bash
TASK_COUNT=$(python simulation_runners/repeatability_experiment.py \
  show-plan \
  --config /path/to/my_experiment.json \
  --count-only)

sbatch \
  --array="0-$((TASK_COUNT - 1))%20" \
  --export=ALL,EXPERIMENT_CONFIG=/path/to/my_experiment.json \
  hpc_scripts/repeatability_experiment_array.slurm
```

### Useful Environment Overrides

- `PIPELINE_DIR`
- `LOG_DIR`
- `OVERWRITE_OUTPUT=1`
- `FORCE_MESH=1`

Use `OVERWRITE_OUTPUT=1` only when you want to rerun completed repeat workspaces.

Use `FORCE_MESH=1` when:

- you changed the meshing logic
- you changed segmentation preprocessing
- you want to regenerate the shared mesh cache for `fixed_mesh`

## Local / Sequential Execution

For smoke tests:

```bash
python simulation_runners/repeatability_experiment.py \
  run-all \
  --config /path/to/my_experiment.json \
  --max-subjects 1 \
  --dry-run
```

To run all repeats for one subject and one condition sequentially:

```bash
python simulation_runners/repeatability_experiment.py \
  run-condition \
  --config /path/to/my_experiment.json \
  --subject sub-CC721888 \
  --condition fixed_mesh
```

## Analysis

The old single-condition report still exists:

- `post/mesh_repeat_report.py`

The new paired-condition comparison entry point is:

- `post/repeatability_experiment_report.py`

It runs the existing single-condition analysis separately for each condition, then writes a direct condition-comparison layer on top.

### Minimal Paired Analysis

```bash
python post/repeatability_experiment_report.py \
  --config /path/to/my_experiment.json \
  --all-subjects
```

### Single Subject Paired Analysis

```bash
python post/repeatability_experiment_report.py \
  --config /path/to/my_experiment.json \
  --subject sub-CC721888
```

### Override ROI Manually

```bash
python post/repeatability_experiment_report.py \
  --config /path/to/my_experiment.json \
  --all-subjects \
  --roi-name Left-Hippocampus \
  --roi-labels 17
```

### Disable Cohort Comparison

```bash
python post/repeatability_experiment_report.py \
  --config /path/to/my_experiment.json \
  --all-subjects \
  --skip-cohort
```

### Fault-Tolerant Batch Behavior

In batch mode, the paired report now continues past subject-level failures such as:

- missing atlas for one subject
- missing repeat folders for one subject
- bad or incomplete per-subject outputs

Those failures are recorded instead of aborting the whole job.

Batch-level summary files now include:

- `_analysis/paired_condition_summary.json`
- `_analysis/paired_condition_summary.csv`
- `_analysis/paired_condition_failures.json`

Each failed subject records its error type, error message, and traceback in the JSON failure summary.

### Slurm Submission

A dedicated wrapper is available:

- `hpc_scripts/repeatability_experiment_report.slurm`

Submit all subjects with ROI settings taken from the config:

```bash
sbatch \
  --export=ALL,EXPERIMENT_CONFIG=/path/to/my_experiment.json,PIPELINE_DIR=/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/mesh_repeat_analysis \
  hpc_scripts/repeatability_experiment_report.slurm
```

Submit all subjects while setting the ROI explicitly:

```bash
sbatch \
  --export=ALL,EXPERIMENT_CONFIG=/path/to/my_experiment.json,PIPELINE_DIR=/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/mesh_repeat_analysis,ROI_PRESET=left-hippocampus,ATLAS_DIR=/home/boyan/sandbox/Jake_Data/atlases \
  hpc_scripts/repeatability_experiment_report.slurm
```

Useful environment overrides for the Slurm wrapper:

- `SUBJECT_ID`
- `CONDITIONS`
- `MAX_SUBJECTS`
- `OUTPUT_DIR`
- `ROI_PRESET`
- `ROI_NAME`
- `ROI_LABELS`
- `ATLAS_DIR`
- `COMPARE_COHORT_ROOT`
- `COHORT_REGION_NAME`
- `COHORT_REGION_LABEL`
- `COHORT_METRIC`
- `COMPARE_METRIC`
- `REFERENCE_REPEAT`
- `SPATIAL_PERCENTILE`
- `SKIP_COHORT=1`
- `LOG_DIR`
- `LOG_FILE`
- `JSONL_LOG_FILE`

## What The Paired Analysis Produces

For each subject and condition:

- `summary.csv/json`
- `repeatability_stats.csv/json`
- `repeatability_report.md`
- `parameter_consistency.json`
- `label_diff_frequency.nii.gz`
- `roi_mask_on_t1.nii.gz`
- `roi_outline_on_t1.png`
- `roi_outline_on_mean_ti.png`
- `repeat_qc/repeat_###_roi_outline_on_ti.png`
- `median_roi_by_repeat.png`
- `mesh_nodes_by_repeat.png`
- `mesh_nodes_by_repeat_bar.png`

For each subject across conditions:

- `condition_comparison.json`
- `condition_comparison.md`
- `condition_metric_std_comparison.png`
- `condition_metric_cv_comparison.png`
- `median_roi_by_condition.png`
- `mesh_nodes_by_condition.png`

Batch-level outputs:

- `_analysis/paired_condition_summary.json`
- `_analysis/paired_condition_summary.csv`
- `_analysis/paired_condition_failures.json`

## Condition Comparison Logic

The paired report treats one condition as the baseline and one as the comparison:

- baseline defaults to `remesh` when present
- comparison defaults to `fixed_mesh` when present

For each key metric it computes:

- baseline mean / SD / CV
- comparison mean / SD / CV
- `comparison / baseline` SD ratio
- percent reduction in SD from baseline to comparison
- `comparison / baseline` CV ratio
- percent reduction in CV from baseline to comparison

This gives you the exact contrast you asked for:

- E-field variation with new meshes
- E-field variation with the same mesh
- mesh variation only in the remesh condition

## Important Interpretation Notes

In the new experiment:

- `remesh` captures meshing variation plus FEM variation
- `fixed_mesh` should suppress mesh variation and leave mainly FEM / solver variation
- the difference between the two conditions is therefore an estimate of how much variability the remeshing step introduces

Expected behavior:

- `mesh_nodes` variation should be present in `remesh`
- `mesh_nodes` variation should be near zero in `fixed_mesh`
- label-difference metrics should be much smaller in `fixed_mesh`
- ROI and whole-head TI variation should usually be smaller in `fixed_mesh`

If those patterns do not appear, that is an important result and should be investigated rather than ignored.

## ROI Selection

ROI selection is explicit. There is no silent M1 fallback anymore.

Use one of:

- `--roi-preset left-hippocampus`
- `--roi-preset right-hippocampus`
- `--roi-preset left-m1`
- `--roi-name ... --roi-labels ...`

Current preset labels:

- `left-hippocampus` -> `17`
- `right-hippocampus` -> `53`
- `left-m1` -> `1022`

## Notes On Robustness

The new code was written to reduce first-run failure modes:

- one JSON config drives both simulation and analysis
- task enumeration is deterministic
- task manifests are written per repeat
- condition manifests are written per subject-condition
- `fixed_mesh` uses a dedicated shared mesh cache rather than an implicit external mesh
- the paired analysis reuses the already-tested single-condition metric path instead of reimplementing the core calculations

## Legacy Files

These remain in the repository for older workflows:

- `simulation_runners/TI_runner_multi-core_repeat.py`
- `simulation_runners/TI_runner_batch_reuse_mesh.py`
- `hpc_scripts/my_jobArray.slurm`
- `hpc_scripts/batch_reuse_mesh.slurm`

For the new `40 remesh + 40 fixed_mesh` experiment, use the new unified runner instead.
