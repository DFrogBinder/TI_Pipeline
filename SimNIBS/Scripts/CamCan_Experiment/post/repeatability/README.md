# Repeatability Analysis

This folder contains the across-repeat analysis layer for repeated TI simulation experiments.

## What belongs here

- `analyze_subject_metrics.py`
  Compares `subject_metrics.json` outputs across repeated dataset runs such as
  `Left_Hippocampus_Data_01` to `Left_Hippocampus_Data_10`.
  It performs repeat-level summaries, experiment-level repeatability analysis,
  subject-level variation analysis, figure generation, and optional log-based
  failure auditing.

## How this differs from the rest of `post/`

- `post_process.py`
  Generates per-subject post-processing outputs for one dataset run.

- `post_population.py`
  Aggregates subjects within one dataset run.

- `robustness_analysis.py`
  Computes anatomical and targeting robustness measures within one dataset.

- `run_post_processing_batch.py`
  Runs the post-processing pipeline across repeated dataset folders, but does not
  itself compare repeats statistically.

## Recommended workflow

1. Run `run_post_processing_batch.py` or the standard `post` pipeline to generate
   `subject_metrics.json` for each repeat dataset.
2. Run `repeatability/analyze_subject_metrics.py` on the repeated-dataset root or
   on a single repeat-batch dataset root that contains all repeat folders.
3. Review outputs under `<dataset_root>/subject_metrics_analysis`.

## Typical use

```bash
python3 repeatability/analyze_subject_metrics.py /path/to/Left_Hippocampus_Post_Data
```

If matching execution logs exist, the script can also audit failure patterns and
paired run transitions.
