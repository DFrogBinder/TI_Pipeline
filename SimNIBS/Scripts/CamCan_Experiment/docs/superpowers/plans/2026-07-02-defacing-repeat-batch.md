# Defacing Repeat Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible intact-vs-defaced repeat-batch workflow for `sub-CCMe` across `left-hippocampus` and `left-m1`, including local defacing/staging, pure-CHARM simulation support, HPC launch tooling, tests, and end-user docs.

**Architecture:** Extend the simulation path first so it can consume either compressed or uncompressed subject images and run with no custom segmentation map. Then add a defacing/staging CLI that generates a T1-derived face mask, applies it to both T1 and T2, and materializes the four experiment roots. Finally, add a parent-root HPC launcher that submits one manifest row per repeat/subject task through the existing runner and validation path.

**Tech Stack:** Python, nibabel, numpy, subprocess/FSL, existing SimNIBS runners, Slurm shell scripts, pytest.

---

## File Structure

- Modify: `CamCan_Experiment/simulation/TI_runner_multi-core.py`
- Modify: `CamCan_Experiment/simulation/prepare_inplace_rerun.py`
- Modify: `CamCan_Experiment/tests/test_prepare_inplace_rerun.py`
- Create: `CamCan_Experiment/utils/subject_inputs.py`
- Create: `CamCan_Experiment/tests/test_subject_inputs.py`
- Create: `defacing_experiment/prepare_defacing_repeat_batch.py`
- Delete: `defacing_experiment/deface_batch.py`
- Modify: `defacing_experiment/deface_fsl_batch.py`
- Create: `defacing_experiment/README.md`
- Create: `CamCan_Experiment/HPC_scripts/defacing_repeat_array.slurm`
- Create: `CamCan_Experiment/HPC_scripts/submit_defacing_repeat_batch.sh`
- Create: `CamCan_Experiment/tests/test_defacing_repeat_batch.py`
- Create: `CamCan_Experiment/tests/test_defacing_repeat_hpc.py`
- Modify: `CamCan_Experiment/docs/README.md`
- Create: `CamCan_Experiment/docs/DEFACING_EXPERIMENT_TUTORIAL.md`

## Tasks

### Task 1: Add shared subject-input resolution helpers

- [ ] Create a helper module that resolves canonical T1, T2, and optional segmentation paths from `.nii` or `.nii.gz`.
- [ ] Add tests covering compressed, uncompressed, and missing-path cases.
- [ ] Reuse these helpers from simulation and preflight code.

### Task 2: Add pure-CHARM simulation support

- [ ] Update the multi-core runner to use resolved T1/T2 paths instead of hard-coded `.nii`.
- [ ] Detect whether a custom segmentation exists.
- [ ] Preserve current merge/remesh behavior when it exists.
- [ ] Add a no-custom-segmentation branch that skips merge/remesh and uses the CHARM-produced mesh directly.
- [ ] Keep validation and logging behavior intact.

### Task 3: Update repeat-preflight code for optional custom segmentation

- [ ] Adjust preflight input checks so custom segmentation is optional for the defacing experiment path.
- [ ] Keep the current stricter checks available where existing workflows still rely on them.
- [ ] Update tests to cover the new optional-segmentation mode.

### Task 4: Build the defacing/staging CLI

- [ ] Remove the unused brain-only defacing script.
- [ ] Keep the FSL defacing helper as the face-removal backend.
- [ ] Add a new prep CLI that:
  - validates intact T1/T2
  - runs `fsl_deface` on T1 and saves the mask
  - resamples the mask to T2 space and applies it
  - stages the four condition/ROI roots
  - creates 10 repeat datasets per root
  - writes manifests for staged data and generated defaced outputs
- [ ] Add tests for mask resampling, staged naming, and manifest contents.

### Task 5: Add parent-root HPC launch tooling

- [ ] Add a Slurm array task script that reads a manifest row, sets `TI_SIM_ROOT`, and runs the standard simulation runner in full-rerun mode.
- [ ] Add a submit wrapper that discovers all repeat roots under one parent, writes/reads a manifest, chunks large arrays, and passes the required exports.
- [ ] Add source-level tests and shell syntax checks.

### Task 6: Write documentation and usage tutorials

- [ ] Add a focused `defacing_experiment` README for local generation/staging.
- [ ] Add a CamCan experiment tutorial for HPC submission and downstream usage.
- [ ] Update the main CamCan docs to reference the new experiment flow.

### Task 7: Verify end-to-end readiness

- [ ] Run targeted pytest suites for the new helpers, staging logic, and HPC script checks.
- [ ] Run syntax/compile checks on the modified Python modules and shell scripts.
- [ ] Summarize any residual operational assumptions, especially FSL availability and HPC upload expectations.
