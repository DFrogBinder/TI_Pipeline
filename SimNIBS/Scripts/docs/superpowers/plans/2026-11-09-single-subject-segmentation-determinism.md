# Single-subject Segmentation Determinism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated, fail-closed Stanage workflow that runs ten SimNIBS 3.2.6 headreco segmentations and ten SimNIBS 4.0.1 CHARM segmentations of the same subject and reports within-version voxel determinism.

**Architecture:** A new `simnibs_segmentation_determinism` Python package owns configuration, input provenance, repeat runners, receipt validation, NIfTI analysis and CLI operations. Thin Slurm scripts load the pinned modules and call those interfaces; a shell submitter provides read-only preflight, initialization, complete two-arm submission and status. Existing v3 compatibility and CHARM runner patterns are reused without changing their production defaults.

**Tech Stack:** Python 3.9-compatible standard library, NumPy, NiBabel, pytest, Bash, Slurm, SimNIBS 3.2.6 plus MATLAB R2023b, SimNIBS 4.0.1.

**Spec:** `docs/superpowers/specs/2026-11-09-single-subject-segmentation-determinism-design.md`

## Global Constraints

- Subject is `sub-CC110174`, selected by `random.Random(20261106).choice(subjects)` from the pinned balanced-ten list.
- Source root is read-only `/mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_corrected`.
- Output root defaults to `/mnt/parscratch/users/cop23bi/simnibs_segmentation_determinism_v1`.
- Each arm has ten repeats; scientific task count is 20 and scheduler task count is 23.
- Every repeat passes the identical canonical subject ID; only the isolated parent directory and receipt identify the repeat.
- V3 runs `headreco preparevols --noclean` only; v4 runs CHARM registration, atlas initialization and segmentation only.
- Surface preparation, cleaning, meshing, electrode placement and FEM are forbidden.
- Existing repeatability campaigns, corrected label roots and consolidated datasets are protected read-only.
- An incomplete or failed task never receives a completion receipt.
- Probability thresholds `1e-7`, `1e-6`, and `1e-5` are descriptive; exact array equality defines bitwise determinism.
- No HPC jobs are submitted by tests or by `scope`, `preflight`, `init`, `status`, or `analyze`.

## Review Focus

- Ambiguous `.nii`/`.nii.gz` source pairs must fail rather than select one silently; pin in Task 1 tests.
- Repeat workspaces must not change the canonical subject ID or reuse another repeat's output; pin in Tasks 2 and 3 tests.
- Forbidden downstream headreco/CHARM stages must be rejected before a receipt is written; pin in Tasks 2, 3 and 6 tests.
- Header-only, gzip-only and voxel-content differences must be distinguished; pin in Task 4 tests.
- Partial arrays and retried elements must produce an incomplete audit rather than a terminal success receipt; pin in Tasks 5 and 6 tests.

---

### Task 1: Package settings, scope and read-only preflight

**Files:**
- Create: `simnibs_segmentation_determinism/__init__.py`
- Create: `simnibs_segmentation_determinism/settings.py`
- Create: `simnibs_segmentation_determinism/provenance.py`
- Create: `simnibs_segmentation_determinism/tests/conftest.py`
- Create: `simnibs_segmentation_determinism/tests/test_settings.py`
- Create: `simnibs_segmentation_determinism/tests/test_preflight.py`

**Interfaces:**
- Produces: `ExperimentConfig`, `SubjectInputs`, `scope_payload(config)`, `resolve_subject_inputs(config)`, `preflight(config)` and atomic JSON/hash helpers.
- Consumes: no new package interfaces.

- [ ] **Step 1: Write failing settings and preflight tests**

Test that defaults contain the exact subject, seed, modules, roots, repeat count and scope counts; the pinned cohort hashes validate; seeded selection returns `sub-CC110174`; one T1/T2 pair resolves; ambiguous extensions, missing files, cohort-hash drift and protected-root collisions fail; preflight leaves an absent output root absent.

- [ ] **Step 2: Run the focused tests and confirm they fail because the package is absent**

Run: `./simnibs_env/bin/python -m pytest simnibs_segmentation_determinism/tests/test_settings.py simnibs_segmentation_determinism/tests/test_preflight.py -q`

Expected: collection/import failure for `simnibs_segmentation_determinism`.

- [ ] **Step 3: Implement configuration and provenance interfaces**

Define immutable dataclasses:

```python
@dataclass(frozen=True)
class ExperimentConfig: ...

@dataclass(frozen=True)
class SubjectInputs:
    subject: str
    t1: Path
    t2: Path
    t1_sha256: str
    t2_sha256: str
```

Implement `ExperimentConfig.from_env()`, `scope_payload(config) -> dict[str, object]`, `resolve_subject_inputs(config) -> SubjectInputs`, and `preflight(config) -> dict[str, object]`. Use exact path containment checks rather than string prefixes. Keep preflight read-only.

- [ ] **Step 4: Run focused tests and verify they pass**

Run: `./simnibs_env/bin/python -m pytest simnibs_segmentation_determinism/tests/test_settings.py simnibs_segmentation_determinism/tests/test_preflight.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the package foundation**

```bash
git add simnibs_segmentation_determinism/__init__.py simnibs_segmentation_determinism/settings.py simnibs_segmentation_determinism/provenance.py simnibs_segmentation_determinism/tests
git commit -m "feat: add segmentation determinism preflight"
```

### Task 2: SimNIBS 3.2.6 segmentation-only repeat runner

**Files:**
- Create: `simnibs_segmentation_determinism/headreco_runner.py`
- Create: `simnibs_segmentation_determinism/tests/test_headreco_runner.py`
- Read/reuse: `simnibs_326_repeatability/cat12_compat.py`
- Read/reuse: `simnibs_326_repeatability/hpc/activate_compat.sh`

**Interfaces:**
- Consumes: `ExperimentConfig`, `SubjectInputs`, atomic JSON/hash helpers.
- Produces: `HeadrecoRepeatRequest`, `run_headreco_repeat(request) -> dict[str, object]`, `validate_headreco_repeat(request) -> dict[str, object]`.

- [ ] **Step 1: Write failing runner tests with a fake headreco process**

Assert that the command is exactly the compatibility wrapper followed by `headreco preparevols --noclean sub-CC110174 T1 T2`; ten requests retain the same subject ID but different repeat roots; required SPM/CAT probability and binary patterns are promoted atomically; input hashes are checked; missing artifacts, unexpected artifact inventories, forbidden-stage log text and nonzero subprocess exits leave no completion receipt.

- [ ] **Step 2: Run the headreco runner tests and confirm the expected import failure**

Run: `./simnibs_env/bin/python -m pytest simnibs_segmentation_determinism/tests/test_headreco_runner.py -q`

Expected: FAIL because `headreco_runner` is absent.

- [ ] **Step 3: Implement the request-driven v3 runner**

Define:

```python
@dataclass(frozen=True)
class HeadrecoRepeatRequest: ...

def run_headreco_repeat(request: HeadrecoRepeatRequest) -> dict[str, object]: ...
def validate_headreco_repeat(request: HeadrecoRepeatRequest) -> dict[str, object]: ...
```

Execute in a fresh repeat-local temporary directory, call the existing CAT12 wrapper, discover only the pinned `preparevols` NIfTI patterns, copy them with relative paths into `artifacts/`, record canonical subject/module/input/command/artifact hashes, scan the captured command/log for forbidden modes, and atomically publish the receipt last.

- [ ] **Step 4: Run the headreco runner tests and the existing compatibility tests**

Run: `./simnibs_env/bin/python -m pytest simnibs_segmentation_determinism/tests/test_headreco_runner.py simnibs_326_repeatability/tests/test_simnibs_326_pipeline.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the v3 repeat runner**

```bash
git add simnibs_segmentation_determinism/headreco_runner.py simnibs_segmentation_determinism/tests/test_headreco_runner.py
git commit -m "feat: add headreco segmentation repeat runner"
```

### Task 3: SimNIBS 4.0.1 CHARM segmentation-only repeat runner

**Files:**
- Create: `simnibs_segmentation_determinism/charm_runner.py`
- Create: `simnibs_segmentation_determinism/tests/test_charm_runner.py`
- Read/reuse: `CamCan_Experiment/charm_segmentation_batch/run_charm_segmentation.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, `SubjectInputs`, atomic JSON/hash helpers.
- Produces: `CharmRepeatRequest`, `run_charm_repeat(request) -> dict[str, object]`, `validate_charm_repeat(request) -> dict[str, object]`.

- [ ] **Step 1: Write failing CHARM runner tests**

Assert that every command uses `charm sub-CC110174 T1 T2 --registerT2 --initatlas --segment --forceqform`; repeat roots differ while subject ID does not; `--surfaces` and `--mesh` never appear; only a valid nonempty `tissue_labeling_upsampled.nii.gz` is promoted; source/destination hashes match; missing or multiple maps, timeouts, forbidden flags and input-hash drift leave no completion receipt.

- [ ] **Step 2: Run the focused tests and verify they fail because the runner is absent**

Run: `./simnibs_env/bin/python -m pytest simnibs_segmentation_determinism/tests/test_charm_runner.py -q`

Expected: FAIL on import.

- [ ] **Step 3: Implement the isolated CHARM runner**

Define the request dataclass and the two run/validate functions. Reuse the established process-group timeout and atomic-copy semantics, but keep all work and receipts under the new experiment root. Delete only task-local temporary work after successful promotion.

- [ ] **Step 4: Run new and established CHARM tests**

Run: `./simnibs_env/bin/python -m pytest simnibs_segmentation_determinism/tests/test_charm_runner.py CamCan_Experiment/charm_segmentation_batch -q`

Expected: PASS or, if the established directory contains no pytest files, the new tests pass with no regression collection errors.

- [ ] **Step 5: Commit the v4 repeat runner**

```bash
git add simnibs_segmentation_determinism/charm_runner.py simnibs_segmentation_determinism/tests/test_charm_runner.py
git commit -m "feat: add CHARM segmentation repeat runner"
```

### Task 4: NIfTI determinism analysis

**Files:**
- Create: `simnibs_segmentation_determinism/analysis.py`
- Create: `simnibs_segmentation_determinism/tests/test_analysis.py`

**Interfaces:**
- Consumes: validated artifact manifests from Tasks 2 and 3.
- Produces: `inspect_nifti(path) -> NiftiFingerprint`, `compare_niftis(reference, candidate, kind) -> NiftiComparison`, `analyze_arm(...) -> dict[str, object]`.

- [ ] **Step 1: Write synthetic-NIfTI failing tests**

Create images proving: identical arrays with different gzip bytes remain voxel-identical; affine-only changes fail geometric determinism; one label transition changes counts and Dice; NaNs compare in stable positions; floating differences are counted separately above `1e-7`, `1e-6`, and `1e-5`; an artifact differing from repeat 001 triggers all 45 pairwise comparisons.

- [ ] **Step 2: Run analysis tests and confirm the import failure**

Run: `./simnibs_env/bin/python -m pytest simnibs_segmentation_determinism/tests/test_analysis.py -q`

Expected: FAIL because `analysis` is absent.

- [ ] **Step 3: Implement normalized image inspection and comparison**

Define frozen result dataclasses. Hash canonical C-contiguous unscaled arrays with dtype/shape included, compare qform/sform/effective affine explicitly, classify integer/binary maps as discrete, and use zero relative tolerance for fixed absolute thresholds. Produce transition and Dice records only for discrete maps.

- [ ] **Step 4: Run analysis tests**

Run: `./simnibs_env/bin/python -m pytest simnibs_segmentation_determinism/tests/test_analysis.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the analysis engine**

```bash
git add simnibs_segmentation_determinism/analysis.py simnibs_segmentation_determinism/tests/test_analysis.py
git commit -m "feat: analyze segmentation determinism"
```

### Task 5: Collector, receipts and command-line workflow

**Files:**
- Create: `simnibs_segmentation_determinism/workflow.py`
- Create: `simnibs_segmentation_determinism/cli.py`
- Create: `simnibs_segmentation_determinism/__main__.py`
- Create: `simnibs_segmentation_determinism/tests/test_workflow.py`
- Create: `simnibs_segmentation_determinism/tests/test_cli.py`

**Interfaces:**
- Consumes: settings/preflight, repeat validators and analysis interfaces.
- Produces: CLI commands `scope`, `preflight`, `init`, `run-repeat`, `collect`, `status`, and stable JSON receipts.

- [ ] **Step 1: Write failing workflow and CLI tests**

Assert exact scope JSON; preflight is read-only; init is idempotent only for a matching plan; `run-repeat` maps arm/index to repeat 001-010; collect with 19/20 valid receipts writes an incomplete audit and no terminal receipt; retry metadata is preserved; 20/20 valid receipts produce the tabular outputs, report, determinism manifest and terminal receipt; status distinguishes uninitialized, prepared, in-progress, incomplete and complete.

- [ ] **Step 2: Run workflow tests and verify the import failure**

Run: `./simnibs_env/bin/python -m pytest simnibs_segmentation_determinism/tests/test_workflow.py simnibs_segmentation_determinism/tests/test_cli.py -q`

Expected: FAIL because workflow/CLI modules are absent.

- [ ] **Step 3: Implement workflow state transitions and CLI parsing**

All JSON output uses atomic writes and stable sorted serialization. `init` stores the plan hash and source manifest. `collect` validates receipts rather than trusting directory counts. `status` is read-only.

- [ ] **Step 4: Run workflow, CLI and analysis tests**

Run: `./simnibs_env/bin/python -m pytest simnibs_segmentation_determinism/tests/test_workflow.py simnibs_segmentation_determinism/tests/test_cli.py simnibs_segmentation_determinism/tests/test_analysis.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the workflow**

```bash
git add simnibs_segmentation_determinism/workflow.py simnibs_segmentation_determinism/cli.py simnibs_segmentation_determinism/__main__.py simnibs_segmentation_determinism/tests/test_workflow.py simnibs_segmentation_determinism/tests/test_cli.py
git commit -m "feat: add segmentation determinism workflow"
```

### Task 6: Slurm launchers and guarded submitter

**Files:**
- Create: `simnibs_segmentation_determinism/hpc/module_preflight_326.slurm`
- Create: `simnibs_segmentation_determinism/hpc/module_preflight_401.slurm`
- Create: `simnibs_segmentation_determinism/hpc/headreco_array.slurm`
- Create: `simnibs_segmentation_determinism/hpc/charm_array.slurm`
- Create: `simnibs_segmentation_determinism/hpc/collect.slurm`
- Create: `simnibs_segmentation_determinism/hpc/submit.sh`
- Create: `simnibs_segmentation_determinism/tests/test_hpc_launchers.py`

**Interfaces:**
- Consumes: package CLI and pinned module/resource constants.
- Produces: operational commands `scope`, `preflight`, `init`, `run`, `status`, and `analyze`.

- [ ] **Step 1: Write static and fake-sbatch launcher tests**

Assert exact `0-9%10` arrays, 8 CPU/32G/08:00/sheffield resources, append-only logs, exact-element retry bounded at one, module pins, v3 compatibility activation, dependency graph, 23 scheduler tasks in scope, `afterany` collector, no `headreco all|preparecat|cleanvols|surfacemesh|volumemesh`, no CHARM `--surface|--mesh`, and zero `sbatch` calls for every command except `run`.

- [ ] **Step 2: Run launcher tests and confirm files are absent**

Run: `./simnibs_env/bin/python -m pytest simnibs_segmentation_determinism/tests/test_hpc_launchers.py -q`

Expected: FAIL on missing launchers.

- [ ] **Step 3: Implement thin Slurm scripts and the fail-closed submitter**

Load modules only in Slurm scripts, pass explicit environment paths, use parseable job IDs, append all IDs to `job_ids.tsv`, preserve attempt-specific logs, and submit no array when its module-preflight submission fails. The submitter prints the complete numerical scope before actions.

- [ ] **Step 4: Verify Python tests and shell quality**

Run:

```bash
./simnibs_env/bin/python -m pytest simnibs_segmentation_determinism/tests/test_hpc_launchers.py -q
bash -n simnibs_segmentation_determinism/hpc/submit.sh simnibs_segmentation_determinism/hpc/*.slurm
shellcheck simnibs_segmentation_determinism/hpc/submit.sh simnibs_segmentation_determinism/hpc/*.slurm
```

Expected: tests pass; Bash and ShellCheck return zero.

- [ ] **Step 5: Commit the HPC interface**

```bash
git add simnibs_segmentation_determinism/hpc simnibs_segmentation_determinism/tests/test_hpc_launchers.py
git commit -m "feat: add segmentation determinism HPC launchers"
```

### Task 7: End-to-end validation and operator documentation

**Files:**
- Create: `simnibs_segmentation_determinism/README.md`
- Create: `simnibs_segmentation_determinism/tests/test_end_to_end.py`
- Modify: `docs/superpowers/specs/2026-11-09-single-subject-segmentation-determinism-design.md` only if implementation reveals a factual mismatch.

**Interfaces:**
- Consumes: all earlier task interfaces.
- Produces: a tested local synthetic workflow and a copy-paste-safe Stanage runbook that stops before submission approval.

- [ ] **Step 1: Write a synthetic end-to-end failing test**

Use fake runner outputs for ten repeats per arm, run initialization and collection, and assert 20 validated receipts, per-arm determinism classifications, expected TSV/report files, source immutability and a terminal receipt whose hashes verify. Add an incomplete case proving one missing repeat cannot complete.

- [ ] **Step 2: Run the end-to-end test and confirm any missing integration fails**

Run: `./simnibs_env/bin/python -m pytest simnibs_segmentation_determinism/tests/test_end_to_end.py -q`

Expected: FAIL until all integration paths are wired.

- [ ] **Step 3: Complete integration and write the README**

Document scientific scope, stage boundaries, output interpretation, candidate resources, preflight/init/run/status/analyze commands, receipt gates, retry semantics and the explicit statement that implementation does not authorize submission. Do not include a command that submits while the dataset checksum remains active.

- [ ] **Step 4: Run full verification**

Run:

```bash
./simnibs_env/bin/python -m pytest simnibs_segmentation_determinism/tests -q
./simnibs_env/bin/python -m pytest simnibs_326_repeatability/tests CamCan_Experiment/charm_segmentation_batch -q
bash -n simnibs_segmentation_determinism/hpc/submit.sh simnibs_segmentation_determinism/hpc/*.slurm
shellcheck simnibs_segmentation_determinism/hpc/submit.sh simnibs_segmentation_determinism/hpc/*.slurm
git diff --check
```

Expected: all tests pass; Bash, ShellCheck and diff checks return zero.

- [ ] **Step 5: Perform a manual safety review**

Confirm the final diff contains no writes to historical roots, no meshing/FEM invocation, no silent subject-ID variation, no unbounded requeue and no submission performed during development.

- [ ] **Step 6: Commit the validated package**

```bash
git add simnibs_segmentation_determinism docs/superpowers/specs/2026-11-09-single-subject-segmentation-determinism-design.md docs/superpowers/plans/2026-11-09-single-subject-segmentation-determinism.md
git commit -m "docs: add segmentation determinism runbook"
```
