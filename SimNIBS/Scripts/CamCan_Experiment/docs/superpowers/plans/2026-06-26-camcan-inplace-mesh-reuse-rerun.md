# CamCan In-Place Mesh-Reuse Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a safe in-place CamCan rerun workflow that deletes generated outputs by explicit preflight, reuses existing meshes, and submits one Slurm array per ROI root across all repeats and subjects.

**Architecture:** Add a pure `mesh_reuse.py` helper for testable mesh path resolution, a `prepare_inplace_rerun.py` CLI for manifest generation, cleanup, and validation, a `--reuse-existing-mesh` mode in `TI_runner_multi-core.py`, and two HPC scripts for manifest-driven one-array-per-ROI submission. The runner remains montage-preset driven through `targets.csv`.

**Tech Stack:** Python stdlib, existing CamCan simulation modules, pytest, Bash/Slurm.

---

## File Structure

- Create `CamCan_Experiment/simulation/mesh_reuse.py`: pure helpers for resolving existing mesh paths without importing SimNIBS.
- Create `CamCan_Experiment/simulation/prepare_inplace_rerun.py`: discovery, manifest writing, explicit cleanup, and validation CLI.
- Modify `CamCan_Experiment/simulation/TI_runner_multi-core.py`: add `--reuse-existing-mesh` and use `mesh_reuse.resolve_existing_mesh`.
- Create `CamCan_Experiment/HPC_scripts/camcan_inplace_rerun_array.slurm`: Slurm task runner for one manifest row.
- Create `CamCan_Experiment/HPC_scripts/submit_camcan_inplace_rerun.sh`: one-command ROI submission wrapper.
- Create `CamCan_Experiment/tests/test_mesh_reuse.py`: pure path-resolution tests.
- Create `CamCan_Experiment/tests/test_prepare_inplace_rerun.py`: manifest, cleanup, and validation tests.
- Create `CamCan_Experiment/tests/test_camcan_inplace_rerun_hpc.py`: source-level and shell syntax tests for runner/scripts.

## Task 1: Mesh-Reuse Path Helper

**Files:**
- Create: `CamCan_Experiment/simulation/mesh_reuse.py`
- Test: `CamCan_Experiment/tests/test_mesh_reuse.py`

- [ ] **Step 1: Write failing mesh resolution tests**

```python
from pathlib import Path

from simulation.mesh_reuse import candidate_mesh_paths, resolve_existing_mesh


def test_candidate_mesh_paths_prefers_standard_m2m_subject_layout(tmp_path):
    anat = tmp_path / "sub-CC000001" / "anat"
    paths = candidate_mesh_paths(anat, "sub-CC000001")

    assert paths[0] == anat / "m2m_sub-CC000001" / "sub-CC000001.msh"
    assert anat / "m2m_sub-CC000001" / "sub-CC000001.msh" in paths


def test_resolve_existing_mesh_uses_first_existing_candidate(tmp_path):
    anat = tmp_path / "sub-CC000001" / "anat"
    fallback = anat / "m2m_sub-CC000001" / "sub-CC000001.msh"
    preferred = anat / "m2m_sub-CC000001" / "sub-CC000001.msh"
    fallback.parent.mkdir(parents=True)
    fallback.write_text("mesh", encoding="utf-8")

    assert resolve_existing_mesh(anat, "sub-CC000001") == preferred


def test_resolve_existing_mesh_returns_none_when_missing(tmp_path):
    assert resolve_existing_mesh(tmp_path / "anat", "sub-CC000001") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest CamCan_Experiment/tests/test_mesh_reuse.py -q`

Expected: fails with `ModuleNotFoundError: No module named 'simulation.mesh_reuse'`.

- [ ] **Step 3: Implement helper**

```python
from __future__ import annotations

from pathlib import Path


def subject_suffix(subject: str) -> str:
    return subject.split("-")[-1].upper()


def candidate_mesh_paths(anat_dir: str | Path, subject: str) -> tuple[Path, ...]:
    anat = Path(anat_dir)
    suffix = subject_suffix(subject)
    candidates = [
        anat / f"m2m_{subject}" / f"{subject}.msh",
        anat / f"m2m_sub-{suffix}" / f"{subject}.msh",
        anat / f"m2m_sub-{suffix}" / f"sub-{suffix}.msh",
    ]
    unique = []
    seen = set()
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return tuple(unique)


def resolve_existing_mesh(anat_dir: str | Path, subject: str) -> Path | None:
    for path in candidate_mesh_paths(anat_dir, subject):
        if path.is_file():
            return path
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest CamCan_Experiment/tests/test_mesh_reuse.py -q`

Expected: all tests pass.

## Task 2: Preflight Manifest, Cleanup, and Validation

**Files:**
- Create: `CamCan_Experiment/simulation/prepare_inplace_rerun.py`
- Test: `CamCan_Experiment/tests/test_prepare_inplace_rerun.py`

- [ ] **Step 1: Write failing preflight tests**

Add tests that:

```python
from pathlib import Path

from simulation.prepare_inplace_rerun import (
    discover_repeat_datasets,
    read_tsv,
    run_preflight,
    validate_manifest,
)


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _seed_subject(dataset: Path, subject: str, *, mesh: bool = True) -> None:
    anat = dataset / subject / "anat"
    _write(anat / f"{subject}_T1w.nii")
    _write(anat / f"{subject}_T2w.nii")
    _write(anat / f"{subject}_T1w_ras_1mm_T1andT2_masks.nii")
    if mesh:
        _write(anat / f"m2m_{subject}" / f"{subject}.msh", "mesh")


def test_discover_repeat_datasets_sorts_numeric_repeat_ids(tmp_path):
    for name in ("Left_M1_Data_10", "Left_M1_Data_01", "notes"):
        (tmp_path / name).mkdir()

    datasets = discover_repeat_datasets(tmp_path)

    assert [item.name for item in datasets] == ["Left_M1_Data_01", "Left_M1_Data_10"]


def test_preflight_dry_run_writes_ready_and_blocked_rows_without_deleting(tmp_path):
    run_01 = tmp_path / "Left_M1_Data_01"
    run_02 = tmp_path / "Left_M1_Data_02"
    _seed_subject(run_01, "sub-01")
    _seed_subject(run_02, "sub-01", mesh=False)
    _write(run_01 / "sub-01" / "anat" / "SimNIBS" / "old.txt")

    manifest = tmp_path / "manifest.tsv"
    cleanup = tmp_path / "cleanup.tsv"
    result = run_preflight(tmp_path, manifest=manifest, cleanup_manifest=cleanup, apply=False)

    rows = read_tsv(manifest)
    assert result["ready_tasks"] == 1
    assert result["blocked_tasks"] == 1
    assert [row["status"] for row in rows] == ["ready", "blocked"]
    assert (run_01 / "sub-01" / "anat" / "SimNIBS" / "old.txt").is_file()
    assert any(row["action"] == "would_delete" for row in read_tsv(cleanup))


def test_preflight_apply_deletes_generated_outputs_and_preserves_inputs(tmp_path):
    run_01 = tmp_path / "Left_M1_Data_01"
    _seed_subject(run_01, "sub-01")
    anat = run_01 / "sub-01" / "anat"
    _write(anat / "SimNIBS" / "old.txt")
    _write(anat / "post" / "old.txt")
    _write(anat / "skin_mask.nii.gz")

    run_preflight(tmp_path, manifest=tmp_path / "manifest.tsv", cleanup_manifest=tmp_path / "cleanup.tsv", apply=True)

    assert not (anat / "SimNIBS").exists()
    assert not (anat / "post").exists()
    assert not (anat / "skin_mask.nii.gz").exists()
    assert (anat / "m2m_sub-01" / "sub-01.msh").is_file()
    assert (anat / "sub-01_T1w.nii").is_file()


def test_validate_manifest_reports_complete_and_incomplete_rows(tmp_path):
    run_01 = tmp_path / "Left_M1_Data_01"
    _seed_subject(run_01, "sub-01")
    output = run_01 / "sub-01" / "anat" / "SimNIBS"
    _write(output / "Output" / "sub-01" / "TI.msh")
    _write(output / "Output" / "sub-01" / "Volume_Base" / "TI_Volumetric_Base.nii.gz")
    _write(output / "Output" / "sub-01" / "Volume_Labels" / "TI_Volumetric_Labels.nii.gz")
    _write(output / "ti_brain_only.nii.gz")
    manifest = tmp_path / "manifest.tsv"
    validation = tmp_path / "validation.tsv"
    run_preflight(tmp_path, manifest=manifest, cleanup_manifest=tmp_path / "cleanup.tsv", apply=False)

    result = validate_manifest(manifest, summary_path=validation, check_nifti=False)

    assert result["complete_tasks"] == 1
    assert read_tsv(validation)[0]["status"] == "complete"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest CamCan_Experiment/tests/test_prepare_inplace_rerun.py -q`

Expected: fails with `ModuleNotFoundError: No module named 'simulation.prepare_inplace_rerun'`.

- [ ] **Step 3: Implement the CLI and pure functions**

Implement dataclasses `RepeatDataset` and `TaskRow`, functions
`discover_repeat_datasets()`, `run_preflight()`, `write_tsv()`, `read_tsv()`,
`cleanup_generated_outputs()`, and `validate_manifest()`. The CLI exposes
`preflight` and `validate` subcommands.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest CamCan_Experiment/tests/test_prepare_inplace_rerun.py -q`

Expected: all tests pass.

## Task 3: Runner Mesh-Reuse Mode

**Files:**
- Modify: `CamCan_Experiment/simulation/TI_runner_multi-core.py`
- Test: `CamCan_Experiment/tests/test_camcan_inplace_rerun_hpc.py`

- [ ] **Step 1: Write failing source-level runner tests**

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_multi_core_runner_exposes_reuse_existing_mesh_flag():
    source = (ROOT / "simulation" / "TI_runner_multi-core.py").read_text(encoding="utf-8")

    assert "--reuse-existing-mesh" in source
    assert "resolve_existing_mesh" in source
    assert "reuse_existing_mesh" in source
    assert "mesh_reuse_enabled" in source


def test_inplace_rerun_slurm_scripts_have_expected_entrypoints():
    array_script = ROOT / "HPC_scripts" / "camcan_inplace_rerun_array.slurm"
    submit_script = ROOT / "HPC_scripts" / "submit_camcan_inplace_rerun.sh"

    assert array_script.is_file()
    assert submit_script.is_file()
```

- [ ] **Step 2: Run test to verify runner flag assertion fails**

Run: `pytest CamCan_Experiment/tests/test_camcan_inplace_rerun_hpc.py::test_multi_core_runner_exposes_reuse_existing_mesh_flag -q`

Expected: fails because the runner does not yet contain `--reuse-existing-mesh`.

- [ ] **Step 3: Modify runner minimally**

Add:

```python
from simulation.mesh_reuse import resolve_existing_mesh
```

and add a parser flag:

```python
parser.add_argument(
    "--reuse-existing-mesh",
    action="store_true",
    help="Skip CHARM/remeshing and run simulations using an existing m2m mesh.",
)
```

Thread the value into a module-global `REUSE_EXISTING_MESH`, log
`mesh_reuse_enabled`, and in `process_subject()` resolve `fnamehead` from
`resolve_existing_mesh(subject_dir, subject)` before meshing. If reuse is true
and no mesh exists, raise `SimulationInputError`. If reuse is true, skip the
entire CHARM/remesh block and do not call `cleanup_subject_mesh_outputs()`.

- [ ] **Step 4: Run source-level runner test**

Run: `pytest CamCan_Experiment/tests/test_camcan_inplace_rerun_hpc.py::test_multi_core_runner_exposes_reuse_existing_mesh_flag -q`

Expected: test passes.

## Task 4: Slurm Array and Submit Wrapper

**Files:**
- Create: `CamCan_Experiment/HPC_scripts/camcan_inplace_rerun_array.slurm`
- Create: `CamCan_Experiment/HPC_scripts/submit_camcan_inplace_rerun.sh`
- Test: `CamCan_Experiment/tests/test_camcan_inplace_rerun_hpc.py`

- [ ] **Step 1: Extend failing tests for script syntax and tokens**

```python
import subprocess


def test_inplace_rerun_shell_scripts_pass_bash_syntax_check():
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "HPC_scripts/camcan_inplace_rerun_array.slurm",
        "HPC_scripts/submit_camcan_inplace_rerun.sh",
    ):
        result = subprocess.run(["bash", "-n", str(root / relative)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_inplace_rerun_scripts_use_manifest_and_reuse_mesh_flag():
    root = Path(__file__).resolve().parents[1]
    array_source = (root / "HPC_scripts" / "camcan_inplace_rerun_array.slurm").read_text(encoding="utf-8")
    submit_source = (root / "HPC_scripts" / "submit_camcan_inplace_rerun.sh").read_text(encoding="utf-8")

    assert "TI_INPLACE_RERUN_MANIFEST" in array_source
    assert "--reuse-existing-mesh" in array_source
    assert "TASK_OFFSET" in array_source
    assert "MONTAGE_PRESET" in submit_source
    assert "MAX_ARRAY_TASKS" in submit_source
```

- [ ] **Step 2: Run tests to verify they fail before scripts exist**

Run: `pytest CamCan_Experiment/tests/test_camcan_inplace_rerun_hpc.py -q`

Expected: failures for missing scripts.

- [ ] **Step 3: Implement scripts**

Use existing `my_jobArray.slurm` and `submit_current_repair_jobArray.sh` patterns:

- array script reads manifest line `TASK_OFFSET + SLURM_ARRAY_TASK_ID + 2`;
- skips non-`ready` manifest rows;
- sets `TI_SIM_ROOT="$DATASET_ROOT"`;
- calls `TI_runner_multi-core.py --subject "$SUBJECT" --montage-preset "$TI_MONTAGE_PRESET" --reuse-existing-mesh`;
- validates with `validate_simulation_outputs.py`;
- requeues incomplete ready tasks using retry files;
- submit wrapper counts manifest data rows and chunks with `MAX_ARRAY_TASKS`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest CamCan_Experiment/tests/test_camcan_inplace_rerun_hpc.py -q`

Expected: all tests pass.

## Task 5: Focused Regression Suite and Final Review

**Files:**
- All files above.

- [ ] **Step 1: Run focused CamCan tests**

Run:

```bash
pytest \
  CamCan_Experiment/tests/test_mesh_reuse.py \
  CamCan_Experiment/tests/test_prepare_inplace_rerun.py \
  CamCan_Experiment/tests/test_camcan_inplace_rerun_hpc.py \
  CamCan_Experiment/tests/test_validate_simulation_outputs.py \
  CamCan_Experiment/simulation/test_target_montages.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run syntax checks**

Run:

```bash
python -m py_compile \
  CamCan_Experiment/simulation/mesh_reuse.py \
  CamCan_Experiment/simulation/prepare_inplace_rerun.py
bash -n CamCan_Experiment/HPC_scripts/camcan_inplace_rerun_array.slurm
bash -n CamCan_Experiment/HPC_scripts/submit_camcan_inplace_rerun.sh
```

Expected: all commands exit 0.

- [ ] **Step 3: Review git diff**

Run: `git diff --stat && git diff --check`

Expected: diff is scoped to the rerun workflow, no whitespace errors.

- [ ] **Step 4: Commit implementation files only**

Run:

```bash
git add \
  CamCan_Experiment/simulation/mesh_reuse.py \
  CamCan_Experiment/simulation/prepare_inplace_rerun.py \
  CamCan_Experiment/simulation/TI_runner_multi-core.py \
  CamCan_Experiment/HPC_scripts/camcan_inplace_rerun_array.slurm \
  CamCan_Experiment/HPC_scripts/submit_camcan_inplace_rerun.sh \
  CamCan_Experiment/tests/test_mesh_reuse.py \
  CamCan_Experiment/tests/test_prepare_inplace_rerun.py \
  CamCan_Experiment/tests/test_camcan_inplace_rerun_hpc.py \
  CamCan_Experiment/docs/superpowers/plans/2026-06-26-camcan-inplace-mesh-reuse-rerun.md
git commit -m "feat: add CamCan in-place mesh reuse rerun workflow"
```

Expected: commit succeeds without staging `utils/targets.csv` or backup files.
