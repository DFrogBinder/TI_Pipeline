# SimNIBS 3.2.6 Corrected-Label Repeatability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated, preflight-gated SimNIBS 3.2.6 workflow that projects the existing corrected v4 label maps into one retained v3-compatible map per participant, creates genuinely new v3 surfaces and tetrahedral meshes, and re-runs the complete balanced-10 repeatability experiment without rerunning SPM/CAT12.

**Architecture:** A new package owns configuration, corrected-label harmonization, allowlisted staging of legacy coordinate references, a version-pinned v3 meshing adapter, head-model QC/promotion, and HPC orchestration. The established v3 simulation and repeatability machinery is copied into the new package and changed only at its head-model boundary, while the completed v3/v4 packages and data roots remain read only.

**Tech Stack:** Python 3, NumPy, SciPy, NiBabel, SimNIBS 3.2.6 internal meshing APIs, pytest, Bash/Slurm, ShellCheck, Ruff, and the existing `ti-post` interpreter for Matplotlib QC rendering.

**Spec:** `SimNIBS/Scripts/docs/superpowers/specs/2026-10-22-simnibs326-corrected-label-repeatability-design.md`

## Global Constraints

- Work only on `feature/simnibs326-corrected-label-pipeline` in the existing isolated worktree.
- Add `simnibs_326_corrected_label_repeatability/`; do not change the completed `simnibs_326_repeatability/` behavior.
- Never copy or modify the corrected source label maps; record their absolute paths and SHA-256 hashes.
- Retain exactly one harmonized categorical NIfTI per participant on the staged `T1fs_conform` grid; source-grid conversions and input masks are temporary.
- Accept source labels `0,1,2,3,5,6,7,8,9,10` and optional `11`; reject every other label, including source label 4.
- Use physical-mm nearest-tissue assignment for blood with priority `WM, GM, CSF, external soft tissue` on exact ties.
- Generate new surfaces and meshes with SimNIBS 3.2.6; reuse no old tissue mask, surface, or mesh.
- Use the existing v3 reference scaffold only for the explicit coordinate/reference allowlist and immutable provenance.
- Preserve the full scope: 10 participants x 2 targets x 2 conditions x 40 repeats = 1,600 simulations and 3,200 component FEM solves.
- Preserve the working simulation profile: `SimNIBS/3.2.6-foss-2023a`, `sheffield`, 8 CPUs, 32 GB, 8 hours, `%50`, one retry, append-only Slurm output, immutable attempt logs, and element-qualified requeue.
- Deliberate operational delta: do not load MATLAB or invoke SPM/CAT12 in the new workflow; fail the v3 API probe before scientific arrays if the internal meshing API differs.
- Do not submit any HPC job during implementation or local verification.
- A completed head-model collection and a content-addressed human-review promotion receipt are mandatory before simulation submission.

## File Structure

```text
simnibs_326_corrected_label_repeatability/
  __init__.py                 package identity
  __main__.py                 `python -m` entry point
  settings.py                 pinned subjects, roots, resources, and path isolation
  labels.py                   pure label projection and blood assignment
  label_io.py                 NIfTI validation, categorical resampling, retained map, temporary masks
  reference_stage.py          legacy allowlist staging and checksum receipts
  v3_meshing.py               version probe, surface generation, volume meshing, mesh validation
  headmodels.py               per-subject build, collection, QC manifest, promotion
  render_qc.py                deferred Matplotlib renderer for the ti-post interpreter
  pipeline.py                 campaign configuration, receipts, status, and CLI dispatch
  simulation_runner.py        v3 FEM adapter using the new head-model manifest
  operational_profile.json   inherited working settings plus unverified new-stage deltas
  README.md                   safe HPC handoff and interpretation boundary
  hpc/
    activate_compat.sh
    module_preflight.slurm
    reference_stage_array.slurm
    headmodel_array.slurm
    headmodel_collect.slurm
    simulation_array.slurm
    control.slurm
    optimizer_roi_array.slurm
    optimizer_roi_collect.slurm
    analyze_optimizer_roi.slurm
    submit.sh
  tests/
    conftest.py
    test_settings.py
    test_labels.py
    test_label_io.py
    test_reference_stage.py
    test_v3_meshing.py
    test_headmodels.py
    test_pipeline.py
    test_simulation_runner.py
    test_hpc_launchers.py
```

## Review Focus

- A source image containing NaN, a fractional label, label 4, or another unknown value must fail before any retained output is written; Task 2 pins each case.
- A source/reference affine combination with no valid overlap, or a resample that empties a required tissue, must fail and leave no final map; Task 3 pins both cases.
- A partially staged destination, a changed legacy source between hashing and copying, a symlink, or a path escape must never be treated as a valid reference; Task 4 pins all four cases.
- A v3 module with the right version string but a changed meshing signature or missing mesh-cleanup method must fail module preflight before head-model submission; Task 5 pins the probe contract.
- A stale/tampered head-model manifest or promotion receipt must prevent plan creation and simulation execution; Tasks 6 and 7 pin both boundaries.

---

### Task 1: Pinned Configuration, Scope, and Root Isolation

**Files:**
- Create: `SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/__init__.py`
- Create: `SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/settings.py`
- Create: `SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/conftest.py`
- Create: `SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_settings.py`

**Interfaces:**
- Consumes: the exact subjects, target records, default roots, and scheduler values in the spec and existing v3 settings.
- Produces: `CampaignPaths`, `SubjectPaths`, `campaign_paths_from_environment(environ=None)`, `subject_paths(paths, subject)`, `assert_root_isolation(paths)`, `campaign_scope()`, `target_settings(key)`, and constants used by every later task.

- [ ] **Step 1: Write failing configuration tests**

  Add tests asserting the literal ten-subject order, two target roots, 40 repeats, 1,600 simulations, 3,200 FEM solves, `0-399%50` arrays, the `SIMNIBS326H_` override prefix, and all three harmonized writable roots. Add behavior tests proving equality, ancestor, and descendant collisions with the corrected source, legacy v3 reference, atlas, completed v3 roots, or completed v4 roots raise before writes; disjoint roots return a serializable isolation report.

- [ ] **Step 2: Run the tests and verify RED**

  Run: `python3 -m pytest -q SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_settings.py`

  Expected: collection fails because the new package/settings module does not exist.

- [ ] **Step 3: Implement the minimal settings contract**

  Use frozen dataclasses and `Path.resolve(strict=False)`. `SubjectPaths` must expose the existing corrected label/T1 paths, legacy marker/m2m paths, new staged m2m path, one retained harmonized-map path, head-model mesh paths, and receipt paths. Root checks must compare all writable roots pairwise and against every protected root.

- [ ] **Step 4: Run the focused and legacy baseline tests**

  Run: `python3 -m pytest -q SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_settings.py SimNIBS/Scripts/simnibs_326_repeatability/tests/test_simnibs_326_pipeline.py`

  Expected: all new settings tests and all 20 legacy v3 tests pass.

- [ ] **Step 5: Commit**

  ```bash
  git add SimNIBS/Scripts/simnibs_326_corrected_label_repeatability
  git commit -m "feat: pin harmonized v3 campaign settings"
  ```

### Task 2: Deterministic Tissue Harmonization

**Files:**
- Create: `SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/labels.py`
- Create: `SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_labels.py`

**Interfaces:**
- Consumes: an in-memory three-dimensional source-label array and `(x, y, z)` voxel spacing in millimetres.
- Produces: `HarmonizationStats`, `validate_source_labels(data) -> np.ndarray`, and `harmonize_labels(labels, spacing_mm) -> tuple[np.ndarray, HarmonizationStats]`; the returned `uint8` geometry uses `0-6` plus optional air label `11`.

- [ ] **Step 1: Write failing mapping and validation tests**

  Use literal synthetic arrays to prove `1/2/3/5/6` are preserved, `7/8 -> 4`, `10 -> 5`, `11 -> 11`, background stays zero, and nonblood voxels never change beyond that table. Add separate tests for NaN, fractional values, source label 4, and unknown label 12; each must raise a descriptive `LabelContractError`.

- [ ] **Step 2: Run the validation tests and verify RED**

  Run: `python3 -m pytest -q SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_labels.py -k 'mapping or rejects'`

  Expected: import/collection fails because `labels.py` is absent.

- [ ] **Step 3: Implement source validation and nonblood mapping**

  Convert only finite, exactly integral data to an integer array. Validate the complete observed label set before allocating an output. Return stats containing literal source/output counts and the observed label set.

- [ ] **Step 4: Run the mapping tests and verify GREEN**

  Run: the command from Step 2.

  Expected: all selected tests pass.

- [ ] **Step 5: Write failing physical-distance blood tests**

  Add independent fixtures proving: anisotropic spacing changes the nearest recipient; exact equal distances choose WM before GM before CSF before soft tissue; separate blood voxels can map to different recipients; bone/eyes/air/background are never recipients; and blood with no eligible recipient raises. Assert literal recipient and tie counts.

- [ ] **Step 6: Run the blood tests and verify RED**

  Run: `python3 -m pytest -q SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_labels.py -k blood`

  Expected: failures show that blood assignment is not implemented.

- [ ] **Step 7: Implement blood assignment**

  Compute `scipy.ndimage.distance_transform_edt(~recipient_mask, sampling=spacing_mm)` for each present eligible class in priority order `[1, 2, 3, 5]`. Stack distances in that order; `argmin` implements the specified tie priority. Apply assignments only at original label-9 voxels and record per-class/tie counts.

- [ ] **Step 8: Run the complete label suite**

  Run: `python3 -m pytest -q SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_labels.py`

  Expected: all label tests pass with no warnings.

- [ ] **Step 9: Commit**

  ```bash
  git add SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/labels.py SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_labels.py
  git commit -m "feat: harmonize corrected tissue labels"
  ```

### Task 3: One Retained Map, Categorical Resampling, and Temporary Masks

**Files:**
- Create: `SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/label_io.py`
- Create: `SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_label_io.py`

**Interfaces:**
- Consumes: Task 1 `SubjectPaths` and Task 2 `harmonize_labels`.
- Produces: `SubjectLabelReceipt`, `resample_geometry_to_reference(source_img, reference_img)`, `harmonize_subject_labels(subject_paths)`, and `temporary_surface_masks(harmonized_path, scratch_root)` as a context manager yielding the exact `MASK_*.nii.gz` paths.

- [ ] **Step 1: Write failing NIfTI/resampling tests**

  Use tiny real NiBabel images to prove nearest-neighbour categorical resampling, preservation of affine/qform/sform and `uint8`, rejection of non-overlapping grids, and failure when any required class `1-6` becomes empty. Assert that no output file remains after a failed conversion.

- [ ] **Step 2: Run the resampling tests and verify RED**

  Run: `python3 -m pytest -q SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_label_io.py -k resample`

  Expected: import/collection fails because `label_io.py` is absent.

- [ ] **Step 3: Implement categorical resampling**

  Harmonize in memory, wrap the geometry in a NiBabel image using the source affine/header, call `nibabel.processing.resample_from_to(..., order=0)`, revalidate integer/allowed labels, and atomically rename a temporary output only after all checks pass.

- [ ] **Step 4: Run the resampling tests and verify GREEN**

  Run: the command from Step 2.

  Expected: all selected tests pass.

- [ ] **Step 5: Write failing retention and temporary-mask tests**

  Assert that a successful subject conversion writes only `<subject>_v3_harmonized_t1fs_grid.nii.gz` plus its JSON receipt under `harmonized_labels`, never copies the source label file, and records both hashes/grids/counts. Exercise `temporary_surface_masks` and assert mutually exclusive WM/GM/CSF/BONE/SKIN/EYES/AIR masks during the context and their removal afterward, including exception cleanup.

- [ ] **Step 6: Run the new tests and verify RED**

  Run: `python3 -m pytest -q SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_label_io.py -k 'retains or masks or cleanup'`

  Expected: failures identify missing subject conversion and mask lifecycle behavior.

- [ ] **Step 7: Implement retained-map receipts and temporary masks**

  Write receipts atomically. Mask names and label mapping are `WM=1`, `GM=2`, `CSF=3`, `BONE=4`, `SKIN=5`, `EYES=6`, `AIR=11`; use the retained map as the sole reconstruction source.

- [ ] **Step 8: Run Tasks 2-3 tests**

  Run: `python3 -m pytest -q SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_labels.py SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_label_io.py`

  Expected: all tests pass.

- [ ] **Step 9: Commit**

  ```bash
  git add SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/label_io.py SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_label_io.py
  git commit -m "feat: persist one harmonized v3 label map"
  ```

### Task 4: Allowlisted Legacy Reference Staging

**Files:**
- Create: `SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/reference_stage.py`
- Create: `SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_reference_stage.py`

**Interfaces:**
- Consumes: Task 1 `SubjectPaths` and the legacy ready marker/m2m tree.
- Produces: `StageReceipt`, `stage_subject_reference(subject_paths)`, `validate_staged_reference(subject_paths)`, and `collect_reference_stage(paths) -> dict`.

- [ ] **Step 1: Write failing allowlist/staging tests**

  Build a realistic temporary legacy tree and assert that staging copies only `T1fs_conform.nii.gz`, `T1fs_nu_conform.nii.gz`, regular files under `eeg_positions/` and `toMNI/`, and the old ready-marker content under a provenance-only name. Assert no `.msh`, `.stl`, `.off`, `segment`, `mask_prep`, or control image appears in the destination.

- [ ] **Step 2: Run the allowlist test and verify RED**

  Run: `python3 -m pytest -q SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_reference_stage.py -k allowlist`

  Expected: import/collection fails because `reference_stage.py` is absent.

- [ ] **Step 3: Implement atomic allowlisted staging**

  Enumerate regular files without following links, copy into a sibling temporary directory, hash source and copy, re-hash the source immediately before promotion, and atomically rename only a complete tree. Reuse a destination only when its receipt and every hash match; otherwise fail without replacing it.

- [ ] **Step 4: Run the allowlist test and verify GREEN**

  Run: the command from Step 2.

  Expected: selected test passes.

- [ ] **Step 5: Write failing adversarial staging tests**

  Add separate tests for a symlink anywhere in the allowlist, `..`/resolved path escape, a source changed after its first hash, an incomplete existing destination, and a forged receipt. Add a cohort test requiring ten valid subject receipts before writing the collection receipt.

- [ ] **Step 6: Run adversarial tests and verify RED**

  Run: `python3 -m pytest -q SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_reference_stage.py -k 'symlink or escape or changed or incomplete or forged or cohort'`

  Expected: the missing safeguards fail their assertions.

- [ ] **Step 7: Implement validation and cohort collection**

  Bind each receipt to subject, absolute source/destination roots, relative allowlist, byte counts, SHA-256 values, Git commit, and UTC timestamp. The collector writes only after all ten receipts validate.

- [ ] **Step 8: Run the complete staging suite**

  Run: `python3 -m pytest -q SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_reference_stage.py`

  Expected: all staging tests pass.

- [ ] **Step 9: Commit**

  ```bash
  git add SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/reference_stage.py SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_reference_stage.py
  git commit -m "feat: stage v3 reference metadata safely"
  ```

### Task 5: Version-Pinned SimNIBS 3.2.6 Meshing Adapter

**Files:**
- Create: `SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/v3_meshing.py`
- Create: `SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_v3_meshing.py`

**Interfaces:**
- Consumes: Task 3 temporary mask paths and the installed `simnibs.msh.headreco`, `hmutils`, and `mesh_io` modules.
- Produces: `V3MeshingAPI`, `SurfaceResult`, `MeshValidation`, `load_v3_meshing_api()`, `probe_v3_meshing_api(api)`, `build_surfaces(api, mask_paths, m2m_dir, scratch_dir, vertex_density=0.5)`, `build_volume_mesh(api, subject, surfaces, relabel_surfaces, m2m_dir, output_mesh, keep_air=False)`, and `validate_mesh(api, mesh_path)`.

- [ ] **Step 1: Write failing API-probe tests**

  Use complete lightweight fake modules to prove the probe accepts version `3.2.6`, `make_surface_meshes(input_files, out_dir, temp_dir, vertex_density=0.5, catsurf=False)`, `make_volume_mesh(subject_id, input_files, out_dir, keep_air)`, `relabel_compartments`, `volume_mesh_relabel`, buffered `read_msh`, `write_msh`, and mesh cleanup methods. Separately remove/change each required callable/signature and assert a fail-closed report.

- [ ] **Step 2: Run the probe tests and verify RED**

  Run: `python3 -m pytest -q SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_v3_meshing.py -k probe`

  Expected: import/collection fails because `v3_meshing.py` is absent.

- [ ] **Step 3: Implement lazy imports and the exact API probe**

  Keep SimNIBS imports out of module scope. Record module file paths, version, inspected signatures, meshfix/gmsh resolution, and the expected cleanup methods without running a mesher.

- [ ] **Step 4: Run probe tests and verify GREEN**

  Run: the command from Step 2.

  Expected: all probe tests pass.

- [ ] **Step 5: Write failing surface/volume call-order tests**

  Assert that surfaces call v3 `make_surface_meshes` with sorted `MASK_*.nii.gz`, `vertex_density=0.5`, and `catsurf=False`. For volume meshing, assert the official v3.2.6 sequence: `make_volume_mesh`; buffered read; optional ventricle/relabel adjustment; `fix_surface_labels`; `fix_thin_tetrahedra`; tetrahedral ordering; triangle ordering; final write; then independent tag/positive-volume validation. Assert the raw gmsh mesh is removed only after the final mesh validates.

- [ ] **Step 6: Run adapter tests and verify RED**

  Run: `python3 -m pytest -q SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_v3_meshing.py -k 'surface or volume or cleanup or tag'`

  Expected: failures show the adapter operations are missing.

- [ ] **Step 7: Implement the exact v3 meshing sequence**

  Follow the official v3.2.6 `headreco.py` meshing sections, but stop after final mesh creation/validation instead of creating redundant control volumes and MNI-warped derivatives. Keep all external binary work inside the installed v3 API.

- [ ] **Step 8: Run the complete meshing-adapter suite**

  Run: `python3 -m pytest -q SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_v3_meshing.py`

  Expected: all adapter tests pass without importing the locally installed SimNIBS 4 package.

- [ ] **Step 9: Commit**

  ```bash
  git add SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/v3_meshing.py SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_v3_meshing.py
  git commit -m "feat: add pinned SimNIBS 3.2.6 meshing adapter"
  ```

### Task 6: Head-Model Build, QC Collection, and Promotion

**Files:**
- Create: `SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/headmodels.py`
- Create: `SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/render_qc.py`
- Create: `SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_headmodels.py`

**Interfaces:**
- Consumes: Task 3 label products, Task 4 staged references, and Task 5 meshing adapter.
- Produces: `build_headmodel(paths, subject, api)`, `validate_headmodel(paths, subject, api)`, `write_qc_request(paths, subject)`, `render_qc_request(request_path)`, `collect_headmodels(paths, render_python)`, `validate_headmodel_manifest(paths)`, and `promote_headmodels(paths, manifest_sha256, confirm_reviewed)`.

- [ ] **Step 1: Write failing head-model lifecycle tests**

  Exercise real temporary NIfTIs plus a narrow fake meshing adapter. Assert the order: validate staged reference, harmonize, create temporary masks, create new surfaces, create new mesh, validate, write marker, remove input masks. Assert retries reuse only a fully hash-valid marker and rebuild/fail safely for partial state. Assert the marker binds source, retained-map, staged-reference, surface, mesh, API-probe, and Git hashes.

- [ ] **Step 2: Run lifecycle tests and verify RED**

  Run: `python3 -m pytest -q SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_headmodels.py -k 'build or retry or marker'`

  Expected: import/collection fails because `headmodels.py` is absent.

- [ ] **Step 3: Implement build and validation**

  Use a participant-owned scratch directory and atomic markers. Require expected tags `1-6`, finite nodes, tetrahedra, positive volumes, and at least the required WM/GM/CSF/bone/skin surfaces. Compare against any legacy mesh/surface hashes and reject reuse. Do not write a cohort-complete marker here.

- [ ] **Step 4: Run lifecycle tests and verify GREEN**

  Run: the command from Step 2.

  Expected: all selected tests pass.

- [ ] **Step 5: Write failing renderer, collector, and promotion tests**

  Assert the renderer lazily imports Matplotlib, shows T1 plus harmonized categorical overlays on three orthogonal slices, and verifies a request hash before writing PNG. Assert the collector requires ten valid markers and ten PNGs, writes one map/QC TSV plus manifest/checksums, and never copies source maps. Assert promotion requires the exact manifest hash and explicit `confirm_reviewed=True`; stale/tampered hashes and missing panels fail without a receipt.

- [ ] **Step 6: Run collection tests and verify RED**

  Run: `python3 -m pytest -q SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_headmodels.py -k 'render or collect or promote or tampered'`

  Expected: failures identify missing collection/promotion behavior.

- [ ] **Step 7: Implement renderer, collection, and promotion**

  The numeric/head-model interpreter writes content-addressed render requests; the configured ti-post executable renders them. Promotion writes only `_pipeline/headmodels/promoted.json` and never edits scientific artifacts.

- [ ] **Step 8: Run the complete head-model suite**

  Run: `python3 -m pytest -q SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_headmodels.py`

  Expected: all head-model tests pass with Matplotlib absent from the system interpreter.

- [ ] **Step 9: Commit**

  ```bash
  git add SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/headmodels.py SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/render_qc.py SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_headmodels.py
  git commit -m "feat: build and promote harmonized v3 head models"
  ```

### Task 7: Full Repeatability Campaign and v3 FEM Adapter

**Files:**
- Create: `SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/simulation_runner.py`
- Create: `SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/pipeline.py`
- Create: `SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/__main__.py`
- Create: `SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_simulation_runner.py`
- Create: `SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 1 settings, Task 5 direct volume mesher, Task 6 promoted manifest, existing `ti_current_repair` experiment configuration/validation/median-selection/ROI tools, and the completed v3 runner behavior.
- Produces: a v3.2.6 backend named `simnibs-3.2.6-harmonized-corrected-label-v2`; `campaign_scope`, `preflight`, `initialize`, `module_preflight`, receipt validators, remesh/fixed completion, status, and CLI commands defined by the spec.

- [ ] **Step 1: Write failing simulation-boundary tests**

  Pin max-TI behavior and v3 label export from the legacy tests. Assert a remesh task copies only the promoted new surfaces/reference files into an isolated workspace, calls Task 5 `build_volume_mesh`, and records promoted-manifest/head-model/map/mesh hashes. Assert fixed tasks consume the selected immutable mesh. Reject configs pointing at old v3, v4, source, or unpromoted roots.

- [ ] **Step 2: Run simulation tests and verify RED**

  Run: `python3 -m pytest -q SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_simulation_runner.py`

  Expected: import/collection fails because the new runner is absent.

- [ ] **Step 3: Implement the new runner from the validated legacy behavior**

  Copy the established v3 adapter semantics into the new package, replace `_copy_scaffold_for_volumemesh`/CLI remeshing with promoted-headmodel staging plus Task 5 direct volume meshing, and change provenance/root guards. Preserve SESSION execution, Grossman max-TI, `ElementData.to_nifti(method="assign")`, output layout, validators, conductivities, and confirmed stimulation records.

- [ ] **Step 4: Run new and legacy runner tests**

  Run: `python3 -m pytest -q SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_simulation_runner.py SimNIBS/Scripts/simnibs_326_repeatability/tests/test_simnibs_326_pipeline.py`

  Expected: new tests and all 20 legacy tests pass.

- [ ] **Step 5: Write failing campaign/receipt tests**

  Assert read-only preflight validates ten source/staged inputs and labels without creating roots; init writes two 400-remesh configs and campaign manifests only under new roots; module preflight validates v3 meshing/FEM APIs without MATLAB; remesh completion requires 400/400; fixed preparation keeps the established median rule; final completion requires 800/800 per target; status distinguishes initialized, staged, headmodels-built, promoted, remesh-complete, and final-complete. Add stale/tampered promotion tests at plan creation and task execution.

- [ ] **Step 6: Run campaign tests and verify RED**

  Run: `python3 -m pytest -q SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_pipeline.py`

  Expected: failures identify missing campaign and receipt operations.

- [ ] **Step 7: Implement pipeline and CLI**

  Port the proven config/receipt logic rather than importing mutable globals from the legacy package. Commands are `preflight`, `init`, `module-preflight`, `validate-module-receipt`, `stage-reference`, `collect-references`, `build-headmodel`, `validate-headmodel`, `render-qc`, `collect-headmodels`, `promote-headmodels`, `complete-remesh`, `prepare-fixed`, `complete-final`, `validate-final`, `show-plan`, and `status`.

- [ ] **Step 8: Run the complete Python package suites**

  Run: `python3 -m pytest -q SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests SimNIBS/Scripts/simnibs_326_repeatability/tests SimNIBS/Scripts/ti_current_repair/tests/test_staged_current_repair_pipeline.py SimNIBS/Scripts/ti_current_repair/tests/test_repeatability_optimizer_roi_metrics.py`

  Expected: all selected new, legacy-v3, and maintained shared-runner tests pass.

- [ ] **Step 9: Commit**

  ```bash
  git add SimNIBS/Scripts/simnibs_326_corrected_label_repeatability
  git commit -m "feat: integrate harmonized v3 repeatability campaign"
  ```

### Task 8: Slurm Launchers and Safe Submission Orchestration

**Files:**
- Create: every file under `SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/hpc/`
- Create: `SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_hpc_launchers.py`

**Interfaces:**
- Consumes: Task 7 CLI and the inherited working scheduler profile.
- Produces: copy-paste-safe `hpc/submit.sh` modes `preflight`, `init`, `module-check`, `stage-references`, `headmodels`, `promote`, `remesh`, `post-remesh`, `fixed`, `all-remesh`, `all-fixed`, `analysis`, `analysis-status`, and `status`.

- [ ] **Step 1: Write failing launcher behavior tests**

  Execute `submit.sh` with a fake `sbatch` that records arguments. Assert `preflight` and `status` submit nothing; `stage-references` creates `0-9%10`; `headmodels` requires completed reference/module receipts and creates `0-9%10` plus an `afterok` collector; simulation arrays are `0-399%50`; all-remesh remains two sequential 400-task target chains; all-fixed remains two sequential 400-task arrays. Assert every submission prints the numerical Scope block first.

- [ ] **Step 2: Run launcher tests and verify RED**

  Run: `python3 -m pytest -q SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_hpc_launchers.py`

  Expected: failures show the launchers are absent.

- [ ] **Step 3: Implement Slurm scripts and submitter**

  Copy the known-good retry/requeue/logging structure. Keep SimNIBS module/resources unchanged. Remove the old MATLAB module load/export and CAT12 overlay from new jobs; retain only the bounded NumPy/NiBabel compatibility shim. Use `SIMNIBS326H_RENDER_PYTHON` for the collector renderer and validate it before submission. Require explicit resolved-root confirmation for mutating initialization/promotion operations; the default mode remains read-only preflight.

- [ ] **Step 4: Run executable launcher tests and verify GREEN**

  Run: the command from Step 2.

  Expected: all launcher behavior tests pass.

- [ ] **Step 5: Run static shell verification**

  Run: `bash -n SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/hpc/*.sh SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/hpc/*.slurm`

  Expected: exit 0 with no output.

  Run: `shellcheck SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/hpc/*.sh SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/hpc/*.slurm`

  Expected: exit 0 with no findings.

- [ ] **Step 6: Review operational deltas**

  Compare every new launcher against the corresponding legacy launcher and confirm in the task log: unchanged module/version, partition, CPUs, memory, time, concurrency, task counts, retry cap, exact-element requeue, append mode, source/output roots, and dependencies; intentional removals are MATLAB/CAT12 only.

- [ ] **Step 7: Commit**

  ```bash
  git add SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/hpc SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests/test_hpc_launchers.py
  git commit -m "feat: add gated harmonized v3 Slurm workflow"
  ```

### Task 9: Documentation, Operational Profile, and Full Verification

**Files:**
- Create: `SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/README.md`
- Create: `SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/operational_profile.json`
- Modify only if implementation evidence requires clarification: `SimNIBS/Scripts/docs/superpowers/specs/2026-10-22-simnibs326-corrected-label-repeatability-design.md`

**Interfaces:**
- Consumes: all prior tasks and their verified command surface.
- Produces: an auditable HPC handoff that clearly separates implemented/local verification, future module preflight, future full head-model execution, and the not-yet-submitted 1,600-simulation campaign.

- [ ] **Step 1: Write README and candidate operational profile**

  Document the scientific comparison, exact label mapping, one-map storage policy, legacy allowlist, new-root layout, API boundary, promotion gate, deterministic-remesh interpretation, and commands for pull, preflight, init, staging, head models, inspection, promotion, remesh, fixed, analysis, and status. Mark the new harmonization/meshing stage `candidate` until Stanage evidence exists; inherit only the confirmed v3 module/simulation profile as working.

- [ ] **Step 2: Run package and regression tests fresh**

  Run: `python3 -m pytest -q SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/tests SimNIBS/Scripts/simnibs_326_repeatability/tests SimNIBS/Scripts/ti_current_repair/tests/test_staged_current_repair_pipeline.py SimNIBS/Scripts/ti_current_repair/tests/test_repeatability_optimizer_roi_metrics.py`

  Expected: zero failures.

- [ ] **Step 3: Run Python/static verification**

  Run: `python3 -m compileall -q SimNIBS/Scripts/simnibs_326_corrected_label_repeatability`

  Expected: exit 0.

  Run: `ruff check SimNIBS/Scripts/simnibs_326_corrected_label_repeatability`

  Expected: exit 0 with no findings.

  Run: `bash -n SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/hpc/*.sh SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/hpc/*.slurm && shellcheck SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/hpc/*.sh SimNIBS/Scripts/simnibs_326_corrected_label_repeatability/hpc/*.slurm`

  Expected: exit 0 with no findings.

- [ ] **Step 4: Run real local label/QC integration**

  Create only synthetic data under `/tmp`; run the real CLI through harmonization, categorical resampling, temporary-mask creation/cleanup, QC-request generation, and QC rendering with `/home/boyan/anaconda3/envs/simnibs_post/bin/python3.12`. Assert one retained map, one receipt, one PNG, no source copy, and no surviving temporary masks.

  Expected: exit 0 and the integration manifest reports the expected single retained NIfTI.

- [ ] **Step 5: Run final repository checks**

  Run: `git diff --check`

  Expected: exit 0 with no output.

  Run: `git status --short`

  Expected: only the intended package, plan, and any explicitly ledgered spec clarification are present before the final commit.

- [ ] **Step 6: Commit**

  ```bash
  git add SimNIBS/Scripts/simnibs_326_corrected_label_repeatability SimNIBS/Scripts/docs/superpowers
  git commit -m "docs: document harmonized v3 workflow handoff"
  ```

- [ ] **Step 7: Run final whole-branch review and verification**

  Build the executing-plans review package from the branch merge base through `HEAD`. Under the current no-delegation constraint, perform a separate author whole-branch review using the review package and explicitly report that it is weaker than a fresh reviewer. Resolve Critical/Important findings in one TDD fix pass and rerun Steps 2-5 before any completion claim.

## HPC Handoff Boundary

Implementation completion authorizes only repository delivery. The first HPC action after merge is the full ten-subject read-only preflight. A real module API probe, reference staging, ten-subject head-model array, QC inspection, and promotion are separate evidence gates. None of those gates silently releases the 1,600 simulations.
