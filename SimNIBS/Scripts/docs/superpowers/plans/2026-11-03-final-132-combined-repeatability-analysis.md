# Final-132 Combined Repeatability Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a provenance-safe, read-only 9+1 analysis package covering both SimNIBS versions, both targets, both mesh conditions, and all 3,200 technical-repeat metric rows.

**Architecture:** A new `repeatability_combined_analysis` package assembles four canonical schema-v2 input tables from the completed base-nine and replacement sources, invokes the existing analyzers unchanged for established outputs, and adds a separate condition-inclusive participant-level analysis. A guarded workflow snapshots source hashes before one lightweight Slurm job, writes through an attempt directory, promotes atomically, and later rebinds the analysis provenance to the completed combined dataset using its source mapping.

**Tech Stack:** Python 3.9+, dataclasses, pathlib, csv/json/hashlib/tarfile, pandas, NumPy, SciPy, Matplotlib, pytest, Bash, Slurm.

**Spec:** `docs/superpowers/specs/2026-11-03-final-132-combined-repeatability-analysis-design.md`

## Global Constraints

- Canonical cohort is the ordered 9+1 subject list in the spec; `sub-CC510415` replaces `sub-CC520083`.
- Exact scope is 10 subjects, 2 versions, 2 targets, 2 conditions, 40 repeats, 80 cells, and 3,200 rows.
- Participant is the inferential unit; runs are technical replicates.
- Primary outcome is `roi_median_v_per_m`; mean and p95 are sensitivities.
- Sources, the active combined-dataset copy, and existing canonical analyzers remain read-only.
- Output root is `/mnt/parscratch/users/cop23bi/final_132_repeatability_combined_analysis_v1`.
- No meshing, FEM simulation, implicit NIfTI extraction, source deletion, or dataset mutation.
- Missing schema-v2 replacement metrics fail closed and report the exact 160-run refresh requirement.
- `preflight`, `scope`, `status`, and `rebind-preflight` submit no jobs.
- Only explicit `run` may create the owned root and submit one 4-CPU, 16-GB, 2-hour Sheffield job.
- All output promotion is atomic and all retained files receive SHA-256 provenance.

## Review Focus

- Source table with the right row count but the wrong cohort must fail before output creation; pinned in Task 2 assembly tests.
- Replacement v4 summary files that are not schema-v2 optimizer-ROI tables must not be silently adapted; pinned in Task 2 missing-schema tests.
- Zero or near-zero technical variance must yield valid raw-SD contrasts and an explicitly undefined CV where necessary; pinned in Task 3 statistics tests.
- Canonical analyzer invocation must use the assembled 9+1 tables rather than legacy default roots; pinned in Task 4 orchestration tests.
- A source hash change after snapshot but before execution must leave only an incomplete attempt and never a completion manifest; pinned in Task 5 workflow tests.

---

### Task 1: Code-owned scope and source profile

**Files:**
- Create: `repeatability_combined_analysis/__init__.py`
- Create: `repeatability_combined_analysis/__main__.py`
- Create: `repeatability_combined_analysis/settings.py`
- Test: `repeatability_combined_analysis/tests/test_settings.py`

**Interfaces:**
- Consumes: production paths and cohort fixed by the spec.
- Produces: `AnalysisConfig`, `SourcePaths`, `SUBJECTS`, `VERSIONS`, `TARGETS`, `CONDITIONS`, `METRICS`, `PRIMARY_METRIC`, and `production_config()`.

- [ ] **Step 1: Write the failing scope tests**

  Add tests asserting the exact ordered subjects, replacement/exclusion IDs,
  expected cell/row counts, output root, ten metric-table source paths, ten
  terminal receipt paths, and absence of the active combined dataset from all
  scientific input paths.

- [ ] **Step 2: Run the tests and confirm failure**

  Run: `python3 -m pytest repeatability_combined_analysis/tests/test_settings.py -q`

  Expected: FAIL because the package and settings do not exist.

- [ ] **Step 3: Implement the immutable settings API**

  Define frozen dataclasses:

  ```python
  @dataclass(frozen=True)
  class SourcePaths:
      v3_base9_left_csv: Path
      v3_base9_right_csv: Path
      v3_replacement_left_csv: Path
      v3_replacement_right_csv: Path
      v4_historical_left_csv: Path
      v4_historical_right_csv: Path
      v4_corrected_fixed_left_csv: Path
      v4_corrected_fixed_right_csv: Path
      v4_replacement_left_csv: Path
      v4_replacement_right_csv: Path
      receipts: tuple[Path, ...]

  @dataclass(frozen=True)
  class AnalysisConfig:
      output_root: Path
      sources: SourcePaths
      subjects: tuple[str, ...] = SUBJECTS
      repeat_count: int = 40
  ```

  Add computed properties for `expected_cells == 80`,
  `expected_rows_per_table == 800`, and `expected_rows == 3200`. Point
  production defaults only at harmonized base-nine, corrected v4 historical
  and spherical-fixed roots, and `sub-CC510415` replacement roots.

- [ ] **Step 4: Run the settings tests**

  Run: `python3 -m pytest repeatability_combined_analysis/tests/test_settings.py -q`

  Expected: PASS.

- [ ] **Step 5: Commit the scope contract**

  ```bash
  git add repeatability_combined_analysis/__init__.py repeatability_combined_analysis/__main__.py repeatability_combined_analysis/settings.py repeatability_combined_analysis/tests/test_settings.py
  git commit -m "feat: define combined repeatability analysis scope"
  ```

### Task 2: Fail-closed canonical input assembly

**Files:**
- Create: `repeatability_combined_analysis/assembly.py`
- Create: `repeatability_combined_analysis/receipts.py`
- Create: `repeatability_combined_analysis/tests/conftest.py`
- Create: `repeatability_combined_analysis/tests/test_assembly.py`
- Test: `repeatability_combined_analysis/tests/test_receipts.py`

**Interfaces:**
- Consumes: `AnalysisConfig` and `SourcePaths` from Task 1.
- Produces: `AssemblyPlan`, `SourceSelection`, `build_assembly_plan(config) -> AssemblyPlan`, `materialize_inputs(plan, destination) -> dict[str, Path]`, and `validate_terminal_receipts(config) -> list[dict[str, object]]`.

- [ ] **Step 1: Build synthetic source fixtures and failing happy-path tests**

  Generate schema-v2 CSV fixtures with the same source shapes as production:
  v3 base-nine tables with 720 rows, v3 replacement tables with 80 rows, v4
  historical tables with 800 rows, v4 corrected-fixed tables with 400 rows,
  and v4 replacement tables with 80 rows. Assert four 800-row outputs, exact
  subject order/set, 3,200 total rows, and zero excluded-subject rows.

- [ ] **Step 2: Add failing source-role and provenance tests**

  Assert that v4 canonical outputs take only `remesh` from historical tables,
  only `fixed_mesh` from spherical-fixed tables, and both conditions from the
  replacement table. Assert every source selection records path, size,
  SHA-256, observed rows, selected rows, rejected rows, conditions, and
  subjects.

- [ ] **Step 3: Add failing validation and review-focus tests**

  Cover wrong cohort with valid row count, excluded subject surviving assembly,
  missing replacement, duplicate keys, missing `repeat_040`, wrong ROI or
  volume, missing mean/p95 columns, negative/non-finite values, inconsistent
  support accounting, changing within-subject ROI geometry, schema version 1,
  and a replacement v4 report-summary CSV masquerading as an optimizer table.
  The last case must raise an error containing `160 existing NIfTI metrics`
  and must not create output.

- [ ] **Step 4: Add failing receipt tests**

  Exercise the differing v3/v4 receipt shapes while requiring `status` or
  terminal validation status to be `complete`, the expected target/repeat
  scope, and source receipt hash capture. Reject missing, malformed, selected
  subject mismatches, incomplete stages, and wrong repeat counts.

- [ ] **Step 5: Run the new tests and confirm failure**

  Run: `python3 -m pytest repeatability_combined_analysis/tests/test_assembly.py repeatability_combined_analysis/tests/test_receipts.py -q`

  Expected: FAIL because assembly and receipt APIs do not exist.

- [ ] **Step 6: Implement table validation and selection**

  Implement `validate_metric_table(frame, *, target, allowed_conditions,
  expected_subjects, expected_rows)` and source-role-specific selection. Use
  pandas for parsing, but write canonical tables with stable column order and
  stable sort order `(subject_order, condition_order, repeat_number)`.

- [ ] **Step 7: Implement immutable assembly plans**

  `build_assembly_plan` reads and validates without writing. The plan holds
  source hashes and selected row hashes. `materialize_inputs` rechecks all
  source hashes before writing four CSV files and `assembly_manifest.json`
  atomically.

- [ ] **Step 8: Implement receipt normalization**

  Normalize heterogeneous terminal receipts to target, subject scope,
  conditions, repeats, status, path, and SHA-256 while retaining the raw
  receipt path. Fail when a receipt cannot prove its assigned source scope.

- [ ] **Step 9: Run assembly and receipt tests**

  Run: `python3 -m pytest repeatability_combined_analysis/tests/test_assembly.py repeatability_combined_analysis/tests/test_receipts.py -q`

  Expected: PASS.

- [ ] **Step 10: Commit canonical assembly**

  ```bash
  git add repeatability_combined_analysis/assembly.py repeatability_combined_analysis/receipts.py repeatability_combined_analysis/tests/conftest.py repeatability_combined_analysis/tests/test_assembly.py repeatability_combined_analysis/tests/test_receipts.py
  git commit -m "feat: assemble canonical 9+1 repeatability metrics"
  ```

### Task 3: Condition-inclusive participant-level statistics

**Files:**
- Create: `repeatability_combined_analysis/statistics.py`
- Create: `repeatability_combined_analysis/render.py`
- Test: `repeatability_combined_analysis/tests/test_statistics.py`
- Test: `repeatability_combined_analysis/tests/test_render.py`

**Interfaces:**
- Consumes: four canonical 800-row tables from Task 2.
- Produces: `summarize_cells(frame) -> DataFrame`, `exact_sign_flip(values) -> dict`, `holm_adjust(values) -> ndarray`, `build_contrasts(cell_summary, outcome) -> dict[str, DataFrame]`, `build_support_audit(frame) -> tuple[DataFrame, dict]`, and `render_comprehensive_outputs(frame, output_dir) -> dict`.

- [ ] **Step 1: Write failing cell-summary tests**

  Assert 80 participant cells per run-level metric after combining all four
  tables and exact mean, median, SD, CV, IQR, range, support minimum, and
  affected-run counts for controlled fixtures. Assert CV is null, not infinite,
  when the denominator is zero.

- [ ] **Step 2: Write failing contrast arithmetic tests**

  Pin the signs for `v3 - v4`, `fixed - remesh`, and
  `(v3 - v4)_fixed - (v3 - v4)_remesh`. Assert 12/12/6 central rows and
  12/12/6 variability rows, with ten participant contrasts behind every row.

- [ ] **Step 3: Write failing inference tests**

  Assert exact 1,024-assignment two-sided sign-flip p-values, stable Holm
  adjustment within each named family, defined/undefined Cohen's dz behavior,
  and valid raw-SD contrasts when one or both cells have zero SD.

- [ ] **Step 4: Write failing support-audit tests**

  Assert group summaries by version/target/condition/subject, no causal wording,
  and exact counts for complete versus incomplete finite support.

- [ ] **Step 5: Write failing rendering tests**

  Replace Matplotlib writes with test placeholders and assert four target-level
  figure families, report/caption files, portable relative artifact paths, and
  explicit participant/technical-repeat interpretation text.

- [ ] **Step 6: Run statistics/render tests and confirm failure**

  Run: `python3 -m pytest repeatability_combined_analysis/tests/test_statistics.py repeatability_combined_analysis/tests/test_render.py -q`

  Expected: FAIL because the analysis functions do not exist.

- [ ] **Step 7: Implement participant-cell summaries and exact inference**

  Use deterministic ordering, raw participant contrasts, exact sign enumeration,
  and separate Holm families named in the spec. Use run SD as the primary
  technical-variability outcome and retain CV as descriptive only.

- [ ] **Step 8: Implement figures and prose outputs**

  Render participant-level central-value, technical-SD, paired-version, and
  interaction figures for both targets. Reports must label primary versus
  sensitivity metrics, fixed-mesh interpretation, aligned-not-identical ROI
  construction, and continuous-field interpretation limits.

- [ ] **Step 9: Run statistics and rendering tests**

  Run: `python3 -m pytest repeatability_combined_analysis/tests/test_statistics.py repeatability_combined_analysis/tests/test_render.py -q`

  Expected: PASS.

- [ ] **Step 10: Commit comprehensive statistics**

  ```bash
  git add repeatability_combined_analysis/statistics.py repeatability_combined_analysis/render.py repeatability_combined_analysis/tests/test_statistics.py repeatability_combined_analysis/tests/test_render.py
  git commit -m "feat: add condition-inclusive repeatability analysis"
  ```

### Task 4: Established analyzer orchestration

**Files:**
- Create: `repeatability_combined_analysis/established.py`
- Test: `repeatability_combined_analysis/tests/test_established.py`

**Interfaces:**
- Consumes: four canonical input paths from Task 2.
- Produces: `run_established(inputs, output_root) -> dict[str, object]` and `canonical_code_manifest() -> dict[str, object]`.

- [ ] **Step 1: Write failing orchestration tests**

  Monkeypatch the two canonical analyzers and assert calls for within-v3,
  within-v4, and cross-version-remesh with the four assembled paths,
  `expected_subjects=10`, seed `20260803`, and 20,000 rank draws.

- [ ] **Step 2: Add legacy-root and code-hash tests**

  Assert no production default from the legacy v3 comparison launcher reaches
  any canonical call. Assert the manifest records paths and SHA-256 hashes for
  `analyze_repeatability_optimizer_roi_metrics.py` and
  `compare_simnibs_version_remesh_distributions.py`.

- [ ] **Step 3: Add output validation tests**

  Require 1,600 rows from each within-version analysis and 3,200 input rows /
  1,600 selected remesh rows from the cross-version analysis. Reject missing or
  inconsistent canonical manifests.

- [ ] **Step 4: Run the orchestration tests and confirm failure**

  Run: `python3 -m pytest repeatability_combined_analysis/tests/test_established.py -q`

  Expected: FAIL because `established.py` does not exist.

- [ ] **Step 5: Implement dynamic imports and canonical calls**

  Load canonical scripts from repository-relative paths, invoke their public
  `run` functions in fresh output directories, and validate their returned and
  written manifests. Do not modify or copy their implementations.

- [ ] **Step 6: Run orchestration tests**

  Run: `python3 -m pytest repeatability_combined_analysis/tests/test_established.py -q`

  Expected: PASS.

- [ ] **Step 7: Commit established-analysis integration**

  ```bash
  git add repeatability_combined_analysis/established.py repeatability_combined_analysis/tests/test_established.py
  git commit -m "feat: orchestrate established repeatability analyses"
  ```

### Task 5: Guarded workflow, manifests, archive, and rebind check

**Files:**
- Create: `repeatability_combined_analysis/workflow.py`
- Create: `repeatability_combined_analysis/cli.py`
- Test: `repeatability_combined_analysis/tests/test_workflow.py`
- Test: `repeatability_combined_analysis/tests/test_cli.py`

**Interfaces:**
- Consumes: scope, assembly, statistics, rendering, and established analyzers from Tasks 1-4.
- Produces: `preflight(config)`, `prepare(config, confirmed_root)`, `execute(config)`, `status(config)`, `rebind_preflight(config, dataset_root)`, and CLI commands `scope|preflight|prepare|execute|status|rebind-preflight`.

- [ ] **Step 1: Write failing preflight and prepare tests**

  Assert preflight validates all sources and receipts but creates nothing.
  Assert prepare requires exact output-root confirmation, creates only an owned
  workflow root plus immutable `input_plan.json`, and refuses an unowned,
  mismatched, or incomplete existing root.

- [ ] **Step 2: Write failing execution and TOCTOU tests**

  Assert execute rechecks every snapshotted hash, writes through
  `.attempt-<id>`, promotes only a complete result, and rejects a changed source
  after snapshot without writing `analysis_manifest.json`. A matching complete
  result must be idempotent.

- [ ] **Step 3: Write failing artifact/package tests**

  Assert all design outputs exist, `SHA256SUMS` validates, the tar archive
  contains relative paths and no attempt/workflow files, the sidecar matches,
  JSON contains no NaN literals, and the completion manifest records Git commit,
  code hashes, environment versions, 3,200 rows, 80 cells, and zero simulations.

- [ ] **Step 4: Write failing status and rebind tests**

  Status must distinguish absent/prepared/running-or-incomplete/complete/invalid
  using independent hash checks. Rebind must require the combined dataset's
  completion receipt, locate every analysis source through
  `provenance/source_mapping.tsv`, verify mapped destination hashes equal the
  assembly source hashes, and perform no writes.

- [ ] **Step 5: Write failing CLI tests**

  Assert `scope` and `preflight` print the exact numeric scope, `prepare`
  requires confirmation, `execute` cannot bypass ownership, and all read-only
  modes leave the filesystem unchanged.

- [ ] **Step 6: Run workflow/CLI tests and confirm failure**

  Run: `python3 -m pytest repeatability_combined_analysis/tests/test_workflow.py repeatability_combined_analysis/tests/test_cli.py -q`

  Expected: FAIL because workflow and CLI do not exist.

- [ ] **Step 7: Implement the guarded state machine**

  Use atomic JSON/text writes, exact owner and plan hashes, attempt-directory
  cleanup only within the owned analysis root, independent manifest validation,
  and deterministic artifact inventory generation.

- [ ] **Step 8: Implement archive and checksum generation**

  Archive only the promoted scientific package; exclude `_workflow`, attempts,
  the archive itself, and its sidecar. Write the sidecar after atomic archive
  promotion.

- [ ] **Step 9: Implement read-only dataset rebind**

  Parse the consolidation source mapping by exact source path. Require each
  assembly source to map to a regular destination under the completed dataset
  and compare SHA-256; report bindings without altering either root.

- [ ] **Step 10: Run workflow and CLI tests**

  Run: `python3 -m pytest repeatability_combined_analysis/tests/test_workflow.py repeatability_combined_analysis/tests/test_cli.py -q`

  Expected: PASS.

- [ ] **Step 11: Commit guarded execution**

  ```bash
  git add repeatability_combined_analysis/workflow.py repeatability_combined_analysis/cli.py repeatability_combined_analysis/tests/test_workflow.py repeatability_combined_analysis/tests/test_cli.py
  git commit -m "feat: guard combined analysis execution and provenance"
  ```

### Task 6: Slurm wrapper and operational runbook

**Files:**
- Create: `repeatability_combined_analysis/hpc/analysis.slurm`
- Create: `repeatability_combined_analysis/hpc/submit.sh`
- Create: `repeatability_combined_analysis/README.md`
- Test: `repeatability_combined_analysis/tests/test_hpc_launcher.py`

**Interfaces:**
- Consumes: CLI from Task 5.
- Produces: operator modes `scope|preflight|run|status|rebind-preflight` and one analysis job invoking the internal `prepare`/`execute` stages.

- [ ] **Step 1: Write failing launcher scope tests**

  Run the launcher with fake production fixtures and assert it prints 10
  subjects, 80 cells, 3,200 rows, zero simulations, one job, 4 CPU, 16 GB,
  2 hours, Sheffield, read-only sources, isolated output, and no `sbatch` call
  for `scope` or `preflight`.

- [ ] **Step 2: Write failing submission and status tests**

  Stub `sbatch` and assert `run` performs preflight, exact-root prepare,
  records the parsed numeric job ID, and submits exactly one job. Assert status
  combines `sacct` evidence when available with independent package validation.

- [ ] **Step 3: Write failing environment tests**

  Require an executable analysis Python with pandas/NumPy/SciPy/Matplotlib,
  repository commit capture, `PYTHONNOUSERSITE=1`, explicit `MPLCONFIGDIR`, and
  refusal to use an output root other than the code-owned production value.

- [ ] **Step 4: Run launcher tests and confirm failure**

  Run: `python3 -m pytest repeatability_combined_analysis/tests/test_hpc_launcher.py -q`

  Expected: FAIL because launcher files do not exist.

- [ ] **Step 5: Implement the Slurm script**

  Use `set -euo pipefail`, 4 CPU, 16 GB, 2 hours, Sheffield, and the prepared
  input-plan hash. Execute only `python -m repeatability_combined_analysis
  execute` and print the validated final manifest.

- [ ] **Step 6: Implement the shell wrapper**

  Make `scope`, `preflight`, `status`, and `rebind-preflight` read-only. `run`
  must call preflight, prepare with exact confirmation, submit one job, and
  write `_workflow/submission.json` atomically without touching scientific
  sources or the combined dataset.

- [ ] **Step 7: Write the runbook**

  Document the source-bound nature, exact inputs and outputs, explicit run
  command, progress/status commands, missing-v4-metrics blocker, validation,
  download/checksum commands, rebind step after dataset completion, and the
  fact that no source deletion is performed.

- [ ] **Step 8: Run launcher tests**

  Run: `python3 -m pytest repeatability_combined_analysis/tests/test_hpc_launcher.py -q`

  Expected: PASS.

- [ ] **Step 9: Commit launcher and runbook**

  ```bash
  git add repeatability_combined_analysis/hpc/analysis.slurm repeatability_combined_analysis/hpc/submit.sh repeatability_combined_analysis/README.md repeatability_combined_analysis/tests/test_hpc_launcher.py
  git commit -m "feat: add guarded combined analysis launcher"
  ```

### Task 7: Full verification and handoff

**Files:**
- Modify if required: files created in Tasks 1-6 only
- Test: `repeatability_combined_analysis/tests/`

**Interfaces:**
- Consumes: complete package from Tasks 1-6.
- Produces: locally verified implementation and exact HPC preflight instructions; does not submit.

- [ ] **Step 1: Run the focused suite**

  Run: `python3 -m pytest repeatability_combined_analysis/tests -q`

  Expected: all tests PASS.

- [ ] **Step 2: Run affected canonical suites**

  Run:

  ```bash
  python3 -m pytest \
    ti_current_repair/tests/test_analyze_repeatability_optimizer_roi_metrics.py \
    ti_current_repair/tests/test_compare_simnibs_version_remesh_distributions.py \
    mesh_repeat_analysis/tests/test_prepare_spherical_fixed_paper_inputs.py -q
  ```

  Expected: all tests PASS; adjust the third path only if the repository's
  existing test filename differs, and record the actual command.

- [ ] **Step 3: Run static shell and repository checks**

  Run:

  ```bash
  bash -n repeatability_combined_analysis/hpc/submit.sh
  bash -n repeatability_combined_analysis/hpc/analysis.slurm
  git diff --check
  git status --short
  ```

  Expected: shell syntax and diff checks pass; status lists only intended work.

- [ ] **Step 4: Exercise a complete synthetic end-to-end run**

  Use test-only config injection to preflight, prepare, execute, status, archive,
  checksum, and rebind against a synthetic completed-dataset mapping. Assert
  exactly 3,200 rows, 80 cells, complete manifests, and no writes under any
  synthetic source.

- [ ] **Step 5: Audit generated scientific language and figures**

  Inspect the synthetic report and rendered figures for participant-level
  labeling, primary/sensitivity distinction, fixed-mesh control wording,
  aligned-not-identical ROI wording, and absence of therapeutic threshold
  claims.

- [ ] **Step 6: Run verification-before-completion checks**

  Re-run focused tests, canonical affected tests, shell checks, and
  `git diff --check` immediately before reporting completion.

- [ ] **Step 7: Commit final fixes and documentation**

  ```bash
  git add repeatability_combined_analysis docs/superpowers/specs/2026-11-03-final-132-combined-repeatability-analysis-design.md docs/superpowers/plans/2026-11-03-final-132-combined-repeatability-analysis.md
  git commit -m "docs: finalize combined repeatability analysis workflow"
  ```

- [ ] **Step 8: Hand off without HPC submission**

  Report the verified commit, test evidence, exact source-bound preflight
  command, expected numeric scope, output root, and the fact that the active
  copy remains untouched. Await the user's HPC output before any `run` command.
