# Combined Repeatability Dataset Consolidation and Retirement Design

## Intent

Create one physically self-contained, analysis-ready repeatability dataset at:

`/mnt/parscratch/users/cop23bi/final_132_repeatability_combined_dataset_v1`

The dataset replaces the scattered SimNIBS 3.2.6 and SimNIBS 4 campaign roots
as the retained working copy for the repeatability paper. It contains the nine
retained balanced-cohort subjects plus `sub-CC510415`, which replaces
`sub-CC520083`. After the copy has been fully validated, a separate guarded
workflow can archive obsolete v3 reference/scaffold roots to Shared and submit
the exact approved parscratch roots for deletion.

The purpose is to make later, unforeseen analyses straightforward without
leaving the only usable copy split across historical experiment layouts.

## Goals

- Produce one full physical copy: no symlinks, hard-link dependence, or paths
  that require the old simulation roots to remain.
- Present the scientific simulations as four canonical experiment arms:
  SimNIBS 3.2.6 and SimNIBS 4, each with left hippocampus and right M1.
- Retain 40 `remesh` and 40 `fixed_mesh` repeats for every subject, version,
  and target: 3,200 simulation records in total.
- Replace `sub-CC520083` with `sub-CC510415` and fail if either the excluded
  subject enters the destination or the replacement is missing.
- Retain campaign configuration, completion receipts, source-root provenance,
  per-run and aggregate analyses, nested-repeatability results, manifests,
  figures, and logs needed to interpret or reproduce the data.
- Record every copied source/destination mapping and validate it using SHA-256
  content comparison before writing a dataset completion receipt.
- Make copy and verification resumable and idempotent.
- Archive obsolete v3 reference/scaffold roots to
  `/shared/boyan/Shared/retired_repeatability_v3_sources_v1` on the login node,
  with an independent full-checksum completion receipt.
- Make retirement programmatic, exact-root allowlisted, dry-run by default,
  and impossible before both required completion receipts validate.

## Non-goals

- Do not regenerate meshes, fields, NIfTIs, ROI metrics, or paper analyses.
- Do not reinterpret scientific results or alter any scientific file.
- Do not change active SimNIBS launchers to read from `/shared`; Shared remains
  login-node-only.
- Do not delete primary MRI, atlas, corrected-segmentation, CHARM, or reusable
  scaffold inputs merely because simulations have been consolidated.
- Do not automatically delete anything at the end of `copy`, `verify`, or
  `checksum`.
- Do not treat submitted deletion jobs as proof that data was removed.

## Anti-goals

- A technically successful copy that still depends on an old source path is a
  failure.
- A whole-root dump containing the excluded subject or unexplained duplicate
  conditions is a failure.
- A completion receipt based only on file counts, sizes, timestamps, or Slurm
  success is a failure.
- Deleting a source because its destination merely exists is a failure.
- Allowing a caller-provided arbitrary path into the retirement allowlist is a
  failure.

## Constraints

- The HPC repository path is
  `/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts`.
- The output root is exactly
  `/mnt/parscratch/users/cop23bi/final_132_repeatability_combined_dataset_v1`
  unless an explicit test-only override is provided.
- The destination must be absent or owned by this workflow through a matching
  owner receipt. A partial or unowned directory is never adopted.
- Source roots are read-only to this workflow.
- Copy and verification are login-node filesystem operations with no Slurm
  array and no reduced scientific scope.
- `rsync` preserves directory structure, timestamps, and physical file
  contents. Symlinked scientific inputs are materialized as physical
  destination files only after preflight proves their targets resolve inside
  an approved source root; broken, cyclic, or out-of-scope links fail closed.
- Every destructive target must be a canonical, existing top-level directory
  immediately below `/mnt/parscratch/users/cop23bi` or the exact replacement
  subject root explicitly listed by this design.
- Dataset and archive completion receipts are immutable inputs to retirement;
  retirement records their SHA-256 values.

## Cohort and scientific scope

The canonical subject order replaces the faulty subject in place:

1. `sub-CC110174`
2. `sub-CC121144`
3. `sub-CC310407`
4. `sub-CC320616`
5. `sub-CC420071`
6. `sub-CC410432`
7. `sub-CC510415`
8. `sub-CC520127`
9. `sub-CC610631`
10. `sub-CC720941`

The required matrix is:

- versions: `simnibs326`, `simnibs4`;
- targets: `left_hippocampus`, `right_m1`;
- conditions: `remesh`, `fixed_mesh`;
- repeats: 40 per subject-condition;
- expected subject-condition cells: 80;
- expected simulation records: 3,200;
- excluded subject occurrences in the canonical scientific tree: zero.

## Destination layout

```text
final_132_repeatability_combined_dataset_v1/
├── README.md
├── subjects.txt
├── dataset_manifest.json
├── data/
│   ├── simnibs326/
│   │   ├── left_hippocampus/
│   │   │   └── sub-*_repeatability/{remesh,fixed_mesh}/...
│   │   └── right_m1/
│   │       └── sub-*_repeatability/{remesh,fixed_mesh}/...
│   └── simnibs4/
│       ├── left_hippocampus/
│       │   └── sub-*_repeatability/{remesh,fixed_mesh}/...
│       └── right_m1/
│           └── sub-*_repeatability/{remesh,fixed_mesh}/...
├── headmodels/
│   ├── simnibs326/
│   │   └── sub-*/...
│   └── simnibs4/
│       └── sub-*/...
├── analyses/
│   ├── repeatability_paper_update_v1/
│   ├── simnibs326_base9/
│   └── replacement_subject/
├── supplemental/
│   └── nested_40x40_v1/
├── provenance/
│   ├── source_roots/
│   ├── configs/historical/
│   ├── configs/canonical/
│   ├── receipts/
│   └── source_mapping.tsv
└── _consolidation/
    ├── owner.json
    ├── planned_items.json
    ├── copy_status.json
    ├── verification/
    └── complete.json
```

Each canonical target directory remains compatible with the existing
`<subject>_repeatability/<condition>/...` experiment layout. The workflow
merges only condition subtrees; it never merges or overwrites two sources into
the same condition. Root-level pipeline metadata from historical campaigns is
copied under `provenance/source_roots/<source-key>/` so conflicting `_pipeline`,
`_analysis`, `_post_processing`, `_figures`, or log directories remain intact
and attributable.

Scientific files are copied byte-for-byte. Reorganization changes only their
parent directory in the consolidated dataset. `source_mapping.tsv` records the
absolute original path, canonical destination, role, subject, version, target,
condition, and verification status.

Historical configs and receipts remain byte-identical under
`provenance/configs/historical` even when they contain absolute paths to retired
roots. The workflow additionally generates canonical-path analysis configs
under `provenance/configs/canonical`. Each generated config records its source
config and source SHA-256, changes only dataset-root path fields, and is
validated against the consolidated tree. This keeps the audit trail intact
while allowing supported post-processing to run after retirement.

## Authoritative source groups

### SimNIBS 4 scientific arms

For the retained nine subjects:

- left remesh:
  `/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10`
- right remesh:
  `/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_right_m1`
- left fixed:
  `/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_spherical_fixed_v1`
- right fixed:
  `/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_right_m1_spherical_fixed_v1`

For `sub-CC510415`, both conditions come from the corresponding target below:

- `/mnt/parscratch/users/cop23bi/simnibs_subject_replacements/sub-CC510415/simnibs4/left_hippocampus`
- `/mnt/parscratch/users/cop23bi/simnibs_subject_replacements/sub-CC510415/simnibs4/right_m1`

### SimNIBS 3.2.6 scientific arms

For the retained nine subjects, both conditions come from:

- `/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_simnibs326_harmonized_base9_left_hippocampus_v2`
- `/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_simnibs326_harmonized_base9_right_m1_v2`

For `sub-CC510415`, both conditions come from:

- `/mnt/parscratch/users/cop23bi/simnibs_subject_replacements/sub-CC510415/simnibs326/left_hippocampus`
- `/mnt/parscratch/users/cop23bi/simnibs_subject_replacements/sub-CC510415/simnibs326/right_m1`

### Retained analyses and provenance

- `/mnt/parscratch/users/cop23bi/final_132_repeatability_paper_update_v1`
- the complete physical tree under
  `/mnt/parscratch/users/cop23bi/final_132_repeatability_nested_40x40_v1`,
  including its underlying simulation outputs, analyses, manifests, and
  figures;
- root-level pipeline, analysis, post-processing, figure, log, config, and
  completion metadata from every scientific arm above;
- the replacement-subject v3 final-analysis directory and its verified archive;
- relevant harmonized-v3 head-model manifests, promotion receipts, label QC,
  and selected replacement-head-model metadata;
- the complete harmonized-v3 head-model bundles for the nine retained subjects
  from
  `/mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_simnibs326_harmonized_v2`,
  plus the corresponding replacement bundle under
  `/mnt/parscratch/users/cop23bi/simnibs_subject_replacements/sub-CC510415/simnibs326/headmodels`;
- the matching ten corrected-v4 subject scaffold subtrees under
  `/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Scaffolds/subjects`, plus
  their subject-specific provenance records under the scaffold root's
  `results/` directory.

The v3 head-model bundle includes the retained harmonized label map, generated
surfaces, initial tetrahedral mesh, staged T1fs references, EEG positions, MNI
transforms, completion receipts, QC, checksums, manifest, and promotion record.
The v4 bundle copies the complete subject-specific corrected scaffold for the
same ten subjects rather than only the three staged NIfTI inputs. The global
v4 scaffold root contains subjects outside this study and remains a protected
reusable input; it is neither copied wholesale nor retired.

The live preflight must resolve and inventory these paths. Missing required
scientific conditions, terminal receipts, or analysis roots block copying.
Optional diagnostic logs may be absent only when the planned item labels them
optional; optionality never applies to simulation records or completion
receipts.

## Workflow interface

Add `repeatability_dataset_consolidation/` with a dependency-free Python core,
a thin login-node shell entry point, tests, and a runbook.

The shell interface exposes these modes:

- `preflight`: read-only source, cohort, receipt, collision, and count audit;
- `prepare`: create an owned empty dataset root and immutable copy plan;
- `copy`: resumably copy every planned item;
- `verify`: read-only size/mtime comparison for every item;
- `checksum`: full-content comparison and, only on complete success, write
  `_consolidation/complete.json`;
- `status`: report planned, copied, verified, checksummed, and blocked counts;
- `archive-preflight`, `archive-copy`, `archive-checksum`: perform the separate
  login-node archive workflow for obsolete v3 sources;
- `retire-plan`: validate both completion receipts and print the exact deletion
  targets without deleting or submitting anything;
- `retire-submit`: require exact confirmations, revalidate all gates, and
  submit one guarded deletion job per allowlisted target using the existing
  hardened deletion launcher;
- `retire-status`: validate terminal deletion receipts and independently check
  target absence.

`preflight`, `status`, `archive-preflight`, `retire-plan`, and
`retire-status` are non-destructive. `copy` and archive copying never delete
sources. `retire-submit` is the only mode that can initiate deletion.

## Copy, verification, and completion

`prepare` creates an immutable plan containing each source/destination pair,
source identity, item role, required/optional state, and expected cohort cell.
Reruns accept an existing root only when `owner.json` and the plan hash match.

`copy` invokes resumable `rsync` separately for every planned subtree. It does
not use `--delete` against the dataset root. Approved in-scope symlinks are
materialized so the final dataset has no dependency on an old path. A copied
item is not considered verified merely because `rsync` returns success.

`verify` performs a read-only size/mtime comparison. `checksum` performs a
read-only content comparison. The completion receipt is written atomically only
when:

- every required planned item reports checksum match;
- all 10 subjects and 80 subject-condition cells are present;
- every cell contains exactly 40 validated repeat records;
- all 3,200 expected simulation records are present;
- `sub-CC520083` is absent from the canonical scientific tree;
- `sub-CC510415` is present in all eight version-target-condition cells;
- no symlink remains in the canonical scientific tree;
- every required source completion receipt and copied analysis manifest is
  present;
- a per-file destination inventory records relative path, byte size, and
  SHA-256 for every regular file;
- the manifest, source mapping, and dataset inventory hashes are recorded.

## Legacy-v3 Shared archive

The following obsolete/reference roots are archived as complete, unchanged
trees before they can be retired from parscratch:

- `final_132_repeatability_balanced_10_simnibs326_left_hippocampus_v1`
- `final_132_repeatability_balanced_10_simnibs326_right_m1_v1`
- `ti_dataset_final_132_balanced_10_simnibs326_headreco`
- `simnibs326_subject_scaffolds_v1`

The destination is
`/shared/boyan/Shared/retired_repeatability_v3_sources_v1/<source-name>`.
Archive copying is resumable and full-checksum verified. Its completion receipt
lives outside every deletion target under
`/shared/boyan/Shared/.repeatability_retirement_receipts/` and records all four
source/destination mappings and hashes. Because Shared is login-node-only,
these modes never submit Slurm work.

## Retirement policy

The exact retirement allowlist is split into two classes.

### Consolidated working sources

These may be deleted after the consolidated dataset completion receipt passes:

- the four SimNIBS 4 main/fixed roots listed above;
- the two SimNIBS 3.2.6 harmonized base-nine target roots;
- `/mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_simnibs326_harmonized_v2`,
  after all nine retained head-model bundles and their shared cohort receipts
  have checksum-matched the consolidated dataset;
- `/mnt/parscratch/users/cop23bi/simnibs_subject_replacements/sub-CC510415`;
- `/mnt/parscratch/users/cop23bi/final_132_repeatability_paper_update_v1`;
- `/mnt/parscratch/users/cop23bi/final_132_repeatability_nested_40x40_v1`.

### Archived obsolete v3 sources

The four legacy-v3 roots listed in the Shared archive section additionally
require the Shared archive completion receipt to pass.

The allowlist is code-owned and exact. CLI arguments may select a subset of
the allowlist but cannot add paths. Default retirement behavior is a dry run.
Submission requires all of:

- the exact dataset root confirmation;
- the exact dataset completion-receipt SHA-256;
- the exact Shared archive completion-receipt SHA-256 when archived roots are
  selected;
- an explicit `--confirm-retirement` flag;
- zero active Slurm jobs whose command, working directory, or job name is bound
  to a selected source, insofar as scheduler inspection can determine;
- immediate revalidation of canonical path, root containment, source identity,
  and receipt bindings.

The wrapper delegates deletion to the existing hardened one-directory-per-job
launcher. Completion is established only when every external terminal receipt
says `complete` and every target is independently absent. A retirement summary
receipt is then written outside all deleted roots under
`/mnt/parscratch/users/cop23bi/deletion_job_logs/repeatability_dataset_v1/`.

Primary inputs remain outside the retirement allowlist, including
`CamCan_Corrected_v4_Scaffolds`, `CamCanMRI`, `charm_segmentations`, corrected
label roots, `ti_dataset`, `ZIPs`, and `MNI152_SimNIBS401_inputs`.

## Failure behavior

- Missing, incomplete, ambiguous, or multiply mapped source conditions fail
  preflight before destination creation.
- An unowned or differently planned destination fails closed.
- Interrupted copies remain resumable but cannot produce completion receipts.
- Any content difference blocks completion and retirement.
- A changed source after plan creation invalidates the affected verification.
- A missing or changed completion receipt blocks retirement.
- A selected target outside the code-owned allowlist blocks retirement.
- A failed deletion job leaves a failed terminal receipt and is never reported
  as removed.

## Approaches considered

### Manifest-driven canonical physical copy — selected

Merge subject-condition subtrees into four canonical experiment roots, retain
historical metadata separately, and bind every copy to a source mapping and
checksum receipt. This is easiest to analyse later and permits safe retirement
without losing provenance.

### Whole-root mirror

Copy each historical root unchanged below the new dataset. This is the
simplest archive, but it preserves incompatible layouts, duplicates metadata,
keeps the replacement subject separate, and risks retaining the excluded
subject. It does not meet the analysis-readiness goal.

### Link-based catalogue

Build a clean index using symlinks or hard links into existing roots. This
would be fast and space-efficient, but deletion would break the dataset and it
would not be a full physical copy.

## Success criteria

- Preflight reports the exact 10-subject, 2-version, 2-target, 2-condition,
  40-repeat scope and makes no writes.
- Copy can resume after interruption without overwriting unrelated data.
- Checksum validation reports all required items matched and writes one atomic
  dataset completion receipt.
- The consolidated canonical tree contains 3,200 validated simulation records,
  includes `sub-CC510415`, and excludes `sub-CC520083`.
- The Shared archive checksum receipt covers all four obsolete v3 roots.
- Retirement dry-run prints only the exact allowlisted paths.
- Retirement refuses missing/stale receipts, arbitrary paths, source changes,
  wrong confirmations, and active-job conflicts.
- Real retirement submission creates one guarded job and external receipt per
  approved target; status requires terminal receipts plus independent absence.
- Existing HPC migration, deletion, SimNIBS 3.2.6, and replacement tests remain
  green.
