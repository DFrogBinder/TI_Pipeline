# Generic replacement-subject workflow

This folder orchestrates a complete replacement run for one eligible final-132
participant in both SimNIBS 3.2.6 and SimNIBS 4. The subject is supplied at run
time; changing candidate does not require editing source code.

The workflow creates new meshes in both versions. Per version it runs two targets
(left hippocampus and right M1), two conditions (remesh and fixed mesh), and 40
repeats: 160 simulation tasks and 320 component FEM solves. Across both versions
that is 320 simulation tasks and 640 component solves.

## Safety model

- `preflight` is read-only and submits nothing.
- `prepare` atomically creates a subject-specific root and stages exactly three
  corrected v4 inputs.
- The arbitrary-subject legacy v3 reference root is mandatory; there is no
  unverified default.
- The v3 head-model collection must be checksum-valid, visually reviewed, and
  explicitly promoted before either version can initialize or submit FEM work.
- Right M1 is blocked until the same version's left-hippocampus terminal receipt
  proves all 80 simulations complete.
- Existing full10, base9, historical v3, and historical v4 roots are protected
  and are never adopted or modified.
- Scheduler submission is not treated as completion. Use `status`, the delegated
  pipeline status, and terminal receipts.

## Prerequisite: create and validate the subject's v3 scaffold

For a subject that has not previously been processed with SimNIBS 3.2.6, first use
`simnibs_326_subject_scaffold_creator`. It runs the pinned
`SimNIBS/3.2.6-foss-2023a` `headreco all --noclean` workflow, retains complete
diagnostic output, and writes hash-bound owner and completion receipts. Follow that
package's runbook through terminal accounting and validation; submitting the scaffold
job is not sufficient.

After validation reports `status: ready`, the creator's output root is the replacement
workflow's legacy reference root:

```bash
SUBJECT=sub-CC510415
SCAFFOLD_CREATOR=simnibs_326_subject_scaffold_creator/hpc/submit.sh
LEGACY_V3_ROOT=/mnt/parscratch/users/cop23bi/simnibs326_subject_scaffolds_v1

bash "$SCAFFOLD_CREATOR" validate \
    "--subject=$SUBJECT" \
    "--output-root=$LEGACY_V3_ROOT"
```

This prerequisite creates no FEM simulations. The v3 head-model QC/promotion gate and
all matched v3/v4 simulation commands below are unchanged.

## Set the candidate once

Run from `~/Repos/TI_Pipeline/SimNIBS/Scripts` on the HPC login node:

```bash
SUBJECT=sub-CC510415
REPLACES=sub-CC520083
LEGACY_V3_ROOT=/mnt/parscratch/users/cop23bi/simnibs326_subject_scaffolds_v1
WORK_ROOT=/mnt/parscratch/users/cop23bi/simnibs_subject_replacements

REPLACEMENT=simnibs_subject_replacement/hpc/submit.sh
COMMON=(
    "--subject=$SUBJECT"
    "--replaces=$REPLACES"
    "--legacy-v3-root=$LEGACY_V3_ROOT"
    "--work-root=$WORK_ROOT"
)
```

Optional overrides are `--final132-subjects=...`,
`--corrected-v4-scaffold=...`, `--atlas-root=...`, `--targets-csv=...`, and
`--work-root=...`. The defaults point to the established final-132 list,
corrected-v4 scaffold, atlas, repository targets file, and subject-replacement
parscratch root.

## 1. Audit and prepare

```bash
bash "$REPLACEMENT" preflight "${COMMON[@]}"
bash "$REPLACEMENT" prepare "${COMMON[@]}"
bash "$REPLACEMENT" status "${COMMON[@]}"
```

Preflight must report one reference task, one v3 head-model task, 160 simulations
per version, 320 simulations overall, and zero submitted jobs.

## 2. Build and review the new v3 head

```bash
bash "$REPLACEMENT" v3-module-check "${COMMON[@]}"
bash "$REPLACEMENT" v3-stage-references "${COMMON[@]}"
```

The module check and reference staging may run together. Wait for both to finish
successfully and for their receipts to exist, then submit the head model:

```bash
bash "$REPLACEMENT" v3-headmodels "${COMMON[@]}"
```

If collection needs to be repeated without rebuilding the mesh:

```bash
bash "$REPLACEMENT" v3-collect-headmodels "${COMMON[@]}"
```

Download and inspect the one QC panel under:

```text
$WORK_ROOT/$SUBJECT/simnibs326/headmodels/_pipeline/headmodels/external1/qc_panels/
```

After visual review, calculate the manifest hash and promote it:

```bash
HEADS="$WORK_ROOT/$SUBJECT/simnibs326/headmodels/_pipeline/headmodels/external1"
MANIFEST_SHA256="$(sha256sum "$HEADS/manifest.json" | awk '{print $1}')"

bash "$REPLACEMENT" v3-promote "${COMMON[@]}" \
    "--manifest-sha256=$MANIFEST_SHA256" \
    --confirm-reviewed
```

Promotion failure blocks all following simulation commands, including v4.

## 3. Run v3 left, then v3 right

Initialize the v3 configs and run the complete left target. `v3-left-remesh`
submits only the left remesh array and its completion/ROI/median-seed preparation
chain. Once that chain is terminal and the fixed config exists, submit fixed:

```bash
bash "$REPLACEMENT" v3-init "${COMMON[@]}"
bash "$REPLACEMENT" v3-left-remesh "${COMMON[@]}"
bash "$REPLACEMENT" v3-status "${COMMON[@]}"
bash "$REPLACEMENT" v3-left-fixed "${COMMON[@]}"
```

Only after the left final receipt is complete:

```bash
bash "$REPLACEMENT" v3-right-remesh "${COMMON[@]}"
bash "$REPLACEMENT" v3-status "${COMMON[@]}"
bash "$REPLACEMENT" v3-right-fixed "${COMMON[@]}"
bash "$REPLACEMENT" v3-analysis "${COMMON[@]}"
```

If both target-level ROI collectors succeeded but only the final aggregation
failed, rerun the aggregation without duplicating extraction jobs:

```bash
bash "$REPLACEMENT" v3-analysis-final "${COMMON[@]}"
```

`--dependency=JOBID` may be supplied to the v3 remesh/fixed modes when an
explicit scheduler dependency is needed. The singleton analysis artifacts are
QC and provenance inputs for the later 9+1 aggregation, not cohort-level
inference.

## 4. Run v4 left, then v4 right

Each target is fully new: 40 new v4 remesh simulations, target-specific median
mesh selection, physical fixed-mesh seeding, and 40 fixed simulations.

```bash
bash "$REPLACEMENT" v4-init-left "${COMMON[@]}"
bash "$REPLACEMENT" v4-preflight-left "${COMMON[@]}"
bash "$REPLACEMENT" v4-submit-left "${COMMON[@]}"
bash "$REPLACEMENT" status "${COMMON[@]}"
```

The right-target commands remain blocked until the v4 left terminal receipt is
complete:

```bash
bash "$REPLACEMENT" v4-init-right "${COMMON[@]}"
bash "$REPLACEMENT" v4-preflight-right "${COMMON[@]}"
bash "$REPLACEMENT" v4-submit-right "${COMMON[@]}"
bash "$REPLACEMENT" status "${COMMON[@]}"
```

If a candidate fails during v3 head meshing, preserve its diagnostic evidence,
select another eligible final-132 subject, change only `SUBJECT`, and restart at
read-only preflight. A new subject automatically receives a separate root.
