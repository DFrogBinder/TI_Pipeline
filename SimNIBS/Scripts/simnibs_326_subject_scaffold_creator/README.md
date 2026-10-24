# Generic SimNIBS 3.2.6 subject scaffold creator

This package builds one complete legacy-v3 reference scaffold for a runtime-selected
final-132 participant. It is pinned to `SimNIBS/3.2.6-foss-2023a` with
`MATLAB/2023b` and the existing hash-verified CAT12 compatibility overlay. It runs
`headreco all --noclean`; it is not a SimNIBS 4/CHARM workflow.

The creator retains the complete diagnostic output from both successful and failed
attempts. It builds only the upstream v3 scaffold and does not submit FEM simulations.
Matched corrected-label v3 and v4 head models and simulations remain the responsibility
of `simnibs_subject_replacement` after this scaffold validates.

## Safety and output contract

- `preflight`, `status`, and `validate` are read-only and never call `sbatch`.
- `module-check` and `submit` require an exact `--confirm-output-root` value.
- The selected subject must be a canonical `sub-CC######` ID occurring exactly once in
  the authoritative final-132 list.
- T1, T2, and corrected-label paths are resolved from the corrected-v4 result receipt;
  both copied and gzip-compressed source records are hash-checked.
- The default isolated output root is
  `/mnt/parscratch/users/cop23bi/simnibs326_subject_scaffolds_v1`.
- A partial subject tree can be rebuilt only when its creator `owner.json` still matches
  the subject, source files and source-receipt hashes, authoritative final-132 list,
  output root, module, CAT12 patch, and workflow.
- Each subject completion receipt embeds the module/CAT12 evidence used for its build.
  A later equivalent module check may safely replace the shared current receipt without
  invalidating an earlier subject, while both the historical and current evidence remain
  available for audit.
- The Slurm worker is one array element (`0-0%1`), allows one exact-element retry, and
  keeps append-only scheduler output plus a distinct timestamped attempt log.

For `sub-CC510415`, the retained scaffold is under:

```text
/mnt/parscratch/users/cop23bi/simnibs326_subject_scaffolds_v1/
  sub-CC510415/anat/
    .simnibs326_scaffold_ready.json
    sub-CC510415.msh
    m2m_sub-CC510415/
      sub-CC510415.msh
      T1fs_conform.nii.gz
      T1fs_nu_conform.nii.gz
      eeg_positions/
      toMNI/
      *.stl
      ...complete headreco --noclean diagnostic products...
  _simnibs326/
    module_preflight.json
    subjects/sub-CC510415/{owner.json,complete.json}
    logs/sub-CC510415/
```

## 1. Read-only preflight

Run from `~/Repos/TI_Pipeline/SimNIBS/Scripts` on the Stanage login node:

```bash
cd ~/Repos/TI_Pipeline/SimNIBS/Scripts

SUBJECT=sub-CC510415
SCAFFOLD_ROOT=/mnt/parscratch/users/cop23bi/simnibs326_subject_scaffolds_v1
CREATOR=simnibs_326_subject_scaffold_creator/hpc/submit.sh

CREATOR_ARGS=(
    "--subject=$SUBJECT"
    "--output-root=$SCAFFOLD_ROOT"
)

bash "$CREATOR" preflight "${CREATOR_ARGS[@]}"
```

The JSON must report `status: ready`, the exact `SimNIBS/3.2.6-foss-2023a`
module, three hash-valid sources, zero jobs submitted, and the intended isolated output
root. Optional overrides are `--corrected-source-root=...` and
`--final132-subjects=...`.

## 2. Submit the module check and scaffold

The normal submission command creates a two-job chain: the existing module/CAT12 check,
then one dependent scaffold element.

```bash
bash "$CREATOR" submit "${CREATOR_ARGS[@]}" \
    "--confirm-output-root=$SCAFFOLD_ROOT"
```

To submit only the module check:

```bash
bash "$CREATOR" module-check "${CREATOR_ARGS[@]}" \
    "--confirm-output-root=$SCAFFOLD_ROOT"
```

Scheduler submission is not completion. Record both returned job IDs and wait for
terminal accounting before validation:

```bash
JOBS=<module-job-id>,<scaffold-array-job-id>

sacct -D -X -j "$JOBS" \
    --format=JobIDRaw,JobName%28,State,ExitCode,Elapsed,Restarts,Start,End

bash "$CREATOR" status "${CREATOR_ARGS[@]}"
bash "$CREATOR" validate "${CREATOR_ARGS[@]}"
```

Completion requires terminal zero-exit jobs and `validate` returning `status: ready`.
Inspect retry history as well as final state; a preserved `REQUEUED` first attempt is not
an unresolved failure when the qualified element later completes and validation passes.

## 3. Diagnose a failed candidate

Do not remove or manually adopt the subject tree. Inspect the immutable logs and the full
partial `--noclean` output:

```bash
SUBJECT_META="$SCAFFOLD_ROOT/_simnibs326/subjects/$SUBJECT"
SUBJECT_LOGS="$SCAFFOLD_ROOT/_simnibs326/logs/$SUBJECT"
SUBJECT_ANAT="$SCAFFOLD_ROOT/$SUBJECT/anat"

find "$SUBJECT_LOGS" -maxdepth 2 -type f -print | sort
find "$SUBJECT_ANAT" -maxdepth 3 -type f -printf '%s\t%p\n' | sort -n
sed -n '1,240p' "$SUBJECT_META/owner.json"
```

The worker automatically performs at most one exact-element retry. If it exhausts that
retry, preserve the owner receipt, logs, and partial subject tree for diagnosis. A later
rerun with the same bound request is allowed to rebuild it cleanly; a changed subject,
source receipt, or output root requires its own owner binding.

## 4. Hand off to the matched v3/v4 replacement workflow

Only after creator validation returns `ready`, pass the creator root—not the subject's
`anat` directory—as `--legacy-v3-root`:

```bash
LEGACY_V3_ROOT="$SCAFFOLD_ROOT"
REPLACEMENT=simnibs_subject_replacement/hpc/submit.sh

REPLACEMENT_ARGS=(
    "--subject=$SUBJECT"
    "--replaces=sub-CC520083"
    "--legacy-v3-root=$LEGACY_V3_ROOT"
)

bash "$REPLACEMENT" preflight "${REPLACEMENT_ARGS[@]}"
bash "$REPLACEMENT" prepare "${REPLACEMENT_ARGS[@]}"
```

Continue with the replacement runbook's unchanged v3 reference/head-model review,
promotion, v3 left/right campaigns, and v4 left/right campaigns. The replacement wrapper
still blocks simulations until the new v3 head has passed checksum collection and human
QC promotion.
