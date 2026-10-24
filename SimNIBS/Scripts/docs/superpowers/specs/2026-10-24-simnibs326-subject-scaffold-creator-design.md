# SimNIBS 3.2.6 Subject Scaffold Creator Design

## Purpose

Build a complete, reusable SimNIBS 3.2.6 `headreco` scaffold for one runtime-selected
final-132 subject. The first intended subject is `sub-CC510415`, but no subject ID is
hardcoded into the implementation.

The creator is an upstream prerequisite for `simnibs_subject_replacement`. It creates
the legacy v3 reference tree that the replacement workflow requires. It does not build
the corrected-label v3 head model and does not submit either the v3 or v4 FEM campaigns.

## Version and runtime contract

- SimNIBS module: `SimNIBS/3.2.6-foss-2023a`.
- MATLAB module: `MATLAB/2023b`, overridable only through the existing
  `SIMNIBS326_MATLAB_MODULE` mechanism.
- Segmentation command: SimNIBS 3.2.6 `headreco all --noclean`.
- The existing, hash-pinned CAT12 compatibility overlay is reused without copying or
  independently reimplementing it.
- The creator is explicitly not a SimNIBS 4 or CHARM workflow.

The observed runtime SimNIBS version, loaded module set, CAT12 patch identity, source
hashes, mesh hashes, and exact command are recorded in the completion evidence.

## Architecture

Create a separate `simnibs_326_subject_scaffold_creator/` package containing the
subject-facing CLI, validation, HPC submission, documentation, and tests. Extract the
existing reusable scaffold implementation from
`simnibs_326_repeatability/scaffold_runner.py` into a small shared core. The original
balanced-10 runner remains a backwards-compatible adapter around that core.

The new creator resolves the selected subject's T1, T2, and corrected-label provenance
from the established final-132 corrected-v4 scaffold inventory. It passes those exact
source paths to the shared builder; it does not create an unnecessary second input copy.

The shared core owns:

- the output layout;
- the `headreco` command;
- rebuilding an owned partial attempt from scratch;
- copying the native mesh into the compatibility location;
- completion-marker generation; and
- full scaffold validation.

The creator owns:

- canonical subject-ID and final-132 membership checks;
- corrected-source receipt and hash validation;
- output-root isolation and ownership;
- read-only preflight and status reporting;
- the one-subject Slurm chain; and
- creator-specific receipts and operational provenance.

## Inputs

The CLI accepts:

- `--subject=sub-CC######`;
- `--corrected-source-root`, defaulting to
  `/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Scaffolds`;
- `--output-root`, defaulting to
  `/mnt/parscratch/users/cop23bi/simnibs326_subject_scaffolds_v1`;
- `--final132-subjects`, defaulting to the repository's authoritative final-132
  `subjects.txt`; and
- an exact `--confirm-output-root` value for commands that can submit or write.

The subject must occur exactly once in the authoritative final-132 list. The corrected
source receipt must resolve exactly one valid T1, one valid T2, and one corrected label,
and every recorded hash must verify before submission or building.

## Output layout and retention

For subject `sub-CC510415`, the complete scaffold is retained under:

```text
<output-root>/sub-CC510415/anat/
  .simnibs326_scaffold_ready.json
  sub-CC510415.msh
  m2m_sub-CC510415/
    T1fs_conform.nii.gz
    T1fs_nu_conform.nii.gz
    eeg_positions/
    toMNI/
    *.stl
    sub-CC510415.msh
    ...all other headreco --noclean diagnostic products...
```

No successful `headreco` products are pruned. The complete output is intentionally kept
for diagnosis, visual inspection, future reuse, and exact provenance. The downstream
replacement workflow continues to stage only its restricted reference allowlist.

Creator-owned evidence lives separately under:

```text
<output-root>/_simnibs326/
  module_preflight.json
  subjects/<subject>/owner.json
  subjects/<subject>/complete.json
  logs/<subject>/...
```

The subject output may be deleted and rebuilt after a failed attempt only when its
creator ownership receipt matches the requested subject, source inventory, output root,
workflow, and SimNIBS module. An existing unowned or mismatched tree fails closed.

## Commands and scheduler behavior

The user-facing wrapper provides:

- `preflight`: read-only input, scope, ownership, and isolation audit; submits nothing;
- `module-check`: submit the existing SimNIBS 3.2.6/MATLAB/CAT12 module preflight;
- `submit`: run preflight, then submit the module check and one dependent scaffold task;
- `status`: report evidence and scheduler-independent output state without mutation; and
- `validate`: hash and structure validation of a completed scaffold.

The scaffold task is a one-element array (`0-0%1`) so the established exact-array-element
retry behavior remains available. It uses the existing resources: Sheffield partition,
8 CPUs, 32 GB, 8 hours, a 7-hour bounded worker timeout, one retry, append-only scheduler
output, and immutable timestamped attempt logs.

Submission is never implicit. Implementation and tests must not submit any HPC jobs.

## Validation and downstream handoff

A scaffold is ready only when all of the following hold:

- the source T1, T2, and corrected-label records remain hash-valid;
- `T1fs_conform.nii.gz` and `T1fs_nu_conform.nii.gz` exist and are non-empty;
- `eeg_positions/` and `toMNI/` exist and contain files;
- at least five STL surfaces exist;
- the native and compatibility meshes exist, are non-empty, and have equal SHA-256;
- the completion marker is valid and bound to the subject, inputs, module, and outputs;
- the creator owner and completion receipts are valid and bound to the same request; and
- no source file or protected production root was modified.

After validation, the existing replacement wrapper receives the creator output root as
its `--legacy-v3-root`. That wrapper stages the restricted reference tree, constructs and
reviews the corrected-label SimNIBS 3.2.6 head model, and only then releases the matched
v3 and v4 simulations.

## Failure behavior

- Invalid, duplicate, or non-final-132 subject IDs fail before writes or submission.
- Missing, empty, corrupt, or hash-mismatched source evidence fails before submission.
- Output overlap with source, established campaign, atlas, or repository inputs fails.
- A mismatched module receipt blocks the scaffold job.
- A failed `headreco` attempt preserves its immutable attempt log and complete partial
  `--noclean` output for inspection until the bounded retry begins.
- Retry exhaustion leaves the subject incomplete and never writes a ready receipt.
- Re-running a ready, hash-valid request is idempotent and reports reuse.

