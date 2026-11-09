# Single-subject SimNIBS segmentation determinism experiment

## Intent

Determine whether repeated segmentation of the same T1- and T2-weighted MRI
is deterministic within each of two production-relevant implementations:

1. SimNIBS 3.2.6 `headreco` using SPM12 and CAT12; and
2. SimNIBS 4.0.1 CHARM.

This is a small, isolated methods experiment. It does not compare electric
fields and does not attempt to make the different v3 and v4 tissue taxonomies
voxel-equivalent. The scientific question is whether each implementation
reproduces its own segmentation when its inputs and settings are unchanged.

## Selected approach

Run ten independent segmentation-only realizations per implementation for one
seed-selected participant. Each realization starts in a new working directory
from the same checksum-pinned source images. Only segmentation artifacts and
provenance are retained. A dependent collector performs within-arm image and
header comparisons after both arrays terminate.

The selected two-arm design was chosen over a three-arm design containing
SimNIBS 4.5.0 because the immediate question concerns the production versions
used in the repeatability study: SimNIBS 3.2.6 and SimNIBS 4.0.1. A 4.5.0 arm
can be added later as a separately versioned experiment.

Running complete `headreco all` and CHARM mesh-generation workflows was
rejected because it would confound segmentation with surface preparation and
tetrahedral meshing.

## Scientific scope

- Subject count: 1.
- Candidate subject: `sub-CC110174`, selected with seed `20261106`.
- Implementations: 2.
- Repeats per implementation: 10.
- Scientific segmentation tasks: 20.
- Expected successful repeat directories: 20.
- Expected primary hard-label/binary result sets: 10 per implementation.
- Surface-generation tasks: 0.
- Tetrahedral-meshing tasks: 0.
- FEM simulation tasks: 0.
- Source MRI files modified: 0.

The authoritative cohort record is the tracked file:

```text
output/spreadsheet/final_132_repeatability_balanced_10/subjects.txt
SHA-256: 3cd591021466d414c5ab5019944b6cb698dd9de69577cf3d7cd10cc878027767
```

The corresponding tracked selection audit is:

```text
output/spreadsheet/final_132_repeatability_balanced_10/selection_audit.xlsx
SHA-256: 3d802d1308aee36dc46f9a6c547b35ef9eb4a5f1b96023032afb52513952868d
```

Preflight verifies both tracked checksums and confirms that `sub-CC110174` is
the first unique subject in the pinned cohort file. The seeded selection is
reproduced as `random.Random(20261106).choice(subjects)` over the exact ordered
subject list, and must return `sub-CC110174`. If the files or selection differ,
or either MRI is absent, preflight fails; subject selection is not silently
repeated with a different cohort.

## Source and isolation contract

The default Stanage source is the read-only balanced-ten corrected dataset:

```text
/mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_corrected
```

The experiment reads exactly:

```text
sub-CC110174/anat/sub-CC110174_T1w.nii[.gz]
sub-CC110174/anat/sub-CC110174_T2w.nii[.gz]
```

Preflight resolves one unambiguous T1 and one unambiguous T2, records their
absolute paths, sizes and SHA-256 hashes, and verifies that every task receives
those same files. The source root, existing repeatability roots, consolidated
dataset, corrected label maps and atlases are protected read-only roots.

The proposed isolated output root is:

```text
/mnt/parscratch/users/cop23bi/simnibs_segmentation_determinism_v1
```

Initialization refuses an existing non-workflow directory and never deletes
or overwrites another campaign. Regeneration is limited to an explicitly
named repeat under this root.

## Stage boundaries

### SimNIBS 3.2.6 headreco arm

Module and compatibility boundary:

```text
SimNIBS/3.2.6-foss-2023a
MATLAB/2023b
```

The arm runs only:

```text
headreco preparevols --noclean sub-CC110174 <T1> <T2>
```

The exact argument ordering is checked against the installed command before
submission. The existing bounded NumPy/NiBabel compatibility shim and the
hash-verified CAT12 XML compatibility overlay from
`simnibs_326_repeatability` are reused rather than reimplemented.

The official SimNIBS 3.2.6 source defines `preparevols` as the stage that:

- reorients/conforms the inputs;
- coregisters T2 to T1;
- runs SPM12 segmentation;
- runs CAT12 segmentation;
- binarizes the posterior probability maps; and
- writes the associated transformations and templates.

The workflow must not invoke `preparecat`, `cleanvols`, `surfacemesh`,
`volumemesh`, `all`, electrode placement or simulation. The runner searches
its complete command history and retained log for those forbidden downstream
stage invocations and fails the repeat if any are present.

The retained scientific artifacts are the conformed image geometry and the
SPM/CAT probability and binarized tissue maps created by `preparevols`.
Artifact discovery is pattern-based but fail-closed: the expected patterns and
counts are recorded during module preflight, and every repeat must produce the
same relative artifact set before it can be marked complete. Surface files
created internally by CAT12 are not promoted as primary results and no later
surface-preparation stage is called.

### SimNIBS 4.0.1 CHARM arm

Module boundary:

```text
SimNIBS/4.0.1-foss-2023a
```

The established segmentation-only invocation is reused:

```text
charm sub-CC110174 <T1> <T2> \
  --registerT2 --initatlas --segment --forceqform
```

`--surfaces` and `--mesh` are forbidden. The primary scientific artifact is:

```text
m2m_sub-CC110174/label_prep/tissue_labeling_upsampled.nii.gz
```

The existing `CamCan_Experiment/charm_segmentation_batch` runner may provide
the process, timeout, atomic-copy and provenance patterns, but this experiment
uses new repeat-specific output paths and must not write into the historical
cohort collection.

## Independence of repeats

Every repeat passes the identical canonical subject ID `sub-CC110174` to the
segmentation tool. Repeat identity is encoded only in the parent output path,
array index and receipts; it is not an input to headreco or CHARM. Each repeat
uses a new task-local working directory, so the identical `m2m_sub-CC110174`
name cannot collide across runs. No repeat may read another repeat's `m2m_*`,
temporary directory, completion marker or segmentation output. Input MRIs may
be copied into local scratch, but their hashes must be checked before
execution.

Outputs are promoted atomically only after all required artifacts validate.
An existing valid repeat is reused only by an explicit resume operation; a
normal new run never treats another repeat as a cache. A failed attempt retains
its immutable log and cannot leave a success marker.

At most one exact-array-element retry is allowed for an operational failure.
Retry provenance is retained and the scientific report identifies repeats
that required a retry. A retry starts from an empty task-local work directory
and does not increase the intended count of ten successful realizations.

## Proposed output layout

```text
simnibs_segmentation_determinism_v1/
├── _pipeline/
│   ├── config.json
│   ├── source_manifest.json
│   ├── preflight.json
│   ├── submission.json
│   ├── job_ids.tsv
│   ├── logs/
│   └── complete.json
├── simnibs326_headreco/
│   ├── repeat_001/
│   │   ├── artifacts/
│   │   ├── artifact_manifest.json
│   │   └── complete.json
│   └── repeat_002/ ... repeat_010/
├── simnibs401_charm/
│   ├── repeat_001/
│   │   ├── tissue_labeling_upsampled.nii.gz
│   │   ├── artifact_manifest.json
│   │   └── complete.json
│   └── repeat_002/ ... repeat_010/
└── analysis/
    ├── determinism_manifest.json
    ├── artifact_inventory.tsv
    ├── repeat_comparisons.tsv
    ├── tissue_summary.tsv
    ├── changed_voxel_summary.tsv
    ├── figures/
    └── report.md
```

## Preflight and submission design

The workflow provides separate, idempotent commands for:

- `scope`: print the exact scientific and scheduler scope;
- `preflight`: read-only validation; submit nothing and create no output root;
- `init`: create the isolated configuration and manifests; submit nothing;
- `run`: submit the complete two-arm experiment;
- `status`: read-only inspection of jobs, repeats and receipts;
- `analyze`: rerun only the read-only collector/analysis after all outputs are
  complete.

`run` submits:

1. one SimNIBS 3.2.6/MATLAB compute-node module preflight;
2. one SimNIBS 4.0.1 compute-node module preflight;
3. a `0-9%10` v3 array after the v3 preflight succeeds;
4. a `0-9%10` v4 array after the v4 preflight succeeds; and
5. one read-only collector after both arrays terminate.

This is 20 scientific segmentation tasks and 23 scheduler tasks when array
elements are counted individually. The collector uses `afterany` so it can
write an incomplete audit even if a repeat fails, but it writes the terminal
success receipt only when all 20 repeat receipts validate.

The initial candidate resource profile preserves the nearest established
settings for both arms:

```text
partition: sheffield
CPUs per segmentation task: 8
memory per segmentation task: 32G
time per segmentation task: 08:00:00
array concurrency: 10 per arm
SimNIBS 3.2.6 MATLAB module: MATLAB/2023b
```

These settings are candidate settings for this experiment until Stanage
evidence is recorded. No submission occurs while the current multi-terabyte
dataset checksum process is active on the same login host unless the user
explicitly chooses to overlap them; code development and read-only local tests
can continue independently.

## Determinism analysis

Determinism is evaluated within each implementation. V3 and v4 tissue labels
are not directly compared because their tissue models and output structures
differ.

For every retained NIfTI artifact, the analysis records:

- relative artifact name and semantic class;
- dimensions and voxel sizes;
- qform, sform and effective affine;
- data type and scaling fields;
- file SHA-256;
- canonical uncompressed array SHA-256;
- finite/non-finite voxel counts; and
- per-label voxel count and physical volume for discrete maps.

Each repeat is compared with repeat 001. If a map differs, all 45 pairwise
repeat comparisons are calculated for that artifact. Discrete maps report
changed-voxel count and fraction, transition counts and tissue-wise Dice.
Floating probability maps report exact-array equality, maximum and mean
absolute difference, and voxel counts with absolute differences greater than
`1e-7`, `1e-6`, and `1e-5`, using zero relative tolerance. These fixed
thresholds distinguish changes around float32 precision from progressively
larger numerical differences. They are descriptive only and do not redefine
bitwise determinism.

Compressed-file hashes alone are not used to classify determinism because
gzip metadata may change while the NIfTI arrays remain identical.

The report classifies separately:

- geometric determinism: identical grid, affine and header geometry;
- discrete segmentation determinism: identical hard-label/binary arrays;
- probability-map determinism: identical floating arrays; and
- operational determinism: identical artifact inventory and successful stage
  completion without retry-dependent differences.

## Completion gates

The experiment is complete only when:

- the authoritative subject and both input hashes are recorded;
- both module preflights report ready;
- exactly ten valid repeat receipts exist for each arm;
- every repeat uses the pinned input hashes and expected module version;
- no repeat reports a forbidden downstream stage;
- the artifact inventory is consistent within each arm;
- the collector and analysis finish successfully;
- `determinism_manifest.json` and `report.md` exist; and
- `_pipeline/complete.json` records the source hashes, code commit, job IDs,
  repeat receipts and analysis hashes.

No scientific conclusion is inferred from scheduler completion alone.

## Goals

- Answer whether segmentation results are repeatable for unchanged inputs.
- Preserve enough evidence to distinguish image-content differences from
  compression, header or execution differences.
- Keep both existing simulation campaigns and the active consolidated dataset
  untouched.
- Make failures and retries visible rather than silently replacing outputs.

## Non-goals

- Comparing v3 and v4 tissue labels voxel-for-voxel.
- Determining which segmentation is anatomically more accurate.
- Testing surface reconstruction, tetrahedral meshing, conductivities,
  electrode placement, FEM solvers or electric fields.
- Testing SimNIBS 4.5.0.
- Expanding beyond one participant or ten repeats per arm.

## Anti-goals

- Calling `headreco all` or CHARM `--mesh` for convenience.
- Reusing an existing head model or segmentation as one of the ten repeats.
- Treating `.nii.gz` byte inequality as voxel inequality.
- Hiding failed attempts or scheduler retries.
- Writing into or deleting any completed repeatability campaign.

## Implementation and validation expectations

Implementation should be a new isolated package rather than behavior added to
the production repeatability workflows. It may import small, stable helpers,
but existing launchers retain their defaults.

Tests must cover scope counts, subject/input fail-closed behavior, protected
root collisions, exact repeat-to-array mapping, forbidden stage rejection,
module/version receipts, atomic promotion, retry provenance, NIfTI geometry
and voxel comparisons, gzip-hash independence, incomplete collection and
terminal receipt generation. Modified shell scripts require `bash -n` and
ShellCheck; Python requires the repository's focused test and lint checks.

Before any HPC command is handed over, the launcher must print the numerical
scope and a read-only preflight must pass on the current Stanage checkout.
