# Modular CHARM Meshing Campaign Design

## Intent

Refactor the collected-segmentation meshing workflow into a reusable, configuration-driven CHARM meshing campaign. The immediate use is exactly one new tetrahedral mesh for every validated, uncorrected SimNIBS 4.0.1 CHARM segmentation. The same core must also support corrected maps, repeated segmentations of one participant, and later explicitly supported SimNIBS versions without copying or rewriting campaign code.

## Current constraints

The existing direct mesher is scientifically suitable and has a working Stanage history, but its orchestration is rigid:

- the worker loads `SimNIBS/4.0.1-foss-2023a` directly;
- `charm.ini` is selected implicitly from the imported installation;
- the submitter defaults to 474 unique subjects and describes every run as CamCAN;
- one subject is one task and one output path, so segmentation repetitions collide;
- paths, scientific settings and scheduler resources are distributed across CLI arguments and environment variables;
- runtime SimNIBS compatibility is recorded after meshing but not required before meshing.

The validated numerical implementation must be preserved: load an immutable label map, use CHARM's `create_mesh` with `charm.ini` settings, write and reload the mesh, validate tetrahedra and tissue tags, atomically install the result, confirm that the source-label hash is unchanged, and write provenance.

## Goals

1. Make all campaign inputs explicit in a versioned JSON configuration.
2. Generate exactly one mesh per ready manifest row.
3. Support repeated segmentation realizations of the same participant through a distinct `realization_id` and collision-free output key.
4. Require an explicit SimNIBS version, module, Python command, meshing backend and settings policy.
5. Fail closed if the imported SimNIBS version does not satisfy the configured version.
6. Record configuration, settings, input and output hashes in every result.
7. Keep paths within a declared output root and refuse duplicate output paths.
8. Preserve dry-run, preflight, resume, retry and aggregate validation behaviour.
9. Keep the current Python CLI, shell submitter and result layout operational for existing campaigns.
10. Keep HPC execution user-operated; local work covers implementation and validation only.

## Non-goals

- Do not run CHARM segmentation, cortical surfaces, EEG transformations, FEM simulation or field extraction.
- Do not mutate the source segmentation maps.
- Do not claim support for an arbitrary SimNIBS release. A release is supported only when its adapter contract and tests pass.
- Do not silently translate a SimNIBS 4.0.1 segmentation into a 4.6 methodological condition.
- Do not change established SimNIBS 4.0.1 mesh settings or the current legacy output layout.
- Do not automatically submit a reduced pilot or full HPC campaign.

## Selected architecture

### Versioned campaign configuration

Add `CamCan_Experiment/charm_mesh_campaign/` with a standard-library-only configuration layer. A configuration contains:

- `schema_version` and `campaign_id`;
- input manifest path, delimiter, ready status and column mapping;
- declared input segmentation version;
- output root and relative mesh/result path templates;
- expected task count and optional explicit exclusions;
- meshing backend, expected SimNIBS version, module, Python command and settings path/policy;
- required mesh properties and tissue-tag rules;
- partition, CPU, memory, time, concurrency, packing, local staging and retry settings;
- resume/reuse and overwrite policies.

Configuration paths are resolved relative to the configuration file unless already absolute. The normalized configuration is SHA-256 hashed. Unknown fields, missing required fields, unsafe templates, unsupported backends and unsupported schema versions are errors.

### Generic task manifest

The campaign preflight reads the configured input manifest through explicit column mappings and emits an immutable normalized task TSV. Each row contains:

- `task_id`;
- `subject_id`;
- `realization_id`;
- `task_key`;
- input label path/hash/size;
- mesh and result paths;
- configuration hash;
- status and message.

`task_key` is unique even when one subject has several segmentation realizations. Mesh/result templates must resolve beneath the output root and must be unique across tasks. Preflight verifies every input hash and exact task count before returning `ready`.

### Meshing backend contract

The initial backend is `direct_charm_create_mesh`. It reuses the established meshing primitive and CHARM settings translation. Before loading label data it verifies:

- imported SimNIBS version equals the configured version under a normalized comparison;
- resolved settings file exists and its SHA-256 matches the preflight record;
- configured backend is supported by the active adapter.

The adapter filters mesh options against the installed `create_mesh` signature as the existing implementation does. A future SimNIBS version can add or update an adapter without changing discovery, task identity, provenance, Slurm or validation code. `meshmesh` is not substituted for the validated v4.0.1 backend.

### Orchestration

Add a Python CLI with `preflight`, `run-task`, `validate`, `describe` and `submit` commands. `submit` prints the complete scope and exact `sbatch` command; it submits only with an explicit `--submit` flag. The worker shell script accepts the module and Python command from the immutable campaign configuration/export rather than embedding v4.0.1.

Scientific settings and scheduler settings remain separate in the configuration and provenance. Worker packing remains optional. The default is one mesh worker per array element.

### Compatibility boundary

The following remain valid:

- imports and function signatures exercised by `test_mesh_collected_segmentations.py`;
- `mesh_collected_segmentations.py preflight|run-task|validate`;
- `submit_collected_meshes.sh` environment variables;
- `mesh_collected_segmentations_array.slurm` defaults;
- legacy output layout `subjects/<subject>/anat/m2m_<subject>/<subject>.msh`;
- existing result markers and validation logic.

Legacy scripts become or remain compatibility wrappers over validated primitives. New campaign configuration is opt-in. Default legacy behaviour is not changed, except that the worker's current hard-coded module is expressed as a defaulted environment variable so existing calls resolve to the same module.

## Immediate uncorrected-v4.0.1 campaign

Provide a candidate example configuration for:

- input collection manifest: `/mnt/parscratch/users/cop23bi/charm_segmentations/collection/charm_segmentation_manifest.tsv`;
- segmentation and meshing version: SimNIBS 4.0.1;
- module: `SimNIBS/4.0.1-foss-2023a`;
- backend: `direct_charm_create_mesh`;
- output root: `/mnt/parscratch/users/cop23bi/charm_segmentation_uncorrected_meshes_v1`;
- resources inherited from the established direct-meshing profile: Sheffield, 8 CPUs, 32 GB and 8 hours, concurrency 50, retries 2;
- no segmentation, surfaces or simulations.

The example uses the recorded 652-task cohort as a candidate expectation. The live HPC preflight remains authoritative and must report the exact ready count before any user-operated submission.

## Provenance and validation

Every result records:

- campaign ID and configuration hash;
- task, participant and realization identity;
- input label path/hash/size and tissue tags;
- backend, expected/actual SimNIBS versions, module and Python command;
- settings path/hash;
- mesh path/hash/size, tetrahedra and tissue tags;
- resource-relevant thread/staging information;
- start/completion timestamps;
- explicit false flags for segmentation, surfaces and simulation.

Validation requires the result identity and all recorded hashes to match the current immutable task. A changed configuration, settings file, label or mesh invalidates reuse.

## Backward-compatibility validation

Before completion:

1. Run all new configuration/campaign tests.
2. Run the existing collected-meshing tests unchanged.
3. Run existing CHARM-only remesh, cohort-pipeline and HPC-launcher tests.
4. Run Bash syntax and ShellCheck on new and modified launchers.
5. Compare legacy submitter scope, resource defaults, array construction, result paths and provenance fields before and after.
6. Run a synthetic legacy task and a synthetic configuration-driven task and compare mesh/QC payloads except for the new provenance fields.
7. Report any broader repository collection failures separately rather than representing them as regressions caused by this work.

## Considered alternatives

### Parameterize the current scripts only

This is smaller but leaves input schema, task identity, output layout and version logic entangled. It does not safely support repeated segmentations and would require further rewrites.

### Replace the workflow with Snakemake or Nextflow

This would provide general orchestration but adds a new operational dependency and discards proven Slurm/retry behaviour. It is disproportionate to a focused CHARM meshing workflow.

The selected configuration-driven core plus compatibility wrappers provides reuse without changing the validated scientific operation.
