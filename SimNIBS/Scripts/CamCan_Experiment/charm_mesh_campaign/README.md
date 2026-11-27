# Configuration-driven CHARM meshing

This workflow creates exactly one mesh per validated CHARM
tissue-label map. It does not run segmentation, cortical reconstruction,
surfaces, FEM simulation, or electric-field extraction. The original label
files are read-only and are checked against their recorded SHA-256 hashes
before and after meshing.

The current backend is the existing direct CHARM `create_mesh` implementation.
The campaign config makes the input manifest, output layout, software version,
module, Python executable, CHARM settings policy, quality rules and Slurm
resources explicit. Existing `charm_segmentation_batch` commands remain valid.

## Configuration contract

The JSON schema is intentionally strict. Unknown fields, unsupported backends
or software versions, unsafe output templates, mixed segmentation/meshing
versions, and ambiguous settings are rejected. `realization_id` is part of the
task identity, so ten independent segmentations of one participant produce ten
tasks and ten distinct output paths—not 100 meshes.

When an inventory has only one segmentation per participant, set the
`realization_id` column to `null`; the normalized identity is then `seg_001`.
For repeated segmentations, point it at the inventory column containing stable
realization IDs such as `seg_001` through `seg_010`.

## Preliminary uncorrected SimNIBS 4.0.1 campaign

`configs/uncorrected_v401.example.json` contains the candidate paths and the
established Stanage resource profile. Its expected 652 tasks are **pending live preflight**.
The actual collection manifest remains authoritative; do not alter
the expected count merely to make a blocked preflight pass.

Run these commands from this directory on Stanage after pulling the branch:

```bash
cd /users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/charm_mesh_campaign
export PYTHONPATH=..

python -m charm_mesh_campaign.cli describe --config configs/uncorrected_v401.example.json
python -m charm_mesh_campaign.cli preflight --config configs/uncorrected_v401.example.json
python -m charm_mesh_campaign.cli submit --config configs/uncorrected_v401.example.json
```

The third command is a dry run: it prints the complete `sbatch` command but
does not submit anything. Inspect the preflight JSON and task manifest first.
Only the following explicit command submits the full validated scope:

```bash
python -m charm_mesh_campaign.cli submit --config configs/uncorrected_v401.example.json --submit
```

After the user-operated array has finished:

```bash
python -m charm_mesh_campaign.cli validate --config configs/uncorrected_v401.example.json
```

Use `--skip-hashes` only for a quick status check; the final campaign receipt
should use the full hash validation. A changed configuration, settings file,
source label, result marker or mesh invalidates reuse. With the example's
`overwrite: false`, stale outputs cause a task to stop rather than being
silently replaced.

## Adapting another campaign

Copy the example config to a new versioned file and change the manifest,
column mapping, output root, expected full task count, SimNIBS contract and
resources. Keep corrected, uncorrected, v4.0.1 and v4.6 products in separate
output roots. A new SimNIBS version is usable only after it is included in the
supported-version registry and its adapter tests pass; changing a module string
alone does not establish methodological compatibility.

The machine-readable outputs are:

- `campaign/task_manifest.tsv`: normalized one-row-per-segmentation work list;
- `campaign/preflight.json`: full-scope readiness and configuration hash;
- `results/<task_key>.json`: input, settings, software, mesh and QC provenance;
- `campaign/submission_receipt.json`: written only after explicit submission.
