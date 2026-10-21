# SimNIBS 3.2.6 finite-support forensic audit

This isolated workflow determines where finite values are lost inside the
optimizer-matched target ROI in the completed balanced-10 v3 campaign and
compares the same retained products with v4. It is read-only with respect to
all simulation, mesh, atlas, and log sources.

## Fixed full scope

- 10 participants, left hippocampus and right M1.
- `remesh` and `fixed_mesh`, 40 repeats each.
- 1,600 v3 runs and 1,600 matched v4 runs.
- Direct `TImax` element-field audit of all 1,600 retained v3 `TI.msh` files.
- Immutable attempt-log audit of all 1,600 v3 production task histories.
- 40 paired v3/v4 panels (repeat 001 for every subject-target-condition cell).
- 20 audit array tasks plus one collector; zero meshing or FEM jobs.

For every run the workflow separates:

1. non-finite values already present in direct v3 ROI mesh elements;
2. non-finite values appearing in the raw `msh2nii` field;
3. finite raw values removed by the `[WM, GM] == [1, 2]` tissue mask;
4. final-field inconsistencies not explained by that mask;
5. atlas/grid resampling and optimizer-sphere provenance.

The mesh localization uses tetrahedral barycentres transformed onto the exact
voxel ROI grid. A non-finite mesh element elsewhere in the head therefore does
not count as target-ROI evidence.

## Running on Stanage

From the project root:

```bash
bash simnibs_326_finite_support_audit/hpc/submit.sh scope
bash simnibs_326_finite_support_audit/hpc/submit.sh preflight
bash simnibs_326_finite_support_audit/hpc/submit.sh run
bash simnibs_326_finite_support_audit/hpc/submit.sh status
```

`run` always repeats the full read-only inventory before submission. It then
submits array `0-19%10` and an `afterok` collector. No job is submitted by
`scope` or `preflight`.

The array deliberately uses two Python environments. The SimNIBS 3.2.6 module
Python performs all NIfTI and direct mesh-field calculations. The existing
`$HOME/.conda/envs/ti-post/bin/python` interpreter renders the diagnostic
panels. `preflight` verifies that the rendering interpreter has Matplotlib,
Nibabel, and NumPy before any job is submitted. Override only the renderer with
`FINITE_SUPPORT_RENDER_PYTHON` if required; do not replace the SimNIBS 3.2.6
interpreter used for mesh inspection.

The default output root is:

```text
/mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_simnibs326_headreco/_simnibs326/finite_support_forensics_v1
```

Override it only with `FINITE_SUPPORT_AUDIT_ROOT`; the preflight rejects any
path overlapping the v3 roots, v4 roots, or atlas source.

## Collected outputs

- `all_runs.csv`: 1,600 paired run-level records;
- `audit_summary.csv`: 40 subject-target-condition summaries;
- `audit_summary.json`: full scope, cause counts, and grouped statistics;
- `audit_report.md`: one human-readable interpretation with the 40 group summaries;
- `cells/.../figures/*.png`: 40 paired panels;
- `cells/.../render_request.json` and `render_complete.json`: cross-environment
  rendering request and hashed completion evidence;
- `production_sacct.tsv`: scheduler snapshot for the four production arrays;
- `manifest.json` and `checksums.sha256`: auditable file inventory;
- `finite_support_forensics_v1.tar.gz` plus SHA-256 sidecar.

The collector fails closed unless all 20 cell receipts, 1,600 unique paired
rows, and 40 panels are present and the per-cell CSV, rendering request, and
panel checksums agree.
