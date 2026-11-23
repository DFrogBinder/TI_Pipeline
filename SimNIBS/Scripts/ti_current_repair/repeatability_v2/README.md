# Repeatability pipeline v2

This package is the isolated implementation of the approved native-mesh
repeatability rewrite. Existing v1 scripts and historical SimNIBS 3.2.6/4.0.1
sources are not modified.

## Scientific contract

- Production is exactly SimNIBS 4.6.0, the canonical 9+1 cohort, two targets,
  two conditions, and 40 repeats.
- Native `.msh` `ElementData` stored and analysed as float64 is authoritative.
  NIfTI output is float32 qualitative QC under `_derived_qc` only.
- Remesh repeats rebuild the corrected-map head and both electrode pairs.
- Fixed repeats solve the two immutable, pair-specific electrode-augmented
  caches directly through `fem.tdcs`; electrode placement is unavailable in
  that path.
- The three outcomes are TI in the established target ROI, pair-1 E magnitude
  in that exact ROI, and pair-1 E magnitude in the 100 mm3 GM-only ROI under
  F8/Fp2.

## Main campaign gates

Create a schema-v2 operational profile containing the canonical
`StudyConfigV2`, new and mutually isolated output/cache roots, the confirmed
montage preset names, and all subject paths. The configured hashes are checked
against the actual corrected segmentations, atlases, and `targets.csv` during
preflight; T1/T2 inputs must also exist and cannot be symbolic links.

```bash
export V2_PROFILE=/absolute/path/repeatability-v2-profile.json
export V2_EVIDENCE=/absolute/path/preflight-evidence.json
export V2_OUTPUT_ROOT=/absolute/new/path/repeatability-v2-production
export SIMNIBS_MODULE=SimNIBS/4.6.0
export EXPECTED_SIMNIBS_VERSION=4.6.0

ti_current_repair/repeatability_v2/hpc/submit.sh preflight
```

`submit.sh run` also requires completed `V2_LOCAL_VALIDATION` and
`V2_HPC_PILOT` receipts, including a passed local-versus-HPC comparison. It
submits the exact eight-stage, 1,600-simulation graph and permits one automatic
retry. Preflight itself creates no output root and submits no job.

## Post-campaign entry points

- `v3_audit.audit_v3_fixed_geometry` makes the retain/full-800-replacement
  decision and `probe_existing_mesh_solver` fails closed on the v3 API.
- `segmentation_propagation.run_charm45_propagation` executes the isolated
  five-map, ten-simulation CHARM 4.5 propagation with concrete correction,
  meshing, and simulation callbacks.
- `repeatability_combined_analysis_v2.workflow.run_final_analysis` enforces the
  exact 9,600-row native matrix and writes participant-level outputs.
- `repeatability_figure_review.v2_figures.render_v2_figure_package` writes the
  matched-target, under-anode, grayscale, source-table, and pipeline artifacts.
- `packaging.build_checksummed_package` and
  `packaging.verify_checksummed_package` create and independently validate an
  immutable handoff package.
- `retirement.build_retirement_candidate_manifest` inventories candidates;
  even dual approval only makes them eligible for a separately reviewed manual
  workflow. This implementation contains no delete operation.

