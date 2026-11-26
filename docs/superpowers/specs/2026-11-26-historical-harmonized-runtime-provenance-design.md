# Historical Harmonized Runtime Provenance Design

## Intent

Allow the read-only historical native-field preflight to validate the existing
SimNIBS 3.2.6 harmonized head-model task manifests in the completed combined
repeatability dataset. The production manifests already contain complete
runtime provenance; the consumer currently checks the wrong producer key.

## Scope

- Read `simnibs_326_harmonized_runtime` for consolidated `simnibs326` runs.
- Require the production SimNIBS version, module, and harmonized backend.
- Keep the existing subject, condition, repeat, stimulation, montage, and
  current checks unchanged.
- Use production-shaped tests and reject legacy-only or mismatched runtime
  provenance.

## Non-goals

- Do not modify the combined dataset or any task manifest.
- Do not change extraction, ROI construction, field metrics, or analysis.
- Do not change the historical launcher, Slurm resources, arrays, paths, or
  duplicate-validation behavior in this minimal repair.
- Do not require historical `config` paths to remain present after source
  retirement.

## Approaches

1. Rewrite or supplement the consolidated manifests. Rejected because the
   completed dataset is immutable and already records the correct provenance.
2. Accept either the legacy or harmonized runtime key. Rejected because that
   weakens the provenance boundary and could admit a different v3 workflow.
3. Require the harmonized runtime key and exact producer identity. Selected
   because it matches the production dataset and remains fail closed.

## Validation contract

For every consolidated `simnibs326` task manifest, require a mapping at
`simnibs_326_harmonized_runtime` with:

- `simnibs_version`: `3.2.6`
- `simnibs_module`: `SimNIBS/3.2.6-foss-2023a`
- `backend`: `simnibs-3.2.6-harmonized-corrected-label-v2`

The already recorded hashes remain preserved as provenance but are not given
new validation semantics in this minimal correction. Historical path strings
are not dereferenced.
