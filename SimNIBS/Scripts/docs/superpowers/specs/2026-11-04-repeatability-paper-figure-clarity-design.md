# Repeatability paper-figure clarity redesign

## Intent

Create a second, image-only paper handoff in which a reader can understand every title, axis, legend and panel without knowing the analysis pipeline. Preserve all numerical values, statistical definitions and source files.

## Scope

- Start from the 12 PNG files currently present in `final_132_repeatability_paper_figures_v1`.
- Retain its three-folder structure: SimNIBS 3.2.6, SimNIBS 4, and cross-version comparisons.
- Leave the removed SimNIBS 3.2.6 ranking-summary figure out.
- Write a new `final_132_repeatability_paper_figures_v2_clarity` directory. Do not overwrite v1, the downloaded analysis, or the manuscript.

## Selected approach

The user selected a full reader-facing clarity pass. This preserves each scientific calculation while allowing titles, axis labels, legends, panel annotations, spacing and ordering to change where necessary. A wording-only pass was rejected because some displays also need layout and legend improvements. Manuscript restructuring was deferred because the current request is limited to figures.

## Reader-facing terminology

- Use `temporal-interference field` or `TIS field` rather than `optimizer-ROI field`, `TI field`, or `|E|` on the figure itself.
- Use `target region` on axes. The Methods and captions retain the exact optimizer-matched ROI definition.
- Use `Median temporal-interference field in target region (V/m)` for the primary spatial-median outcome.
- Use `Mean temporal-interference field in target region (V/m)` for the bootstrap spatial-mean outcome.
- Use `Run-to-run variability (SD, V/m)` instead of `within-cell SD`.
- Use `New mesh each run` and `Same mesh reused` instead of relying on `remesh` and `fixed mesh` alone.
- Spell out version differences and interaction contrasts in ordinary language. Do not use unexplained `difference-in-differences` wording.
- Use `participant`, never `subject`, in reader-facing labels.

## Visual rules

- Every title states the scientific question or displayed quantity, not an implementation detail.
- Panel titles identify the target only when the target is not already clear from the axes.
- Condition colours remain consistent: blue for a new mesh each run and orange for reuse of one mesh.
- Cross-version figures label `SimNIBS 3.2.6` and `SimNIBS 4` explicitly.
- Legends sit outside dense data regions where practical.
- Participant identifiers may omit the `sub-CC` prefix when space is limited, but the label must say `Participant`.
- Technical definitions, bootstrap mechanics and ROI-construction details belong in the caption, not in axis labels.

## Legacy figures

The mesh-element and tissue-volume figures do not have their source metric tables in the downloaded combined-analysis package. They will be retained only if their existing labels pass the reader-facing audit. They will not be reconstructed from pixels or altered in a way that could change plotted evidence.

## Validation

- Regenerate all non-legacy figures from the existing canonical CSV inputs.
- Confirm that the 12-file output contains PNG files only.
- Fail if regenerated labels contain `optimizer-ROI`, `|E|`, `within-cell`, `difference-in-differences`, or `Subject IDs`.
- Verify each PNG with Pillow and visually inspect every figure at full resolution and in contact sheets.
- Confirm the source analysis, v1 handoff and manuscript remain unchanged.

## Non-goals

- No recalculation of scientific results.
- No change to the manuscript, captions, tables or numbering in this task.
- No restoration of the removed SimNIBS 3.2.6 ranking-summary figure.
- No attempt to fabricate unavailable mesh or tissue source metrics.
