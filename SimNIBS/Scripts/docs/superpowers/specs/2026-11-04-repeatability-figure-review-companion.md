# Repeatability Figure Review Companion Design

## Goal

Build a separate, versioned companion package for
`final_132_repeatability_combined_analysis_v1` that audits every supplied
figure, repairs misleading presentation, restores paper-style equivalents
where the downloaded data permit, and explains every figure in an illustrated
guide. The downloaded analysis and manuscript remain immutable.

## Source boundary

- Read only from
  `/home/boyan/sandbox/final_132_repeatability_combined_analysis_v1` and the
  current manuscript at
  `/home/boyan/sandbox/Jake_Data/Repeatability-Data/Current/Manuscript`.
- Treat the 3,200-row canonical analysis tables as the authoritative source for
  new field and bootstrap figures.
- Do not infer mesh-element counts or tissue volumes from file sizes or field
  summaries. The downloaded package does not contain those measures.
- Describe version-specific optimizer ROIs as aligned constructions, not
  identical masks.

## Deliverable

Create `final_132_repeatability_combined_analysis_v1_figure_review_v1` as a
companion package containing:

1. `FIGURE_GUIDE.md` and `FIGURE_GUIDE.pdf`, with an illustrated explanation
   and quality assessment of all 20 supplied PNG figures.
2. `figure_inventory.csv` with one row per supplied PNG and its dimensions,
   checksum, family, target, and audit disposition.
3. `paper_equivalence.csv` mapping all five current manuscript figures to an
   updated equivalent, legacy evidence, or an explicit data gap.
4. Repaired, combined figures for central values, technical variability,
   paired version contrasts, version-condition interactions, cross-version
   distribution location, and deterministic v3 ranking behavior.
5. Updated 9+1 paper equivalents for target-field repeatability,
   single-run ranking uncertainty, and bootstrap precision.
6. Immutable copies of the two legacy structural paper figures, labelled in
   the guide and manifest. The tissue illustration remains descriptive because
   its selected participants are retained; the element-count figure is not an
   updated 9+1 result because it includes the replaced cohort member.
7. `review_manifest.json` and `SHA256SUMS` covering every deliverable.

## Presentation rules

- Use two-panel target layouts where the paper compares hippocampus and M1.
- Avoid clipped boundary points and visually blank deterministic heatmaps.
- Use log scaling with an explicit plotting floor for technical SD values that
  span zero to approximately 7e-3 V/m.
- Use neutral paired lines unless participant identity is directly encoded.
- Use horizontal participant-labelled dot plots for interaction effects.
- Explain that technical repeats estimate within-cell variability and that the
  participant is the inferential unit.
- Keep fixed-mesh results framed as mechanistic controls.
- Do not use a therapeutic or biological threshold interpretation.

## Paper-equivalence policy

- Rebuild Figures 1 and 2 from the updated SimNIBS 4 established outputs.
- Recompute Figure 3 locally from the updated v4 remesh `roi_mean_v_per_m`
  records using the established 20,000-resample, seed-0 bootstrap method.
- Preserve Figure 4 only as legacy evidence and mark the current-cohort
  equivalent blocked pending mesh-metric extraction.
- Preserve Figure 5 as a legacy v4 illustration with explicit provenance; do
  not imply it represents a new cross-version analysis.

## Verification

- Validate source row counts, subject sets, conditions, targets, and finite
  metrics before rendering.
- Test inventory and equivalence statuses, percentile boundary handling,
  technical-SD floor encoding, bootstrap reproducibility, and manifest hashes.
- Render every PNG and every page of the PDF for visual inspection.
- Verify the original downloaded analysis tree is unchanged by comparing a
  pre/post file manifest.
