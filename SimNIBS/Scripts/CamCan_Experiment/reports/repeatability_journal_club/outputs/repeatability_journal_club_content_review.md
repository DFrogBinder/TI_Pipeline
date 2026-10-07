# Repeatability journal-club deck: content review

## Working thesis

Within the evaluated CHARM–SimNIBS workflow, tetrahedral mesh realization is the dominant source of repeated-run variability; percent-level field differences and close subject rankings should therefore be interpreted with mesh-induced uncertainty in mind.

## Assumed format

- Internal lab journal club
- 16–21 minute core talk plus discussion
- Core slides 1–16; backup slides 17–21
- Repeatability data freeze: corrected spherical-ROI and fully nested analysis generated 11 September 2026
- Software-version sensitivity: fixed MNI152 comparison supplied 25 September 2026

## Slide map

| # | Slide | Purpose | Figure/data status |
|---:|---|---|---|
| 1 | Mesh realization is part of the result | Open with the scientific claim rather than a generic project title. | Complete |
| 2 | The field estimate is mesh-dependent | State the entire argument and the principal numerical results up front. | Complete |
| 3 | Identical inputs still pass through a variable numerical representation | Explain where remeshing sits in the end-to-end simulation and what was held fixed. | Complete |
| 4 | Two complementary experiments separate pattern from source | Show how the 10-person cohort and fully nested subject answer different questions. | Complete |
| 5 | The contrast changes one thing: the mesh realization | Define exactly what each arm measures, the primary endpoint, and the inferential unit. | Complete |
| 6 | Hippocampal fields vary by 1.81–3.65% under remeshing | Show the participant-level spread in the deep target and the collapse under mesh reuse. | Complete |
| 7 | M1 shows the same pattern: 1.62–2.79% remesh CV | Demonstrate that the field variability pattern replicates in a superficial target. | Complete |
| 8 | Percent-level spread matters when subjects are closely matched | Translate CV into a decision-relevant consequence: uncertainty in subject ordering. | Complete |
| 9 | Remeshing changes the numerical head model—not only the final scalar | Provide structural evidence that each remesh run is a genuinely different discretization. | Complete |
| 10 | The fully nested case isolates between-mesh from within-mesh variation | Explain the 40-by-40 crossing and the variance components before showing the result. | Complete |
| 11 | The nested case assigns virtually all observed variance to mesh generation | Deliver the strongest source-attribution evidence and immediately qualify its scope. | Complete |
| 12 | Version sensitivity is systematic in amplitude—not spatial pattern | Show that software version can shift amplitude even when geometry and spatial pattern are held constant. | Complete |
| 13 | Within-version evidence converges on mesh realization | Synthesize the argument without overclaiming causality or accuracy. | Complete |
| 14 | The practical response depends on the scientific comparison | Convert the findings into decisions for simulation design, ranking, and reporting. | Complete |
| 15 | The result is strong about repeatability—and deliberately narrow about validity | Make the precise claim boundary visible rather than relegating it to a footnote. | Complete |
| 16 | If the claim is smaller than the mesh noise, the mesh belongs in the claim | Close with an actionable thesis and journal-club discussion prompts. | Complete |
| 17 | Appendix • full ranking analysis: left hippocampus | Provide the heatmap and Kendall distribution for questions. | Complete |
| 18 | Appendix • full ranking analysis: right M1 | Provide the heatmap and Kendall distribution for questions. | Complete |
| 19 | Appendix • cohort and stimulation parameters | Provide exact sample and montage details for methods questions. | Complete |
| 20 | Appendix • the current results use the corrected spherical-ROI workflow | Document the resolved selector mismatch and the finite-support sensitivity audit. | Complete |
| 21 | References and evidence sources | Provide literature context and identify the internal corrected data freeze. | Complete |

## Placeholders / optional additions after content approval

No critical scientific figure or analysis is missing from this draft, and no placeholders are required. Optional additions are:

- laboratory or university branding on the title slide
- an anatomical rendering of the two spherical ROIs if a preferred house figure exists
- a mesh close-up that visually localizes boundary differences between two remesh realizations
- final author/affiliation wording if this will be reused outside the lab

## Version guardrail

The deck deliberately uses the corrected spherical-ROI values rather than the older manuscript tables. The provenance slide documents the 18/20 fixed-mesh selection changes.
