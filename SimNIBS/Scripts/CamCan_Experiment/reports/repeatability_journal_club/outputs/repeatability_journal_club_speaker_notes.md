# Repeatability journal-club speaker notes

Core talk: slides 1–16 (about 16–21 minutes). Slides 17–21 are backup.

## 1. Mesh realization is part of the result

Opening (about 40 seconds): Individualized TI simulations are usually spoken about as though one anatomy and one montage produce one field map. This experiment asks whether that output is actually unique when the tetrahedral mesh is regenerated. The answer is no: in this workflow, the mesh realization contributes percent-level uncertainty.

        Preview the two experiments: a ten-participant remesh-versus-fixed comparison and one fully nested 40-by-40 case that separates between-mesh from within-mesh variation.

## 2. The field estimate is mesh-dependent

The cohort analysis shows remesh CVs of 1.81–3.65% in left hippocampus and 1.62–2.79% in right M1. A single remesh repeat often changes at least one pairwise subject ordering. In the fully nested subject, 99.9996% of observed variance is assigned to differences between meshes.

        Emphasize the boundary: the nested percentage is conditional on one subject, one target, and this workflow. It is not a claim that all simulation uncertainty everywhere is mesh-driven.

## 3. Identical inputs still pass through a variable numerical representation

Walk left to right. The anatomy, labels, montage, currents, conductivities, software, and ROI stay fixed. The experimental factor is whether CHARM creates a fresh tetrahedral mesh or the same mesh workspace is reused.

        The study is not comparing two biological states. It is measuring numerical repeatability of the modelling pipeline.

## 4. Two complementary experiments separate pattern from source

The cohort contrast asks whether remesh-versus-fixed behavior is consistent across ten participants and both a deep and superficial target. The fully nested experiment asks where variation enters by crossing 40 independently generated meshes with 40 repeated solutions per mesh.

        Do not combine the two sample sizes as though 3,200 were independent biological observations. These are technical simulation records.

## 5. The contrast changes one thing: the mesh realization

The remesh arm includes everything that can vary after the fixed inputs, including tetrahedralization. The fixed arm measures only residual solution and post-processing variation conditional on one selected mesh.

        The fixed mesh is the remesh geometry nearest the median spherical-ROI result for that participant and target. This deliberately makes it representative of the remesh distribution; it does not make it the most anatomically accurate mesh.

        Mention the correction only if useful here: the current fixed arm was rebuilt after aligning mesh selection with the optimizer-matched spherical ROI.

## 6. Hippocampal fields vary by 1.81–3.65% under remeshing

Read the plot as within-participant technical variability, not between-person uncertainty. Each pale blue cloud is 40 independently remeshed simulations. The orange fixed-mesh repeats are visually collapsed.

        The corrected spherical-ROI remesh CV range is 1.81–3.65%, with a median of 2.40%. The smallest participant-level SD reduction after mesh reuse is 99.19%.

## 7. M1 shows the same pattern: 1.62–2.79% remesh CV

The right-M1 result is qualitatively the same. Corrected spherical-ROI remesh CVs range from 1.62% to 2.79%, median 2.54%. The fixed-mesh SD reduction is at least 99.78%.

        This replication supports robustness across the two evaluated targets, but not universal generalization to all montages, meshing tools, or ROIs.

## 8. Percent-level spread matters when subjects are closely matched

The reference ordering uses each participant's 40-repeat mean. In each of 20,000 draws, one repeat is selected independently per participant.

        At least one pair reverses in 95.7% of hippocampal draws and 88.5% of M1 draws. Median agreement remains high, so the message is not that all rankings are random. It is that close pairs are sensitive to which valid mesh was selected.

        These are uncertainty measures, not p-values.

## 9. Remeshing changes the numerical head model—not only the final scalar

The heatmap shows small but structured shifts in the proportion of volume assigned to tissue classes across remesh realizations. Total tetrahedral element count also varies substantially within participant, while it is constant when the mesh is reused.

        Do not claim that a particular tissue transition caused the field changes. Element count is coarse, and local boundaries, element quality, electrode–skin geometry, and interpolation can also matter.

## 10. The fully nested case isolates between-mesh from within-mesh variation

Each outer level is a freshly generated mesh. Each mesh is then copied and solved 40 times. The balanced one-way random-effects model separates variance among mesh means from residual variation among repeated solutions conditional on the same geometry.

        The subject was selected once from the ten eligible participants with a fixed seed and persisted. The target was prespecified as left hippocampus.

## 11. The nested case assigns virtually all observed variance to mesh generation

Each point is the CV across 40 repeated solutions on one mesh. Most are exactly or effectively zero. The dashed line is the pooled within-mesh CV.

        The between-mesh CV is 2.018%; pooled within-mesh CV is 0.0040%. The component SD ratio is 501-fold, and 99.9996% of total observed variance is attributed to mesh generation.

        Keep saying 'in this participant' and 'in this workflow.'

## 12. Version sensitivity is systematic in amplitude—not spatial pattern

This is a separate sensitivity analysis on one fixed MNI152 model. The mesh, reference image, ROI definitions, montages, conductivities, currents, and electrode geometry were held constant; only the SimNIBS runtime version changed.

        SimNIBS 4.0.1 produced a lower mean ROI field than 4.5.0 in all 4 ROIs. The relative shift ranged from −1.13% to −2.97%. Whole-brain spatial agreement remained extremely high, with Pearson r at least 0.99983; relative L2 differences ranged from 0.82% to 3.54%.

        The interpretation is a small, systematic amplitude shift with the spatial pattern largely preserved. Do not pool this fixed-model version comparison with the participant or remeshing distributions, and do not attach population inference to the four ROIs.

## 13. Within-version evidence converges on mesh realization

The conclusion does not rest on a single plot. The main cohort shows the condition contrast, the nested experiment attributes the variance source, and structural metrics show that remeshing creates different numerical head models.

        This is evidence that mesh realization is part of the computational specification. It is not evidence that any one realization is anatomically truer.

## 14. The practical response depends on the scientific comparison

Mesh reuse is useful for controlled comparisons because it removes a nuisance source. It is not automatically the best strategy for absolute dosimetry or accuracy claims.

        If the effect of interest is a few percent, either average or model multiple mesh realizations, or demonstrate empirically that mesh uncertainty is negligible relative to that contrast. Preserve and checksum the chosen mesh so the model can be reproduced.

## 15. The result is strong about repeatability—and deliberately narrow about validity

The experiment is designed to quantify numerical repeatability, not validation against measured fields. Ten anatomical datasets and two targets provide replication, but not broad coverage of every montage or software stack.

        The representative fixed mesh was selected using the primary field outcome, so its central value is expected to sit near the remesh center. The meaningful fixed-arm result is the near-zero spread conditional on that mesh—not independent accuracy.

## 16. If the claim is smaller than the mesh noise, the mesh belongs in the claim

Pause on the take-home sentence, then invite discussion. These questions are intentionally methodological: whether to average or standardize, how to set a numerical tolerance, and what experiment would most efficiently test generalization.

## 17. Appendix • full ranking analysis: left hippocampus

Use only if the audience wants to inspect which pairs drive ranking uncertainty.

## 18. Appendix • full ranking analysis: right M1

Use only if the audience wants to inspect which pairs drive ranking uncertainty.

## 19. Appendix • cohort and stimulation parameters

The cohort is balanced by recorded sex and covers five age bands. Both target experiments use the same ten participants.

## 20. Appendix • the current results use the corrected spherical-ROI workflow

The historical fixed mesh was selected using the anatomical parcel median, while the paper endpoint was refreshed to the optimizer-matched spherical ROI. Recomputing selection changed 18 of 20 participant–target representatives. The current presentation uses the regenerated spherical-fixed runs.

        Some spherical ROI voxels were non-finite in a subset of runs, but support remained at least 97% in hippocampus and 99% in M1. Within-group correlations with the field metric were weak and are descriptive only.

## 21. References and evidence sources

These references support the TI, SimNIBS, finite-element, and CamCAN context. The numerical results in the deck come from the corrected local analysis artifacts, not the older manuscript tables.
