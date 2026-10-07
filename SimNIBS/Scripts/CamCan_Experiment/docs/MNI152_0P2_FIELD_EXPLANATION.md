# Why the re-simulated MNI152 mean target field is not exactly 0.20 V/m

## Summary

The MNI152 mean target fields shown in the current figures should not be
expected to equal exactly 0.20 V/m.

There are two distinct reasons:

1. **The optimizer did not enforce an equality of exactly 0.20 V/m.** It
   selected the first available discrete Pareto solution that met or exceeded
   the 0.20 V/m selection threshold.
2. **The figures show an independent SimNIBS re-evaluation**, rather than the
   target-field value stored by the optimizer. This re-evaluation uses a
   different field calculation and a voxel-grid reconstruction of the
   tetrahedral optimizer ROI.

The resulting MNI152 values are therefore validation measurements of the
transferred montage, not copies of the optimizer's recorded target field.

## Evidence from the original optimizer results

Inspection of the original MNI152 Pareto `.mat` files shows that
`ParetoBest.TI_free.Emin` selected the first discrete Pareto point above
0.20 V/m. The preceding available point was approximately 0.15 V/m in every
ROI:

| ROI | Previous Pareto point (V/m) | Selected optimizer TargetField (V/m) |
| --- | ---: | ---: |
| Left hippocampus | 0.150624 | 0.201260 |
| Left M1 | 0.151390 | 0.207106 |
| Right DLPFC | 0.150538 | 0.204181 |
| Right thalamus | 0.151880 | 0.200135 |

Thus, 0.20 V/m acted as a **selection threshold**, not an exactly constrained
solution. The exhaustive search evaluated discrete electrode configurations
and current combinations, so the minimum feasible solution above the threshold
was generally slightly greater than 0.20 V/m.

These optimizer-reported values are also recorded in `utils/targets.csv`.

## Comparison with the re-simulated MNI152 measurements

The current manuscript figures use means recalculated from the independently
generated MNI152 SimNIBS TI fields:

| ROI | Optimizer TargetField (V/m) | Re-simulated MNI mean (V/m) | Difference from optimizer |
| --- | ---: | ---: | ---: |
| Left hippocampus | 0.201260 | 0.210052 | +4.37% |
| Left M1 | 0.207106 | 0.188939 | -8.77% |
| Right DLPFC | 0.204181 | 0.176744 | -13.44% |
| Right thalamus | 0.200135 | 0.202151 | +1.01% |

These are not plotting errors. The analysis loads the re-simulated MNI152 TI
volume, reconstructs the target ROI, and calculates the mean from the resulting
voxel values.

## Why the independent re-evaluation differs

### Different target representations

The optimization and manuscript analysis implement the same conceptual ROI,
but not the same numerical sampling:

- The original `MakeROIs.m` method constructs the ROI on a tetrahedral volume
  mesh. It selects tetrahedral elements according to their centre coordinates
  and calculates the selected volume from the individual element volumes.
- The current post-processing constructs a voxel-grid equivalent after
  resampling the anatomical atlas to the TI NIfTI grid. It selects voxel
  centres and calculates the mean over the finite selected voxels.
- Exporting a mesh field to a NIfTI volume introduces an additional
  interpolation step.

The voxel-grid ROI is an appropriate reproducible approximation for applying
one analysis consistently across the CamCan cohort, but it is not numerically
identical to the optimizer's tetrahedral ROI.

### Independent forward-field calculation

The MNI152 fields were regenerated using SimNIBS 4.5.0 and
`TI.get_maxTI`, rather than copied from the optimizer's field library. Any
difference in solver implementation, mesh, electrode representation, field
interpolation, or numerical precision can change the reconstructed mean.

The MNI152 provenance confirms that the intended electrode pairs, currents,
and `targets.csv` version were used. Therefore, the available evidence does
not indicate that the wrong montage was simulated.

The SimNIBS logs also report current-calibration errors ranging from 1.2% to
10.25%, depending on the montage. These may contribute to numerical
differences, although they cannot explain the complete pattern by themselves:
right DLPFC has the largest target-field discrepancy but the smallest reported
calibration error.

### SimNIBS version difference

The current MNI152 baselines were generated with SimNIBS 4.5.0, whereas the
CamCan cohort simulations were generated on the HPC using SimNIBS 4.0.1.
This version difference is another potential source of numerical variation and
should ideally be removed before the final manuscript analysis.

## Important distinction for interpreting the figures

The following quantities should not be conflated:

- **0.20 V/m optimizer selection threshold**
- **optimizer-reported MNI152 TargetField**
- **re-simulated MNI152 mean target field**
- **percentage of target voxels at or above 0.20 V/m**

Even if the mean field were exactly 0.20 V/m, this would not imply 100% target
coverage at 0.20 V/m. A spatially heterogeneous target can have a mean of
0.20 V/m while many target voxels remain below the threshold.

## Recommended manuscript treatment

1. Retain the actual re-simulated MNI152 values in population-comparison
   figures. They were calculated using the same post-processing definitions as
   the subject results and should not be forced or rescaled to 0.20 V/m.
2. Label 0.20 V/m as the **optimizer selection threshold** or
   **evaluation threshold**, rather than the expected MNI152 mean.
3. Label the plotted MNI152 value explicitly as the
   **re-simulated MNI152 mean**.
4. Include a supplementary validation table comparing the optimizer-reported
   TargetField with the independently re-simulated value for each ROI.
5. Regenerate the four MNI152 baselines using SimNIBS 4.0.1, matching the
   software version used for the CamCan cohort. This will remove the
   cross-version confound but will not guarantee exact agreement with the
   optimizer because the field and ROI representations will still differ.
6. If exact optimizer-domain replication is required, the selected montages
   must be evaluated using the original optimizer forward fields, original
   tetrahedral mesh, and original ROI element membership. Alternatively, the
   optimization would need to be repeated using the same SimNIBS fields and
   voxel-grid ROI used by the manuscript analysis.

## Conclusion

The premise that every MNI152 mean should equal exactly 0.20 V/m is not
supported by the optimizer outputs. The optimizer selected discrete solutions
slightly above a 0.20 V/m threshold, and the manuscript pipeline then
independently re-simulated and re-sampled those montages. The observed
differences are therefore method-comparison discrepancies that should be
quantified and reported, rather than corrected by forcing the MNI152 values to
0.20 V/m.
