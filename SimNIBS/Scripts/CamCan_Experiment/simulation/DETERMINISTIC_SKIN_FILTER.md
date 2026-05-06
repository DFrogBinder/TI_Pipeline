# Deterministic Skin Filter

## Why this exists
ROAST-based custom segmentations can produce noisy or jagged scalp labels.  
That noise can propagate into CHARM remeshing and result in rough outer head surfaces.

This filter applies a deterministic morphology cleanup to the skin label before remeshing.

## Where it runs in the pipeline
File: `simulation/TI_runner_multi-core_skin-filter.py`

Meshing mode now has two exclusive workflows controlled by `mergeSegmentationMaps`.

Replacement workflow (`mergeSegmentationMaps = False`):
1. CHARM initial run creates baseline files.
2. Custom segmentation is loaded and converted to integer labels.
3. `smooth_skin_segmentation(...)` is applied only if `applyDeterministicSkinFilter = True`.
4. The resulting custom segmentation replaces CHARM's `tissue_labeling_upsampled.nii.gz`.
5. CHARM remeshing (`charm <subject> --mesh`) runs on the replaced segmentation.

Merge workflow (`mergeSegmentationMaps = True`):
1. CHARM initial run creates baseline files.
2. Custom and CHARM segmentations are loaded.
3. The custom segmentation is converted to integer labels.
4. CHARM's segmentation is resampled to the custom grid if needed.
5. `merge_segmentation_maps(...)` creates the merged segmentation.
6. The merged segmentation replaces CHARM's `tissue_labeling_upsampled.nii.gz`.
7. CHARM remeshing (`charm <subject> --mesh`) runs on the merged segmentation.

So the deterministic skin filter affects final mesh geometry only in replacement mode. In merge mode it is ignored.

## Implementation location
File: `utils/skin_filter.py`  
Function: `smooth_skin_segmentation(...)`

## Plain-language behavior
The filter only modifies skin/background voxels. It does not relabel inner tissues.

At a high level:
1. Build a binary mask of skin voxels (default label `5`).
2. Run binary closing:
   - fills small holes and narrow gaps in skin.
3. Run binary opening:
   - removes small spikes/islands after closing.
4. Optionally keep only the largest connected skin component:
   - removes detached fragments.
5. Apply the cleaned mask back to segmentation:
   - voxels removed from skin become background (default `0`)
   - new skin voxels can only be added where current label is skin or background

This is deterministic: same input + same parameters -> same output.

## Function signature
```python
smooth_skin_segmentation(
    segmentation,
    *,
    skin_label=5,
    background_label=0,
    closing_voxels=2,
    opening_voxels=1,
    keep_largest_component=True,
    output_path=None,
) -> (nib.Nifti1Image, dict)
```

## Parameters and effects
| Parameter | Meaning | Typical impact |
|---|---|---|
| `skin_label` | Label ID treated as skin | Must match your segmentation convention |
| `background_label` | Label used when removing skin voxels | Usually `0` |
| `closing_voxels` | Number of closing iterations | Higher -> smoother and thicker/filled skin |
| `opening_voxels` | Number of opening iterations | Higher -> removes more small protrusions |
| `keep_largest_component` | Keep only largest skin component | Removes disconnected skin islands |
| `output_path` | Optional NIfTI save path | Useful for QA inspection |

## Current runner controls
Top-of-file flags in `simulation/TI_runner_multi-core_skin-filter.py`:

```python
mergeSegmentationMaps = False
applyDeterministicSkinFilter = True
skinFilterLabelId = 5
skinFilterBackgroundId = 0
skinFilterClosingVoxels = 2
skinFilterOpeningVoxels = 1
skinFilterKeepLargestComponent = True
saveSkinFilterPreview = True
```

Behavior:
- If `mergeSegmentationMaps = True`, `applyDeterministicSkinFilter` is ignored and the merge path is used.
- If `mergeSegmentationMaps = False` and `applyDeterministicSkinFilter = True`, the filtered segmentation is written as `<subject>_T1w_ras_1mm_T1andT2_masks_skin_smoothed.nii` when `saveSkinFilterPreview = True`.
- If `mergeSegmentationMaps = False`, `applyDeterministicSkinFilter = True`, and `saveSkinFilterPreview = False`, the smoothed segmentation is written as `<subject>_T1w_ras_1mm_T1andT2_masks_replaced.nii` because that file is used for the CHARM replacement step.
- If `mergeSegmentationMaps = False` and `applyDeterministicSkinFilter = False`, the unsmoothed custom segmentation is written as `<subject>_T1w_ras_1mm_T1andT2_masks_replaced.nii` before replacing CHARM's segmentation.

## Debug/log outputs
The function returns a debug dictionary and the runner logs it via `skin_filter_applied` when replacement mode uses smoothing.

Related runner events now include:
- `segmentation_strategy`
- `segmentation_merge_applied`
- `skin_filter_applied`
- `skin_filter_skipped`
- `skin_filter_ignored`

Returned metrics include:
- `initial_skin_voxels`
- `final_skin_voxels`
- `voxels_added`
- `voxels_removed`
- `connected_components_before_keep`
- parameter values used

These metrics help verify if filtering is too weak or too aggressive.

## Tuning recommendations
Start conservative and increase gradually.

Suggested progression:
1. `closing=1`, `opening=0` for minimal cleanup.
2. `closing=2`, `opening=1` for moderate smoothing (current default).
3. Increase only one step at a time and inspect preview output + mesh quality.

Watch for over-smoothing signs:
- scalp unnaturally thick or inflated
- removal of valid anatomical detail
- strong jumps in `voxels_added`/`voxels_removed`

## Dependency notes
`smooth_skin_segmentation` uses `scipy.ndimage` morphology and connected components.
Ensure SciPy is available in the environment used for simulation jobs.

## Safety properties
- deterministic (no randomness)
- limited scope (edits only skin/background labels)
- preserves image affine/header in output
- fails fast for invalid parameters (negative iteration counts)
