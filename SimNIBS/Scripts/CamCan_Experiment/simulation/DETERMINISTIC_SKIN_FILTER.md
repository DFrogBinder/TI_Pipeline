# Deterministic Skin Filter

## Why this exists
ROAST-based custom segmentations can produce noisy or jagged scalp labels.  
That noise can propagate into CHARM remeshing and result in rough outer head surfaces.

This filter applies a deterministic morphology cleanup to the skin label before remeshing.

## Where it runs in the pipeline
File: `simulation/TI_runner_multi-core.py`

Order of operations in meshing mode:
1. CHARM initial run creates baseline files.
2. Custom segmentation is loaded and converted to integer labels.
3. `smooth_skin_segmentation(...)` is applied (if enabled).
4. Filtered segmentation is used for replacement/merge behavior.
5. CHARM remeshing (`charm <subject> --mesh`) runs on the updated segmentation.

So the filter affects the geometry used for final mesh generation.

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
Top-of-file flags in `simulation/TI_runner_multi-core.py`:

```python
applyDeterministicSkinFilter = True
skinFilterLabelId = 5
skinFilterBackgroundId = 0
skinFilterClosingVoxels = 2
skinFilterOpeningVoxels = 1
skinFilterKeepLargestComponent = True
saveSkinFilterPreview = True
```

If `saveSkinFilterPreview = True`, the filtered segmentation is written as:
`<subject>_T1w_ras_1mm_T1andT2_masks_skin_smoothed.nii`

## Debug/log outputs
The function returns a debug dictionary and the runner logs it via `skin_filter_applied`.

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
