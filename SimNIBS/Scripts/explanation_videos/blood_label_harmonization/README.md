# Blood-label harmonization animation

This silent 40-second schematic explains how the corrected multi-tissue v4
label map is projected into a v3-compatible six-tissue model.

The central point is that blood is **not** globally reassigned to CSF, scalp,
or any other single class. After removing the explicit blood label, each former
blood region inherits the local compartment established by the nested tissue
envelopes: white matter, grey matter, CSF, or external soft tissue in the
schematic. Compact and spongy bone are fused into generic bone, muscle is
absorbed into external soft tissue, eyes stay separate, and internal air stays
excluded.

This is a conceptual two-dimensional explanation of the intended deterministic
mask/envelope projection. It is not subject-specific anatomy and does not by
itself validate a particular voxel-wise implementation.

## View the finished animation

- [`output/blood_label_harmonization.mp4`](output/blood_label_harmonization.mp4):
  1920x1080, 30 fps, H.264, silent, 40 seconds.
- [`output/blood_label_harmonization.gif`](output/blood_label_harmonization.gif):
  960x540, 12 fps, 40-second shareable preview.
- [`output/qa_report.md`](output/qa_report.md): metadata, checksums, and visual
  inspection record.

## Reproduce the animation

Run these commands from `SimNIBS/Scripts`:

```bash
conda env create -f explanation_videos/environment.yml
conda activate explanation-videos

python -m pytest \
    explanation_videos/blood_label_harmonization/tests \
    -q

# Fast 854x480/15-fps smoke render without a GIF.
python -m explanation_videos.blood_label_harmonization.render \
    --quality low \
    --skip-gif

# Final 1920x1080/30-fps MP4 and 960x540/12-fps GIF.
python -m explanation_videos.blood_label_harmonization.render \
    --quality final
```

The renderer stores transient Manim files under `output/.manim/`, not in the
project root. Override the delivery directory with `--output-dir PATH`. Set
`EXPLANATION_VIDEO_CACHE_DIR` only if `/tmp/explanation-videos-cache` is not an
appropriate writable Fontconfig cache location.

## Source layout

- `storyboard.py`: testable scientific mapping, tissue palette, captions, and
  timing.
- `scene.py`: Manim geometry and animation.
- `render.py`: path-safe MP4/GIF rendering wrapper.
- `tests/`: scientific-contract, rendering-command, and visual-layout tests.
- `output/`: final delivery artifacts and QA record.
