# Scientific explanation videos

This directory contains reproducible, silent scientific animations used to
explain analysis and modelling decisions in the SimNIBS projects.

## Reusable environment

Create the dedicated environment once from `SimNIBS/Scripts`:

```bash
conda env create -f explanation_videos/environment.yml
conda activate explanation-videos
```

If the environment already exists and the definition changes:

```bash
conda env update -n explanation-videos \
    -f explanation_videos/environment.yml \
    --prune
```

The environment currently provides Python 3.11, Manim 0.19, FFmpeg 7, and
pytest. Each video lives in its own subdirectory with its own scene, tests,
render wrapper, outputs, and scientific explanation.

## Available videos

- [`blood_label_harmonization`](blood_label_harmonization/README.md): explains
  how an explicit blood class is removed locally when corrected v4 labels are
  projected into a v3-compatible six-tissue model.
