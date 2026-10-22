# Blood-label Harmonization Animation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and render a reusable silent Manim animation explaining local blood-label reassignment during v4-to-v3 tissue harmonization.

**Architecture:** A pure-Python storyboard module owns the scientific contract and timeline without importing Manim. A Manim scene consumes that contract to draw the schematic, while a small rendering CLI provides reproducible MP4/GIF commands and validates final media.

**Tech Stack:** Python 3.11, Manim 0.19, FFmpeg/FFprobe, pytest, Conda.

**Spec:** `docs/superpowers/specs/2026-10-22-blood-label-harmonization-animation-design.md`

## Global Constraints

- The final animation is silent, captioned, 16:9, 1920x1080, 30 fps, and approximately 35-45 seconds.
- Blood must be shown as locally inherited by the enclosing WM, GM, CSF, or external soft-tissue compartment, never globally reassigned to one tissue.
- The harmonized result has exactly six conductive tissues: WM, GM, CSF, generic bone, external soft tissue, and eyes; internal air stays excluded.
- Source, tests, environment definition, instructions, and outputs live under `explanation_videos/`.
- Existing uncommitted files in the original checkout must remain untouched.

## Review Focus

- A future edit must not turn the position-dependent blood mapping into one global target; Task 1 pins all four local cases.
- The six-tissue output must not accidentally retain blood, muscle, either separate bone class, or internal air; Task 1 tests exact membership.
- A timing edit must keep total duration between 35 and 45 seconds and preserve storyboard order; Task 1 tests both.
- Paths containing spaces must remain safe in render command construction; Task 2 tests argument lists rather than shell strings.
- Final media must remain silent, 1920x1080, 30 fps, and playable; Task 3 verifies FFprobe metadata and representative frames.

---

### Task 1: Scientific storyboard contract

**Files:**
- Create: `explanation_videos/environment.yml`
- Create: `explanation_videos/blood_label_harmonization/__init__.py`
- Create: `explanation_videos/blood_label_harmonization/storyboard.py`
- Create: `explanation_videos/blood_label_harmonization/tests/test_storyboard.py`

**Interfaces:**
- Produces: `Tissue`, `Beat`, `TISSUES`, `HARMONIZED_TISSUES`, `local_blood_replacement(str) -> str`, and `storyboard() -> tuple[Beat, ...]`.
- Consumes: no earlier task interface.

- [ ] **Step 1: Write failing scientific-contract tests**

Test exact six-tissue membership, all four local blood replacements, rejection of unsupported enclosing labels, absence of air from conductive tissues, ordered beat identifiers, and total duration within 35-45 seconds.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest explanation_videos/blood_label_harmonization/tests/test_storyboard.py -q`

Expected: FAIL because `storyboard.py` does not exist.

- [ ] **Step 3: Implement the pure storyboard contract and environment definition**

Use immutable dataclasses and literal, colour-blind-conscious hex colours. Keep Manim out of this module.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest explanation_videos/blood_label_harmonization/tests/test_storyboard.py -q`

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit**

Commit: `feat: define blood-label animation storyboard`

### Task 2: Manim scene and reproducible renderer

**Files:**
- Create: `explanation_videos/blood_label_harmonization/scene.py`
- Create: `explanation_videos/blood_label_harmonization/render.py`
- Create: `explanation_videos/blood_label_harmonization/tests/test_render.py`

**Interfaces:**
- Consumes: Task 1 `TISSUES`, `HARMONIZED_TISSUES`, `local_blood_replacement`, and `storyboard`.
- Produces: Manim scene class `BloodLabelHarmonization`; `manim_command(...) -> list[str]`; `gif_command(...) -> list[str]`; CLI `main(...) -> int`.

- [ ] **Step 1: Write failing renderer-contract tests**

Test exact scene name, path-safe argument lists, 1920x1080/30-fps final settings, low-quality smoke settings, GIF conversion settings, and rejection of a missing MP4.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest explanation_videos/blood_label_harmonization/tests/test_render.py -q`

Expected: FAIL because `render.py` does not exist.

- [ ] **Step 3: Implement the renderer and Manim scene**

Draw a stylised nested sagittal cross-section, separate red vessel segments by local compartment, animate rejected global mappings, build nested envelopes, recolour former blood regions, merge tissue classes, and finish with the approved side-by-side caption.

- [ ] **Step 4: Run unit tests and verify GREEN**

Run: `python -m pytest explanation_videos/blood_label_harmonization/tests -q`

Expected: all Task 1 and Task 2 unit tests pass.

- [ ] **Step 5: Run a real low-resolution Manim smoke render**

Run the renderer with `--quality low --skip-gif` and verify the MP4 exists and is non-empty.

- [ ] **Step 6: Commit**

Commit: `feat: animate local blood-label harmonization`

### Task 3: Final render, documentation, and media QA

**Files:**
- Create: `explanation_videos/README.md`
- Create: `explanation_videos/blood_label_harmonization/README.md`
- Create: `explanation_videos/blood_label_harmonization/output/.gitignore`
- Create: rendered MP4 and GIF under `explanation_videos/blood_label_harmonization/output/`

**Interfaces:**
- Consumes: Task 2 renderer CLI and `BloodLabelHarmonization` scene.
- Produces: reusable setup/render instructions, final MP4/GIF, and recorded QA evidence.

- [ ] **Step 1: Render the final MP4 and GIF in `explanation-videos`**

Run the renderer at final quality. Expected: both media files are non-empty.

- [ ] **Step 2: Verify media metadata**

Use FFprobe to confirm 1920x1080, 30 fps, 35-45 seconds, H.264 video, and no audio stream.

- [ ] **Step 3: Extract and visually inspect representative frames**

Inspect frames from the initial model, rejected mappings, local reassignment, and final comparison. Fix clipping, illegible text, misleading geometry, or colour problems before acceptance.

- [ ] **Step 4: Document setup, rendering, outputs, and scientific caveat**

Give copy-paste commands for environment creation/activation, unit tests, smoke render, and final render.

- [ ] **Step 5: Run the package suite**

Run: `python -m pytest explanation_videos/blood_label_harmonization/tests -q`

Expected: all tests pass without warnings.

- [ ] **Step 6: Commit**

Commit: `docs: add reusable explanation-video workflow`
