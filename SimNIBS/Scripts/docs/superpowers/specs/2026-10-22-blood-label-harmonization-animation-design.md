# Blood-label harmonization animation design

## Purpose

Create a silent, captioned scientific animation that explains how a corrected
multi-tissue SimNIBS v4 label map is projected into a v3-compatible six-tissue
model. The animation must make the blood-label treatment unambiguous without
implying that every blood voxel is globally reassigned to one tissue.

## Audience and scientific boundary

The primary audience is the research team and supervisors. The animation is a
conceptual two-dimensional schematic of deterministic mask/envelope logic, not
a literal rendering of one participant, a claim about vascular anatomy, or a
voxel-perfect representation of a meshing implementation.

## Scientific message

- The corrected v4 label map contains white matter, grey matter, CSF, scalp,
  eyes, compact bone, spongy bone, blood, muscle, and optionally internal air.
- A v3-compatible model contains white matter, grey matter, CSF, generic bone,
  generic external soft tissue, and eyes.
- Compact and spongy bone become one generic-bone compartment.
- Muscle becomes part of generic external soft tissue.
- Blood is not globally mapped to CSF or globally mapped to scalp.
- Once the explicit blood class is removed, each former blood region is filled
  by its local enclosing compartment as nested cumulative tissue envelopes are
  resolved: WM, GM, CSF, or external soft tissue in this schematic.
- Internal air remains excluded/void.

The final caption must read: "Blood is not assigned globally. Each former
blood region inherits its local enclosing tissue compartment."

## Storyboard

1. Introduce the corrected v4 model as a stylised sagittal head cross-section
   with separate tissue layers and red vessel segments.
2. Show and reject two global shortcuts: `all blood -> CSF` and
   `all blood -> scalp`.
3. Build the nested cumulative envelopes from the inside outward: WM, GM, CSF,
   both bone classes, and the whole external soft-tissue envelope; keep eyes
   separate.
4. Remove the red blood class and recolour each former vessel segment according
   to its local enclosing compartment.
5. Merge compact and spongy bone into generic bone and absorb muscle into
   generic external soft tissue.
6. End with the full corrected-v4 schematic beside the harmonized six-tissue
   schematic and the final caption.

## Visual and delivery requirements

- Silent; all explanation is carried by concise on-screen captions.
- Approximately 35-45 seconds, 16:9, 1920x1080, 30 fps.
- Clean scientific style with a dark neutral background, colour-blind-conscious
  tissue palette, persistent legend when useful, and no dense equations.
- Deliver a high-quality MP4 and a smaller shareable GIF.
- Keep source, tests, environment definition, instructions, and outputs under
  `explanation_videos/`.
- Use a dedicated Conda environment named `explanation-videos` with Manim and
  FFmpeg so future explanation videos can reuse it.

## Verification

- Unit tests pin the scientific mapping, six-tissue result, storyboard order,
  duration boundary, and command construction.
- A low-resolution smoke render proves that Manim can execute the real scene.
- The final MP4 and representative extracted frames are inspected visually.
- FFprobe verifies dimensions, frame rate, duration, video codec, and absence
  of an audio stream.
