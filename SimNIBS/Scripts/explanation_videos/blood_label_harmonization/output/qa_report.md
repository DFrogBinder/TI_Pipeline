# Blood-label harmonization animation QA

Verified: 2026-10-22

## Final MP4

- File: `blood_label_harmonization.mp4`
- SHA-256: `6729bbe960e590aafaadefcd390a88f3b342a30537e44b579b67a432dae182e0`
- Size: 4,557,527 bytes
- Duration: 40.000 seconds
- Video: H.264 High, 1920x1080, 30 fps
- Streams: one video stream; no audio stream

## Shareable GIF

- File: `blood_label_harmonization.gif`
- SHA-256: `78ab8c44b9f36a2ef5fd8db24488a44485667d9e1c9ebd4eced01376f3cb188f`
- Size: 4,482,350 bytes
- Duration: 40.000 seconds
- Video: GIF, 960x540, 12 fps

## Visual inspection

Frames at 5, 10, 16, 24, 31, and 39 seconds were extracted from the final MP4
and inspected as a contact sheet. The final 39-second frame was also inspected
at its native 1920x1080 resolution.

Accepted checks:

- titles, legends, mapping rows, and bottom captions stay inside the frame;
- the final caveat and comparison labels do not overlap;
- the explicit blood colour is distinguishable from scalp and muscle;
- the two rejected global mappings are visibly crossed out;
- the nested envelopes progress from inner to outer compartments;
- local blood segments change to their enclosing WM, GM, CSF, or soft-tissue
  colour;
- the final side-by-side comparison preserves eyes, merges both bone classes,
  absorbs muscle into external soft tissue, and shows no explicit blood class;
- no clipping, corrupted frames, or unintended audio was observed.

Scientific boundary: this is a conceptual schematic of intended harmonization
logic, not a subject-specific anatomical or voxel-wise validation result.
