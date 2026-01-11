# Electrode Workshop (SimNIBS)

Minimal SimNIBS-native scripts for placing rectangular electrodes and running
temporal interference (TI) simulations in SimNIBS 4.5.

## Contents
- `place_plank_electrode.py`: place one or two rectangular electrodes (native SimNIBS shapes).
- `ti_custom_planks.py`: run a 4-rectangle TI setup and compute `TI.msh`.
- `functions.py`: legacy post-processing utilities (currently unused in the minimal flow).

## Dependencies
You need a SimNIBS 4.5 environment plus:
- numpy (pulled in by SimNIBS)

## Typical usage

Place a single rectangular electrode:
```python
from place_plank_electrode import run_placement

run_placement({
    "m2m": "/path/to/m2m_subject",
    "out": "./tdcs_custom_out",
    "center": "AF4",
    "length_mm": 50,
    "width_mm": 25,
})
```

Run a 4-rectangle TI setup (AF4/PO4 and AF3/PO3 by default):
```python
from ti_custom_planks import run_ti

run_ti({
    "m2m": "/path/to/m2m_subject",
    "fnamehead": "/path/to/m2m_subject/subject.msh",
    "out": "./TI_CustomPlanks",
    "current_ma": 2.0,
    "element_size_mm": 0.5,
    "length_mm": 50,
    "width_mm": 25,
})
```

## Notes
- Provide `*_xyz` entries (e.g., `right_center_xyz`) if EEG label resolution isn't available.
- Sweep `length_mm` while keeping `width_mm` fixed to study length effects.
- If your SimNIBS build expects `rectangle` instead of `rect`, update `el.shape` in the scripts.
