import csv
from pathlib import Path

from target_montages import (
    DEFAULT_ELECTRODE_CONDUCTIVITY,
    DEFAULT_ELECTRODE_RADIUS_MM,
    DEFAULT_ELECTRODE_SHAPE,
    DEFAULT_ELECTRODE_THICKNESS_MM,
    MONTAGE_PRESETS,
    preset_key_for_roi,
)


TARGETS_CSV = Path(__file__).resolve().parents[2] / "utils" / "targets.csv"
RUNNER_FILES = [
    "TI_runner_MNI152.py",
    "TI_runner_multi-core.py",
    "TI_runner_multi-core_skin-filter.py",
    "TI_runner_single-core.py",
]


def test_montage_presets_match_targets_csv_stimulation_parameters():
    with TARGETS_CSV.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    assert set(MONTAGE_PRESETS) == {preset_key_for_roi(row["roi"]) for row in rows}

    for row in rows:
        preset = MONTAGE_PRESETS[preset_key_for_roi(row["roi"])]
        pair1_anode, pair1_cathode = row["pair1"].split("-")
        pair2_anode, pair2_cathode = row["pair2"].split("-")

        assert (preset.pair1.anode, preset.pair1.cathode) == (
            pair1_anode,
            pair1_cathode,
        )
        assert (preset.pair2.anode, preset.pair2.cathode) == (
            pair2_anode,
            pair2_cathode,
        )
        assert preset.pair1.current_a == float(row["current1"]) * 1e-3
        assert preset.pair2.current_a == float(row["current2"]) * 1e-3


def test_montage_presets_use_shared_non_stimulation_parameters():
    for preset in MONTAGE_PRESETS.values():
        assert preset.electrode_radius_mm == DEFAULT_ELECTRODE_RADIUS_MM
        assert preset.electrode_thickness_mm == DEFAULT_ELECTRODE_THICKNESS_MM
        assert preset.electrode_shape == DEFAULT_ELECTRODE_SHAPE
        assert preset.electrode_conductivity == DEFAULT_ELECTRODE_CONDUCTIVITY


def test_all_ti_runners_use_shared_csv_montage_source():
    root = Path(__file__).resolve().parent
    required_tokens = [
        "from target_montages import",
        "resolve_montage_preset",
        "montage.electrode_radius_mm",
        "montage.electrode_thickness_mm",
        "montage.electrode_shape",
        "montage.electrode_conductivity",
    ]

    for runner in RUNNER_FILES:
        source = (root / runner).read_text()
        for token in required_tokens:
            assert token in source, f"{runner} does not use shared montage token: {token}"
