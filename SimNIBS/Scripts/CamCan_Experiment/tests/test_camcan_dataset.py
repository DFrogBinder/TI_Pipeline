from pathlib import Path

import pytest

from utils.camcan_dataset import (
    CAMCAN_ROI_CONFIGS,
    electrode_names_for_config,
    parse_dataset_name,
    validate_individualized_target_table,
    validate_dataset_montage,
)
from post.camcan_electrodes import resolve_camcan_post_electrodes


TARGETS_CSV = Path(__file__).resolve().parents[2] / "utils" / "targets.csv"


def test_camcan_dataset_mapping_uses_confirmed_optimized_electrodes():
    expected = {
        "Left_M1": ("F1", "F2", "C3", "CP3"),
        "Left_Hippocampus": ("F8", "P8", "T7", "P7"),
        "Right_DLPC": ("AF4", "F4", "FC2", "C2"),
        "Right_Thalamus": ("F5", "TP7", "FT8", "P8"),
    }
    assert {
        config.dataset_prefix: electrode_names_for_config(config, TARGETS_CSV)
        for config in CAMCAN_ROI_CONFIGS
    } == expected


def test_dataset_parser_and_montage_validation_fail_closed():
    config, repeat = parse_dataset_name("Right_DLPC_Data_10")
    assert config.montage_preset == "right-dlpfc"
    assert repeat == "10"
    assert validate_dataset_montage(["Right_DLPC_Data_01"], "right-dlpc") == config

    with pytest.raises(ValueError, match="does not match"):
        validate_dataset_montage(["Right_DLPC_Data_01"], "left-m1")
    with pytest.raises(ValueError, match="mixes multiple ROI"):
        validate_dataset_montage(
            ["Right_DLPC_Data_01", "Left_M1_Data_01"],
            "right-dlpfc",
        )


def test_post_electrodes_are_derived_from_batch_roi_and_targets_csv(tmp_path):
    (tmp_path / "Right_Thalamus_Data_01").mkdir()
    resolved = resolve_camcan_post_electrodes(tmp_path, TARGETS_CSV)

    assert resolved.names == ("F5", "TP7", "FT8", "P8")
    assert "eeg_positions/EEG10-10_UI_Jurak_2007.csv" in (
        resolved.eeg_positions_path_template
    )


def test_individualized_best_worst_table_is_complete_and_fail_closed():
    cohort_dir = (
        Path(__file__).resolve().parents[1]
        / "cohort_pipeline"
        / "cohorts"
        / "optimized_best_worst_7"
    )
    subjects = (cohort_dir / "subjects.txt").read_text().splitlines()
    rows = validate_individualized_target_table(
        cohort_dir / "individualized_targets.csv",
        subjects=subjects,
    )

    assert len(rows) == 28
    assert rows[("sub-CC120795", "Right_DLPC")]["pair1"] == "F2-F4"
    assert rows[("sub-CC120795", "Right_DLPC")]["pair2"] == "FT8-T8"
    assert rows[("sub-CC420100", "Right_Thalamus")][
        "pareto_selection"
    ] == "TI_free.Emin"
