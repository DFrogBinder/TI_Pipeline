import pytest

from simulation.montage_presets import (
    infer_montage_from_directory,
    resolve_montage_preset,
)


def test_infer_montage_from_repeat_directory_name():
    montage = infer_montage_from_directory("/tmp/Left_Hippocampus_Data_01")

    assert montage.name == "left-hippocampus"
    assert montage.roi_name == "Left-Hippocampus"
    assert montage.pair1.as_signed_tuple() == ("F10", 2e-3, "P8", -2e-3)


def test_infer_montage_supports_m1_aliases():
    left = infer_montage_from_directory("/tmp/Left_M1_Data_01")
    right = infer_montage_from_directory("/tmp/Right_M1_Data_01")

    assert left.name == "left-m1"
    assert left.roi_name == "ctx-lh-precentral"
    assert right.name == "right-m1"
    assert right.roi_name == "ctx-rh-precentral"


def test_resolve_montage_preset_accepts_roi_alias_and_preset_name():
    assert resolve_montage_preset("right_dlpfc").name == "right-dlpfc"
    assert resolve_montage_preset("right-thalamus").roi_name == "Right-Thalamus-Proper"


def test_resolve_montage_preset_rejects_roi_without_simulation_preset():
    with pytest.raises(ValueError, match="No simulation montage preset"):
        resolve_montage_preset("left_amygdala")
