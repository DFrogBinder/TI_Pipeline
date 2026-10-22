from __future__ import annotations

import pytest

from explanation_videos.blood_label_harmonization.storyboard import (
    HARMONIZED_TISSUES,
    TISSUES,
    local_blood_replacement,
    storyboard,
)


def test_harmonized_model_contains_exactly_six_supported_tissues() -> None:
    assert HARMONIZED_TISSUES == (
        "white_matter",
        "grey_matter",
        "csf",
        "generic_bone",
        "external_soft_tissue",
        "eyes",
    )
    assert not {
        "blood",
        "muscle",
        "compact_bone",
        "spongy_bone",
        "internal_air",
    }.intersection(HARMONIZED_TISSUES)


@pytest.mark.parametrize(
    ("enclosing_tissue", "expected"),
    (
        ("white_matter", "white_matter"),
        ("grey_matter", "grey_matter"),
        ("csf", "csf"),
        ("external_soft_tissue", "external_soft_tissue"),
    ),
)
def test_blood_inherits_its_local_enclosing_compartment(
    enclosing_tissue: str,
    expected: str,
) -> None:
    assert local_blood_replacement(enclosing_tissue) == expected


def test_blood_replacement_rejects_unsupported_enclosing_tissue() -> None:
    with pytest.raises(ValueError, match="Unsupported enclosing tissue"):
        local_blood_replacement("blood")


def test_internal_air_is_defined_as_excluded_not_conductive() -> None:
    assert TISSUES["internal_air"].conductive is False
    assert "internal_air" not in HARMONIZED_TISSUES


def test_storyboard_preserves_scientific_sequence_and_duration() -> None:
    beats = storyboard()

    assert tuple(beat.identifier for beat in beats) == (
        "corrected_v4",
        "reject_global_mapping",
        "build_nested_envelopes",
        "local_inheritance",
        "merge_tissues",
        "final_comparison",
    )
    assert 35.0 <= sum(beat.duration_seconds for beat in beats) <= 45.0
    assert beats[-1].caption == (
        "Blood is not assigned globally. Each former blood region inherits "
        "its local enclosing tissue compartment."
    )
