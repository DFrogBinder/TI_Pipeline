from __future__ import annotations

import numpy as np
import pytest

from simnibs_326_corrected_label_repeatability.labels import (
    LabelContractError,
    harmonize_labels,
    validate_source_labels,
)


def test_mapping_preserves_and_combines_only_declared_nonblood_labels() -> None:
    source = np.array([0, 1, 2, 3, 5, 6, 7, 8, 10, 11], dtype=np.float32).reshape(
        2, 5, 1
    )

    output, stats = harmonize_labels(source, (1.0, 1.0, 1.0))

    assert output.dtype == np.uint8
    assert output.ravel().tolist() == [0, 1, 2, 3, 5, 6, 4, 4, 5, 11]
    assert stats.observed_source_labels == (0, 1, 2, 3, 5, 6, 7, 8, 10, 11)
    assert stats.source_counts == {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        5: 1,
        6: 1,
        7: 1,
        8: 1,
        10: 1,
        11: 1,
    }
    assert stats.output_counts == {0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 1, 11: 1}
    assert stats.blood_voxels == 0
    assert stats.blood_recipient_counts == {1: 0, 2: 0, 3: 0, 5: 0}
    assert stats.blood_tie_voxels == 0


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.nan, "finite"),
        (2.5, "integer"),
        (4.0, "unsupported.*4"),
        (12.0, "unsupported.*12"),
    ],
)
def test_rejects_invalid_source_values(value: float, message: str) -> None:
    source = np.ones((2, 2, 2), dtype=float)
    source[0, 0, 0] = value

    with pytest.raises(LabelContractError, match=message):
        validate_source_labels(source)


@pytest.mark.parametrize(
    "source",
    [np.ones((2, 2), dtype=int), np.ones((1, 1, 1, 1), dtype=int)],
)
def test_rejects_non_three_dimensional_data(source: np.ndarray) -> None:
    with pytest.raises(LabelContractError, match="three-dimensional"):
        validate_source_labels(source)


@pytest.mark.parametrize(
    "spacing",
    [(0.0, 1.0, 1.0), (-1.0, 1.0, 1.0), (1.0, np.nan, 1.0), (1.0, 1.0)],
)
def test_rejects_invalid_spacing(spacing: tuple[float, ...]) -> None:
    with pytest.raises(LabelContractError, match="spacing"):
        harmonize_labels(np.ones((2, 2, 2), dtype=int), spacing)


def test_blood_uses_physical_distance_with_anisotropic_spacing() -> None:
    source = np.zeros((3, 3, 1), dtype=int)
    source[1, 1, 0] = 9
    source[0, 1, 0] = 1
    source[1, 0, 0] = 2

    output, stats = harmonize_labels(source, (3.0, 1.0, 1.0))

    assert output[1, 1, 0] == 2
    assert stats.spacing_mm == (3.0, 1.0, 1.0)
    assert stats.blood_recipient_counts == {1: 0, 2: 1, 3: 0, 5: 0}
    assert stats.blood_tie_voxels == 0


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [(1, 2, 1), (2, 3, 2), (3, 5, 3)],
)
def test_blood_exact_ties_follow_declared_priority(
    first: int,
    second: int,
    expected: int,
) -> None:
    source = np.zeros((3, 1, 1), dtype=int)
    source[:, 0, 0] = [first, 9, second]

    output, stats = harmonize_labels(source, (1.0, 1.0, 1.0))

    assert output[1, 0, 0] == expected
    assert stats.blood_tie_voxels == 1
    assert stats.blood_recipient_counts[expected] == 1


def test_separate_blood_voxels_can_map_to_different_recipients() -> None:
    source = np.zeros((5, 1, 1), dtype=int)
    source[:, 0, 0] = [1, 9, 0, 9, 2]

    output, stats = harmonize_labels(source, (1.0, 1.0, 1.0))

    assert output[:, 0, 0].tolist() == [1, 1, 0, 2, 2]
    assert stats.blood_recipient_counts == {1: 1, 2: 1, 3: 0, 5: 0}


def test_bone_eyes_air_and_background_are_never_blood_recipients() -> None:
    source = np.zeros((7, 1, 1), dtype=int)
    source[:, 0, 0] = [5, 0, 4, 9, 6, 11, 0]
    source[2, 0, 0] = 7

    output, stats = harmonize_labels(source, (1.0, 1.0, 1.0))

    assert output[3, 0, 0] == 5
    assert stats.blood_recipient_counts == {1: 0, 2: 0, 3: 0, 5: 1}


def test_blood_without_any_eligible_recipient_fails() -> None:
    source = np.array([[[9]], [[7]], [[6]], [[11]]], dtype=int)

    with pytest.raises(LabelContractError, match="no eligible recipient"):
        harmonize_labels(source, (1.0, 1.0, 1.0))
