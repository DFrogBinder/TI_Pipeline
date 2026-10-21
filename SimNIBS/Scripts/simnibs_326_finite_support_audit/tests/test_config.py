from __future__ import annotations

import pytest

from simnibs_326_finite_support_audit.config import (
    SUBJECTS,
    cell_from_index,
    production_task_id,
)


def test_cell_and_production_task_mapping_are_subject_major():
    target0, subject_index0, subject0 = cell_from_index(0)
    target10, subject_index10, subject10 = cell_from_index(10)
    assert (target0.key, subject_index0, subject0) == (
        "left-hippocampus",
        0,
        SUBJECTS[0],
    )
    assert (target10.key, subject_index10, subject10) == ("right-m1", 0, SUBJECTS[0])
    assert production_task_id(3, 1) == 120
    assert production_task_id(3, 40) == 159


def test_cell_mapping_rejects_partial_or_expanded_scope():
    with pytest.raises(IndexError):
        cell_from_index(20)
    with pytest.raises(ValueError):
        production_task_id(0, 41)
