from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from simnibs_subject_replacement.settings import (
    ReplacementConfig,
    assert_replacement_root_isolation,
    replacement_paths,
    replacement_scope,
    validate_candidate,
)


def _config(layout: dict[str, Path | str]) -> ReplacementConfig:
    return ReplacementConfig(
        subject=str(layout["subject"]),
        replaces=str(layout["replaces"]),
        final132_subjects_file=Path(layout["subjects_file"]),
        corrected_v4_scaffold_root=Path(layout["scaffold"]),
        legacy_v3_root=Path(layout["legacy"]),
        atlas_root=Path(layout["atlas"]),
        targets_csv=Path(layout["targets"]),
        work_root=Path(layout["work"]),
    )


def test_valid_candidate_and_subject_derived_paths(
    replacement_layout: dict[str, Path | str],
) -> None:
    config = _config(replacement_layout)

    cohort = validate_candidate(config)
    paths = replacement_paths(config)
    scope = replacement_scope(config)

    assert cohort == ("sub-CC110056", "sub-CC110098")
    assert paths.root == Path(replacement_layout["work"]) / config.subject
    assert config.subject in str(paths.v3_headmodel_root)
    assert config.subject in str(paths.v3_left_root)
    assert config.subject in str(paths.v3_right_root)
    assert config.subject in str(paths.v4_left_root)
    assert config.subject in str(paths.v4_right_root)
    assert len(
        {
            paths.v3_headmodel_root,
            paths.v3_left_root,
            paths.v3_right_root,
            paths.v4_corrected_root,
            paths.v4_left_root,
            paths.v4_right_root,
        }
    ) == 6
    assert scope["per_version"]["simulation_tasks"] == 160
    assert scope["per_version"]["component_fem_solves"] == 320
    assert scope["both_versions"]["simulation_tasks"] == 320
    assert scope["both_versions"]["component_fem_solves"] == 640


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("subject", "../sub-CC110056", "canonical"),
        ("subject", "sub-CC999999", "exactly once"),
        ("subject", "sub-CC110174", "retained base-nine"),
        ("subject", "sub-CC520083", "differ from"),
        ("replaces", "not-a-subject", "canonical"),
    ],
)
def test_candidate_validation_fails_closed(
    replacement_layout: dict[str, Path | str],
    field: str,
    value: str,
    match: str,
) -> None:
    config = replace(_config(replacement_layout), **{field: value})

    with pytest.raises(ValueError, match=match):
        validate_candidate(config)


def test_duplicate_authoritative_membership_is_rejected(
    replacement_layout: dict[str, Path | str],
) -> None:
    config = _config(replacement_layout)
    config.final132_subjects_file.write_text(
        f"{config.subject}\n{config.subject}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_candidate(config)


@pytest.mark.parametrize(
    "field",
    [
        "v3_headmodel_root",
        "v3_left_root",
        "v3_right_root",
        "v4_corrected_root",
        "v4_left_root",
        "v4_right_root",
    ],
)
def test_writable_roots_must_not_overlap_sources_or_each_other(
    replacement_layout: dict[str, Path | str],
    field: str,
) -> None:
    config = _config(replacement_layout)
    paths = replacement_paths(config)
    unsafe = replace(paths, **{field: config.corrected_v4_scaffold_root})

    with pytest.raises(ValueError, match="overlap"):
        assert_replacement_root_isolation(config, unsafe)
