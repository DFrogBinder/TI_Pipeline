from pathlib import Path

from repeatability_dataset_consolidation.settings import (
    EXCLUDED_SUBJECT,
    SUBJECTS,
    production_config,
    retirement_allowlist,
)


EXPECTED_SUBJECTS = (
    "sub-CC110174",
    "sub-CC121144",
    "sub-CC310407",
    "sub-CC320616",
    "sub-CC420071",
    "sub-CC410432",
    "sub-CC510415",
    "sub-CC520127",
    "sub-CC610631",
    "sub-CC720941",
)


def test_production_scope_is_the_complete_replacement_cohort() -> None:
    config = production_config()

    assert SUBJECTS == EXPECTED_SUBJECTS
    assert config.subjects == EXPECTED_SUBJECTS
    assert EXCLUDED_SUBJECT == "sub-CC520083"
    assert EXCLUDED_SUBJECT not in config.subjects
    assert config.repeat_count == 40
    assert config.expected_cells == 80
    assert config.expected_simulations == 3200


def test_production_roots_keep_shared_archive_and_v4_scaffolds_separate() -> None:
    config = production_config()

    assert config.output_root == Path(
        "/mnt/parscratch/users/cop23bi/final_132_repeatability_combined_dataset_v1"
    )
    assert config.shared_archive_root == Path(
        "/shared/boyan/Shared/retired_repeatability_v3_sources_v1"
    )
    assert config.sources.v4_scaffolds == Path(
        "/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Scaffolds"
    )
    assert config.sources.v4_scaffolds not in retirement_allowlist(config).values()
    assert config.protected_materialization_roots == (
        Path(
            "/mnt/parscratch/users/cop23bi/"
            "ti_dataset_final_132_balanced_10_corrected"
        ),
    )
    assert not set(config.protected_materialization_roots) & set(
        retirement_allowlist(config).values()
    )


def test_archive_and_retirement_scopes_are_exact() -> None:
    config = production_config()

    assert config.archive_sources == {
        "legacy_v3_left": config.sources.legacy_v3_left,
        "legacy_v3_right": config.sources.legacy_v3_right,
        "legacy_v3_headreco": config.sources.legacy_v3_headreco,
        "replacement_v3_scaffolds": config.sources.replacement_v3_scaffolds,
    }
    assert set(retirement_allowlist(config)) == {
        "v4_left_remesh",
        "v4_right_remesh",
        "v4_left_fixed",
        "v4_right_fixed",
        "v3_left",
        "v3_right",
        "v3_headmodels",
        "replacement_subject",
        "paper_analysis",
        "nested_analysis",
        "legacy_v3_left",
        "legacy_v3_right",
        "legacy_v3_headreco",
        "replacement_v3_scaffolds",
    }
