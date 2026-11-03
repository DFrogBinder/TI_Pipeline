from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from repeatability_combined_analysis.settings import (
    CONDITIONS,
    EXCLUDED_SUBJECT,
    METRICS,
    PRIMARY_METRIC,
    REPLACEMENT_SUBJECT,
    SUBJECTS,
    TARGETS,
    VERSIONS,
    production_config,
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


def test_production_scope_is_the_exact_9_plus_1_matrix() -> None:
    config = production_config()

    assert SUBJECTS == EXPECTED_SUBJECTS
    assert config.subjects == EXPECTED_SUBJECTS
    assert REPLACEMENT_SUBJECT == "sub-CC510415"
    assert EXCLUDED_SUBJECT == "sub-CC520083"
    assert VERSIONS == ("simnibs326", "simnibs4")
    assert TARGETS == ("left_hippocampus", "right_m1")
    assert CONDITIONS == ("remesh", "fixed_mesh")
    assert METRICS == (
        "roi_median_v_per_m",
        "roi_mean_v_per_m",
        "roi_p95_v_per_m",
    )
    assert PRIMARY_METRIC == "roi_median_v_per_m"
    assert config.repeat_count == 40
    assert config.expected_cells == 80
    assert config.expected_rows_per_table == 800
    assert config.expected_rows == 3200
    assert config.output_root == Path(
        "/mnt/parscratch/users/cop23bi/"
        "final_132_repeatability_combined_analysis_v1"
    )


def test_production_sources_name_ten_metric_tables_and_ten_receipts() -> None:
    sources = production_config().sources

    assert len(sources.metric_tables) == 10
    assert len(sources.receipts) == 10
    assert sources.v3_base9_left_csv == Path(
        "/mnt/parscratch/users/cop23bi/"
        "final_132_repeatability_balanced_10_simnibs326_harmonized_"
        "base9_left_hippocampus_v2/_post_processing/"
        "repeatability_optimizer_roi_metrics_simnibs326h_paired_v2/"
        "optimizer_roi_metrics.csv"
    )
    assert sources.v4_corrected_fixed_right_csv == Path(
        "/mnt/parscratch/users/cop23bi/"
        "final_132_repeatability_balanced_10_right_m1_spherical_fixed_v1/"
        "_post_processing/optimizer_roi_metrics_v1/optimizer_roi_metrics.csv"
    )
    assert sources.v4_replacement_left_csv == Path(
        "/mnt/parscratch/users/cop23bi/simnibs_subject_replacements/"
        "sub-CC510415/simnibs4/left_hippocampus/_post_processing/"
        "repeatability_optimizer_roi_metrics_v1/optimizer_roi_metrics.csv"
    )
    assert all(path.name == "complete.json" for path in sources.receipts)


def test_scientific_sources_do_not_point_into_active_combined_dataset() -> None:
    config = production_config()
    active_copy = Path(
        "/mnt/parscratch/users/cop23bi/"
        "final_132_repeatability_combined_dataset_v1"
    )

    for path in (*config.sources.metric_tables.values(), *config.sources.receipts):
        assert path != active_copy
        assert active_copy not in path.parents


def test_production_configuration_is_immutable() -> None:
    config = production_config()

    with pytest.raises(FrozenInstanceError):
        config.repeat_count = 20  # type: ignore[misc]
