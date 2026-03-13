import pytest

from utils.roi_registry import (
    FASTSURFER_ROI_ALIASES,
    match_fastsurfer_roi_from_directory,
    resolve_fastsurfer_roi_name,
)


def test_alias_dictionary_contains_expected_hippocampus_aliases():
    aliases = FASTSURFER_ROI_ALIASES["Left-Hippocampus"]
    assert "left_hippocampus" in aliases
    assert "lh_hippocampus" in aliases


def test_resolve_fastsurfer_roi_name_supports_right_m1_aliases():
    match = resolve_fastsurfer_roi_name("right_m1")
    assert match.canonical_name == "ctx-rh-precentral"

    typo_match = resolve_fastsurfer_roi_name("rigth_m1")
    assert typo_match.canonical_name == "ctx-rh-precentral"


def test_match_fastsurfer_roi_from_directory_uses_longest_specific_alias():
    match = match_fastsurfer_roi_from_directory("/tmp/Left_Hippocampus_Data_test")
    assert match.canonical_name == "Left-Hippocampus"
    assert match.matched_alias == "left_hippocampus"


def test_ambiguous_aliases_raise_clear_errors():
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_fastsurfer_roi_name("hippocampus")

    with pytest.raises(ValueError, match="ambiguous"):
        match_fastsurfer_roi_from_directory("hippocampus_data")


def test_pipeline_exits_before_analysis_for_unknown_dataset_roi():
    from post.run_post_processing import (
        PipelineConfig,
        PopulationConfig,
        PostBatchConfig,
        _resolve_pipeline_rois,
    )

    cfg = PipelineConfig(
        post=PostBatchConfig(
            root="/tmp/Not_A_Recognized_Target",
            atlas_mode="fastsurfer",
            plot_roi=None,
        ),
        population=PopulationConfig(enabled=False),
    )

    with pytest.raises(SystemExit, match="No analysis was started for this dataset"):
        _resolve_pipeline_rois(cfg)
