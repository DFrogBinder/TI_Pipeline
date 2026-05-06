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


@pytest.mark.parametrize(
    ("dataset_root", "canonical_name", "matched_alias"),
    [
        ("/tmp/Right_Thalamus_Data", "Right-Thalamus-Proper", "right_thalamus"),
        ("/tmp/Left_Accumbens_Data", "Left-Accumbens-area", "left_accumbens"),
    ],
)
def test_match_fastsurfer_roi_from_directory_supports_shorthand_region_names(
    dataset_root: str,
    canonical_name: str,
    matched_alias: str,
):
    match = match_fastsurfer_roi_from_directory(dataset_root)
    assert match.canonical_name == canonical_name
    assert match.matched_alias == matched_alias


def test_ambiguous_aliases_raise_clear_errors():
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_fastsurfer_roi_name("hippocampus")

    with pytest.raises(ValueError, match="ambiguous"):
        match_fastsurfer_roi_from_directory("hippocampus_data")

    with pytest.raises(ValueError, match="ambiguous"):
        resolve_fastsurfer_roi_name("thalamus")


def test_resolve_fastsurfer_roi_name_supports_shorthand_thalamus_aliases():
    match = resolve_fastsurfer_roi_name("right_thalamus")
    assert match.canonical_name == "Right-Thalamus-Proper"

    match = resolve_fastsurfer_roi_name("left_accumbens")
    assert match.canonical_name == "Left-Accumbens-area"


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


def test_pipeline_resolves_shorthand_dataset_roi_to_canonical_label():
    from post.run_post_processing import (
        PipelineConfig,
        PopulationConfig,
        PostBatchConfig,
        _resolve_pipeline_rois,
    )

    cfg = PipelineConfig(
        post=PostBatchConfig(
            root="/tmp/Right_Thalamus_Data",
            atlas_mode="fastsurfer",
            plot_roi=None,
        ),
        population=PopulationConfig(enabled=False),
    )

    _resolve_pipeline_rois(cfg)

    assert cfg.post.plot_roi == "Right-Thalamus-Proper"
    assert cfg.population.target_roi == "Right-Thalamus-Proper"
