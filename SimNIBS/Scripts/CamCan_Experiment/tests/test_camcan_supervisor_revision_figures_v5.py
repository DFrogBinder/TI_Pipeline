import json
from pathlib import Path

import numpy as np
import pandas as pd

from post import build_camcan_supervisor_revision_figures_v5 as revision


def _synthetic_inputs(tmp_path):
    cohort = tmp_path / "cohort"
    personalized = tmp_path / "personalized"
    cohort.mkdir()
    personalized.mkdir()
    threshold_table = (
        tmp_path / revision.Path("mni152_simnibs401_roi_thresholds.csv").name
    )
    threshold_rows = []
    for index, roi in enumerate(revision.ROI_ORDER):
        mean = 0.17 + index * 0.01
        threshold_rows.append(
            {
                "roi": roi,
                "roi_group": "superficial" if index < 2 else "deep",
                "roi_order": index + 1,
                "threshold_v_per_m": mean,
                "mni_min_v_per_m": mean - 0.03,
                "mni_mean_v_per_m": mean,
                "mni_max_p99_9_v_per_m": mean + 0.05,
                "simnibs_version": "4.0.1",
                "source_table_sha256": "synthetic",
            }
        )
    pd.DataFrame(threshold_rows).to_csv(threshold_table, index=False)
    thresholds = [row["threshold_v_per_m"] for row in threshold_rows]
    cohort_manifest = {
        "analysis_schema_version": 4,
        "status": "complete",
        "subjects": 132,
        "repeat_level_records": 5280,
        "subject_level_records": 528,
        "mni_baselines": 4,
        "thresholds_v_per_m": thresholds,
    }
    (cohort / "analysis_manifest.json").write_text(json.dumps(cohort_manifest))
    personalized_manifest = {
        "comparison_schema_version": 3,
        "manuscript_analysis_schema_version": 4,
        "status": "complete",
        "subject_roi_configurations": 28,
        "repeat_level_records": 560,
        "condition_repeat_mean_records": 56,
        "thresholds_v_per_m": thresholds,
    }
    (personalized / "analysis_manifest.json").write_text(
        json.dumps(personalized_manifest)
    )

    cohort_rows = []
    mni_rows = []
    paired_rows = []
    slugs = [revision.threshold_slug(value) for value in thresholds]
    for roi_index, roi in enumerate(revision.ROI_ORDER):
        mean = thresholds[roi_index]
        mni_row = {
            "subject": "MNI152",
            "roi": roi,
            "roi_min_v_per_m": mean - 0.03,
            "roi_mean_v_per_m": mean,
            "roi_robust_max_p99_9_v_per_m": mean + 0.05,
        }
        for slug_index, slug in enumerate(slugs):
            mni_row[f"target_coverage_percent_ge_{slug}"] = 42 + slug_index
            mni_row[f"off_target_coverage_percent_ge_{slug}"] = 4 + slug_index
        mni_rows.append(mni_row)
        for subject_index in range(132):
            row = {
                "subject": f"sub-{subject_index:03d}",
                "roi": roi,
                "roi_min_v_per_m": mean - 0.04 + subject_index / 10000,
                "roi_mean_v_per_m": mean - 0.02 + subject_index / 3500,
                "roi_robust_max_p99_9_v_per_m": mean + 0.04
                + subject_index / 5000,
            }
            for slug_index, slug in enumerate(slugs):
                target = 10 + subject_index * 0.55 + slug_index
                off_target = 0.4 + target * (0.08 + 0.01 * roi_index)
                if (
                    roi == "Right_Thalamus"
                    and subject_index == 131
                    and slug_index == roi_index
                ):
                    off_target = 0.0
                row[f"target_coverage_percent_ge_{slug}"] = target
                row[f"off_target_coverage_percent_ge_{slug}"] = off_target
            cohort_rows.append(row)
        for subject_index in range(7):
            row = {
                "subject": f"sub-P{subject_index:02d}",
                "roi": roi,
            }
            for slug_index, slug in enumerate(slugs):
                generic_target = 15 + 8 * subject_index + slug_index
                personalized_target = generic_target + 12
                generic_off = 1.2 + roi_index + subject_index * 0.15
                personalized_off = generic_off * 0.65
                if (
                    roi == "Left_Hippocampus"
                    and subject_index == 6
                    and slug_index == roi_index
                ):
                    personalized_off = 0.0
                for metric, generic, optimized in (
                    (
                        f"target_coverage_percent_ge_{slug}",
                        generic_target,
                        personalized_target,
                    ),
                    (
                        f"off_target_coverage_percent_ge_{slug}",
                        generic_off,
                        personalized_off,
                    ),
                ):
                    row[f"{metric}__generic_repeat_mean"] = generic
                    row[f"{metric}__personalized_repeat_mean"] = optimized
            paired_rows.append(row)
    pd.DataFrame(cohort_rows).to_csv(
        cohort / "subject_level_repeat_mean_metrics.csv", index=False
    )
    pd.DataFrame(mni_rows).to_csv(
        cohort / "mni152_baseline_metrics.csv", index=False
    )
    pd.DataFrame(paired_rows).to_csv(
        personalized / "paired_personalized_vs_generic.csv", index=False
    )
    return cohort, personalized, threshold_table


def test_builds_only_final_four_figures_and_audits_infinity(tmp_path):
    cohort, personalized, threshold_table = _synthetic_inputs(tmp_path)
    output = tmp_path / "figures_v5"
    payload = revision.build(
        cohort,
        personalized,
        threshold_table,
        output,
        force=False,
    )

    assert payload["status"] == "complete"
    assert payload["axis_scale_policy"] == "linear only"
    assert payload["ratio_definition"] == "target coverage / off-target coverage"
    assert sorted(path.stem for path in (output / "figures").glob("*.png")) == sorted(
        revision.EXPECTED_FIGURE_STEMS
    )
    assert not list((output / "figures").glob("*.pdf"))
    ratio_audit = pd.read_csv(
        output / "tables" / "table_ratio_zero_denominator_audit.csv"
    ).set_index("roi")
    assert ratio_audit.loc[
        "Right_Thalamus", "infinite_censored_subjects"
    ] == 1
    personalized_table = pd.read_csv(
        output / "tables" / "table_personalization_subject_changes.csv"
    )
    censored = personalized_table.loc[
        (personalized_table["roi"] == "Left_Hippocampus")
        & (personalized_table["condition"] == "personalized")
        & (personalized_table["panel"] == "target_to_off_target_ratio")
    ]
    assert "infinite" in set(censored["ratio_status"])
    note = (
        output / "SUPERVISOR_NOTE_RATIO_ZERO_DENOMINATORS.md"
    ).read_text()
    assert "positive infinity" in note
    assert "0/0" in note
    captions = (output / "figure_captions.md").read_text()
    assert "Pale violins show the population density" in captions
    assert "Generic-to-personalized changes" in captions
    assert "ordinary least-squares fits" in captions
    assert "The analysis contains 132 adults" not in captions
    assert "independently remeshed simulations" not in captions
    assert len(pd.read_csv(output / "figure_captions.csv")) == len(
        revision.EXPECTED_FIGURE_STEMS
    )
    assert payload["figure_revision_schema_version"] == 9
    assert payload["population_centering"].startswith("none")
    assert "absolute MNI152 value" in payload["population_mni_markers"]
    assert not (
        output / "figures" / "figure_mni152_absolute_mean_target_field.png"
    ).exists()
    assert (
        output
        / "figures"
        / (
            "figure_personalization_subject_changes_all_rois_"
            "at_mni_roi_threshold.png"
        )
    ).is_file()


def test_population_violin_preserves_style_and_marks_absolute_mni():
    figure, axis = revision.plt.subplots()
    values = [
        np.linspace(0.10 + index * 0.01, 0.25 + index * 0.01, 132)
        for index in range(4)
    ]
    mni_values = np.asarray([0.17, 0.18, 0.19, 0.20])
    artists = revision._violin(
        axis,
        values,
        color=revision.BLUE,
        rng=np.random.default_rng(20260729),
        mni_values=mni_values,
    )

    assert len(artists["bodies"]) == 4
    assert all(body.get_alpha() == 0.20 for body in artists["bodies"])
    assert len(artists["samples"]) == 4
    assert all(len(sample.get_offsets()) == 44 for sample in artists["samples"])
    np.testing.assert_allclose(
        artists["mni"].get_offsets(),
        np.column_stack((np.arange(1, 5), mni_values)),
    )
    revision.plt.close(figure)


def test_refuses_old_fixed_threshold_aggregates(tmp_path):
    cohort, personalized, threshold_table = _synthetic_inputs(tmp_path)
    cohort_manifest = json.loads(
        (cohort / "analysis_manifest.json").read_text()
    )
    cohort_manifest["thresholds_v_per_m"] = [0.2, 0.18, 0.15]
    (cohort / "analysis_manifest.json").write_text(
        json.dumps(cohort_manifest)
    )
    try:
        revision.build(
            cohort,
            personalized,
            threshold_table,
            tmp_path / "must_fail",
            force=False,
        )
    except RuntimeError as error:
        assert "does not contain required threshold" in str(error)
    else:
        raise AssertionError("fixed-threshold aggregates were accepted")


def test_mni_threshold_wrapper_uses_nonempty_command_arrays():
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "cohort_pipeline"
        / "submit_mni_threshold_manuscript_reanalysis.sh"
    )
    text = wrapper.read_text(encoding="utf-8")

    assert "EXTRA_ARGS" not in text
    assert "COHORT_COMMAND=(" in text
    assert "PERSONALIZED_COMMAND=(" in text
    assert '"${COHORT_COMMAND[@]}"' in text
    assert '"${PERSONALIZED_COMMAND[@]}"' in text
    assert "COHORT_SOURCE_RECEIPT=" in text
    assert "MANUSCRIPT_SOURCE_VALIDATION_RECEIPT=" in text


def test_v5_personalized_collector_skips_legacy_fixed_threshold_figures():
    collector = (
        Path(__file__).resolve().parents[1]
        / "cohort_pipeline"
        / "cohort_personalized_comparison_collect.slurm"
    )
    text = collector.read_text(encoding="utf-8")

    assert "COLLECT_COMMAND=(" in text
    assert "package_camcan_publication_inputs_v5.py" in text
    assert "COLLECT_COMMAND+=(--skip-legacy-figures)" in text
    assert '"${COLLECT_COMMAND[@]}"' in text
