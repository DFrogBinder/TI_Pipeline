import json

import numpy as np
import pandas as pd

from post import build_camcan_supervisor_revision_figures as revision


def _write_synthetic_inputs(tmp_path):
    cohort_dir = tmp_path / "cohort"
    personalized_dir = tmp_path / "personalized"
    cohort_dir.mkdir()
    personalized_dir.mkdir()

    cohort_manifest = {
        "analysis_schema_version": 4,
        "status": "complete",
        "subjects": 132,
        "repeat_level_records": 5280,
        "subject_level_records": 528,
        "mni_baselines": 4,
        "thresholds_v_per_m": [0.2, 0.18, 0.15],
    }
    (cohort_dir / "analysis_manifest.json").write_text(
        json.dumps(cohort_manifest),
        encoding="utf-8",
    )
    subject_rows = []
    mni_rows = []
    for roi_index, roi in enumerate(revision.ROI_ORDER):
        for subject_index in range(132):
            coverage = subject_index / 131 * 100.0
            mean = 0.10 + coverage / 500.0 + roi_index * 0.01
            off_target = (
                0.4 + 0.03 * coverage
                if roi not in revision.DEEP_ROIS
                else 0.6 + 0.08 * np.expm1(coverage / 22.0)
            )
            subject_rows.append(
                {
                    "subject": f"sub-{subject_index:03d}",
                    "roi": roi,
                    "roi_min_v_per_m": mean - 0.035,
                    "roi_mean_v_per_m": mean,
                    "roi_median_v_per_m": mean - 0.005,
                    "roi_robust_max_p99_9_v_per_m": mean + 0.045,
                    "target_coverage_percent_ge_0p2": coverage,
                    "off_target_coverage_percent_ge_0p2": off_target,
                }
            )
        mni_rows.append(
            {
                "subject": "MNI152",
                "roi": roi,
                "roi_min_v_per_m": 0.17 + roi_index * 0.01,
                "roi_mean_v_per_m": 0.20 + roi_index * 0.01,
                "roi_median_v_per_m": 0.195 + roi_index * 0.01,
                "roi_robust_max_p99_9_v_per_m": 0.25 + roi_index * 0.01,
                "target_coverage_percent_ge_0p2": 50.0,
                "off_target_coverage_percent_ge_0p2": 2.0 + roi_index,
            }
        )
    pd.DataFrame(subject_rows).to_csv(
        cohort_dir / "subject_level_repeat_mean_metrics.csv",
        index=False,
    )
    pd.DataFrame(mni_rows).to_csv(
        cohort_dir / "mni152_baseline_metrics.csv",
        index=False,
    )

    personalized_manifest = {
        "comparison_schema_version": 3,
        "manuscript_analysis_schema_version": 4,
        "status": "complete",
        "subject_roi_configurations": 28,
        "repeat_level_records": 560,
        "condition_repeat_mean_records": 56,
    }
    (personalized_dir / "analysis_manifest.json").write_text(
        json.dumps(personalized_manifest),
        encoding="utf-8",
    )
    paired_rows = []
    repeat_rows = []
    subjects = [f"sub-{index:02d}" for index in range(7)]
    pair_index = 0
    for roi_index, roi in enumerate(revision.ROI_ORDER):
        for subject_index, subject in enumerate(subjects):
            generic_target = float(subject_index * 12)
            personalized_target = min(100.0, generic_target + 20.0)
            generic_off = 0.5 + roi_index + subject_index * 0.2
            personalized_off = max(0.05, generic_off * 0.7)
            row = {
                "pair_index": pair_index,
                "subject": subject,
                "roi": roi,
                "selection_role": "cross_target",
            }
            metrics = {
                "roi_min_v_per_m": (0.12 + subject_index * 0.01, 0.16),
                "roi_mean_v_per_m": (0.16 + subject_index * 0.01, 0.20),
                "roi_median_v_per_m": (0.15 + subject_index * 0.01, 0.195),
                "roi_robust_max_p99_9_v_per_m": (
                    0.23 + subject_index * 0.01,
                    0.27,
                ),
                "target_coverage_percent_ge_0p2": (
                    generic_target,
                    personalized_target,
                ),
                "off_target_coverage_percent_ge_0p2": (
                    generic_off,
                    personalized_off,
                ),
            }
            for metric, (generic, personalized) in metrics.items():
                row[f"{metric}__generic_repeat_mean"] = generic
                row[f"{metric}__generic_repeat_sd"] = 0.01
                row[f"{metric}__personalized_repeat_mean"] = personalized
                row[f"{metric}__personalized_repeat_sd"] = 0.01
            paired_rows.append(row)
            for condition_index, condition in enumerate(revision.CONDITIONS):
                for repeat_index in range(10):
                    repeat_rows.append(
                        {
                            "pair_index": pair_index,
                            "subject": subject,
                            "roi": roi,
                            "condition": condition,
                            "repeat": f"{repeat_index + 1:02d}",
                            "roi_min_v_per_m": (
                                metrics["roi_min_v_per_m"][condition_index]
                                + repeat_index / 10_000
                            ),
                            "roi_mean_v_per_m": (
                                metrics["roi_mean_v_per_m"][condition_index]
                                + repeat_index / 10_000
                            ),
                            "roi_median_v_per_m": (
                                metrics["roi_median_v_per_m"][condition_index]
                                + repeat_index / 10_000
                            ),
                            "roi_robust_max_p99_9_v_per_m": (
                                metrics["roi_robust_max_p99_9_v_per_m"][
                                    condition_index
                                ]
                                + repeat_index / 10_000
                            ),
                        }
                    )
            pair_index += 1
    pd.DataFrame(paired_rows).to_csv(
        personalized_dir / "paired_personalized_vs_generic.csv",
        index=False,
    )
    pd.DataFrame(repeat_rows).to_csv(
        personalized_dir / "repeat_level_metrics.csv",
        index=False,
    )
    return cohort_dir, personalized_dir


def test_supervisor_revision_builds_complete_figure_set(tmp_path):
    cohort_dir, personalized_dir = _write_synthetic_inputs(tmp_path)
    output_dir = tmp_path / "out"

    result = revision.build(
        cohort_dir,
        personalized_dir,
        output_dir,
        force=False,
    )

    assert result["status"] == "complete"
    assert result["best_worst_visual_encoding"] is False
    assert result["trajectory_arrows"] is False
    assert result["condition_connecting_lines"] is False
    assert result["subject_row_condition_offset"] is False
    assert result["field_summaries"] == [
        "minimum",
        "mean",
        "median",
        "maximum (P99.9)",
    ]
    assert result["panel_roi_order"] == [
        "Left_Hippocampus",
        "Left_M1",
        "Right_Thalamus",
        "Right_DLPC",
    ]
    assert result["personalized_summary_split_by_roi"] is True
    assert result["personalized_ratio_split_by_roi"] is True
    assert (
        result["personalized_repeat_distributions_split_by_roi_and_statistic"]
        is True
    )
    assert len(list((output_dir / "figures").glob("*.png"))) == 32
    assert len(list((output_dir / "figures").glob("*.pdf"))) == 32
    assert (output_dir / "tables" / "table_descriptive_fit_statistics.csv").is_file()
    fits = pd.read_csv(output_dir / "tables" / "table_descriptive_fit_statistics.csv")
    assert set(fits["model"]) == {"linear"}
    assert (
        output_dir / "tables" / "table_deep_target_linear_vs_exponential.csv"
    ).is_file()
    assert (output_dir / "figure_captions.md").is_file()


def test_mni_relative_v4_centres_metrics_without_mutating_sources(tmp_path):
    cohort_dir, personalized_dir = _write_synthetic_inputs(tmp_path)
    _, subjects, mni = revision.load_cohort(cohort_dir)
    _, paired, repeats = revision.load_personalized(personalized_dir)
    original_subjects = subjects.copy(deep=True)
    original_mni = mni.copy(deep=True)
    original_paired = paired.copy(deep=True)
    original_repeats = repeats.copy(deep=True)

    (
        relative_subjects,
        relative_mni,
        relative_paired,
        relative_repeats,
        audit,
    ) = revision.make_mni_relative_tables(subjects, mni, paired, repeats)

    metrics = [
        *(metric for metric, _, _ in revision.FIELD_METRICS),
        "target_coverage_percent_ge_0p2",
        "off_target_coverage_percent_ge_0p2",
    ]
    for roi in revision.ROI_ORDER:
        subject_mask = subjects["roi"] == roi
        paired_mask = paired["roi"] == roi
        repeat_mask = repeats["roi"] == roi
        for metric in metrics:
            baseline = float(mni.loc[roi, metric])
            assert float(relative_mni.loc[roi, metric]) == 0.0
            np.testing.assert_allclose(
                relative_subjects.loc[subject_mask, metric],
                subjects.loc[subject_mask, metric] - baseline,
            )
            for condition in revision.CONDITIONS:
                column = f"{metric}__{condition}_repeat_mean"
                np.testing.assert_allclose(
                    relative_paired.loc[paired_mask, column],
                    paired.loc[paired_mask, column] - baseline,
                )
            if metric in repeats:
                np.testing.assert_allclose(
                    relative_repeats.loc[repeat_mask, metric],
                    repeats.loc[repeat_mask, metric] - baseline,
                )

        target = subjects.loc[
            subject_mask, "target_coverage_percent_ge_0p2"
        ].to_numpy(dtype=float)
        off_target = subjects.loc[
            subject_mask, "off_target_coverage_percent_ge_0p2"
        ].to_numpy(dtype=float)
        mni_ratio = float(
            mni.loc[roi, "off_target_coverage_percent_ge_0p2"]
            / mni.loc[roi, "target_coverage_percent_ge_0p2"]
        )
        expected = np.full(len(target), np.nan)
        valid = target > 0
        expected[valid] = off_target[valid] / target[valid] - mni_ratio
        np.testing.assert_allclose(
            relative_subjects.loc[
                subject_mask, revision.MNI_RELATIVE_RATIO_COHORT
            ],
            expected,
            equal_nan=True,
        )

    assert len(audit) == len(revision.ROI_ORDER) * (len(metrics) + 2)
    pd.testing.assert_frame_equal(subjects, original_subjects)
    pd.testing.assert_frame_equal(mni, original_mni)
    pd.testing.assert_frame_equal(paired, original_paired)
    pd.testing.assert_frame_equal(repeats, original_repeats)

    output_dir = tmp_path / "v4"
    result = revision.build(
        cohort_dir,
        personalized_dir,
        output_dir,
        force=False,
        mni_relative=True,
    )
    assert result["status"] == "complete"
    assert result["figure_revision_schema_version"] == 5
    assert result["figure_revision_variant"] == "mni_relative_v4"
    assert result["mni_relative"] is True
    assert result["threshold_v_per_m"] == 0.2
    assert result["roi_specific_mni_mean_threshold_recalculation"] is False
    assert len(list((output_dir / "figures").glob("*.png"))) == 32
    assert len(list((output_dir / "figures").glob("*.pdf"))) == 32
    assert (
        output_dir / "tables" / "table_mni152_reference_values.csv"
    ).is_file()
    assert (
        output_dir / "tables" / "table_mni_relative_transform_audit.csv"
    ).is_file()
