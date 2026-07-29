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
    assert result["field_summaries"] == [
        "minimum",
        "mean",
        "median",
        "robust maximum (P99.9)",
    ]
    assert len(list((output_dir / "figures").glob("*.png"))) == 12
    assert len(list((output_dir / "figures").glob("*.pdf"))) == 12
    assert (output_dir / "tables" / "table_descriptive_fit_statistics.csv").is_file()
    fits = pd.read_csv(output_dir / "tables" / "table_descriptive_fit_statistics.csv")
    assert set(fits["model"]) == {"linear"}
    assert (
        output_dir / "tables" / "table_deep_target_linear_vs_exponential.csv"
    ).is_file()
    assert (output_dir / "figure_captions.md").is_file()
