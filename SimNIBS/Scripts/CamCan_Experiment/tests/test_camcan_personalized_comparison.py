import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from post import camcan_personalized_comparison as comparison


def test_repository_allowlist_contains_all_28_optimized_configurations():
    pipeline_dir = Path(__file__).resolve().parents[1] / "cohort_pipeline"
    cohort_dir = pipeline_dir / "cohorts" / "optimized_best_worst_7"
    scripts_dir = pipeline_dir.parents[1]

    frame = comparison.build_allowlist(
        cohort_config=cohort_dir / "cohort.json",
        individualized_targets_csv=cohort_dir / "individualized_targets.csv",
        generic_targets_csv=scripts_dir / "utils" / "targets.csv",
    )

    observed = {
        (row.roi, row.selection_role): row.subject
        for row in frame.loc[
            frame["selection_role"].isin(["best", "worst"])
        ].itertuples()
    }
    assert observed == {
        ("Left_Hippocampus", "best"): "sub-CC410182",
        ("Left_Hippocampus", "worst"): "sub-CC420100",
        ("Left_M1", "best"): "sub-CC221775",
        ("Left_M1", "worst"): "sub-CC320776",
        ("Right_DLPC", "best"): "sub-CC120795",
        ("Right_DLPC", "worst"): "sub-CC410387",
        ("Right_Thalamus", "best"): "sub-CC610061",
        ("Right_Thalamus", "worst"): "sub-CC420100",
    }
    assert len(frame) == 28
    assert frame["subject"].nunique() == 7
    assert frame.groupby("roi").size().eq(7).all()
    assert frame["selection_role"].eq("cross_target").sum() == 20
    assert frame["personalized_pareto_selection"].eq("TI_free.Emin").all()


def _source_fixture(
    root: Path,
    *,
    pair: dict,
    repeat: str,
    targets_hash: str,
    individualized_hash: str | None,
) -> None:
    ti = comparison._ti_path(root, pair["roi"], repeat, pair["subject"])
    ti.parent.mkdir(parents=True, exist_ok=True)
    ti.write_bytes(b"ti")
    marker = comparison._marker_path(root, pair["roi"], repeat, pair["subject"])
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "status": "complete",
                "subject": pair["subject"],
                "roi": pair["roi"],
                "repeat_id": repeat,
                "targets_csv_sha256": targets_hash,
                "individualized_targets_csv_sha256": individualized_hash,
                "mesh_sha256": "mesh",
                "corrected_label_sha256": "label",
                "pareto_selection": (
                    "TI_free.Emin" if individualized_hash is not None else None
                ),
                "optimized_configuration": (
                    pair["personalized_configuration"]
                    if individualized_hash is not None
                    else None
                ),
                "optimized_pair1": (
                    pair["personalized_pair1"]
                    if individualized_hash is not None
                    else None
                ),
                "optimized_pair2": (
                    pair["personalized_pair2"]
                    if individualized_hash is not None
                    else None
                ),
                "optimized_current1_ma": (
                    pair["personalized_current1_ma"]
                    if individualized_hash is not None
                    else None
                ),
                "optimized_current2_ma": (
                    pair["personalized_current2_ma"]
                    if individualized_hash is not None
                    else None
                ),
                "optimized_e_target_v_per_m": (
                    pair["personalized_optimization_e_target_v_per_m"]
                    if individualized_hash is not None
                    else None
                ),
                "optimized_stimulated_volume": (
                    pair["personalized_optimization_stimulated_volume"]
                    if individualized_hash is not None
                    else None
                ),
            }
        ),
        encoding="utf-8",
    )


def test_source_validation_distinguishes_generic_and_personalized(tmp_path):
    pair = {
        "subject": "sub-01",
        "roi": "Left_M1",
        "personalized_configuration": 12,
        "personalized_pair1": "F1-F2",
        "personalized_pair2": "C3-CP3",
        "personalized_current1_ma": 2.0,
        "personalized_current2_ma": 0.8,
        "personalized_optimization_e_target_v_per_m": 0.2,
        "personalized_optimization_stimulated_volume": 0.3,
    }
    generic_root = tmp_path / "generic"
    personalized_root = tmp_path / "personalized"
    _source_fixture(
        generic_root,
        pair=pair,
        repeat="01",
        targets_hash="targets",
        individualized_hash=None,
    )
    _source_fixture(
        personalized_root,
        pair=pair,
        repeat="01",
        targets_hash="targets",
        individualized_hash="individualized",
    )

    generic = comparison.validate_source_record(
        condition="generic",
        pair=pair,
        repeat="01",
        study_root=generic_root,
        targets_sha256="targets",
        individualized_targets_sha256="individualized",
    )
    personalized = comparison.validate_source_record(
        condition="personalized",
        pair=pair,
        repeat="01",
        study_root=personalized_root,
        targets_sha256="targets",
        individualized_targets_sha256="individualized",
    )

    assert generic["corrected_label_sha256"] == personalized["corrected_label_sha256"]
    marker = comparison._marker_path(generic_root, pair["roi"], "01", pair["subject"])
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["individualized_targets_csv_sha256"] = "unexpected"
    marker.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpectedly uses individualized"):
        comparison.validate_source_record(
            condition="generic",
            pair=pair,
            repeat="01",
            study_root=generic_root,
            targets_sha256="targets",
            individualized_targets_sha256="individualized",
        )


def test_collector_requires_and_aggregates_exact_560_records(monkeypatch, tmp_path):
    output_root = tmp_path / "comparison"
    allowlist_rows = []
    metric_names = comparison.manuscript_metric_names(comparison.DEFAULT_THRESHOLDS)
    for pair_index in range(comparison.EXPECTED_CONFIGURATIONS):
        roi = comparison.ROI_ORDER[pair_index // comparison.EXPECTED_SUBJECTS]
        subject_index = pair_index % comparison.EXPECTED_SUBJECTS
        allowlist_rows.append(
            {
                "pair_index": pair_index,
                "subject": f"sub-{subject_index:02d}",
                "roi": roi,
                "selection_role": (
                    "best"
                    if subject_index == 0
                    else "worst" if subject_index == 1 else "cross_target"
                ),
            }
        )
    allowlist = pd.DataFrame(allowlist_rows)
    allowlist_path = output_root / "selection_allowlist.csv"
    allowlist_path.parent.mkdir(parents=True)
    allowlist.to_csv(allowlist_path, index=False)

    for pair in allowlist_rows:
        summary_path = (
            output_root / "pair_summaries" / f"pair_{pair['pair_index']:02d}.json"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "pair_index": pair["pair_index"],
                    "records": 20,
                }
            ),
            encoding="utf-8",
        )
        canonical = comparison.match_fastsurfer_roi_from_directory(
            f"{pair['roi']}_Data_01"
        ).canonical_name
        for condition_index, condition in enumerate(comparison.CONDITIONS):
            for repeat_number, repeat in enumerate(comparison.REPEATS, start=1):
                path = (
                    output_root
                    / "repeat_records"
                    / condition
                    / pair["roi"]
                    / pair["subject"]
                    / f"repeat_{repeat}.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                value = float(repeat_number + condition_index * 10)
                path.write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "config_fingerprint": f"{pair['pair_index']}-{condition}-{repeat}",
                            "pair_index": pair["pair_index"],
                            "subject": pair["subject"],
                            "roi": pair["roi"],
                            "canonical_roi": canonical,
                            "selection_role": pair["selection_role"],
                            "condition": condition,
                            "repeat": repeat,
                            "source": {
                                "ti_path": "ti",
                                "marker_path": "marker",
                                "mesh_sha256": "mesh",
                                "corrected_label_sha256": "label",
                            },
                            "roi_definition": {
                                "method": "synthetic test ROI",
                                "requested_volume_mm3": 100.0,
                                "achieved_volume_mm3": 100.0,
                                "radius_mm": 3.0,
                                "target_volume_reached": True,
                            },
                            "metrics": {name: value for name in metric_names},
                        }
                    ),
                    encoding="utf-8",
                )

    monkeypatch.setattr(comparison, "_write_paired_dumbbell", lambda *a, **k: None)
    monkeypatch.setattr(
        comparison,
        "_write_effectiveness_trajectories",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(comparison, "_write_repeat_distribution", lambda *a, **k: None)
    manifest = comparison.collect_analysis(
        allowlist_path=allowlist_path,
        output_root=output_root,
        thresholds=comparison.DEFAULT_THRESHOLDS,
        top_percentile=comparison.DEFAULT_TOP_PERCENTILE,
        robust_max_percentile=comparison.DEFAULT_ROBUST_MAX_PERCENTILE,
        upper_tail_fraction=comparison.DEFAULT_UPPER_TAIL_FRACTION,
    )

    assert manifest["status"] == "complete"
    assert manifest["subject_roi_configurations"] == 28
    assert manifest["repeat_level_records"] == 560
    assert manifest["condition_repeat_mean_records"] == 56
    assert manifest["excluded_personalized_simulations"] == 0
    condition_frame = pd.read_csv(
        output_root / "results" / "condition_repeat_mean_metrics.csv"
    )
    generic = condition_frame.loc[
        (condition_frame["pair_index"] == 0)
        & (condition_frame["condition"] == "generic"),
        "roi_median_v_per_m",
    ].iloc[0]
    personalized = condition_frame.loc[
        (condition_frame["pair_index"] == 0)
        & (condition_frame["condition"] == "personalized"),
        "roi_median_v_per_m",
    ].iloc[0]
    assert generic == pytest.approx(5.5)
    assert personalized == pytest.approx(15.5)
    long = pd.read_csv(
        output_root / "results" / "paired_personalized_vs_generic_long.csv"
    )
    row = long.loc[
        (long["pair_index"] == 0) & (long["metric"] == "roi_median_v_per_m")
    ].iloc[0]
    assert row["absolute_change_personalized_minus_generic"] == pytest.approx(10.0)
    assert (output_root / "results" / "table_stimulation_parameters.csv").is_file()
    assert (output_root / "results" / "paired_personalized_vs_generic.csv").is_file()


def test_submitter_declares_all_configuration_optimizer_scope():
    pipeline_dir = Path(__file__).resolve().parents[1] / "cohort_pipeline"
    text = (pipeline_dir / "submit_personalized_vs_generic_analysis.sh").read_text(
        encoding="utf-8"
    )

    assert '--array="0-27%${MAX_CONCURRENT_PAIRS}"' in text
    assert '--dependency="afterok:${PAIR_JOB}"' in text
    assert "required repeat-level metric inputs: 560" in text
    assert "personalized simulations excluded as out of scope: 0" in text
    assert "MakeROIs.m-equivalent" in text
    assert "repeat pairing across conditions: none" in text
    assert "source simulations: read-only" in text


def test_submitter_preflight_validates_exact_synthetic_scope(tmp_path):
    pipeline_dir = Path(__file__).resolve().parents[1] / "cohort_pipeline"
    cohort_dir = pipeline_dir / "cohorts" / "optimized_best_worst_7"
    scripts_dir = pipeline_dir.parents[1]
    allowlist = comparison.build_allowlist(
        cohort_config=cohort_dir / "cohort.json",
        individualized_targets_csv=cohort_dir / "individualized_targets.csv",
        generic_targets_csv=scripts_dir / "utils" / "targets.csv",
    )
    targets_hash = comparison.sha256_file(scripts_dir / "utils" / "targets.csv")
    individualized_hash = comparison.sha256_file(
        cohort_dir / "individualized_targets.csv"
    )
    generic_root = tmp_path / "generic"
    personalized_root = tmp_path / "personalized"
    atlas_root = tmp_path / "atlases"
    atlas_root.mkdir()
    for pair in allowlist.to_dict(orient="records"):
        (atlas_root / f"{pair['subject']}.nii.gz").write_bytes(b"atlas")
        for repeat in comparison.REPEATS:
            _source_fixture(
                generic_root,
                pair=pair,
                repeat=repeat,
                targets_hash=targets_hash,
                individualized_hash=None,
            )
            _source_fixture(
                personalized_root,
                pair=pair,
                repeat=repeat,
                targets_hash=targets_hash,
                individualized_hash=individualized_hash,
            )
    generic_receipt = tmp_path / "generic_complete.tsv"
    personalized_receipt = tmp_path / "personalized_complete.tsv"
    generic_receipt.write_text("status\tcomplete\n", encoding="utf-8")
    personalized_receipt.write_text("status\tcomplete\n", encoding="utf-8")

    completed = subprocess.run(
        [
            "bash",
            str(pipeline_dir / "submit_personalized_vs_generic_analysis.sh"),
            "--preflight",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GENERIC_STUDY_ROOT": str(generic_root),
            "PERSONALIZED_STUDY_ROOT": str(personalized_root),
            "FASTSURFER_ROOT": str(atlas_root),
            "GENERIC_CHAIN_RECEIPT": str(generic_receipt),
            "PERSONALIZED_CHAIN_RECEIPT": str(personalized_receipt),
            "COMPARISON_ROOT": str(tmp_path / "comparison"),
            "PYTHON": sys.executable,
            "MPLCONFIGDIR": str(tmp_path / "mpl"),
        },
    )

    assert "subject-ROI configurations: 28" in completed.stdout
    assert "required repeat-level metric inputs: 560" in completed.stdout
    assert "personalized simulations excluded as out of scope: 0" in completed.stdout
    assert "Preflight passed without submitting jobs." in completed.stdout
    preflight = json.loads(
        (tmp_path / "comparison" / "preflight.json").read_text(encoding="utf-8")
    )
    assert preflight["status"] == "ready"
    assert preflight["subject_roi_configurations"] == 28
    assert preflight["required_generic_inputs"] == 280
    assert preflight["required_personalized_inputs"] == 280


def test_comparison_figures_write_png_and_pdf(tmp_path):
    condition_rows = []
    repeat_rows = []
    for pair_index in range(comparison.EXPECTED_CONFIGURATIONS):
        roi = comparison.ROI_ORDER[pair_index // comparison.EXPECTED_SUBJECTS]
        subject_index = pair_index % comparison.EXPECTED_SUBJECTS
        role = (
            "best"
            if subject_index == 0
            else "worst" if subject_index == 1 else "cross_target"
        )
        subject = f"sub-{subject_index:02d}"
        for condition_index, condition in enumerate(comparison.CONDITIONS):
            condition_rows.append(
                {
                    "pair_index": pair_index,
                    "subject": subject,
                    "roi": roi,
                    "selection_role": role,
                    "condition": condition,
                    "roi_min_v_per_m": 0.08 + 0.02 * condition_index,
                    "roi_mean_v_per_m": 0.09 + 0.02 * condition_index,
                    "roi_median_v_per_m": 0.10 + 0.02 * condition_index,
                    "target_coverage_percent_ge_0p2": 18 + 10 * condition_index,
                    "off_target_coverage_percent_ge_0p2": 7 - condition_index,
                    "target_coverage_percent_ge_0p18": 20 + 10 * condition_index,
                    "off_target_coverage_percent_ge_0p18": 8 - condition_index,
                    "threshold_localization_percent_in_roi_ge_0p18": 25
                    + 8 * condition_index,
                }
            )
            for repeat_number in range(10):
                repeat_rows.append(
                    {
                        "pair_index": pair_index,
                        "subject": subject,
                        "roi": roi,
                        "selection_role": role,
                        "condition": condition,
                        "repeat": f"{repeat_number + 1:02d}",
                        "roi_min_v_per_m": (
                            0.08 + 0.02 * condition_index + repeat_number / 1000
                        ),
                        "roi_mean_v_per_m": (
                            0.09 + 0.02 * condition_index + repeat_number / 1000
                        ),
                        "roi_median_v_per_m": (
                            0.10 + 0.02 * condition_index + repeat_number / 1000
                        ),
                    }
                )
    condition_frame = pd.DataFrame(condition_rows)
    repeat_frame = pd.DataFrame(repeat_rows)
    bases = (
        tmp_path / "dumbbell",
        tmp_path / "trajectories",
        tmp_path / "repeats",
    )
    comparison._write_paired_dumbbell(condition_frame, bases[0])
    comparison._write_effectiveness_trajectories(
        condition_frame,
        threshold=0.18,
        output_base=bases[1],
    )
    comparison._write_repeat_distribution(repeat_frame, bases[2])

    for base in bases:
        assert base.with_suffix(".png").stat().st_size > 0
        assert base.with_suffix(".pdf").stat().st_size > 0
