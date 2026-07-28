import json
import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from post import camcan_manuscript_analysis as manuscript
from post import optimizer_target_roi


TEST_ROI_DEFINITION = {
    "method": "synthetic test ROI",
    "requested_volume_mm3": 100.0,
    "achieved_volume_mm3": 100.0,
    "radius_mm": 3.0,
    "target_volume_reached": True,
}


def test_optimizer_roi_matches_make_rois_growth_and_volume_classes():
    anatomical = np.ones((21, 21, 21), dtype=bool)
    image = nib.Nifti1Image(np.zeros(anatomical.shape), np.eye(4))

    cortical = optimizer_target_roi.build_optimizer_target_roi(
        anatomical_mask=anatomical,
        reference_img=image,
        roi="Left_M1",
    )
    subcortical = optimizer_target_roi.build_optimizer_target_roi(
        anatomical_mask=anatomical,
        reference_img=image,
        roi="Left_Hippocampus",
    )

    assert cortical.metadata["requested_volume_mm3"] == 100.0
    assert cortical.metadata["achieved_volume_mm3"] == pytest.approx(
        cortical.mask.sum()
    )
    assert cortical.metadata["achieved_volume_mm3"] >= 100.0
    assert cortical.metadata["radius_mm"] == pytest.approx(3.01)
    assert subcortical.metadata["requested_volume_mm3"] == 200.0
    assert subcortical.metadata["achieved_volume_mm3"] >= 200.0
    assert subcortical.metadata["radius_mm"] > cortical.metadata["radius_mm"]
    assert np.all(cortical.mask <= anatomical)
    assert np.all(subcortical.mask <= anatomical)


def test_optimizer_roi_centroid_uses_world_coordinates_and_stays_in_parcel():
    anatomical = np.zeros((9, 9, 9), dtype=bool)
    anatomical[1, 2, 3] = True
    anatomical[3, 4, 5] = True
    affine = np.array(
        [[2.0, 0.0, 0.0, 10.0], [0.0, 3.0, 0.0, -5.0], [0.0, 0.0, 4.0, 7.0], [0, 0, 0, 1]]
    )
    image = nib.Nifti1Image(np.zeros(anatomical.shape), affine)

    target = optimizer_target_roi.build_optimizer_target_roi(
        anatomical_mask=anatomical,
        reference_img=image,
        roi="Right_Thalamus",
    )

    expected_world = nib.affines.apply_affine(affine, [2.0, 3.0, 4.0])
    observed_world = np.array(
        [
            target.metadata["centre_world_x_mm"],
            target.metadata["centre_world_y_mm"],
            target.metadata["centre_world_z_mm"],
        ]
    )
    assert observed_world == pytest.approx(expected_world)
    assert np.all(target.mask <= anatomical)
    assert target.metadata["target_volume_reached"] is False


def test_atlas_validation_requires_all_four_direct_roi_labels(tmp_path):
    atlas_path = tmp_path / "atlas.nii.gz"
    atlas = np.zeros((4, 4, 4), dtype=np.int32)
    for index, label in enumerate((17, 11129, 12115, 49)):
        atlas[index, 0, 0] = label
    nib.save(nib.Nifti1Image(atlas, np.eye(4)), atlas_path)

    result = manuscript.validate_atlas_rois(atlas_path)
    assert result["status"] == "complete"

    atlas[2, 0, 0] = 0
    nib.save(nib.Nifti1Image(atlas, np.eye(4)), atlas_path)
    with pytest.raises(ValueError, match="Right_DLPC"):
        manuscript.validate_atlas_rois(atlas_path)


def test_manuscript_metrics_use_supplied_roi_denominator_and_all_thresholds():
    ti_data = np.array(
        [[[0.20, 0.18], [0.15, np.nan]], [[0.10, 0.30], [0.00, 0.40]]],
        dtype=np.float32,
    )
    roi_mask = np.array(
        [[[True, True], [True, True]], [[False, False], [False, False]]],
        dtype=bool,
    )
    ti_img = nib.Nifti1Image(ti_data, np.eye(4))

    metrics = manuscript.compute_manuscript_metrics(
        ti_img=ti_img,
        ti_data=ti_data,
        roi_mask=roi_mask,
    )

    assert metrics["roi_voxels"] == 4
    assert metrics["roi_min_v_per_m"] == pytest.approx(0.15)
    assert metrics["roi_finite_voxels"] == 3
    assert metrics["roi_nonfinite_voxels"] == 1
    assert metrics["target_coverage_voxels_ge_0p18"] == 2
    assert metrics["target_coverage_percent_ge_0p18"] == pytest.approx(50.0)
    assert metrics["target_coverage_voxels_ge_0p15"] == 3
    assert metrics["target_coverage_percent_ge_0p15"] == pytest.approx(75.0)
    assert metrics["whole_brain_coverage_percent_ge_0p18"] == pytest.approx(
        4 / 7 * 100.0
    )
    assert metrics["off_target_coverage_percent_ge_0p18"] == pytest.approx(
        2 / 4 * 100.0
    )
    assert metrics["threshold_localization_percent_in_roi_ge_0p18"] == pytest.approx(
        2 / 4 * 100.0
    )


def test_top_five_metrics_are_coverage_and_localization():
    ti_data = np.arange(1, 101, dtype=np.float32).reshape(10, 10, 1)
    roi_mask = np.zeros_like(ti_data, dtype=bool)
    roi_mask[9, :, :] = True
    ti_img = nib.Nifti1Image(ti_data, np.eye(4))

    metrics = manuscript.compute_manuscript_metrics(
        ti_img=ti_img,
        ti_data=ti_data,
        roi_mask=roi_mask,
    )

    assert metrics["top_5_percent_voxels"] == 5
    assert metrics["top_5_percent_target_voxels"] == 5
    assert metrics["top_5_percent_target_coverage_percent"] == pytest.approx(50.0)
    assert metrics["top_5_percent_localization_percent_in_roi"] == pytest.approx(
        100.0
    )


def test_threshold_localization_is_zero_when_no_voxel_reaches_threshold():
    ti_data = np.full((2, 2, 2), 0.10, dtype=np.float32)
    roi_mask = np.zeros_like(ti_data, dtype=bool)
    roi_mask[0, 0, 0] = True
    ti_img = nib.Nifti1Image(ti_data, np.eye(4))

    metrics = manuscript.compute_manuscript_metrics(
        ti_img=ti_img,
        ti_data=ti_data,
        roi_mask=roi_mask,
        thresholds=(0.18,),
    )

    assert metrics["whole_brain_coverage_voxels_ge_0p18"] == 0
    assert metrics["target_coverage_voxels_ge_0p18"] == 0
    assert metrics["threshold_localization_percent_in_roi_ge_0p18"] == 0.0


def test_subject_extraction_writes_and_reuses_fingerprinted_record(tmp_path):
    dataset_root = tmp_path / "Left_Hippocampus_Data_01"
    subject = "sub-01"
    ti_path = (
        dataset_root
        / subject
        / "anat"
        / "SimNIBS"
        / "ti_brain_only.nii.gz"
    )
    ti_path.parent.mkdir(parents=True)
    ti_data = np.full((3, 3, 3), 0.1, dtype=np.float32)
    ti_data[1, 1, 1] = 0.2
    nib.save(nib.Nifti1Image(ti_data, np.eye(4)), ti_path)

    atlas_root = tmp_path / "atlases"
    atlas_root.mkdir()
    atlas_data = np.zeros((3, 3, 3), dtype=np.int16)
    atlas_data[1, 1, 1] = 17
    atlas_data[1, 1, 2] = 17
    nib.save(
        nib.Nifti1Image(atlas_data, np.eye(4)),
        atlas_root / f"{subject}.nii.gz",
    )
    task = manuscript.ExtractionTask(
        dataset_root=str(dataset_root),
        subject=subject,
        roi="Left_Hippocampus",
        repeat="01",
        canonical_roi="Left-Hippocampus",
        atlas_root=str(atlas_root),
        thresholds=manuscript.DEFAULT_THRESHOLDS_V_PER_M,
        top_percentile=manuscript.DEFAULT_TOP_PERCENTILE,
        robust_max_percentile=manuscript.DEFAULT_ROBUST_MAX_PERCENTILE,
        upper_tail_fraction=manuscript.DEFAULT_UPPER_TAIL_FRACTION,
        force=False,
    )

    first = manuscript._extract_subject(task)
    second = manuscript._extract_subject(task)

    assert first["status"] == "complete"
    assert second["status"] == "skipped"
    payload = json.loads(Path(first["output"]).read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["metrics"]["roi_voxels"] == 2
    assert payload["metrics"]["target_coverage_percent_ge_0p18"] == pytest.approx(
        50.0
    )


def test_collector_arithmetic_means_repeat_metrics(monkeypatch, tmp_path):
    study_root = tmp_path / "study"
    subjects_file = tmp_path / "subjects.txt"
    subjects_file.write_text("sub-01\nsub-02\n", encoding="utf-8")
    metric_names = manuscript.manuscript_metric_names()

    for roi in manuscript.ROI_ORDER:
        canonical = manuscript.match_fastsurfer_roi_from_directory(
            f"{roi}_Data_01"
        ).canonical_name
        for repeat_number in range(1, 11):
            repeat = f"{repeat_number:02d}"
            for subject_index, subject in enumerate(("sub-01", "sub-02"), start=1):
                path = (
                    study_root
                    / "runs"
                    / f"{roi}_Runs"
                    / f"{roi}_Data_{repeat}"
                    / subject
                    / "anat"
                    / "post"
                    / manuscript.METRIC_MARKER_FILENAME
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                metrics = {
                    name: float(repeat_number + subject_index)
                    for name in metric_names
                }
                if (
                    roi == "Left_Hippocampus"
                    and repeat_number == 1
                    and subject == "sub-01"
                ):
                    metrics["whole_brain_coverage_voxels_ge_0p18"] = 0
                    metrics["target_coverage_voxels_ge_0p18"] = 0
                    metrics["threshold_localization_percent_in_roi_ge_0p18"] = None
                path.write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "config_fingerprint": "test",
                            "subject": subject,
                            "roi": roi,
                            "repeat": repeat,
                            "canonical_roi": canonical,
                            "roi_definition": TEST_ROI_DEFINITION,
                            "metrics": metrics,
                        }
                    ),
                    encoding="utf-8",
                )

    monkeypatch.setattr(
        manuscript,
        "_compute_mni_record",
        lambda roi, **kwargs: {
            "subject": "MNI152",
            "roi": roi,
            **{name: 1.0 for name in metric_names},
        },
    )
    monkeypatch.setattr(
        manuscript,
        "_write_effectiveness_spread_figure",
        lambda **kwargs: None,
    )
    out_dir = tmp_path / "out"
    manifest = manuscript.collect_analysis(
        study_root=study_root,
        subjects_file=subjects_file,
        mni_atlas_path=tmp_path / "mni.nii.gz",
        mni_baseline_parent=tmp_path / "baselines",
        out_dir=out_dir,
        thresholds=manuscript.DEFAULT_THRESHOLDS_V_PER_M,
        top_percentile=manuscript.DEFAULT_TOP_PERCENTILE,
        robust_max_percentile=manuscript.DEFAULT_ROBUST_MAX_PERCENTILE,
        upper_tail_fraction=manuscript.DEFAULT_UPPER_TAIL_FRACTION,
    )

    assert manifest["status"] == "complete"
    assert manifest["repeat_level_records"] == 80
    assert manifest["subject_level_records"] == 8
    assert manifest["zero_denominator_localization_values_repaired"] == 1
    assert manifest["zero_denominator_localization_repairs_by_metric"] == {
        "threshold_localization_percent_in_roi_ge_0p2": 0,
        "threshold_localization_percent_in_roi_ge_0p18": 1,
        "threshold_localization_percent_in_roi_ge_0p15": 0,
    }
    repeat_frame = manuscript.pd.read_csv(out_dir / "repeat_level_metrics.csv")
    repaired = repeat_frame.loc[
        (repeat_frame["roi"] == "Left_Hippocampus")
        & (repeat_frame["subject"] == "sub-01")
        & (repeat_frame["repeat"] == 1),
        "threshold_localization_percent_in_roi_ge_0p18",
    ].iloc[0]
    assert repaired == 0.0
    subject_frame = manuscript.pd.read_csv(
        out_dir / "subject_level_repeat_mean_metrics.csv"
    )
    row = subject_frame.loc[
        (subject_frame["roi"] == "Left_Hippocampus")
        & (subject_frame["subject"] == "sub-01")
    ].iloc[0]
    assert row["roi_median_v_per_m"] == pytest.approx(6.5)
    assert row["repeat_count"] == 10
    assert (out_dir / "table_main_long.csv").is_file()
    assert (out_dir / "table_supplementary_descriptive_statistics.csv").is_file()


def test_collector_rejects_nonfinite_repeat_metrics(monkeypatch, tmp_path):
    study_root = tmp_path / "study"
    subjects_file = tmp_path / "subjects.txt"
    subjects_file.write_text("sub-01\n", encoding="utf-8")
    metric_names = manuscript.manuscript_metric_names()

    for roi in manuscript.ROI_ORDER:
        canonical = manuscript.match_fastsurfer_roi_from_directory(
            f"{roi}_Data_01"
        ).canonical_name
        for repeat_number in range(1, 11):
            repeat = f"{repeat_number:02d}"
            path = (
                study_root
                / "runs"
                / f"{roi}_Runs"
                / f"{roi}_Data_{repeat}"
                / "sub-01"
                / "anat"
                / "post"
                / manuscript.METRIC_MARKER_FILENAME
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            metrics = {name: 1.0 for name in metric_names}
            if roi == "Right_DLPC" and repeat == "01":
                metrics["threshold_localization_percent_in_roi_ge_0p18"] = None
            path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "config_fingerprint": "test",
                        "subject": "sub-01",
                        "roi": roi,
                        "repeat": repeat,
                        "canonical_roi": canonical,
                        "roi_definition": TEST_ROI_DEFINITION,
                        "metrics": metrics,
                    }
                ),
                encoding="utf-8",
            )

    with pytest.raises(
        RuntimeError,
        match="outside the defined 0/0 case",
    ):
        manuscript.collect_analysis(
            study_root=study_root,
            subjects_file=subjects_file,
            mni_atlas_path=tmp_path / "mni.nii.gz",
            mni_baseline_parent=tmp_path / "baselines",
            out_dir=tmp_path / "out",
            thresholds=manuscript.DEFAULT_THRESHOLDS_V_PER_M,
            top_percentile=manuscript.DEFAULT_TOP_PERCENTILE,
            robust_max_percentile=manuscript.DEFAULT_ROBUST_MAX_PERCENTILE,
            upper_tail_fraction=manuscript.DEFAULT_UPPER_TAIL_FRACTION,
        )


def test_csv_only_reaggregation_repairs_zero_denominator_localization(
    monkeypatch, tmp_path
):
    input_dir = tmp_path / "schema1"
    input_dir.mkdir()
    metric_names = manuscript.manuscript_metric_names()
    rows = []
    for roi in manuscript.ROI_ORDER:
        canonical = manuscript.match_fastsurfer_roi_from_directory(
            f"{roi}_Data_01"
        ).canonical_name
        for repeat_number in range(1, 11):
            row = {
                "subject": "sub-01",
                "roi": roi,
                "repeat": repeat_number,
                "canonical_roi": canonical,
                **{name: 1.0 for name in metric_names},
            }
            if roi == "Left_Hippocampus" and repeat_number == 1:
                row["whole_brain_coverage_voxels_ge_0p18"] = 0
                row["target_coverage_voxels_ge_0p18"] = 0
                row["threshold_localization_percent_in_roi_ge_0p18"] = np.nan
            rows.append(row)
    manuscript.pd.DataFrame(rows).to_csv(
        input_dir / "repeat_level_metrics.csv", index=False
    )
    manuscript.pd.DataFrame(
        [
            {
                "subject": "MNI152",
                "roi": roi,
                **{name: 1.0 for name in metric_names},
            }
            for roi in manuscript.ROI_ORDER
        ]
    ).to_csv(input_dir / "mni152_baseline_metrics.csv", index=False)
    (input_dir / "analysis_manifest.json").write_text(
        json.dumps({"analysis_schema_version": manuscript.ANALYSIS_SCHEMA_VERSION}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        manuscript,
        "_write_effectiveness_spread_figure",
        lambda **kwargs: None,
    )

    out_dir = tmp_path / "schema2"
    manifest = manuscript.reaggregate_existing_analysis(
        input_dir=input_dir,
        out_dir=out_dir,
        thresholds=manuscript.DEFAULT_THRESHOLDS_V_PER_M,
        top_percentile=manuscript.DEFAULT_TOP_PERCENTILE,
        robust_max_percentile=manuscript.DEFAULT_ROBUST_MAX_PERCENTILE,
        upper_tail_fraction=manuscript.DEFAULT_UPPER_TAIL_FRACTION,
    )

    assert manifest["status"] == "complete"
    assert manifest["execution_mode"] == "csv_only_reaggregation"
    assert manifest["image_metric_extraction_rerun"] is False
    assert manifest["zero_denominator_localization_values_repaired"] == 1
    repeat_frame = manuscript.pd.read_csv(out_dir / "repeat_level_metrics.csv")
    repaired = repeat_frame.loc[
        (repeat_frame["roi"] == "Left_Hippocampus")
        & (repeat_frame["repeat"] == 1),
        "threshold_localization_percent_in_roi_ge_0p18",
    ].iloc[0]
    assert repaired == 0.0
    subject_frame = manuscript.pd.read_csv(
        out_dir / "subject_level_repeat_mean_metrics.csv"
    )
    subject_value = subject_frame.loc[
        subject_frame["roi"] == "Left_Hippocampus",
        "threshold_localization_percent_in_roi_ge_0p18",
    ].iloc[0]
    assert subject_value == pytest.approx(0.9)
    assert subject_frame["repeat_count"].eq(10).all()


def test_csv_only_reaggregation_rejects_pre_optimizer_schema(tmp_path):
    input_dir = tmp_path / "schema2"
    input_dir.mkdir()
    (input_dir / "repeat_level_metrics.csv").write_text("subject\n", encoding="utf-8")
    (input_dir / "mni152_baseline_metrics.csv").write_text("subject\n", encoding="utf-8")
    (input_dir / "analysis_manifest.json").write_text(
        json.dumps({"analysis_schema_version": 2}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="cannot convert"):
        manuscript.reaggregate_existing_analysis(
            input_dir=input_dir,
            out_dir=tmp_path / "out",
            thresholds=manuscript.DEFAULT_THRESHOLDS_V_PER_M,
            top_percentile=manuscript.DEFAULT_TOP_PERCENTILE,
            robust_max_percentile=manuscript.DEFAULT_ROBUST_MAX_PERCENTILE,
            upper_tail_fraction=manuscript.DEFAULT_UPPER_TAIL_FRACTION,
        )


def test_effectiveness_spread_figure_writes_png_and_pdf(tmp_path):
    subject_rows = []
    mni_rows = []
    for roi_index, roi in enumerate(manuscript.ROI_ORDER):
        for subject_index in range(4):
            subject_rows.append(
                {
                    "subject": f"sub-{subject_index:02d}",
                    "roi": roi,
                    "target_coverage_percent_ge_0p18": 20 + roi_index * 12 + subject_index,
                    "off_target_coverage_percent_ge_0p18": 5 + roi_index + subject_index,
                }
            )
        mni_rows.append(
            {
                "subject": "MNI152",
                "roi": roi,
                "target_coverage_percent_ge_0p18": 30 + roi_index * 10,
                "off_target_coverage_percent_ge_0p18": 7 + roi_index,
            }
        )
    output_base = tmp_path / "effectiveness_spread"
    manuscript._write_effectiveness_spread_figure(
        subject_frame=manuscript.pd.DataFrame(subject_rows),
        mni_frame=manuscript.pd.DataFrame(mni_rows),
        threshold=0.18,
        spread_scope="off_target",
        output_base=output_base,
    )

    assert output_base.with_suffix(".png").stat().st_size > 0
    assert output_base.with_suffix(".pdf").stat().st_size > 0


def test_manuscript_submitter_uses_40_resumable_jobs_then_one_collector():
    pipeline_dir = Path(__file__).resolve().parents[1] / "cohort_pipeline"
    text = (pipeline_dir / "submit_cohort_manuscript_analysis.sh").read_text(
        encoding="utf-8"
    )

    assert 'MAX_CONCURRENT_DATASETS="${MAX_CONCURRENT_DATASETS:-40}"' in text
    assert '--array="0-39%${MAX_CONCURRENT_DATASETS}"' in text
    assert '--dependency="afterok:${SUBJECT_JOB}"' in text
    assert "MANUSCRIPT_THRESHOLDS_COLON:-0.20:0.18:0.15" in text
    assert "optimizer_matched_analysis" in text
    assert "100 mm3 cortical; 200 mm3 subcortical" in text
    assert "MANUSCRIPT_ROBUST_MAX_PERCENTILE:-99.9" in text
    assert "existing simulations are read-only" in text


def test_manuscript_submitter_preflight_accepts_complete_synthetic_scope(tmp_path):
    pipeline_dir = Path(__file__).resolve().parents[1] / "cohort_pipeline"
    study_root = tmp_path / "study"
    campaign_root = study_root / "campaigns" / "test"
    post_root = campaign_root / "post_processing"
    subject = "sub-01"
    subjects_file = tmp_path / "subjects.txt"
    subjects_file.write_text(f"{subject}\n", encoding="utf-8")
    study_config = tmp_path / "study.json"
    study_config.write_text(
        json.dumps({"hpc_study_root": str(study_root)}), encoding="utf-8"
    )
    cohort_config = tmp_path / "cohort.json"
    cohort_config.write_text(
        json.dumps({"expected_subjects": 1}), encoding="utf-8"
    )
    manifest = campaign_root / "simulation_tasks.tsv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("task_id\n", encoding="utf-8")
    receipt = campaign_root / "release_state" / "chain_complete.tsv"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("status\tcomplete\n", encoding="utf-8")

    atlas_root = tmp_path / "atlases"
    atlas_root.mkdir()
    (atlas_root / f"{subject}.nii.gz").write_bytes(b"atlas")
    mni_atlas = atlas_root / "sub-mni152.nii.gz"
    mni_atlas.write_bytes(b"mni")
    baseline_parent = tmp_path / "MNI152-data"
    for roi in manuscript.ROI_ORDER:
        for repeat_number in range(1, 11):
            ti_path = (
                study_root
                / "runs"
                / f"{roi}_Runs"
                / f"{roi}_Data_{repeat_number:02d}"
                / subject
                / "anat"
                / "SimNIBS"
                / "ti_brain_only.nii.gz"
            )
            ti_path.parent.mkdir(parents=True)
            ti_path.write_bytes(b"ti")
        baseline_ti = (
            baseline_parent
            / manuscript.MNI_BASELINE_NAMES[roi]
            / "MNI152"
            / "anat"
            / "SimNIBS"
            / "ti_brain_only.nii.gz"
        )
        baseline_ti.parent.mkdir(parents=True)
        baseline_ti.write_bytes(b"mni-ti")

    fake_workflow = tmp_path / "workflow.py"
    fake_workflow.write_text(
        "import json,pathlib,sys\n"
        "args=sys.argv\n"
        "summary=pathlib.Path(args[args.index('--summary')+1])\n"
        "summary.parent.mkdir(parents=True,exist_ok=True)\n"
        "summary.write_text('task_id\\tstatus\\n',encoding='utf-8')\n"
        "summary.with_suffix('.json').write_text("
        "json.dumps({'status':'complete','complete':40}),encoding='utf-8')\n",
        encoding="utf-8",
    )
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/bash\nprintf '%s\\n' manuscript_analysis_dependencies=ready\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    completed = subprocess.run(
        [
            "bash",
            str(pipeline_dir / "submit_cohort_manuscript_analysis.sh"),
            "test",
            "--preflight",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "STUDY_CONFIG": str(study_config),
            "COHORT_CONFIG": str(cohort_config),
            "STUDY_ROOT": str(study_root),
            "CAMPAIGN_ROOT": str(campaign_root),
            "POST_CAMPAIGN_ROOT": str(post_root),
            "SUBJECTS_FILE": str(subjects_file),
            "SIMULATION_MANIFEST": str(manifest),
            "CHAIN_RECEIPT": str(receipt),
            "WORKFLOW_PY": str(fake_workflow),
            "FASTSURFER_ROOT": str(atlas_root),
            "MNI_FIXED_ATLAS_PATH": str(mni_atlas),
            "MNI_BASELINE_PARENT": str(baseline_parent),
            "PYTHON": str(fake_python),
        },
    )

    assert "repeat-level metric records: 40" in completed.stdout
    assert "thresholds: 0.20 and 0.18 and 0.15 V/m" in completed.stdout
    assert "Preflight passed without submitting jobs." in completed.stdout
