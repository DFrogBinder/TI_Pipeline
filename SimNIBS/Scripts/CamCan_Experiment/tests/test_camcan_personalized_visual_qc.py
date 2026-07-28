import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from post import camcan_personalized_visual_qc as visual_qc


def _pair(pair_index: int = 0) -> dict:
    return {
        "pair_index": pair_index,
        "subject": f"sub-{pair_index:02d}",
        "roi": "Left_Hippocampus",
        "selection_role": "best" if pair_index % 2 == 0 else "worst",
        "generic_pair1": "F8-P8",
        "generic_pair2": "T7-P7",
        "generic_current1_ma": 2.0,
        "generic_current2_ma": 1.588656,
        "personalized_pair1": "Fpz-F4",
        "personalized_pair2": "P9-I1",
        "personalized_current1_ma": 2.0,
        "personalized_current2_ma": 2.0,
    }


def test_condition_specific_electrodes_do_not_reuse_generic_montage():
    pair = _pair()
    assert visual_qc._electrode_names(pair, "generic") == (
        "F8",
        "P8",
        "T7",
        "P7",
    )
    assert visual_qc._electrode_names(pair, "personalized") == (
        "Fpz",
        "F4",
        "P9",
        "I1",
    )


def test_source_dataset_root_is_derived_from_validated_ti_layout(tmp_path):
    source = (
        tmp_path
        / "Left_Hippocampus_Data_01"
        / "sub-01"
        / "anat"
        / "SimNIBS"
        / "ti_brain_only.nii.gz"
    )
    assert visual_qc._source_dataset_root(source, "sub-01") == (
        tmp_path / "Left_Hippocampus_Data_01"
    )


def test_common_scale_paired_renderer_writes_a_nonempty_png(tmp_path):
    shape = (21, 21, 21)
    affine = np.eye(4)
    grid = np.indices(shape, dtype=float)
    radius = np.sqrt(sum((axis - 10.0) ** 2 for axis in grid))
    t1_data = np.maximum(0.0, 100.0 - 5.0 * radius)
    generic_data = np.exp(-((radius / 5.0) ** 2)) * 0.24
    personalized_data = np.exp(-((radius / 4.0) ** 2)) * 0.30
    atlas_data = np.zeros(shape, dtype=np.int16)
    atlas_data[8:13, 8:13, 8:13] = 17

    t1_path = tmp_path / "t1.nii.gz"
    generic_path = tmp_path / "generic.nii.gz"
    personalized_path = tmp_path / "personalized.nii.gz"
    atlas_path = tmp_path / "atlas.nii.gz"
    nib.save(nib.Nifti1Image(t1_data, affine), t1_path)
    nib.save(nib.Nifti1Image(generic_data, affine), generic_path)
    nib.save(nib.Nifti1Image(personalized_data, affine), personalized_path)
    nib.save(nib.Nifti1Image(atlas_data, affine), atlas_path)

    pair = _pair()
    pair["subject"] = "sub-00"
    base_record = {
        "pair_index": 0,
        "subject": "sub-00",
        "roi": "Left_Hippocampus",
        "canonical_roi": "Left-Hippocampus",
        "selection_role": "best",
        "repeat": "01",
        "source_t1_path": str(t1_path),
        "atlas_path": str(atlas_path),
    }
    generic = {
        **base_record,
        "condition": "generic",
        "source": {"ti_path": str(generic_path)},
    }
    personalized = {
        **base_record,
        "condition": "personalized",
        "source": {"ti_path": str(personalized_path)},
    }
    output = tmp_path / "paired.png"
    payload = visual_qc._render_paired_repeat(
        pair=pair,
        repeat="01",
        generic_record=generic,
        personalized_record=personalized,
        vmax=0.30,
        threshold=0.20,
        output_path=output,
    )

    assert output.stat().st_size > 0
    assert payload["common_vmax_v_per_m"] == 0.30
    assert payload["threshold_v_per_m"] == 0.20


def test_zero_support_threshold_writes_explicit_legacy_placeholders(tmp_path):
    shape = (21, 21, 21)
    affine = np.eye(4)
    grid = np.indices(shape, dtype=float)
    radius = np.sqrt(sum((axis - 10.0) ** 2 for axis in grid))
    t1_data = np.maximum(0.0, 100.0 - 5.0 * radius)
    ti_data = np.exp(-((radius / 5.0) ** 2)) * 0.19
    atlas_data = np.zeros(shape, dtype=np.int16)
    atlas_data[8:13, 8:13, 8:13] = 17

    t1_path = tmp_path / "t1.nii.gz"
    ti_path = tmp_path / "ti.nii.gz"
    atlas_path = tmp_path / "atlas.nii.gz"
    nib.save(nib.Nifti1Image(t1_data, affine), t1_path)
    nib.save(nib.Nifti1Image(ti_data, affine), ti_path)
    nib.save(nib.Nifti1Image(atlas_data, affine), atlas_path)
    metrics = {
        "qc_meta": {
            "checks": {
                "overlays": {
                    "missing_overlay_types": [
                        "context_threshold",
                        "roi_focus_threshold",
                    ]
                }
            }
        },
        "threshold_qc": {
            "whole_brain": {
                "overlay_threshold": {
                    "threshold": 0.20,
                    "voxels": 0,
                    "has_voxels": False,
                }
            }
        },
    }

    paths = visual_qc._write_expected_empty_threshold_overlays(
        metrics=metrics,
        output_dir=tmp_path / "post",
        subject="sub-00",
        canonical_roi="Left-Hippocampus",
        ti_path=ti_path,
        t1_path=t1_path,
        atlas_path=atlas_path,
    )

    assert len(paths) == 2
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    assert paths[0].name.endswith("_context_sub-00_above0.20.png")
    assert paths[1].name.endswith("_roi_focus_sub-00_above0.20.png")


def test_zero_support_direct_recount_handles_legacy_metrics_schema(tmp_path):
    """Mirror the HPC records whose metrics lacked the expected QC nesting."""

    shape = (21, 21, 21)
    affine = np.eye(4)
    grid = np.indices(shape, dtype=float)
    radius = np.sqrt(sum((axis - 10.0) ** 2 for axis in grid))
    t1_data = np.maximum(0.0, 100.0 - 5.0 * radius)
    ti_data = np.exp(-((radius / 5.0) ** 2)) * 0.19
    atlas_data = np.zeros(shape, dtype=np.int16)
    atlas_data[8:13, 8:13, 8:13] = 17

    t1_path = tmp_path / "t1.nii.gz"
    ti_path = tmp_path / "ti.nii.gz"
    atlas_path = tmp_path / "atlas.nii.gz"
    nib.save(nib.Nifti1Image(t1_data, affine), t1_path)
    nib.save(nib.Nifti1Image(ti_data, affine), ti_path)
    nib.save(nib.Nifti1Image(atlas_data, affine), atlas_path)

    existing = []
    for index in range(5):
        path = tmp_path / f"existing_{index}.png"
        path.write_bytes(b"existing")
        existing.append(path)

    paths = visual_qc._write_expected_empty_threshold_overlays(
        metrics={},
        output_dir=tmp_path / "post",
        subject="sub-00",
        canonical_roi="Left-Hippocampus",
        ti_path=ti_path,
        t1_path=t1_path,
        atlas_path=atlas_path,
        existing_overlay_paths=existing,
    )

    assert len(paths) == 2
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_direct_recount_does_not_repair_nonzero_threshold_support(tmp_path):
    shape = (9, 9, 9)
    affine = np.eye(4)
    ti_data = np.zeros(shape, dtype=float)
    ti_data[4, 4, 4] = 0.21
    ti_path = tmp_path / "ti.nii.gz"
    nib.save(nib.Nifti1Image(ti_data, affine), ti_path)
    existing = []
    for index in range(5):
        path = tmp_path / f"existing_{index}.png"
        path.write_bytes(b"existing")
        existing.append(path)

    paths = visual_qc._write_expected_empty_threshold_overlays(
        metrics={},
        output_dir=tmp_path / "post",
        subject="sub-00",
        canonical_roi="Left-Hippocampus",
        ti_path=ti_path,
        t1_path=tmp_path / "unused-t1.nii.gz",
        atlas_path=tmp_path / "unused-atlas.nii.gz",
        existing_overlay_paths=existing,
    )

    assert paths == []


def test_missing_threshold_overlays_are_not_repaired_when_support_is_nonzero(
    tmp_path,
):
    metrics = {
        "qc_meta": {
            "checks": {
                "overlays": {
                    "missing_overlay_types": [
                        "context_threshold",
                        "roi_focus_threshold",
                    ]
                }
            }
        },
        "threshold_qc": {
            "whole_brain": {
                "overlay_threshold": {
                    "threshold": 0.20,
                    "voxels": 1,
                    "has_voxels": True,
                }
            }
        },
    }

    paths = visual_qc._write_expected_empty_threshold_overlays(
        metrics=metrics,
        output_dir=tmp_path,
        subject="sub-00",
        canonical_roi="Left-Hippocampus",
        ti_path=tmp_path / "missing-ti.nii.gz",
        t1_path=tmp_path / "missing-t1.nii.gz",
        atlas_path=tmp_path / "missing-atlas.nii.gz",
    )

    assert paths == []


def test_visual_qc_collector_requires_exact_selected_product(tmp_path):
    output_root = tmp_path / "visual_qc"
    allowlist_rows = []
    for pair_index in range(8):
        pair = _pair(pair_index)
        pair["roi"] = (
            "Left_Hippocampus",
            "Left_Hippocampus",
            "Left_M1",
            "Left_M1",
            "Right_DLPC",
            "Right_DLPC",
            "Right_Thalamus",
            "Right_Thalamus",
        )[pair_index]
        allowlist_rows.append(pair)
        visual_rows = []
        for mode in ("full_field", "above_0p20"):
            for repeat in visual_qc.REPEATS:
                image = (
                    output_root
                    / "paired_fields"
                    / f"pair_{pair_index:02d}"
                    / f"repeat_{repeat}_{mode}.png"
                )
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(b"png")
                visual_rows.append(
                    {
                        "path": str(image),
                        "repeat": repeat,
                        "mode": mode,
                        "common_vmax_v_per_m": 0.3,
                    }
                )
        visual_index = (
            output_root
            / "pair_summaries"
            / f"pair_{pair_index:02d}_visual_index.csv"
        )
        visual_index.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(visual_rows).to_csv(visual_index, index=False)
        reports = []
        for mode in ("full_field", "above_0p20"):
            report = (
                output_root
                / "pair_reports"
                / f"pair_{pair_index:02d}_{mode}_10_repeats.pdf"
            )
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_bytes(b"pdf")
            reports.append(str(report))
        summary = {
            "status": "complete",
            "pair_index": pair_index,
            "subject": pair["subject"],
            "roi": pair["roi"],
            "selection_role": pair["selection_role"],
            "post_records": 20,
            "legacy_overlays": 140,
            "paired_visualizations": 20,
            "pair_reports": 2,
            "visual_index": str(visual_index),
            "report_paths": reports,
        }
        (
            output_root / "pair_summaries" / f"pair_{pair_index:02d}.json"
        ).write_text(json.dumps(summary), encoding="utf-8")
        for condition in visual_qc.CONDITIONS:
            for repeat in visual_qc.REPEATS:
                record = visual_qc._post_record_path(
                    output_root,
                    pair_index,
                    condition,
                    repeat,
                )
                record.parent.mkdir(parents=True, exist_ok=True)
                post_output = (
                    output_root
                    / "post_products"
                    / f"pair_{pair_index:02d}"
                    / condition
                    / f"repeat_{repeat}"
                )
                post_output.mkdir(parents=True, exist_ok=True)
                subject_metrics = post_output / "subject_metrics.json"
                subject_metrics.write_text("{}", encoding="utf-8")
                record.write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "pair_index": pair_index,
                            "subject": pair["subject"],
                            "roi": pair["roi"],
                            "selection_role": pair["selection_role"],
                            "condition": condition,
                            "repeat": repeat,
                            "post_output_dir": str(post_output),
                            "subject_metrics_path": str(subject_metrics),
                        }
                    ),
                    encoding="utf-8",
                )

    allowlist_path = output_root / "selection_allowlist.csv"
    pd.DataFrame(allowlist_rows).to_csv(allowlist_path, index=False)
    manifest = visual_qc.collect_qc(
        allowlist_path=allowlist_path,
        output_root=output_root,
    )

    assert manifest["status"] == "complete"
    assert manifest["selected_subject_roi_pairs"] == 8
    assert manifest["full_post_records"] == 160
    assert manifest["paired_visualization_pngs"] == 160
    assert manifest["multipage_pair_reports"] == 16
    assert manifest["excluded_out_of_scope_personalized_simulations"] == 200
    assert (output_root / "results" / "README.md").is_file()


def test_submitter_declares_strict_qc_scope_and_isolated_outputs():
    pipeline_dir = Path(__file__).resolve().parents[1] / "cohort_pipeline"
    text = (
        pipeline_dir / "submit_personalized_vs_generic_visual_qc.sh"
    ).read_text(encoding="utf-8")

    assert 'PAIR_ARRAY_SPEC="${PAIR_ARRAY_SPEC:-0-7%${MAX_CONCURRENT_PAIRS}}"' in text
    assert '--array="${PAIR_ARRAY_SPEC}"' in text
    assert "resumable recovery of pair" in text
    assert "Submit with: ${SUBMIT_COMMAND}" in text
    assert "final validated product scope remains: 8 pairs" in text
    assert "full subject-level post-processing records: 160" in text
    assert "personalized simulations excluded as out of scope: 200" in text
    assert "existing source post directories modified: no" in text
    assert "VISUAL_QC_ROOT" in text
