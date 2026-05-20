from copy import deepcopy

from post.utils.backfill_threshold_qc import backfill_payload


def _base_payload():
    return {
        "whole_brain_voxels": 100,
        "voxel_volume_mm3": 2.0,
        "extended_metrics": {
            "focality_threshold_v_per_m": 0.2,
            "focality_voxels_gt_threshold": 0,
            "focality_volume_mm3_gt_threshold": 0.0,
        },
        "extended_metrics_meta": {"status": "complete"},
        "subject_metrics_meta": {
            "status": "complete",
            "extended_metrics_status": "complete",
            "qc_status": "complete",
            "blocking_qc_checks": [],
            "nonblocking_qc_checks": ["overlays"],
        },
        "qc_meta": {"status": "complete", "error_checks": []},
        "rois": {
            "ctx_rh_G_front_middle": {
                "roi_voxels": 12,
                "focality_in_roi_voxels_gt_threshold": 0,
                "focality_in_roi_volume_mm3_gt_threshold": 0.0,
            }
        },
    }


def test_backfill_clears_spurious_overlay_nonblocking_qc_on_complete_payload():
    payload = _base_payload()

    assert backfill_payload(payload, default_threshold=0.2)

    subject_meta = payload["subject_metrics_meta"]
    assert subject_meta["status"] == "complete"
    assert subject_meta["nonblocking_qc_checks"] == []
    assert payload["threshold_qc"]["whole_brain"]["overlay_threshold"]["voxels"] == 0
    assert (
        payload["rois"]["ctx_rh_G_front_middle"]["threshold_qc"]["overlay_threshold"]["reason"]
        == "no_roi_voxels_at_or_above_overlay_threshold"
    )


def test_backfill_promotes_overlay_only_partial_payload_to_analysis_complete():
    payload = deepcopy(_base_payload())
    payload["subject_metrics_meta"].update(
        {"status": "partial", "qc_status": "partial", "nonblocking_qc_checks": []}
    )
    payload["qc_meta"] = {"status": "partial", "error_checks": ["overlays"]}

    assert backfill_payload(payload, default_threshold=0.2)

    subject_meta = payload["subject_metrics_meta"]
    assert subject_meta["status"] == "complete"
    assert subject_meta["blocking_qc_checks"] == []
    assert subject_meta["nonblocking_qc_checks"] == ["overlays"]
