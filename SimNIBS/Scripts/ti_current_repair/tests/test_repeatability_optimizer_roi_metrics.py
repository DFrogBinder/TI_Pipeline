import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


CURRENT_REPAIR_ROOT = Path(__file__).resolve().parents[1]
if str(CURRENT_REPAIR_ROOT) not in sys.path:
    sys.path.insert(0, str(CURRENT_REPAIR_ROOT))

from post import extract_repeatability_optimizer_roi_metrics as extractor
from stimulation_config import resolve_confirmed_stimulation


def _write_experiment(
    root: Path,
    *,
    roi_preset: str,
    label_id: int,
    run_count: int = 2,
) -> Path:
    subject = "sub-test"
    atlas_root = root / "atlases"
    atlas_root.mkdir(parents=True)
    atlas_data = np.full((11, 11, 11), label_id, dtype=np.int16)
    nib.save(
        nib.Nifti1Image(atlas_data, np.eye(4)),
        str(atlas_root / f"{subject}.nii.gz"),
    )

    config_path = root / "_pipeline" / "configs" / "paired_analysis.json"
    config_path.parent.mkdir(parents=True)
    config = {
        "source_root": str(root / "source"),
        "experiment_root": str(root),
        "subjects": [subject],
        "conditions": [
            {
                "name": "remesh",
                "mesh_mode": "remesh",
                "repeat_count": run_count,
            },
            {
                "name": "fixed_mesh",
                "mesh_mode": "fixed_mesh",
                "repeat_count": run_count,
            },
        ],
        "stimulation": resolve_confirmed_stimulation(roi_preset).to_dict(),
        "analysis": {
            "roi_preset": roi_preset,
            "compare_metric": "median_roi",
            "atlas_dir": str(atlas_root),
        },
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    receipt_path = root / "_pipeline" / "workflow" / "complete.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "scope": {
                    "subject_count": 1,
                    "repeats_per_condition": run_count,
                    "roi": roi_preset,
                },
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    for condition_index, condition in enumerate(("remesh", "fixed_mesh")):
        for run_number in range(1, run_count + 1):
            ti_path = (
                root
                / f"{subject}_repeatability"
                / condition
                / "repeats"
                / f"repeat_{run_number:03d}"
                / subject
                / "anat"
                / "SimNIBS"
                / "ti_brain_only.nii.gz"
            )
            ti_path.parent.mkdir(parents=True)
            value = float(1 + condition_index * run_count + run_number)
            nib.save(
                nib.Nifti1Image(
                    np.full((11, 11, 11), value, dtype=np.float32),
                    np.eye(4),
                ),
                str(ti_path),
            )
    return config_path


@pytest.mark.parametrize(
    ("roi_preset", "label_id", "expected_roi", "expected_volume"),
    [
        ("left-hippocampus", 17, "Left_Hippocampus", 200.0),
        ("right-m1", 12129, "Right_M1", 100.0),
    ],
)
def test_extract_and_collect_optimizer_roi_metrics(
    tmp_path: Path,
    roi_preset: str,
    label_id: int,
    expected_roi: str,
    expected_volume: float,
) -> None:
    root = tmp_path / roi_preset
    config_path = _write_experiment(
        root,
        roi_preset=roi_preset,
        label_id=label_id,
    )
    output_root = root / "_post_processing" / "optimizer_roi_metrics"

    preflight = extractor.preflight(
        config_path=config_path,
        output_root=output_root,
    )
    assert preflight["status"] == "ready"
    assert preflight["expected_fields"] == 4
    assert preflight["optimizer_roi"]["roi"] == expected_roi

    receipt = extractor.extract_subject(
        config_path=config_path,
        subject_index=0,
        output_root=output_root,
    )
    assert receipt["status"] == "complete"
    assert receipt["rows"] == 4
    assert receipt["grid_realizations"] == 1

    archive_path = output_root / "metrics.tar.gz"
    manifest = extractor.collect(
        config_path=config_path,
        output_root=output_root,
        archive_path=archive_path,
    )
    assert manifest["status"] == "complete"
    assert manifest["validation"]["rows"] == 4
    assert archive_path.is_file()
    assert archive_path.with_suffix(".gz.sha256").is_file()

    with (output_root / "optimizer_roi_metrics.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["roi"] for row in rows} == {expected_roi}
    assert {
        float(row["requested_roi_volume_mm3"]) for row in rows
    } == {expected_volume}
    assert [float(row["roi_median_v_per_m"]) for row in rows] == [
        2.0,
        3.0,
        4.0,
        5.0,
    ]
    assert all(
        int(row["roi_voxels"]) == int(row["finite_roi_voxels"])
        for row in rows
    )


def test_missing_target_label_fails_visible(tmp_path: Path) -> None:
    root = tmp_path / "missing-label"
    config_path = _write_experiment(
        root,
        roi_preset="right-m1",
        label_id=17,
        run_count=1,
    )
    with pytest.raises(ValueError, match="lacks required labels"):
        extractor.extract_subject(
            config_path=config_path,
            subject_index=0,
            output_root=root / "output",
        )
