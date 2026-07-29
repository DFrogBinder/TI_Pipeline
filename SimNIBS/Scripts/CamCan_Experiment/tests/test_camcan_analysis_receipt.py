import json
import subprocess
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "post"
    / "validate_camcan_analysis_receipt.py"
)


def _write_receipt(path: Path, **overrides) -> Path:
    payload = {
        "analysis_schema_version": 4,
        "status": "complete",
        "subjects": 132,
        "rois": [
            "Left_Hippocampus",
            "Left_M1",
            "Right_DLPC",
            "Right_Thalamus",
        ],
        "repeats_per_subject_roi": 10,
        "repeat_level_records": 5280,
        "subject_level_records": 528,
        "mni_baselines": 4,
        "execution_mode": "full_image_metric_extraction_and_aggregation",
    }
    payload.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _command(
    receipt: Path,
    post_root: Path,
    output_json: Path,
    output_summary: Path,
) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--receipt",
        str(receipt),
        "--expected-subjects",
        "132",
        "--expected-records",
        "5280",
        "--expected-post-campaign-root",
        str(post_root),
        "--output-json",
        str(output_json),
        "--output-summary",
        str(output_summary),
    ]


def test_accepts_complete_schema4_full_image_receipt(tmp_path):
    post_root = tmp_path / "post_processing"
    receipt = _write_receipt(
        post_root
        / "optimizer_matched_analysis_schema4"
        / "results"
        / "analysis_manifest.json"
    )
    output_json = tmp_path / "new" / "simulations.json"
    output_summary = tmp_path / "new" / "simulations.tsv"

    result = subprocess.run(
        _command(receipt, post_root, output_json, output_summary),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert result.returncode == 0
    assert payload["status"] == "complete"
    assert payload["complete"] == 5280
    assert payload["validation_mode"] == "prior_complete_analysis_receipt"
    assert payload["source_analysis_schema_version"] == 4
    assert len(payload["source_analysis_receipt_sha256"]) == 64
    assert output_summary.is_file()


def test_rejects_incomplete_or_wrong_schema_receipt(tmp_path):
    post_root = tmp_path / "post_processing"
    receipt = _write_receipt(
        post_root / "old" / "analysis_manifest.json",
        analysis_schema_version=3,
    )

    result = subprocess.run(
        _command(
            receipt,
            post_root,
            tmp_path / "simulations.json",
            tmp_path / "simulations.tsv",
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "analysis_schema_version=3" in result.stderr


def test_rejects_receipt_outside_expected_campaign(tmp_path):
    post_root = tmp_path / "post_processing"
    receipt = _write_receipt(tmp_path / "other" / "analysis_manifest.json")

    result = subprocess.run(
        _command(
            receipt,
            post_root,
            tmp_path / "simulations.json",
            tmp_path / "simulations.tsv",
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "outside the expected post-processing campaign root" in result.stderr
