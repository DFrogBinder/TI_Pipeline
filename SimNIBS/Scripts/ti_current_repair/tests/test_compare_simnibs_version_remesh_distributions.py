import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CURRENT_REPAIR_ROOT = Path(__file__).resolve().parents[1]
if str(CURRENT_REPAIR_ROOT) not in sys.path:
    sys.path.insert(0, str(CURRENT_REPAIR_ROOT))

from post import compare_simnibs_version_remesh_distributions as comparison  # noqa: E402


def _target_frame(target: str, version: str) -> pd.DataFrame:
    meta = comparison.TARGETS[target]
    roi_voxels = int(meta["requested_volume_mm3"])
    rows = []
    metric_offsets = {
        "roi_mean_v_per_m": 0.010,
        "roi_median_v_per_m": 0.000,
        "roi_p95_v_per_m": 0.025,
    }
    for subject_index in range(10):
        subject = f"sub-CC{subject_index:06d}"
        for condition in ("remesh", "fixed_mesh"):
            for repeat_index in range(1, 41):
                base = 0.10 + 0.01 * subject_index
                spread = 0.001 * repeat_index if condition == "remesh" else 0.0
                version_shift = 0.005 if version == "v3" and condition == "remesh" else 0.0
                nonfinite = (
                    1
                    if version == "v3"
                    and target == "left_hippocampus"
                    and condition == "remesh"
                    and repeat_index % 10 == 0
                    else 0
                )
                finite = roi_voxels - nonfinite
                row = {
                    "schema_version": 2,
                    "subject": subject,
                    "condition": condition,
                    "repeat_tag": f"repeat_{repeat_index:03d}",
                    "roi": meta["roi"],
                    "roi_voxels": roi_voxels,
                    "requested_roi_volume_mm3": meta["requested_volume_mm3"],
                    "achieved_roi_volume_mm3": float(roi_voxels),
                    "roi_radius_mm": 3.5 + 0.05 * subject_index,
                    "finite_roi_voxels": finite,
                    "nonfinite_roi_voxels": nonfinite,
                    "finite_roi_fraction": finite / roi_voxels,
                }
                for metric, offset in metric_offsets.items():
                    row[metric] = base + spread + version_shift + offset
                rows.append(row)
    return pd.DataFrame(rows)


def _write_inputs(root: Path) -> dict[str, Path]:
    paths = {}
    for version in ("v3", "v4"):
        for target in comparison.TARGETS:
            path = root / f"{version}_{target}.csv"
            _target_frame(target, version).to_csv(path, index=False)
            paths[f"{version}_{target}_csv"] = path
    return paths


def _args(paths: dict[str, Path], out_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(**paths, out_dir=out_dir)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_empirical_percentile_and_common_language_probability_half_weight_ties() -> None:
    assert comparison.empirical_midrank_percentile(2.0, np.array([1, 2, 3, 4])) == 37.5
    assert comparison.common_language_probability(
        np.array([2.0, 4.0]), np.array([1.0, 2.0])
    ) == 0.875


def test_complete_inputs_produce_the_approved_comparison_scope(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    frame = comparison.load_inputs(**paths)
    tables = comparison.build_comparison_tables(frame)

    assert len(frame) == 3200
    assert len(frame.query("condition == 'remesh'")) == 1600
    assert len(tables["run_percentiles"]) == 2400
    assert len(tables["participant_metrics"]) == 60
    assert len(tables["primary_summary"]) == 20
    assert len(tables["participant_inference"]) == 6
    assert set(tables["primary_summary"]["metric"]) == {"roi_median_v_per_m"}
    assert tables["participant_inference"][
        "holm_adjusted_p_across_six_tests"
    ].tolist() == pytest.approx([12 / 1024] * 6)

    row = tables["participant_metrics"].query(
        "target == 'left_hippocampus' and subject == 'sub-CC000000' "
        "and metric == 'roi_median_v_per_m'"
    ).iloc[0]
    v3 = np.array([0.10 + 0.001 * index + 0.005 for index in range(1, 41)])
    v4 = np.array([0.10 + 0.001 * index for index in range(1, 41)])
    assert row["v3_median_v_per_m"] == pytest.approx(np.median(v3))
    assert row["v4_median_v_per_m"] == pytest.approx(np.median(v4))
    assert row["probability_v3_greater_than_v4"] == pytest.approx(
        987.5 / 1600.0
    )
    assert row["v3_median_location_in_v4_observed_range"] == "within"
    assert row["v3_median_percentile_in_v4"] == pytest.approx(62.5)


def test_input_validation_rejects_version_cohort_mismatch(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    v3_left = pd.read_csv(paths["v3_left_hippocampus_csv"])
    v3_left.loc[v3_left["subject"] == "sub-CC000009", "subject"] = "sub-CC999999"
    v3_left.to_csv(paths["v3_left_hippocampus_csv"], index=False)

    with pytest.raises(RuntimeError, match="subject sets differ"):
        comparison.load_inputs(**paths)


def test_input_validation_rejects_wrong_roi_provenance(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    v4_right = pd.read_csv(paths["v4_right_m1_csv"])
    v4_right["roi"] = "Right_Precentral_Gyrus"
    v4_right.to_csv(paths["v4_right_m1_csv"], index=False)

    with pytest.raises(RuntimeError, match="unexpected ROI values"):
        comparison.load_inputs(**paths)


def test_input_validation_rejects_negative_field_magnitude(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    v3_right = pd.read_csv(paths["v3_right_m1_csv"])
    v3_right.loc[0, "roi_median_v_per_m"] = -0.01
    v3_right.to_csv(paths["v3_right_m1_csv"], index=False)

    with pytest.raises(RuntimeError, match="negative scalar ROI metrics"):
        comparison.load_inputs(**paths)


def test_exact_sign_flip_uses_ten_participants_as_inferential_units() -> None:
    result = comparison.exact_paired_sign_flip(np.ones(10, dtype=float))

    assert result["participants"] == 10
    assert result["permutations"] == 1024
    assert result["two_sided_p"] == pytest.approx(2 / 1024)


def test_preflight_is_read_only_and_reports_full_scope(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    out_dir = tmp_path / "must_not_exist"

    summary = comparison.preflight(_args(paths, out_dir))

    assert summary["status"] == "ready"
    assert summary["input_rows"] == 3200
    assert summary["selected_remesh_rows"] == 1600
    assert summary["participants"] == 10
    assert summary["targets"] == 2
    assert not out_dir.exists()


def test_run_writes_figures_tables_report_and_hashed_manifest(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    out_dir = tmp_path / "analysis"

    result = comparison.run(_args(paths, out_dir))

    assert result["status"] == "complete"
    assert result["input_rows"] == 3200
    assert result["selected_remesh_rows"] == 1600
    assert result["run_percentile_rows"] == 2400
    assert result["participant_target_primary_rows"] == 20
    assert result["participant_inference_rows"] == 6
    assert result["inferential_unit"] == "participant"
    assert result["technical_repeats_are_inferential_units"] is False

    required = {
        "v3_run_percentiles_in_v4.csv",
        "participant_metric_summary.csv",
        "participant_target_primary_summary.csv",
        "participant_level_inference.csv",
        "left_hippocampus_v4_distribution_with_v3_overlay.png",
        "left_hippocampus_v4_distribution_with_v3_overlay.pdf",
        "left_hippocampus_v3_percentile_location.png",
        "left_hippocampus_v3_percentile_location.pdf",
        "right_m1_v4_distribution_with_v3_overlay.png",
        "right_m1_v4_distribution_with_v3_overlay.pdf",
        "right_m1_v3_percentile_location.png",
        "right_m1_v3_percentile_location.pdf",
        "figure_captions.md",
        "analysis_report.md",
        "SHA256SUMS",
        "analysis_manifest.json",
    }
    assert required.issubset({path.name for path in out_dir.iterdir()})

    manifest = json.loads((out_dir / "analysis_manifest.json").read_text())
    assert manifest == result
    assert manifest["inputs"]["v3_left_hippocampus_csv"]["sha256"] == _sha256(
        paths["v3_left_hippocampus_csv"]
    )
    for artifact in manifest["artifacts"]:
        path = out_dir / artifact["path"]
        assert path.is_file()
        assert artifact["size_bytes"] == path.stat().st_size
        assert artifact["sha256"] == _sha256(path)

    report = (out_dir / "analysis_report.md").read_text()
    assert "technical replicates, not inferential units" in report
    assert "aligned optimizer-matched spherical ROI constructions" in report


def test_hpc_launcher_reports_full_read_only_scope_without_submission() -> None:
    launcher = (
        CURRENT_REPAIR_ROOT
        / "hpc_scripts"
        / "submit_compare_simnibs_version_remesh_distributions.sh"
    )
    slurm = (
        CURRENT_REPAIR_ROOT
        / "hpc_scripts"
        / "compare_simnibs_version_remesh_distributions.slurm"
    )

    completed = subprocess.run(
        ["bash", str(launcher), "scope"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "selected remesh metric rows: 1,600" in completed.stdout
    assert "inferential units: 10 participants" in completed.stdout
    assert "meshing/FEM tasks: 0" in completed.stdout
    assert "source policy: read-only" in completed.stdout
    assert "sbatch" not in completed.stdout.lower()
    slurm_text = slurm.read_text()
    assert "compare_simnibs_version_remesh_distributions.py" in slurm_text
    assert "analysis_manifest.json" in slurm_text
    assert "finite_support_audit" not in slurm_text


def test_hpc_preflight_validates_inputs_without_creating_output(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    analysis_root = tmp_path / "preflight_must_not_create_this"
    launcher = (
        CURRENT_REPAIR_ROOT
        / "hpc_scripts"
        / "submit_compare_simnibs_version_remesh_distributions.sh"
    )
    env = os.environ.copy()
    env.update(
        {
            "PIPELINE_DIR": str(CURRENT_REPAIR_ROOT),
            "ANALYSIS_PYTHON": sys.executable,
            "ANALYSIS_ROOT": str(analysis_root),
            "V3_LEFT_HIPPOCAMPUS_CSV": str(paths["v3_left_hippocampus_csv"]),
            "V3_RIGHT_M1_CSV": str(paths["v3_right_m1_csv"]),
            "V4_LEFT_HIPPOCAMPUS_CSV": str(paths["v4_left_hippocampus_csv"]),
            "V4_RIGHT_M1_CSV": str(paths["v4_right_m1_csv"]),
        }
    )

    completed = subprocess.run(
        ["bash", str(launcher), "preflight"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert '"status": "ready"' in completed.stdout
    assert "no job was submitted and no output was created" in completed.stdout
    assert not analysis_root.exists()
