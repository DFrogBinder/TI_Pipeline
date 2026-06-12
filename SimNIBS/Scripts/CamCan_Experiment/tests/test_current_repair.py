import sys
from pathlib import Path

import pytest

CAMCAN_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = CAMCAN_ROOT.parents[0]
for path in (CAMCAN_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from simulation.repair_current_bug import (  # noqa: E402
    build_camcan_repair_tasks,
    camcan_scale_factor,
    should_skip_unaffected_preset,
)
from simulation.target_montages import PairSpec, resolve_montage_preset  # noqa: E402
from ti_current_repair.core import RepairKey, validate_same_repair_key  # noqa: E402


def test_camcan_scale_factor_uses_pair2_over_pair1_current_from_targets_csv():
    montage = resolve_montage_preset("left-hippocampus")

    assert camcan_scale_factor(montage) == pytest.approx(
        montage.pair2.current_a / montage.pair1.current_a
    )


def test_camcan_unaffected_presets_are_skipped_by_default():
    pair1 = PairSpec("A", "B", 0.002)
    pair2 = PairSpec("C", "D", 0.002)

    assert should_skip_unaffected_preset(pair1, pair2)
    assert not should_skip_unaffected_preset(pair1, pair2, include_unaffected=True)


def test_camcan_plan_preserves_dataset_subject_layout(tmp_path):
    dataset = tmp_path / "original" / "Left_Hippocampus_Data_01"
    subject = dataset / "sub-01"
    subject.mkdir(parents=True)

    tasks = build_camcan_repair_tasks(
        original_root=tmp_path / "original",
        output_root=tmp_path / "scaled",
        montage_preset="left-hippocampus",
        datasets=["Left_Hippocampus_Data_01"],
        subjects=["sub-01"],
    )

    assert len(tasks) == 1
    task = tasks[0]
    assert task.key == RepairKey(
        experiment="camcan",
        subject="sub-01",
        dataset="Left_Hippocampus_Data_01",
    )
    assert task.original_subject_root == subject
    assert task.output_anat_dir == (
        tmp_path / "scaled" / "Left_Hippocampus_Data_01" / "sub-01" / "anat"
    )
    assert task.output_dir == task.output_anat_dir / "SimNIBS" / "Output" / "sub-01"


def test_camcan_comparison_rejects_repeatability_keys():
    camcan_key = RepairKey(
        experiment="camcan",
        subject="sub-01",
        dataset="Left_Hippocampus_Data_01",
    )
    repeatability_key = RepairKey(
        experiment="repeatability",
        subject="sub-01",
        condition="remesh",
        repeat_tag="repeat_001",
    )

    with pytest.raises(ValueError, match="camcan"):
        validate_same_repair_key(camcan_key, repeatability_key, experiment="camcan")
