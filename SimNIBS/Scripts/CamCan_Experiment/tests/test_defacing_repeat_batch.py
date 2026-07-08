from pathlib import Path

import nibabel as nib
import numpy as np

from defacing_experiment.prepare_defacing_repeat_batch import (
    DEFAULT_REPEATS,
    apply_keep_mask,
    build_repeat_batch_specs,
    resample_keep_mask_to_target,
    stage_repeat_batches,
)


def test_build_repeat_batch_specs_creates_four_expected_arms(tmp_path):
    specs = build_repeat_batch_specs(
        out_root=tmp_path,
        subject="sub-CCMe",
        repeats=2,
        targets=("left-hippocampus", "left-m1"),
    )

    keys = [(spec.target, spec.condition) for spec in specs]
    assert keys == [
        ("left-hippocampus", "intact"),
        ("left-hippocampus", "defaced"),
        ("left-m1", "intact"),
        ("left-m1", "defaced"),
    ]

    assert specs[0].parent_root == tmp_path / "Left_Hippocampus_Intact"
    assert specs[1].parent_root == tmp_path / "Left_Hippocampus_Defaced"
    assert specs[2].parent_root == tmp_path / "Left_M1_Intact"
    assert specs[3].parent_root == tmp_path / "Left_M1_Defaced"
    assert specs[0].repeat_ids == ("01", "02")


def test_defacing_experiment_default_repeat_count_is_40_per_arm(tmp_path):
    specs = build_repeat_batch_specs(
        out_root=tmp_path,
        subject="sub-CCMe",
        repeats=DEFAULT_REPEATS,
    )

    assert DEFAULT_REPEATS == 40
    assert len(specs) == 4
    for spec in specs:
        assert len(spec.repeat_ids) == 40
        assert spec.repeat_ids[0] == "01"
        assert spec.repeat_ids[-1] == "40"


def test_apply_keep_mask_zeroes_voxels_outside_mask():
    source = nib.Nifti1Image(
        np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        affine=np.eye(4),
    )
    keep_mask = nib.Nifti1Image(
        np.array(
            [
                [[1, 0], [1, 0]],
                [[1, 1], [0, 0]],
            ],
            dtype=np.uint8,
        ),
        affine=np.eye(4),
    )

    out = apply_keep_mask(source, keep_mask)
    out_data = np.asanyarray(out.dataobj)

    assert out_data[0, 0, 0] == 0
    assert out_data[0, 0, 1] == 0
    assert out_data[1, 1, 0] == 0
    assert out_data[1, 1, 1] == 0
    assert out_data[0, 1, 0] == source.get_fdata()[0, 1, 0]


def test_resample_keep_mask_to_target_returns_target_grid():
    keep_mask = nib.Nifti1Image(
        np.ones((2, 2, 2), dtype=np.uint8),
        affine=np.diag([1.0, 1.0, 1.0, 1.0]),
    )
    target = nib.Nifti1Image(
        np.zeros((4, 2, 2), dtype=np.float32),
        affine=np.diag([0.5, 1.0, 1.0, 1.0]),
    )

    out = resample_keep_mask_to_target(keep_mask, target)

    assert out.shape == target.shape
    assert np.allclose(out.affine, target.affine)


def test_stage_repeat_batches_writes_manifest_and_canonical_repeat_inputs(tmp_path):
    intact_t1 = tmp_path / "inputs" / "sub-CCMe_T1w.nii.gz"
    intact_t2 = tmp_path / "inputs" / "sub-CCMe_T2w.nii"
    defaced_t1 = tmp_path / "generated" / "sub-CCMe_desc-deface_T1w.nii.gz"
    defaced_t2 = tmp_path / "generated" / "sub-CCMe_desc-deface_T2w.nii"
    for path in (intact_t1, intact_t2, defaced_t1, defaced_t2):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")

    parent_roots = stage_repeat_batches(
        out_root=tmp_path / "experiment",
        subject="sub-CCMe",
        intact_t1=intact_t1,
        intact_t2=intact_t2,
        defaced_t1=defaced_t1,
        defaced_t2=defaced_t2,
        repeats=1,
        targets=("left-m1",),
        force=False,
    )

    assert parent_roots == (
        tmp_path / "experiment" / "Left_M1_Intact",
        tmp_path / "experiment" / "Left_M1_Defaced",
    )

    intact_repeat = parent_roots[0] / "Left_M1_Data_01" / "sub-CCMe" / "anat"
    defaced_repeat = parent_roots[1] / "Left_M1_Data_01" / "sub-CCMe" / "anat"
    assert (intact_repeat / "sub-CCMe_T1w.nii.gz").read_text(encoding="utf-8") == "sub-CCMe_T1w.nii.gz"
    assert (defaced_repeat / "sub-CCMe_T1w.nii.gz").read_text(encoding="utf-8") == "sub-CCMe_desc-deface_T1w.nii.gz"
    assert (parent_roots[0] / "slurm" / "manifest.tsv").is_file()
    assert (parent_roots[1] / "slurm" / "manifest.tsv").is_file()
