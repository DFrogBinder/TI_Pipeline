import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from post.compare_mni152_baseline_versions import (
    compare_volumes,
    numeric_metric_deltas,
)
from simulation.target_montages import resolve_montage_preset
from simulation.validate_mni152_baseline import validate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_random_nifti(path: Path, *, seed: int) -> None:
    rng = np.random.default_rng(seed)
    data = rng.random((40, 40, 40), dtype=np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def test_voxelwise_version_comparison_reports_exact_difference(tmp_path):
    old_path = tmp_path / "old.nii.gz"
    new_path = tmp_path / "new.nii.gz"
    old = np.arange(27, dtype=np.float32).reshape((3, 3, 3)) / 100
    new = old + 0.01
    old[0, 0, 0] = np.nan
    new[0, 0, 0] = np.nan
    nib.save(nib.Nifti1Image(old, np.eye(4)), str(old_path))
    nib.save(nib.Nifti1Image(new, np.eye(4)), str(new_path))

    result = compare_volumes(old_path, new_path)

    assert result["shared_finite_voxels"] == 26
    assert np.isclose(result["signed_mean_difference_v_per_m"], 0.01)
    assert np.isclose(result["mean_absolute_difference_v_per_m"], 0.01)
    assert np.isclose(result["root_mean_square_difference_v_per_m"], 0.01)
    assert np.isclose(result["maximum_absolute_difference_v_per_m"], 0.01)
    assert np.isclose(result["pearson_r"], 1.0)


def test_numeric_metric_deltas_use_4p0p1_minus_4p5p0_direction():
    rows = numeric_metric_deltas(
        "Left_M1",
        {"subject": "MNI152", "roi": "Left_M1", "roi_mean_v_per_m": 0.2},
        {"subject": "MNI152", "roi": "Left_M1", "roi_mean_v_per_m": 0.21},
    )

    assert len(rows) == 1
    assert rows[0]["roi"] == "Left_M1"
    assert rows[0]["metric"] == "roi_mean_v_per_m"
    assert rows[0]["simnibs_4p5p0"] == 0.2
    assert rows[0]["simnibs_4p0p1"] == 0.21
    assert np.isclose(rows[0]["delta_4p0p1_minus_4p5p0"], 0.01)
    assert np.isclose(rows[0]["percent_delta_relative_to_4p5p0"], 5.0)


def test_mni_baseline_validator_checks_version_hashes_and_montage(tmp_path):
    output_subject = "MNI152-left-m1"
    simnibs_root = tmp_path / output_subject / "anat" / "SimNIBS"
    output_dir = simnibs_root / "Output" / "MNI152"
    (output_dir / "Volume_Base").mkdir(parents=True)
    (output_dir / "Volume_Labels").mkdir(parents=True)

    reference_t1 = tmp_path / "T1.nii.gz"
    ti_brain = simnibs_root / "ti_brain_only.nii.gz"
    ti_volume = output_dir / "Volume_Base" / "TI_Volumetric_Base_TImax.nii.gz"
    ti_labels = output_dir / "Volume_Labels" / "TI_Volumetric_Labels.nii.gz"
    _write_random_nifti(reference_t1, seed=1)
    _write_random_nifti(ti_brain, seed=2)
    _write_random_nifti(ti_volume, seed=3)
    _write_random_nifti(ti_labels, seed=4)
    for name in (
        "TI.msh",
        "MNI152_TDCS_1_scalar.msh",
        "MNI152_TDCS_2_scalar.msh",
    ):
        (output_dir / name).write_bytes(b"0" * 1_000_000)

    montage = resolve_montage_preset("left-m1")
    provenance = {
        "status": "complete",
        "output_subject": output_subject,
        "preset": "left-m1",
        "software": {"simnibs_version": "4.0.1"},
        "inputs": {
            "mni_mesh_sha256": "mesh-hash",
            "reference_t1_sha256": "t1-hash",
            "targets_csv_sha256": "targets-hash",
        },
        "stimulation": {
            "pair1": {
                "anode": montage.pair1.anode,
                "cathode": montage.pair1.cathode,
                "current_a": montage.pair1.current_amp,
            },
            "pair2": {
                "anode": montage.pair2.anode,
                "cathode": montage.pair2.cathode,
                "current_a": montage.pair2.current_amp,
            },
        },
        "outputs": {
            "ti_brain_only": {
                "sha256": _sha256(ti_brain),
            }
        },
        "padding": "x" * 600,
    }
    (simnibs_root / "mni_baseline_provenance.json").write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )
    summary = tmp_path / "validation_records" / f"{output_subject}.json"
    args = argparse.Namespace(
        output_parent=tmp_path,
        output_subject=output_subject,
        preset="left-m1",
        reference_t1=reference_t1,
        expected_simnibs_version="4.0.1",
        expected_mni_mesh_sha256="mesh-hash",
        expected_reference_t1_sha256="t1-hash",
        expected_targets_sha256="targets-hash",
        summary=summary,
    )

    result = validate(args)

    assert result["status"] == "complete"
    assert result["simnibs_version"] == "4.0.1"
    assert result["nifti"]["finite_voxels"] == 40**3
    assert summary.is_file()


def test_hpc_launcher_is_four_task_isolated_simnibs401_validation():
    root = Path(__file__).resolve().parents[1]
    submit = (
        root
        / "HPC_scripts"
        / "submit_mni152_simnibs401_validation.sh"
    ).read_text()
    worker = (
        root
        / "HPC_scripts"
        / "mni152_simnibs401_validation_array.slurm"
    ).read_text()

    assert '--array="0-3%${MAX_CONCURRENT}"' in submit
    assert 'MNI401_OUTPUT_PARENT="${MNI401_OUTPUT_PARENT:-' in submit
    assert '[ "${MNI401_OUTPUT_PARENT}" = "${MNI45_BASELINE_PARENT}" ]' in submit
    assert 'module load SimNIBS/4.0.1-foss-2023a' in worker
    assert '--expected-simnibs-version "4.0.1"' in worker
    assert "--output-subject" in worker
    assert "#SBATCH --requeue" not in worker
