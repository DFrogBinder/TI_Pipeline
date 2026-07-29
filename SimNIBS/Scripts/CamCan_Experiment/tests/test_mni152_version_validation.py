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
            "head_model_manifest_sha256": "manifest-hash",
            "mni_mesh_sha256": "mesh-hash",
            "reference_t1_sha256": "t1-hash",
            "eeg_cap_sha256": "cap-hash",
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
        expected_head_model_manifest_sha256="manifest-hash",
        expected_mni_mesh_sha256="mesh-hash",
        expected_reference_t1_sha256="t1-hash",
        expected_eeg_cap_sha256="cap-hash",
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
    assert (
        'MNI_INPUT_ROOT="${MNI_INPUT_ROOT:-/mnt/parscratch/users/cop23bi/'
        'MNI152_SimNIBS401_inputs/m2m_MNI152}"'
    ) in submit
    assert "${MNI45_BASELINE_PARENT}/m2m_MNI152" not in submit
    assert "mni152_head_model_manifest.sha256" in submit
    assert "Upload the complete local directory before running preflight" in submit
    assert "Existing 4.5.0 baseline is incomplete" not in submit
    assert "MNI45_BASELINE_PARENT=" not in submit
    assert 'module load SimNIBS/4.0.1-foss-2023a' in worker
    assert 'session.eeg_cap = str(eeg_cap_path)' in (
        root / "simulation" / "TI_runner_MNI152.py"
    ).read_text()
    assert '--expected-simnibs-version "4.0.1"' in worker
    assert "--output-subject" in worker
    assert "#SBATCH --requeue" not in worker
    collector = (
        root
        / "HPC_scripts"
        / "mni152_simnibs401_validation_collect.slurm"
    ).read_text()
    assert "mni152_simnibs_4p0p1_validated_results.tar.gz" in collector
    assert "compare_mni152_baseline_versions.py" not in collector
    assert "simnibs-4p5-parent" not in collector


def test_mni152_head_model_manifest_covers_complete_local_bundle():
    root = Path(__file__).resolve().parents[1]
    manifest = (
        root / "simulation" / "mni152_head_model_manifest.sha256"
    ).read_text(encoding="utf-8").splitlines()

    assert len(manifest) == 17
    assert any(line.endswith("  MNI152.msh") for line in manifest)
    assert any(line.endswith("  T1.nii.gz") for line in manifest)
    assert any(
        line.endswith("  eeg_positions/EEG10-10_UI_Jurak_2007.csv")
        for line in manifest
    )
    assert any(line.endswith("  final_tissues.nii.gz") for line in manifest)
