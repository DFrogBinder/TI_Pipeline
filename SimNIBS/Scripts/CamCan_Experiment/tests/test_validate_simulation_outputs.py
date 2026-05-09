from simulation.validate_simulation_outputs import validate_subject_outputs


def _write_file(path, payload=b"ok"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_validate_subject_outputs_fails_when_outputs_missing(tmp_path):
    result = validate_subject_outputs(
        tmp_path,
        "sub-CC000001",
        check_nifti=False,
    )

    assert result.ok is False
    assert {check.name for check in result.checks if not check.ok} == {
        "ti_mesh",
        "ti_volume",
        "ti_labels",
        "ti_brain_only",
    }


def test_validate_subject_outputs_passes_when_required_outputs_exist(tmp_path):
    subject = "sub-CC000001"
    output_dir = tmp_path / subject / "anat" / "SimNIBS" / "Output" / subject
    _write_file(output_dir / "TI.msh")
    _write_file(output_dir / "Volume_Base" / "TI_Volumetric_Base.nii.gz")
    _write_file(output_dir / "Volume_Labels" / "TI_Volumetric_Labels.nii.gz")
    _write_file(tmp_path / subject / "anat" / "SimNIBS" / "ti_brain_only.nii.gz")

    result = validate_subject_outputs(
        tmp_path,
        subject,
        check_nifti=False,
    )

    assert result.ok is True
    assert all(check.ok for check in result.checks)
