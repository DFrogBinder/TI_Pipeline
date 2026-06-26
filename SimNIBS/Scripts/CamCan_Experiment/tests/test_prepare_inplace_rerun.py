from pathlib import Path

from simulation.prepare_inplace_rerun import (
    discover_repeat_datasets,
    read_tsv,
    run_preflight,
    validate_manifest,
)


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _seed_subject(dataset: Path, subject: str, *, mesh: bool = True) -> None:
    anat = dataset / subject / "anat"
    _write(anat / f"{subject}_T1w.nii")
    _write(anat / f"{subject}_T2w.nii")
    _write(anat / f"{subject}_T1w_ras_1mm_T1andT2_masks.nii")
    if mesh:
        _write(anat / f"m2m_{subject}" / f"{subject}.msh", "mesh")


def test_discover_repeat_datasets_sorts_numeric_repeat_ids(tmp_path):
    for name in ("Left_M1_Data_10", "Left_M1_Data_01", "notes"):
        (tmp_path / name).mkdir()

    datasets = discover_repeat_datasets(tmp_path)

    assert [item.name for item in datasets] == [
        "Left_M1_Data_01",
        "Left_M1_Data_10",
    ]


def test_preflight_dry_run_writes_ready_and_blocked_rows_without_deleting(tmp_path):
    run_01 = tmp_path / "Left_M1_Data_01"
    run_02 = tmp_path / "Left_M1_Data_02"
    _seed_subject(run_01, "sub-01")
    _seed_subject(run_02, "sub-01", mesh=False)
    _write(run_01 / "sub-01" / "anat" / "SimNIBS" / "old.txt")

    manifest = tmp_path / "manifest.tsv"
    cleanup = tmp_path / "cleanup.tsv"
    result = run_preflight(
        tmp_path,
        manifest=manifest,
        cleanup_manifest=cleanup,
        apply=False,
    )

    rows = read_tsv(manifest)
    assert result["ready_tasks"] == 1
    assert result["blocked_tasks"] == 1
    assert [row["status"] for row in rows] == ["ready", "blocked"]
    assert (run_01 / "sub-01" / "anat" / "SimNIBS" / "old.txt").is_file()
    assert any(row["action"] == "would_delete" for row in read_tsv(cleanup))


def test_preflight_apply_deletes_generated_outputs_and_preserves_inputs(tmp_path):
    run_01 = tmp_path / "Left_M1_Data_01"
    _seed_subject(run_01, "sub-01")
    anat = run_01 / "sub-01" / "anat"
    _write(anat / "SimNIBS" / "old.txt")
    _write(anat / "post" / "old.txt")
    _write(anat / "skin_mask.nii.gz")

    run_preflight(
        tmp_path,
        manifest=tmp_path / "manifest.tsv",
        cleanup_manifest=tmp_path / "cleanup.tsv",
        apply=True,
    )

    assert not (anat / "SimNIBS").exists()
    assert not (anat / "post").exists()
    assert not (anat / "skin_mask.nii.gz").exists()
    assert (anat / "m2m_sub-01" / "sub-01.msh").is_file()
    assert (anat / "sub-01_T1w.nii").is_file()
    assert (anat / "sub-01_T2w.nii").is_file()
    assert (anat / "sub-01_T1w_ras_1mm_T1andT2_masks.nii").is_file()


def test_validate_manifest_reports_complete_rows(tmp_path):
    run_01 = tmp_path / "Left_M1_Data_01"
    _seed_subject(run_01, "sub-01")
    output = run_01 / "sub-01" / "anat" / "SimNIBS"
    _write(output / "Output" / "sub-01" / "TI.msh")
    _write(output / "Output" / "sub-01" / "Volume_Base" / "TI_Volumetric_Base.nii.gz")
    _write(output / "Output" / "sub-01" / "Volume_Labels" / "TI_Volumetric_Labels.nii.gz")
    _write(output / "ti_brain_only.nii.gz")
    manifest = tmp_path / "manifest.tsv"
    validation = tmp_path / "validation.tsv"
    run_preflight(
        tmp_path,
        manifest=manifest,
        cleanup_manifest=tmp_path / "cleanup.tsv",
        apply=False,
    )

    result = validate_manifest(manifest, summary_path=validation, check_nifti=False)

    assert result["complete_tasks"] == 1
    assert result["incomplete_tasks"] == 0
    assert read_tsv(validation)[0]["status"] == "complete"
