import csv
import subprocess
from pathlib import Path

from charm_only_remesh.workflow import (
    build_remesh_manifest,
    read_tsv,
    remove_installation_backups,
    remove_roast_segmentations,
    run_remesh_task,
    select_install_rows,
    validate_remesh_results,
)
from utils.camcan_dataset import sha256_file


def _write(path: Path, data: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
    return path


def _install_manifest(tmp_path: Path):
    roi_root = tmp_path / "Analised-Data"
    dataset = roi_root / "Left_M1_Runs" / "Left_M1_Data_01"
    subject = "sub-01"
    anat = dataset / subject / "anat"
    label = _write(
        anat / f"m2m_{subject}" / "label_prep" / "tissue_labeling_upsampled.nii.gz",
        "charm-only-label",
    )
    mesh = _write(anat / f"m2m_{subject}" / f"{subject}.msh", "old-mesh")
    source = _write(tmp_path / "all-seg-maps" / f"{subject}_tissue_labeling.nii.gz", "charm-only-label")
    backup = _write(
        tmp_path / "install-backups" / "run-01" / "Left_M1_Data_01" / f"{subject}.nii.gz",
        "old-roast-label",
    )
    install = tmp_path / "apply_manifest.tsv"
    with install.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "dataset",
                "subject",
                "source",
                "destination",
                "backup",
                "before_sha256",
                "installed_sha256",
                "status",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "Left_M1_Data_01",
                "subject": subject,
                "source": source,
                "destination": label,
                "backup": backup,
                "before_sha256": sha256_file(backup),
                "installed_sha256": sha256_file(label),
                "status": "installed",
            }
        )
    return roi_root, install, anat, label, mesh


def test_roi_scope_selects_exact_repeat_and_subject_set():
    rows = [
        {"dataset": "Left_Hippocampus_Data_01", "subject": "sub-01"},
        {"dataset": "Left_M1_Data_01", "subject": "sub-01"},
    ]

    selected = select_install_rows(
        rows,
        roi_prefix="Left_Hippocampus",
        expected_targets=1,
        expected_repeats=1,
        expected_subjects=1,
    )

    assert selected == [rows[0]]


def test_roi_scope_refuses_duplicate_subject_rows():
    row = {"dataset": "Left_Hippocampus_Data_01", "subject": "sub-01"}

    try:
        select_install_rows(
            [row, row.copy()],
            roi_prefix="Left_Hippocampus",
            expected_targets=2,
            expected_repeats=1,
            expected_subjects=1,
        )
    except ValueError as exc:
        assert "unique_subjects=1" in str(exc)
    else:
        raise AssertionError("duplicate ROI subject rows must block the scope")


def test_roi_scope_refuses_different_subject_sets_between_repeats():
    rows = [
        {"dataset": "Left_Hippocampus_Data_01", "subject": "sub-01"},
        {"dataset": "Left_Hippocampus_Data_02", "subject": "sub-02"},
    ]

    try:
        select_install_rows(
            rows,
            roi_prefix="Left_Hippocampus",
            expected_targets=2,
            expected_repeats=2,
            expected_subjects=1,
        )
    except ValueError as exc:
        assert "different subject set" in str(exc)
    else:
        raise AssertionError("different repeat subject sets must block the scope")


def test_installation_backup_removal_is_audited_then_deleted(tmp_path):
    _, install, _, _, _ = _install_manifest(tmp_path)
    backup_root = tmp_path / "install-backups" / "run-01"
    backup = next(backup_root.rglob("*.nii.gz"))

    audit = remove_installation_backups(
        install_manifest=install,
        backup_root=backup_root,
        report=tmp_path / "backup-audit.tsv",
        expected_backups=1,
    )
    assert audit["status"] == "audit"
    assert backup.is_file()

    result = remove_installation_backups(
        install_manifest=install,
        backup_root=backup_root,
        report=tmp_path / "backup-delete.tsv",
        apply_delete=True,
        external_backup_confirmed=True,
        expected_backups=1,
    )
    assert result["status"] == "complete"
    assert result["deleted"] == 1
    assert result["backup_root_removed"] is True
    assert not backup.exists()


def test_installation_backup_removal_refuses_duplicate_paths(tmp_path):
    _, install, _, _, _ = _install_manifest(tmp_path)
    lines = install.read_text(encoding="utf-8").splitlines(keepends=True)
    install.write_text("".join((lines[0], lines[1], lines[1])), encoding="utf-8")
    backup_root = tmp_path / "install-backups" / "run-01"
    backup = next(backup_root.rglob("*.nii.gz"))

    result = remove_installation_backups(
        install_manifest=install,
        backup_root=backup_root,
        report=tmp_path / "duplicate-backup.tsv",
        apply_delete=True,
        external_backup_confirmed=True,
        expected_backups=2,
    )

    assert result["status"] == "failed"
    assert result["deleted"] == 0
    assert backup.is_file()


def test_roast_removal_is_dry_run_by_default_and_preflight_refuses_roast(tmp_path):
    roi_root, install, anat, _, _ = _install_manifest(tmp_path)
    roast = _write(anat / "sub-01_T1w_ras_1mm_T1andT2_masks.nii", "roast")

    audit = remove_roast_segmentations(
        install_manifest=install,
        roi_root=roi_root,
        report=tmp_path / "removal.tsv",
        expected_targets=1,
    )
    assert audit["status"] == "audit"
    assert roast.is_file()

    try:
        remove_roast_segmentations(
            install_manifest=install,
            roi_root=roi_root,
            report=tmp_path / "unconfirmed.tsv",
            apply_delete=True,
            expected_targets=1,
        )
    except ValueError as exc:
        assert "confirm-external-backup" in str(exc)
    else:
        raise AssertionError("destructive ROAST removal must require confirmation")
    assert roast.is_file()

    blocked = build_remesh_manifest(
        install_manifest=install,
        roi_root=roi_root,
        manifest=tmp_path / "blocked.tsv",
        summary=tmp_path / "blocked.json",
        expected_targets=1,
    )
    assert blocked["status"] == "blocked"

    complete = remove_roast_segmentations(
        install_manifest=install,
        roi_root=roi_root,
        report=tmp_path / "applied.tsv",
        apply_delete=True,
        external_backup_confirmed=True,
        expected_targets=1,
    )
    assert complete["status"] == "complete"
    assert not roast.exists()
    assert read_tsv(tmp_path / "applied.tsv")[0]["action"] == "deleted"


def test_run_task_invokes_only_mesh_mode_and_removes_old_mesh(tmp_path, monkeypatch):
    roi_root, install, anat, label, mesh = _install_manifest(tmp_path)
    manifest = tmp_path / "remesh.tsv"
    ready = build_remesh_manifest(
        install_manifest=install,
        roi_root=roi_root,
        manifest=manifest,
        summary=tmp_path / "summary.json",
        expected_targets=1,
    )
    assert ready["status"] == "ready"

    calls = []

    def fake_run(command, *, cwd, capture_output, text):
        calls.append((command, cwd))
        _write(mesh, "new-charm-mesh")
        return subprocess.CompletedProcess(command, 0, "mesh complete\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_remesh_task(
        manifest=manifest,
        task_index=0,
        result_dir=tmp_path / "results",
        load_mesh=False,
    )

    assert calls == [(["charm", "sub-01", "--mesh"], anat)]
    assert result["status"] == "complete"
    assert result["label_sha256_before"] == sha256_file(label)
    assert result["removed_previous_mesh_path"] == str(mesh)
    assert result["removed_previous_mesh_sha256"]
    assert not (tmp_path / "results" / "_mesh_backups").exists()
    assert mesh.read_text(encoding="utf-8") == "new-charm-mesh"
    validation = validate_remesh_results(
        manifest=manifest,
        result_dir=tmp_path / "results",
        summary=tmp_path / "validation.tsv",
        load_mesh=False,
        task_indices=[0],
    )
    assert validation["status"] == "complete"


def test_run_task_refuses_mesh_path_outside_subject(tmp_path, monkeypatch):
    roi_root, install, _, _, mesh = _install_manifest(tmp_path)
    manifest = tmp_path / "remesh.tsv"
    build_remesh_manifest(
        install_manifest=install,
        roi_root=roi_root,
        manifest=manifest,
        summary=tmp_path / "summary.json",
        expected_targets=1,
    )
    outside_mesh = _write(tmp_path / "must-not-delete.msh", "unrelated")
    text = manifest.read_text(encoding="utf-8").replace(str(mesh), str(outside_mesh))
    manifest.write_text(text, encoding="utf-8")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("CHARM must not run for an unsafe manifest")
        ),
    )

    try:
        run_remesh_task(
            manifest=manifest,
            task_index=0,
            result_dir=tmp_path / "results",
            load_mesh=False,
        )
    except ValueError as exc:
        assert "outside the allowed subject locations" in str(exc)
    else:
        raise AssertionError("unsafe mesh path must fail the task")

    assert outside_mesh.read_text(encoding="utf-8") == "unrelated"
    assert mesh.read_text(encoding="utf-8") == "old-mesh"


def test_failed_remesh_leaves_obsolete_mesh_removed(tmp_path, monkeypatch):
    roi_root, install, _, _, mesh = _install_manifest(tmp_path)
    manifest = tmp_path / "remesh.tsv"
    build_remesh_manifest(
        install_manifest=install,
        roi_root=roi_root,
        manifest=manifest,
        summary=tmp_path / "summary.json",
        expected_targets=1,
    )

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1, "", "failed"),
    )
    try:
        run_remesh_task(
            manifest=manifest,
            task_index=0,
            result_dir=tmp_path / "results",
            load_mesh=False,
        )
    except subprocess.CalledProcessError:
        pass
    else:
        raise AssertionError("failed CHARM command should fail the task")

    assert not mesh.exists()


def test_remesh_restores_installed_label_if_charm_changes_it(tmp_path, monkeypatch):
    roi_root, install, _, label, mesh = _install_manifest(tmp_path)
    manifest = tmp_path / "remesh.tsv"
    build_remesh_manifest(
        install_manifest=install,
        roi_root=roi_root,
        manifest=manifest,
        summary=tmp_path / "summary.json",
        expected_targets=1,
    )

    def bad_run(command, **kwargs):
        label.write_text("unexpected-change", encoding="utf-8")
        mesh.write_text("new-mesh", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", bad_run)
    try:
        run_remesh_task(
            manifest=manifest,
            task_index=0,
            result_dir=tmp_path / "results",
            load_mesh=False,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("a changed installed label must fail the task")

    assert label.read_text(encoding="utf-8") == "charm-only-label"
    assert not mesh.exists()
