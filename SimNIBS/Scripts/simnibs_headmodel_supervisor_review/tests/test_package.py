from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Callable, Optional

import pytest

from simnibs_headmodel_supervisor_review.package import (
    ReviewPackageError,
    audit_subject,
    build_package,
    load_selection,
    preflight,
    select_subject,
    validate_package,
)


def test_label_exporter_alias_imports_with_python39_union_semantics() -> None:
    package_source = Path(
        __import__(
            "simnibs_headmodel_supervisor_review.package", fromlist=["__file__"]
        ).__file__
    ).read_text(encoding="utf-8")
    tree = ast.parse(package_source)
    alias = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "LabelExporter"
            for target in node.targets
        )
    )

    class Python39Type(type):
        def __or__(cls, other):
            raise TypeError("unsupported operand type(s) for |")

    class LegacyPath(metaclass=Python39Type):
        pass

    namespace = {
        "Callable": Callable,
        "Optional": Optional,
        "Path": LegacyPath,
    }
    exec(
        compile(ast.Module(body=[alias], type_ignores=[]), "package.py", "exec"),
        namespace,
    )

    assert "LabelExporter" in namespace


def test_audit_resolves_exact_eight_artifacts_from_distinct_provenance_branches(
    review_layout: tuple[object, tuple[str, ...]],
) -> None:
    paths, subjects = review_layout

    sources = audit_subject(paths, subjects[0])

    assert sources.subject == subjects[0]
    assert [artifact.key for artifact in sources.artifacts] == [
        "original_charm_labels",
        "corrected_charm_labels",
        "harmonized_v3_labels",
        "original_headreco_labels",
        "subject_atlas",
        "v3_original_headreco_mesh",
        "v3_harmonized_mesh",
        "v4_corrected_charm_median_mesh",
    ]
    assert sources.artifacts[3].mode == "mesh_tags_to_nifti"
    assert sources.artifacts[3].reference_path.name == "T1fs_conform.nii.gz"
    assert sources.artifacts[5].source_path != sources.artifacts[6].source_path
    assert sources.artifacts[7].metadata["selected_repeat_tag"] == "repeat_001"


def test_audit_rejects_receipt_path_outside_protected_root(
    review_layout: tuple[object, tuple[str, ...]], tmp_path: Path
) -> None:
    paths, subjects = review_layout
    receipt = paths.corrected_v4_scaffold_root / "results" / f"{subjects[0]}.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    outside = tmp_path / "outside-label.nii.gz"
    outside.write_bytes(b"outside")
    payload["installed_label"] = str(outside)
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReviewPackageError, match="outside protected root"):
        audit_subject(paths, subjects[0])


def test_preflight_fails_if_any_of_nine_subjects_is_incomplete(
    review_layout: tuple[object, tuple[str, ...]],
) -> None:
    paths, subjects = review_layout
    missing = paths.atlas_root / f"{subjects[-1]}.nii.gz"
    missing.unlink()

    with pytest.raises(ReviewPackageError, match=subjects[-1]):
        preflight(paths)

    assert not paths.output_root.exists()


def test_selection_is_random_once_then_reused_immutably(
    review_layout: tuple[object, tuple[str, ...]],
) -> None:
    paths, subjects = review_layout
    calls = 0

    def choose_last(candidates: tuple[str, ...]) -> str:
        nonlocal calls
        calls += 1
        assert candidates == subjects
        return candidates[-1]

    first = select_subject(paths, chooser=choose_last)
    second = select_subject(
        paths,
        chooser=lambda _: (_ for _ in ()).throw(AssertionError("must not redraw")),
    )

    assert first["selected_subject"] == subjects[-1]
    assert second == first
    assert calls == 1
    assert first["candidate_subjects"] == list(subjects)
    assert first["candidate_count"] == 9
    assert first["artifact_count"] == 8
    assert len(first["audit_sha256"]) == 64
    assert load_selection(paths)["selected_subject"] == subjects[-1]


def test_selection_rejects_stale_source_audit(
    review_layout: tuple[object, tuple[str, ...]],
) -> None:
    paths, subjects = review_layout
    select_subject(paths, chooser=lambda candidates: candidates[0])
    atlas = paths.atlas_root / f"{subjects[0]}.nii.gz"
    atlas.write_bytes(b"changed-after-selection")

    with pytest.raises(ReviewPackageError, match="selection receipt is stale"):
        load_selection(paths, verify_current=True)


def test_build_materializes_exact_eight_artifacts_manifest_checksums_and_archive(
    review_layout: tuple[object, tuple[str, ...]],
) -> None:
    paths, subjects = review_layout
    selected = subjects[0]
    select_subject(paths, chooser=lambda candidates: candidates[0])
    source_before = {
        path: path.read_bytes()
        for root in paths.protected_roots
        for path in root.rglob("*")
        if path.is_file()
    }

    def exporter(mesh: Path, reference: Path, destination: Path) -> Path:
        destination.write_bytes(b"derived:" + mesh.read_bytes() + b":" + reference.read_bytes())
        return destination

    result = build_package(paths, exporter=exporter)

    assert result["action"] == "built"
    assert result["subject"] == selected
    expected_names = {
        "01_original_charm_tissue_labels.nii.gz",
        "02_corrected_charm_tissue_labels.nii.gz",
        "03_harmonized_v3_tissue_labels.nii.gz",
        "04_original_headreco_mesh_tissue_labels.nii.gz",
        "05_subject_atlas.nii.gz",
        "06_v3_original_headreco_mesh.msh",
        "07_v3_harmonized_mesh.msh",
        "08_v4_corrected_charm_median_mesh.msh",
    }
    manifest = json.loads((paths.package_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["subject"] == selected
    assert manifest["scientific_artifact_count"] == 8
    assert {row["destination"] for row in manifest["artifacts"]} == expected_names
    for row in manifest["artifacts"]:
        assert "provenance_paths" not in row
        for receipt in row["provenance"]:
            assert Path(receipt["path"]).is_file()
            assert len(receipt["sha256"]) == 64
    assert {path.name for path in paths.package_dir.iterdir()} == expected_names | {
        "README.md",
        "manifest.json",
        "checksums.sha256",
    }
    derived = paths.package_dir / "04_original_headreco_mesh_tissue_labels.nii.gz"
    assert derived.read_bytes().startswith(b"derived:legacy-mesh-")
    assert paths.archive_path.is_file()
    assert paths.completion_receipt.is_file()
    assert validate_package(paths)["status"] == "complete"
    assert all(path.read_bytes() == content for path, content in source_before.items())


def test_failed_export_never_promotes_package(
    review_layout: tuple[object, tuple[str, ...]],
) -> None:
    paths, _ = review_layout
    select_subject(paths, chooser=lambda candidates: candidates[0])

    def fail_export(mesh: Path, reference: Path, destination: Path) -> Path:
        raise RuntimeError("export failed")

    with pytest.raises(RuntimeError, match="export failed"):
        build_package(paths, exporter=fail_export)

    assert not paths.package_dir.exists()
    assert not paths.archive_path.exists()
    assert not paths.completion_receipt.exists()


def test_completed_package_is_reused_but_tampering_fails_validation(
    review_layout: tuple[object, tuple[str, ...]],
) -> None:
    paths, _ = review_layout
    select_subject(paths, chooser=lambda candidates: candidates[0])

    def exporter(mesh: Path, reference: Path, destination: Path) -> Path:
        destination.write_bytes(b"derived-labels")
        return destination

    build_package(paths, exporter=exporter)
    reused = build_package(
        paths,
        exporter=lambda *_: (_ for _ in ()).throw(AssertionError("must not rebuild")),
    )
    assert reused["action"] == "reused"

    (paths.package_dir / "07_v3_harmonized_mesh.msh").write_bytes(b"tampered")
    with pytest.raises(ReviewPackageError, match="checksum mismatch"):
        validate_package(paths)


def test_unowned_partial_output_is_not_overwritten(
    review_layout: tuple[object, tuple[str, ...]],
) -> None:
    paths, _ = review_layout
    select_subject(paths, chooser=lambda candidates: candidates[0])
    paths.package_dir.mkdir()
    (paths.package_dir / "unknown.txt").write_text("do not replace", encoding="utf-8")

    with pytest.raises(ReviewPackageError, match="partial or unowned"):
        build_package(paths, exporter=lambda *_: None)

    assert (paths.package_dir / "unknown.txt").read_text(encoding="utf-8") == "do not replace"


def test_validation_rejects_wrong_subject_and_artifact_count_in_manifest(
    review_layout: tuple[object, tuple[str, ...]],
) -> None:
    paths, _ = review_layout
    select_subject(paths, chooser=lambda candidates: candidates[0])

    def exporter(mesh: Path, reference: Path, destination: Path) -> Path:
        destination.write_bytes(b"derived-labels")
        return destination

    build_package(paths, exporter=exporter)
    manifest_path = paths.package_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["subject"] = "sub-WRONG"
    manifest["scientific_artifact_count"] = 7
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ReviewPackageError, match="invalid package binding"):
        validate_package(paths)
