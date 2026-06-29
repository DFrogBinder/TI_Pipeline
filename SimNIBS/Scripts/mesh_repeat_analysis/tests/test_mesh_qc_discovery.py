from pathlib import Path

from mesh_repeat_analysis.post.mesh_qc.discovery import discover_meshes, infer_record


def test_infer_record_extracts_roi_subject_and_repeat_from_path(tmp_path):
    path = (
        tmp_path
        / "Left_Hippocampus"
        / "sub-CC110056"
        / "repeat_07"
        / "SimNIBS"
        / "head.msh"
    )
    path.parent.mkdir(parents=True)
    path.write_text("$MeshFormat\n", encoding="utf-8")

    record = infer_record(path, root=tmp_path)

    assert record.roi == "Left_Hippocampus"
    assert record.subject == "sub-CC110056"
    assert record.repeat == "repeat_07"


def test_discover_meshes_returns_sorted_records(tmp_path):
    second = tmp_path / "M1" / "sub-CC2" / "repeat_02" / "mesh.msh"
    first = tmp_path / "M1" / "sub-CC1" / "repeat_01" / "mesh.msh"
    second.parent.mkdir(parents=True)
    first.parent.mkdir(parents=True)
    second.write_text("$MeshFormat\n", encoding="utf-8")
    first.write_text("$MeshFormat\n", encoding="utf-8")

    records = discover_meshes(tmp_path, mesh_glob="*.msh")

    assert [Path(r.path).name for r in records] == ["mesh.msh", "mesh.msh"]
    assert [r.subject for r in records] == ["sub-CC1", "sub-CC2"]
    assert [r.repeat for r in records] == ["repeat_01", "repeat_02"]


def test_discover_meshes_defaults_to_m2m_directories_only(tmp_path):
    wanted = tmp_path / "M1" / "sub-CC1" / "repeat_01" / "m2m_sub-CC1" / "head.msh"
    extra = tmp_path / "M1" / "sub-CC1" / "repeat_01" / "SimNIBS" / "field.msh"
    wanted.parent.mkdir(parents=True)
    extra.parent.mkdir(parents=True)
    wanted.write_text("$MeshFormat\n", encoding="utf-8")
    extra.write_text("$MeshFormat\n", encoding="utf-8")

    default_records = discover_meshes(tmp_path)
    explicit_records = discover_meshes(tmp_path, mesh_glob="*.msh")

    assert [r.path for r in default_records] == [wanted]
    assert [r.path for r in explicit_records] == [extra, wanted]
