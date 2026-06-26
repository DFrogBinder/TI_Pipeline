from simulation.mesh_reuse import candidate_mesh_paths, resolve_existing_mesh


def test_candidate_mesh_paths_deduplicates_standard_sub_subject_layout(tmp_path):
    anat = tmp_path / "sub-CC000001" / "anat"

    paths = candidate_mesh_paths(anat, "sub-CC000001")

    assert paths == (
        anat / "m2m_sub-CC000001" / "sub-CC000001.msh",
    )


def test_candidate_mesh_paths_includes_legacy_sub_folder_for_unprefixed_subject(tmp_path):
    anat = tmp_path / "CC000001" / "anat"

    paths = candidate_mesh_paths(anat, "CC000001")

    assert paths == (
        anat / "m2m_CC000001" / "CC000001.msh",
        anat / "m2m_sub-CC000001" / "CC000001.msh",
        anat / "m2m_sub-CC000001" / "sub-CC000001.msh",
    )


def test_resolve_existing_mesh_uses_first_existing_candidate(tmp_path):
    anat = tmp_path / "CC000001" / "anat"
    fallback = anat / "m2m_sub-CC000001" / "sub-CC000001.msh"
    fallback.parent.mkdir(parents=True)
    fallback.write_text("mesh", encoding="utf-8")

    assert resolve_existing_mesh(anat, "CC000001") == fallback


def test_resolve_existing_mesh_returns_none_when_missing(tmp_path):
    assert resolve_existing_mesh(tmp_path / "anat", "sub-CC000001") is None
