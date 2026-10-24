from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from simnibs_326_corrected_label_repeatability.settings import (
    BASE9_SUBJECTS,
    ENV_PREFIX,
    REPEAT_COUNT,
    SUBJECTS,
    assert_root_isolation,
    campaign_paths_from_environment,
    campaign_scope,
    cohort_name,
    cohort_subjects,
    load_external_manifest,
    subject_paths,
    target_settings,
)


EXPECTED_SUBJECTS = (
    "sub-CC110174",
    "sub-CC121144",
    "sub-CC310407",
    "sub-CC320616",
    "sub-CC420071",
    "sub-CC410432",
    "sub-CC520083",
    "sub-CC520127",
    "sub-CC610631",
    "sub-CC720941",
)


def test_pins_full_campaign_scope_and_default_roots() -> None:
    paths = campaign_paths_from_environment({})
    scope = campaign_scope(paths)

    assert SUBJECTS == EXPECTED_SUBJECTS
    assert REPEAT_COUNT == 40
    assert ENV_PREFIX == "SIMNIBS326H_"
    assert paths.headmodel_root == Path(
        "/mnt/parscratch/users/cop23bi/"
        "ti_dataset_final_132_balanced_10_simnibs326_harmonized_v2"
    )
    assert paths.left_root == Path(
        "/mnt/parscratch/users/cop23bi/"
        "final_132_repeatability_balanced_10_simnibs326_harmonized_"
        "left_hippocampus_v2"
    )
    assert paths.right_root == Path(
        "/mnt/parscratch/users/cop23bi/"
        "final_132_repeatability_balanced_10_simnibs326_harmonized_right_m1_v2"
    )
    assert scope["subjects"] == list(EXPECTED_SUBJECTS)
    assert scope["targets"] == ["left-hippocampus", "right-m1"]
    assert scope["conditions"] == ["remesh", "fixed_mesh"]
    assert scope["total_simulation_tasks"] == 1600
    assert scope["component_fem_solves"] == 3200
    assert scope["arrays"] == {
        "left_hippocampus_remesh": "0-399%50",
        "left_hippocampus_fixed": "0-399%50",
        "right_m1_remesh": "0-399%50",
        "right_m1_fixed": "0-399%50",
    }
    assert target_settings("left-hippocampus", paths).experiment_root == paths.left_root
    assert target_settings("right-m1", paths).experiment_root == paths.right_root


def test_base9_profile_excludes_only_failed_subject_and_uses_isolated_roots() -> None:
    environ = {"SIMNIBS326H_COHORT": "base9"}

    paths = campaign_paths_from_environment(environ)
    scope = campaign_scope(paths, environ=environ)

    assert cohort_name(environ) == "base9"
    assert cohort_subjects(environ) == BASE9_SUBJECTS
    assert BASE9_SUBJECTS == tuple(
        subject for subject in EXPECTED_SUBJECTS if subject != "sub-CC520083"
    )
    assert scope["subjects"] == list(BASE9_SUBJECTS)
    assert scope["subject_count"] == 9
    assert scope["total_simulation_tasks"] == 1440
    assert scope["component_fem_solves"] == 2880
    assert scope["arrays"] == {
        "left_hippocampus_remesh": "0-359%50",
        "left_hippocampus_fixed": "0-359%50",
        "right_m1_remesh": "0-359%50",
        "right_m1_fixed": "0-359%50",
    }
    assert paths.left_root == Path(
        "/mnt/parscratch/users/cop23bi/"
        "final_132_repeatability_balanced_10_simnibs326_harmonized_"
        "base9_left_hippocampus_v2"
    )
    assert paths.right_root == Path(
        "/mnt/parscratch/users/cop23bi/"
        "final_132_repeatability_balanced_10_simnibs326_harmonized_"
        "base9_right_m1_v2"
    )


def test_base9_profile_uses_only_namespaced_base9_root_overrides(tmp_path: Path) -> None:
    environ = {
        "SIMNIBS326H_COHORT": "base9",
        "SIMNIBS326H_LEFT_ROOT": str(tmp_path / "full-left"),
        "SIMNIBS326H_RIGHT_ROOT": str(tmp_path / "full-right"),
        "SIMNIBS326H_BASE9_LEFT_ROOT": str(tmp_path / "base9-left"),
        "SIMNIBS326H_BASE9_RIGHT_ROOT": str(tmp_path / "base9-right"),
    }

    paths = campaign_paths_from_environment(environ)

    assert paths.left_root == (tmp_path / "base9-left").resolve()
    assert paths.right_root == (tmp_path / "base9-right").resolve()


def test_base9_profile_rejects_collision_with_full10_writable_roots(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared-result-root"
    environ = {
        "SIMNIBS326H_COHORT": "base9",
        "SIMNIBS326H_LEFT_ROOT": str(shared),
        "SIMNIBS326H_BASE9_LEFT_ROOT": str(shared),
        "SIMNIBS326H_RIGHT_ROOT": str(tmp_path / "full-right"),
        "SIMNIBS326H_BASE9_RIGHT_ROOT": str(tmp_path / "base9-right"),
    }

    with pytest.raises(ValueError, match="base9.*full10 writable result root"):
        campaign_paths_from_environment(environ)


def test_unknown_cohort_profile_fails_closed() -> None:
    environ = {"SIMNIBS326H_COHORT": "partial"}

    with pytest.raises(ValueError, match="Unsupported cohort profile"):
        cohort_subjects(environ)
    with pytest.raises(ValueError, match="Unsupported cohort profile"):
        campaign_paths_from_environment(environ)


def _write_external_manifest(
    tmp_path: Path,
    *,
    subject: str = "sub-CC999999",
    **overrides: object,
) -> Path:
    path = tmp_path / "replacement_manifest.json"
    payload: dict[str, object] = {
        "schema_version": 1,
        "workflow": "simnibs_subject_replacement",
        "profile": "external1",
        "status": "prepared",
        "subject": subject,
        "subjects": [subject],
        "paths": {
            "manifest": str(path.resolve()),
            "v4_corrected_root": str((tmp_path / "corrected").resolve()),
            "v3_headmodel_root": str((tmp_path / "headmodels").resolve()),
            "v3_left_root": str((tmp_path / "left").resolve()),
            "v3_right_root": str((tmp_path / "right").resolve()),
        },
        "inputs": {
            "legacy_reference": {
                "anat_root": str((tmp_path / "legacy" / subject / "anat").resolve())
            },
            "atlas": {
                "path": str((tmp_path / "atlas" / f"{subject}.nii.gz").resolve())
            },
            "targets_csv": {
                "path": str((tmp_path / "targets.csv").resolve())
            },
        },
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_external_profile_loads_one_manifest_bound_subject(tmp_path: Path) -> None:
    manifest_path = _write_external_manifest(tmp_path)
    environ = {
        "SIMNIBS326H_COHORT": "external1",
        "SIMNIBS326H_EXTERNAL_MANIFEST": str(manifest_path),
    }

    manifest = load_external_manifest(environ)

    assert manifest["subject"] == "sub-CC999999"
    assert cohort_name(environ) == "external1"
    assert cohort_subjects(environ) == ("sub-CC999999",)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"profile": "base9"}, "profile"),
        ({"status": "draft"}, "prepared"),
        ({"subjects": ["sub-CC999999", "sub-CC888888"]}, "exactly one"),
        ({"subject": "../sub-CC999999", "subjects": ["../sub-CC999999"]}, "canonical"),
        ({"subjects": ["sub-CC888888"]}, "subject.*subjects"),
    ],
)
def test_external_manifest_rejects_invalid_binding(
    tmp_path: Path,
    overrides: dict[str, object],
    match: str,
) -> None:
    manifest_path = _write_external_manifest(tmp_path, **overrides)
    environ = {
        "SIMNIBS326H_COHORT": "external1",
        "SIMNIBS326H_EXTERNAL_MANIFEST": str(manifest_path),
    }

    with pytest.raises(ValueError, match=match):
        cohort_subjects(environ)


def test_external_manifest_rejects_different_self_path(tmp_path: Path) -> None:
    manifest_path = _write_external_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["paths"]["manifest"] = str(tmp_path / "other.json")
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="different path"):
        cohort_subjects(
            {
                "SIMNIBS326H_COHORT": "external1",
                "SIMNIBS326H_EXTERNAL_MANIFEST": str(manifest_path),
            }
        )


def test_external_profile_rejects_environment_root_not_bound_to_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = _write_external_manifest(tmp_path)
    environ = {
        "SIMNIBS326H_COHORT": "external1",
        "SIMNIBS326H_EXTERNAL_MANIFEST": str(manifest_path),
        "SIMNIBS326H_CORRECTED_ROOT": str(tmp_path / "wrong-corrected"),
        "SIMNIBS326H_LEGACY_REFERENCE_ROOT": str(tmp_path / "legacy"),
        "SIMNIBS326H_ATLAS_ROOT": str(tmp_path / "atlas"),
        "SIMNIBS326H_TARGETS_CSV": str(tmp_path / "targets.csv"),
        "SIMNIBS326H_HEADMODEL_ROOT": str(tmp_path / "headmodels"),
        "SIMNIBS326H_LEFT_ROOT": str(tmp_path / "left"),
        "SIMNIBS326H_RIGHT_ROOT": str(tmp_path / "right"),
    }

    with pytest.raises(ValueError, match="CORRECTED_ROOT.*manifest"):
        campaign_paths_from_environment(environ)


def test_external_manifest_is_mandatory_absolute_and_valid_json(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="EXTERNAL_MANIFEST"):
        cohort_subjects({"SIMNIBS326H_COHORT": "external1"})

    with pytest.raises(ValueError, match="absolute"):
        cohort_subjects(
            {
                "SIMNIBS326H_COHORT": "external1",
                "SIMNIBS326H_EXTERNAL_MANIFEST": "relative/manifest.json",
            }
        )

    malformed = tmp_path / "malformed.json"
    malformed.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="valid JSON"):
        cohort_subjects(
            {
                "SIMNIBS326H_COHORT": "external1",
                "SIMNIBS326H_EXTERNAL_MANIFEST": str(malformed),
            }
        )


def test_external_profile_has_singleton_scope_and_subject_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = "sub-CC999999"
    manifest_path = _write_external_manifest(tmp_path, subject=subject)
    environ = {
        "SIMNIBS326H_COHORT": "external1",
        "SIMNIBS326H_EXTERNAL_MANIFEST": str(manifest_path),
        "SIMNIBS326H_CORRECTED_ROOT": str(tmp_path / "corrected"),
        "SIMNIBS326H_LEGACY_REFERENCE_ROOT": str(tmp_path / "legacy"),
        "SIMNIBS326H_ATLAS_ROOT": str(tmp_path / "atlas"),
        "SIMNIBS326H_TARGETS_CSV": str(tmp_path / "targets.csv"),
        "SIMNIBS326H_HEADMODEL_ROOT": str(tmp_path / "headmodels"),
        "SIMNIBS326H_LEFT_ROOT": str(tmp_path / "left"),
        "SIMNIBS326H_RIGHT_ROOT": str(tmp_path / "right"),
    }
    paths = campaign_paths_from_environment(environ)
    scope = campaign_scope(paths, environ=environ)
    for key, value in environ.items():
        monkeypatch.setenv(key, value)

    assert scope["subject_count"] == 1
    assert scope["total_simulation_tasks"] == 160
    assert scope["component_fem_solves"] == 320
    assert set(scope["arrays"].values()) == {"0-39%50"}
    assert subject_paths(paths, subject).subject == subject
    with pytest.raises(ValueError, match="active external1 subject"):
        subject_paths(paths, SUBJECTS[0])


def test_environment_overrides_share_one_prefix(tmp_path: Path) -> None:
    environ = {
        "SIMNIBS326H_CORRECTED_ROOT": str(tmp_path / "corrected"),
        "SIMNIBS326H_LEGACY_REFERENCE_ROOT": str(tmp_path / "legacy"),
        "SIMNIBS326H_HEADMODEL_ROOT": str(tmp_path / "headmodels"),
        "SIMNIBS326H_LEFT_ROOT": str(tmp_path / "left"),
        "SIMNIBS326H_RIGHT_ROOT": str(tmp_path / "right"),
        "SIMNIBS326H_ATLAS_ROOT": str(tmp_path / "atlases"),
    }

    paths = campaign_paths_from_environment(environ)

    assert paths.corrected_root == (tmp_path / "corrected").resolve()
    assert paths.legacy_reference_root == (tmp_path / "legacy").resolve()
    assert paths.headmodel_root == (tmp_path / "headmodels").resolve()
    assert paths.left_root == (tmp_path / "left").resolve()
    assert paths.right_root == (tmp_path / "right").resolve()
    assert paths.atlas_root == (tmp_path / "atlases").resolve()


@pytest.mark.parametrize("relationship", ["equal", "ancestor", "descendant"])
@pytest.mark.parametrize(
    "protected_field",
    [
        "corrected_root",
        "legacy_reference_root",
        "atlas_root",
        "completed_v3_left_root",
        "completed_v3_right_root",
        "completed_v4_left_root",
        "completed_v4_right_root",
    ],
)
def test_root_isolation_rejects_protected_collisions(
    tmp_path: Path,
    protected_field: str,
    relationship: str,
) -> None:
    paths = _disjoint_paths(tmp_path)
    protected = getattr(paths, protected_field)
    if relationship == "equal":
        unsafe = protected
    elif relationship == "ancestor":
        unsafe = protected.parent
    else:
        unsafe = protected / "child"

    with pytest.raises(RuntimeError, match="overlap"):
        assert_root_isolation(replace(paths, headmodel_root=unsafe))


@pytest.mark.parametrize("relationship", ["equal", "ancestor", "descendant"])
def test_root_isolation_rejects_writable_root_collisions(
    tmp_path: Path,
    relationship: str,
) -> None:
    paths = _disjoint_paths(tmp_path)
    if relationship == "equal":
        unsafe = paths.headmodel_root
    elif relationship == "ancestor":
        unsafe = paths.headmodel_root.parent
    else:
        unsafe = paths.headmodel_root / "child"

    with pytest.raises(RuntimeError, match="overlap"):
        assert_root_isolation(replace(paths, left_root=unsafe))


def test_disjoint_root_report_is_json_serializable(tmp_path: Path) -> None:
    report = assert_root_isolation(_disjoint_paths(tmp_path))

    encoded = json.dumps(report, sort_keys=True)
    assert "headmodel_root" in encoded
    assert "corrected_root" in encoded


def test_subject_paths_expose_inputs_outputs_and_receipts(tmp_path: Path) -> None:
    paths = _disjoint_paths(tmp_path)
    subject = SUBJECTS[0]

    resolved = subject_paths(paths, subject)

    assert resolved.corrected_label == (
        paths.corrected_root
        / subject
        / "anat"
        / f"{subject}_T1w_ras_1mm_T1andT2_masks.nii"
    )
    assert resolved.corrected_t1 == (
        paths.corrected_root / subject / "anat" / f"{subject}_T1w.nii"
    )
    assert resolved.corrected_t2 == (
        paths.corrected_root / subject / "anat" / f"{subject}_T2w.nii"
    )
    assert resolved.legacy_m2m == (
        paths.legacy_reference_root / subject / "anat" / f"m2m_{subject}"
    )
    assert resolved.legacy_ready_marker.name == ".simnibs326_scaffold_ready.json"
    assert resolved.staged_m2m == paths.headmodel_root / subject / "anat" / f"m2m_{subject}"
    assert resolved.harmonized_label == (
        paths.headmodel_root
        / "harmonized_labels"
        / f"{subject}_v3_harmonized_t1fs_grid.nii.gz"
    )
    assert resolved.head_mesh == paths.headmodel_root / subject / "anat" / f"{subject}.msh"
    assert resolved.reference_receipt.name == "reference_stage.json"
    assert resolved.headmodel_receipt.name == "headmodel_complete.json"


def test_unknown_subject_and_target_fail_closed(tmp_path: Path) -> None:
    paths = _disjoint_paths(tmp_path)
    with pytest.raises(ValueError, match="Unsupported subject"):
        subject_paths(paths, "sub-UNKNOWN")
    with pytest.raises(ValueError, match="Unsupported target"):
        target_settings("other", paths)


def _disjoint_paths(tmp_path: Path):
    defaults = campaign_paths_from_environment({})
    replacements = {
        field: (tmp_path / field).resolve()
        for field in (
            "corrected_root",
            "legacy_reference_root",
            "atlas_root",
            "headmodel_root",
            "left_root",
            "right_root",
            "completed_v3_left_root",
            "completed_v3_right_root",
            "completed_v4_left_root",
            "completed_v4_right_root",
        )
    }
    return replace(defaults, **replacements)
