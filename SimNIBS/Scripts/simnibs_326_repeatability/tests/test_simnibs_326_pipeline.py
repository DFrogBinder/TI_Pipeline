from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import simulation_runner  # noqa: E402
import settings as campaign_settings  # noqa: E402
import cat12_compat  # noqa: E402
from settings import MAX_RETRIES, MATLAB_MODULE, SIMNIBS_MODULE, SUBJECTS  # noqa: E402


PIPELINE_SPEC = importlib.util.spec_from_file_location(
    "simnibs326_campaign_pipeline",
    PACKAGE_ROOT / "pipeline.py",
)
assert PIPELINE_SPEC is not None and PIPELINE_SPEC.loader is not None
pipeline = importlib.util.module_from_spec(PIPELINE_SPEC)
PIPELINE_SPEC.loader.exec_module(pipeline)


def test_full_campaign_scope_is_exact() -> None:
    scope = pipeline.campaign_scope()
    assert scope["simnibs_module"] == "SimNIBS/3.2.6-foss-2023a"
    assert scope["subject_count"] == 10
    assert scope["repeats_per_condition"] == 40
    assert scope["remesh_tasks"] == 800
    assert scope["fixed_mesh_tasks"] == 800
    assert scope["total_simulation_tasks"] == 1600
    assert scope["component_fem_solves"] == 3200
    assert scope["arrays"]["left_hippocampus_remesh"] == "0-399%50"
    assert scope["arrays"]["right_m1_fixed"] == "0-399%50"


def test_remesh_configs_pin_backend_and_confirmed_currents() -> None:
    left = pipeline.remesh_config("left-hippocampus")
    right = pipeline.remesh_config("right-m1")
    assert left[simulation_runner.CONFIG_RUNTIME_KEY]["simnibs_module"] == SIMNIBS_MODULE
    assert left[simulation_runner.CONFIG_RUNTIME_KEY]["backend"] == simulation_runner.BACKEND_NAME
    assert left["subjects"] == list(SUBJECTS)
    assert left["conditions"] == [
        {
            "name": "remesh",
            "mesh_mode": "remesh",
            "repeat_count": 40,
            "description": (
                "Fresh SimNIBS 3.2.6 headreco volumemesh realization from the "
                "participant's immutable v3 scaffold."
            ),
        }
    ]
    assert left["stimulation"]["pair1"]["current_a"] == 0.002
    assert left["stimulation"]["pair2"]["current_a"] == 0.0015886564694485628
    assert right["stimulation"]["pair1"]["current_a"] == 0.002
    assert right["stimulation"]["pair2"]["current_a"] == 0.0006324555320336759


def test_default_v3_outputs_are_isolated_from_protected_charm_roots() -> None:
    isolation = campaign_settings.assert_output_root_isolation()
    writable = isolation["writable_v3_roots"]
    protected = isolation["protected_read_only_roots"]

    assert writable == {
        "v3_scaffolds": (
            "/mnt/parscratch/users/cop23bi/"
            "ti_dataset_final_132_balanced_10_simnibs326_headreco"
        ),
        "v3_left_hippocampus": (
            "/mnt/parscratch/users/cop23bi/"
            "final_132_repeatability_balanced_10_simnibs326_left_hippocampus_v1"
        ),
        "v3_right_m1": (
            "/mnt/parscratch/users/cop23bi/"
            "final_132_repeatability_balanced_10_simnibs326_right_m1_v1"
        ),
    }
    assert protected["left_hippocampus_experiment"].endswith(
        "/final_132_repeatability_balanced_10"
    )
    assert protected["right_m1_experiment"].endswith(
        "/final_132_repeatability_balanced_10_right_m1"
    )


def test_v3_writable_root_collision_with_charm_fails_closed(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        campaign_settings,
        "SCAFFOLD_ROOT",
        campaign_settings.PROTECTED_CHARM_ROOTS["left_hippocampus_experiment"],
    )
    with pytest.raises(RuntimeError, match="protected source/CHARM root"):
        campaign_settings.assert_output_root_isolation()


def test_v3_runner_rejects_config_that_targets_charm_root(tmp_path: Path) -> None:
    config = pipeline.remesh_config("left-hippocampus")
    config["experiment_root"] = str(
        campaign_settings.PROTECTED_CHARM_ROOTS["left_hippocampus_experiment"]
    )
    config_path = tmp_path / "bad_root.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(SystemExit, match="not a dedicated SimNIBS 3.2.6 root"):
        simulation_runner._validate_campaign_config(config_path)


def test_internal_max_ti_matches_basic_geometries() -> None:
    E1 = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    E2 = np.array([[0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    observed = simulation_runner.max_ti(E1, E2)
    assert np.allclose(observed, [1.0, np.sqrt(2.0), 0.0])


def test_scaffold_clone_is_minimal_and_forces_fresh_mesh(
    tmp_path: Path,
    monkeypatch,
) -> None:
    subject = SUBJECTS[0]
    scaffold_root = tmp_path / "scaffolds"
    source = scaffold_root / subject / "anat" / f"m2m_{subject}"
    source.mkdir(parents=True)
    (source.parent / ".simnibs326_scaffold_ready.json").write_text("{}\n")
    for index in range(5):
        (source / f"surface_{index}.stl").write_text("surface\n")
    (source / "T1fs_conform.nii.gz").write_bytes(b"t1")
    (source / "T1fs_nu_conform.nii.gz").write_bytes(b"nu")
    (source / f"{subject}.msh").write_bytes(b"old mesh")
    (source / f"{subject}_final_contr.nii.gz").write_bytes(b"old qc")
    (source / "toMNI").mkdir()
    (source / "toMNI" / "warp.nii.gz").write_bytes(b"warp")
    (source / "segment").mkdir()
    (source / "segment" / "large.nii").write_bytes(b"not copied")

    monkeypatch.setattr(simulation_runner, "SCAFFOLD_ROOT", scaffold_root)
    destination = tmp_path / "repeat" / f"m2m_{subject}"
    simulation_runner._copy_scaffold_for_volumemesh(
        subject=subject,
        destination=destination,
    )

    assert len(list(destination.glob("*.stl"))) == 5
    assert (destination / "T1fs_conform.nii.gz").is_file()
    assert (destination / "toMNI" / "warp.nii.gz").is_file()
    assert not (destination / f"{subject}.msh").exists()
    assert not (destination / f"{subject}_final_contr.nii.gz").exists()
    assert not (destination / "segment").exists()


def test_hpc_launchers_do_not_load_simnibs_4() -> None:
    hpc_root = PACKAGE_ROOT / "hpc"
    launchers = list(hpc_root.glob("*.slurm")) + [hpc_root / "submit.sh"]
    assert launchers
    for launcher in launchers:
        text = launcher.read_text(encoding="utf-8")
        assert "SimNIBS/4.0.1" not in text
    assert "SimNIBS/3.2.6-foss-2023a" in (
        hpc_root / "simulation_array.slurm"
    ).read_text(encoding="utf-8")


def test_headreco_jobs_load_pinned_matlab_dependency() -> None:
    assert MATLAB_MODULE == "MATLAB/2023b"
    hpc_root = PACKAGE_ROOT / "hpc"
    for name in (
        "module_preflight.slurm",
        "scaffold_array.slurm",
        "simulation_array.slurm",
    ):
        text = (hpc_root / name).read_text(encoding="utf-8")
        assert (
            'SIMNIBS326_MATLAB_MODULE="${SIMNIBS326_MATLAB_MODULE:-MATLAB/2023b}"'
            in text
        )
        assert 'module load "${SIMNIBS326_MATLAB_MODULE}"' in text


def test_hpc_retries_are_bounded_element_scoped_and_auditable() -> None:
    assert MAX_RETRIES == 1
    hpc_root = PACKAGE_ROOT / "hpc"
    for name in ("scaffold_array.slurm", "simulation_array.slurm"):
        text = (hpc_root / name).read_text(encoding="utf-8")
        assert "#SBATCH --open-mode=append" in text
        assert 'MAX_RETRIES="${SIMNIBS326_MAX_RETRIES:-1}"' in text
        assert 'REQUEUE_JOB_ID="${ARRAY_JOB_ID}_${TASK_ID}"' in text
        assert 'scontrol requeue "$REQUEUE_JOB_ID"' in text
        assert 'scontrol requeue "$SLURM_JOB_ID"' not in text
        assert "_retry_${RETRY_COUNT}_${ATTEMPT_STAMP}.log" in text
        assert 'if [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; then' in text

    submit = (hpc_root / "submit.sh").read_text(encoding="utf-8")
    assert submit.count("--open-mode=append") == 2
    assert "SIMNIBS326_MAX_RETRIES=$MAX_RETRIES" in submit
    assert "unlimited validation-gated requeue" not in submit


def test_cat12_compatibility_patch_is_bounded() -> None:
    segment = (
        "before\n"
        + cat12_compat.SEGMENT_PRE_INIT_INSERTION_POINT
        + "\nmiddle\n"
        + cat12_compat.SEGMENT_POST_INIT_INSERTION_POINT
        + "\nafter\n"
    )
    xml = (
        "before\n"
        + cat12_compat.XML_WRITE_ERROR
        + "\nmiddle\n"
        + cat12_compat.XML_WRITE_ERROR
        + "\nafter\n"
    )

    patched_segment, patched_xml, replacements = cat12_compat._patch_source_text(
        segment_text=segment,
        xml_text=xml,
    )

    assert replacements == {
        "segment_pre_init_path_insertions": 1,
        "segment_post_init_path_reassertions": 1,
        "xml_error_to_warning_replacements": 2,
    }
    assert "SIMNIBS326_CAT12_COMPAT_DIR" in patched_segment
    assert "CAT12 compatibility overlay is not first" in patched_segment
    assert "SIMNIBS326_CAT12_COMPAT_ACTIVE_POST_INIT" in patched_segment
    assert "rmpath(compat_dir);" in patched_segment
    assert "clear cat_io_xml" in patched_segment
    assert cat12_compat.XML_WRITE_ERROR not in patched_xml
    assert patched_xml.count(cat12_compat.XML_WRITE_WARNING) == 2


def test_cat12_compatibility_rejects_unexpected_source_layout() -> None:
    with pytest.raises(RuntimeError, match="exactly two fatal"):
        cat12_compat._patch_source_text(
            segment_text=(
                cat12_compat.SEGMENT_PRE_INIT_INSERTION_POINT
                + "\n"
                + cat12_compat.SEGMENT_POST_INIT_INSERTION_POINT
            ),
            xml_text=cat12_compat.XML_WRITE_ERROR,
        )
    with pytest.raises(RuntimeError, match="pre-init compatibility"):
        cat12_compat._patch_source_text(
            segment_text="no insertion point",
            xml_text=(cat12_compat.XML_WRITE_ERROR + "\n") * 2,
        )
    with pytest.raises(RuntimeError, match="post-init compatibility"):
        cat12_compat._patch_source_text(
            segment_text=cat12_compat.SEGMENT_PRE_INIT_INSERTION_POINT,
            xml_text=(cat12_compat.XML_WRITE_ERROR + "\n") * 2,
        )


def test_headreco_matlab_command_gets_one_compatibility_path() -> None:
    command = (
        "matlab -nosplash -nodesktop -r \"addpath('/msh','/spm12');"
        "try,segment_SPM('x'); segment_CAT('y');catch ME,rethrow(ME);end,exit;\""
    )
    overlay = Path("/tmp/simnibs326_cat12_overlay")
    observed = cat12_compat._inject_matlab_overlay(command, overlay)
    assert observed.count("simnibs326_cat12_overlay") == 1
    assert "addpath('/tmp/simnibs326_cat12_overlay','-begin');try," in observed
    assert cat12_compat._inject_matlab_overlay("matlab -batch version", overlay) == (
        "matlab -batch version"
    )


def test_module_receipt_requires_cat12_compatibility(tmp_path: Path) -> None:
    receipt = tmp_path / "module_preflight.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "ready",
                "expected_module": SIMNIBS_MODULE,
                "expected_matlab_module": MATLAB_MODULE,
                "cat12_compatibility": {
                    "status": "ready",
                    "patch_id": cat12_compat.PATCH_ID,
                    "source_sha256": cat12_compat.EXPECTED_SOURCE_SHA256,
                    "replacements": {
                        "segment_pre_init_path_insertions": 1,
                        "segment_post_init_path_reassertions": 1,
                        "xml_error_to_warning_replacements": 2,
                    },
                    "spm_jobman_initcfg_tested": True,
                    "post_init_reassertion_tested": True,
                    "mat_report_created": True,
                },
            }
        ),
        encoding="utf-8",
    )
    result = pipeline.validate_module_preflight_receipt(receipt)
    assert result["status"] == "ready"
    assert result["cat12_compatibility_patch"] == cat12_compat.PATCH_ID

    receipt.write_text(json.dumps({"status": "ready"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="CAT12 compatibility evidence is missing"):
        pipeline.validate_module_preflight_receipt(receipt)


def test_scaffold_runner_uses_cat12_compatibility_wrapper() -> None:
    text = (PACKAGE_ROOT / "scaffold_runner.py").read_text(encoding="utf-8")
    assert '"cat12_compat.py"' in text
    assert '"headreco"' in text
    launcher = (PACKAGE_ROOT / "hpc" / "scaffold_array.slurm").read_text(
        encoding="utf-8"
    )
    assert "validate-module-receipt" in launcher
    submit = (PACKAGE_ROOT / "hpc" / "submit.sh").read_text(encoding="utf-8")
    assert "PREFLIGHT_RECEIPT=$PREFLIGHT_RECEIPT" in submit
    assert 'CAT12_COMPAT="$PIPELINE_DIR/cat12_compat.py"' in submit


def test_compat_shim_covers_numpy_and_nibabel_5() -> None:
    text = (PACKAGE_ROOT / "hpc" / "activate_compat.sh").read_text(
        encoding="utf-8"
    )
    assert '{"bool": bool, "int": int, "float": float}' in text
    assert "_DataobjImage.get_data = _legacy_get_data" in text


def test_v3_label_export_matches_v4_tag_assignment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import nibabel as nib

    calls: dict[str, object] = {}

    class FakeElements:
        tag1 = np.array([1, 2, 3, 5], dtype=np.int32)

    class FakeMesh:
        elm = FakeElements()

        def crop_mesh(self, *, elm_type: int):
            calls["elm_type"] = elm_type
            return self

    class FakeElementData:
        def __init__(self, values) -> None:
            calls["values"] = np.asarray(values).copy()
            self.mesh = None

        def to_nifti(
            self,
            dimensions,
            affine,
            *,
            fn: str,
            qform,
            method: str,
        ) -> None:
            calls["dimensions"] = tuple(int(value) for value in dimensions)
            calls["affine"] = np.asarray(affine).copy()
            calls["qform"] = np.asarray(qform).copy()
            calls["method"] = method
            calls["mesh"] = self.mesh
            Path(fn).write_bytes(b"label-nifti")

    class FakeMeshIO:
        ElementData = FakeElementData

        @staticmethod
        def read_msh(path: str):
            calls["mesh_path"] = path
            return FakeMesh()

    reference = tmp_path / "reference.nii.gz"
    nib.save(
        nib.Nifti1Image(np.zeros((3, 4, 5), dtype=np.uint8), np.eye(4)),
        reference,
    )
    mesh_path = tmp_path / "TI.msh"
    mesh_path.write_bytes(b"mesh")

    monkeypatch.setattr(simulation_runner.runner, "SIM_MODULE", object())
    monkeypatch.setattr(simulation_runner.runner, "SIM_MESH_IO", FakeMeshIO)
    output = simulation_runner._write_label_volume_326(
        mesh_path,
        reference,
        tmp_path / "TI_Volumetric_Labels",
    )

    assert output == tmp_path / "TI_Volumetric_Labels.nii.gz"
    assert output.read_bytes() == b"label-nifti"
    assert calls["mesh_path"] == str(mesh_path)
    assert calls["elm_type"] == 4
    assert np.array_equal(calls["values"], [1, 2, 3, 5])
    assert calls["dimensions"] == (3, 4, 5)
    assert calls["method"] == "assign"
    assert isinstance(calls["mesh"], FakeMesh)


def test_v3_backend_installs_internal_label_exporter() -> None:
    source = (PACKAGE_ROOT / "simulation_runner.py").read_text(encoding="utf-8")
    assert "runner.SIM_LABEL_EXPORTER = _write_label_volume_326" in source
    assert Path(simulation_runner.runner.__file__).resolve() == (
        PACKAGE_ROOT.parent
        / "ti_current_repair"
        / "simulation_runners"
        / "repeatability_experiment.py"
    ).resolve()
    shared_source = Path(simulation_runner.runner.__file__).read_text(encoding="utf-8")
    assert "SIM_LABEL_EXPORTER = None" in shared_source
    assert "SIM_LABEL_EXPORTER(" in shared_source
