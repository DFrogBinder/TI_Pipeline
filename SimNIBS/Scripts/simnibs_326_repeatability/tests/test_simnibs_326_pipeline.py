from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import simulation_runner  # noqa: E402
from settings import MATLAB_MODULE, SIMNIBS_MODULE, SUBJECTS  # noqa: E402


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


def test_compat_shim_covers_numpy_and_nibabel_5() -> None:
    text = (PACKAGE_ROOT / "hpc" / "activate_compat.sh").read_text(
        encoding="utf-8"
    )
    assert '{"bool": bool, "int": int, "float": float}' in text
    assert "_DataobjImage.get_data = _legacy_get_data" in text
