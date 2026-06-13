import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from ti_current_repair.core import (  # noqa: E402
    ElectrodePair,
    RepairKey,
    RepairTask,
    SimulationSpec,
    array_agreement_metrics,
    build_ti_mesh,
    iter_with_rich_progress,
    repair_pair2_rerun_run,
    rerun_pair2_scalar_mesh,
    scale_mesh_e_field,
    validate_same_repair_key,
    write_comparison_summary,
)
from mesh_repeat_analysis.simulation_runners.repair_current_bug import (  # noqa: E402
    CAMCAN_CONDUCTIVITIES,
    REPEATABILITY_CURRENT_SPEC,
    build_repeatability_repair_tasks,
    repeatability_simulation_spec,
)


class FakeField:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=float)


class FakeMesh:
    def __init__(self, e_field):
        self.field = {"E": FakeField(e_field)}
        self.elmdata = ["source"]
        self.added_fields = {}
        self.crop_calls = []

    def crop_mesh(self, tags):
        self.crop_calls.append(np.asarray(tags))
        return self

    def add_element_field(self, value, name):
        self.added_fields[name] = np.asarray(value, dtype=float)


def test_scaled_repair_only_changes_pair2_e_field():
    pair1 = FakeMesh([[1, 0, 0], [0, 1, 0]])
    pair2 = FakeMesh([[2, 0, 0], [0, 2, 0]])

    scaled_pair2 = scale_mesh_e_field(pair2, 0.5)

    assert np.array_equal(pair1.field["E"].value, [[1, 0, 0], [0, 1, 0]])
    assert np.array_equal(pair2.field["E"].value, [[2, 0, 0], [0, 2, 0]])
    assert np.array_equal(scaled_pair2.field["E"].value, [[1, 0, 0], [0, 1, 0]])


def test_recomputed_timax_uses_original_pair1_and_corrected_pair2():
    pair1 = FakeMesh([[1, 0, 0], [0, 1, 0]])
    pair2 = FakeMesh([[0, 2, 0], [0, 0, 3]])

    ti_mesh, timax = build_ti_mesh(
        pair1,
        pair2,
        get_max_ti=lambda e1, e2: np.linalg.norm(e1 + e2, axis=1),
    )

    assert np.allclose(timax, [np.sqrt(5), np.sqrt(10)])
    assert np.allclose(ti_mesh.added_fields["TImax"], timax)
    assert ti_mesh.elmdata == []


def test_array_agreement_metrics_zero_for_identical_and_nonzero_for_perturbation():
    identical = array_agreement_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    perturbed = array_agreement_metrics([1.0, 2.0, 3.0], [1.0, 2.5, 5.0])

    assert identical.max_abs == 0.0
    assert identical.rmse == 0.0
    assert perturbed.max_abs == 2.0
    assert perturbed.rmse > 0.0


def test_comparison_metric_records_shape_mismatch_without_raising():
    import ti_current_repair.core as core

    metrics = core._agreement_metrics_dict_or_error(
        np.zeros((2, 3)),
        np.zeros((3, 3)),
    )

    assert metrics["status"] == "shape_mismatch"
    assert metrics["left_shape"] == [2, 3]
    assert metrics["right_shape"] == [3, 3]
    assert "Shape mismatch" in metrics["error"]
    assert np.isnan(metrics["max_abs"])
    assert metrics["n_values"] == 0


def test_comparison_summary_writes_csv_for_shape_mismatch(tmp_path):
    row = {
        "experiment": "repeatability",
        "subject": "sub-01",
        "dataset": None,
        "condition": "remesh",
        "repeat_tag": "repeat_001",
        "pair2_e": {
            "status": "shape_mismatch",
            "error": "Shape mismatch: (2, 3) vs (3, 3)",
            "left_shape": [2, 3],
            "right_shape": [3, 3],
            "max_abs": float("nan"),
            "rmse": float("nan"),
        },
        "timax": {
            "status": "ok",
            "error": None,
            "left_shape": [2],
            "right_shape": [2],
            "max_abs": 0.0,
            "rmse": 0.0,
        },
        "ti_brain_only": {
            "status": "ok",
            "error": None,
            "left_shape": [2],
            "right_shape": [2],
            "max_abs": 0.0,
            "rmse": 0.0,
        },
        "left_root": "/scaled",
        "right_root": "/pair2-rerun",
    }

    summary = write_comparison_summary([row], tmp_path)

    assert summary["summary_csv"].exists()
    assert "shape_mismatch" in summary["summary_csv"].read_text(encoding="utf-8")


def test_rich_progress_wrapper_advances_after_each_item():
    events = []

    class FakeProgress:
        def __enter__(self):
            events.append(("enter",))
            return self

        def __exit__(self, exc_type, exc, traceback):
            events.append(("exit",))

        def add_task(self, description, *, total):
            events.append(("add_task", description, total))
            return 7

        def advance(self, task_id):
            events.append(("advance", task_id))

    items = list(
        iter_with_rich_progress(
            ["a", "b", "c"],
            total=3,
            description="Comparing runs",
            enabled=True,
            progress_factory=FakeProgress,
        )
    )

    assert items == ["a", "b", "c"]
    assert events == [
        ("enter",),
        ("add_task", "Comparing runs", 3),
        ("advance", 7),
        ("advance", 7),
        ("advance", 7),
        ("exit",),
    ]


def test_rich_progress_wrapper_skips_factory_when_disabled():
    def fail_factory():
        raise AssertionError("progress factory should not be used")

    items = list(
        iter_with_rich_progress(
            [1, 2],
            total=2,
            description="Comparing runs",
            enabled=False,
            progress_factory=fail_factory,
        )
    )

    assert items == [1, 2]


def test_repeatability_scale_factor_matches_current_bug():
    assert REPEATABILITY_CURRENT_SPEC.pair1_label == "F10-P8"
    assert REPEATABILITY_CURRENT_SPEC.pair2_label == "T7-P7"
    assert REPEATABILITY_CURRENT_SPEC.scale_factor == pytest.approx(1.588656 / 2.0)


def test_repeatability_pair2_rerun_uses_camcan_electrode_properties(tmp_path):
    task = build_repeatability_repair_tasks(
        config_path=_write_config(tmp_path),
        original_root=tmp_path / "original",
        output_root=tmp_path / "pair2-rerun",
        subjects=["sub-01"],
        conditions=["remesh"],
        repeats=[1],
    )[0]

    spec = repeatability_simulation_spec(task)

    assert spec.electrode_radius_mm == 10.0
    assert spec.electrode_thickness_mm == 2.0
    assert spec.electrode_conductivity == 1.4
    assert spec.conductivity_by_name == CAMCAN_CONDUCTIVITIES
    assert spec.conductivity_by_index == {}


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "source_root": str(tmp_path / "source"),
                "experiment_root": str(tmp_path / "original"),
                "subjects": ["sub-01"],
                "conditions": [
                    {"name": "remesh", "mesh_mode": "remesh", "repeat_count": 3}
                ],
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_repeatability_plan_preserves_existing_validation_layout(tmp_path):
    config_path = _write_config(tmp_path)

    tasks = build_repeatability_repair_tasks(
        config_path=config_path,
        original_root=tmp_path / "original",
        output_root=tmp_path / "scaled",
        subjects=["sub-01"],
        conditions=["remesh"],
        repeats=[2],
    )

    assert len(tasks) == 1
    task = tasks[0]
    assert task.key == RepairKey(
        experiment="repeatability",
        subject="sub-01",
        condition="remesh",
        repeat_tag="repeat_002",
    )
    assert task.original_subject_root == (
        tmp_path
        / "original"
        / "sub-01_repeatability"
        / "remesh"
        / "repeats"
        / "repeat_002"
        / "sub-01"
    )
    assert task.output_anat_dir == (
        tmp_path
        / "scaled"
        / "sub-01_repeatability"
        / "remesh"
        / "repeats"
        / "repeat_002"
        / "sub-01"
        / "anat"
    )
    assert task.output_dir == task.output_anat_dir / "SimNIBS" / "Output" / "sub-01"


def test_repeatability_comparison_rejects_mismatched_keys():
    left = RepairKey(
        experiment="repeatability",
        subject="sub-01",
        condition="remesh",
        repeat_tag="repeat_001",
    )
    right = RepairKey(
        experiment="repeatability",
        subject="sub-01",
        condition="fixed_mesh",
        repeat_tag="repeat_001",
    )

    with pytest.raises(ValueError, match="different runs"):
        validate_same_repair_key(left, right, experiment="repeatability")


def test_pair2_rerun_stages_new_simulation_inside_repair_root(tmp_path, monkeypatch):
    import ti_current_repair.core as core

    subject = "sub-01"
    original_subject_root = tmp_path / "original" / subject
    original_output = original_subject_root / "anat" / "SimNIBS" / "Output" / subject
    original_output.mkdir(parents=True)
    (original_output / f"{subject}_TDCS_1_scalar.msh").write_text("pair1", encoding="utf-8")
    (original_output / f"{subject}_TDCS_2_scalar.msh").write_text("wrong pair2", encoding="utf-8")

    output_subject_root = tmp_path / "pair2-rerun" / subject
    task = RepairTask(
        key=RepairKey(experiment="repeatability", subject=subject),
        original_subject_root=original_subject_root,
        output_subject_root=output_subject_root,
        original_anat_dir=original_subject_root / "anat",
        output_anat_dir=output_subject_root / "anat",
        current_spec=REPEATABILITY_CURRENT_SPEC,
    )
    spec = SimulationSpec(
        subject=subject,
        head_mesh=task.output_anat_dir / f"m2m_{subject}" / f"{subject}.msh",
        pair2=ElectrodePair("T7", "P7", REPEATABILITY_CURRENT_SPEC.pair2_intended_current_a),
        electrode_radius_mm=10.0,
        electrode_thickness_mm=2.0,
        electrode_shape="ellipse",
        electrode_conductivity=1.4,
    )
    seen = {}

    def fake_rerun_pair2_scalar_mesh(simulation_spec, *, staging_pathfem, **_kwargs):
        seen["staging_pathfem"] = staging_pathfem
        staging_pathfem.mkdir(parents=True)
        pair2_path = staging_pathfem / f"{simulation_spec.subject}_TDCS_1_scalar.msh"
        pair2_path.write_text("corrected pair2", encoding="utf-8")
        return pair2_path

    def fake_recompute_ti_outputs(**_kwargs):
        return {"ti_msh": task.output_dir / "TI.msh"}

    monkeypatch.setattr(core, "rerun_pair2_scalar_mesh", fake_rerun_pair2_scalar_mesh)
    monkeypatch.setattr(core, "recompute_ti_outputs", fake_recompute_ti_outputs)

    repair_pair2_rerun_run(task, spec, overwrite=True, regenerate_volumes=False)

    assert seen["staging_pathfem"].is_relative_to(task.output_sim_root)
    assert not seen["staging_pathfem"].is_relative_to(task.original_sim_root)
    assert (task.output_dir / f"{subject}_TDCS_2_scalar.msh").read_text(encoding="utf-8") == "corrected pair2"
    assert (task.original_dir / f"{subject}_TDCS_2_scalar.msh").read_text(encoding="utf-8") == "wrong pair2"


def test_pair2_rerun_disables_gmsh_visualization(tmp_path):
    class FakeElectrode:
        pass

    class FakeTdcs:
        def __init__(self):
            self.cond = []
            self.currents = []

        def add_electrode(self):
            electrode = FakeElectrode()
            return electrode

    class FakeSession:
        def __init__(self):
            self.open_in_gmsh = True

        def add_tdcslist(self):
            self.tdcs = FakeTdcs()
            return self.tdcs

    class FakeSimStructModule:
        SESSION = FakeSession

    seen = {}

    class FakeSimModule:
        @staticmethod
        def run_simnibs(session):
            seen["open_in_gmsh"] = session.open_in_gmsh
            Path(session.pathfem).mkdir(parents=True, exist_ok=True)
            (Path(session.pathfem) / "sub-01_TDCS_1_scalar.msh").write_text(
                "pair2", encoding="utf-8"
            )

    spec = SimulationSpec(
        subject="sub-01",
        head_mesh=tmp_path / "sub-01.msh",
        pair2=ElectrodePair("T7", "P7", REPEATABILITY_CURRENT_SPEC.pair2_intended_current_a),
        electrode_radius_mm=10.0,
        electrode_thickness_mm=2.0,
        electrode_shape="ellipse",
        electrode_conductivity=1.4,
    )

    rerun_pair2_scalar_mesh(
        spec,
        staging_pathfem=tmp_path / "staging",
        sim_module=FakeSimModule,
        sim_struct_module=FakeSimStructModule,
    )

    assert seen["open_in_gmsh"] is False
