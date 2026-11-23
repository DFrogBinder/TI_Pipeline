from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ti_current_repair.repeatability_v2.segmentation_propagation import (
    plan_charm45_propagation,
    run_charm45_propagation,
)
from ti_current_repair.repeatability_v2.v3_audit import (
    audit_v3_fixed_geometry,
    build_v3_replacement_plan,
    probe_existing_mesh_solver,
)


def _v3_rows() -> list[dict[str, object]]:
    rows = []
    for subject_index in range(10):
        subject = f"sub-{subject_index:02d}"
        for target in ("left_hippocampus", "right_m1"):
            for repeat in range(1, 41):
                for pair in (1, 2):
                    rows.append(
                        {
                            "subject": subject,
                            "target": target,
                            "condition": "fixed_mesh",
                            "repeat": repeat,
                            "pair": pair,
                            "geometry_sha256": f"{subject}-{target}-pair{pair}",
                        }
                    )
    return rows


def test_v3_audit_retains_only_exactly_one_pair_geometry_per_case() -> None:
    result = audit_v3_fixed_geometry(_v3_rows())
    assert result["status"] == "retain_existing_fixed_results"
    assert result["fixed_simulations_audited"] == 800
    assert result["replacement_simulations"] == 0
    assert result["original_data_policy"] == "preserve"


def test_v3_audit_one_geometry_drift_triggers_full_isolated_fixed_replacement() -> None:
    rows = _v3_rows()
    rows[-1]["geometry_sha256"] = "changed"
    result = audit_v3_fixed_geometry(rows)
    assert result["status"] == "full_fixed_replacement_required"
    assert result["replacement_simulations"] == 800
    assert result["replacement_component_solves"] == 1600
    assert result["affected_cases"] == 1
    assert result["original_data_policy"] == "preserve"
    plan = build_v3_replacement_plan(
        audit=result,
        output_root=Path("/isolated/v3-fixed-replacement-v2"),
    )
    assert plan["array"] == "0-799%50"
    assert plan["simulations"] == 800
    assert plan["component_solves"] == 1600
    assert plan["electrode_placement_allowed"] is False


def test_v3_existing_mesh_capability_probe_rejects_placement_only_api() -> None:
    class Fem:
        @staticmethod
        def tdcs(mesh, cond, currents, electrode_surface_tags, solver_options=None):
            return None

    assert probe_existing_mesh_solver(Fem)["status"] == "ready"

    class BadFem:
        @staticmethod
        def tdcs(session):
            return None

    with pytest.raises(ValueError, match="existing-mesh"):
        probe_existing_mesh_solver(BadFem)


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def test_charm45_plan_requires_five_maps_and_runs_full_propagation(tmp_path: Path) -> None:
    sources = tuple(
        _write(tmp_path / "sources" / f"segmentation_{index}.nii.gz", f"seg-{index}")
        for index in range(1, 6)
    )
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    plan = plan_charm45_propagation(
        subject="sub-CC420071",
        segmentation_paths=sources,
        correction_profile_sha256="a" * 64,
        output_root=tmp_path / "isolated-output",
    )
    assert plan["mesh_tasks"] == 5
    assert plan["ti_simulations"] == 10
    assert plan["component_solves"] == 20
    assert not (tmp_path / "isolated-output").exists()

    def correct(source: Path, output: Path) -> Path:
        return _write(output, f"corrected:{source.read_text()}")

    def mesh(corrected: Path, output: Path) -> Path:
        return _write(output, f"mesh:{corrected.read_text()}")

    def simulate(head: Path, target: str, output: Path) -> dict[str, object]:
        _write(output / "TI.msh", f"ti:{target}:{head.read_text()}")
        return {"status": "complete", "target": target, "component_solves": 2}

    receipt = run_charm45_propagation(
        plan=plan,
        correct_segmentation=correct,
        build_mesh=mesh,
        simulate_and_extract=simulate,
    )

    assert receipt["status"] == "complete"
    assert receipt["mesh_tasks"] == 5
    assert receipt["ti_simulations"] == 10
    assert receipt["interpretation"] == "full_segmentation_to_field_propagation"
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources} == before


def test_charm45_plan_rejects_partial_scope(tmp_path: Path) -> None:
    paths = tuple(_write(tmp_path / f"seg-{index}.nii.gz", "x") for index in range(4))
    with pytest.raises(ValueError, match="exactly five"):
        plan_charm45_propagation(
            subject="sub-CC420071",
            segmentation_paths=paths,
            correction_profile_sha256="a" * 64,
            output_root=tmp_path / "output",
        )


def test_charm45_plan_rejects_non_sha256_correction_identity(tmp_path: Path) -> None:
    paths = tuple(
        _write(tmp_path / f"seg-{index}.nii.gz", "x") for index in range(5)
    )
    with pytest.raises(ValueError, match="SHA-256"):
        plan_charm45_propagation(
            subject="sub-CC420071",
            segmentation_paths=paths,
            correction_profile_sha256="not-a-digest".ljust(64, "x"),
            output_root=tmp_path / "output",
        )
