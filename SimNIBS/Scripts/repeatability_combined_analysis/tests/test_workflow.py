from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import pytest

from repeatability_combined_analysis import workflow
from repeatability_combined_analysis.assembly import build_assembly_plan


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_analysis_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_established(inputs: dict[str, Path], output_root: Path) -> dict:
        for directory, payload in (
            (
                "within_simnibs326",
                {"status": "complete", "rows": 1600, "subject_count": 10},
            ),
            (
                "within_simnibs4",
                {"status": "complete", "rows": 1600, "subject_count": 10},
            ),
            (
                "cross_version_remesh",
                {
                    "status": "complete",
                    "input_rows": 3200,
                    "selected_remesh_rows": 1600,
                    "participants": 10,
                },
            ),
        ):
            current = output_root / directory
            current.mkdir(parents=True, exist_ok=True)
            (current / "analysis_manifest.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
        result = {
            "status": "complete",
            "within_simnibs326": {"status": "complete", "rows": 1600},
            "within_simnibs4": {"status": "complete", "rows": 1600},
            "cross_version_remesh": {
                "status": "complete",
                "input_rows": 3200,
                "selected_remesh_rows": 1600,
            },
        }
        (output_root / "established_manifest.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        return result

    def fake_render(frame, output_dir: Path) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        files = (
            "participant_cell_summary.csv",
            "central_value_contrasts.csv",
            "technical_variability_contrasts.csv",
            "finite_support_audit.csv",
            "finite_support_summary.json",
            "analysis_report.md",
            "figure_captions.md",
        )
        for name in files:
            content = "{}\n" if name.endswith(".json") else "test\n"
            (output_dir / name).write_text(content, encoding="utf-8")
        figures = []
        for target in ("left_hippocampus", "right_m1"):
            for family in (
                "central_values",
                "technical_sd",
                "paired_version_contrasts",
                "version_condition_interaction",
            ):
                relative = f"figures/{target}/{family}.png"
                path = output_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"png")
                figures.append(relative)
        manifest = {
            "status": "complete",
            "run_level_rows": len(frame),
            "participant_cells": 80,
            "artifacts": [*files, *figures],
        }
        (output_dir / "comprehensive_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return {
            "status": "complete",
            "manifest": "comprehensive_manifest.json",
            "figures": figures,
            "artifacts": [*files, *figures, "comprehensive_manifest.json"],
        }

    monkeypatch.setattr(workflow, "run_established", fake_established)
    monkeypatch.setattr(workflow, "render_comprehensive_outputs", fake_render)


def test_preflight_is_read_only_and_validates_sources_and_receipts(
    analysis_config,
) -> None:
    before = sorted(
        str(path.relative_to(analysis_config.output_root.parent))
        for path in analysis_config.output_root.parent.rglob("*")
    )
    result = workflow.preflight(analysis_config)
    after = sorted(
        str(path.relative_to(analysis_config.output_root.parent))
        for path in analysis_config.output_root.parent.rglob("*")
    )
    assert result["status"] == "ready"
    assert result["scope"] == {
        "subjects": 10,
        "versions": 2,
        "targets": 2,
        "conditions": 2,
        "participant_cells": 80,
        "run_level_rows": 3200,
        "meshing_tasks": 0,
        "fem_simulation_tasks": 0,
    }
    assert len(result["plan_sha256"]) == 64
    assert result["source_count"] == 10
    assert result["terminal_receipt_count"] == 10
    assert before == after
    assert not analysis_config.output_root.exists()


def test_prepare_requires_exact_root_and_creates_only_ownership_records(
    analysis_config,
) -> None:
    with pytest.raises(workflow.WorkflowError, match="exactly equal"):
        workflow.prepare(analysis_config, "/wrong/root")
    assert not analysis_config.output_root.exists()

    result = workflow.prepare(
        analysis_config, str(analysis_config.output_root)
    )
    assert result["status"] == "prepared"
    files = sorted(
        path.relative_to(analysis_config.output_root).as_posix()
        for path in analysis_config.output_root.rglob("*")
        if path.is_file()
    )
    assert files == ["_workflow/input_plan.json", "_workflow/owner.json"]
    plan = json.loads(
        (analysis_config.output_root / "_workflow/input_plan.json").read_text()
    )
    owner = json.loads(
        (analysis_config.output_root / "_workflow/owner.json").read_text()
    )
    assert owner["output_root"] == str(analysis_config.output_root.resolve())
    assert owner["plan_sha256"] == plan["plan_sha256"]
    assert workflow.prepare(
        analysis_config, str(analysis_config.output_root)
    )["action"] == "reused"


def test_prepare_refuses_unowned_mismatched_and_incomplete_roots(
    analysis_config,
) -> None:
    analysis_config.output_root.mkdir()
    with pytest.raises(workflow.WorkflowError, match="unowned"):
        workflow.prepare(analysis_config, str(analysis_config.output_root))

    shutil.rmtree(analysis_config.output_root)
    workflow.prepare(analysis_config, str(analysis_config.output_root))
    owner_path = analysis_config.output_root / "_workflow/owner.json"
    owner = json.loads(owner_path.read_text())
    owner["plan_sha256"] = "0" * 64
    owner_path.write_text(json.dumps(owner), encoding="utf-8")
    with pytest.raises(workflow.WorkflowError, match="different plan"):
        workflow.prepare(analysis_config, str(analysis_config.output_root))

    owner["plan_sha256"] = json.loads(
        (analysis_config.output_root / "_workflow/input_plan.json").read_text()
    )["plan_sha256"]
    owner_path.write_text(json.dumps(owner), encoding="utf-8")
    (analysis_config.output_root / "_workflow/input_plan.json").unlink()
    with pytest.raises(workflow.WorkflowError, match="incomplete"):
        workflow.prepare(analysis_config, str(analysis_config.output_root))


def test_prepare_refuses_partial_outputs_in_otherwise_owned_root(
    analysis_config,
) -> None:
    workflow.prepare(analysis_config, str(analysis_config.output_root))
    (analysis_config.output_root / "README.md").write_text(
        "partial", encoding="utf-8"
    )
    with pytest.raises(workflow.WorkflowError, match="incomplete"):
        workflow.prepare(analysis_config, str(analysis_config.output_root))


def test_status_recomputes_input_plan_hash_instead_of_trusting_field(
    analysis_config,
) -> None:
    workflow.prepare(analysis_config, str(analysis_config.output_root))
    plan_path = analysis_config.output_root / "_workflow/input_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["scope"]["run_level_rows"] = 1
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    result = workflow.status(analysis_config)
    assert result["state"] == "invalid"
    assert "hash" in result["error"].lower()


def test_execute_rechecks_snapshot_before_creating_attempt(
    analysis_config,
) -> None:
    workflow.prepare(analysis_config, str(analysis_config.output_root))
    source = analysis_config.sources.v3_replacement_left_csv
    with source.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(workflow.WorkflowError, match="changed after preparation"):
        workflow.execute(analysis_config)
    assert not (analysis_config.output_root / "analysis_manifest.json").exists()
    assert not list(analysis_config.output_root.glob(".attempt-*"))


def test_execution_failure_is_wrapped_and_leaves_private_attempt_evidence(
    analysis_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow.prepare(analysis_config, str(analysis_config.output_root))

    def fail_established(_inputs, _output_root):
        raise RuntimeError("synthetic canonical failure")

    monkeypatch.setattr(workflow, "run_established", fail_established)
    with pytest.raises(workflow.WorkflowError, match="synthetic canonical failure"):
        workflow.execute(analysis_config)
    assert len(list(analysis_config.output_root.glob(".attempt-*"))) == 1
    assert workflow.status(analysis_config)["state"] == "running-or-incomplete"


def test_execute_promotes_complete_package_and_is_idempotent(
    analysis_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_analysis_stubs(monkeypatch)
    workflow.prepare(analysis_config, str(analysis_config.output_root))
    result = workflow.execute(analysis_config)
    assert result["status"] == "complete"
    assert result["action"] == "created"
    assert not list(analysis_config.output_root.glob(".attempt-*"))
    assert workflow.execute(analysis_config)["action"] == "reused"


def test_complete_package_has_checksums_portable_archive_and_no_nan(
    analysis_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_analysis_stubs(monkeypatch)
    workflow.prepare(analysis_config, str(analysis_config.output_root))
    workflow.execute(analysis_config)
    root = analysis_config.output_root
    manifest_path = root / "analysis_manifest.json"
    archive = root / f"{root.name}.tar.gz"
    sidecar = archive.with_suffix(archive.suffix + ".sha256")

    for relative in (
        "README.md",
        "subjects.txt",
        "inputs/assembly_manifest.json",
        "established/established_manifest.json",
        "comprehensive/participant_cell_summary.csv",
        "comprehensive/central_value_contrasts.csv",
        "comprehensive/technical_variability_contrasts.csv",
        "comprehensive/finite_support_audit.csv",
        "comprehensive/finite_support_summary.json",
        "comprehensive/analysis_report.md",
        "comprehensive/figure_captions.md",
        "provenance/owner.json",
        "provenance/source_receipts.json",
        "provenance/analysis_code.json",
        "provenance/run_environment.json",
        "analysis_manifest.json",
        "SHA256SUMS",
    ):
        assert (root / relative).is_file(), relative
    assert archive.is_file() and sidecar.is_file()

    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert sha256(root / relative) == expected
    expected_archive, archive_name = sidecar.read_text().strip().split("  ")
    assert archive_name == archive.name
    assert expected_archive == sha256(archive)
    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
    assert names
    assert all(not name.startswith("/") for name in names)
    assert not any("_workflow" in name or ".attempt-" in name for name in names)
    assert archive.name not in names and sidecar.name not in names

    json_paths = list(root.rglob("*.json"))
    assert all("NaN" not in path.read_text(encoding="utf-8") for path in json_paths)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete"
    assert len(manifest["git_commit"]) == 40
    assert manifest["scope"]["run_level_rows"] == 3200
    assert manifest["scope"]["participant_cells"] == 80
    assert manifest["scope"]["meshing_tasks"] == 0
    assert manifest["scope"]["fem_simulation_tasks"] == 0
    assert manifest["analysis_code"]
    assert manifest["environment"]["python"]


def test_status_distinguishes_all_states_and_independently_checks_hashes(
    analysis_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert workflow.status(analysis_config)["state"] == "absent"
    workflow.prepare(analysis_config, str(analysis_config.output_root))
    assert workflow.status(analysis_config)["state"] == "prepared"
    attempt = analysis_config.output_root / ".attempt-test"
    attempt.mkdir()
    assert workflow.status(analysis_config)["state"] == "running-or-incomplete"
    attempt.rmdir()

    install_analysis_stubs(monkeypatch)
    workflow.execute(analysis_config)
    assert workflow.status(analysis_config)["state"] == "complete"
    (analysis_config.output_root / "README.md").write_text(
        "corrupted", encoding="utf-8"
    )
    result = workflow.status(analysis_config)
    assert result["state"] == "invalid"
    assert "checksum" in result["error"].lower()


def test_rebind_is_read_only_and_verifies_exact_mapped_hashes(
    analysis_config, tmp_path: Path
) -> None:
    plan = build_assembly_plan(analysis_config)
    dataset = tmp_path / "combined_dataset"
    (dataset / "_consolidation").mkdir(parents=True)
    (dataset / "provenance").mkdir()
    (dataset / "_consolidation/complete.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "workflow": "repeatability_dataset_consolidation",
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for index, selection in enumerate(plan.selections):
        destination = dataset / "mapped" / f"source-{index}.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(selection.path, destination)
        rows.append(
            {
                "source": str(selection.path),
                "destination": str(destination),
                "role": "analysis_metric_table",
                "subject": "",
                "version": "",
                "target": "",
                "condition": "",
                "verification_status": "checksum_matched",
            }
        )
    mapping = dataset / "provenance/source_mapping.tsv"
    with mapping.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    before = {
        path.relative_to(dataset).as_posix(): (path.stat().st_mtime_ns, path.read_bytes())
        for path in dataset.rglob("*")
        if path.is_file()
    }
    result = workflow.rebind_preflight(analysis_config, dataset)
    after = {
        path.relative_to(dataset).as_posix(): (path.stat().st_mtime_ns, path.read_bytes())
        for path in dataset.rglob("*")
        if path.is_file()
    }
    assert result["status"] == "matched"
    assert result["bindings"] == 10
    assert before == after

    rows[0]["destination"] = str(dataset / "outside-missing.csv")
    with mapping.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(workflow.WorkflowError, match="Mapped destination"):
        workflow.rebind_preflight(analysis_config, dataset)


def test_rebind_requires_completed_dataset(analysis_config, tmp_path: Path) -> None:
    dataset = tmp_path / "combined_dataset"
    dataset.mkdir()
    with pytest.raises(workflow.WorkflowError, match="completion receipt"):
        workflow.rebind_preflight(analysis_config, dataset)
