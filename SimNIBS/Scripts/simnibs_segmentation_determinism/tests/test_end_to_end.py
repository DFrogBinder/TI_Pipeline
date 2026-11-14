from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

import simnibs_segmentation_determinism.charm_runner as charm_runner
import simnibs_segmentation_determinism.headreco_runner as headreco_runner
from simnibs_segmentation_determinism.extension_import import (
    install_extension,
    install_preflight,
)
from simnibs_segmentation_determinism.headreco_runner import expected_headreco_artifacts
from simnibs_segmentation_determinism.provenance import sha256_file
from simnibs_segmentation_determinism.visualizations import build_visualizations
from simnibs_segmentation_determinism.workflow import (
    collect,
    initialize,
    run_repeat,
    runtime_preflight,
)


PACKAGE = Path(__file__).resolve().parents[1]


def _save(path: Path, *, discrete: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        np.ones((2, 2, 2), dtype=np.uint8)
        if discrete
        else np.full((2, 2, 2), 0.25, dtype=np.float32)
    )
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def _install_fake_tools(monkeypatch, *, vary_charm=False):
    monkeypatch.setenv("SEGDET_GIT_COMMIT", "synthetic-test-commit")

    def fake_headreco(command, *, cwd, timeout_seconds, log_path):
        del timeout_seconds
        subject = command[5]
        root = Path(cwd) / f"m2m_{subject}"
        for relative in expected_headreco_artifacts(subject):
            _save(root / relative, discrete="_bin.nii" in relative)
        Path(log_path).write_text("preparevols complete\n", encoding="utf-8")
        return "preparevols complete\n"

    def fake_charm(command, *, cwd, timeout_seconds, log_path):
        del timeout_seconds
        subject = command[1]
        label_map = (
            Path(cwd) / f"m2m_{subject}/label_prep/tissue_labeling_upsampled.nii.gz"
        )
        if vary_charm:
            repeat_index = int(Path(cwd).parent.name.split("_")[-1])
            labels = np.array([*range(1, 11), 1, 1], dtype=np.uint8)
            labels[:10] = np.roll(labels[:10], repeat_index - 1)
            data = labels.reshape((3, 2, 2))
            label_map.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(data, np.eye(4)), str(label_map))
        else:
            _save(label_map, discrete=True)
        Path(log_path).write_text("segmentation complete\n", encoding="utf-8")
        return "segmentation complete\n"

    monkeypatch.setattr(headreco_runner, "run_with_timeout", fake_headreco)
    monkeypatch.setattr(charm_runner, "run_with_timeout", fake_charm)


def _run_repeats(config, *, omit=None):
    for arm in ("simnibs326_headreco", "simnibs401_charm"):
        for array_index in range(10):
            if (arm, array_index) == omit:
                continue
            run_repeat(config, arm=arm, array_index=array_index, attempt=1)


def _install_workflow_evidence(config):
    ready = runtime_preflight(config)
    receipt_root = config.output_root / "_pipeline/module_preflight"
    receipt_root.mkdir(parents=True, exist_ok=True)
    for name in ("simnibs326.json", "simnibs401.json"):
        (receipt_root / name).write_text(json.dumps(ready), encoding="utf-8")
    (receipt_root / "simnibs326_cat12_probe.json").write_text(
        json.dumps({
            "status": "ready",
            "patch_id": "simnibs326-cat12-r2023b-post-init-xml-warning-v2",
            "spm_jobman_initcfg_tested": True,
            "post_init_reassertion_tested": True,
            "mat_report_created": True,
        }),
        encoding="utf-8",
    )
    workflow_root = config.output_root / "_pipeline/workflow"
    workflow_root.mkdir(parents=True, exist_ok=True)
    (workflow_root / "submission.json").write_text(
        json.dumps({
            "status": "submitted",
            "jobs": {
                "module_preflight_326": "1",
                "module_preflight_401": "2",
                "simnibs326_headreco_array": "3",
                "simnibs401_charm_array": "4",
                "collector": "5",
            },
        }),
        encoding="utf-8",
    )


def test_synthetic_twenty_repeat_workflow_completes_with_verified_receipts(
    configured_experiment, dataset_receipt_writer, monkeypatch
):
    config = configured_experiment
    source_hashes = {
        path: sha256_file(path)
        for path in (config.source_root / config.subject / "anat").iterdir()
    }
    initialize(config)
    _install_workflow_evidence(config)
    _install_fake_tools(monkeypatch)

    _run_repeats(config)
    terminal = collect(config)

    assert terminal["status"] == "complete"
    assert terminal["validated_repeats"] == 20
    assert sum(
        1
        for arm in ("simnibs326_headreco", "simnibs401_charm")
        for path in (config.output_root / arm).glob("repeat_*/complete.json")
    ) == 20
    manifest_path = config.output_root / "analysis/determinism_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for arm in ("simnibs326_headreco", "simnibs401_charm"):
        assert manifest["arms"][arm]["geometric_deterministic"] is True
        assert manifest["arms"][arm]["discrete_segmentation_deterministic"] is True
        expected_probability = True if arm == "simnibs326_headreco" else None
        assert manifest["arms"][arm]["probability_map_deterministic"] is expected_probability
    for name, expected_hash in terminal["analysis_artifacts"].items():
        assert sha256_file(config.output_root / "analysis" / name) == expected_hash
    terminal_path = config.output_root / "_pipeline/complete.json"
    sidecar = terminal_path.with_name("complete.json.sha256").read_text(encoding="utf-8")
    assert sidecar.split()[0] == sha256_file(terminal_path)
    assert {
        path: sha256_file(path)
        for path in (config.source_root / config.subject / "anat").iterdir()
    } == source_hashes

    dataset_receipt_writer(config)
    assert install_preflight(config)["status"] == "ready"
    installed = install_extension(
        config,
        confirm_dataset_root=config.dataset_root,
    )
    assert installed["status"] == "complete"
    assert (config.extension_root / "_import/complete.json").is_file()


def test_one_missing_repeat_cannot_create_terminal_receipt(configured_experiment, monkeypatch):
    config = configured_experiment
    initialize(config)
    _install_workflow_evidence(config)
    _install_fake_tools(monkeypatch)
    _run_repeats(config, omit=("simnibs401_charm", 9))

    result = collect(config)

    assert result["status"] == "incomplete"
    assert result["validated_repeats"] == 19
    assert not (config.output_root / "_pipeline/complete.json").exists()


def test_complete_workflow_exports_a_verified_separate_figure_package(
    configured_experiment, monkeypatch
):
    config = configured_experiment
    initialize(config)
    _install_workflow_evidence(config)
    _install_fake_tools(monkeypatch, vary_charm=True)
    _run_repeats(config)
    collect(config)
    terminal = config.output_root / "_pipeline/complete.json"
    terminal_hash = sha256_file(terminal)

    result = build_visualizations(config)

    assert result["status"] == "complete"
    assert result["action"] == "created"
    assert result["figure_count"] == 6
    assert sha256_file(terminal) == terminal_hash
    output = config.output_root / "visualizations_v1"
    assert len(tuple(output.glob("*.png"))) == 6
    assert len(tuple(output.glob("*.pdf"))) == 6
    for line in (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        assert sha256_file(output / relative) == expected
    assert build_visualizations(config)["action"] == "reused"


def test_operator_readme_contains_submission_and_interpretation_gates():
    readme = (PACKAGE / "README.md").read_text(encoding="utf-8")

    assert "does not authorize automatic HPC submission" in readme
    assert "may run while dataset consolidation continues" in readme
    assert "headreco preparevols --noclean" in readme
    assert "--registerT2 --initatlas --segment --forceqform" in readme
    assert "gzip" in readme.lower()
    assert "20 scientific segmentation tasks" in readme
    assert "23 scheduler tasks" in readme
    assert "final_132_repeatability_combined_dataset_v1" in readme
    assert "extensions/segmentation_determinism_v1" in readme
    assert "segmentation_determinism_staging_v1" in readme
    assert "--confirm-dataset-root" in readme
    assert "staged copies" in readme
    assert "simnibs_segmentation_determinism_v1" not in readme
    assert "ti_dataset_final_132_balanced_10_corrected" in readme
