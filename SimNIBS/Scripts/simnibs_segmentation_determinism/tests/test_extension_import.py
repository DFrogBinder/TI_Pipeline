from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from simnibs_segmentation_determinism import extension_import
from simnibs_segmentation_determinism.extension_import import (
    ExtensionImportError,
    install_extension,
    install_preflight,
)
from simnibs_segmentation_determinism.provenance import sha256_file


def _snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def test_install_preflight_requires_completed_staging_and_dataset(
    configured_experiment, dataset_receipt_writer, staging_completer
):
    config = configured_experiment
    dataset_receipt_writer(config)

    with pytest.raises(ExtensionImportError, match="staging completion"):
        install_preflight(config)

    staging_completer(config)
    config.dataset_completion_file.unlink()
    with pytest.raises(Exception, match="dataset completion"):
        install_preflight(config)


def test_install_preflight_is_read_only_and_reports_exact_roots(
    configured_experiment, dataset_receipt_writer, staging_completer
):
    config = configured_experiment
    terminal = staging_completer(config)
    dataset_receipt_writer(config)

    payload = install_preflight(config)

    assert payload["status"] == "ready"
    assert payload["staging_root"] == str(config.output_root.resolve())
    assert payload["extension_root"] == str(config.extension_root.resolve())
    assert payload["staging_completion_sha256"] == sha256_file(terminal)
    assert payload["source_outputs_modified"] is False
    assert not config.extension_root.exists()


def test_install_requires_exact_dataset_confirmation_without_writing(
    configured_experiment, dataset_receipt_writer, staging_completer
):
    config = configured_experiment
    staging_completer(config)
    dataset_receipt_writer(config)

    with pytest.raises(ExtensionImportError, match="confirmation"):
        install_extension(config, confirm_dataset_root=config.dataset_root.parent)

    assert not config.extension_root.exists()


def test_install_rejects_foreign_and_completed_destinations(
    configured_experiment, dataset_receipt_writer, staging_completer
):
    config = configured_experiment
    staging_completer(config)
    dataset_receipt_writer(config)
    config.extension_root.mkdir(parents=True)
    (config.extension_root / "foreign.txt").write_text("foreign", encoding="utf-8")

    with pytest.raises(ExtensionImportError, match="foreign"):
        install_preflight(config)


def test_interrupted_owned_install_resumes(
    configured_experiment, dataset_receipt_writer, staging_completer, monkeypatch
):
    config = configured_experiment
    staging_completer(config)
    dataset_receipt_writer(config)
    original = extension_import._copy_with_rsync
    attempts = 0

    def interrupted(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            source = config.output_root / "inputs/cohort/subjects.txt"
            partial = config.extension_root / "inputs/cohort/subjects.txt"
            partial.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, partial)
            raise ExtensionImportError("synthetic interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(extension_import, "_copy_with_rsync", interrupted)
    with pytest.raises(ExtensionImportError, match="synthetic interruption"):
        install_extension(config, confirm_dataset_root=config.dataset_root)

    assert install_preflight(config)["action"] == "resume"
    result = install_extension(config, confirm_dataset_root=config.dataset_root)
    assert result["status"] == "complete"
    assert attempts == 2


def test_checksum_verification_detects_destination_only_file(
    configured_experiment, dataset_receipt_writer, staging_completer, monkeypatch
):
    config = configured_experiment
    staging_completer(config)
    dataset_receipt_writer(config)
    original = extension_import._copy_with_rsync

    def copy_then_add_extra(*args, **kwargs):
        original(*args, **kwargs)
        (config.extension_root / "destination-only.txt").write_text(
            "not in staging", encoding="utf-8"
        )

    monkeypatch.setattr(extension_import, "_copy_with_rsync", copy_then_add_extra)
    with pytest.raises(ExtensionImportError, match="checksum verification"):
        install_extension(config, confirm_dataset_root=config.dataset_root)

    assert not (config.extension_root / "_import/complete.json").exists()


def test_install_revalidates_staging_receipt_bindings_after_copy(
    configured_experiment, dataset_receipt_writer, staging_completer, monkeypatch
):
    config = configured_experiment
    staging_completer(config)
    dataset_receipt_writer(config)
    original = extension_import._copy_with_rsync

    def mutate_then_copy(*args, **kwargs):
        analysis = config.output_root / "analysis/determinism_manifest.json"
        analysis.write_text('{"status":"mutated"}\n', encoding="utf-8")
        original(*args, **kwargs)

    monkeypatch.setattr(extension_import, "_copy_with_rsync", mutate_then_copy)
    with pytest.raises(ExtensionImportError, match="staging|artifact"):
        install_extension(config, confirm_dataset_root=config.dataset_root)

    assert not (config.extension_root / "_import/complete.json").exists()


def test_successful_install_is_checksum_verified_and_source_immutable(
    configured_experiment, dataset_receipt_writer, staging_completer
):
    config = configured_experiment
    staging_completer(config)
    dataset_receipt_writer(config)
    before = _snapshot(config.output_root)

    result = install_extension(config, confirm_dataset_root=config.dataset_root)

    assert result["status"] == "complete"
    receipt = config.extension_root / "_import/complete.json"
    assert receipt.is_file()
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["checksum_verification"] == "matched"
    assert payload["source_outputs_modified"] is False
    assert _snapshot(config.output_root) == before
    assert (config.extension_root / "inputs/cohort/subjects.txt").is_file()

    with pytest.raises(ExtensionImportError, match="already complete"):
        install_preflight(config)
