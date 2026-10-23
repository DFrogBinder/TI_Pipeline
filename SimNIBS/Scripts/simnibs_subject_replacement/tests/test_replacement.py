from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from simnibs_subject_replacement.replacement import (
    load_prepared_manifest,
    preflight,
    prepare,
    require_left_complete,
    require_v3_promotion,
)
from simnibs_subject_replacement.settings import ReplacementConfig, replacement_paths


def _config(layout: dict[str, Path | str]) -> ReplacementConfig:
    return ReplacementConfig(
        subject=str(layout["subject"]),
        replaces=str(layout["replaces"]),
        final132_subjects_file=Path(layout["subjects_file"]),
        corrected_v4_scaffold_root=Path(layout["scaffold"]),
        legacy_v3_root=Path(layout["legacy"]),
        atlas_root=Path(layout["atlas"]),
        targets_csv=Path(layout["targets"]),
        work_root=Path(layout["work"]),
    )


def test_preflight_hashes_all_inputs_without_writing(
    replacement_layout: dict[str, Path | str],
) -> None:
    config = _config(replacement_layout)
    paths = replacement_paths(config)

    result = preflight(config)

    assert result["status"] == "ready"
    assert result["profile"] == "external1"
    assert result["subject"] == config.subject
    assert result["subjects"] == [config.subject]
    assert result["scope"]["both_versions"]["simulation_tasks"] == 320
    assert len(result["inputs"]["authoritative_subjects"]["sha256"]) == 64
    assert len(result["inputs"]["atlas"]["sha256"]) == 64
    assert len(result["inputs"]["targets_csv"]["sha256"]) == 64
    assert len(result["inputs"]["legacy_reference"]["files"]) >= 5
    assert len(result["inputs"]["corrected_v4"]) == 3
    assert not paths.root.exists()


def test_preflight_rejects_missing_or_hash_mismatched_sources_without_writing(
    replacement_layout: dict[str, Path | str],
) -> None:
    config = _config(replacement_layout)
    paths = replacement_paths(config)
    atlas = config.atlas_root / f"{config.subject}.nii.gz"
    atlas.unlink()

    with pytest.raises((FileNotFoundError, ValueError), match="atlas"):
        preflight(config)
    assert not paths.root.exists()

    atlas.write_bytes(b"atlas")
    record = config.corrected_v4_scaffold_root / "results" / f"{config.subject}.json"
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["source_t1_sha256"] = "0" * 64
    record.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        preflight(config)
    assert not paths.root.exists()


def test_prepare_atomically_stages_exact_singleton_and_reuses_exact_manifest(
    replacement_layout: dict[str, Path | str],
) -> None:
    config = _config(replacement_layout)
    paths = replacement_paths(config)

    created = prepare(config)
    reused = prepare(config)
    manifest = load_prepared_manifest(paths.manifest)

    assert created["action"] == "created"
    assert reused["action"] == "reused"
    assert manifest["status"] == "prepared"
    assert manifest["subject"] == config.subject
    assert manifest["subjects"] == [config.subject]
    assert manifest["paths"]["manifest"] == str(paths.manifest)
    assert paths.singleton_subjects_file.read_text(encoding="utf-8") == f"{config.subject}\n"
    staged = sorted(
        path.name
        for path in (paths.v4_corrected_root / config.subject / "anat").iterdir()
        if path.is_file()
    )
    assert staged == [
        f"{config.subject}_T1w.nii",
        f"{config.subject}_T1w_ras_1mm_T1andT2_masks.nii",
        f"{config.subject}_T2w.nii",
    ]
    metadata = json.loads(
        (paths.v4_corrected_root / "dataset_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["subject_count"] == 1
    assert metadata["subjects"] == [config.subject]


def test_prepare_rejects_nonempty_unowned_or_stale_root(
    replacement_layout: dict[str, Path | str],
) -> None:
    config = _config(replacement_layout)
    paths = replacement_paths(config)
    paths.root.mkdir(parents=True)
    (paths.root / "unowned.txt").write_text("do not adopt", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unowned"):
        prepare(config)
    assert (paths.root / "unowned.txt").read_text(encoding="utf-8") == "do not adopt"

    other = replace(config, work_root=config.work_root / "other")
    other_paths = replacement_paths(other)
    prepare(other)
    payload = json.loads(other_paths.manifest.read_text(encoding="utf-8"))
    payload["replaces"] = "sub-CC999999"
    other_paths.manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale|mismatch|bound"):
        prepare(other)


def _write_v3_promotion(manifest: dict[str, object]) -> None:
    import hashlib

    paths = manifest["paths"]
    subject = manifest["subject"]
    collection = Path(paths["v3_headmodel_root"]) / "_pipeline/headmodels/external1"
    headmodel_manifest = collection / "manifest.json"
    headmodel_manifest.parent.mkdir(parents=True)
    headmodel_manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "headmodel_root": paths["v3_headmodel_root"],
                "subject_count": 1,
                "subject_order": [subject],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(headmodel_manifest.read_bytes()).hexdigest()
    (collection / "promoted.json").write_text(
        json.dumps(
            {
                "status": "promoted",
                "subjects": [subject],
                "confirm_reviewed": True,
                "manifest_path": str(headmodel_manifest),
                "manifest_sha256": digest,
            }
        ),
        encoding="utf-8",
    )


def _write_left_completion(manifest: dict[str, object], version: str) -> None:
    import hashlib

    paths = manifest["paths"]
    subject = manifest["subject"]
    root = Path(paths[f"{version}_left_root"])
    config = root / "_pipeline/configs/paired_analysis.json"
    config.parent.mkdir(parents=True)
    config.write_text(
        json.dumps(
            {
                "subjects": [subject],
                "conditions": [
                    {"name": "remesh", "repeat_count": 40},
                    {"name": "fixed_mesh", "repeat_count": 40},
                ],
            }
        ),
        encoding="utf-8",
    )
    completion = root / "_pipeline/workflow/complete.json"
    completion.parent.mkdir(parents=True, exist_ok=True)
    if version == "v3":
        payload = {
            "status": "complete",
            "stage": "final",
            "config": str(config),
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "scope": {"target": "left-hippocampus", "subject_count": 1},
            "validation": {"total_tasks": 80},
        }
    else:
        payload = {
            "status": "complete",
            "scope": {
                "roi": "left-hippocampus",
                "subjects": [subject],
                "subject_count": 1,
                "repeats_per_condition": 40,
                "total_simulation_tasks": 80,
            },
        }
    completion.write_text(json.dumps(payload), encoding="utf-8")


def test_simulation_gate_requires_exact_reviewed_v3_promotion(
    replacement_layout: dict[str, Path | str],
) -> None:
    paths = replacement_paths(_config(replacement_layout))
    prepare(_config(replacement_layout))
    manifest = load_prepared_manifest(paths.manifest)

    with pytest.raises(RuntimeError, match="promotion"):
        require_v3_promotion(manifest)

    _write_v3_promotion(manifest)
    result = require_v3_promotion(manifest)
    assert result["status"] == "ready"
    assert result["subjects"] == [manifest["subject"]]

    promotion = Path(result["promotion_path"])
    payload = json.loads(promotion.read_text(encoding="utf-8"))
    payload["subjects"] = ["sub-CC999999"]
    promotion.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="subject"):
        require_v3_promotion(manifest)


@pytest.mark.parametrize("version", ["v3", "v4"])
def test_right_target_gate_requires_same_version_left_terminal_receipt(
    replacement_layout: dict[str, Path | str],
    version: str,
) -> None:
    paths = replacement_paths(_config(replacement_layout))
    prepare(_config(replacement_layout))
    manifest = load_prepared_manifest(paths.manifest)
    _write_v3_promotion(manifest)

    with pytest.raises(RuntimeError, match="left-hippocampus"):
        require_left_complete(manifest, version)

    _write_left_completion(manifest, version)
    result = require_left_complete(manifest, version)
    assert result["status"] == "ready"
    assert result["version"] == version
    assert result["subject"] == manifest["subject"]
