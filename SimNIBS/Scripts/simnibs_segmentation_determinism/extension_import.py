from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Mapping, Sequence

from .provenance import (
    ProvenanceError,
    path_is_within,
    sha256_file,
    validate_sha256_sidecar,
    write_json_atomic,
    write_sha256_sidecar,
)
from .settings import (
    ExperimentConfig,
    validate_dataset_receipts,
    validate_output_isolation,
)


class ExtensionImportError(RuntimeError):
    """Raised when the extension cannot be installed without ambiguity."""


def _read_json(path: Path, label: str) -> Dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ExtensionImportError(f"Missing or unsafe {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExtensionImportError(f"Unreadable {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ExtensionImportError(f"Invalid {label}: {path}")
    return payload


def _require_bound_file(
    root: Path,
    path: Path,
    expected_sha256: object,
    expected_size: object = None,
) -> None:
    resolved = path.resolve(strict=False)
    if (
        not path_is_within(resolved, root)
        or path.is_symlink()
        or not resolved.is_file()
        or not isinstance(expected_sha256, str)
        or sha256_file(resolved) != expected_sha256
        or (
            expected_size is not None
            and (
                not isinstance(expected_size, int)
                or resolved.stat().st_size != expected_size
            )
        )
    ):
        raise ExtensionImportError(f"Staging receipt artifact is invalid: {path}")


def _validate_repeat_receipt(
    config: ExperimentConfig,
    record: Mapping[str, object],
) -> tuple[str, int]:
    arm = record.get("arm")
    repeat_index = record.get("repeat_index")
    if arm not in {"simnibs326_headreco", "simnibs401_charm"} or not isinstance(
        repeat_index, int
    ):
        raise ExtensionImportError("Staging repeat-receipt identity is invalid")
    expected_path = (
        config.output_root / str(arm) / f"repeat_{repeat_index:03d}/complete.json"
    )
    observed_path = Path(str(record.get("path", "")))
    if observed_path != expected_path:
        raise ExtensionImportError("Staging repeat-receipt path is noncanonical")
    _require_bound_file(config.output_root, observed_path, record.get("sha256"))
    try:
        validate_sha256_sidecar(observed_path)
    except ProvenanceError as exc:
        raise ExtensionImportError(
            f"Staging repeat-receipt sidecar is invalid: {observed_path}"
        ) from exc
    receipt = _read_json(observed_path, "staging repeat receipt")
    if (
        receipt.get("status") != "complete"
        or receipt.get("arm") != arm
        or receipt.get("repeat_index") != repeat_index
        or receipt.get("source_outputs_modified") is not False
    ):
        raise ExtensionImportError("Staging repeat receipt binding is invalid")

    for label, hash_label in (
        ("artifact_manifest", "artifact_manifest_sha256"),
        ("log", "log_sha256"),
    ):
        path = Path(str(receipt.get(label, "")))
        _require_bound_file(config.output_root, path, receipt.get(hash_label))
        if label == "artifact_manifest":
            try:
                validate_sha256_sidecar(path)
            except ProvenanceError as exc:
                raise ExtensionImportError(
                    f"Staging artifact-manifest sidecar is invalid: {path}"
                ) from exc

    inputs = receipt.get("inputs")
    if not isinstance(inputs, dict):
        raise ExtensionImportError("Staging repeat input binding is invalid")
    for modality in ("t1", "t2"):
        input_record = inputs.get(modality)
        if not isinstance(input_record, dict):
            raise ExtensionImportError("Staging repeat input record is invalid")
        _require_bound_file(
            config.output_root,
            Path(str(input_record.get("path", ""))),
            input_record.get("sha256"),
        )

    artifacts = (
        receipt.get("artifacts")
        if arm == "simnibs326_headreco"
        else [receipt.get("artifact")]
    )
    if not isinstance(artifacts, list) or not artifacts:
        raise ExtensionImportError("Staging repeat artifact inventory is invalid")
    artifact_root = (
        observed_path.parent / "artifacts"
        if arm == "simnibs326_headreco"
        else observed_path.parent
    )
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ExtensionImportError("Staging repeat artifact record is invalid")
        relative = Path(str(artifact.get("relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ExtensionImportError("Staging repeat artifact path is invalid")
        _require_bound_file(
            config.output_root,
            artifact_root / relative,
            artifact.get("sha256"),
            artifact.get("size_bytes"),
        )
    return str(arm), repeat_index


def _validate_staging_completion(config: ExperimentConfig) -> Dict[str, object]:
    # Imported locally to keep the CLI module graph acyclic while reusing the
    # same staged-input and ownership checks as compute tasks.
    from .workflow import runtime_preflight

    try:
        runtime_preflight(config)
    except (RuntimeError, OSError, ValueError) as exc:
        raise ExtensionImportError(
            f"Invalid staging completion state: {config.output_root}"
        ) from exc
    terminal = config.output_root / "_pipeline/complete.json"
    try:
        terminal_sha256 = validate_sha256_sidecar(terminal)
    except ProvenanceError as exc:
        raise ExtensionImportError(
            f"Invalid staging completion receipt: {terminal}"
        ) from exc
    payload = _read_json(terminal, "staging completion receipt")
    config_path = config.output_root / "_pipeline/config.json"
    try:
        validate_sha256_sidecar(config_path)
    except ProvenanceError as exc:
        raise ExtensionImportError(f"Invalid staging workflow config: {config_path}") from exc
    stored = _read_json(config_path, "staging workflow config")
    scope = payload.get("scope")
    if not isinstance(scope, dict):
        raise ExtensionImportError("Staging completion scope is invalid")
    expected_staging = str(config.output_root.resolve())
    expected_extension = str(config.extension_root.resolve(strict=False))
    if (
        payload.get("status") != "complete"
        or payload.get("workflow") != "simnibs_segmentation_determinism"
        or payload.get("source_outputs_modified") is not False
        or payload.get("plan_sha256") != stored.get("plan_sha256")
        or scope.get("staging_root") != expected_staging
        or scope.get("future_extension_root") != expected_extension
        or scope.get("scientific_segmentation_tasks") != 20
        or payload.get("validated_repeats") != 20
    ):
        raise ExtensionImportError("Staging completion receipt binding is invalid")
    stored_config = stored.get("config")
    if (
        stored.get("workflow") != "simnibs_segmentation_determinism"
        or not isinstance(stored_config, dict)
        or stored_config.get("output_root") != str(config.output_root)
    ):
        raise ExtensionImportError("Staging workflow ownership is invalid")

    repeat_receipts = payload.get("repeat_receipts")
    if not isinstance(repeat_receipts, list) or len(repeat_receipts) != 20:
        raise ExtensionImportError("Staging repeat-receipt inventory is incomplete")
    observed_repeats = set()
    for record in repeat_receipts:
        if not isinstance(record, dict):
            raise ExtensionImportError("Staging repeat-receipt record is invalid")
        observed_repeats.add(_validate_repeat_receipt(config, record))
    expected_repeats = {
        (arm, repeat_index)
        for arm in ("simnibs326_headreco", "simnibs401_charm")
        for repeat_index in range(1, 11)
    }
    if observed_repeats != expected_repeats:
        raise ExtensionImportError("Staging repeat-receipt identities are incomplete")

    artifacts = payload.get("analysis_artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ExtensionImportError("Staging analysis-artifact inventory is incomplete")
    for name, digest in artifacts.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ExtensionImportError("Staging analysis-artifact name is invalid")
        _require_bound_file(config.output_root, config.output_root / "analysis" / name, digest)
    return {
        "path": str(terminal.resolve()),
        "sha256": terminal_sha256,
        "plan_sha256": payload["plan_sha256"],
    }


def _owner_payload(
    config: ExperimentConfig,
    staging: Mapping[str, object],
    dataset: Mapping[str, object],
) -> Dict[str, object]:
    completion = dataset.get("completion")
    if not isinstance(completion, dict):
        raise ExtensionImportError("Dataset completion binding is invalid")
    return {
        "schema_version": 1,
        "status": "importing",
        "workflow": "simnibs_segmentation_determinism_extension_import",
        "staging_root": str(config.output_root.resolve()),
        "extension_root": str(config.extension_root.resolve(strict=False)),
        "dataset_root": str(config.dataset_root.resolve()),
        "staging_completion_sha256": staging["sha256"],
        "dataset_completion_sha256": completion.get("sha256"),
        "plan_sha256": staging["plan_sha256"],
        "source_outputs_modified": False,
    }


def _destination_action(extension_root: Path, expected_owner: Mapping[str, object]) -> str:
    if not extension_root.exists() and not extension_root.is_symlink():
        return "create"
    if not extension_root.is_dir() or extension_root.is_symlink():
        raise ExtensionImportError(f"Extension destination is foreign: {extension_root}")
    complete = extension_root / "_import/complete.json"
    if complete.exists() or complete.is_symlink():
        raise ExtensionImportError(f"Extension destination is already complete: {extension_root}")
    owner_path = extension_root / "_import/owner.json"
    if not owner_path.is_file() or owner_path.is_symlink():
        raise ExtensionImportError(f"Extension destination is foreign: {extension_root}")
    try:
        validate_sha256_sidecar(owner_path)
    except ProvenanceError as exc:
        raise ExtensionImportError(
            f"Extension destination owner is invalid: {owner_path}"
        ) from exc
    observed = _read_json(owner_path, "extension import owner")
    if observed != expected_owner:
        raise ExtensionImportError(f"Extension destination ownership differs: {extension_root}")
    return "resume"


def install_preflight(config: ExperimentConfig) -> Dict[str, object]:
    validate_output_isolation(config)
    staging = _validate_staging_completion(config)
    try:
        dataset = validate_dataset_receipts(config)
    except ProvenanceError as exc:
        raise ExtensionImportError(str(exc)) from exc
    owner = _owner_payload(config, staging, dataset)
    action = _destination_action(config.extension_root, owner)
    return {
        "schema_version": 1,
        "status": "ready",
        "action": action,
        "workflow": "simnibs_segmentation_determinism_extension_import",
        "staging_root": str(config.output_root.resolve()),
        "extension_root": str(config.extension_root.resolve(strict=False)),
        "dataset_root": str(config.dataset_root.resolve()),
        "staging_completion_sha256": staging["sha256"],
        "dataset_completion_sha256": owner["dataset_completion_sha256"],
        "source_outputs_modified": False,
        "jobs_submitted": 0,
        "deletion_performed": False,
    }


def _run_rsync(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(arguments),
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise ExtensionImportError(f"Could not execute rsync: {exc}") from exc


def _copy_with_rsync(source: Path, destination: Path) -> None:
    result = _run_rsync(
        (
            "rsync",
            "-rltH",
            "--copy-links",
            "--partial",
            "--protect-args",
            "--checksum",
            "--exclude=/_import/",
            f"{source}/",
            f"{destination}/",
        )
    )
    if result.returncode != 0:
        raise ExtensionImportError(
            f"Extension rsync copy failed ({result.returncode}): {result.stderr.strip()}"
        )


def _checksum_changes(source: Path, destination: Path) -> list[str]:
    result = _run_rsync(
        (
            "rsync",
            "-rltH",
            "--copy-links",
            "--protect-args",
            "--dry-run",
            "--itemize-changes",
            "--out-format=%i|%n%L",
            "--delete",
            "--checksum",
            "--exclude=/_import/",
            f"{source}/",
            f"{destination}/",
        )
    )
    if result.returncode != 0:
        raise ExtensionImportError(
            "Extension checksum verification failed "
            f"({result.returncode}): {result.stderr.strip()}"
        )
    return [line for line in result.stdout.splitlines() if line.strip()]


def install_extension(
    config: ExperimentConfig, *, confirm_dataset_root: Path
) -> Dict[str, object]:
    confirmed = Path(confirm_dataset_root).expanduser().resolve(strict=False)
    expected_dataset = config.dataset_root.resolve(strict=False)
    if confirmed != expected_dataset:
        raise ExtensionImportError(
            f"Dataset-root confirmation differs: {confirmed} != {expected_dataset}"
        )
    ready = install_preflight(config)
    staging_sha256 = str(ready["staging_completion_sha256"])
    dataset = validate_dataset_receipts(config)
    staging = _validate_staging_completion(config)
    owner = _owner_payload(config, staging, dataset)

    import_root = config.extension_root / "_import"
    import_root.mkdir(parents=True, exist_ok=True)
    owner_path = import_root / "owner.json"
    if ready["action"] == "create":
        write_json_atomic(owner_path, owner)
        write_sha256_sidecar(owner_path)
    else:
        observed = _read_json(owner_path, "extension import owner")
        if observed != owner:
            raise ExtensionImportError("Extension import ownership changed")

    _copy_with_rsync(config.output_root, config.extension_root)
    changes = _checksum_changes(config.output_root, config.extension_root)
    if changes:
        preview = "; ".join(changes[:10])
        raise ExtensionImportError(
            f"Extension checksum verification found changes: {preview}"
        )
    final_staging = _validate_staging_completion(config)
    if final_staging != staging:
        raise ExtensionImportError("Staging receipt binding changed during import")
    final_dataset = validate_dataset_receipts(config)
    if final_dataset != dataset:
        raise ExtensionImportError("Dataset receipt binding changed during import")

    completion = {
        "schema_version": 1,
        "status": "complete",
        "workflow": "simnibs_segmentation_determinism_extension_import",
        "staging_root": str(config.output_root.resolve()),
        "extension_root": str(config.extension_root.resolve()),
        "dataset_root": str(config.dataset_root.resolve()),
        "staging_completion_sha256": staging_sha256,
        "dataset_completion_sha256": ready["dataset_completion_sha256"],
        "plan_sha256": staging["plan_sha256"],
        "checksum_verification": "matched",
        "source_outputs_modified": False,
        "deletion_performed": False,
        "jobs_submitted": 0,
    }
    completion_path = import_root / "complete.json"
    write_json_atomic(completion_path, completion)
    write_sha256_sidecar(completion_path)
    return completion
