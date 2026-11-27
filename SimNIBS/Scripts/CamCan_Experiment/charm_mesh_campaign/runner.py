"""Execute and validate configuration-driven CHARM meshing tasks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from charm_segmentation_batch import mesh_collected_segmentations as legacy

from .config import CampaignConfig
from .tasks import TaskRecord, read_task_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task(config: CampaignConfig, task_index: int) -> TaskRecord:
    tasks = read_task_manifest(config.output.task_manifest)
    if task_index < 0 or task_index >= len(tasks):
        raise IndexError(
            f"task index {task_index} is outside manifest range 0..{len(tasks) - 1}"
        )
    task = tasks[task_index]
    if task.task_id != task_index:
        raise ValueError(
            f"task manifest id {task.task_id} does not match index {task_index}"
        )
    if task.status != "ready":
        raise ValueError(f"task {task_index} is not ready: {task.message}")
    if task.config_sha256 != config.sha256:
        raise ValueError("task manifest configuration hash differs from active config")
    if not task.label_path.is_file():
        raise FileNotFoundError(task.label_path)
    if task.label_path.stat().st_size != task.label_bytes:
        raise ValueError("task label byte count changed after preflight")
    if _sha256(task.label_path) != task.label_sha256:
        raise ValueError("task label SHA-256 changed after preflight")
    for output in (task.mesh_path, task.result_path):
        try:
            output.resolve().relative_to(config.output.root.resolve())
        except ValueError as exc:
            raise ValueError(f"task output is outside campaign root: {output}") from exc
    return task


def run_task(
    config: CampaignConfig,
    task_index: int,
    staging_root: str | Path | None = None,
) -> dict[str, object]:
    """Run exactly one normalized segmentation-to-mesh task."""

    task = _task(config, task_index)
    if staging_root is None:
        resolved_staging = config.output.root / ".staging"
        resolved_staging.mkdir(parents=True, exist_ok=True)
    else:
        resolved_staging = Path(staging_root).expanduser().resolve(strict=True)
    provenance = {
        "campaign_id": config.campaign_id,
        "config_sha256": config.sha256,
        "task_key": task.task_key,
        "realization_id": task.realization_id,
        "simnibs_module": config.software.module,
        "simnibs_python": config.software.python,
        "segmentation_version": config.input.segmentation_version,
        "staging_mode": "node_local" if staging_root is not None else "output_root",
    }
    return legacy.create_mesh_from_label(
        subject=task.subject_id,
        label_path=task.label_path,
        expected_label_hash=task.label_sha256,
        mesh_path=task.mesh_path,
        result_path=task.result_path,
        task_index=task.task_id,
        settings_path=config.software.settings_path,
        staging_root=resolved_staging,
        provenance=provenance,
        expected_simnibs_version=config.software.expected_version,
        required_tissue_tags=config.quality.required_tissue_tags,
        reuse_validated=config.resume.reuse_validated,
        overwrite=config.resume.overwrite,
    )


def validate_campaign(
    config: CampaignConfig, *, verify_hashes: bool = True
) -> dict[str, object]:
    """Validate task results without modifying source maps or meshes."""

    tasks = read_task_manifest(config.output.task_manifest)
    failures: list[dict[str, object]] = []
    complete = 0
    for task in tasks:
        messages: list[str] = []
        if task.config_sha256 != config.sha256:
            messages.append("task config hash mismatch")
        if not task.result_path.is_file():
            messages.append("result missing")
        else:
            try:
                result = json.loads(task.result_path.read_text(encoding="utf-8"))
                expected = {
                    "status": "complete",
                    "subject": task.subject_id,
                    "realization_id": task.realization_id,
                    "task_key": task.task_key,
                    "config_sha256": config.sha256,
                    "label_sha256_after": task.label_sha256,
                    "mesh_path": str(task.mesh_path),
                    "expected_simnibs_version": config.software.expected_version,
                }
                for key, value in expected.items():
                    if result.get(key) != value:
                        messages.append(f"result {key} mismatch")
                if not task.mesh_path.is_file() or task.mesh_path.stat().st_size <= 0:
                    messages.append("mesh missing or empty")
                elif task.mesh_path.stat().st_size != result.get("mesh_bytes"):
                    messages.append("mesh byte count mismatch")
                elif verify_hashes and _sha256(task.mesh_path) != result.get("mesh_sha256"):
                    messages.append("mesh SHA-256 mismatch")
                if verify_hashes and _sha256(task.label_path) != task.label_sha256:
                    messages.append("label SHA-256 mismatch")
                settings_path = Path(str(result.get("settings_path", "")))
                if verify_hashes:
                    if not settings_path.is_file():
                        messages.append("settings file missing")
                    elif _sha256(settings_path) != result.get("settings_sha256"):
                        messages.append("settings SHA-256 mismatch")
                actual_version = legacy._normalized_simnibs_version(
                    result.get("simnibs_version", "unknown")
                )
                expected_version = legacy._normalized_simnibs_version(
                    config.software.expected_version
                )
                if actual_version != expected_version:
                    messages.append("result SimNIBS version mismatch")
            except (OSError, ValueError, TypeError) as exc:
                messages.append(str(exc))
        if messages:
            failures.append({"task_key": task.task_key, "messages": messages})
        else:
            complete += 1
    return {
        "status": "complete" if complete == len(tasks) else "incomplete",
        "tasks": len(tasks),
        "complete": complete,
        "incomplete": len(tasks) - complete,
        "hashes_verified": verify_hashes,
        "failures": failures,
    }
