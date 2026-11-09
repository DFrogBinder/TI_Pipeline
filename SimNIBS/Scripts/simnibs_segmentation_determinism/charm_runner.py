from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .provenance import (
    sha256_file,
    validate_sha256_sidecar,
    write_json_atomic,
    write_sha256_sidecar,
)
from .settings import ExperimentConfig, SubjectInputs


MAP_BASENAME = "tissue_labeling_upsampled.nii.gz"
APPROVED_FLAGS = ("--registerT2", "--initatlas", "--segment", "--forceqform")


class RepeatRunError(RuntimeError):
    """Raised when a CHARM repeat cannot be proven complete."""


class CharmTimeout(RepeatRunError):
    """Raised after terminating a CHARM process group at its deadline."""


@dataclass(frozen=True)
class CharmRepeatRequest:
    config: ExperimentConfig
    inputs: SubjectInputs
    repeat_index: int
    attempt: int = 1
    charm_bin: str = "charm"
    flags: Tuple[str, ...] = APPROVED_FLAGS
    timeout_seconds: int = 8 * 60 * 60

    def __post_init__(self) -> None:
        if not 1 <= self.repeat_index <= self.config.repeats_per_arm:
            raise ValueError("repeat_index is outside the configured repeat scope")
        if not 1 <= self.attempt <= self.config.max_retries + 1:
            raise ValueError("attempt exceeds the bounded retry scope")
        if self.inputs.subject != self.config.subject:
            raise ValueError("request subject differs from the canonical configured subject")

    @property
    def repeat_root(self) -> Path:
        return (
            self.config.output_root
            / "simnibs401_charm"
            / f"repeat_{self.repeat_index:03d}"
        )

    @property
    def command(self) -> List[str]:
        return [
            self.charm_bin,
            self.inputs.subject,
            str(self.inputs.t1),
            str(self.inputs.t2),
            *self.flags,
        ]


def run_with_timeout(
    command: List[str], *, cwd: Path, timeout_seconds: int, log_path: Path
) -> str:
    """Run CHARM in its own process group and retain a complete task log."""
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise CharmTimeout(f"CHARM timed out after {timeout_seconds} seconds") from exc
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if return_code != 0:
        raise RepeatRunError(f"CHARM returned nonzero exit status {return_code}")
    return text


def _assert_inputs_unchanged(request: CharmRepeatRequest) -> None:
    for label, path, expected in (
        ("T1", request.inputs.t1, request.inputs.t1_sha256),
        ("T2", request.inputs.t2, request.inputs.t2_sha256),
    ):
        if not path.is_file():
            raise RepeatRunError(f"{label} input disappeared: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise RepeatRunError(
                f"{label} input SHA-256 drift: expected {expected}, observed {observed}"
            )


def _assert_stage_boundary(request: CharmRepeatRequest, log_text: str) -> None:
    if request.flags != APPROVED_FLAGS:
        raise RepeatRunError(
            f"CHARM flags differ from the segmentation-only contract: {request.flags}"
        )
    forbidden = {"--surfaces", "--surface", "--mesh"}
    if forbidden.intersection(request.command):
        raise RepeatRunError("CHARM command requests forbidden surfaces or meshing")
    if re.search(r"\bcharm\b[^\n]*(?:--surfaces?|--mesh)\b", log_text, re.I):
        raise RepeatRunError("CHARM log contains forbidden surfaces or mesh invocation")


def _assert_recorded_command(recorded: List[str], request: CharmRepeatRequest) -> None:
    if len(recorded) != len(request.command):
        raise RepeatRunError("Receipt CHARM command has an unexpected length")
    if Path(str(recorded[0])).name != "charm":
        raise RepeatRunError("Receipt command does not use the CHARM executable")
    if recorded[1:] != request.command[1:]:
        raise RepeatRunError("Receipt command differs from the approved CHARM contract")


def _quarantine_incomplete_publication(request: CharmRepeatRequest) -> None:
    candidates = [
        request.repeat_root / MAP_BASENAME,
        request.repeat_root / "artifact_manifest.json",
        request.repeat_root / "artifact_manifest.json.sha256",
        request.repeat_root / "complete.json",
        request.repeat_root / "complete.json.sha256",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return
    if request.attempt <= 1:
        raise RepeatRunError("Refusing a non-fresh repeat attempt workspace")
    quarantine = request.repeat_root / f"_failed_publication_attempt_{request.attempt - 1:03d}"
    if quarantine.exists():
        raise RepeatRunError(f"Failed-publication quarantine already exists: {quarantine}")
    quarantine.mkdir()
    for path in existing:
        os.replace(str(path), str(quarantine / path.name))


def _find_single_map(work_root: Path, subject: str) -> Path:
    matches = sorted(work_root.glob(f"m2m_*/label_prep/{MAP_BASENAME}"))
    if not matches:
        raise RepeatRunError(f"CHARM did not create {MAP_BASENAME}")
    if len(matches) != 1:
        raise RepeatRunError(f"Multiple CHARM tissue maps found: {matches}")
    expected = work_root / f"m2m_{subject}" / "label_prep" / MAP_BASENAME
    if matches[0] != expected:
        raise RepeatRunError(
            f"CHARM tissue map uses a noncanonical subject root: {matches[0]}"
        )
    if matches[0].stat().st_size <= 0:
        raise RepeatRunError(f"CHARM tissue map is empty: {matches[0]}")
    return matches[0]


def _copy_atomic_verified(source: Path, destination: Path) -> str:
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        shutil.copy2(str(source), str(temporary))
        source_hash = sha256_file(source)
        copied_hash = sha256_file(temporary)
        if source_hash != copied_hash:
            raise RepeatRunError("Copied CHARM map differs from generated map")
        os.replace(str(temporary), str(destination))
        return source_hash
    finally:
        if temporary.exists():
            temporary.unlink()


def run_charm_repeat(request: CharmRepeatRequest) -> Dict[str, object]:
    _assert_inputs_unchanged(request)
    _assert_stage_boundary(request, "")
    repeat_root = request.repeat_root
    receipt_path = repeat_root / "complete.json"
    if receipt_path.exists():
        try:
            validate_sha256_sidecar(receipt_path)
        except RuntimeError:
            pass
        else:
            raise RepeatRunError(f"Repeat already has a valid completion receipt: {receipt_path}")
    repeat_root.mkdir(parents=True, exist_ok=True)
    work_root = repeat_root / f"_work_attempt_{request.attempt:03d}"
    destination = repeat_root / MAP_BASENAME
    _quarantine_incomplete_publication(request)
    if work_root.exists() or destination.exists():
        raise RepeatRunError("Refusing a non-fresh repeat attempt workspace")
    work_root.mkdir()
    logs_root = repeat_root / "logs"
    logs_root.mkdir(exist_ok=True)
    log_path = logs_root / f"attempt_{request.attempt:03d}.log"

    try:
        log_text = run_with_timeout(
            request.command,
            cwd=work_root,
            timeout_seconds=request.timeout_seconds,
            log_path=log_path,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RepeatRunError(f"CHARM execution failed: {exc}") from exc
    _assert_stage_boundary(request, log_text)
    _assert_inputs_unchanged(request)
    generated = _find_single_map(work_root, request.inputs.subject)
    generated_hash = _copy_atomic_verified(generated, destination)

    artifact = {
        "relative_path": MAP_BASENAME,
        "size_bytes": destination.stat().st_size,
        "sha256": generated_hash,
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "arm": "simnibs401_charm",
        "stage": "segment",
        "subject": request.inputs.subject,
        "repeat_index": request.repeat_index,
        "attempt": request.attempt,
        "artifact": artifact,
    }
    manifest_path = repeat_root / "artifact_manifest.json"
    manifest_sha256 = write_json_atomic(manifest_path, manifest)
    write_sha256_sidecar(manifest_path)
    shutil.rmtree(work_root)
    receipt = {
        **manifest,
        "module": request.config.simnibs401_module,
        "command": request.command,
        "inputs": {
            "t1": {"path": str(request.inputs.t1), "sha256": request.inputs.t1_sha256},
            "t2": {"path": str(request.inputs.t2), "sha256": request.inputs.t2_sha256},
        },
        "generated_map_sha256": generated_hash,
        "archived_map_sha256": sha256_file(destination),
        "log": str(log_path),
        "log_sha256": sha256_file(log_path),
        "artifact_manifest": str(manifest_path),
        "artifact_manifest_sha256": manifest_sha256,
        "surfaces_requested": False,
        "mesh_requested": False,
        "simulation_requested": False,
        "temporary_charm_outputs_deleted": not work_root.exists(),
        "source_outputs_modified": False,
    }
    write_json_atomic(receipt_path, receipt)
    write_sha256_sidecar(receipt_path)
    return receipt


def validate_charm_repeat(request: CharmRepeatRequest) -> Dict[str, object]:
    receipt_path = request.repeat_root / "complete.json"
    if not receipt_path.is_file():
        raise RepeatRunError(f"Missing completion receipt: {receipt_path}")
    try:
        validate_sha256_sidecar(receipt_path)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, RuntimeError) as exc:
        raise RepeatRunError(f"Invalid completion receipt: {receipt_path}") from exc
    expected_fields = {
        "status": "complete",
        "arm": "simnibs401_charm",
        "stage": "segment",
        "subject": request.inputs.subject,
        "repeat_index": request.repeat_index,
        "module": request.config.simnibs401_module,
        "surfaces_requested": False,
        "mesh_requested": False,
        "simulation_requested": False,
    }
    for field, expected in expected_fields.items():
        if receipt.get(field) != expected:
            raise RepeatRunError(
                f"Receipt {field} mismatch: expected {expected!r}, observed {receipt.get(field)!r}"
            )
    if receipt.get("attempt") not in range(1, request.config.max_retries + 2):
        raise RepeatRunError("Receipt attempt is outside the bounded retry scope")
    recorded_command = receipt.get("command")
    if not isinstance(recorded_command, list):
        raise RepeatRunError("Receipt CHARM command is missing")
    _assert_recorded_command(recorded_command, request)
    _assert_stage_boundary(request, "")
    for modality, expected_path, expected_hash in (
        ("t1", request.inputs.t1, request.inputs.t1_sha256),
        ("t2", request.inputs.t2, request.inputs.t2_sha256),
    ):
        observed = receipt.get("inputs", {}).get(modality, {})
        if observed.get("path") != str(expected_path) or observed.get("sha256") != expected_hash:
            raise RepeatRunError(f"Receipt {modality.upper()} provenance mismatch")
    artifact = receipt.get("artifact")
    if not isinstance(artifact, dict) or artifact.get("relative_path") != MAP_BASENAME:
        raise RepeatRunError("Receipt CHARM artifact is invalid")
    path = request.repeat_root / MAP_BASENAME
    if not path.is_file() or path.stat().st_size != artifact.get("size_bytes"):
        raise RepeatRunError("Promoted CHARM map is missing or has the wrong size")
    observed_hash = sha256_file(path)
    if observed_hash != artifact.get("sha256"):
        raise RepeatRunError("Promoted CHARM map SHA-256 mismatch")
    manifest_path = request.repeat_root / "artifact_manifest.json"
    try:
        validate_sha256_sidecar(manifest_path)
    except RuntimeError as exc:
        raise RepeatRunError("Artifact manifest sidecar is invalid") from exc
    if sha256_file(manifest_path) != receipt.get("artifact_manifest_sha256"):
        raise RepeatRunError("Artifact manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact") != artifact:
        raise RepeatRunError("Artifact manifest differs from receipt")
    attempt = int(receipt["attempt"])
    expected_log = request.repeat_root / f"logs/attempt_{attempt:03d}.log"
    if receipt.get("log") != str(expected_log) or not expected_log.is_file():
        raise RepeatRunError("Retained log path is missing or unexpected")
    if sha256_file(expected_log) != receipt.get("log_sha256"):
        raise RepeatRunError("Retained log SHA-256 mismatch")
    _assert_stage_boundary(request, expected_log.read_text(encoding="utf-8", errors="replace"))
    if receipt.get("temporary_charm_outputs_deleted") is not True:
        raise RepeatRunError("Receipt does not prove CHARM temporary-output cleanup")
    if (request.repeat_root / f"_work_attempt_{attempt:03d}").exists():
        raise RepeatRunError("CHARM temporary-output directory still exists")
    return receipt
