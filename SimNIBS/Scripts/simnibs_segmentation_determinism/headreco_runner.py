from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from .provenance import sha256_file, write_json_atomic, write_sha256_sidecar
from .settings import ExperimentConfig, SubjectInputs


class RepeatRunError(RuntimeError):
    """Raised when a repeat cannot be proven complete and stage-bounded."""


FORBIDDEN_LOG_PATTERNS = (
    re.compile(r"\bheadreco\s+(?:all|preparecat|cleanvols|surfacemesh|volumemesh)\b", re.I),
    re.compile(r"PREPARING SURFACES FROM CAT12", re.I),
    re.compile(r"CLEANING SEGMENTATIONS", re.I),
    re.compile(r"CREATING VOLUME MESH", re.I),
)


def expected_headreco_artifacts(subject: str) -> List[str]:
    del subject  # Names are canonical because the working m2m root already carries the subject ID.
    artifacts = [
        "T1fs_conform.nii.gz",
        "T1fs_nu_conform.nii.gz",
        "T2_conform.nii.gz",
    ]
    artifacts.extend(f"segment/spm/c{index}T1fs_conform.nii" for index in range(1, 7))
    artifacts.extend(
        f"segment/spm/binarized/c{index}T1fs_conform_bin.nii.gz" for index in range(1, 7)
    )
    artifacts.extend(f"segment/cat/mri/p{index}T1fs_conform.nii" for index in range(1, 4))
    artifacts.extend(
        f"segment/cat/binarized/p{index}T1fs_conform_bin.nii.gz" for index in range(1, 4)
    )
    return artifacts


@dataclass(frozen=True)
class HeadrecoRepeatRequest:
    config: ExperimentConfig
    inputs: SubjectInputs
    repeat_index: int
    attempt: int = 1
    compatibility_script: Path = Path("simnibs_326_repeatability/cat12_compat.py")
    python_executable: Path = Path("python3")
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
            / "simnibs326_headreco"
            / f"repeat_{self.repeat_index:03d}"
        )

    @property
    def command(self) -> List[str]:
        return [
            str(self.python_executable),
            str(self.compatibility_script),
            "headreco",
            "preparevols",
            "--noclean",
            self.inputs.subject,
            str(self.inputs.t1),
            str(self.inputs.t2),
        ]


def _assert_inputs_unchanged(request: HeadrecoRepeatRequest) -> None:
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


def _assert_stage_boundary(command: Sequence[str], log_text: str) -> None:
    expected_tail = ["headreco", "preparevols", "--noclean"]
    if list(command[2:5]) != expected_tail:
        raise RepeatRunError(f"Unexpected headreco stage command: {list(command)}")
    combined = " ".join(command) + "\n" + log_text
    for pattern in FORBIDDEN_LOG_PATTERNS:
        if pattern.search(combined):
            raise RepeatRunError(
                f"Forbidden downstream headreco stage evidence: {pattern.pattern}"
            )


def _discover_artifacts(subject_root: Path, subject: str) -> List[Path]:
    required = {Path(relative) for relative in expected_headreco_artifacts(subject)}
    missing = sorted(str(path) for path in required if not (subject_root / path).is_file())
    if missing:
        raise RepeatRunError("Missing required preparevols artifacts: " + ", ".join(missing))

    allowed = set(required)
    # CAT revisions may additionally expose p0 or further numeric posterior maps.
    for parent in (Path("segment/cat/mri"), Path("segment/cat/binarized")):
        directory = subject_root / parent
        if directory.is_dir():
            for path in directory.glob("p*T1fs_conform*.nii*"):
                if re.fullmatch(r"p\d+T1fs_conform(?:_bin)?\.nii(?:\.gz)?", path.name):
                    allowed.add(path.relative_to(subject_root))

    observed_primary = set()
    for path in subject_root.glob("*conform*.nii*"):
        observed_primary.add(path.relative_to(subject_root))
    for pattern in (
        "segment/spm/c*T1fs_conform*.nii*",
        "segment/spm/binarized/c*T1fs_conform*.nii*",
        "segment/cat/mri/p*T1fs_conform*.nii*",
        "segment/cat/binarized/p*T1fs_conform*.nii*",
    ):
        observed_primary.update(path.relative_to(subject_root) for path in subject_root.glob(pattern))
    unexpected = sorted(str(path) for path in observed_primary - allowed)
    if unexpected:
        raise RepeatRunError("Unexpected preparevols artifact inventory: " + ", ".join(unexpected))

    selected = sorted(subject_root / path for path in allowed)
    empty = [str(path.relative_to(subject_root)) for path in selected if path.stat().st_size == 0]
    if empty:
        raise RepeatRunError("Empty preparevols artifacts: " + ", ".join(empty))
    return selected


def _artifact_records(subject_root: Path, promoted_root: Path, sources: Sequence[Path]) -> List[Dict[str, object]]:
    records = []
    for source in sources:
        relative = source.relative_to(subject_root)
        destination = promoted_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(destination))
        records.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
    return sorted(records, key=lambda record: str(record["relative_path"]))


def run_headreco_repeat(request: HeadrecoRepeatRequest) -> Dict[str, object]:
    _assert_inputs_unchanged(request)
    repeat_root = request.repeat_root
    receipt_path = repeat_root / "complete.json"
    if receipt_path.exists():
        raise RepeatRunError(f"Repeat already has a completion receipt: {receipt_path}")
    repeat_root.mkdir(parents=True, exist_ok=True)
    work_root = repeat_root / f"_work_attempt_{request.attempt:03d}"
    staging_root = repeat_root / f".artifacts_attempt_{request.attempt:03d}"
    artifacts_root = repeat_root / "artifacts"
    if work_root.exists() or staging_root.exists() or artifacts_root.exists():
        raise RepeatRunError("Refusing a non-fresh repeat attempt workspace")
    work_root.mkdir()
    logs_root = repeat_root / "logs"
    logs_root.mkdir(exist_ok=True)
    log_path = logs_root / f"attempt_{request.attempt:03d}.log"

    try:
        result = subprocess.run(
            request.command,
            cwd=str(work_root),
            text=True,
            capture_output=True,
            check=False,
            timeout=request.timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RepeatRunError(f"headreco execution failed: {exc}") from exc

    log_text = (result.stdout or "") + (result.stderr or "")
    log_path.write_text(log_text, encoding="utf-8")
    if result.returncode != 0:
        raise RepeatRunError(f"headreco returned nonzero exit status {result.returncode}")
    _assert_stage_boundary(request.command, log_text)
    _assert_inputs_unchanged(request)

    subject_root = work_root / f"m2m_{request.inputs.subject}"
    if not subject_root.is_dir():
        raise RepeatRunError(f"Missing headreco subject directory: {subject_root}")
    sources = _discover_artifacts(subject_root, request.inputs.subject)
    staging_root.mkdir()
    records = _artifact_records(subject_root, staging_root, sources)
    os.replace(str(staging_root), str(artifacts_root))

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "arm": "simnibs326_headreco",
        "stage": "preparevols",
        "subject": request.inputs.subject,
        "repeat_index": request.repeat_index,
        "attempt": request.attempt,
        "artifacts": records,
    }
    manifest_path = repeat_root / "artifact_manifest.json"
    manifest_sha256 = write_json_atomic(manifest_path, manifest)
    write_sha256_sidecar(manifest_path)
    receipt = {
        **manifest,
        "module": request.config.simnibs326_module,
        "matlab_module": request.config.matlab_module,
        "command": request.command,
        "inputs": {
            "t1": {"path": str(request.inputs.t1), "sha256": request.inputs.t1_sha256},
            "t2": {"path": str(request.inputs.t2), "sha256": request.inputs.t2_sha256},
        },
        "log": str(log_path),
        "log_sha256": sha256_file(log_path),
        "artifact_manifest": str(manifest_path),
        "artifact_manifest_sha256": manifest_sha256,
        "forbidden_stage_evidence": [],
        "source_outputs_modified": False,
    }
    write_json_atomic(receipt_path, receipt)
    write_sha256_sidecar(receipt_path)
    return receipt


def validate_headreco_repeat(request: HeadrecoRepeatRequest) -> Dict[str, object]:
    receipt_path = request.repeat_root / "complete.json"
    if not receipt_path.is_file():
        raise RepeatRunError(f"Missing completion receipt: {receipt_path}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RepeatRunError(f"Invalid completion receipt: {receipt_path}") from exc
    expected_fields = {
        "status": "complete",
        "arm": "simnibs326_headreco",
        "stage": "preparevols",
        "subject": request.inputs.subject,
        "repeat_index": request.repeat_index,
        "module": request.config.simnibs326_module,
        "matlab_module": request.config.matlab_module,
    }
    for field, expected in expected_fields.items():
        if receipt.get(field) != expected:
            raise RepeatRunError(
                f"Receipt {field} mismatch: expected {expected!r}, observed {receipt.get(field)!r}"
            )
    if receipt.get("attempt") not in range(1, request.config.max_retries + 2):
        raise RepeatRunError("Receipt attempt is outside the bounded retry scope")
    if receipt.get("command") != request.command:
        raise RepeatRunError("Receipt command differs from the approved preparevols command")
    _assert_stage_boundary(receipt["command"], "")
    for modality, expected_path, expected_hash in (
        ("t1", request.inputs.t1, request.inputs.t1_sha256),
        ("t2", request.inputs.t2, request.inputs.t2_sha256),
    ):
        observed = receipt.get("inputs", {}).get(modality, {})
        if observed.get("path") != str(expected_path) or observed.get("sha256") != expected_hash:
            raise RepeatRunError(f"Receipt {modality.upper()} provenance mismatch")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RepeatRunError("Receipt artifact inventory is missing")
    for record in artifacts:
        path = request.repeat_root / "artifacts" / record["relative_path"]
        if not path.is_file():
            raise RepeatRunError(f"Promoted artifact is missing: {path}")
        if path.stat().st_size != record.get("size_bytes"):
            raise RepeatRunError(f"Promoted artifact size mismatch: {path}")
        observed_hash = sha256_file(path)
        if observed_hash != record.get("sha256"):
            raise RepeatRunError(f"Promoted artifact SHA-256 mismatch: {path}")
    manifest_path = request.repeat_root / "artifact_manifest.json"
    if sha256_file(manifest_path) != receipt.get("artifact_manifest_sha256"):
        raise RepeatRunError("Artifact manifest SHA-256 mismatch")
    return receipt
