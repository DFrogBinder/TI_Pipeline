from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


SUBJECT_LEVEL_STAGE = "subject_level"
POPULATION_WITHIN_RUN_STAGE = "population_within_run"
ACROSS_REPEATS_STAGE = "across_repeats"
FIGURE_GENERATION_STAGE = "figure_generation"

PIPELINE_STAGE_ORDER = (
    SUBJECT_LEVEL_STAGE,
    POPULATION_WITHIN_RUN_STAGE,
    ACROSS_REPEATS_STAGE,
    FIGURE_GENERATION_STAGE,
)

PIPELINE_STAGE_LABELS = {
    SUBJECT_LEVEL_STAGE: "Subject-level metrics",
    POPULATION_WITHIN_RUN_STAGE: "Population (within run)-level metrics",
    ACROSS_REPEATS_STAGE: "Across-repeats-level metrics",
    FIGURE_GENERATION_STAGE: "Post-processing figure generation",
}


@dataclass
class PipelineStageResult:
    stage: str
    status: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {"stage": self.stage, "status": self.status}
        payload.update(self.details)
        return payload


def stage_ok(stage: str, **details: Any) -> Dict[str, Any]:
    return PipelineStageResult(stage=stage, status="ok", details=details).to_dict()


def stage_skipped(stage: str, *, reason: str, **details: Any) -> Dict[str, Any]:
    return PipelineStageResult(
        stage=stage,
        status="skipped",
        details={"reason": reason, **details},
    ).to_dict()


def stage_failed(stage: str, *, error: str, **details: Any) -> Dict[str, Any]:
    return PipelineStageResult(
        stage=stage,
        status="failed",
        details={"error": error, **details},
    ).to_dict()


def stage_partial(stage: str, *, warning: str, **details: Any) -> Dict[str, Any]:
    return PipelineStageResult(
        stage=stage,
        status="partial",
        details={"warning": warning, **details},
    ).to_dict()


def load_subject_metrics_payload(path: Path) -> Dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def subject_metrics_payload_status(payload: Dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    subject_meta = payload.get("subject_metrics_meta")
    if isinstance(subject_meta, dict):
        status = subject_meta.get("status")
        return str(status) if isinstance(status, str) else None
    meta = payload.get("extended_metrics_meta")
    if not isinstance(meta, dict):
        return None
    status = meta.get("status")
    return str(status) if isinstance(status, str) else None


def subject_metrics_payload_complete(payload: Dict[str, Any] | None) -> bool:
    return subject_metrics_payload_status(payload) == "complete"


def subject_metrics_file_complete(path: Path) -> bool:
    return subject_metrics_payload_complete(load_subject_metrics_payload(path))


def _subject_metrics_extended_status(payload: Dict[str, Any]) -> str | None:
    subject_meta = payload.get("subject_metrics_meta")
    if isinstance(subject_meta, dict):
        status = subject_meta.get("extended_metrics_status")
        if isinstance(status, str):
            return status

    meta = payload.get("extended_metrics_meta")
    if isinstance(meta, dict):
        status = meta.get("status")
        if isinstance(status, str):
            return status
    return None


def _subject_metrics_blocking_qc_checks(payload: Dict[str, Any]) -> list[str] | None:
    qc_meta = payload.get("qc_meta")
    if not isinstance(qc_meta, dict):
        return None

    checks = qc_meta.get("error_checks")
    if not isinstance(checks, list):
        return None

    return [str(check) for check in checks if str(check) != "overlays"]


def subject_metrics_payload_analysis_complete(payload: Dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False

    if subject_metrics_payload_complete(payload):
        return True

    if _subject_metrics_extended_status(payload) != "complete":
        return False

    blocking_qc_checks = _subject_metrics_blocking_qc_checks(payload)
    return blocking_qc_checks is not None and not blocking_qc_checks


def subject_metrics_file_analysis_complete(path: Path) -> bool:
    return subject_metrics_payload_analysis_complete(load_subject_metrics_payload(path))
