"""Normalize and validate heterogeneous terminal campaign receipts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .settings import (
    EXCLUDED_SUBJECT,
    REPLACEMENT_SUBJECT,
    AnalysisConfig,
)


class ReceiptError(RuntimeError):
    """Raised when a terminal receipt cannot prove its assigned scope."""


@dataclass(frozen=True)
class _Expectation:
    target: str
    subjects: tuple[str, ...]
    conditions: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expectations(config: AnalysisConfig) -> tuple[_Expectation, ...]:
    base9 = tuple(subject for subject in config.subjects if subject != REPLACEMENT_SUBJECT)
    original10 = (*base9, EXCLUDED_SUBJECT)
    replacement = (REPLACEMENT_SUBJECT,)
    both = ("remesh", "fixed_mesh")
    return (
        _Expectation("left_hippocampus", base9, both),
        _Expectation("right_m1", base9, both),
        _Expectation("left_hippocampus", replacement, both),
        _Expectation("right_m1", replacement, both),
        _Expectation("left_hippocampus", original10, ("remesh",)),
        _Expectation("right_m1", original10, ("remesh",)),
        _Expectation("left_hippocampus", original10, ("fixed_mesh",)),
        _Expectation("right_m1", original10, ("fixed_mesh",)),
        _Expectation("left_hippocampus", replacement, both),
        _Expectation("right_m1", replacement, both),
    )


def _normalize_target(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "left_hippocampus": "left_hippocampus",
        "right_m1": "right_m1",
    }
    return aliases.get(text, text)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReceiptError(f"Terminal receipt is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReceiptError(f"Terminal receipt is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ReceiptError(f"Terminal receipt JSON is not an object: {path}")
    return payload


def validate_terminal_receipts(
    config: AnalysisConfig,
) -> list[dict[str, object]]:
    expectations = _expectations(config)
    if len(config.sources.receipts) != len(expectations):
        raise ReceiptError(
            f"Expected {len(expectations)} terminal receipts, "
            f"observed {len(config.sources.receipts)}"
        )
    normalized: list[dict[str, object]] = []
    for path, expected in zip(config.sources.receipts, expectations):
        payload = _load(path)
        if payload.get("status") != "complete":
            raise ReceiptError(f"Terminal receipt is not complete: {path}")
        validation = payload.get("validation")
        if isinstance(validation, dict) and validation.get("status") not in (
            None,
            "complete",
        ):
            raise ReceiptError(f"Terminal receipt validation is not complete: {path}")
        scope = payload.get("scope")
        if not isinstance(scope, dict):
            raise ReceiptError(f"Terminal receipt has no scope: {path}")
        observed_target = _normalize_target(scope.get("target", scope.get("roi")))
        if observed_target != expected.target:
            raise ReceiptError(
                f"Terminal receipt target differs: expected {expected.target}, "
                f"observed {observed_target or 'missing'} in {path}"
            )
        try:
            repeats = int(scope.get("repeats_per_condition", -1))
        except (TypeError, ValueError) as exc:
            raise ReceiptError(f"Invalid repeat count in {path}") from exc
        if repeats != config.repeat_count:
            raise ReceiptError(
                f"Terminal receipt repeat count differs: expected "
                f"{config.repeat_count}, observed {repeats} in {path}"
            )
        try:
            subject_count = int(scope.get("subject_count", -1))
        except (TypeError, ValueError) as exc:
            raise ReceiptError(f"Invalid subject count in {path}") from exc
        if subject_count != len(expected.subjects):
            raise ReceiptError(
                f"Terminal receipt subject count differs in {path}"
            )
        observed_subjects = scope.get("subjects")
        if observed_subjects is not None and set(map(str, observed_subjects)) != set(
            expected.subjects
        ):
            raise ReceiptError(f"Terminal receipt subjects differ in {path}")

        condition_counts: dict[str, int] = {}
        expected_tasks = repeats * subject_count
        conditions_payload = scope.get("conditions")
        if isinstance(conditions_payload, dict):
            for condition, value in conditions_payload.items():
                try:
                    if isinstance(value, bool):
                        raise ValueError("boolean is not a count")
                    observed_count = int(value)
                    if isinstance(value, float) and not value.is_integer():
                        raise ValueError("fractional value is not a count")
                except (TypeError, ValueError) as exc:
                    raise ReceiptError(
                        f"Invalid {condition} condition count in {path}"
                    ) from exc
                if observed_count not in (repeats, expected_tasks):
                    raise ReceiptError(
                        f"Terminal receipt {condition} condition count differs: "
                        f"expected {repeats} repeats or {expected_tasks} total tasks, "
                        f"observed {observed_count} in {path}"
                    )
                condition_counts[str(condition)] = repeats
        for condition in ("remesh", "fixed_mesh"):
            task_key = f"{condition}_tasks"
            if task_key in scope:
                task_count = int(scope[task_key])
                if task_count > 0:
                    if task_count != expected_tasks:
                        raise ReceiptError(
                            f"Terminal receipt {condition} task count differs: "
                            f"expected {expected_tasks}, observed {task_count} in {path}"
                        )
                    condition_counts.setdefault(condition, repeats)
        missing_conditions = set(expected.conditions).difference(condition_counts)
        if missing_conditions:
            raise ReceiptError(
                f"Terminal receipt does not prove conditions "
                f"{sorted(missing_conditions)} in {path}"
            )
        if any(condition_counts[condition] != repeats for condition in expected.conditions):
            raise ReceiptError(f"Terminal receipt condition repeat scope differs in {path}")

        normalized.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "status": "complete",
                "target": expected.target,
                "subjects": list(expected.subjects),
                "subject_count": len(expected.subjects),
                "conditions": list(expected.conditions),
                "repeats_per_condition": repeats,
            }
        )
    return normalized
