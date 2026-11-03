from __future__ import annotations

import json

import pytest

from repeatability_combined_analysis.receipts import (
    ReceiptError,
    validate_terminal_receipts,
)


def test_v3_and_v4_terminal_receipts_normalize_to_expected_scope(
    analysis_config,
) -> None:
    receipts = validate_terminal_receipts(analysis_config)

    assert len(receipts) == 10
    assert {receipt["status"] for receipt in receipts} == {"complete"}
    assert {receipt["target"] for receipt in receipts} == {
        "left_hippocampus",
        "right_m1",
    }
    assert {receipt["repeats_per_condition"] for receipt in receipts} == {40}
    assert all(len(str(receipt["sha256"])) == 64 for receipt in receipts)
    assert all(receipt["path"].endswith("complete.json") for receipt in receipts)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing", "missing"),
        ("malformed", "JSON"),
        ("incomplete", "complete"),
        ("wrong_subject", "subjects"),
        ("wrong_repeats", "repeat"),
        ("wrong_target", "target"),
    ],
)
def test_invalid_terminal_receipts_fail_closed(
    analysis_config,
    mutation: str,
    match: str,
) -> None:
    path = analysis_config.sources.receipts[0]
    if mutation == "missing":
        path.unlink()
    elif mutation == "malformed":
        path.write_text("not-json", encoding="utf-8")
    else:
        payload = json.loads(path.read_text())
        if mutation == "incomplete":
            payload["status"] = "running"
        elif mutation == "wrong_subject":
            payload["scope"]["subjects"][0] = "sub-CC999999"
        elif mutation == "wrong_repeats":
            payload["scope"]["repeats_per_condition"] = 39
        elif mutation == "wrong_target":
            payload["scope"]["target"] = "right_m1"
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReceiptError, match=match):
        validate_terminal_receipts(analysis_config)


def test_v4_receipt_task_totals_must_match_subjects_times_repeats(
    analysis_config,
) -> None:
    path = analysis_config.sources.receipts[4]
    payload = json.loads(path.read_text())
    payload["scope"]["remesh_tasks"] = 399
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReceiptError, match="task count"):
        validate_terminal_receipts(analysis_config)
