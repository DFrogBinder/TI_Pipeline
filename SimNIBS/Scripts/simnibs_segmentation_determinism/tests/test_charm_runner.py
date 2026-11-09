from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import simnibs_segmentation_determinism.charm_runner as charm_runner
from simnibs_segmentation_determinism.charm_runner import (
    CharmRepeatRequest,
    RepeatRunError,
    run_charm_repeat,
    validate_charm_repeat,
)
from simnibs_segmentation_determinism.provenance import sha256_file
from simnibs_segmentation_determinism.settings import resolve_subject_inputs


def _request(config, repeat_index=1, attempt=1):
    return CharmRepeatRequest(
        config=config,
        inputs=resolve_subject_inputs(config),
        repeat_index=repeat_index,
        attempt=attempt,
        charm_bin="/module/bin/charm",
    )


def _write_map(cwd: Path, subject="sub-CC110174", payload=b"charm map"):
    path = cwd / f"m2m_{subject}" / "label_prep/tissue_labeling_upsampled.nii.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _successful_fake(calls, log_text="segmentation complete\n"):
    def fake(command, *, cwd, timeout_seconds, log_path):
        calls.append((command, cwd, timeout_seconds, log_path))
        _write_map(Path(cwd))
        Path(log_path).write_text(log_text, encoding="utf-8")
        return log_text

    return fake


def test_command_is_exact_segmentation_only_invocation(configured_experiment, monkeypatch):
    request = _request(configured_experiment)
    calls = []
    monkeypatch.setattr(charm_runner, "run_with_timeout", _successful_fake(calls))

    receipt = run_charm_repeat(request)

    assert calls[0][0] == [
        "/module/bin/charm",
        "sub-CC110174",
        str(request.inputs.t1),
        str(request.inputs.t2),
        "--registerT2",
        "--initatlas",
        "--segment",
        "--forceqform",
    ]
    assert "--surfaces" not in calls[0][0]
    assert "--mesh" not in calls[0][0]
    assert receipt["command"] == calls[0][0]
    assert receipt["module"] == "SimNIBS/4.0.1-foss-2023a"


def test_ten_requests_keep_subject_constant_and_repeat_roots_unique(configured_experiment):
    requests = [_request(configured_experiment, repeat_index=index) for index in range(1, 11)]

    assert {request.inputs.subject for request in requests} == {"sub-CC110174"}
    assert len({request.repeat_root for request in requests}) == 10
    assert requests[0].repeat_root.name == "repeat_001"
    assert requests[-1].repeat_root.name == "repeat_010"


def test_only_nonempty_tissue_map_is_promoted_byte_identically(
    configured_experiment, monkeypatch
):
    request = _request(configured_experiment)
    monkeypatch.setattr(charm_runner, "run_with_timeout", _successful_fake([]))

    receipt = run_charm_repeat(request)
    validated = validate_charm_repeat(request)

    destination = request.repeat_root / "tissue_labeling_upsampled.nii.gz"
    assert destination.is_file()
    assert sha256_file(destination) == receipt["artifact"]["sha256"]
    assert validated == receipt
    assert not (request.repeat_root / "_work_attempt_001").exists()
    assert sorted(path.name for path in request.repeat_root.iterdir()) == [
        "artifact_manifest.json",
        "artifact_manifest.json.sha256",
        "complete.json",
        "complete.json.sha256",
        "logs",
        "tissue_labeling_upsampled.nii.gz",
    ]


@pytest.mark.parametrize(
    "failure",
    ["missing", "multiple", "empty", "timeout", "forbidden_flags", "forbidden_log", "input_drift"],
)
def test_failures_never_publish_completion_receipt(
    configured_experiment, monkeypatch, failure
):
    request = _request(configured_experiment)
    if failure == "forbidden_flags":
        request = replace(request, flags=request.flags + ("--mesh",))

    def fake(command, *, cwd, timeout_seconds, log_path):
        del timeout_seconds
        if failure == "timeout":
            raise charm_runner.CharmTimeout("deadline")
        if failure != "missing":
            _write_map(Path(cwd), payload=b"" if failure == "empty" else b"map")
        if failure == "multiple":
            _write_map(Path(cwd), subject="sub-OTHER", payload=b"other")
        text = "charm --mesh invoked\n" if failure == "forbidden_log" else "ok\n"
        Path(log_path).write_text(text, encoding="utf-8")
        return text

    monkeypatch.setattr(charm_runner, "run_with_timeout", fake)
    if failure == "input_drift":
        request.inputs.t2.write_bytes(b"changed after preflight")

    with pytest.raises((RepeatRunError, charm_runner.CharmTimeout)):
        run_charm_repeat(request)

    assert not (request.repeat_root / "complete.json").exists()


def test_validation_detects_map_tampering(configured_experiment, monkeypatch):
    request = _request(configured_experiment)
    monkeypatch.setattr(charm_runner, "run_with_timeout", _successful_fake([]))
    run_charm_repeat(request)
    map_path = request.repeat_root / "tissue_labeling_upsampled.nii.gz"
    map_path.write_bytes(b"x" * map_path.stat().st_size)

    with pytest.raises(RepeatRunError, match="SHA-256"):
        validate_charm_repeat(request)


def test_validation_accepts_different_charm_path_but_rescans_log(
    configured_experiment, monkeypatch
):
    request = _request(configured_experiment)
    monkeypatch.setattr(charm_runner, "run_with_timeout", _successful_fake([]))
    run_charm_repeat(request)

    collector_request = replace(request, charm_bin="charm")
    assert validate_charm_repeat(collector_request)["status"] == "complete"

    log = request.repeat_root / "logs/attempt_001.log"
    log.write_text("charm --mesh invoked\n", encoding="utf-8")
    with pytest.raises(RepeatRunError, match="log|forbidden|SHA-256"):
        validate_charm_repeat(collector_request)


def test_retry_quarantines_incomplete_publication(configured_experiment, monkeypatch):
    first = _request(configured_experiment, attempt=1)
    monkeypatch.setattr(charm_runner, "run_with_timeout", _successful_fake([]))
    original_write = charm_runner.write_json_atomic
    calls = {"count": 0}

    def fail_first_manifest(path, payload):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("publication failure")
        return original_write(path, payload)

    monkeypatch.setattr(charm_runner, "write_json_atomic", fail_first_manifest)
    with pytest.raises(OSError, match="publication failure"):
        run_charm_repeat(first)

    monkeypatch.setattr(charm_runner, "write_json_atomic", original_write)
    second = _request(configured_experiment, attempt=2)
    receipt = run_charm_repeat(second)
    assert receipt["attempt"] == 2
    assert (
        second.repeat_root
        / "_failed_publication_attempt_001/tissue_labeling_upsampled.nii.gz"
    ).is_file()


def test_retry_recovers_when_receipt_sidecar_publication_fails(
    configured_experiment, monkeypatch
):
    first = _request(configured_experiment, attempt=1)
    monkeypatch.setattr(charm_runner, "run_with_timeout", _successful_fake([]))
    original_sidecar = charm_runner.write_sha256_sidecar
    calls = {"count": 0}

    def fail_receipt_sidecar(path):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("receipt sidecar failure")
        return original_sidecar(path)

    monkeypatch.setattr(charm_runner, "write_sha256_sidecar", fail_receipt_sidecar)
    with pytest.raises(OSError, match="receipt sidecar failure"):
        run_charm_repeat(first)

    monkeypatch.setattr(charm_runner, "write_sha256_sidecar", original_sidecar)
    second = _request(configured_experiment, attempt=2)
    assert run_charm_repeat(second)["attempt"] == 2
    assert (second.repeat_root / "_failed_publication_attempt_001/complete.json").is_file()
