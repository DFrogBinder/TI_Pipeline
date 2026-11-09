from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import simnibs_segmentation_determinism.headreco_runner as headreco_runner
from simnibs_segmentation_determinism.headreco_runner import (
    HeadrecoRepeatRequest,
    RepeatRunError,
    expected_headreco_artifacts,
    run_headreco_repeat,
    validate_headreco_repeat,
)
from simnibs_segmentation_determinism.settings import resolve_subject_inputs


def _request(config, repeat_index=1, attempt=1):
    return HeadrecoRepeatRequest(
        config=config,
        inputs=resolve_subject_inputs(config),
        repeat_index=repeat_index,
        attempt=attempt,
        compatibility_script=Path("/pipeline/simnibs_326_repeatability/cat12_compat.py"),
        python_executable=Path("/module/python"),
    )


def _write_expected_outputs(cwd: Path, subject: str, extra_spm_probability=False):
    root = cwd / f"m2m_{subject}"
    for relative in expected_headreco_artifacts(subject):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((relative + "\n").encode("utf-8"))
    if extra_spm_probability:
        path = root / "segment/spm/c7T1fs_conform.nii"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unexpected")


def _successful_fake(record, stdout="preparevols complete\n", extra=False):
    def fake_run(command, *, cwd, timeout_seconds, log_path):
        del timeout_seconds
        kwargs = {"cwd": str(cwd), "log_path": log_path}
        record.append((command, kwargs))
        _write_expected_outputs(Path(cwd), "sub-CC110174", extra_spm_probability=extra)
        Path(log_path).write_text(stdout, encoding="utf-8")
        return stdout

    return fake_run


def test_command_is_exactly_compatibility_wrapper_and_preparevols(
    configured_experiment, monkeypatch
):
    request = _request(configured_experiment)
    calls = []
    monkeypatch.setattr(headreco_runner, "run_with_timeout", _successful_fake(calls))

    receipt = run_headreco_repeat(request)

    assert calls[0][0] == [
        "/module/python",
        "/pipeline/simnibs_326_repeatability/cat12_compat.py",
        "headreco",
        "preparevols",
        "--noclean",
        "sub-CC110174",
        str(request.inputs.t1),
        str(request.inputs.t2),
    ]
    assert receipt["command"] == calls[0][0]
    assert receipt["stage"] == "preparevols"
    assert receipt["module"] == "SimNIBS/3.2.6-foss-2023a"
    assert receipt["status"] == "complete"


def test_ten_requests_keep_subject_constant_and_repeat_roots_unique(configured_experiment):
    requests = [_request(configured_experiment, repeat_index=index) for index in range(1, 11)]

    assert {request.inputs.subject for request in requests} == {"sub-CC110174"}
    assert len({request.repeat_root for request in requests}) == 10
    assert requests[0].repeat_root.name == "repeat_001"
    assert requests[-1].repeat_root.name == "repeat_010"


def test_expected_probability_and_binary_maps_are_promoted_and_validate(
    configured_experiment, monkeypatch
):
    request = _request(configured_experiment)
    monkeypatch.setattr(headreco_runner, "run_with_timeout", _successful_fake([]))

    run_headreco_repeat(request)
    validated = validate_headreco_repeat(request)

    artifact_paths = [entry["relative_path"] for entry in validated["artifacts"]]
    assert artifact_paths == sorted(expected_headreco_artifacts(request.inputs.subject))
    assert all((request.repeat_root / "artifacts" / path).is_file() for path in artifact_paths)
    assert (request.repeat_root / "artifact_manifest.json").is_file()
    assert (request.repeat_root / "complete.json").is_file()
    assert not any(path.name.startswith(".artifacts") for path in request.repeat_root.iterdir())


@pytest.mark.parametrize(
    "failure",
    ["missing", "unexpected", "forbidden", "nonzero", "input_drift"],
)
def test_failures_never_publish_completion_receipt(
    configured_experiment, monkeypatch, failure
):
    request = _request(configured_experiment)

    def fake_run(command, *, cwd, timeout_seconds, log_path):
        del timeout_seconds
        _write_expected_outputs(
            Path(cwd),
            request.inputs.subject,
            extra_spm_probability=failure == "unexpected",
        )
        if failure == "missing":
            (Path(cwd) / f"m2m_{request.inputs.subject}" / expected_headreco_artifacts(request.inputs.subject)[0]).unlink()
        stdout = "headreco volumemesh was invoked\n" if failure == "forbidden" else "ok\n"
        Path(log_path).write_text(stdout, encoding="utf-8")
        if failure == "nonzero":
            raise RepeatRunError("headreco returned nonzero exit status 9")
        return stdout

    monkeypatch.setattr(headreco_runner, "run_with_timeout", fake_run)
    if failure == "input_drift":
        request.inputs.t1.write_bytes(b"changed after preflight")

    with pytest.raises(RepeatRunError):
        run_headreco_repeat(request)

    assert not (request.repeat_root / "complete.json").exists()


def test_validation_detects_promoted_artifact_tampering(configured_experiment, monkeypatch):
    request = _request(configured_experiment)
    monkeypatch.setattr(headreco_runner, "run_with_timeout", _successful_fake([]))
    run_headreco_repeat(request)
    receipt = json.loads((request.repeat_root / "complete.json").read_text(encoding="utf-8"))
    artifact = request.repeat_root / "artifacts" / receipt["artifacts"][0]["relative_path"]
    artifact.write_bytes(b"x" * artifact.stat().st_size)

    with pytest.raises(RepeatRunError, match="artifact SHA-256"):
        validate_headreco_repeat(request)


def test_validation_accepts_different_collector_python_but_rescans_log(
    configured_experiment, monkeypatch
):
    request = _request(configured_experiment)
    monkeypatch.setattr(headreco_runner, "run_with_timeout", _successful_fake([]))
    run_headreco_repeat(request)

    collector_request = HeadrecoRepeatRequest(
        config=request.config,
        inputs=request.inputs,
        repeat_index=1,
        compatibility_script=request.compatibility_script,
        python_executable=Path("/different/module/python3"),
    )
    assert validate_headreco_repeat(collector_request)["status"] == "complete"

    log = request.repeat_root / "logs/attempt_001.log"
    log.write_text("headreco volumemesh was invoked\n", encoding="utf-8")
    with pytest.raises(RepeatRunError, match="log|forbidden|SHA-256"):
        validate_headreco_repeat(collector_request)


def test_timeout_retains_streamed_log_and_terminates_process_group(tmp_path):
    log = tmp_path / "timeout.log"
    with pytest.raises(RepeatRunError, match="timed out"):
        headreco_runner.run_with_timeout(
            [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(60)"],
            cwd=tmp_path,
            timeout_seconds=1,
            log_path=log,
        )
    assert "started" in log.read_text(encoding="utf-8")


def test_retry_quarantines_incomplete_publication(configured_experiment, monkeypatch):
    first = _request(configured_experiment, attempt=1)
    monkeypatch.setattr(headreco_runner, "run_with_timeout", _successful_fake([]))
    original_write = headreco_runner.write_json_atomic
    calls = {"count": 0}

    def fail_first_manifest(path, payload):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("publication failure")
        return original_write(path, payload)

    monkeypatch.setattr(headreco_runner, "write_json_atomic", fail_first_manifest)
    with pytest.raises(OSError, match="publication failure"):
        run_headreco_repeat(first)

    monkeypatch.setattr(headreco_runner, "write_json_atomic", original_write)
    second = _request(configured_experiment, attempt=2)
    receipt = run_headreco_repeat(second)
    assert receipt["attempt"] == 2
    assert (second.repeat_root / "_failed_publication_attempt_001/artifacts").is_dir()


def test_retry_recovers_when_receipt_sidecar_publication_fails(
    configured_experiment, monkeypatch
):
    first = _request(configured_experiment, attempt=1)
    monkeypatch.setattr(headreco_runner, "run_with_timeout", _successful_fake([]))
    original_sidecar = headreco_runner.write_sha256_sidecar
    calls = {"count": 0}

    def fail_receipt_sidecar(path):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("receipt sidecar failure")
        return original_sidecar(path)

    monkeypatch.setattr(headreco_runner, "write_sha256_sidecar", fail_receipt_sidecar)
    with pytest.raises(OSError, match="receipt sidecar failure"):
        run_headreco_repeat(first)

    monkeypatch.setattr(headreco_runner, "write_sha256_sidecar", original_sidecar)
    second = _request(configured_experiment, attempt=2)
    assert run_headreco_repeat(second)["attempt"] == 2
    assert (second.repeat_root / "_failed_publication_attempt_001/complete.json").is_file()
