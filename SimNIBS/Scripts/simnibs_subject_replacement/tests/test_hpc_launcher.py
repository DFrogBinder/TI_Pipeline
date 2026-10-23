from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from simnibs_subject_replacement.replacement import load_prepared_manifest, prepare
from simnibs_subject_replacement.settings import ReplacementConfig, replacement_paths


PACKAGE = Path(__file__).resolve().parents[1]
SUBMIT = PACKAGE / "hpc" / "submit.sh"


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


def _common_args(config: ReplacementConfig) -> list[str]:
    return [
        f"--subject={config.subject}",
        f"--replaces={config.replaces}",
        f"--legacy-v3-root={config.legacy_v3_root}",
        f"--final132-subjects={config.final132_subjects_file}",
        f"--corrected-v4-scaffold={config.corrected_v4_scaffold_root}",
        f"--atlas-root={config.atlas_root}",
        f"--targets-csv={config.targets_csv}",
        f"--work-root={config.work_root}",
    ]


def _run(
    env: dict[str, str], config: ReplacementConfig, mode: str, *extra: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SUBMIT), mode, *_common_args(config), *extra],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_wrapper_requires_runtime_identity_and_legacy_root() -> None:
    result = subprocess.run(
        ["bash", str(SUBMIT), "preflight"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--subject" in result.stderr
    assert "--replaces" in result.stderr
    assert "--legacy-v3-root" in result.stderr


def test_preflight_and_prepare_are_explicit_and_submit_nothing(
    replacement_layout: dict[str, Path | str],
) -> None:
    config = _config(replacement_layout)
    env = os.environ.copy()
    env["PYTHON_BIN"] = sys.executable
    calls = config.work_root.parent / "v3.calls"
    fake_v3 = config.work_root.parent / "fake-v3-submit.sh"
    fake_v3.write_text(f"#!/bin/bash\nprintf '%s\\n' \"$*\" >> {calls}\n", encoding="utf-8")
    fake_v3.chmod(0o755)
    env["V3_SUBMIT"] = str(fake_v3)

    audited = _run(env, config, "preflight")
    assert audited.returncode == 0, audited.stderr
    assert '"simulation_tasks": 320' in audited.stdout
    assert not replacement_paths(config).root.exists()
    assert not calls.exists()

    prepared = _run(env, config, "prepare")
    assert prepared.returncode == 0, prepared.stderr
    assert '"action": "created"' in prepared.stdout
    assert replacement_paths(config).manifest.is_file()
    assert not calls.exists()


def test_v3_delegation_exports_manifest_bound_singleton_roots(
    replacement_layout: dict[str, Path | str],
) -> None:
    config = _config(replacement_layout)
    paths = replacement_paths(config)
    prepare(config)
    calls = config.work_root.parent / "v3.calls"
    env_dump = config.work_root.parent / "v3.env"
    fake_v3 = config.work_root.parent / "fake-v3-submit.sh"
    fake_v3.write_text(
        "#!/bin/bash\n"
        f"printf '%s\\n' \"$*\" >> {calls}\n"
        f"printf '%s\\n' \"$SIMNIBS326H_COHORT|$SIMNIBS326H_EXTERNAL_MANIFEST|$SIMNIBS326H_HEADMODEL_ROOT|$SIMNIBS326H_LEFT_ROOT|$SIMNIBS326H_RIGHT_ROOT\" > {env_dump}\n",
        encoding="utf-8",
    )
    fake_v3.chmod(0o755)
    env = os.environ.copy()
    env.update({"PYTHON_BIN": sys.executable, "V3_SUBMIT": str(fake_v3)})

    result = _run(env, config, "v3-stage-references")

    assert result.returncode == 0, result.stderr
    assert calls.read_text(encoding="utf-8").strip() == "stage-references --cohort=external1"
    assert env_dump.read_text(encoding="utf-8").strip() == "|".join(
        [
            "external1",
            str(paths.manifest),
            str(paths.v3_headmodel_root),
            str(paths.v3_left_root),
            str(paths.v3_right_root),
        ]
    )


def test_right_v3_is_blocked_until_left_terminal_receipt(
    replacement_layout: dict[str, Path | str],
) -> None:
    from simnibs_subject_replacement.tests.test_replacement import (
        _write_left_completion,
        _write_v3_promotion,
    )

    config = _config(replacement_layout)
    paths = replacement_paths(config)
    prepare(config)
    manifest = load_prepared_manifest(paths.manifest)
    _write_v3_promotion(manifest)
    calls = config.work_root.parent / "v3.calls"
    fake_v3 = config.work_root.parent / "fake-v3-submit.sh"
    fake_v3.write_text(f"#!/bin/bash\nprintf '%s\\n' \"$*\" >> {calls}\n", encoding="utf-8")
    fake_v3.chmod(0o755)
    env = os.environ.copy()
    env.update({"PYTHON_BIN": sys.executable, "V3_SUBMIT": str(fake_v3)})

    blocked = _run(env, config, "v3-right-remesh")
    assert blocked.returncode != 0
    assert "left-hippocampus" in blocked.stderr
    assert not calls.exists()

    _write_left_completion(manifest, "v3")
    released = _run(env, config, "v3-right-remesh")
    assert released.returncode == 0, released.stderr
    assert calls.read_text(encoding="utf-8").strip() == (
        "target-remesh right-m1 --cohort=external1"
    )


def test_wrapper_lists_full_v3_and_v4_command_surface() -> None:
    text = SUBMIT.read_text(encoding="utf-8")
    expected = {
        "preflight", "prepare", "v3-module-check", "v3-stage-references",
        "v3-headmodels", "v3-collect-headmodels", "v3-promote", "v3-init",
        "v3-left-remesh", "v3-left-fixed", "v3-right-remesh",
        "v3-right-fixed", "v3-analysis", "v3-status", "v4-init-left",
        "v4-preflight-left", "v4-submit-left", "v4-init-right",
        "v4-preflight-right", "v4-submit-right", "status",
    }
    assert all(mode in text for mode in expected)
    assert "sub-CC110056" not in text
    assert "sub-CC510415" not in text
