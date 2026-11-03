from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from repeatability_combined_analysis import established


def assembled_inputs(tmp_path: Path) -> dict[str, Path]:
    inputs = {}
    for version in ("simnibs326", "simnibs4"):
        for target in ("left_hippocampus", "right_m1"):
            path = tmp_path / "inputs" / f"{version}_{target}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("placeholder\n", encoding="utf-8")
            inputs[f"{version}_{target}"] = path
    return inputs


def test_orchestration_calls_canonical_analyzers_with_exact_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, argparse.Namespace]] = []

    def within_run(args: argparse.Namespace) -> dict[str, object]:
        calls.append(("within", args))
        args.out_dir.mkdir(parents=True)
        result = {
            "status": "complete",
            "rows": 1600,
            "subject_count": 10,
            "analysis_parameters": {"seed": 20260803, "rank_draws": 20000},
        }
        (args.out_dir / "analysis_manifest.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        return result

    def cross_run(args: argparse.Namespace) -> dict[str, object]:
        calls.append(("cross", args))
        args.out_dir.mkdir(parents=True)
        result = {
            "status": "complete",
            "input_rows": 3200,
            "selected_remesh_rows": 1600,
            "participants": 10,
        }
        (args.out_dir / "analysis_manifest.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        return result

    monkeypatch.setattr(
        established,
        "_load_canonical_modules",
        lambda: (SimpleNamespace(run=within_run), SimpleNamespace(run=cross_run)),
    )
    inputs = assembled_inputs(tmp_path)
    output_root = tmp_path / "established"
    result = established.run_established(inputs, output_root)

    assert [name for name, _ in calls] == ["within", "within", "cross"]
    v3 = calls[0][1]
    v4 = calls[1][1]
    cross = calls[2][1]
    assert v3.left_csv == inputs["simnibs326_left_hippocampus"]
    assert v3.right_csv == inputs["simnibs326_right_m1"]
    assert v4.left_csv == inputs["simnibs4_left_hippocampus"]
    assert v4.right_csv == inputs["simnibs4_right_m1"]
    for args in (v3, v4):
        assert args.expected_subjects == 10
        assert args.seed == 20260803
        assert args.rank_draws == 20_000
    assert cross.v3_left_hippocampus_csv == inputs["simnibs326_left_hippocampus"]
    assert cross.v3_right_m1_csv == inputs["simnibs326_right_m1"]
    assert cross.v4_left_hippocampus_csv == inputs["simnibs4_left_hippocampus"]
    assert cross.v4_right_m1_csv == inputs["simnibs4_right_m1"]
    assert "legacy" not in " ".join(str(value) for value in vars(cross).values()).lower()
    assert result["status"] == "complete"
    assert result["within_simnibs326"]["rows"] == 1600
    assert result["within_simnibs4"]["rows"] == 1600
    assert result["cross_version_remesh"]["selected_remesh_rows"] == 1600


def test_code_manifest_hashes_both_unchanged_canonical_scripts() -> None:
    manifest = established.canonical_code_manifest()
    assert set(manifest) == {
        "within_version_optimizer_roi_analysis",
        "cross_version_remesh_distribution_comparison",
    }
    expected_names = {
        "analyze_repeatability_optimizer_roi_metrics.py",
        "compare_simnibs_version_remesh_distributions.py",
    }
    assert {Path(item["path"]).name for item in manifest.values()} == expected_names
    for item in manifest.values():
        path = Path(item["path"])
        assert path.is_file()
        assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert len(item["sha256"]) == 64


@pytest.mark.parametrize(
    ("bad_result", "message"),
    (
        ({"status": "complete", "rows": 1599, "subject_count": 10}, "1,600"),
        ({"status": "failed", "rows": 1600, "subject_count": 10}, "complete"),
    ),
)
def test_rejects_bad_within_version_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_result: dict[str, object],
    message: str,
) -> None:
    def within_run(args: argparse.Namespace) -> dict[str, object]:
        args.out_dir.mkdir(parents=True)
        (args.out_dir / "analysis_manifest.json").write_text(
            json.dumps(bad_result), encoding="utf-8"
        )
        return bad_result

    monkeypatch.setattr(
        established,
        "_load_canonical_modules",
        lambda: (
            SimpleNamespace(run=within_run),
            SimpleNamespace(run=lambda _args: {}),
        ),
    )
    with pytest.raises(established.EstablishedAnalysisError, match=message):
        established.run_established(
            assembled_inputs(tmp_path), tmp_path / "established"
        )


def test_rejects_inconsistent_written_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    returned = {
        "status": "complete",
        "rows": 1600,
        "subject_count": 10,
        "analysis_parameters": {"seed": 20260803, "rank_draws": 20000},
    }

    def within_run(args: argparse.Namespace) -> dict[str, object]:
        args.out_dir.mkdir(parents=True)
        written = dict(returned)
        written["rows"] = 800
        (args.out_dir / "analysis_manifest.json").write_text(
            json.dumps(written), encoding="utf-8"
        )
        return returned

    monkeypatch.setattr(
        established,
        "_load_canonical_modules",
        lambda: (
            SimpleNamespace(run=within_run),
            SimpleNamespace(run=lambda _args: {}),
        ),
    )
    with pytest.raises(
        established.EstablishedAnalysisError, match="returned and written"
    ):
        established.run_established(
            assembled_inputs(tmp_path), tmp_path / "established"
        )


def test_accepts_json_safe_null_for_returned_nonfinite_descriptive_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core = {
        "status": "complete",
        "rows": 1600,
        "subject_count": 10,
        "analysis_parameters": {"seed": 20260803, "rank_draws": 20000},
    }

    def within_run(args: argparse.Namespace) -> dict[str, object]:
        args.out_dir.mkdir(parents=True)
        returned = {**core, "support_audit": {"correlation": float("nan")}}
        written = {**core, "support_audit": {"correlation": None}}
        (args.out_dir / "analysis_manifest.json").write_text(
            json.dumps(written), encoding="utf-8"
        )
        return returned

    cross_result = {
        "status": "complete",
        "input_rows": 3200,
        "selected_remesh_rows": 1600,
        "participants": 10,
    }

    def cross_run(args: argparse.Namespace) -> dict[str, object]:
        args.out_dir.mkdir(parents=True)
        (args.out_dir / "analysis_manifest.json").write_text(
            json.dumps(cross_result), encoding="utf-8"
        )
        return cross_result

    monkeypatch.setattr(
        established,
        "_load_canonical_modules",
        lambda: (SimpleNamespace(run=within_run), SimpleNamespace(run=cross_run)),
    )
    result = established.run_established(
        assembled_inputs(tmp_path), tmp_path / "established"
    )
    assert result["status"] == "complete"
    assert result["within_simnibs326"]["support_audit"]["correlation"] is None


def test_rejects_bad_cross_version_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    within_result = {
        "status": "complete",
        "rows": 1600,
        "subject_count": 10,
        "analysis_parameters": {"seed": 20260803, "rank_draws": 20000},
    }

    def within_run(args: argparse.Namespace) -> dict[str, object]:
        args.out_dir.mkdir(parents=True)
        (args.out_dir / "analysis_manifest.json").write_text(
            json.dumps(within_result), encoding="utf-8"
        )
        return within_result

    def cross_run(args: argparse.Namespace) -> dict[str, object]:
        args.out_dir.mkdir(parents=True)
        result = {
            "status": "complete",
            "input_rows": 3200,
            "selected_remesh_rows": 800,
            "participants": 10,
        }
        (args.out_dir / "analysis_manifest.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        return result

    monkeypatch.setattr(
        established,
        "_load_canonical_modules",
        lambda: (SimpleNamespace(run=within_run), SimpleNamespace(run=cross_run)),
    )
    with pytest.raises(established.EstablishedAnalysisError, match="1,600"):
        established.run_established(
            assembled_inputs(tmp_path), tmp_path / "established"
        )


def test_requires_all_four_named_assembled_inputs(tmp_path: Path) -> None:
    inputs = assembled_inputs(tmp_path)
    inputs.pop("simnibs4_right_m1")
    with pytest.raises(established.EstablishedAnalysisError, match="four canonical"):
        established.run_established(inputs, tmp_path / "established")

