"""Orchestrate the unchanged canonical repeatability analyzers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any


EXPECTED_INPUT_KEYS = {
    "simnibs326_left_hippocampus",
    "simnibs326_right_m1",
    "simnibs4_left_hippocampus",
    "simnibs4_right_m1",
}
SEED = 20260803
RANK_DRAWS = 20_000
EXPECTED_SUBJECTS = 10


class EstablishedAnalysisError(RuntimeError):
    """Raised when canonical analysis scope or evidence is incomplete."""


def _scripts_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _canonical_paths() -> tuple[Path, Path]:
    root = _scripts_root()
    return (
        root / "ti_current_repair" / "post" / "analyze_repeatability_optimizer_roi_metrics.py",
        root / "ti_current_repair" / "post" / "compare_simnibs_version_remesh_distributions.py",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_code_manifest() -> dict[str, dict[str, str]]:
    """Hash the exact two established implementations used by the wrapper."""

    within_path, cross_path = _canonical_paths()
    paths = {
        "within_version_optimizer_roi_analysis": within_path,
        "cross_version_remesh_distribution_comparison": cross_path,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise EstablishedAnalysisError(
            f"Canonical analysis implementation is missing: {missing}"
        )
    return {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in paths.items()
    }


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise EstablishedAnalysisError(f"Cannot load canonical module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "run", None)):
        raise EstablishedAnalysisError(f"Canonical module has no run function: {path}")
    return module


def _load_canonical_modules() -> tuple[ModuleType, ModuleType]:
    within_path, cross_path = _canonical_paths()
    return (
        _load_module(within_path, "_combined_repeatability_within"),
        _load_module(cross_path, "_combined_repeatability_cross"),
    )


def _read_written_manifest(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "analysis_manifest.json"
    if not path.is_file():
        raise EstablishedAnalysisError(
            f"Canonical analyzer did not write its manifest: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EstablishedAnalysisError(
            f"Cannot read canonical analysis manifest: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise EstablishedAnalysisError(f"Invalid canonical manifest: {path}")
    return payload


def _require_return_matches_written(
    returned: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    written = _read_written_manifest(output_dir)
    scalar_scope_keys = (
        "status",
        "rows",
        "subject_count",
        "input_rows",
        "selected_remesh_rows",
        "participants",
    )
    scope_mismatch = any(
        returned.get(key) != written.get(key)
        for key in scalar_scope_keys
        if key in returned or key in written
    )
    returned_parameters = returned.get("analysis_parameters")
    written_parameters = written.get("analysis_parameters")
    parameter_mismatch = any(
        (returned_parameters or {}).get(key)
        != (written_parameters or {}).get(key)
        for key in ("seed", "rank_draws")
    )
    if scope_mismatch or parameter_mismatch:
        raise EstablishedAnalysisError(
            "Canonical analyzer returned and written manifests are inconsistent"
        )
    return written


def _validate_within(result: dict[str, Any], label: str) -> None:
    if result.get("status") != "complete":
        raise EstablishedAnalysisError(f"{label} is not complete")
    if result.get("rows") != 1600:
        raise EstablishedAnalysisError(f"{label} must contain exactly 1,600 rows")
    if result.get("subject_count") != EXPECTED_SUBJECTS:
        raise EstablishedAnalysisError(f"{label} must contain 10 participants")
    parameters = result.get("analysis_parameters")
    if not isinstance(parameters, dict) or parameters.get("seed") != SEED or parameters.get("rank_draws") != RANK_DRAWS:
        raise EstablishedAnalysisError(
            f"{label} does not prove seed {SEED} and {RANK_DRAWS} rank draws"
        )


def _validate_cross(result: dict[str, Any]) -> None:
    if result.get("status") != "complete":
        raise EstablishedAnalysisError("Cross-version analysis is not complete")
    if result.get("input_rows") != 3200:
        raise EstablishedAnalysisError(
            "Cross-version analysis must validate exactly 3,200 input rows"
        )
    if result.get("selected_remesh_rows") != 1600:
        raise EstablishedAnalysisError(
            "Cross-version analysis must select exactly 1,600 remesh rows"
        )
    if result.get("participants") != EXPECTED_SUBJECTS:
        raise EstablishedAnalysisError(
            "Cross-version analysis must contain 10 participants"
        )


def _validate_inputs(inputs: dict[str, Path]) -> dict[str, Path]:
    if set(inputs) != EXPECTED_INPUT_KEYS:
        raise EstablishedAnalysisError(
            "Established analysis requires exactly the four canonical assembled inputs"
        )
    normalized = {key: Path(path).resolve() for key, path in inputs.items()}
    missing = [str(path) for path in normalized.values() if not path.is_file()]
    if missing:
        raise EstablishedAnalysisError(f"Canonical assembled input is missing: {missing}")
    return normalized


def run_established(
    inputs: dict[str, Path], output_root: Path
) -> dict[str, Any]:
    """Run both within-version packages and the official remesh comparison."""

    paths = _validate_inputs(inputs)
    output_root = Path(output_root)
    if output_root.exists():
        raise EstablishedAnalysisError(
            f"Refusing to overwrite established analysis root: {output_root}"
        )
    output_root.mkdir(parents=True)
    within_module, cross_module = _load_canonical_modules()

    results: dict[str, dict[str, Any]] = {}
    for version in ("simnibs326", "simnibs4"):
        label = f"within_{version}"
        out_dir = output_root / label
        args = argparse.Namespace(
            left_csv=paths[f"{version}_left_hippocampus"],
            right_csv=paths[f"{version}_right_m1"],
            out_dir=out_dir,
            seed=SEED,
            rank_draws=RANK_DRAWS,
            expected_subjects=EXPECTED_SUBJECTS,
        )
        returned = within_module.run(args)
        if not isinstance(returned, dict):
            raise EstablishedAnalysisError(f"{label} returned no manifest")
        _validate_within(returned, label)
        written = _require_return_matches_written(returned, out_dir)
        _validate_within(written, label)
        results[label] = written

    cross_dir = output_root / "cross_version_remesh"
    cross_args = argparse.Namespace(
        v3_left_hippocampus_csv=paths["simnibs326_left_hippocampus"],
        v3_right_m1_csv=paths["simnibs326_right_m1"],
        v4_left_hippocampus_csv=paths["simnibs4_left_hippocampus"],
        v4_right_m1_csv=paths["simnibs4_right_m1"],
        out_dir=cross_dir,
    )
    returned_cross = cross_module.run(cross_args)
    if not isinstance(returned_cross, dict):
        raise EstablishedAnalysisError("Cross-version analyzer returned no manifest")
    _validate_cross(returned_cross)
    written_cross = _require_return_matches_written(returned_cross, cross_dir)
    _validate_cross(written_cross)
    results["cross_version_remesh"] = written_cross

    manifest: dict[str, Any] = {
        "status": "complete",
        "schema_version": 1,
        "canonical_code": canonical_code_manifest(),
        "within_simnibs326": results["within_simnibs326"],
        "within_simnibs4": results["within_simnibs4"],
        "cross_version_remesh": results["cross_version_remesh"],
        "scope": {
            "participants": EXPECTED_SUBJECTS,
            "within_version_rows": 1600,
            "cross_version_input_rows": 3200,
            "cross_version_selected_remesh_rows": 1600,
            "seed": SEED,
            "rank_draws": RANK_DRAWS,
        },
    }
    (output_root / "established_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest

