#!/usr/bin/env python3
"""Run the completed repeatability-paper analyses and build a download bundle.

This analysis-only driver reads the immutable historical remesh metrics, the
corrected spherical-median fixed-mesh metrics, and the completed 40-by-40
nested metrics. It performs no meshing or field simulation and does not modify
any source experiment directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Any


SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEFT_HISTORICAL = Path(
    "/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10/"
    "_post_processing/repeatability_optimizer_roi_metrics_v1/"
    "optimizer_roi_metrics.csv"
)
DEFAULT_LEFT_CORRECTED = Path(
    "/mnt/parscratch/users/cop23bi/"
    "final_132_repeatability_balanced_10_spherical_fixed_v1/"
    "_post_processing/optimizer_roi_metrics_v1/optimizer_roi_metrics.csv"
)
DEFAULT_RIGHT_HISTORICAL = Path(
    "/mnt/parscratch/users/cop23bi/"
    "final_132_repeatability_balanced_10_right_m1/"
    "_post_processing/repeatability_optimizer_roi_metrics_v1/"
    "optimizer_roi_metrics.csv"
)
DEFAULT_RIGHT_CORRECTED = Path(
    "/mnt/parscratch/users/cop23bi/"
    "final_132_repeatability_balanced_10_right_m1_spherical_fixed_v1/"
    "_post_processing/optimizer_roi_metrics_v1/optimizer_roi_metrics.csv"
)
DEFAULT_NESTED_METRICS = Path(
    "/mnt/parscratch/users/cop23bi/final_132_repeatability_nested_40x40_v1/"
    "_post_processing/optimizer_roi_metrics_v1/optimizer_roi_metrics.csv"
)
DEFAULT_NESTED_VARIANCE = Path(
    "/mnt/parscratch/users/cop23bi/final_132_repeatability_nested_40x40_v1/"
    "_analysis/nested_variance/nested_variance_components.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/parscratch/users/cop23bi/final_132_repeatability_paper_update_v1"
)


def _load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _build_bundle(*, output_root: Path, payload_paths: list[Path]) -> dict[str, Any]:
    bundle = output_root / "repeatability_paper_analysis_v1.tar.gz"
    temporary = bundle.with_name(f".{bundle.name}.tmp-{os.getpid()}")
    with tarfile.open(temporary, mode="w:gz") as archive:
        for path in payload_paths:
            archive.add(path, arcname=path.relative_to(output_root))
    temporary.replace(bundle)
    digest = _sha256(bundle)
    checksum = bundle.with_suffix(bundle.suffix + ".sha256")
    checksum.write_text(f"{digest}  {bundle.name}\n", encoding="utf-8")
    return {
        "path": str(bundle),
        "sha256": digest,
        "checksum_path": str(checksum),
        "size_bytes": bundle.stat().st_size,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_root / ".mplconfig"))

    prepare_module = _load_module(
        "prepare_spherical_fixed_paper_inputs",
        SCRIPTS_ROOT
        / "mesh_repeat_analysis/post/prepare_spherical_fixed_paper_inputs.py",
    )
    fixed_module = _load_module(
        "analyze_repeatability_optimizer_roi_metrics",
        SCRIPTS_ROOT
        / "ti_current_repair/post/analyze_repeatability_optimizer_roi_metrics.py",
    )
    nested_module = _load_module(
        "plot_nested_repeatability",
        SCRIPTS_ROOT / "mesh_repeat_analysis/post/plot_nested_repeatability.py",
    )

    fixed_input_dir = output_root / "corrected_fixed_inputs"
    fixed_analysis_dir = output_root / "corrected_fixed_analysis"
    nested_analysis_dir = output_root / "nested_analysis"

    prepared = prepare_module.run(
        argparse.Namespace(
            left_historical_csv=args.left_historical_csv,
            left_corrected_fixed_csv=args.left_corrected_fixed_csv,
            right_historical_csv=args.right_historical_csv,
            right_corrected_fixed_csv=args.right_corrected_fixed_csv,
            output_dir=fixed_input_dir,
        )
    )
    fixed = fixed_module.run(
        argparse.Namespace(
            left_csv=(
                fixed_input_dir / "left_hippocampus_optimizer_roi_metrics.csv"
            ),
            right_csv=fixed_input_dir / "right_m1_optimizer_roi_metrics.csv",
            out_dir=fixed_analysis_dir,
            seed=args.rank_seed,
            rank_draws=args.rank_draws,
        )
    )
    nested = nested_module.run(
        metrics_csv=args.nested_metrics_csv,
        variance_json=args.nested_variance_json,
        output_dir=nested_analysis_dir,
        metric="roi_median_v_per_m",
        expected_meshes=40,
        expected_repeats=40,
    )

    expected_figures = [
        fixed_analysis_dir
        / "figures/left_hippocampus/01_primary_median_roi_repeat_distributions.png",
        fixed_analysis_dir
        / "figures/left_hippocampus/02_single_repeat_subject_ranking_uncertainty.png",
        fixed_analysis_dir
        / "figures/right_m1/01_primary_median_roi_repeat_distributions.png",
        fixed_analysis_dir
        / "figures/right_m1/02_single_repeat_subject_ranking_uncertainty.png",
        nested_analysis_dir / "nested_mesh_by_solver_repeatability.png",
        nested_analysis_dir / "nested_mesh_by_solver_repeatability.svg",
    ]
    missing = [str(path) for path in expected_figures if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Expected figures were not produced: {missing}")

    summary = {
        "schema_version": 1,
        "status": "complete",
        "analysis": "repeatability paper spherical-fixed and nested update",
        "source_outputs_modified": False,
        "scope": {
            "targets": 2,
            "participants_per_target": 10,
            "conditions_per_target": 2,
            "technical_repeats_per_condition": 40,
            "corrected_fixed_analysis_rows": 1600,
            "nested_participants": 1,
            "nested_outer_meshes": 40,
            "nested_repeats_per_mesh": 40,
            "nested_analysis_rows": 1600,
            "simulations_run": 0,
        },
        "fixed_analysis": {
            "status": fixed["status"],
            "rows": fixed["rows"],
            "targets": fixed["targets"],
            "analysis_manifest": str(
                fixed_analysis_dir / "analysis_manifest.json"
            ),
        },
        "nested_analysis": {
            "status": nested["status"],
            "subject": nested["subject"],
            "roi": nested["roi"],
            "observations": nested["observations"],
            "figure_manifest": str(
                nested_analysis_dir / "nested_figure_manifest.json"
            ),
        },
        "input_assembly": {
            "status": prepared["status"],
            "observed_rows": prepared["observed_rows"],
            "manifest": str(fixed_input_dir / "assembly_manifest.json"),
        },
        "figures": [str(path) for path in expected_figures],
    }
    summary_path = output_root / "paper_analysis_summary.json"
    _write_json_atomic(summary_path, summary)
    payload_paths = [
        fixed_input_dir,
        fixed_analysis_dir,
        nested_analysis_dir,
        summary_path,
    ]
    bundle = _build_bundle(output_root=output_root, payload_paths=payload_paths)
    result = {
        **summary,
        "download_bundle": bundle,
    }
    _write_json_atomic(output_root / "paper_analysis_manifest.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--left-historical-csv", type=Path, default=DEFAULT_LEFT_HISTORICAL
    )
    parser.add_argument(
        "--left-corrected-fixed-csv", type=Path, default=DEFAULT_LEFT_CORRECTED
    )
    parser.add_argument(
        "--right-historical-csv", type=Path, default=DEFAULT_RIGHT_HISTORICAL
    )
    parser.add_argument(
        "--right-corrected-fixed-csv", type=Path, default=DEFAULT_RIGHT_CORRECTED
    )
    parser.add_argument(
        "--nested-metrics-csv", type=Path, default=DEFAULT_NESTED_METRICS
    )
    parser.add_argument(
        "--nested-variance-json", type=Path, default=DEFAULT_NESTED_VARIANCE
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rank-seed", type=int, default=20260803)
    parser.add_argument("--rank-draws", type=int, default=20_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
