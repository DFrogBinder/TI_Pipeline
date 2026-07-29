#!/usr/bin/env python3
"""Validate and package the exact inputs for final CamCan figures.

The supervisor-revision figures combine two independently collected analyses:
the final-132 cohort and the 7-subject personalized comparison. This utility
adds a small, fail-closed publication bundle to either results directory before
that directory is archived. The bundle contains the exact renderer, hashes of
every required source file, and self-contained reconstruction instructions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path


PACKAGE_SCHEMA_VERSION = 1
RENDERER_NAME = "build_camcan_supervisor_revision_figures.py"
EXPECTED_FIGURE_STEMS = [
    "figure_population_target_offtarget_relationship_ge_0p20",
    "figure_population_mean_field_offtarget_relationship_ge_0p20",
    "figure_population_minimum_field_offtarget_relationship_ge_0p20",
    "figure_population_maximum_p99_9_field_offtarget_relationship_ge_0p20",
    "figure_population_target_field_distributions",
    "figure_population_offtarget_target_ratio_ge_0p20",
    "figure_mni152_percentile_context_ge_0p20",
    "figure_personalization_effectiveness_spread_ge_0p20",
    "figure_personalization_all_subject_changes_left_hippocampus",
    "figure_personalization_all_subject_changes_left_m1",
    "figure_personalization_all_subject_changes_right_dlpc",
    "figure_personalization_all_subject_changes_right_thalamus",
    "figure_personalization_target_offtarget_ratio_ge_0p20_left_hippocampus",
    "figure_personalization_target_offtarget_ratio_ge_0p20_left_m1",
    "figure_personalization_target_offtarget_ratio_ge_0p20_right_dlpc",
    "figure_personalization_target_offtarget_ratio_ge_0p20_right_thalamus",
    *[
        f"figure_personalization_repeat_distributions_{roi}_{statistic}"
        for roi in (
            "left_hippocampus",
            "left_m1",
            "right_dlpc",
            "right_thalamus",
        )
        for statistic in (
            "minimum",
            "mean",
            "median",
            "maximum_p99_9",
        )
    ],
]

COMMON_METRICS = {
    "roi_min_v_per_m",
    "roi_mean_v_per_m",
    "roi_median_v_per_m",
    "roi_robust_max_p99_9_v_per_m",
    "target_coverage_percent_ge_0p2",
    "off_target_coverage_percent_ge_0p2",
}

PAIRED_COLUMNS = {
    f"{metric}__{condition}_repeat_{summary}"
    for metric in COMMON_METRICS
    for condition in ("generic", "personalized")
    for summary in ("mean", "sd")
}

ROLE_CONFIG = {
    "cohort": {
        "manifest_expectations": {
            "analysis_schema_version": 4,
            "status": "complete",
            "subjects": 132,
            "repeat_level_records": 5280,
            "subject_level_records": 528,
            "mni_baselines": 4,
            "thresholds_v_per_m": [0.2, 0.18, 0.15],
        },
        "tables": {
            "subject_level_repeat_mean_metrics.csv": {
                "rows": 528,
                "columns": {"subject", "roi"} | COMMON_METRICS,
            },
            "mni152_baseline_metrics.csv": {
                "rows": 4,
                "columns": {"subject", "roi"} | COMMON_METRICS,
            },
        },
        "companion_role": "personalized",
        "companion_schema": {
            "comparison_schema_version": 3,
            "manuscript_analysis_schema_version": 4,
        },
    },
    "personalized": {
        "manifest_expectations": {
            "comparison_schema_version": 3,
            "manuscript_analysis_schema_version": 4,
            "status": "complete",
            "subject_roi_configurations": 28,
            "repeat_level_records": 560,
            "condition_repeat_mean_records": 56,
            "thresholds_v_per_m": [0.2, 0.18, 0.15],
        },
        "tables": {
            "paired_personalized_vs_generic.csv": {
                "rows": 28,
                "columns": {"subject", "roi"} | PAIRED_COLUMNS,
            },
            "repeat_level_metrics.csv": {
                "rows": 560,
                "columns": {
                    "subject",
                    "roi",
                    "condition",
                    "repeat",
                    "roi_min_v_per_m",
                    "roi_mean_v_per_m",
                    "roi_median_v_per_m",
                    "roi_robust_max_p99_9_v_per_m",
                },
            },
        },
        "companion_role": "cohort",
        "companion_schema": {"analysis_schema_version": 4},
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(path: Path, expectations: dict) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Required publication input is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, expected in expectations.items():
        actual = payload.get(key)
        if actual != expected:
            raise RuntimeError(
                f"Publication input manifest has {key}={actual!r}; "
                f"expected {expected!r}"
            )
    return payload


def validate_csv(path: Path, *, expected_rows: int, required_columns: set[str]) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Required publication input is missing or empty: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(required_columns - columns)
        if missing:
            raise RuntimeError(f"{path.name} is missing columns: {missing}")
        rows = sum(1 for _ in reader)
    if rows != expected_rows:
        raise RuntimeError(
            f"{path.name} has {rows} rows; expected exactly {expected_rows}"
        )


def dependency_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("matplotlib", "numpy", "pandas", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed-in-packaging-environment"
    return versions


def write_requirements(path: Path, versions: dict[str, str]) -> None:
    lines = [
        f"# Python {versions['python']} was used when the bundle was packaged.",
    ]
    for package in ("matplotlib", "numpy", "pandas", "scipy"):
        version = versions[package]
        if version != "not-installed-in-packaging-environment":
            lines.append(f"{package}=={version}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _readme(role: str) -> str:
    companion = ROLE_CONFIG[role]["companion_role"]
    return f"""# CamCan publication-figure reconstruction bundle

This is the **{role}** half of a two-archive figure bundle. It contains the
validated source tables required by the manuscript-ready renderer, plus the
exact renderer version used for this analysis. The companion **{companion}**
archive is also required because the final figure set combines the final-132
population analysis with the personalized-versus-generic comparison.

## Integrity check

Run this from the extracted results directory:

```bash
sha256sum -c publication_inputs.sha256
```

Every line must report `OK`. Machine-readable provenance is in
`publication_figure_input_manifest.json`.

## Reconstruct all polished figures

Extract the two result archives into separate directories named `cohort` and
`personalized`, then run:

```bash
python3 cohort/publication_tools/{RENDERER_NAME} \\
  --cohort-dir cohort \\
  --personalized-dir personalized \\
  --out-dir manuscript_ready_figures
```

The renderer validates both analysis schemas and record counts before plotting.
It writes 32 figures as 400-dpi PNG and vector PDF, self-contained captions,
derived statistics tables, and `figure_revision_manifest.json`. Use `--force`
only when intentionally replacing an existing output directory.

Required Python packages and the versions present when this bundle was created
are recorded in `publication_figure_input_manifest.json` and pinned in
`publication_requirements.txt`. For pixel-level reproducibility, install those
requirements in a clean environment rather than modifying an unrelated working
environment.
"""


def package_results(role: str, results_dir: Path, renderer: Path) -> dict:
    if role not in ROLE_CONFIG:
        raise ValueError(f"Unknown package role: {role}")
    results_dir = results_dir.resolve()
    renderer = renderer.resolve()
    if not results_dir.is_dir():
        raise RuntimeError(f"Results directory does not exist: {results_dir}")
    if not renderer.is_file() or renderer.stat().st_size == 0:
        raise RuntimeError(f"Publication renderer is missing or empty: {renderer}")

    config = ROLE_CONFIG[role]
    source_paths = [results_dir / "analysis_manifest.json"]
    validate_manifest(source_paths[0], config["manifest_expectations"])
    for relative_path, table_config in config["tables"].items():
        path = results_dir / relative_path
        validate_csv(
            path,
            expected_rows=table_config["rows"],
            required_columns=table_config["columns"],
        )
        source_paths.append(path)

    tools_dir = results_dir / "publication_tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    packaged_renderer = tools_dir / RENDERER_NAME
    shutil.copy2(renderer, packaged_renderer)

    versions = dependency_versions()
    requirements_path = results_dir / "publication_requirements.txt"
    write_requirements(requirements_path, versions)
    hashed_paths = source_paths + [packaged_renderer, requirements_path]
    entries = [
        {
            "path": str(path.relative_to(results_dir)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in hashed_paths
    ]
    checksum_path = results_dir / "publication_inputs.sha256"
    checksum_path.write_text(
        "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in entries),
        encoding="utf-8",
    )

    payload = {
        "publication_input_package_schema_version": PACKAGE_SCHEMA_VERSION,
        "status": "complete",
        "package_role": role,
        "artifact_contract": "exact_source_tables_plus_versioned_renderer",
        "companion_package_role": config["companion_role"],
        "companion_schema_requirements": config["companion_schema"],
        "source_files": entries[: len(source_paths)],
        "renderer": entries[len(source_paths)],
        "requirements": entries[len(source_paths) + 1],
        "dependency_versions_at_packaging": versions,
        "reconstruction_command": (
            f"python3 cohort/publication_tools/{RENDERER_NAME} "
            "--cohort-dir cohort --personalized-dir personalized "
            "--out-dir manuscript_ready_figures"
        ),
        "expected_outputs": {
            "figure_stems": EXPECTED_FIGURE_STEMS,
            "png_count": len(EXPECTED_FIGURE_STEMS),
            "pdf_count": len(EXPECTED_FIGURE_STEMS),
            "captions": ["figure_captions.csv", "figure_captions.md"],
            "derived_tables_directory": "tables",
            "completion_manifest": "figure_revision_manifest.json",
        },
    }
    (results_dir / "publication_figure_input_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (results_dir / "README_PUBLICATION_FIGURES.md").write_text(
        _readme(role),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=sorted(ROLE_CONFIG))
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--renderer", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_results(args.role, args.results_dir, args.renderer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
