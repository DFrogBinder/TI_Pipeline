#!/usr/bin/env python3
"""Package exact source tables for the MNI-threshold v5 figure renderer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path


PACKAGE_SCHEMA_VERSION = 2
RENDERER_NAME = "build_camcan_supervisor_revision_figures_v5.py"
THRESHOLD_TABLE_NAME = "mni152_simnibs401_roi_thresholds.csv"
ROI_ORDER = [
    "Left_M1",
    "Right_DLPC",
    "Left_Hippocampus",
    "Right_Thalamus",
]
EXPECTED_FIGURE_STEMS = [
    "figure_mni152_absolute_field_summaries",
    "figure_personalization_subject_changes_left_m1_at_mni_roi_threshold",
    "figure_personalization_subject_changes_right_dlpc_at_mni_roi_threshold",
    "figure_personalization_subject_changes_left_hippocampus_at_mni_roi_threshold",
    "figure_personalization_subject_changes_right_thalamus_at_mni_roi_threshold",
    "figure_population_mean_field_and_target_offtarget_ratio_mni_relative",
    "figure_population_mean_field_offtarget_relationship_at_mni_roi_threshold",
    "figure_population_target_offtarget_relationship_at_mni_roi_threshold",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def threshold_slug(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".").replace(".", "p")


def load_threshold_table(path: Path) -> tuple[list[float], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4 or [row["roi"] for row in rows] != ROI_ORDER:
        raise RuntimeError(
            "Threshold table must contain the four ROIs in the standard order"
        )
    thresholds = [float(row["threshold_v_per_m"]) for row in rows]
    if any(str(row["simnibs_version"]) != "4.0.1" for row in rows):
        raise RuntimeError("Threshold table must identify SimNIBS 4.0.1")
    return thresholds, [threshold_slug(value) for value in thresholds]


def validate_manifest(
    path: Path,
    expected: dict[str, object],
    thresholds: list[float],
) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty publication input: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"{path.name} has {key}={payload.get(key)!r}; expected {value!r}"
            )
    actual = [float(value) for value in payload.get("thresholds_v_per_m", [])]
    for threshold in thresholds:
        if not any(abs(value - threshold) <= 1e-12 for value in actual):
            raise RuntimeError(
                f"{path.name} does not contain required threshold "
                f"{threshold:.17g} V/m"
            )
    return payload


def validate_csv(
    path: Path,
    *,
    expected_rows: int,
    required_columns: set[str],
) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty publication input: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(required_columns - columns)
        if missing:
            raise RuntimeError(
                f"{path.name} is missing exact MNI-threshold columns: {missing}"
            )
        rows = sum(1 for _ in reader)
    if rows != expected_rows:
        raise RuntimeError(
            f"{path.name} has {rows} rows; expected {expected_rows}"
        )


def dependency_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("matplotlib", "numpy", "pandas", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def write_requirements(path: Path, versions: dict[str, str]) -> None:
    lines = [f"# Packaging Python: {versions['python']}"]
    for package in ("matplotlib", "numpy", "pandas", "scipy"):
        if versions[package] != "not-installed":
            lines.append(f"{package}=={versions[package]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def role_contract(role: str, threshold_slugs: list[str]) -> dict:
    coverage = {
        f"{kind}_coverage_percent_ge_{slug}"
        for slug in threshold_slugs
        for kind in ("target", "off_target")
    }
    fields = {
        "roi_min_v_per_m",
        "roi_mean_v_per_m",
        "roi_robust_max_p99_9_v_per_m",
    }
    if role == "cohort":
        return {
            "manifest": {
                "analysis_schema_version": 4,
                "status": "complete",
                "subjects": 132,
                "repeat_level_records": 5280,
                "subject_level_records": 528,
                "mni_baselines": 4,
            },
            "tables": {
                "subject_level_repeat_mean_metrics.csv": (
                    528,
                    {"subject", "roi"} | fields | coverage,
                ),
                "mni152_baseline_metrics.csv": (
                    4,
                    {"subject", "roi"} | fields | coverage,
                ),
            },
            "companion": "personalized",
        }
    if role == "personalized":
        paired_coverage = {
            f"{metric}__{condition}_repeat_mean"
            for metric in coverage
            for condition in ("generic", "personalized")
        }
        return {
            "manifest": {
                "comparison_schema_version": 3,
                "manuscript_analysis_schema_version": 4,
                "status": "complete",
                "subject_roi_configurations": 28,
                "repeat_level_records": 560,
                "condition_repeat_mean_records": 56,
            },
            "tables": {
                "paired_personalized_vs_generic.csv": (
                    28,
                    {"subject", "roi"} | paired_coverage,
                ),
            },
            "companion": "cohort",
        }
    raise ValueError(role)


def readme(role: str, companion: str) -> str:
    return f"""# CamCan MNI-threshold publication-figure bundle

This is the **{role}** half of the exact v5 figure source bundle. The
**{companion}** archive is also required.

## Verify

```bash
sha256sum -c publication_inputs.sha256
```

## Reconstruct

Extract the archives as `cohort` and `personalized`, then run:

```bash
python3 cohort/publication_tools/{RENDERER_NAME} \\
  --cohort-dir cohort \\
  --personalized-dir personalized \\
  --mni-threshold-table cohort/publication_tools/{THRESHOLD_TABLE_NAME} \\
  --out-dir manuscript_ready_figures_v5
```

The renderer requires exact image-derived coverage metrics at the four
ROI-specific SimNIBS 4.0.1 MNI152 mean fields. It will not interpolate or
relabel the older fixed-threshold metrics.
"""


def package_results(
    role: str,
    results_dir: Path,
    renderer: Path,
    threshold_table: Path,
) -> dict:
    results_dir = results_dir.resolve()
    renderer = renderer.resolve()
    threshold_table = threshold_table.resolve()
    if not results_dir.is_dir():
        raise RuntimeError(f"Results directory does not exist: {results_dir}")
    for path in (renderer, threshold_table):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Required packaging input is missing: {path}")
    thresholds, threshold_slugs = load_threshold_table(threshold_table)
    contract = role_contract(role, threshold_slugs)
    manifest_path = results_dir / "analysis_manifest.json"
    validate_manifest(manifest_path, contract["manifest"], thresholds)
    source_paths = [manifest_path]
    for name, (rows, columns) in contract["tables"].items():
        path = results_dir / name
        validate_csv(path, expected_rows=rows, required_columns=columns)
        source_paths.append(path)

    tools_dir = results_dir / "publication_tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    packaged_renderer = tools_dir / RENDERER_NAME
    packaged_thresholds = tools_dir / THRESHOLD_TABLE_NAME
    shutil.copy2(renderer, packaged_renderer)
    shutil.copy2(threshold_table, packaged_thresholds)
    versions = dependency_versions()
    requirements = results_dir / "publication_requirements.txt"
    write_requirements(requirements, versions)
    hashed_paths = [
        *source_paths,
        packaged_renderer,
        packaged_thresholds,
        requirements,
    ]
    entries = [
        {
            "path": str(path.relative_to(results_dir)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in hashed_paths
    ]
    checksum = results_dir / "publication_inputs.sha256"
    checksum.write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in entries),
        encoding="utf-8",
    )
    payload = {
        "publication_input_package_schema_version": PACKAGE_SCHEMA_VERSION,
        "status": "complete",
        "package_role": role,
        "companion_package_role": contract["companion"],
        "artifact_contract": (
            "exact_mni_threshold_source_tables_plus_versioned_renderer"
        ),
        "source_files": entries,
        "thresholds_v_per_m": thresholds,
        "threshold_policy": "ROI-specific SimNIBS 4.0.1 MNI152 mean fields",
        "dependency_versions_at_packaging": versions,
        "expected_outputs": {
            "figure_stems": EXPECTED_FIGURE_STEMS,
            "png_count": len(EXPECTED_FIGURE_STEMS),
            "pdf_count": len(EXPECTED_FIGURE_STEMS),
        },
        "reconstruction_command": (
            f"python3 cohort/publication_tools/{RENDERER_NAME} "
            "--cohort-dir cohort --personalized-dir personalized "
            f"--mni-threshold-table cohort/publication_tools/{THRESHOLD_TABLE_NAME} "
            "--out-dir manuscript_ready_figures_v5"
        ),
    }
    (results_dir / "publication_figure_input_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (results_dir / "README_PUBLICATION_FIGURES.md").write_text(
        readme(role, contract["companion"]),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("cohort", "personalized"))
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--renderer", required=True, type=Path)
    parser.add_argument("--mni-threshold-table", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_results(
        args.role,
        args.results_dir,
        args.renderer,
        args.mni_threshold_table,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
