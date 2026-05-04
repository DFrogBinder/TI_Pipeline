#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate paired repeatability experiment outputs before post-processing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from experiment_config import (  # noqa: E402
    ExperimentConfig,
    condition_by_name,
    load_experiment_config,
    repeat_tag,
    subject_analysis_root,
    subject_condition_mesh_cache_root,
    subject_condition_repeats_root,
)


def _exists_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _first_match(parent: Path, pattern: str) -> Path | None:
    if not parent.is_dir():
        return None
    matches = sorted(parent.glob(pattern))
    return matches[0] if matches else None


def _load_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _repeat_paths(config: ExperimentConfig, subject: str, condition_name: str, repeat_index: int) -> dict[str, Path]:
    tag = repeat_tag(repeat_index)
    repeat_root = subject_condition_repeats_root(config, subject, condition_name) / tag
    anat_dir = repeat_root / subject / "anat"
    sim_root = anat_dir / "SimNIBS"
    output_root = sim_root / "Output" / subject
    mesh_dir = anat_dir / f"m2m_{subject}"
    return {
        "repeat_root": repeat_root,
        "task_manifest": repeat_root / "task_manifest.json",
        "anat_dir": anat_dir,
        "mesh_dir": mesh_dir,
        "head_mesh": mesh_dir / f"{subject}.msh",
        "sim_root": sim_root,
        "ti_brain_only": sim_root / "ti_brain_only.nii.gz",
        "output_root": output_root,
        "ti_msh": output_root / "TI.msh",
        "volume_labels": output_root / "Volume_Labels",
        "volume_base": output_root / "Volume_Base",
        "volume_masks": output_root / "Volume_Maks",
    }


def _analysis_paths(config: ExperimentConfig, subject: str, condition_name: str) -> dict[str, Path]:
    condition_root = subject_analysis_root(config, subject) / condition_name
    return {
        "condition_root": condition_root,
        "summary_csv": condition_root / "summary.csv",
        "summary_json": condition_root / "summary.json",
        "repeatability_csv": condition_root / "repeatability_stats.csv",
        "repeatability_json": condition_root / "repeatability_stats.json",
        "report_md": condition_root / "repeatability_report.md",
        "parameter_json": condition_root / "parameter_consistency.json",
        "diff_freq": condition_root / "label_diff_frequency.nii.gz",
        "diff_overlay": condition_root / "label_diff_overlay.png",
        "roi_mask": condition_root / "roi_mask_on_t1.nii.gz",
        "roi_t1": condition_root / "roi_outline_on_t1.png",
        "roi_mean_ti": condition_root / "roi_outline_on_mean_ti.png",
        "repeat_qc_dir": condition_root / "repeat_qc",
        "median_roi_plot": condition_root / "median_roi_by_repeat.png",
        "mesh_nodes_plot": condition_root / "mesh_nodes_by_repeat.png",
        "mesh_nodes_bar": condition_root / "mesh_nodes_by_repeat_bar.png",
    }


def _paired_analysis_paths(config: ExperimentConfig, subject: str) -> dict[str, Path]:
    subject_root = subject_analysis_root(config, subject)
    batch_root = config.experiment_root / "_analysis"
    return {
        "subject_root": subject_root,
        "comparison_json": subject_root / "condition_comparison.json",
        "comparison_md": subject_root / "condition_comparison.md",
        "std_plot": subject_root / "condition_metric_std_comparison.png",
        "cv_plot": subject_root / "condition_metric_cv_comparison.png",
        "median_plot": subject_root / "median_roi_by_condition.png",
        "mesh_plot": subject_root / "mesh_nodes_by_condition.png",
        "batch_json": batch_root / "paired_condition_summary.json",
        "batch_csv": batch_root / "paired_condition_summary.csv",
    }


def _check_repeat(
    config: ExperimentConfig,
    *,
    subject: str,
    condition_name: str,
    mesh_mode: str,
    repeat_index: int,
) -> dict[str, object]:
    tag = repeat_tag(repeat_index)
    paths = _repeat_paths(config, subject, condition_name, repeat_index)
    missing: list[str] = []
    warnings: list[str] = []

    required_paths = {
        "repeat_root": paths["repeat_root"],
        "task_manifest": paths["task_manifest"],
        "anat_dir": paths["anat_dir"],
        "head_mesh": paths["head_mesh"],
        "ti_brain_only": paths["ti_brain_only"],
        "ti_msh": paths["ti_msh"],
    }
    for key, path in required_paths.items():
        if key in {"repeat_root", "anat_dir"}:
            if not path.is_dir():
                missing.append(key)
        elif not _exists_nonempty(path):
            missing.append(key)

    label_volume = _first_match(paths["volume_labels"], "TI_Volumetric_*")
    base_volume = _first_match(paths["volume_base"], "TI_Volumetric_*")
    masks_volume = _first_match(paths["volume_masks"], "TI_Volumetric_*")
    if label_volume is None:
        missing.append("volume_labels")
    if base_volume is None:
        missing.append("volume_base")
    if masks_volume is None:
        warnings.append("volume_masks_missing")

    sim_mat = _first_match(paths["output_root"], "simnibs_simulation_*.mat")
    if sim_mat is None:
        warnings.append("simulation_mat_missing")

    manifest = _load_json(paths["task_manifest"])
    if manifest is None:
        warnings.append("task_manifest_unreadable")
    else:
        if manifest.get("repeat_tag") != tag:
            warnings.append("task_manifest_repeat_mismatch")
        if manifest.get("condition") != condition_name:
            warnings.append("task_manifest_condition_mismatch")

    if mesh_mode == "fixed_mesh":
        expected_cache_mesh = (
            subject_condition_mesh_cache_root(config, subject, condition_name)
            / f"m2m_{subject}"
            / f"{subject}.msh"
        )
        if not paths["mesh_dir"].exists():
            missing.append("repeat_mesh_link")
        elif not paths["mesh_dir"].is_symlink():
            warnings.append("repeat_mesh_not_symlink")
        else:
            try:
                if paths["mesh_dir"].resolve() != expected_cache_mesh.parent.resolve():
                    warnings.append("repeat_mesh_link_target_unexpected")
            except FileNotFoundError:
                missing.append("repeat_mesh_link_broken")

    status = "complete" if not missing else "incomplete"
    return {
        "subject": subject,
        "condition": condition_name,
        "mesh_mode": mesh_mode,
        "repeat_index": repeat_index,
        "repeat_tag": tag,
        "status": status,
        "missing": missing,
        "warnings": warnings,
    }


def _check_condition(
    config: ExperimentConfig,
    *,
    subject: str,
    condition_name: str,
    check_analysis: bool,
) -> dict[str, object]:
    condition = condition_by_name(config, condition_name)
    condition_root = subject_condition_repeats_root(config, subject, condition_name).parent
    manifest_path = condition_root / "condition_manifest.json"
    missing: list[str] = []
    warnings: list[str] = []

    if not condition_root.is_dir():
        missing.append("condition_root")
    if not _exists_nonempty(manifest_path):
        missing.append("condition_manifest")

    cache_info: dict[str, object] | None = None
    if condition.mesh_mode == "fixed_mesh":
        cache_anat = subject_condition_mesh_cache_root(config, subject, condition_name)
        cache_mesh_dir = cache_anat / f"m2m_{subject}"
        cache_mesh = cache_mesh_dir / f"{subject}.msh"
        cache_ready = cache_anat / ".mesh_ready.json"
        cache_info = {
            "cache_anat_dir": str(cache_anat),
            "cache_mesh": str(cache_mesh),
            "cache_ready_marker": str(cache_ready),
            "cache_complete": _exists_nonempty(cache_mesh),
            "ready_marker_present": _exists_nonempty(cache_ready),
        }
        if not _exists_nonempty(cache_mesh):
            missing.append("mesh_cache_mesh")
        if not _exists_nonempty(cache_ready):
            warnings.append("mesh_cache_ready_marker_missing")

    repeat_reports = [
        _check_repeat(
            config,
            subject=subject,
            condition_name=condition_name,
            mesh_mode=condition.mesh_mode,
            repeat_index=repeat_index,
        )
        for repeat_index in range(1, condition.repeat_count + 1)
    ]
    complete_repeats = sum(1 for row in repeat_reports if row["status"] == "complete")
    incomplete_repeats = condition.repeat_count - complete_repeats

    analysis_info: dict[str, object] | None = None
    if check_analysis:
        analysis_paths = _analysis_paths(config, subject, condition_name)
        analysis_missing: list[str] = []
        for key in (
            "summary_csv",
            "summary_json",
            "repeatability_csv",
            "repeatability_json",
            "report_md",
            "parameter_json",
            "diff_freq",
            "roi_mask",
            "roi_t1",
            "roi_mean_ti",
            "median_roi_plot",
            "mesh_nodes_plot",
        ):
            if not _exists_nonempty(analysis_paths[key]):
                analysis_missing.append(key)
        if not analysis_paths["repeat_qc_dir"].is_dir():
            analysis_missing.append("repeat_qc_dir")
            repeat_qc_count = 0
        else:
            repeat_qc_count = len(list(analysis_paths["repeat_qc_dir"].glob("*.png")))
            if repeat_qc_count < complete_repeats:
                warnings.append("repeat_qc_png_count_below_complete_repeats")
        if not _exists_nonempty(analysis_paths["mesh_nodes_bar"]):
            warnings.append("mesh_nodes_bar_plot_missing")
        if not _exists_nonempty(analysis_paths["diff_overlay"]):
            warnings.append("label_diff_overlay_missing")
        analysis_info = {
            "analysis_root": str(analysis_paths["condition_root"]),
            "analysis_complete": not analysis_missing,
            "analysis_missing": analysis_missing,
            "repeat_qc_png_count": repeat_qc_count,
        }

    return {
        "subject": subject,
        "condition": condition_name,
        "mesh_mode": condition.mesh_mode,
        "condition_root": str(condition_root),
        "expected_repeats": condition.repeat_count,
        "complete_repeats": complete_repeats,
        "incomplete_repeats": incomplete_repeats,
        "missing": missing,
        "warnings": warnings,
        "mesh_cache": cache_info,
        "repeats": repeat_reports,
        "analysis": analysis_info,
    }


def _check_paired_outputs(
    config: ExperimentConfig,
    *,
    subjects: list[str],
    condition_names: list[str],
    check_analysis: bool,
) -> dict[str, object] | None:
    if not check_analysis or len(condition_names) < 2:
        return None

    subject_reports = []
    for subject in subjects:
        paths = _paired_analysis_paths(config, subject)
        missing: list[str] = []
        for key in (
            "comparison_json",
            "comparison_md",
            "std_plot",
            "cv_plot",
            "median_plot",
            "mesh_plot",
        ):
            if not _exists_nonempty(paths[key]):
                missing.append(key)
        subject_reports.append(
            {
                "subject": subject,
                "analysis_root": str(paths["subject_root"]),
                "status": "complete" if not missing else "incomplete",
                "missing": missing,
            }
        )

    batch_paths = _paired_analysis_paths(config, subjects[0] if subjects else config.subjects[0])
    batch_missing = [
        key
        for key in ("batch_json", "batch_csv")
        if not _exists_nonempty(batch_paths[key])
    ]
    return {
        "subject_reports": subject_reports,
        "batch_status": "complete" if not batch_missing else "incomplete",
        "batch_missing": batch_missing,
        "batch_root": str((config.experiment_root / "_analysis").resolve()),
    }


def _build_summary(
    *,
    config: ExperimentConfig,
    subject_reports: list[dict[str, object]],
    paired_outputs: dict[str, object] | None,
) -> dict[str, object]:
    expected_repeats = sum(int(row["expected_repeats"]) for row in subject_reports)
    complete_repeats = sum(int(row["complete_repeats"]) for row in subject_reports)
    incomplete_repeats = expected_repeats - complete_repeats
    condition_issues = sum(1 for row in subject_reports if row["missing"] or row["incomplete_repeats"])
    analysis_condition_issues = sum(
        1
        for row in subject_reports
        if row.get("analysis") and (
            row["analysis"]["analysis_missing"]  # type: ignore[index]
        )
    )
    paired_subject_issues = 0
    if paired_outputs is not None:
        paired_subject_issues = sum(
            1 for row in paired_outputs["subject_reports"] if row["status"] != "complete"
        )

    return {
        "config_path": str(config.config_path),
        "experiment_root": str(config.experiment_root),
        "subjects": len({row["subject"] for row in subject_reports}),
        "conditions": len(subject_reports),
        "expected_repeats": expected_repeats,
        "complete_repeats": complete_repeats,
        "incomplete_repeats": incomplete_repeats,
        "condition_issues": condition_issues,
        "analysis_condition_issues": analysis_condition_issues,
        "paired_subject_issues": paired_subject_issues,
    }


def _print_report(summary: dict[str, object], subject_reports: list[dict[str, object]], paired_outputs: dict[str, object] | None) -> None:
    print("Validation Summary")
    print(f"- Config: {summary['config_path']}")
    print(f"- Experiment root: {summary['experiment_root']}")
    print(f"- Subjects checked: {summary['subjects']}")
    print(f"- Condition groups checked: {summary['conditions']}")
    print(f"- Expected repeats: {summary['expected_repeats']}")
    print(f"- Complete repeats: {summary['complete_repeats']}")
    print(f"- Incomplete repeats: {summary['incomplete_repeats']}")

    print("\nCondition Status")
    for row in subject_reports:
        status = "OK"
        if row["missing"] or row["incomplete_repeats"]:
            status = "ISSUES"
        print(
            f"- {row['subject']} / {row['condition']}: {status} | "
            f"{row['complete_repeats']}/{row['expected_repeats']} repeats complete"
        )
        if row["missing"]:
            print(f"  missing condition-level items: {', '.join(row['missing'])}")
        if row["warnings"]:
            print(f"  warnings: {', '.join(row['warnings'])}")
        incomplete = [rep for rep in row["repeats"] if rep["status"] != "complete"]
        if incomplete:
            preview = ", ".join(
                f"{rep['repeat_tag']}[{','.join(rep['missing'])}]"
                for rep in incomplete[:6]
            )
            suffix = " ..." if len(incomplete) > 6 else ""
            print(f"  incomplete repeats: {preview}{suffix}")
        if row.get("analysis"):
            analysis = row["analysis"]
            analysis_status = "OK" if analysis["analysis_complete"] else "ISSUES"
            print(
                f"  analysis outputs: {analysis_status} | "
                f"repeat_qc_png_count={analysis['repeat_qc_png_count']}"
            )
            if analysis["analysis_missing"]:
                print(f"  missing analysis items: {', '.join(analysis['analysis_missing'])}")

    if paired_outputs is not None:
        print("\nPaired Analysis Status")
        for row in paired_outputs["subject_reports"]:
            if row["status"] == "complete":
                print(f"- {row['subject']}: OK")
            else:
                print(f"- {row['subject']}: ISSUES | missing: {', '.join(row['missing'])}")
        batch_status = paired_outputs["batch_status"]
        if batch_status == "complete":
            print(f"- Batch summary: OK | {paired_outputs['batch_root']}")
        else:
            print(
                f"- Batch summary: ISSUES | missing: {', '.join(paired_outputs['batch_missing'])}"
            )


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def _save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate paired repeatability experiment outputs before post-processing."
    )
    parser.add_argument("--config", required=True, help="Path to the experiment JSON config.")
    parser.add_argument("--subjects", default=None, help="Optional comma-separated subject subset.")
    parser.add_argument("--conditions", default=None, help="Optional comma-separated condition subset.")
    parser.add_argument(
        "--check-analysis",
        action="store_true",
        help="Also check the expected outputs from post/repeatability_experiment_report.py.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the full validation report as JSON.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to save one condition-level summary row per subject/condition.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 1 if any simulation or analysis outputs are incomplete.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_experiment_config(args.config, validate_paths=False)
    subjects = (
        [value.strip() for value in args.subjects.split(",") if value.strip()]
        if args.subjects
        else config.subjects
    )
    condition_names = (
        [value.strip() for value in args.conditions.split(",") if value.strip()]
        if args.conditions
        else [condition.name for condition in config.conditions]
    )

    subject_reports = [
        _check_condition(
            config,
            subject=subject,
            condition_name=condition_name,
            check_analysis=args.check_analysis,
        )
        for subject in subjects
        for condition_name in condition_names
    ]
    paired_outputs = _check_paired_outputs(
        config,
        subjects=subjects,
        condition_names=condition_names,
        check_analysis=args.check_analysis,
    )
    summary = _build_summary(
        config=config,
        subject_reports=subject_reports,
        paired_outputs=paired_outputs,
    )

    payload = {
        "summary": summary,
        "conditions": subject_reports,
        "paired_analysis": paired_outputs,
    }
    _print_report(summary, subject_reports, paired_outputs)

    if args.output_json:
        _save_json(Path(args.output_json).expanduser().resolve(), payload)
    if args.output_csv:
        csv_rows = [
            {
                "subject": row["subject"],
                "condition": row["condition"],
                "mesh_mode": row["mesh_mode"],
                "expected_repeats": row["expected_repeats"],
                "complete_repeats": row["complete_repeats"],
                "incomplete_repeats": row["incomplete_repeats"],
                "condition_missing_count": len(row["missing"]),
                "warning_count": len(row["warnings"]),
            }
            for row in subject_reports
        ]
        _save_csv(Path(args.output_csv).expanduser().resolve(), csv_rows)

    has_issues = (
        summary["incomplete_repeats"] > 0
        or summary["condition_issues"] > 0
        or summary["analysis_condition_issues"] > 0
        or summary["paired_subject_issues"] > 0
    )
    raise SystemExit(1 if args.strict and has_issues else 0)


if __name__ == "__main__":
    main()
