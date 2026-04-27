"""
Batch runner for repeated dataset roots such as:

    /rootDIR/Left_Hippocampus_Data_01
    /rootDIR/Left_Hippocampus_Data_02
    ...
    /rootDIR/Left_Hippocampus_Data_10

This wrapper discovers each repeat directory under a shared parent root and then
reuses the existing single-dataset pipeline from run_post_processing.py. The ROI
name is therefore inferred from each dataset directory name and is not hard-coded.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post.run_post_processing import (
    PipelineConfig,
    apply_cli_overrides,
    discover_subjects as discover_dataset_subjects,
    make_default_config,
    run_pipeline,
)
from utils.ti_utils import normalize_roi_name

REPEAT_DATASET_PATTERN = re.compile(r"^(?P<roi_prefix>.+)_Data_(?P<repeat>\d+)$")


@dataclass(frozen=True)
class RepeatDataset:
    root: Path
    name: str
    roi_prefix: str
    repeat_id: str


@dataclass
class RepeatBatchConfig:
    batch_root: str
    dataset_glob: str = "*_Data_*"
    repeats: Optional[List[str]] = None
    continue_on_error: bool = True
    summary_filename: Optional[str] = "post_processing_batch_summary.json"
    run_repeatability: bool = True
    repeatability_output_dir: Optional[str] = "repeatability_analysis"
    repeatability_logs_root: Optional[str] = None
    complete_repeat_subjects_only: bool = True


def _parse_repeat_value(value: str | int) -> int:
    try:
        repeat_value = int(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"Invalid repeat identifier {value!r}; expected an integer.") from exc

    if repeat_value < 0:
        raise ValueError(f"Repeat identifier must be non-negative, got {repeat_value}.")
    return repeat_value


def discover_repeat_datasets(
    batch_root: Path,
    dataset_glob: str = "*_Data_*",
    repeats: Optional[Iterable[str]] = None,
) -> List[RepeatDataset]:
    selected_repeats = None
    if repeats is not None:
        selected_repeats = {_parse_repeat_value(repeat) for repeat in repeats}

    datasets: List[RepeatDataset] = []
    for path in batch_root.glob(dataset_glob):
        if not path.is_dir():
            continue

        match = REPEAT_DATASET_PATTERN.match(path.name)
        if not match:
            continue

        repeat_value = _parse_repeat_value(match.group("repeat"))
        if selected_repeats is not None and repeat_value not in selected_repeats:
            continue

        datasets.append(
            RepeatDataset(
                root=path.resolve(),
                name=path.name,
                roi_prefix=match.group("roi_prefix"),
                repeat_id=match.group("repeat"),
            )
        )

    datasets.sort(
        key=lambda item: (
            item.roi_prefix.casefold(),
            _parse_repeat_value(item.repeat_id),
            item.name.casefold(),
        )
    )
    return datasets


def build_dataset_pipeline_config(dataset_root: Path, template: PipelineConfig) -> PipelineConfig:
    cfg = deepcopy(template)
    cfg.post.root = str(dataset_root)
    return cfg


def _resolve_summary_path(cfg: RepeatBatchConfig, batch_root: Path) -> Optional[Path]:
    if not cfg.summary_filename:
        return None

    summary_path = Path(cfg.summary_filename).expanduser()
    if not summary_path.is_absolute():
        summary_path = batch_root / summary_path
    return summary_path


def _resolve_repeatability_output_root(cfg: RepeatBatchConfig, batch_root: Path) -> Optional[Path]:
    if not cfg.repeatability_output_dir:
        return None
    output_root = Path(cfg.repeatability_output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = batch_root / output_root
    return output_root


def _resolve_population_output_root(pipeline_template: PipelineConfig) -> Optional[Path]:
    if not pipeline_template.population.out_dir:
        return None
    return Path(pipeline_template.population.out_dir).expanduser()


def _subject_has_required_outputs(
    dataset_root: Path,
    subject: str,
    *,
    region_filename: str,
    metrics_filename: str,
) -> bool:
    post_root = dataset_root / subject / "anat" / "post"
    return (post_root / region_filename).is_file() and (post_root / metrics_filename).is_file()


def _collect_complete_repeat_subjects(
    *,
    results: list[dict],
    pipeline_template: PipelineConfig,
) -> dict[str, list[str]]:
    by_roi: dict[str, list[set[str]]] = {}
    for result in results:
        if result["status"] != "ok":
            continue
        roi_name = result.get("resolved_target_roi")
        if not roi_name:
            continue
        dataset_root = Path(str(result["dataset_root"])).expanduser().resolve()
        subject_candidates = discover_dataset_subjects(dataset_root, pipeline_template.post.subjects)
        eligible_subjects = {
            subject
            for subject in subject_candidates
            if _subject_has_required_outputs(
                dataset_root,
                subject,
                region_filename=pipeline_template.population.region_filename,
                metrics_filename=pipeline_template.population.metrics_filename,
            )
        }
        by_roi.setdefault(str(roi_name), []).append(eligible_subjects)

    complete_subjects_by_roi: dict[str, list[str]] = {}
    for roi_name, subject_sets in by_roi.items():
        if not subject_sets:
            complete_subjects_by_roi[roi_name] = []
            continue
        complete_subjects_by_roi[roi_name] = sorted(set.intersection(*subject_sets))
    return complete_subjects_by_roi


def _write_complete_subject_manifest(
    *,
    batch_root: Path,
    roi_name: str,
    subjects: Sequence[str],
) -> Path:
    manifest_path = batch_root / f"complete_repeat_subjects__{normalize_roi_name(roi_name)}.txt"
    manifest_path.write_text("\n".join(subjects) + ("\n" if subjects else ""), encoding="utf-8")
    return manifest_path


def _rerun_population_for_complete_subjects(
    *,
    results: list[dict],
    pipeline_template: PipelineConfig,
    complete_subjects_by_roi: dict[str, list[str]],
) -> list[dict]:
    from post.post_population import run_population

    population_results: list[dict] = []
    population_out_dir = _resolve_population_output_root(pipeline_template)
    template_region_csv = (
        Path(pipeline_template.population.template_region_csv).expanduser()
        if pipeline_template.population.template_region_csv
        else None
    )

    for result in results:
        if result["status"] != "ok":
            continue

        roi_name = str(result.get("resolved_target_roi") or "")
        dataset_root = Path(str(result["dataset_root"])).expanduser().resolve()
        subjects = complete_subjects_by_roi.get(roi_name, [])

        if not subjects:
            population_results.append(
                {
                    "dataset_root": str(dataset_root),
                    "roi_name": roi_name,
                    "status": "skipped",
                    "reason": "No complete-case subjects were available across all selected repeats.",
                }
            )
            continue

        try:
            output_path = run_population(
                root=dataset_root,
                subjects=subjects,
                out_dir=population_out_dir,
                region_filename=pipeline_template.population.region_filename,
                metrics_filename=pipeline_template.population.metrics_filename,
                peak_threshold=pipeline_template.population.peak_threshold,
                target_roi=roi_name or (pipeline_template.population.target_roi or "Hippocampus"),
                template_region_csv=template_region_csv,
            )
            population_results.append(
                {
                    "dataset_root": str(dataset_root),
                    "roi_name": roi_name,
                    "status": "ok",
                    "subjects_used": len(subjects),
                    "output_dir": str(output_path),
                }
            )
        except Exception as exc:
            population_results.append(
                {
                    "dataset_root": str(dataset_root),
                    "roi_name": roi_name,
                    "status": "failed",
                    "subjects_used": len(subjects),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return population_results


def _run_repeatability_stage(
    *,
    batch_root: Path,
    cfg: RepeatBatchConfig,
    results: List[dict],
) -> List[dict]:
    if not cfg.run_repeatability:
        return []

    from post.repeatability.analyze_subject_metrics import run_analysis

    by_roi: dict[str, list[dict]] = {}
    for result in results:
        if result["status"] != "ok":
            continue
        roi_name = result.get("resolved_target_roi")
        if not roi_name:
            continue
        by_roi.setdefault(str(roi_name), []).append(result)

    if not by_roi:
        return []

    output_root = _resolve_repeatability_output_root(cfg, batch_root)
    repeatability_results = []
    for roi_name, roi_results in sorted(by_roi.items()):
        roi_output_dir = None
        if output_root is not None:
            roi_output_dir = output_root / normalize_roi_name(roi_name)
        try:
            analysis_result = run_analysis(
                dataset_root=batch_root,
                roi=roi_name,
                output_dir=roi_output_dir,
                logs_root=cfg.repeatability_logs_root,
                complete_case_only=cfg.complete_repeat_subjects_only,
            )
            repeatability_results.append(
                {
                    "roi_name": roi_name,
                    "dataset_count": len(roi_results),
                    "status": "ok",
                    **analysis_result,
                }
            )
        except Exception as exc:
            repeatability_results.append(
                {
                    "roi_name": roi_name,
                    "dataset_count": len(roi_results),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return repeatability_results


def run_repeat_batch(cfg: RepeatBatchConfig, pipeline_template: PipelineConfig) -> dict:
    batch_root = Path(cfg.batch_root).expanduser().resolve()
    if not batch_root.is_dir():
        raise SystemExit(f"Batch root directory not found: {batch_root}")

    datasets = discover_repeat_datasets(
        batch_root=batch_root,
        dataset_glob=cfg.dataset_glob,
        repeats=cfg.repeats,
    )
    if not datasets:
        raise SystemExit(
            f"No repeat datasets matched '{cfg.dataset_glob}' under {batch_root}. "
            "Expected names such as 'Left_Hippocampus_Data_01'."
        )

    print(f"[INFO] Found {len(datasets)} dataset(s) under {batch_root}.")

    results = []
    failed = []
    defer_population_until_complete_case = (
        cfg.complete_repeat_subjects_only and pipeline_template.population.enabled
    )

    for index, dataset in enumerate(datasets, start=1):
        print(f"[INFO] Dataset {index}/{len(datasets)}: {dataset.name}")
        dataset_cfg = build_dataset_pipeline_config(dataset.root, pipeline_template)
        if defer_population_until_complete_case:
            dataset_cfg.population.enabled = False

        status = "ok"
        error = None
        try:
            run_pipeline(dataset_cfg)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            failed.append(dataset.name)
            print(f"[WARN] Dataset failed: {dataset.name} -> {error}")

        results.append(
            {
                "dataset_root": str(dataset.root),
                "dataset_name": dataset.name,
                "roi_prefix": dataset.roi_prefix,
                "repeat_id": dataset.repeat_id,
                "status": status,
                "resolved_plot_roi": dataset_cfg.post.plot_roi,
                "resolved_target_roi": dataset_cfg.population.target_roi,
                "error": error,
            }
        )

        if failed and not cfg.continue_on_error:
            break

    summary = {
        "batch_root": str(batch_root),
        "dataset_glob": cfg.dataset_glob,
        "repeats": list(cfg.repeats) if cfg.repeats is not None else None,
        "total_datasets": len(datasets),
        "processed_datasets": sum(1 for item in results if item["status"] == "ok"),
        "failed_datasets": len(failed),
        "complete_repeat_subjects_only": cfg.complete_repeat_subjects_only,
        "results": results,
    }

    complete_subjects_by_roi = (
        _collect_complete_repeat_subjects(results=results, pipeline_template=pipeline_template)
        if cfg.complete_repeat_subjects_only
        else {}
    )
    if complete_subjects_by_roi:
        manifest_rows = []
        for roi_name, subjects in sorted(complete_subjects_by_roi.items()):
            manifest_path = _write_complete_subject_manifest(
                batch_root=batch_root,
                roi_name=roi_name,
                subjects=subjects,
            )
            manifest_rows.append(
                {
                    "roi_name": roi_name,
                    "n_subjects": len(subjects),
                    "manifest_path": str(manifest_path),
                }
            )
        summary["complete_repeat_subjects"] = manifest_rows

    if defer_population_until_complete_case:
        summary["population_results"] = _rerun_population_for_complete_subjects(
            results=results,
            pipeline_template=pipeline_template,
            complete_subjects_by_roi=complete_subjects_by_roi,
        )

    summary["repeatability_results"] = _run_repeatability_stage(
        batch_root=batch_root,
        cfg=cfg,
        results=results,
    )

    summary_path = _resolve_summary_path(cfg, batch_root)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote batch summary to: {summary_path}")

    if failed:
        raise SystemExit(f"{len(failed)} dataset(s) failed during batch post-processing.")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument(
        "--batch-root",
        default=None,
        help="Parent directory containing repeat datasets such as Left_Hippocampus_Data_01.",
    )
    parser.add_argument(
        "--dataset-glob",
        default=None,
        help="Glob used within batch-root to find dataset directories. Default: *_Data_*",
    )
    parser.add_argument(
        "--repeats",
        nargs="*",
        default=None,
        help="Optional repeat identifiers to include, for example: --repeats 01 02 10",
    )
    parser.add_argument(
        "--summary-filename",
        default=None,
        help="Optional summary filename or absolute path. Use an empty string to disable writing.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first dataset failure instead of continuing through the batch.",
    )
    parser.add_argument(
        "--skip-repeatability",
        action="store_true",
        help="Skip the final across-repeat analysis stage.",
    )
    parser.add_argument(
        "--repeatability-output-dir",
        default=None,
        help=(
            "Optional output directory root for repeatability analysis. "
            "A per-ROI subdirectory will be created under this path."
        ),
    )
    parser.add_argument(
        "--repeatability-logs-root",
        default=None,
        help="Optional logs root for the repeatability analysis stage.",
    )
    parser.add_argument(
        "--allow-incomplete-repeat-subjects",
        action="store_true",
        help=(
            "Allow within-run population summaries and repeatability descriptive outputs to include "
            "subjects that are not present in every selected repeat. By default, only the complete-case "
            "cohort shared across all selected repeats is used."
        ),
    )
    parser.add_argument(
        "--atlas-filename",
        "--fastsurfer-atlas-filename",
        dest="fastsurfer_atlas_filename",
        help=(
            "Atlas override for all subjects. Use either an absolute atlas path "
            "shared by every subject, or a relative path under each subject "
            "directory inside fastsurfer_root, for example "
            "'mri/aparc.DKTatlas+aseg.deep.nii.gz'."
        ),
    )
    return parser


def make_default_pipeline_template() -> PipelineConfig:
    cfg = make_default_config()
    cfg.post.root = "/path/to/rootDIR/Example_Target_Data_01"
    cfg.post.fastsurfer_root = "/mnt/parscratch/users/cop23bi/ZIPs/atlases"
    cfg.post.plot_roi = None
    cfg.population.target_roi = None
    return cfg


def make_default_batch_config() -> RepeatBatchConfig:
    return RepeatBatchConfig(
        batch_root="/path/to/rootDIR",
        dataset_glob="*_Data_*",
        repeats=[f"{idx:02d}" for idx in range(1, 11)],
        continue_on_error=True,
        summary_filename="post_processing_batch_summary.json",
        run_repeatability=True,
        repeatability_output_dir="repeatability_analysis",
        repeatability_logs_root=None,
    )


def apply_batch_cli_overrides(cfg: RepeatBatchConfig, args: argparse.Namespace) -> None:
    if args.batch_root is not None:
        cfg.batch_root = args.batch_root
    if args.dataset_glob is not None:
        cfg.dataset_glob = args.dataset_glob
    if args.repeats is not None:
        cfg.repeats = args.repeats
    if args.summary_filename is not None:
        cfg.summary_filename = args.summary_filename or None
    if args.stop_on_error:
        cfg.continue_on_error = False
    if args.skip_repeatability:
        cfg.run_repeatability = False
    if args.repeatability_output_dir is not None:
        cfg.repeatability_output_dir = args.repeatability_output_dir or None
    if args.repeatability_logs_root is not None:
        cfg.repeatability_logs_root = args.repeatability_logs_root or None
    if args.allow_incomplete_repeat_subjects:
        cfg.complete_repeat_subjects_only = False


def main(argv: Optional[Sequence[str]] = None) -> None:
    batch_cfg = make_default_batch_config()
    pipeline_template = make_default_pipeline_template()

    args = build_arg_parser().parse_args(argv)
    apply_cli_overrides(pipeline_template, args)
    apply_batch_cli_overrides(batch_cfg, args)

    run_repeat_batch(batch_cfg, pipeline_template)


if __name__ == "__main__":
    main()
