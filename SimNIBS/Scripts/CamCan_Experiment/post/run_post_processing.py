"""
Single-dataset orchestration for two explicit post-processing layers:

1. Subject-level metrics
2. Population (within run)-level metrics

Across-repeats-level metrics are orchestrated separately by run_post_processing_batch.py.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post.pipeline_layers import (
    PIPELINE_STAGE_LABELS,
    POPULATION_WITHIN_RUN_STAGE,
    SUBJECT_LEVEL_STAGE,
    load_subject_metrics_payload,
    stage_failed,
    stage_ok,
    stage_partial,
    stage_skipped,
    subject_metrics_payload_complete,
)
from utils.roi_registry import (
    match_fastsurfer_roi_from_directory,
    resolve_fastsurfer_roi_label_ids,
    resolve_fastsurfer_roi_name,
)

if TYPE_CHECKING:
    from post.post_process import PostProcessConfig


def discover_subjects(root: Path, subjects: Optional[Iterable[str]]) -> List[str]:
    if subjects:
        return list(subjects)
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def should_skip_subject(out_dir: Path, pp_cfg: "PostProcessConfig", force: bool) -> bool:
    if force:
        return False
    metrics_path = out_dir / "subject_metrics.json"
    if not metrics_path.is_file():
        return False
    payload = load_subject_metrics_payload(metrics_path)
    if payload is None:
        return False
    if not subject_metrics_payload_complete(payload):
        return False
    try:
        from post.post_process import extended_metrics_fingerprint_for_cfg
        from post.metric_extensions import EXTENDED_METRIC_SCHEMA_VERSION

        expected_fingerprint = extended_metrics_fingerprint_for_cfg(pp_cfg)
    except Exception:
        return False
    meta = payload.get("extended_metrics_meta")
    if not isinstance(meta, dict):
        return False
    if meta.get("schema_version") != EXTENDED_METRIC_SCHEMA_VERSION:
        return False
    return meta.get("config_fingerprint") == expected_fingerprint

@dataclass
class PostBatchConfig:
    root: str
    subjects: Optional[List[str]] = None
    max_workers: Optional[int] = 8
    atlas_mode: str = "auto"  # "auto" | "mni" | "fastsurfer"
    fastsurfer_root: Optional[str] = None
    fs_mri_path: Optional[str] = None
    fastsurfer_atlas_filename: Optional[str] = None
    t1_path: Optional[str] = None
    plot_roi: Optional[str] = None
    percentile: float = 95.0
    hard_threshold: float = 200.0
    overlay_z_offset_mm: float = 0.0
    overlay_full_field: bool = True
    write_region_table: bool = True
    region_percentile: float = 95.0
    offtarget_threshold: float = 0.2
    mni_baseline_root: Optional[str] = None
    mni_fixed_atlas_path: Optional[str] = None
    neighbor_dilation_iter: int = 1
    csf_labels: Optional[List[int]] = None
    skull_labels: Optional[List[int]] = None
    electrode_csv: Optional[str] = None
    electrode_names: Optional[List[str]] = None
    eeg_positions_path_template: Optional[str] = None
    write_neighbor_table: bool = True
    write_neighbor_visualization: bool = True
    write_electrode_table: bool = True
    force: bool = False
    verbose: bool = True


@dataclass
class PopulationConfig:
    enabled: bool = True
    out_dir: Optional[str] = None
    region_filename: str = "region_stats_fastsurfer.csv"
    metrics_filename: str = "subject_metrics.json"
    peak_threshold: float = 0.2
    target_roi: Optional[str] = None
    template_region_csv: Optional[str] = None


@dataclass
class PipelineConfig:
    post: PostBatchConfig
    population: PopulationConfig


def resolve_subject_fastsurfer_atlas_path(cfg: PostBatchConfig, subject: str) -> Optional[str]:
    if not cfg.fastsurfer_atlas_filename:
        return cfg.fs_mri_path

    atlas_filename = Path(cfg.fastsurfer_atlas_filename).expanduser()
    if atlas_filename.is_absolute():
        return str(atlas_filename)

    if not cfg.fastsurfer_root:
        raise ValueError(
            "A relative atlas filename override requires fastsurfer_root to be set."
        )

    return str(Path(cfg.fastsurfer_root).expanduser() / subject / atlas_filename)


def _label_ids_for_roi(roi_name: str) -> list[int]:
    return [int(label_id) for label_id in resolve_fastsurfer_roi_label_ids(roi_name)]


def validate_post_batch_config(cfg: PostBatchConfig) -> None:
    if cfg.mni_baseline_root:
        baseline_root = Path(cfg.mni_baseline_root).expanduser()
        if not baseline_root.exists():
            raise SystemExit(
                f"Configured mni_baseline_root does not exist: {baseline_root}. "
                "MNI baseline comparisons require a baseline root containing "
                "anat/SimNIBS/ti_brain_only.nii.gz, either directly or below an MNI subject directory."
            )
        if not cfg.mni_fixed_atlas_path:
            raise SystemExit(
                "Configured mni_baseline_root without mni_fixed_atlas_path. "
                "MNI baseline comparisons require both paths: the baseline simulation root "
                "and the fixed MNI FastSurfer atlas used to build the ROI mask."
            )

    if cfg.mni_fixed_atlas_path:
        atlas_path = Path(cfg.mni_fixed_atlas_path).expanduser()
        if not atlas_path.is_file():
            raise SystemExit(
                f"Configured mni_fixed_atlas_path is not a file: {atlas_path}. "
                "This path is used for fixed-template neighbor metrics and MNI baseline ROI masks."
            )

    if cfg.electrode_csv:
        electrode_csv = Path(cfg.electrode_csv).expanduser()
        if not electrode_csv.is_file():
            raise SystemExit(
                f"Configured electrode_csv is not a file: {electrode_csv}. "
                "Required columns are subject,electrode,x,y,z."
            )
        with electrode_csv.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
        required = {"subject", "electrode", "x", "y", "z"}
        missing = sorted(required - set(header))
        if missing:
            raise SystemExit(
                f"Configured electrode_csv is missing required column(s): {', '.join(missing)}. "
                "Expected columns: subject,electrode,x,y,z. Coordinates must be millimetres "
                "in the same world coordinate frame as the subject TI image."
            )


def build_post_process_config(root: Path, subject: str, cfg: PostBatchConfig) -> PostProcessConfig:
    from post.post_process import PostProcessConfig

    return PostProcessConfig(
        root_dir=str(root),
        subject=subject,
        atlas_mode=cfg.atlas_mode,
        fastsurfer_root=cfg.fastsurfer_root,
        fs_mri_path=resolve_subject_fastsurfer_atlas_path(cfg, subject),
        t1_path=cfg.t1_path,
        plot_roi=cfg.plot_roi or "ctx-lh-precentral",
        percentile=cfg.percentile,
        hard_threshold=cfg.hard_threshold,
        overlay_z_offset_mm=cfg.overlay_z_offset_mm,
        overlay_full_field=cfg.overlay_full_field,
        write_region_table=cfg.write_region_table,
        region_percentile=cfg.region_percentile,
        offtarget_threshold=cfg.offtarget_threshold,
        mni_baseline_root=cfg.mni_baseline_root,
        mni_fixed_atlas_path=cfg.mni_fixed_atlas_path,
        neighbor_dilation_iter=cfg.neighbor_dilation_iter,
        csf_labels=cfg.csf_labels,
        skull_labels=cfg.skull_labels,
        electrode_csv=cfg.electrode_csv,
        electrode_names=cfg.electrode_names,
        eeg_positions_path_template=cfg.eeg_positions_path_template,
        write_neighbor_table=cfg.write_neighbor_table,
        write_neighbor_visualization=cfg.write_neighbor_visualization,
        write_electrode_table=cfg.write_electrode_table,
        verbose=cfg.verbose,
    )


def resolve_max_workers(cfg: PostBatchConfig, task_count: int) -> int:
    if task_count <= 0:
        return 0

    if cfg.max_workers is not None:
        requested = cfg.max_workers
    else:
        requested = _read_positive_int_env("POST_MAX_WORKERS")
        if requested is None:
            requested = _read_positive_int_env("SLURM_CPUS_PER_TASK")
        if requested is None:
            requested = os.cpu_count() or 1

    if requested < 1:
        raise ValueError("max_workers must be at least 1.")
    return min(task_count, requested)


def _read_positive_int_env(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None

    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc

    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}.")

    return value


def process_subject(pp_cfg: PostProcessConfig) -> dict:
    from post.post_process import run_post_process

    result = run_post_process(pp_cfg)
    return {
        "subject": pp_cfg.subject,
        "extended_status": result["extended_metrics_meta"]["status"],
        "qc_status": result["qc_meta"]["status"],
        "subject_status": result["subject_status"],
        "metrics_path": result["metrics_path"],
    }


def _uses_fastsurfer_aliases(cfg: PostBatchConfig) -> bool:
    return cfg.atlas_mode in {"auto", "fastsurfer"}


def _abort_unknown_roi(dataset_root: str, reason: str) -> None:
    dataset_name = Path(dataset_root).name
    raise SystemExit(
        f"Unrecognized ROI for dataset '{dataset_name}': {reason} "
        "No analysis was started for this dataset."
    )


def _resolve_pipeline_rois(cfg: PipelineConfig) -> None:
    if _uses_fastsurfer_aliases(cfg.post):
        try:
            if cfg.post.plot_roi:
                plot_match = resolve_fastsurfer_roi_name(cfg.post.plot_roi)
                print(
                    f"[INFO] Using configured ROI alias '{cfg.post.plot_roi}' -> "
                    f"'{plot_match.canonical_name}' label id(s) {_label_ids_for_roi(plot_match.canonical_name)}."
                )
            else:
                plot_match = match_fastsurfer_roi_from_directory(cfg.post.root)
                print(
                    f"[INFO] Inferred ROI from directory '{Path(cfg.post.root).name}' via alias "
                    f"'{plot_match.matched_alias}' -> '{plot_match.canonical_name}' "
                    f"label id(s) {_label_ids_for_roi(plot_match.canonical_name)}."
                )
        except ValueError as exc:
            _abort_unknown_roi(cfg.post.root, str(exc))
        cfg.post.plot_roi = plot_match.canonical_name

        if cfg.population.target_roi:
            try:
                target_match = resolve_fastsurfer_roi_name(cfg.population.target_roi)
            except ValueError as exc:
                _abort_unknown_roi(cfg.post.root, str(exc))
            print(
                f"[INFO] Using configured population ROI alias '{cfg.population.target_roi}' -> "
                f"'{target_match.canonical_name}' label id(s) {_label_ids_for_roi(target_match.canonical_name)}."
            )
            cfg.population.target_roi = target_match.canonical_name
        else:
            cfg.population.target_roi = cfg.post.plot_roi
    else:
        if cfg.population.target_roi is None:
            cfg.population.target_roi = cfg.post.plot_roi or "Hippocampus"


def run_batch(cfg: PostBatchConfig) -> dict:
    validate_post_batch_config(cfg)
    root = Path(cfg.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Root directory not found: {root}")

    subjects = discover_subjects(root, cfg.subjects)
    if not subjects:
        raise SystemExit("No subjects found.")

    processed = []
    skipped = []
    failed = []
    incomplete = []
    pending = []

    for subj in subjects:
        pp_cfg = build_post_process_config(root, subj, cfg)
        out_dir = root / subj / "anat" / "post"
        if should_skip_subject(out_dir, pp_cfg, cfg.force):
            skipped.append(subj)
            continue
        pending.append(pp_cfg)

    max_workers = resolve_max_workers(cfg, len(pending))
    if pending:
        print(
            f"[INFO] Running post-processing for {len(pending)} subject(s) "
            f"with up to {max_workers} worker(s)."
        )

    if max_workers <= 1:
        for pp_cfg in pending:
            try:
                subject_result = process_subject(pp_cfg)
                if subject_result["subject_status"] == "complete":
                    processed.append(pp_cfg.subject)
                else:
                    incomplete.append((
                        pp_cfg.subject,
                        "subject_metrics_meta.status="
                        f"{subject_result['subject_status']} "
                        f"(extended={subject_result['extended_status']}, qc={subject_result['qc_status']})",
                    ))
            except Exception as exc:
                failed.append((pp_cfg.subject, f"{type(exc).__name__}: {exc}"))
    else:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=get_context("spawn"),
        ) as executor:
            future_to_subject = {
                executor.submit(process_subject, pp_cfg): pp_cfg.subject
                for pp_cfg in pending
            }
            for future in as_completed(future_to_subject):
                subj = future_to_subject[future]
                try:
                    subject_result = future.result()
                    if subject_result["subject_status"] == "complete":
                        processed.append(subj)
                    else:
                        incomplete.append((
                            subj,
                            "subject_metrics_meta.status="
                            f"{subject_result['subject_status']} "
                            f"(extended={subject_result['extended_status']}, qc={subject_result['qc_status']})",
                        ))
                except Exception as exc:
                    failed.append((subj, f"{type(exc).__name__}: {exc}"))

    processed.sort()
    skipped.sort()
    failed.sort(key=lambda item: item[0])
    incomplete.sort(key=lambda item: item[0])

    print(f"[INFO] Processed {len(processed)} subject(s).")
    if skipped:
        print(f"[INFO] Skipped {len(skipped)} subject(s) (existing outputs).")
    if incomplete:
        print(f"[WARN] Incomplete extended metrics for {len(incomplete)} subject(s).")
        for subj, err in incomplete:
            print(f"  - {subj}: {err}")
    if failed:
        print(f"[WARN] Failed {len(failed)} subject(s).")
        for subj, err in failed:
            print(f"  - {subj}: {err}")

    return {"processed": processed, "skipped": skipped, "failed": failed, "incomplete": incomplete}


def _resolve_population_output_dir(cfg: PopulationConfig) -> Optional[Path]:
    if not cfg.out_dir:
        return None
    return Path(cfg.out_dir).expanduser()


def run_subject_level_stage(cfg: PostBatchConfig) -> dict:
    print(f"[INFO] Stage: {PIPELINE_STAGE_LABELS[SUBJECT_LEVEL_STAGE]}")
    batch_result = run_batch(cfg)
    usable_subjects = batch_result["processed"] + batch_result["skipped"]
    if batch_result["failed"] or batch_result["incomplete"]:
        status_details = {
            "processed_subjects": batch_result["processed"],
            "skipped_subjects": batch_result["skipped"],
            "failed_subjects": batch_result["failed"],
            "incomplete_subjects": batch_result["incomplete"],
            "usable_subject_count": len(usable_subjects),
        }
        if usable_subjects:
            return stage_partial(
                SUBJECT_LEVEL_STAGE,
                warning=(
                    f"{len(batch_result['failed'])} subject exception(s) and "
                    f"{len(batch_result['incomplete'])} subject(s) with incomplete subject metrics/QC. "
                    "Later layers may still use the completed subject subset."
                ),
                **status_details,
            )
        return stage_failed(
            SUBJECT_LEVEL_STAGE,
            error=(
                f"{len(batch_result['failed'])} subject exception(s) and "
                f"{len(batch_result['incomplete'])} subject(s) with incomplete subject metrics/QC. "
                "No complete subject outputs were available for downstream stages."
            ),
            **status_details,
        )
    return stage_ok(
        SUBJECT_LEVEL_STAGE,
        processed_subjects=batch_result["processed"],
        skipped_subjects=batch_result["skipped"],
        failed_subjects=batch_result["failed"],
        incomplete_subjects=batch_result["incomplete"],
    )


def run_population_within_run_stage(cfg: PipelineConfig) -> dict:
    if not cfg.population.enabled:
        return stage_skipped(
            POPULATION_WITHIN_RUN_STAGE,
            reason="Population (within run)-level aggregation was disabled in the pipeline config.",
        )

    print(f"[INFO] Stage: {PIPELINE_STAGE_LABELS[POPULATION_WITHIN_RUN_STAGE]}")
    from post.post_population import run_population

    root = Path(cfg.post.root).expanduser().resolve()
    output_path = run_population(
        root=root,
        subjects=cfg.post.subjects,
        out_dir=_resolve_population_output_dir(cfg.population),
        region_filename=cfg.population.region_filename,
        metrics_filename=cfg.population.metrics_filename,
        peak_threshold=cfg.population.peak_threshold,
        target_roi=cfg.population.target_roi or (cfg.post.plot_roi or "Hippocampus"),
        template_region_csv=Path(cfg.population.template_region_csv).expanduser()
        if cfg.population.template_region_csv
        else None,
    )
    return stage_ok(
        POPULATION_WITHIN_RUN_STAGE,
        output_dir=str(output_path),
        region_filename=cfg.population.region_filename,
        metrics_filename=cfg.population.metrics_filename,
        target_roi=cfg.population.target_roi or (cfg.post.plot_roi or "Hippocampus"),
    )


def run_pipeline(cfg: PipelineConfig, *, raise_on_error: bool = True) -> dict:
    _resolve_pipeline_rois(cfg)
    summary = {
        "resolved_plot_roi": cfg.post.plot_roi,
        "resolved_target_roi": cfg.population.target_roi,
        "stages": {},
    }

    subject_stage = run_subject_level_stage(cfg.post)
    summary["stages"][SUBJECT_LEVEL_STAGE] = subject_stage

    if subject_stage["status"] == "failed":
        population_stage = stage_skipped(
            POPULATION_WITHIN_RUN_STAGE,
            reason="Population (within run)-level metrics were skipped because subject-level processing produced no complete subjects.",
        )
    else:
        try:
            population_stage = run_population_within_run_stage(cfg)
        except Exception as exc:
            population_stage = stage_failed(
                POPULATION_WITHIN_RUN_STAGE,
                error=f"{type(exc).__name__}: {exc}",
            )
    summary["stages"][POPULATION_WITHIN_RUN_STAGE] = population_stage

    if raise_on_error:
        if subject_stage["status"] == "failed":
            raise SystemExit(subject_stage["error"])
        if subject_stage["status"] == "partial":
            raise SystemExit(subject_stage["warning"])
        if population_stage["status"] == "failed":
            raise SystemExit(population_stage["error"])

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.strip())
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


def make_default_config() -> PipelineConfig:
    return PipelineConfig(
        post=PostBatchConfig(
            root="/home/boyan/sandbox/Jake_Data/Left_Thalamus/",
            t1_path=None,
            subjects=None,  # list like ["sub-CC110056", "sub-CC110087"] or None for all
            max_workers=None,  # None -> use all detected CPU cores; lower this if memory is tight
            atlas_mode="fastsurfer",
            fastsurfer_root="/mnt/parscratch/users/cop23bi/ZIPs/atlases",
            fs_mri_path=None,
            fastsurfer_atlas_filename=None,  # e.g. "mri/aparc.DKTatlas+aseg.deep.nii.gz"
            plot_roi=None,  # None -> infer from root dir, e.g. Left_Hippocampus_Data_test
            percentile=95.0,
            hard_threshold=0.2,
            write_region_table=True,
            region_percentile=95.0,
            offtarget_threshold=0.2,
            mni_baseline_root=None,
            mni_fixed_atlas_path=None,
            neighbor_dilation_iter=1,
            csf_labels=[24],
            skull_labels=None,
            electrode_csv=None,
            electrode_names=None,
            eeg_positions_path_template=None,
            write_neighbor_table=True,
            write_neighbor_visualization=True,
            write_electrode_table=True,
            force=False,
            verbose=True,
            overlay_z_offset_mm=0,
            overlay_full_field=True,
        ),
        population=PopulationConfig(
            enabled=False,
            out_dir=None,
            region_filename="region_stats_fastsurfer.csv",
            metrics_filename="subject_metrics.json",
            peak_threshold=0.2,
            target_roi=None,  # None -> reuse resolved post ROI
            template_region_csv=None,
        ),
    )


def apply_cli_overrides(cfg: PipelineConfig, args: argparse.Namespace) -> None:
    if args.fastsurfer_atlas_filename is not None:
        cfg.post.fastsurfer_atlas_filename = args.fastsurfer_atlas_filename


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = make_default_config()
    args = build_arg_parser().parse_args(argv)
    apply_cli_overrides(cfg, args)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
