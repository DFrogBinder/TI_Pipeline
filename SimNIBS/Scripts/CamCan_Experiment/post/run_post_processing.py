"""
Batch runner for subject-level post processing and optional population aggregation.

Edit the config at the bottom to control the full pipeline from a single entrypoint.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post.post_process import PostProcessConfig, run_post_process
from post.post_population import run_population
from utils.roi_registry import match_fastsurfer_roi_from_directory, resolve_fastsurfer_roi_name


def discover_subjects(root: Path, subjects: Optional[Iterable[str]]) -> List[str]:
    if subjects:
        return list(subjects)
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def should_skip_subject(out_dir: Path, force: bool) -> bool:
    if force:
        return False
    metrics_path = out_dir / "subject_metrics.json"
    return metrics_path.is_file()

@dataclass
class PostBatchConfig:
    root: str
    subjects: Optional[List[str]] = None
    max_workers: Optional[int] = 8
    atlas_mode: str = "auto"  # "auto" | "mni" | "fastsurfer"
    fastsurfer_root: Optional[str] = None
    fs_mri_path: Optional[str] = None
    t1_path: Optional[str] = None
    plot_roi: Optional[str] = None
    percentile: float = 95.0
    hard_threshold: float = 200.0
    overlay_z_offset_mm: float = 0.0
    overlay_full_field: bool = False
    write_region_table: bool = True
    region_percentile: float = 95.0
    offtarget_threshold: float = 0.2
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


def build_post_process_config(root: Path, subject: str, cfg: PostBatchConfig) -> PostProcessConfig:
    return PostProcessConfig(
        root_dir=str(root),
        subject=subject,
        atlas_mode=cfg.atlas_mode,
        fastsurfer_root=cfg.fastsurfer_root,
        fs_mri_path=cfg.fs_mri_path,
        t1_path=cfg.t1_path,
        plot_roi=cfg.plot_roi or "ctx-lh-precentral",
        percentile=cfg.percentile,
        hard_threshold=cfg.hard_threshold,
        overlay_z_offset_mm=cfg.overlay_z_offset_mm,
        overlay_full_field=cfg.overlay_full_field,
        write_region_table=cfg.write_region_table,
        region_percentile=cfg.region_percentile,
        offtarget_threshold=cfg.offtarget_threshold,
        verbose=cfg.verbose,
    )


def resolve_max_workers(cfg: PostBatchConfig, task_count: int) -> int:
    if task_count <= 0:
        return 0
    if cfg.max_workers is None:
        requested = os.cpu_count() or 1
    else:
        requested = cfg.max_workers
    if requested < 1:
        raise ValueError("max_workers must be at least 1.")
    return min(task_count, requested)


def process_subject(pp_cfg: PostProcessConfig) -> str:
    run_post_process(pp_cfg)
    return pp_cfg.subject


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
                    f"'{plot_match.canonical_name}'."
                )
            else:
                plot_match = match_fastsurfer_roi_from_directory(cfg.post.root)
                print(
                    f"[INFO] Inferred ROI from directory '{Path(cfg.post.root).name}' via alias "
                    f"'{plot_match.matched_alias}' -> '{plot_match.canonical_name}'."
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
                f"'{target_match.canonical_name}'."
            )
            cfg.population.target_roi = target_match.canonical_name
        else:
            cfg.population.target_roi = cfg.post.plot_roi
    else:
        if cfg.population.target_roi is None:
            cfg.population.target_roi = cfg.post.plot_roi or "Hippocampus"


def run_batch(cfg: PostBatchConfig) -> dict:
    root = Path(cfg.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Root directory not found: {root}")

    subjects = discover_subjects(root, cfg.subjects)
    if not subjects:
        raise SystemExit("No subjects found.")

    processed = []
    skipped = []
    failed = []
    pending = []

    for subj in subjects:
        out_dir = root / subj / "anat" / "post"
        if should_skip_subject(out_dir, cfg.force):
            skipped.append(subj)
            continue
        pending.append(build_post_process_config(root, subj, cfg))

    max_workers = resolve_max_workers(cfg, len(pending))-5
    if pending:
        print(
            f"[INFO] Running post-processing for {len(pending)} subject(s) "
            f"with up to {max_workers} worker(s)."
        )

    if max_workers <= 1:
        for pp_cfg in pending:
            try:
                process_subject(pp_cfg)
                processed.append(pp_cfg.subject)
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
                    future.result()
                    processed.append(subj)
                except Exception as exc:
                    failed.append((subj, f"{type(exc).__name__}: {exc}"))

    processed.sort()
    skipped.sort()
    failed.sort(key=lambda item: item[0])

    print(f"[INFO] Processed {len(processed)} subject(s).")
    if skipped:
        print(f"[INFO] Skipped {len(skipped)} subject(s) (existing outputs).")
    if failed:
        print(f"[WARN] Failed {len(failed)} subject(s).")
        for subj, err in failed:
            print(f"  - {subj}: {err}")

    return {"processed": processed, "skipped": skipped, "failed": failed}


def run_pipeline(cfg: PipelineConfig) -> None:
    _resolve_pipeline_rois(cfg)
    batch_result = run_batch(cfg.post)

    if cfg.population.enabled:
        root = Path(cfg.post.root).expanduser().resolve()
        run_population(
            root=root,
            subjects=cfg.post.subjects,
            out_dir=Path(cfg.population.out_dir).expanduser()
            if cfg.population.out_dir
            else None,
            region_filename=cfg.population.region_filename,
            metrics_filename=cfg.population.metrics_filename,
            peak_threshold=cfg.population.peak_threshold,
            target_roi=cfg.population.target_roi or (cfg.post.plot_roi or "Hippocampus"),
            template_region_csv=Path(cfg.population.template_region_csv).expanduser()
            if cfg.population.template_region_csv
            else None,
        )
    if batch_result["failed"]:
        raise SystemExit("Some subjects failed during post-processing.")


if __name__ == "__main__":
    cfg = PipelineConfig(
        post=PostBatchConfig(
            root="/media/boyan/main/PhD/Rigth_Hippocampus_data",
            t1_path=None,
            subjects=None,  # list like ["sub-CC110056", "sub-CC110087"] or None for all
            max_workers=None,  # None -> use all detected CPU cores; lower this if memory is tight
            atlas_mode="fastsurfer",
            fastsurfer_root='/home/boyan/sandbox/Jake_Data/atlases',
            fs_mri_path=None,
            plot_roi=None,  # None -> infer from root dir, e.g. Left_Hippocampus_Data_test
            percentile=95.0,
            hard_threshold=0.2,
            write_region_table=True,
            region_percentile=95.0,
            offtarget_threshold=0.2,
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
    run_pipeline(cfg)
