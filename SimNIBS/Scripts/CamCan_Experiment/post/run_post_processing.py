"""
Batch runner for subject-level post processing and optional population aggregation.

Edit the config at the bottom to control the full pipeline from a single entrypoint.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post.post_process import PostProcessConfig, run_post_process
from post.post_population import run_population


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
    atlas_mode: str = "auto"  # "auto" | "mni" | "fastsurfer"
    fastsurfer_root: Optional[str] = None
    fs_mri_path: Optional[str] = None
    plot_roi: str = "Hippocampus"
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
    target_roi: str = "Hippocampus"
    template_region_csv: Optional[str] = None


@dataclass
class PipelineConfig:
    post: PostBatchConfig
    population: PopulationConfig


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

    for subj in subjects:
        out_dir = root / subj / "anat" / "post"
        if should_skip_subject(out_dir, cfg.force):
            skipped.append(subj)
            continue

        pp_cfg = PostProcessConfig(
            root_dir=str(root),
            subject=subj,
            atlas_mode=cfg.atlas_mode,
            fastsurfer_root=cfg.fastsurfer_root,
            fs_mri_path=cfg.fs_mri_path,
            plot_roi=cfg.plot_roi,
            percentile=cfg.percentile,
            hard_threshold=cfg.hard_threshold,
            overlay_z_offset_mm=cfg.overlay_z_offset_mm,
            overlay_full_field=cfg.overlay_full_field,
            write_region_table=cfg.write_region_table,
            region_percentile=cfg.region_percentile,
            offtarget_threshold=cfg.offtarget_threshold,
            verbose=cfg.verbose,
        )
        try:
            run_post_process(pp_cfg)
            processed.append(subj)
        except Exception as exc:
            failed.append((subj, str(exc)))

    print(f"[INFO] Processed {len(processed)} subject(s).")
    if skipped:
        print(f"[INFO] Skipped {len(skipped)} subject(s) (existing outputs).")
    if failed:
        print(f"[WARN] Failed {len(failed)} subject(s).")
        for subj, err in failed:
            print(f"  - {subj}: {err}")

    return {"processed": processed, "skipped": skipped, "failed": failed}


def run_pipeline(cfg: PipelineConfig) -> None:
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
            target_roi=cfg.population.target_roi,
            template_region_csv=Path(cfg.population.template_region_csv).expanduser()
            if cfg.population.template_region_csv
            else None,
        )
    if batch_result["failed"]:
        raise SystemExit("Some subjects failed during post-processing.")


if __name__ == "__main__":
    cfg = PipelineConfig(
        post=PostBatchConfig(
            root="/media/boyan/main/PhD/Hippocampus/Hippocampus-data",
            subjects=None,  # list like ["sub-CC110056", "sub-CC110087"] or None for all
            atlas_mode="fastsurfer",
            fastsurfer_root='/home/boyan/sandbox/Jake_Data/atlases',
            fs_mri_path=None,
            plot_roi="Hippocampus",  # M1 (ctx-lh-precentral) / Hippocampus
            percentile=95.0,
            hard_threshold=200.0,
            write_region_table=True,
            region_percentile=95.0,
            offtarget_threshold=0.2,
            force=False,
            verbose=True,
            overlay_z_offset_mm=10.0,
            overlay_full_field=True,
        ),
        population=PopulationConfig(
            enabled=True,
            out_dir=None,
            region_filename="region_stats_fastsurfer.csv",
            metrics_filename="subject_metrics.json",
            peak_threshold=0.2,
            target_roi="ctx-lh-precentral",
            template_region_csv=None,
        ),
    )
    run_pipeline(cfg)
