"""
Single-command CLI entrypoint for the TI post-processing pipeline.

This wrapper hides the current split between the three explicit post-processing
layers:

1. Subject-level metrics
2. Population (within run)-level metrics
3. Across-repeats-level metrics

It chooses the correct orchestration path from one command.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post.run_post_processing import PipelineConfig, make_default_config, run_pipeline
from post.run_post_processing_batch import RepeatBatchConfig, discover_repeat_datasets, run_repeat_batch


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full TI post-processing pipeline from one command. "
            "The wrapper auto-detects whether --root points to a single dataset "
            "or a repeat batch root."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help=(
            "Either a single dataset root such as Left_Hippocampus_Data_01, "
            "or a batch root containing repeated datasets."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "single", "batch"),
        default="auto",
        help="Execution mode. Default: auto-detect from the directory structure.",
    )
    parser.add_argument(
        "--dataset-glob",
        default="*_Data_*",
        help="Batch-mode dataset glob. Default: *_Data_*",
    )
    parser.add_argument(
        "--repeats",
        nargs="*",
        default=None,
        help="Optional repeat identifiers to include in batch mode, for example: 01 02 10",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject IDs to process.",
    )
    parser.add_argument(
        "--roi",
        default=None,
        help="ROI alias or canonical name. If omitted, the pipeline will infer it from the dataset name when possible.",
    )
    parser.add_argument(
        "--fastsurfer-root",
        default=None,
        help="Root directory containing subject-specific FastSurfer outputs or atlas files.",
    )
    parser.add_argument(
        "--atlas-filename",
        "--fastsurfer-atlas-filename",
        dest="fastsurfer_atlas_filename",
        default=None,
        help=(
            "Atlas override for all subjects. Use either an absolute atlas path shared by every subject, "
            "or a relative path under each subject directory inside --fastsurfer-root."
        ),
    )
    parser.add_argument(
        "--atlas-mode",
        choices=("auto", "mni", "fastsurfer"),
        default="auto",
        help="Atlas lookup mode. Default: auto",
    )
    parser.add_argument(
        "--t1-path",
        default=None,
        help="Optional explicit T1 image path. Most subject runs should leave this unset.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum parallel subject workers per dataset.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Percentile used to define the high-field mask. Default: 95",
    )
    parser.add_argument(
        "--hard-threshold",
        type=float,
        default=0.2,
        help="Hard field threshold used in overlay and thresholding logic. Default: 0.2",
    )
    parser.add_argument(
        "--offtarget-threshold",
        type=float,
        default=0.2,
        help="Field threshold used for focality metrics. Default: 0.2",
    )
    parser.add_argument(
        "--mni-baseline-root",
        default=None,
        help=(
            "Root directory of the MNI baseline simulation for this ROI. "
            "The pipeline will extract baseline metrics from its TI volume using the same ROI code path as subjects."
        ),
    )
    parser.add_argument(
        "--mni-fixed-atlas-path",
        default=None,
        help="Path to the fixed MNI atlas used to define ROI neighbor templates.",
    )
    parser.add_argument(
        "--neighbor-dilation-iter",
        type=int,
        default=1,
        help="Neighbor-template dilation iterations on the fixed MNI atlas. Default: 1",
    )
    parser.add_argument(
        "--no-neighbor-visualization",
        action="store_true",
        help=(
            "Disable subject-level fixed-neighbor mask and overlay exports. "
            "Neighbor scalar metrics are still computed when mni_fixed_atlas_path is configured."
        ),
    )
    parser.add_argument(
        "--csf-labels",
        nargs="*",
        type=int,
        default=None,
        help="Optional atlas label ids to treat as CSF.",
    )
    parser.add_argument(
        "--skull-labels",
        nargs="*",
        type=int,
        default=None,
        help="Optional atlas label ids to treat as skull.",
    )
    parser.add_argument(
        "--electrode-csv",
        default=None,
        help="Optional CSV with columns subject,electrode,x,y,z.",
    )
    parser.add_argument(
        "--electrode-dataset-dir",
        default=None,
        help=(
            "Optional directory with per-ROI/per-subject electrode CSVs, "
            "for example <dir>/<roi>/<subject>/electrodes.csv."
        ),
    )
    parser.add_argument(
        "--electrode-names",
        nargs="*",
        default=None,
        help="Optional electrode names to read from subject EEG position files.",
    )
    parser.add_argument(
        "--eeg-positions-path-template",
        default=None,
        help="Optional template path for subject EEG position files. Use {root} and {subject} placeholders.",
    )
    parser.add_argument(
        "--no-population",
        action="store_true",
        help="Disable population (within run)-level aggregation.",
    )
    parser.add_argument(
        "--population-output-dir",
        default=None,
        help="Optional population-analysis output directory for single-dataset mode.",
    )
    parser.add_argument(
        "--template-region-csv",
        default=None,
        help="Optional MNI region-summary CSV used by the within-run population analysis.",
    )
    parser.add_argument(
        "--no-repeatability",
        action="store_true",
        help="Disable the across-repeats-level metrics stage in batch mode.",
    )
    parser.add_argument(
        "--repeatability-output-dir",
        default=None,
        help=(
            "Optional batch-mode repeatability output directory root. "
            "By default, single-ROI repeat batches write directly to <batch_root>/subject_metrics_analysis. "
            "If multiple ROIs are discovered under one batch root, the pipeline falls back to "
            "<batch_root>/repeatability_analysis/<roi> unless you override this path."
        ),
    )
    parser.add_argument(
        "--repeatability-logs-root",
        default=None,
        help="Optional logs directory for the repeatability analysis stage.",
    )
    parser.add_argument(
        "--allow-incomplete-repeat-subjects",
        action="store_true",
        help=(
            "Batch mode only: allow within-run population analysis to use subjects that are not present "
            "in every selected repeat. By default, only the complete-case cohort across repeats is used."
        ),
    )
    parser.add_argument(
        "--no-figure-generation",
        action="store_true",
        help="Disable static PNG/CSV figure generation in batch mode.",
    )
    parser.add_argument(
        "--figure-output-dir",
        default=None,
        help="Optional static figure output directory for batch mode. Default: <root>/post_processing_figures.",
    )
    parser.add_argument(
        "--summary-filename",
        default="post_processing_batch_summary.json",
        help="Batch-mode summary filename. Use an empty string to disable summary writing.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first dataset failure in batch mode.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess subjects even if subject_metrics.json already exists with extended metrics.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging verbosity.",
    )
    return parser.parse_args(argv)


def infer_mode(root: Path, requested_mode: str, dataset_glob: str) -> str:
    if requested_mode != "auto":
        return requested_mode

    if discover_repeat_datasets(root, dataset_glob=dataset_glob):
        return "batch"
    return "single"


def build_pipeline_config(args: argparse.Namespace, root: Path) -> PipelineConfig:
    cfg = make_default_config()
    cfg.post.root = str(root)
    cfg.post.subjects = args.subjects
    cfg.post.max_workers = args.max_workers
    cfg.post.atlas_mode = args.atlas_mode
    cfg.post.fastsurfer_root = args.fastsurfer_root
    cfg.post.fastsurfer_atlas_filename = args.fastsurfer_atlas_filename
    cfg.post.t1_path = args.t1_path
    cfg.post.plot_roi = args.roi
    cfg.post.percentile = args.percentile
    cfg.post.hard_threshold = args.hard_threshold
    cfg.post.offtarget_threshold = args.offtarget_threshold
    cfg.post.mni_baseline_root = args.mni_baseline_root
    cfg.post.mni_fixed_atlas_path = args.mni_fixed_atlas_path
    cfg.post.neighbor_dilation_iter = args.neighbor_dilation_iter
    cfg.post.write_neighbor_visualization = not args.no_neighbor_visualization
    cfg.post.csf_labels = args.csf_labels if args.csf_labels else [24]
    cfg.post.skull_labels = args.skull_labels
    cfg.post.electrode_csv = args.electrode_csv
    cfg.post.electrode_dataset_dir = args.electrode_dataset_dir
    cfg.post.electrode_names = args.electrode_names
    cfg.post.eeg_positions_path_template = args.eeg_positions_path_template
    cfg.post.force = args.force
    cfg.post.verbose = not args.quiet

    cfg.population.enabled = not args.no_population
    cfg.population.out_dir = args.population_output_dir
    cfg.population.target_roi = args.roi
    cfg.population.template_region_csv = args.template_region_csv
    return cfg


def build_batch_config(args: argparse.Namespace, root: Path) -> RepeatBatchConfig:
    summary_filename = args.summary_filename if args.summary_filename else None
    repeatability_output_dir = args.repeatability_output_dir if args.repeatability_output_dir else None
    return RepeatBatchConfig(
        batch_root=str(root),
        dataset_glob=args.dataset_glob,
        repeats=args.repeats,
        continue_on_error=not args.stop_on_error,
        summary_filename=summary_filename,
        run_repeatability=not args.no_repeatability,
        repeatability_output_dir=repeatability_output_dir,
        repeatability_logs_root=args.repeatability_logs_root,
        complete_repeat_subjects_only=not args.allow_incomplete_repeat_subjects,
        run_figure_generation=not args.no_figure_generation,
        figure_output_dir=args.figure_output_dir if args.figure_output_dir else None,
    )


def run_single_mode(args: argparse.Namespace, root: Path) -> None:
    cfg = build_pipeline_config(args, root)
    print(f"[INFO] Running single-dataset pipeline on: {root}")
    print("[INFO] Layers: subject_level -> population_within_run")
    run_pipeline(cfg)
    if not args.no_repeatability:
        print("[INFO] Across-repeats-level metrics were not run because single mode processes only one dataset root.")


def run_batch_mode(args: argparse.Namespace, root: Path) -> None:
    pipeline_cfg = build_pipeline_config(args, root)
    batch_cfg = build_batch_config(args, root)
    print(f"[INFO] Running repeat-batch pipeline on: {root}")
    print("[INFO] Layers: subject_level -> population_within_run -> across_repeats -> figure_generation")
    run_repeat_batch(batch_cfg, pipeline_cfg)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Root directory not found: {root}")

    mode = infer_mode(root, args.mode, args.dataset_glob)
    print(f"[INFO] Resolved pipeline mode: {mode}")

    if mode == "single":
        run_single_mode(args, root)
        return
    if mode == "batch":
        run_batch_mode(args, root)
        return
    raise SystemExit(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    main()
