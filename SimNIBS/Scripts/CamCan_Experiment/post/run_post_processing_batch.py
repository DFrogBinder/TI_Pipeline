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
    make_default_config,
    run_pipeline,
)

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

    for index, dataset in enumerate(datasets, start=1):
        print(f"[INFO] Dataset {index}/{len(datasets)}: {dataset.name}")
        dataset_cfg = build_dataset_pipeline_config(dataset.root, pipeline_template)

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
        "results": results,
    }

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


def main(argv: Optional[Sequence[str]] = None) -> None:
    batch_cfg = make_default_batch_config()
    pipeline_template = make_default_pipeline_template()

    args = build_arg_parser().parse_args(argv)
    apply_cli_overrides(pipeline_template, args)
    apply_batch_cli_overrides(batch_cfg, args)

    run_repeat_batch(batch_cfg, pipeline_template)


if __name__ == "__main__":
    main()
