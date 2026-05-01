"""
Environment-driven entrypoint for batch post-processing.

This exists so Slurm wrappers can configure the batch via environment
variables without launching the Python main module from stdin, which breaks
multiprocessing with the "spawn" start method on HPC.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post.run_post_processing import make_default_config
from post.run_post_processing_batch import RepeatBatchConfig, run_repeat_batch


def read_text(name: str) -> str:
    return os.environ.get(name, "")


def read_optional_text(name: str) -> str | None:
    value = read_text(name).strip()
    return value or None


def read_bool(name: str, *, default: bool) -> bool:
    raw = read_text(name).strip()
    if not raw:
        return default

    normalized = raw.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"Invalid boolean value for {name}: {raw!r}")


def read_optional_int(name: str) -> int | None:
    raw = read_optional_text(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid integer value for {name}: {raw!r}") from exc


def read_float(name: str, *, default: float) -> float:
    raw = read_text(name).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid float value for {name}: {raw!r}") from exc


def read_optional_list(name: str) -> list[str] | None:
    raw = read_optional_text(name)
    if raw is None:
        return None
    return shlex.split(raw)


def build_configs():
    batch_root = read_optional_text("BATCH_ROOT")
    if batch_root is None or batch_root == "/path/to/rootDIR":
        raise SystemExit(
            "Set BATCH_ROOT near the top of HPC_scripts/run_post_processing_batch.slurm "
            "to the parent directory containing repeat datasets."
        )

    pipeline_cfg = make_default_config()
    pipeline_cfg.post.root = batch_root
    pipeline_cfg.post.subjects = read_optional_list("PIPELINE_SUBJECTS")
    pipeline_cfg.post.max_workers = read_optional_int("PIPELINE_MAX_WORKERS")
    pipeline_cfg.post.atlas_mode = read_text("PIPELINE_ATLAS_MODE").strip() or pipeline_cfg.post.atlas_mode
    pipeline_cfg.post.fastsurfer_root = read_optional_text("PIPELINE_FASTSURFER_ROOT")
    pipeline_cfg.post.fs_mri_path = read_optional_text("PIPELINE_FS_MRI_PATH")
    pipeline_cfg.post.fastsurfer_atlas_filename = read_optional_text("PIPELINE_FASTSURFER_ATLAS_FILENAME")
    pipeline_cfg.post.t1_path = read_optional_text("PIPELINE_T1_PATH")
    pipeline_cfg.post.plot_roi = read_optional_text("PIPELINE_PLOT_ROI")
    pipeline_cfg.post.percentile = read_float("PIPELINE_PERCENTILE", default=pipeline_cfg.post.percentile)
    pipeline_cfg.post.hard_threshold = read_float("PIPELINE_HARD_THRESHOLD", default=pipeline_cfg.post.hard_threshold)
    pipeline_cfg.post.overlay_z_offset_mm = read_float(
        "PIPELINE_OVERLAY_Z_OFFSET_MM",
        default=pipeline_cfg.post.overlay_z_offset_mm,
    )
    pipeline_cfg.post.overlay_full_field = read_bool(
        "PIPELINE_OVERLAY_FULL_FIELD",
        default=pipeline_cfg.post.overlay_full_field,
    )
    pipeline_cfg.post.write_region_table = read_bool(
        "PIPELINE_WRITE_REGION_TABLE",
        default=pipeline_cfg.post.write_region_table,
    )
    pipeline_cfg.post.region_percentile = read_float(
        "PIPELINE_REGION_PERCENTILE",
        default=pipeline_cfg.post.region_percentile,
    )
    pipeline_cfg.post.offtarget_threshold = read_float(
        "PIPELINE_OFFTARGET_THRESHOLD",
        default=pipeline_cfg.post.offtarget_threshold,
    )
    pipeline_cfg.post.mni_baseline_root = read_optional_text("PIPELINE_MNI_BASELINE_ROOT")
    pipeline_cfg.post.mni_fixed_atlas_path = read_optional_text("PIPELINE_MNI_FIXED_ATLAS_PATH")
    pipeline_cfg.post.neighbor_dilation_iter = read_optional_int("PIPELINE_NEIGHBOR_DILATION_ITER") or pipeline_cfg.post.neighbor_dilation_iter
    pipeline_cfg.post.csf_labels = [
        int(value) for value in read_optional_list("PIPELINE_CSF_LABELS") or pipeline_cfg.post.csf_labels or []
    ] or None
    pipeline_cfg.post.skull_labels = [
        int(value) for value in read_optional_list("PIPELINE_SKULL_LABELS") or []
    ] or None
    pipeline_cfg.post.electrode_csv = read_optional_text("PIPELINE_ELECTRODE_CSV")
    pipeline_cfg.post.electrode_names = read_optional_list("PIPELINE_ELECTRODE_NAMES")
    pipeline_cfg.post.eeg_positions_path_template = read_optional_text(
        "PIPELINE_EEG_POSITIONS_PATH_TEMPLATE"
    )
    pipeline_cfg.post.write_neighbor_table = read_bool(
        "PIPELINE_WRITE_NEIGHBOR_TABLE",
        default=pipeline_cfg.post.write_neighbor_table,
    )
    pipeline_cfg.post.write_electrode_table = read_bool(
        "PIPELINE_WRITE_ELECTRODE_TABLE",
        default=pipeline_cfg.post.write_electrode_table,
    )
    pipeline_cfg.post.force = read_bool("PIPELINE_FORCE", default=pipeline_cfg.post.force)
    pipeline_cfg.post.verbose = read_bool("PIPELINE_VERBOSE", default=pipeline_cfg.post.verbose)

    pipeline_cfg.population.enabled = read_bool(
        "PIPELINE_POPULATION_ENABLED",
        default=pipeline_cfg.population.enabled,
    )
    pipeline_cfg.population.out_dir = read_optional_text("PIPELINE_POPULATION_OUT_DIR")
    pipeline_cfg.population.region_filename = (
        read_text("PIPELINE_POPULATION_REGION_FILENAME").strip()
        or pipeline_cfg.population.region_filename
    )
    pipeline_cfg.population.metrics_filename = (
        read_text("PIPELINE_POPULATION_METRICS_FILENAME").strip()
        or pipeline_cfg.population.metrics_filename
    )
    pipeline_cfg.population.peak_threshold = read_float(
        "PIPELINE_POPULATION_PEAK_THRESHOLD",
        default=pipeline_cfg.population.peak_threshold,
    )
    pipeline_cfg.population.target_roi = read_optional_text("PIPELINE_POPULATION_TARGET_ROI")
    pipeline_cfg.population.template_region_csv = read_optional_text(
        "PIPELINE_POPULATION_TEMPLATE_REGION_CSV"
    )

    batch_cfg = RepeatBatchConfig(
        batch_root=batch_root,
        dataset_glob=read_text("BATCH_DATASET_GLOB").strip() or "*_Data_*",
        repeats=read_optional_list("BATCH_REPEATS"),
        continue_on_error=not read_bool("BATCH_STOP_ON_ERROR", default=False),
        summary_filename=read_optional_text("BATCH_SUMMARY_FILENAME"),
        run_repeatability=read_bool("PIPELINE_REPEATABILITY_ENABLED", default=True),
        repeatability_output_dir=read_optional_text("PIPELINE_REPEATABILITY_OUTPUT_DIR"),
        repeatability_logs_root=read_optional_text("PIPELINE_REPEATABILITY_LOGS_ROOT"),
        complete_repeat_subjects_only=read_bool(
            "PIPELINE_COMPLETE_REPEAT_SUBJECTS_ONLY",
            default=True,
        ),
    )
    return batch_cfg, pipeline_cfg


def main() -> None:
    repo_dir = Path(os.environ["REPO_DIR"]).expanduser().resolve()
    os.chdir(repo_dir)

    batch_cfg, pipeline_cfg = build_configs()
    run_repeat_batch(batch_cfg, pipeline_cfg)


if __name__ == "__main__":
    main()
