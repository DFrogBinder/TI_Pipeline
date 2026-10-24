"""Thin delegation layer for the existing corrected SimNIBS 4 workflow."""

from __future__ import annotations

from pathlib import Path

from ti_current_repair.pipeline import staged_median_fixed_experiment as v4_pipeline

from .replacement import require_left_complete, require_v3_promotion


TARGETS = ("left-hippocampus", "right-m1")


def _bindings(manifest: dict[str, object], target: str) -> dict[str, str]:
    if target not in TARGETS:
        raise ValueError(f"Unsupported replacement target {target!r}; expected {TARGETS}")
    paths = manifest.get("paths")
    inputs = manifest.get("inputs")
    if not isinstance(paths, dict) or not isinstance(inputs, dict):
        raise RuntimeError("Prepared replacement manifest lacks path/input bindings")
    atlas = inputs.get("atlas")
    targets_csv = inputs.get("targets_csv")
    if not isinstance(atlas, dict) or not isinstance(targets_csv, dict):
        raise RuntimeError("Prepared replacement manifest lacks atlas/target records")
    root_key = "v4_left_root" if target == "left-hippocampus" else "v4_right_root"
    return {
        "subject": str(manifest.get("subject", "")),
        "source_root": str(paths.get("v4_corrected_root", "")),
        "experiment_root": str(paths.get(root_key, "")),
        "atlas_dir": str(Path(str(atlas.get("path", ""))).parent),
        "targets_csv": str(targets_csv.get("path", "")),
    }


def initialize_v4_target(manifest: dict[str, object], target: str) -> int:
    """Initialize one target using the existing v4 median/fixed pipeline."""
    values = _bindings(manifest, target)
    require_v3_promotion(manifest)
    if target == "right-m1":
        require_left_complete(manifest, "v4")
    return v4_pipeline.main(
        [
            "init",
            "--source-root",
            values["source_root"],
            "--experiment-root",
            values["experiment_root"],
            "--subjects",
            values["subject"],
            "--repeat-count",
            "40",
            "--roi-preset",
            target,
            "--montage-preset",
            target,
            "--targets-csv",
            values["targets_csv"],
            "--atlas-dir",
            values["atlas_dir"],
            "--compare-metric",
            "median_roi",
        ]
    )


def run_v4_target(
    manifest: dict[str, object], target: str, dry_run: bool = False
) -> int:
    """Preflight or submit the complete new-mesh-to-fixed v4 target workflow."""
    values = _bindings(manifest, target)
    require_v3_promotion(manifest)
    if target == "right-m1":
        require_left_complete(manifest, "v4")
    argv = [
        "submit-all",
        "--experiment-root",
        values["experiment_root"],
        "--max-concurrent",
        "50",
        "--analysis-max-concurrent",
        "1",
    ]
    if dry_run:
        argv.append("--dry-run")
    return v4_pipeline.main(argv)


def replacement_status(manifest: dict[str, object]) -> dict[str, object]:
    """Return receipt-oriented status without equating submission with completion."""
    values: dict[str, object] = {
        "subject": manifest.get("subject"),
        "v3_promotion": False,
        "targets": {},
    }
    try:
        require_v3_promotion(manifest)
    except RuntimeError:
        pass
    else:
        values["v3_promotion"] = True
    paths = manifest.get("paths", {})
    if not isinstance(paths, dict):
        return values
    target_status: dict[str, object] = {}
    for version in ("v3", "v4"):
        for target, suffix in (
            ("left-hippocampus", "left_root"),
            ("right-m1", "right_root"),
        ):
            root = Path(str(paths.get(f"{version}_{suffix}", "")))
            completion = root / "_pipeline" / "workflow" / "complete.json"
            target_status[f"{version}:{target}"] = {
                "initialized": (root / "_pipeline").is_dir(),
                "terminal_receipt": str(completion),
                "terminal_receipt_present": completion.is_file(),
            }
    values["targets"] = target_status
    return values
