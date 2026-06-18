#!/usr/bin/env python3
"""Run one subject's repeatability report from a Slurm array index."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PIPELINE_DIR = HERE.parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from experiment_config import load_experiment_config  # noqa: E402


def _add_optional(cmd: list[str], flag: str, value: str | None) -> None:
    if value:
        cmd.extend([flag, value])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--subject-index", type=int, required=True)
    parser.add_argument("--conditions", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--roi-preset", default=None)
    parser.add_argument("--roi-name", default=None)
    parser.add_argument("--roi-labels", default=None)
    parser.add_argument("--atlas-dir", default=None)
    parser.add_argument("--reference-repeat", default=None)
    parser.add_argument("--spatial-percentile", default=None)
    parser.add_argument("--compare-metric", default=None)
    parser.add_argument("--compare-cohort-root", default=None)
    parser.add_argument("--cohort-region-name", default=None)
    parser.add_argument("--cohort-region-label", default=None)
    parser.add_argument("--cohort-metric", default=None)
    parser.add_argument("--skip-cohort", action="store_true")
    parser.add_argument("--log-file", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_experiment_config(args.config, validate_paths=False)
    if args.subject_index < 0 or args.subject_index >= len(config.subjects):
        raise SystemExit(
            f"subject index {args.subject_index} out of range 0..{len(config.subjects) - 1}"
        )
    subject = config.subjects[args.subject_index]
    script = PIPELINE_DIR / "post" / "repeatability_experiment_report.py"
    cmd = [
        sys.executable,
        "-u",
        str(script),
        "--config",
        str(Path(args.config).expanduser().resolve()),
        "--subject",
        subject,
        "--skip-batch-summary",
    ]
    _add_optional(cmd, "--conditions", args.conditions)
    _add_optional(cmd, "--output-dir", args.output_dir)
    _add_optional(cmd, "--roi-preset", args.roi_preset)
    _add_optional(cmd, "--roi-name", args.roi_name)
    _add_optional(cmd, "--roi-labels", args.roi_labels)
    _add_optional(cmd, "--atlas-dir", args.atlas_dir)
    _add_optional(cmd, "--reference-repeat", args.reference_repeat)
    _add_optional(cmd, "--spatial-percentile", args.spatial_percentile)
    _add_optional(cmd, "--compare-metric", args.compare_metric)
    _add_optional(cmd, "--compare-cohort-root", args.compare_cohort_root)
    _add_optional(cmd, "--cohort-region-name", args.cohort_region_name)
    _add_optional(cmd, "--cohort-region-label", args.cohort_region_label)
    _add_optional(cmd, "--cohort-metric", args.cohort_metric)
    _add_optional(cmd, "--log-file", args.log_file)
    if args.skip_cohort:
        cmd.append("--skip-cohort")
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{PIPELINE_DIR}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(PIPELINE_DIR)
    )
    print("[INFO] Report subject:", subject)
    print("[INFO] Command:", " ".join(cmd))
    return subprocess.run(cmd, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
