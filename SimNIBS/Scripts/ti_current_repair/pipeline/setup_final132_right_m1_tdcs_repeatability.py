#!/usr/bin/env python3
"""Review the blocked right-M1 tDCS repeatability plan without side effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_STIMULATION_FIELDS = (
    "montage_preset",
    "anode",
    "cathode",
    "current_ma",
    "electrode_radius_mm",
    "electrode_thickness_mm",
    "electrode_shape",
    "electrode_conductivity_s_per_m",
)


def load_and_validate_plan(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        plan = json.load(handle)
    if not isinstance(plan, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    target = plan.get("target", {})
    if target.get("roi_preset") != "right-m1" or target.get("optimizer_roi_name") != "Right_M1":
        raise ValueError("The pending plan must target the optimizer-matched right-M1 ROI.")
    cohort = plan.get("cohort", {})
    subjects = cohort.get("subjects", [])
    if len(subjects) != 10 or len(set(subjects)) != 10:
        raise ValueError("The pending plan must contain ten unique subjects.")
    arms = plan.get("design", {}).get("arms", [])
    task_sum = sum(int(arm.get("tdcs_solves", -1)) for arm in arms)
    if len(arms) != 3 or task_sum != 1200 or plan["design"].get("total_tdcs_solves") != 1200:
        raise ValueError("The pending plan must resolve to three arms and 1,200 tDCS solves.")
    stimulation = plan.get("stimulation_parameters", {})
    missing_keys = [key for key in REQUIRED_STIMULATION_FIELDS if key not in stimulation]
    if missing_keys:
        raise ValueError("Missing stimulation fields: " + ", ".join(missing_keys))
    if plan.get("submission_allowed") is not False:
        raise ValueError("A pending plan must have submission_allowed=false.")
    existing = plan.get("existing_inputs", {})
    if existing.get("policy") != "read_only":
        raise ValueError("Existing v3/v4 experiment roots must be declared read-only.")
    input_roots = [
        Path(existing["simnibs_v4_root"]),
        Path(existing["simnibs_v3_root"]),
    ]
    output_roots = [Path(value) for value in plan.get("proposed_output_roots", {}).values()]
    if len(output_roots) != 2 or len(set(output_roots)) != 2:
        raise ValueError("The proposed v3/v4 output roots must be distinct.")
    for output_root in output_roots:
        for input_root in input_roots:
            if output_root == input_root or output_root in input_root.parents or input_root in output_root.parents:
                raise ValueError(
                    "A proposed output root overlaps a read-only input root: "
                    f"{output_root} and {input_root}."
                )
    return plan


def run(args: argparse.Namespace) -> int:
    plan = load_and_validate_plan(args.plan)
    pending = [
        key for key in REQUIRED_STIMULATION_FIELDS
        if plan["stimulation_parameters"].get(key) is None
    ]
    if args.submit:
        raise RuntimeError(
            "Submission is blocked. This planning command cannot call sbatch; "
            "approve and record the tDCS parameters, complete the mesh audit, "
            "and pass the full 1,200-task replay preflight first."
        )

    summary = {
        "study": plan["study"],
        "status": plan["status"],
        "submission_allowed": plan["submission_allowed"],
        "target": plan["target"],
        "subject_count": plan["cohort"]["subject_count"],
        "repeats_per_arm": plan["cohort"]["repeats_per_arm"],
        "arms": plan["design"]["arms"],
        "total_tdcs_solves": plan["design"]["total_tdcs_solves"],
        "pending_stimulation_parameters": pending,
        "required_gates": plan["required_gates"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Explicit reviewed study plan; no repository-local scientific default is supplied.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Always fails while the plan is pending; retained as a fail-closed guard.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
