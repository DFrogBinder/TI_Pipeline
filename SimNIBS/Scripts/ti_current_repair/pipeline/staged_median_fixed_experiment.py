#!/usr/bin/env python3
"""Staged median-fixed repeatability experiment orchestration."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
PIPELINE_DIR = HERE.parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from experiment_config import repeat_tag  # noqa: E402
from pipeline import provenance  # noqa: E402
from post import make_presentation_figures, seed_fixed_from_median, select_median_remesh_repeats  # noqa: E402
from stimulation_config import (  # noqa: E402
    TARGETS_CSV_PATH,
    StimulationConfig,
    resolve_confirmed_stimulation,
    stimulation_summary,
    validate_stimulation_config,
)


CONFIG_DIRNAME = "configs"
REMESH_CONFIG = "remesh_only.json"
FIXED_CONFIG = "fixed_mesh_only.json"
PAIRED_CONFIG = "paired_analysis.json"
SELECTION_CSV = "median_mesh_selection/median_representative_remesh_repeats.csv"
SEED_MANIFEST = "fixed_seed_manifest.csv"
FIGURE_OUTPUTS = [
    "presentation_condition_summary.csv",
    "01_primary_median_roi_repeat_distributions.png",
    "condition_median_roi_by_repeat.png",
    "presentation_manifest.json",
]
WORKFLOW_STEPS = (
    "after-remesh",
    "select-seed",
    "after-fixed",
    "finalize",
)
WORKFLOW_CONTROLLER_CPUS = 1
WORKFLOW_CONTROLLER_MEMORY = "8G"
WORKFLOW_CONTROLLER_TIME = "08:00:00"
WORKFLOW_PARTITION = "sheffield"
SOURCE_SUFFIXES = (
    "_T1w.nii",
    "_T2w.nii",
    "_T1w_ras_1mm_T1andT2_masks.nii",
)


def _pipeline_root(experiment_root: Path) -> Path:
    return experiment_root / "_pipeline"


def _config_path(experiment_root: Path, name: str) -> Path:
    return _pipeline_root(experiment_root) / CONFIG_DIRNAME / name


def _events_path(experiment_root: Path) -> Path:
    return _pipeline_root(experiment_root) / "events.jsonl"


def _selection_csv(experiment_root: Path) -> Path:
    return _pipeline_root(experiment_root) / SELECTION_CSV


def _seed_manifest(experiment_root: Path) -> Path:
    return _pipeline_root(experiment_root) / SEED_MANIFEST


def _workflow_root(experiment_root: Path) -> Path:
    return _pipeline_root(experiment_root) / "workflow"


def _workflow_submission(experiment_root: Path) -> Path:
    return _workflow_root(experiment_root) / "submission.json"


def _workflow_job_ids(experiment_root: Path) -> Path:
    return _workflow_root(experiment_root) / "job_ids.tsv"


def _workflow_completion(experiment_root: Path) -> Path:
    return _workflow_root(experiment_root) / "complete.json"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    return provenance.write_json(path, payload)


def _subjects_from_arg(value: str) -> list[str]:
    subjects = [item.strip() for item in value.split(",") if item.strip()]
    if not subjects:
        raise argparse.ArgumentTypeError("At least one subject is required.")
    return subjects


def _validate_atlas_dir(atlas_dir: Path, subjects: list[str]) -> Path:
    atlas_dir = atlas_dir.expanduser().resolve()
    if not atlas_dir.is_dir():
        raise SystemExit(f"Atlas directory does not exist: {atlas_dir}")
    missing = [
        atlas_dir / f"{subject}.nii.gz"
        for subject in subjects
        if not (atlas_dir / f"{subject}.nii.gz").is_file()
    ]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise SystemExit(
            "Missing required subject atlas file(s). Atlas filenames must exactly match "
            f"<subject>.nii.gz: {formatted}"
        )
    return atlas_dir


def _base_config(
    *,
    source_root: Path,
    experiment_root: Path,
    subjects: list[str],
    repeat_count: int,
    roi_preset: str,
    atlas_dir: Path,
    compare_metric: str,
    stimulation: StimulationConfig,
) -> dict[str, Any]:
    analysis: dict[str, Any] = {
        "roi_preset": roi_preset,
        "compare_metric": compare_metric,
    }
    analysis["atlas_dir"] = str(atlas_dir)
    return {
        "source_root": str(source_root),
        "experiment_root": str(experiment_root),
        "subjects": subjects,
        "conditions": [
            {
                "name": "remesh",
                "mesh_mode": "remesh",
                "repeat_count": repeat_count,
                "description": "Fresh mesh for every repeat.",
            },
            {
                "name": "fixed_mesh",
                "mesh_mode": "fixed_mesh",
                "repeat_count": repeat_count,
                "description": "Physical copy of selected median remesh mesh reused for fixed repeats.",
            },
        ],
        "stimulation": stimulation.to_dict(),
        "analysis": analysis,
    }


def _config_with_conditions(config: dict[str, Any], names: set[str]) -> dict[str, Any]:
    out = json.loads(json.dumps(config))
    out["conditions"] = [condition for condition in out["conditions"] if condition["name"] in names]
    return out


def _load_pipeline_config(experiment_root: Path, name: str = PAIRED_CONFIG) -> dict[str, Any]:
    path = _config_path(experiment_root, name)
    if not path.is_file():
        raise FileNotFoundError(f"Missing staged pipeline config. Run init first: {path}")
    return _load_json(path)


def _condition_repeat_count(config: dict[str, Any], condition_name: str) -> int:
    for condition in config["conditions"]:
        if condition["name"] == condition_name:
            return int(condition["repeat_count"])
    raise KeyError(condition_name)


def _subject_count(config: dict[str, Any]) -> int:
    return len(config.get("subjects", []))


def _simulation_task_count(config: dict[str, Any]) -> int:
    return sum(_subject_count(config) * int(condition["repeat_count"]) for condition in config["conditions"])


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _workflow_configs(experiment_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    remesh = _load_pipeline_config(experiment_root, REMESH_CONFIG)
    fixed = _load_pipeline_config(experiment_root, FIXED_CONFIG)
    paired = _load_pipeline_config(experiment_root, PAIRED_CONFIG)
    subject_lists = [list(config.get("subjects", [])) for config in (remesh, fixed, paired)]
    if not subject_lists[0] or subject_lists[0] != subject_lists[1] or subject_lists[0] != subject_lists[2]:
        raise ValueError("Workflow configs must contain the same non-empty ordered subject list.")
    roots = [
        (str(config.get("source_root", "")), str(config.get("experiment_root", "")))
        for config in (remesh, fixed, paired)
    ]
    if len(set(roots)) != 1:
        raise ValueError("Workflow configs disagree on source_root or experiment_root.")
    if roots[0][1] != str(experiment_root):
        raise ValueError(
            f"Workflow config experiment_root is {roots[0][1]!r}, expected {str(experiment_root)!r}."
        )
    stimulations = [config.get("stimulation") for config in (remesh, fixed, paired)]
    if stimulations[0] != stimulations[1] or stimulations[0] != stimulations[2]:
        raise ValueError("Workflow configs disagree on stimulation parameters.")
    validate_stimulation_config(stimulations[0])
    remesh_conditions = remesh.get("conditions", [])
    fixed_conditions = fixed.get("conditions", [])
    paired_conditions = paired.get("conditions", [])
    if (
        len(remesh_conditions) != 1
        or remesh_conditions[0].get("name") != "remesh"
        or remesh_conditions[0].get("mesh_mode") != "remesh"
    ):
        raise ValueError("remesh_only.json must contain only the remesh condition.")
    if (
        len(fixed_conditions) != 1
        or fixed_conditions[0].get("name") != "fixed_mesh"
        or fixed_conditions[0].get("mesh_mode") != "fixed_mesh"
    ):
        raise ValueError("fixed_mesh_only.json must contain only the fixed_mesh condition.")
    if [condition.get("name") for condition in paired_conditions] != ["remesh", "fixed_mesh"]:
        raise ValueError("paired_analysis.json must contain remesh followed by fixed_mesh.")
    if int(remesh_conditions[0]["repeat_count"]) != int(fixed_conditions[0]["repeat_count"]):
        raise ValueError("Remesh and fixed-mesh repeat counts must match.")
    return remesh, fixed, paired


def _validate_workflow_inputs(config: dict[str, Any]) -> dict[str, Any]:
    subjects = list(config.get("subjects", []))
    source_root = Path(str(config.get("source_root", "")))
    if not source_root.is_dir():
        raise ValueError(f"Staged source root is not a directory: {source_root}")
    expected_relative_paths = [
        Path(subject) / "anat" / f"{subject}{suffix}"
        for subject in subjects
        for suffix in SOURCE_SUFFIXES
    ]
    source_issues = []
    for relative_path in expected_relative_paths:
        path = source_root / relative_path
        if not path.is_file():
            source_issues.append(f"missing {path}")
        elif path.is_symlink():
            source_issues.append(f"symlinked staged input {path}")
        elif path.stat().st_size <= 0:
            source_issues.append(f"empty {path}")
    if source_issues:
        raise ValueError(
            "Staged source validation failed: " + "; ".join(source_issues[:10])
        )

    subjects_file = source_root / "subjects.txt"
    if subjects_file.is_file():
        staged_subjects = [
            line.strip()
            for line in subjects_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if staged_subjects != subjects:
            raise ValueError(
                f"Staged subjects.txt does not exactly match workflow subjects: {subjects_file}"
            )

    manifest_path = source_root / "dataset_manifest.tsv"
    manifest_hashes_verified: int | str = "not available"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            manifest_rows = [dict(row) for row in csv.DictReader(handle, delimiter="\t")]
        by_destination = {
            row.get("destination", ""): row
            for row in manifest_rows
            if row.get("destination")
        }
        if len(by_destination) != len(manifest_rows):
            raise ValueError(f"Dataset manifest has duplicate or blank destinations: {manifest_path}")
        manifest_issues = []
        verified = 0
        for relative_path in expected_relative_paths:
            key = str(relative_path)
            row = by_destination.get(key)
            if row is None:
                manifest_issues.append(f"missing manifest row {key}")
                continue
            expected_hash = row.get("sha256", "")
            if len(expected_hash) != 64:
                manifest_issues.append(f"invalid manifest SHA-256 for {key}")
                continue
            actual_hash = provenance.file_sha256(source_root / relative_path)
            if actual_hash != expected_hash:
                manifest_issues.append(
                    f"manifest SHA-256 mismatch for {key}: expected {expected_hash}, observed {actual_hash}"
                )
                continue
            verified += 1
        if manifest_issues:
            raise ValueError(
                "Dataset manifest validation failed: " + "; ".join(manifest_issues[:10])
            )
        manifest_hashes_verified = verified

    analysis = config.get("analysis", {})
    atlas_dir = Path(str(analysis.get("atlas_dir", "")))
    if not atlas_dir.is_dir():
        raise ValueError(f"Subject atlas directory is not a directory: {atlas_dir}")
    missing_atlases = [
        atlas_dir / f"{subject}.nii.gz"
        for subject in subjects
        if not (atlas_dir / f"{subject}.nii.gz").is_file()
        or (atlas_dir / f"{subject}.nii.gz").stat().st_size <= 0
    ]
    if missing_atlases:
        raise ValueError(
            "Missing or empty exact subject atlas file(s): "
            + ", ".join(str(path) for path in missing_atlases)
        )
    stimulation = validate_stimulation_config(config.get("stimulation"))
    return {
        "source_root": str(source_root),
        "source_files_expected": len(expected_relative_paths),
        "source_files_ready": len(expected_relative_paths),
        "dataset_manifest": str(manifest_path) if manifest_path.is_file() else None,
        "manifest_hashes_verified": manifest_hashes_verified,
        "atlas_dir": str(atlas_dir),
        "atlases_expected": len(subjects),
        "atlases_ready": len(subjects),
        "stimulation": stimulation_summary(stimulation),
    }


def _workflow_scope(
    experiment_root: Path,
    *,
    max_concurrent: int,
    analysis_max_concurrent: int,
) -> dict[str, Any]:
    remesh, fixed, paired = _workflow_configs(experiment_root)
    subjects = list(paired["subjects"])
    input_validation = _validate_workflow_inputs(paired)
    remesh_tasks = _simulation_task_count(remesh)
    fixed_tasks = _simulation_task_count(fixed)
    roi = str(paired.get("analysis", {}).get("roi_preset", ""))
    source_root = str(paired.get("source_root", ""))
    return {
        "dataset": source_root,
        "roi": roi,
        "subjects": subjects,
        "subject_count": len(subjects),
        "repeats_per_condition": _condition_repeat_count(remesh, "remesh"),
        "remesh_tasks": remesh_tasks,
        "fixed_mesh_tasks": fixed_tasks,
        "total_simulation_tasks": remesh_tasks + fixed_tasks,
        "remesh_array": f"0-{remesh_tasks - 1}%{max_concurrent}",
        "fixed_mesh_array": f"0-{fixed_tasks - 1}%{max_concurrent}",
        "analysis_array": f"0-{len(subjects) - 1}%{analysis_max_concurrent}",
        "expected_ti_msh": remesh_tasks + fixed_tasks,
        "input_validation": input_validation,
        "execution_scope": "full requested experiment; not a smoke or subset",
    }


def _print_workflow_scope(scope: dict[str, Any]) -> None:
    print("Scope:")
    print(f"  dataset: {scope['dataset']}")
    print(f"  ROI: {scope['roi']}")
    print(f"  subjects: {scope['subject_count']}")
    print(f"  repeats per condition: {scope['repeats_per_condition']}")
    print(f"  remesh tasks: {scope['remesh_tasks']} ({scope['remesh_array']})")
    print(f"  fixed-mesh tasks: {scope['fixed_mesh_tasks']} ({scope['fixed_mesh_array']})")
    print(f"  total simulation tasks: {scope['total_simulation_tasks']}")
    print(f"  analysis array: {scope['analysis_array']}")
    print(f"  expected TI.msh outputs: {scope['expected_ti_msh']}")
    input_validation = scope["input_validation"]
    print(
        "  staged inputs: "
        f"{input_validation['source_files_ready']}/"
        f"{input_validation['source_files_expected']} ready"
    )
    print(
        "  subject atlases: "
        f"{input_validation['atlases_ready']}/"
        f"{input_validation['atlases_expected']} ready"
    )
    print(
        "  dataset manifest hashes: "
        f"{input_validation['manifest_hashes_verified']}"
    )
    stimulation = input_validation["stimulation"]
    print(
        "  stimulation: "
        f"{stimulation['montage_preset']} ({stimulation['target_roi']})"
    )
    print(f"  targets.csv: {stimulation['targets_csv']}")
    print(f"  targets.csv SHA-256: {stimulation['targets_csv_sha256']}")
    print(f"  pair 1: {stimulation['pair1']}")
    print(f"  pair 2: {stimulation['pair2']}")
    print(f"  execution: {scope['execution_scope']}")


def _env_subset(env: dict[str, str], keys: list[str]) -> dict[str, str]:
    return {key: env[key] for key in keys if key in env}


def _submit_command_event(
    *,
    experiment_root: Path,
    stage: str,
    command: list[str],
    env: dict[str, str],
    stdout: str,
    stderr: str,
    returncode: int,
    job_id: str | None,
    expected_outputs: dict[str, Any],
) -> None:
    provenance.write_submitted_job_record(
        experiment_root,
        stage=stage,
        command=command,
        env=env,
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        job_id=job_id,
        expected_outputs=expected_outputs,
    )
    provenance.append_event(
        _events_path(experiment_root),
        "submit",
        stage=stage,
        command=command,
        env=env,
        stdout_tail=stdout[-4000:],
        stderr_tail=stderr[-4000:],
        returncode=returncode,
        job_id=job_id,
        expected_outputs=expected_outputs,
    )


def _run_submitter(script: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    run_env = os.environ.copy()
    run_env.update(env)
    return subprocess.run(
        ["bash", str(script)],
        cwd=str(PIPELINE_DIR),
        env=run_env,
        text=True,
        capture_output=True,
        check=False,
    )


def _append_workflow_job(
    experiment_root: Path,
    *,
    stage: str,
    job_id: str,
    dependency: str,
) -> None:
    path = _workflow_job_ids(experiment_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp_utc", "stage", "job_id", "dependency"],
            delimiter="\t",
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "stage": stage,
                "job_id": job_id,
                "dependency": dependency,
            }
        )


def _cancel_job(experiment_root: Path, job_id: str, *, reason: str) -> None:
    command = [os.environ.get("SCANCEL_BIN", "scancel"), job_id]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    provenance.append_event(
        _events_path(experiment_root),
        "workflow_cancel",
        job_id=job_id,
        reason=reason,
        command=command,
        returncode=result.returncode,
        stdout_tail=result.stdout[-4000:],
        stderr_tail=result.stderr[-4000:],
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to cancel unmanaged job {job_id}: {result.stderr or result.stdout}"
        )


def _submit_controller(
    *,
    experiment_root: Path,
    afterok_job_id: str,
    step: str,
    max_concurrent: int,
    analysis_max_concurrent: int,
) -> tuple[int, str | None]:
    if step not in WORKFLOW_STEPS:
        raise ValueError(f"Unsupported workflow step: {step}")
    script = PIPELINE_DIR / "hpc_scripts" / "repeatability_workflow_controller.slurm"
    if not script.is_file():
        raise FileNotFoundError(script)
    log_dir = _workflow_root(experiment_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "PIPELINE_DIR": str(PIPELINE_DIR),
        "EXPERIMENT_ROOT": str(experiment_root),
        "WORKFLOW_STEP": step,
        "MAX_CONCURRENT_TASKS": str(max_concurrent),
        "ANALYSIS_MAX_CONCURRENT_TASKS": str(analysis_max_concurrent),
        "SBATCH_BIN": os.environ.get("SBATCH_BIN", "sbatch"),
        "SCANCEL_BIN": os.environ.get("SCANCEL_BIN", "scancel"),
    }
    invalid_exports = {
        key: value for key, value in env.items() if "," in value or "\n" in value
    }
    if invalid_exports:
        raise ValueError(f"Slurm export values cannot contain commas or newlines: {invalid_exports}")
    dependency = f"afterok:{afterok_job_id}"
    export_vars = "ALL," + ",".join(f"{key}={value}" for key, value in env.items())
    command = [
        os.environ.get("SBATCH_BIN", "sbatch"),
        f"--job-name=ti_repeat_flow_{step}",
        f"--partition={WORKFLOW_PARTITION}",
        f"--cpus-per-task={WORKFLOW_CONTROLLER_CPUS}",
        f"--mem={WORKFLOW_CONTROLLER_MEMORY}",
        f"--time={WORKFLOW_CONTROLLER_TIME}",
        f"--dependency={dependency}",
        f"--output={log_dir / f'{step}-%j.out'}",
        f"--export={export_vars}",
        str(script),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    job_id = provenance.parse_sbatch_job_id(result.stdout)
    returncode = result.returncode
    stderr = result.stderr
    if returncode == 0 and job_id is None:
        returncode = 2
        stderr = (
            f"{stderr}\n" if stderr else ""
        ) + "sbatch succeeded but no controller job ID could be parsed."
    stage = f"workflow-controller-{step}"
    _submit_command_event(
        experiment_root=experiment_root,
        stage=stage,
        command=command,
        env=env,
        stdout=result.stdout,
        stderr=stderr,
        returncode=returncode,
        job_id=job_id,
        expected_outputs={"workflow_step": step},
    )
    if result.stdout:
        print(result.stdout, end="")
    if stderr:
        print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr)
    if returncode == 0 and job_id is not None:
        _append_workflow_job(
            experiment_root,
            stage=stage,
            job_id=job_id,
            dependency=dependency,
        )
    return returncode, job_id


def _attach_controller_or_cancel(
    *,
    experiment_root: Path,
    child_job_id: str,
    step: str,
    max_concurrent: int,
    analysis_max_concurrent: int,
) -> str:
    returncode, controller_job_id = _submit_controller(
        experiment_root=experiment_root,
        afterok_job_id=child_job_id,
        step=step,
        max_concurrent=max_concurrent,
        analysis_max_concurrent=analysis_max_concurrent,
    )
    if returncode != 0 or controller_job_id is None:
        _cancel_job(
            experiment_root,
            child_job_id,
            reason=f"failed to attach workflow controller for {step}",
        )
        raise RuntimeError(
            f"Controller submission for {step} failed; cancelled child job {child_job_id}."
        )
    return controller_job_id


def _simulation_sbatch_command(
    *,
    config: dict[str, Any],
    config_path: Path,
    log_dir: Path,
    max_concurrent: int,
    stage: str,
) -> list[str]:
    task_count = _simulation_task_count(config)
    array_spec = f"0-{task_count - 1}%{max_concurrent}"
    return [
        os.environ.get("SBATCH_BIN", "sbatch"),
        f"--job-name=ti_repeat_{stage}",
        "--cpus-per-task=8",
        "--mem=32G",
        "--time=08:00:00",
        f"--array={array_spec}",
        (
            "EXPERIMENT_CONFIG="
            f"{config_path},PIPELINE_DIR={PIPELINE_DIR},LOG_DIR={log_dir},"
            "OVERWRITE_OUTPUT=0,FORCE_MESH=0"
        ),
        str(PIPELINE_DIR / "hpc_scripts" / "repeatability_experiment_array.slurm"),
    ]


def _report_sbatch_command(
    *,
    config: dict[str, Any],
    config_path: Path,
    log_dir: Path,
    max_concurrent: int,
    conditions: str,
    stage: str,
) -> list[str]:
    array_spec = f"0-{_subject_count(config) - 1}%{max_concurrent}"
    return [
        os.environ.get("SBATCH_BIN", "sbatch"),
        f"--job-name=ti_repeat_{stage}",
        "--cpus-per-task=8",
        "--mem=32G",
        "--time=08:00:00",
        f"--array={array_spec}",
        (
            "EXPERIMENT_CONFIG="
            f"{config_path},LOG_DIR={log_dir},CONDITIONS={conditions},"
            f"PIPELINE_DIR={PIPELINE_DIR},"
            f"REPORT_TASK_PY={PIPELINE_DIR / 'hpc_scripts' / 'repeatability_report_array_task.py'}"
        ),
        str(PIPELINE_DIR / "hpc_scripts" / "repeatability_experiment_report_array.slurm"),
    ]


def command_init(args: argparse.Namespace) -> int:
    source_root = args.source_root.expanduser().resolve()
    experiment_root = args.experiment_root.expanduser().resolve()
    subjects = _subjects_from_arg(args.subjects)
    atlas_dir = _validate_atlas_dir(args.atlas_dir, subjects)
    stimulation = resolve_confirmed_stimulation(
        args.montage_preset,
        targets_csv=args.targets_csv,
    )
    if stimulation.montage_preset != args.roi_preset:
        raise ValueError(
            "Analysis ROI and stimulation montage must match for this repeatability "
            f"workflow: {args.roi_preset!r} != {stimulation.montage_preset!r}."
        )
    base = _base_config(
        source_root=source_root,
        experiment_root=experiment_root,
        subjects=subjects,
        repeat_count=args.repeat_count,
        roi_preset=args.roi_preset,
        atlas_dir=atlas_dir,
        compare_metric=args.compare_metric,
        stimulation=stimulation,
    )
    configs = {
        REMESH_CONFIG: _config_with_conditions(base, {"remesh"}),
        FIXED_CONFIG: _config_with_conditions(base, {"fixed_mesh"}),
        PAIRED_CONFIG: base,
    }
    if args.dry_run:
        print(json.dumps({"configs": configs}, indent=2))
        return 0

    _pipeline_root(experiment_root).mkdir(parents=True, exist_ok=True)
    (_pipeline_root(experiment_root) / "logs").mkdir(exist_ok=True)
    for name, payload in configs.items():
        _write_json(_config_path(experiment_root, name), payload)
    _write_json(
        _pipeline_root(experiment_root) / "experiment_manifest.json",
        {
            "source_root": str(source_root),
            "experiment_root": str(experiment_root),
            "subjects": subjects,
            "repeat_count": args.repeat_count,
            "roi_preset": args.roi_preset,
            "compare_metric": args.compare_metric,
            "atlas_dir": str(atlas_dir),
            "stimulation": stimulation.to_dict(),
            "configs": {name: str(_config_path(experiment_root, name)) for name in configs},
        },
    )
    provenance.append_event(
        _events_path(experiment_root),
        "stage_init",
        source_root=str(source_root),
        experiment_root=str(experiment_root),
        subjects=subjects,
        repeat_count=args.repeat_count,
        atlas_dir=str(atlas_dir),
        stimulation=stimulation.to_dict(),
    )
    provenance.write_stage_status(experiment_root, collect_status(experiment_root))
    print(f"initialized staged pipeline: {_pipeline_root(experiment_root)}")
    return 0


def _submit_simulation_stage_result(
    args: argparse.Namespace,
    *,
    stage: str,
    config_name: str,
) -> tuple[int, str | None]:
    experiment_root = args.experiment_root.expanduser().resolve()
    config_path = _config_path(experiment_root, config_name)
    config = _load_pipeline_config(experiment_root, config_name)
    log_dir = _pipeline_root(experiment_root) / "logs" / stage
    env = {
        "PIPELINE_DIR": str(PIPELINE_DIR),
        "EXPERIMENT_CONFIG": str(config_path),
        "LOG_DIR": str(log_dir),
        "MAX_CONCURRENT_TASKS": str(args.max_concurrent),
        "JOB_NAME": f"ti_repeat_{stage}",
        "OVERWRITE_OUTPUT": "0",
        "FORCE_MESH": "0",
    }
    result = _run_submitter(PIPELINE_DIR / "hpc_scripts" / "submit_repeatability_experiment.sh", env)
    job_id = provenance.parse_sbatch_job_id(result.stdout)
    command = _simulation_sbatch_command(
        config=config,
        config_path=config_path,
        log_dir=log_dir,
        max_concurrent=args.max_concurrent,
        stage=stage,
    )
    _submit_command_event(
        experiment_root=experiment_root,
        stage=stage,
        command=command,
        env=_env_subset(env, sorted(env)),
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
        job_id=job_id,
        expected_outputs={"ti_msh": _simulation_task_count(config)},
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode, job_id


def _submit_simulation_stage(args: argparse.Namespace, *, stage: str, config_name: str) -> int:
    return _submit_simulation_stage_result(
        args,
        stage=stage,
        config_name=config_name,
    )[0]


def _submit_report_stage_result(
    args: argparse.Namespace,
    *,
    stage: str,
    config_name: str,
    conditions: str,
) -> tuple[int, str | None]:
    experiment_root = args.experiment_root.expanduser().resolve()
    config_path = _config_path(experiment_root, config_name)
    config = _load_pipeline_config(experiment_root, config_name)
    log_dir = _pipeline_root(experiment_root) / "logs" / stage
    env = {
        "PIPELINE_DIR": str(PIPELINE_DIR),
        "EXPERIMENT_CONFIG": str(config_path),
        "LOG_DIR": str(log_dir),
        "MAX_CONCURRENT_TASKS": str(args.max_concurrent),
        "JOB_NAME": f"ti_repeat_{stage}",
        "CONDITIONS": conditions,
    }
    result = _run_submitter(PIPELINE_DIR / "hpc_scripts" / "submit_repeatability_report_array.sh", env)
    job_id = provenance.parse_sbatch_job_id(result.stdout)
    command = _report_sbatch_command(
        config=config,
        config_path=config_path,
        log_dir=log_dir,
        max_concurrent=args.max_concurrent,
        conditions=conditions,
        stage=stage,
    )
    _submit_command_event(
        experiment_root=experiment_root,
        stage=stage,
        command=command,
        env=_env_subset(env, sorted(env)),
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
        job_id=job_id,
        expected_outputs={"subject_reports": _subject_count(config)},
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode, job_id


def _submit_report_stage(
    args: argparse.Namespace,
    *,
    stage: str,
    config_name: str,
    conditions: str,
) -> int:
    return _submit_report_stage_result(
        args,
        stage=stage,
        config_name=config_name,
        conditions=conditions,
    )[0]


def command_submit_remesh(args: argparse.Namespace) -> int:
    return _submit_simulation_stage(args, stage="submit-remesh", config_name=REMESH_CONFIG)


def command_submit_fixed(args: argparse.Namespace) -> int:
    return _submit_simulation_stage(args, stage="submit-fixed", config_name=FIXED_CONFIG)


def command_analyze_remesh(args: argparse.Namespace) -> int:
    return _submit_report_stage(args, stage="analyze-remesh", config_name=REMESH_CONFIG, conditions="remesh")


def command_analyze_paired(args: argparse.Namespace) -> int:
    return _submit_report_stage(
        args,
        stage="analyze-paired",
        config_name=PAIRED_CONFIG,
        conditions="",
    )


def command_select_medians(args: argparse.Namespace) -> int:
    experiment_root = args.experiment_root.expanduser().resolve()
    config = _load_pipeline_config(experiment_root, PAIRED_CONFIG)
    output_csv = _selection_csv(experiment_root)
    selections = select_median_remesh_repeats.select_medians(
        experiment_root=experiment_root,
        subjects=list(config["subjects"]),
        metric=args.metric,
        output_csv=output_csv,
    )
    provenance.append_event(
        _events_path(experiment_root),
        "stage_done",
        stage="select-medians",
        output_csv=str(output_csv),
        selected=sum(1 for selection in selections if selection.selection_status == "selected"),
        total=len(selections),
    )
    provenance.write_stage_status(experiment_root, collect_status(experiment_root))
    print(f"wrote {output_csv}")
    return 0


def command_seed_fixed(args: argparse.Namespace) -> int:
    experiment_root = args.experiment_root.expanduser().resolve()
    config = _load_pipeline_config(experiment_root, FIXED_CONFIG)
    repeat_count = _condition_repeat_count(config, "fixed_mesh")
    manifest = seed_fixed_from_median.seed_fixed_meshes(
        experiment_root=experiment_root,
        selection_csv=_selection_csv(experiment_root),
        repeat_count=repeat_count,
        manifest=_seed_manifest(experiment_root),
        overwrite=args.overwrite,
    )
    provenance.append_event(
        _events_path(experiment_root),
        "stage_done",
        stage="seed-fixed",
        manifest=str(manifest),
        repeat_count=repeat_count,
    )
    provenance.write_stage_status(experiment_root, collect_status(experiment_root))
    print(f"wrote {manifest}")
    return 0


def command_make_figures(args: argparse.Namespace) -> int:
    experiment_root = args.experiment_root.expanduser().resolve()
    outputs = make_presentation_figures.make_figures(experiment_root=experiment_root)
    provenance.append_event(
        _events_path(experiment_root),
        "stage_done",
        stage="make-figures",
        **outputs,
    )
    provenance.write_stage_status(experiment_root, collect_status(experiment_root))
    print(json.dumps(outputs, indent=2))
    return 0


def _require_complete(label: str, status: dict[str, Any]) -> None:
    expected = int(status.get("expected", 0))
    observed = int(status.get("observed", 0))
    if expected < 1 or observed != expected:
        raise RuntimeError(
            f"Workflow gate failed for {label}: observed {observed}, expected {expected}."
        )


def _write_workflow_step_receipt(
    experiment_root: Path,
    *,
    step: str,
    payload: dict[str, Any],
) -> Path:
    path = _workflow_root(experiment_root) / "receipts" / f"{step}.json"
    return _write_json(
        path,
        {
            "schema_version": 1,
            "step": step,
            "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **payload,
        },
    )


def _require_submission(
    *,
    stage: str,
    returncode: int,
    job_id: str | None,
) -> str:
    if returncode != 0:
        raise RuntimeError(f"{stage} submission failed with exit code {returncode}.")
    if job_id is None:
        raise RuntimeError(f"{stage} submission returned no parseable Slurm job ID.")
    return job_id


def command_submit_all(args: argparse.Namespace) -> int:
    experiment_root = args.experiment_root.expanduser().resolve()
    scope = _workflow_scope(
        experiment_root,
        max_concurrent=args.max_concurrent,
        analysis_max_concurrent=args.analysis_max_concurrent,
    )
    _print_workflow_scope(scope)
    print(
        "  workflow: remesh -> remesh analysis -> median selection -> physical "
        "fixed seeding -> fixed simulations -> paired analysis -> figures"
    )
    print(
        "  controller resources: "
        f"{WORKFLOW_CONTROLLER_CPUS} CPU, {WORKFLOW_CONTROLLER_MEMORY}, "
        f"{WORKFLOW_CONTROLLER_TIME}, {WORKFLOW_PARTITION}"
    )
    if args.dry_run:
        print("[READY] Workflow preflight passed; no jobs submitted.")
        return 0
    submission_path = _workflow_submission(experiment_root)
    if submission_path.exists():
        raise RuntimeError(
            f"Workflow was already submitted for this experiment: {submission_path}"
        )
    if _workflow_completion(experiment_root).exists():
        raise RuntimeError(
            f"Workflow is already complete: {_workflow_completion(experiment_root)}"
        )
    submitted_job_dir = _pipeline_root(experiment_root) / "submitted_jobs"
    blocking_job_records = []
    failed_pre_submission_records = []
    for record_path in sorted(submitted_job_dir.glob("*.json")):
        try:
            record = _load_json(record_path)
        except (OSError, ValueError, json.JSONDecodeError):
            blocking_job_records.append(record_path)
            continue
        returncode = record.get("returncode")
        job_id = record.get("job_id")
        if isinstance(returncode, int) and returncode != 0 and not job_id:
            failed_pre_submission_records.append(record_path)
        else:
            blocking_job_records.append(record_path)
    if blocking_job_records:
        raise RuntimeError(
            "Refusing to mix the automated chain with prior manual submissions: "
            + ", ".join(str(path) for path in blocking_job_records)
        )
    if failed_pre_submission_records:
        print(
            "[INFO] Retrying after failed pre-sbatch attempt(s); their receipts "
            "will be archived automatically: "
            + ", ".join(str(path) for path in failed_pre_submission_records)
        )

    submit_args = argparse.Namespace(
        experiment_root=experiment_root,
        max_concurrent=args.max_concurrent,
    )
    returncode, remesh_job_id = _submit_simulation_stage_result(
        submit_args,
        stage="submit-remesh",
        config_name=REMESH_CONFIG,
    )
    remesh_job_id = _require_submission(
        stage="remesh",
        returncode=returncode,
        job_id=remesh_job_id,
    )
    _append_workflow_job(
        experiment_root,
        stage="submit-remesh",
        job_id=remesh_job_id,
        dependency="",
    )
    controller_job_id = _attach_controller_or_cancel(
        experiment_root=experiment_root,
        child_job_id=remesh_job_id,
        step="after-remesh",
        max_concurrent=args.max_concurrent,
        analysis_max_concurrent=args.analysis_max_concurrent,
    )
    payload = {
        "schema_version": 1,
        "status": "submitted",
        "submitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": scope,
        "settings": {
            "partition": WORKFLOW_PARTITION,
            "simulation_cpus": 8,
            "simulation_memory": "32G",
            "simulation_time": "08:00:00",
            "max_concurrent_tasks": args.max_concurrent,
            "analysis_max_concurrent_tasks": args.analysis_max_concurrent,
            "controller_cpus": WORKFLOW_CONTROLLER_CPUS,
            "controller_memory": WORKFLOW_CONTROLLER_MEMORY,
            "controller_time": WORKFLOW_CONTROLLER_TIME,
            "dependency": "afterok",
        },
        "initial_jobs": {
            "remesh": remesh_job_id,
            "after_remesh_controller": controller_job_id,
        },
    }
    _write_json(submission_path, payload)
    provenance.append_event(
        _events_path(experiment_root),
        "workflow_submit",
        scope=scope,
        remesh_job_id=remesh_job_id,
        controller_job_id=controller_job_id,
    )
    print(f"[OK] Submitted remesh array: {remesh_job_id}")
    print(
        f"[OK] Attached automatic workflow controller: {controller_job_id} "
        f"(afterok:{remesh_job_id})"
    )
    print(f"[OK] Workflow provenance: {submission_path}")
    return 0


def _submit_next_report(
    *,
    experiment_root: Path,
    stage: str,
    config_name: str,
    conditions: str,
    next_step: str,
    max_concurrent: int,
    analysis_max_concurrent: int,
) -> tuple[str, str]:
    submit_args = argparse.Namespace(
        experiment_root=experiment_root,
        max_concurrent=analysis_max_concurrent,
    )
    returncode, job_id = _submit_report_stage_result(
        submit_args,
        stage=stage,
        config_name=config_name,
        conditions=conditions,
    )
    job_id = _require_submission(stage=stage, returncode=returncode, job_id=job_id)
    _append_workflow_job(
        experiment_root,
        stage=stage,
        job_id=job_id,
        dependency="controller_gate",
    )
    controller_job_id = _attach_controller_or_cancel(
        experiment_root=experiment_root,
        child_job_id=job_id,
        step=next_step,
        max_concurrent=max_concurrent,
        analysis_max_concurrent=analysis_max_concurrent,
    )
    return job_id, controller_job_id


def command_advance_workflow(args: argparse.Namespace) -> int:
    experiment_root = args.experiment_root.expanduser().resolve()
    _, _, paired = _workflow_configs(experiment_root)
    status = collect_status(experiment_root)

    if args.step == "after-remesh":
        _require_complete("remesh TI.msh outputs", status["remesh_ti_msh"])
        report_job, controller_job = _submit_next_report(
            experiment_root=experiment_root,
            stage="analyze-remesh",
            config_name=REMESH_CONFIG,
            conditions="remesh",
            next_step="select-seed",
            max_concurrent=args.max_concurrent,
            analysis_max_concurrent=args.analysis_max_concurrent,
        )
        _write_workflow_step_receipt(
            experiment_root,
            step=args.step,
            payload={
                "remesh_ti_msh": status["remesh_ti_msh"],
                "report_job_id": report_job,
                "next_controller_job_id": controller_job,
            },
        )
        return 0

    if args.step == "select-seed":
        _require_complete("remesh subject summaries", status["remesh_summaries"])
        metric = str(paired.get("analysis", {}).get("compare_metric", "median_roi"))
        _require_summary_metric_coverage(
            paired,
            condition_name="remesh",
            metric=metric,
        )
        command_select_medians(
            argparse.Namespace(experiment_root=experiment_root, metric=metric)
        )
        selection_status = collect_status(experiment_root)["selected_medians"]
        _require_complete("median remesh selections", selection_status)
        command_seed_fixed(
            argparse.Namespace(experiment_root=experiment_root, overwrite=False)
        )
        seed_status = collect_status(experiment_root)["fixed_seed"]
        _require_complete("fixed-mesh seed rows", seed_status)
        if int(seed_status.get("symlink_count", 0)) != 0:
            raise RuntimeError("Workflow gate failed: fixed-mesh seeds contain symlinks.")
        if int(seed_status.get("checksum_mismatches", 0)) != 0:
            raise RuntimeError("Workflow gate failed: fixed-mesh seed checksum mismatch.")

        submit_args = argparse.Namespace(
            experiment_root=experiment_root,
            max_concurrent=args.max_concurrent,
        )
        returncode, fixed_job_id = _submit_simulation_stage_result(
            submit_args,
            stage="submit-fixed",
            config_name=FIXED_CONFIG,
        )
        fixed_job_id = _require_submission(
            stage="fixed-mesh",
            returncode=returncode,
            job_id=fixed_job_id,
        )
        _append_workflow_job(
            experiment_root,
            stage="submit-fixed",
            job_id=fixed_job_id,
            dependency="controller_gate",
        )
        controller_job = _attach_controller_or_cancel(
            experiment_root=experiment_root,
            child_job_id=fixed_job_id,
            step="after-fixed",
            max_concurrent=args.max_concurrent,
            analysis_max_concurrent=args.analysis_max_concurrent,
        )
        _write_workflow_step_receipt(
            experiment_root,
            step=args.step,
            payload={
                "selected_medians": selection_status,
                "fixed_seed": seed_status,
                "fixed_job_id": fixed_job_id,
                "next_controller_job_id": controller_job,
            },
        )
        return 0

    if args.step == "after-fixed":
        _require_complete("fixed-mesh TI.msh outputs", status["fixed_mesh_ti_msh"])
        report_job, controller_job = _submit_next_report(
            experiment_root=experiment_root,
            stage="analyze-paired",
            config_name=PAIRED_CONFIG,
            conditions="",
            next_step="finalize",
            max_concurrent=args.max_concurrent,
            analysis_max_concurrent=args.analysis_max_concurrent,
        )
        _write_workflow_step_receipt(
            experiment_root,
            step=args.step,
            payload={
                "fixed_mesh_ti_msh": status["fixed_mesh_ti_msh"],
                "report_job_id": report_job,
                "next_controller_job_id": controller_job,
            },
        )
        return 0

    if args.step == "finalize":
        _require_complete("remesh subject summaries", status["remesh_summaries"])
        _require_complete("fixed-mesh subject summaries", status["fixed_mesh_summaries"])
        command_make_figures(argparse.Namespace(experiment_root=experiment_root))
        final_status = collect_status(experiment_root)
        _require_complete("presentation figure outputs", final_status["figure_outputs"])
        completion = {
            "schema_version": 1,
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scope": _workflow_scope(
                experiment_root,
                max_concurrent=args.max_concurrent,
                analysis_max_concurrent=args.analysis_max_concurrent,
            ),
            "status_snapshot": final_status,
            "job_ids_file": str(_workflow_job_ids(experiment_root)),
        }
        _write_json(_workflow_completion(experiment_root), completion)
        _write_workflow_step_receipt(
            experiment_root,
            step=args.step,
            payload={"completion": str(_workflow_completion(experiment_root))},
        )
        provenance.append_event(
            _events_path(experiment_root),
            "workflow_complete",
            completion=str(_workflow_completion(experiment_root)),
        )
        print(f"[OK] Full workflow complete: {_workflow_completion(experiment_root)}")
        return 0

    raise ValueError(f"Unsupported workflow step: {args.step}")


def _exists_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _count_ti_msh(config: dict[str, Any], condition_name: str) -> tuple[int, int]:
    repeat_count = _condition_repeat_count(config, condition_name)
    expected = _subject_count(config) * repeat_count
    observed = 0
    root = Path(config["experiment_root"])
    for subject in config["subjects"]:
        for index in range(1, repeat_count + 1):
            path = (
                root
                / f"{subject}_repeatability"
                / condition_name
                / "repeats"
                / repeat_tag(index)
                / subject
                / "anat"
                / "SimNIBS"
                / "Output"
                / subject
                / "TI.msh"
            )
            if _exists_nonempty(path):
                observed += 1
    return expected, observed


def _count_summaries(config: dict[str, Any], condition_name: str) -> tuple[int, int]:
    root = Path(config["experiment_root"])
    repeat_count = _condition_repeat_count(config, condition_name)
    expected_tags = {repeat_tag(index) for index in range(1, repeat_count + 1)}
    expected = _subject_count(config)
    observed = 0
    for subject in config["subjects"]:
        path = root / "_analysis" / subject / condition_name / "summary.csv"
        if not _exists_nonempty(path):
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = [dict(row) for row in csv.DictReader(handle)]
        except (OSError, UnicodeError, csv.Error):
            continue
        tags = [row.get("repeat_tag", "") for row in rows]
        if len(rows) == repeat_count and len(tags) == len(set(tags)) and set(tags) == expected_tags:
            observed += 1
    return expected, observed


def _require_summary_metric_coverage(
    config: dict[str, Any],
    *,
    condition_name: str,
    metric: str,
) -> None:
    root = Path(config["experiment_root"])
    repeat_count = _condition_repeat_count(config, condition_name)
    expected_tags = {repeat_tag(index) for index in range(1, repeat_count + 1)}
    issues = []
    for subject in config["subjects"]:
        path = root / "_analysis" / subject / condition_name / "summary.csv"
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = [dict(row) for row in csv.DictReader(handle)]
        except (OSError, UnicodeError, csv.Error) as exc:
            issues.append(f"{subject}: unreadable summary {path}: {exc}")
            continue
        tags = [row.get("repeat_tag", "") for row in rows]
        finite_tags = set()
        for row in rows:
            try:
                value = float(row.get(metric, ""))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                finite_tags.add(row.get("repeat_tag", ""))
        if (
            len(rows) != repeat_count
            or len(tags) != len(set(tags))
            or set(tags) != expected_tags
            or finite_tags != expected_tags
        ):
            issues.append(
                f"{subject}: expected {repeat_count} unique finite {metric} rows in {path}"
            )
    if issues:
        raise RuntimeError(
            f"Workflow gate failed for {condition_name} {metric} coverage: "
            + "; ".join(issues[:10])
        )


def _count_selected_medians(experiment_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = _selection_csv(experiment_root)
    rows = []
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    return {
        "path": str(path),
        "expected": _subject_count(config),
        "observed": sum(1 for row in rows if row.get("selection_status") == "selected"),
        "rows": len(rows),
    }


def _count_seed_rows(experiment_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = _seed_manifest(experiment_root)
    rows = []
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    expected_repeat_count = _condition_repeat_count(config, "fixed_mesh")
    ok_rows = [row for row in rows if row.get("validation_result") == "ok"]
    validated_rows = 0
    checksum_mismatches = 0
    missing_meshes = 0
    repeat_count_mismatches = 0
    symlink_paths: list[str] = []
    for row in ok_rows:
        subject = row.get("subject", "")
        repeat_anat_dirs = [
            Path(raw)
            for raw in row.get("repeat_anat_dirs", "").split(";")
            if raw
        ]
        if len(repeat_anat_dirs) != expected_repeat_count:
            repeat_count_mismatches += 1
        anat_dirs = [Path(row.get("cache_anat_dir", "")), *repeat_anat_dirs]
        checksum = row.get("mesh_checksum", "")
        row_missing = 0
        row_mismatches = 0
        row_symlinks = []
        for anat_dir in anat_dirs:
            row_symlinks.extend(
                str(path) for path in provenance.find_symlinks(anat_dir)
            )
            mesh = anat_dir / f"m2m_{subject}" / f"{subject}.msh"
            if not mesh.is_file():
                row_missing += 1
            elif not checksum or provenance.file_sha256(mesh) != checksum:
                row_mismatches += 1
        symlink_paths.extend(row_symlinks)
        missing_meshes += row_missing
        checksum_mismatches += row_mismatches
        if (
            len(repeat_anat_dirs) == expected_repeat_count
            and row_missing == 0
            and row_mismatches == 0
            and not row_symlinks
        ):
            validated_rows += 1
    return {
        "path": str(path),
        "expected": _subject_count(config),
        "observed": validated_rows,
        "rows": len(rows),
        "symlink_count": len(symlink_paths),
        "symlinks": symlink_paths[:20],
        "checksum_mismatches": checksum_mismatches,
        "missing_meshes": missing_meshes,
        "repeat_count_mismatches": repeat_count_mismatches,
    }


def _figure_status(experiment_root: Path) -> dict[str, Any]:
    root = experiment_root / "_figures" / "presentation"
    expected_paths = [root / name for name in FIGURE_OUTPUTS]
    return {
        "path": str(root),
        "expected": len(expected_paths),
        "observed": sum(1 for path in expected_paths if _exists_nonempty(path)),
        "missing": [str(path) for path in expected_paths if not _exists_nonempty(path)],
    }


def collect_status(experiment_root: Path) -> dict[str, Any]:
    experiment_root = experiment_root.expanduser().resolve()
    config_path = _config_path(experiment_root, PAIRED_CONFIG)
    if config_path.is_file():
        config = _load_json(config_path)
    else:
        config = {
            "experiment_root": str(experiment_root),
            "subjects": [],
            "conditions": [],
        }
    status: dict[str, Any] = {
        "experiment_root": str(experiment_root),
        "config_path": str(config_path),
        "subjects": len(config.get("subjects", [])),
    }
    for condition_name in ("remesh", "fixed_mesh"):
        try:
            expected, observed = _count_ti_msh(config, condition_name)
            status[f"{condition_name}_ti_msh"] = {"expected": expected, "observed": observed}
            expected_summary, observed_summary = _count_summaries(config, condition_name)
            status[f"{condition_name}_summaries"] = {
                "expected": expected_summary,
                "observed": observed_summary,
            }
        except KeyError:
            status[f"{condition_name}_ti_msh"] = {"expected": 0, "observed": 0}
            status[f"{condition_name}_summaries"] = {"expected": 0, "observed": 0}
    status["selected_medians"] = _count_selected_medians(experiment_root, config)
    status["fixed_seed"] = _count_seed_rows(experiment_root, config)
    status["figure_outputs"] = _figure_status(experiment_root)
    workflow_jobs = _workflow_job_ids(experiment_root)
    job_rows = []
    if workflow_jobs.is_file():
        with workflow_jobs.open("r", encoding="utf-8", newline="") as handle:
            job_rows = [dict(row) for row in csv.DictReader(handle, delimiter="\t")]
    status["workflow"] = {
        "submission": str(_workflow_submission(experiment_root)),
        "submitted": _workflow_submission(experiment_root).is_file(),
        "completion": str(_workflow_completion(experiment_root)),
        "complete": _workflow_completion(experiment_root).is_file(),
        "job_ids": str(workflow_jobs),
        "jobs_recorded": len(job_rows),
        "latest_stage": job_rows[-1]["stage"] if job_rows else None,
        "latest_job_id": job_rows[-1]["job_id"] if job_rows else None,
    }
    return status


def command_status(args: argparse.Namespace) -> int:
    experiment_root = args.experiment_root.expanduser().resolve()
    status = collect_status(experiment_root)
    provenance.write_stage_status(experiment_root, status)
    print(json.dumps(status, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("--source-root", type=Path, required=True)
    init.add_argument("--experiment-root", type=Path, required=True)
    init.add_argument("--subjects", required=True)
    init.add_argument("--repeat-count", type=int, required=True)
    init.add_argument("--roi-preset", default="left-hippocampus")
    init.add_argument(
        "--montage-preset",
        required=True,
        help="Confirmed targets.csv montage preset used for all simulations.",
    )
    init.add_argument(
        "--targets-csv",
        type=Path,
        default=TARGETS_CSV_PATH,
        help=(
            "Path to the confirmed optimized targets.csv. Its SHA-256 must match "
            "the repository-approved hash."
        ),
    )
    init.add_argument("--atlas-dir", type=Path, required=True)
    init.add_argument(
        "--compare-metric",
        default="median_roi",
        choices=("median_roi", "mean_roi", "p95_roi", "peak_roi", "median_head", "mean_head", "p95_head", "peak_head"),
    )
    init.add_argument("--dry-run", action="store_true")
    init.set_defaults(func=command_init)

    for name, func in (
        ("submit-remesh", command_submit_remesh),
        ("submit-fixed", command_submit_fixed),
        ("analyze-remesh", command_analyze_remesh),
        ("analyze-paired", command_analyze_paired),
    ):
        stage = subparsers.add_parser(name)
        stage.add_argument("--experiment-root", type=Path, required=True)
        stage.add_argument("--max-concurrent", type=int, required=True)
        stage.set_defaults(func=func)

    select = subparsers.add_parser("select-medians")
    select.add_argument("--experiment-root", type=Path, required=True)
    select.add_argument("--metric", default="median_roi", choices=("median_roi", "mean_roi", "p95_roi", "peak_roi"))
    select.set_defaults(func=command_select_medians)

    seed = subparsers.add_parser("seed-fixed")
    seed.add_argument("--experiment-root", type=Path, required=True)
    seed.add_argument("--overwrite", action="store_true")
    seed.set_defaults(func=command_seed_fixed)

    figures = subparsers.add_parser("make-figures")
    figures.add_argument("--experiment-root", type=Path, required=True)
    figures.set_defaults(func=command_make_figures)

    submit_all = subparsers.add_parser(
        "submit-all",
        help="Submit the complete dependency-gated remesh-to-fixed workflow.",
    )
    submit_all.add_argument("--experiment-root", type=Path, required=True)
    submit_all.add_argument("--max-concurrent", type=_positive_int, default=50)
    submit_all.add_argument("--analysis-max-concurrent", type=_positive_int, default=10)
    submit_all.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the full scope without submitting jobs.",
    )
    submit_all.set_defaults(func=command_submit_all)

    advance = subparsers.add_parser(
        "advance-workflow",
        help="Internal dependency-controller entry point.",
    )
    advance.add_argument("--experiment-root", type=Path, required=True)
    advance.add_argument("--step", choices=WORKFLOW_STEPS, required=True)
    advance.add_argument("--max-concurrent", type=_positive_int, required=True)
    advance.add_argument("--analysis-max-concurrent", type=_positive_int, required=True)
    advance.set_defaults(func=command_advance_workflow)

    status = subparsers.add_parser("status")
    status.add_argument("--experiment-root", type=Path, required=True)
    status.set_defaults(func=command_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
