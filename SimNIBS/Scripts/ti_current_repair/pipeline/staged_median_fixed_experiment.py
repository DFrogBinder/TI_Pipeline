#!/usr/bin/env python3
"""Staged median-fixed repeatability experiment orchestration."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
PIPELINE_DIR = HERE.parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from experiment_config import repeat_tag  # noqa: E402
from pipeline import provenance  # noqa: E402
from post import make_presentation_figures, seed_fixed_from_median, select_median_remesh_repeats  # noqa: E402


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
        "--time=12:00:00",
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
        "--time=12:00:00",
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
    base = _base_config(
        source_root=source_root,
        experiment_root=experiment_root,
        subjects=subjects,
        repeat_count=args.repeat_count,
        roi_preset=args.roi_preset,
        atlas_dir=atlas_dir,
        compare_metric=args.compare_metric,
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
    )
    provenance.write_stage_status(experiment_root, collect_status(experiment_root))
    print(f"initialized staged pipeline: {_pipeline_root(experiment_root)}")
    return 0


def _submit_simulation_stage(args: argparse.Namespace, *, stage: str, config_name: str) -> int:
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
    return result.returncode


def _submit_report_stage(args: argparse.Namespace, *, stage: str, config_name: str, conditions: str) -> int:
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
    return result.returncode


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
    expected = _subject_count(config)
    observed = sum(
        1
        for subject in config["subjects"]
        if _exists_nonempty(root / "_analysis" / subject / condition_name / "summary.csv")
    )
    return expected, observed


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
    ok_rows = [row for row in rows if row.get("validation_result") == "ok"]
    checksum_mismatches = 0
    symlink_paths: list[str] = []
    for row in ok_rows:
        for key in ("cache_anat_dir",):
            root = Path(row[key])
            symlink_paths.extend(str(path) for path in provenance.find_symlinks(root))
        for raw in row.get("repeat_anat_dirs", "").split(";"):
            if raw:
                symlink_paths.extend(str(path) for path in provenance.find_symlinks(Path(raw)))
        checksum = row.get("mesh_checksum", "")
        cache_mesh = Path(row.get("cache_anat_dir", "")) / f"m2m_{row.get('subject', '')}" / f"{row.get('subject', '')}.msh"
        if checksum and cache_mesh.is_file() and provenance.file_sha256(cache_mesh) != checksum:
            checksum_mismatches += 1
    return {
        "path": str(path),
        "expected": _subject_count(config),
        "observed": len(ok_rows),
        "rows": len(rows),
        "symlink_count": len(symlink_paths),
        "symlinks": symlink_paths[:20],
        "checksum_mismatches": checksum_mismatches,
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

    status = subparsers.add_parser("status")
    status.add_argument("--experiment-root", type=Path, required=True)
    status.set_defaults(func=command_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
