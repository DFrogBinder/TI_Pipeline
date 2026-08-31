#!/usr/bin/env python3
"""Prepare and validate spherical-median fixed and nested repeatability runs.

The completed remesh simulations are immutable inputs.  This module provides
two isolated follow-up designs:

* select each participant's representative remesh geometry using the median
  field in the optimizer-matched parcel-clipped sphere, then rerun only the
  40-repeat fixed-mesh condition; and
* select one participant-target case once, persist that selection, and run a
  40-mesh by 40-within-mesh nested experiment.

The generated configurations are consumed by the established
``simulation_runners/repeatability_experiment.py`` runner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from experiment_config import (  # noqa: E402
    iter_experiment_tasks,
    load_experiment_config,
    subject_condition_mesh_cache_root,
    subject_condition_repeats_root,
)
from pipeline import provenance  # noqa: E402


SCHEMA_VERSION = 1
OPTIMIZER_METRIC_SCHEMA = 2
SPHERICAL_SELECTION_METRIC = "roi_median_v_per_m"
REMESH_CONDITION = "remesh"
FIXED_CONDITION = "fixed_mesh"
DEFAULT_REPEAT_COUNT = 40
DEFAULT_NESTED_SEED = 20260831
NESTED_CONDITION_RE = re.compile(r"mesh_(\d{3})$")
TARGET_SPECS = {
    "left-hippocampus": {
        "roi": "Left_Hippocampus",
        "requested_volume_mm3": 200.0,
    },
    "right-m1": {
        "roi": "Right_M1",
        "requested_volume_mm3": 100.0,
    },
}


@dataclass(frozen=True)
class SphericalMedianSelection:
    subject: str
    selection_status: str
    selected_repeat_tag: str
    metric: str
    metric_value: float
    median_target: float
    selected_abs_delta: float
    selected_m2m_dir: Path
    selected_mesh_path: Path
    mesh_checksum: str
    previous_selected_repeat_tag: str
    selection_changed: bool | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)
    return path


def _write_json_once(path: Path, payload: object) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
        return dict(payload)  # type: ignore[arg-type]
    except FileExistsError:
        return _load_json(path)


def _write_csv_atomic(path: Path, rows: list[dict[str, object]], fields: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)
    return path


def _write_or_validate_json(path: Path, payload: dict[str, Any]) -> Path:
    if path.is_file():
        observed = _load_json(path)
        if observed != payload:
            raise RuntimeError(
                f"Refusing to replace an incompatible initialized artifact: {path}"
            )
        return path
    return _write_json_atomic(path, payload)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _finite_float(value: object, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected finite {label}; got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Expected finite {label}; got {value!r}")
    return parsed


def _repeat_index(tag: str) -> int:
    match = re.fullmatch(r"repeat_(\d{3})", tag)
    if not match:
        raise ValueError(f"Invalid repeat tag: {tag!r}")
    return int(match.group(1))


def _repeat_tag(index: int) -> str:
    return f"repeat_{index:03d}"


def _median(values: Iterable[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a median from no values")
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return 0.5 * (ordered[midpoint - 1] + ordered[midpoint])


def _paired_config_path(source_experiment_root: Path) -> Path:
    return source_experiment_root / "_pipeline" / "configs" / "paired_analysis.json"


def _source_remesh_anat(
    source_experiment_root: Path,
    subject: str,
    repeat_tag: str,
) -> Path:
    return (
        source_experiment_root
        / f"{subject}_repeatability"
        / REMESH_CONDITION
        / "repeats"
        / repeat_tag
        / subject
        / "anat"
    )


def _previous_selection_by_subject(source_experiment_root: Path) -> dict[str, str]:
    candidates = [
        source_experiment_root / "_pipeline" / "fixed_seed_manifest.csv",
        source_experiment_root
        / "_pipeline"
        / "median_mesh_selection"
        / "median_representative_remesh_repeats.csv",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        rows = _read_csv(path)
        return {
            row.get("subject", "").strip(): row.get("selected_repeat_tag", "").strip()
            for row in rows
            if row.get("subject", "").strip()
        }
    return {}


def _validate_optimizer_rows(
    *,
    metrics_csv: Path,
    source_config: dict[str, Any],
    repeat_count: int,
) -> dict[str, list[dict[str, str]]]:
    rows = _read_csv(metrics_csv)
    analysis = source_config.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("Source config is missing analysis metadata")
    target = str(analysis.get("roi_preset", ""))
    try:
        target_spec = TARGET_SPECS[target]
    except KeyError as exc:
        raise ValueError(f"Unsupported spherical target {target!r}") from exc
    subjects = [str(value) for value in source_config.get("subjects", [])]
    if not subjects:
        raise ValueError("Source config has no subjects")

    required = {
        "schema_version",
        "subject",
        "condition",
        "repeat_tag",
        "roi",
        "requested_roi_volume_mm3",
        SPHERICAL_SELECTION_METRIC,
    }
    if not rows:
        raise ValueError(f"Optimizer metric CSV is empty: {metrics_csv}")
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(f"{metrics_csv} is missing columns: {', '.join(missing)}")

    grouped: dict[str, list[dict[str, str]]] = {subject: [] for subject in subjects}
    for row in rows:
        if row.get("condition") != REMESH_CONDITION:
            continue
        subject = row.get("subject", "")
        if subject not in grouped:
            continue
        if int(row["schema_version"]) != OPTIMIZER_METRIC_SCHEMA:
            raise ValueError("Spherical selection requires optimizer metric schema 2")
        if row.get("roi") != target_spec["roi"]:
            raise ValueError(
                f"Unexpected ROI for {subject}: {row.get('roi')} != {target_spec['roi']}"
            )
        volume = _finite_float(
            row.get("requested_roi_volume_mm3"),
            label="requested spherical ROI volume",
        )
        if not math.isclose(
            volume,
            float(target_spec["requested_volume_mm3"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"Unexpected spherical ROI volume for {subject}: {volume}")
        _finite_float(row.get(SPHERICAL_SELECTION_METRIC), label=SPHERICAL_SELECTION_METRIC)
        _repeat_index(row.get("repeat_tag", ""))
        grouped[subject].append(row)

    expected_tags = {_repeat_tag(index) for index in range(1, repeat_count + 1)}
    for subject, subject_rows in grouped.items():
        tags = [row["repeat_tag"] for row in subject_rows]
        if len(tags) != repeat_count or set(tags) != expected_tags or len(tags) != len(set(tags)):
            raise ValueError(
                f"{subject}: expected {repeat_count} unique remesh spherical metric rows"
            )
    return grouped


def select_spherical_medians(
    *,
    source_experiment_root: Path,
    metrics_csv: Path,
    repeat_count: int = DEFAULT_REPEAT_COUNT,
    require_meshes: bool = True,
) -> tuple[dict[str, Any], list[SphericalMedianSelection]]:
    config_path = _paired_config_path(source_experiment_root)
    source_config = _load_json(config_path)
    grouped = _validate_optimizer_rows(
        metrics_csv=metrics_csv,
        source_config=source_config,
        repeat_count=repeat_count,
    )
    previous = _previous_selection_by_subject(source_experiment_root)
    selections: list[SphericalMedianSelection] = []
    for subject in [str(value) for value in source_config["subjects"]]:
        rows = grouped[subject]
        values = [
            _finite_float(row[SPHERICAL_SELECTION_METRIC], label=SPHERICAL_SELECTION_METRIC)
            for row in rows
        ]
        target = _median(values)
        selected = min(
            rows,
            key=lambda row: (
                abs(
                    _finite_float(
                        row[SPHERICAL_SELECTION_METRIC],
                        label=SPHERICAL_SELECTION_METRIC,
                    )
                    - target
                ),
                _repeat_index(row["repeat_tag"]),
            ),
        )
        repeat_tag = selected["repeat_tag"]
        selected_value = _finite_float(
            selected[SPHERICAL_SELECTION_METRIC],
            label=SPHERICAL_SELECTION_METRIC,
        )
        anat = _source_remesh_anat(source_experiment_root, subject, repeat_tag)
        m2m = anat / f"m2m_{subject}"
        mesh = m2m / f"{subject}.msh"
        status = "selected" if mesh.is_file() else "missing_mesh"
        if require_meshes and status != "selected":
            raise FileNotFoundError(mesh)
        previous_tag = previous.get(subject, "")
        selections.append(
            SphericalMedianSelection(
                subject=subject,
                selection_status=status,
                selected_repeat_tag=repeat_tag,
                metric=SPHERICAL_SELECTION_METRIC,
                metric_value=selected_value,
                median_target=target,
                selected_abs_delta=abs(selected_value - target),
                selected_m2m_dir=m2m,
                selected_mesh_path=mesh,
                mesh_checksum=provenance.file_sha256(mesh) if mesh.is_file() else "",
                previous_selected_repeat_tag=previous_tag,
                selection_changed=(repeat_tag != previous_tag) if previous_tag else None,
            )
        )
    return source_config, selections


SELECTION_FIELDS = [
    "subject",
    "selection_status",
    "selected_repeat_tag",
    "metric",
    "metric_value",
    "median_target",
    "selected_abs_delta",
    "selected_m2m_dir",
    "selected_mesh_path",
    "mesh_checksum",
    "previous_selected_repeat_tag",
    "selection_changed",
]


def _selection_rows(selections: list[SphericalMedianSelection]) -> list[dict[str, object]]:
    rows = []
    for selection in selections:
        row = asdict(selection)
        row["selected_m2m_dir"] = str(selection.selected_m2m_dir)
        row["selected_mesh_path"] = str(selection.selected_mesh_path)
        if selection.selection_changed is None:
            row["selection_changed"] = ""
        rows.append(row)
    return rows


def _fixed_config(
    *,
    source_config: dict[str, Any],
    output_root: Path,
    repeat_count: int,
) -> dict[str, Any]:
    return {
        "source_root": source_config["source_root"],
        "experiment_root": str(output_root),
        "subjects": source_config["subjects"],
        "conditions": [
            {
                "name": FIXED_CONDITION,
                "mesh_mode": FIXED_CONDITION,
                "repeat_count": repeat_count,
                "description": (
                    "Reuse the remesh geometry nearest the participant-specific "
                    "median roi_median_v_per_m in the optimizer-matched spherical ROI."
                ),
            }
        ],
        "stimulation": source_config["stimulation"],
        "analysis": source_config["analysis"],
    }


def _nested_config(
    *,
    source_config: dict[str, Any],
    output_root: Path,
    subject: str,
    outer_repeat_count: int,
    inner_repeat_count: int,
) -> dict[str, Any]:
    conditions = []
    for outer_index in range(1, outer_repeat_count + 1):
        tag = _repeat_tag(outer_index)
        conditions.append(
            {
                "name": f"mesh_{outer_index:03d}",
                "mesh_mode": FIXED_CONDITION,
                "repeat_count": inner_repeat_count,
                "description": f"Forty repeated solves using source remesh geometry {tag}.",
            }
        )
    return {
        "source_root": source_config["source_root"],
        "experiment_root": str(output_root),
        "subjects": [subject],
        "conditions": conditions,
        "stimulation": source_config["stimulation"],
        "analysis": source_config["analysis"],
    }


def _mesh_ready_payload(
    *,
    subject: str,
    source_anat: Path,
    source_mesh: Path,
    mesh_checksum: str,
    source_repeat_tag: str,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "mesh_ready",
        "subject": subject,
        "source_anat_dir": str(source_anat),
        "source_mesh_path": str(source_mesh),
        "mesh_checksum": mesh_checksum,
        "source_repeat_tag": source_repeat_tag,
        "seed_mode": "physical_cache_copy",
        "created_utc": _utc_now(),
    }


def _validate_seeded_cache(
    *,
    cache_anat: Path,
    subject: str,
    expected_mesh_checksum: str,
    source_mesh: Path,
) -> None:
    cache_mesh = cache_anat / f"m2m_{subject}" / f"{subject}.msh"
    ready = cache_anat / ".mesh_ready.json"
    if not cache_mesh.is_file() or not ready.is_file():
        raise RuntimeError(f"Incomplete seeded cache: {cache_anat}")
    observed_checksum = provenance.file_sha256(cache_mesh)
    if observed_checksum != expected_mesh_checksum:
        raise RuntimeError(f"Seeded cache checksum mismatch: {cache_mesh}")
    marker = _load_json(ready)
    if marker.get("mesh_checksum") != expected_mesh_checksum:
        raise RuntimeError(f"Seeded cache ready-marker checksum mismatch: {ready}")
    if marker.get("source_mesh_path") != str(source_mesh):
        raise RuntimeError(f"Seeded cache points at a different source mesh: {ready}")


def _seed_cache(
    *,
    cache_anat: Path,
    source_anat: Path,
    subject: str,
    source_repeat_tag: str,
    expected_mesh_checksum: str,
) -> dict[str, object]:
    source_m2m = source_anat / f"m2m_{subject}"
    source_mesh = source_m2m / f"{subject}.msh"
    if not source_mesh.is_file():
        raise FileNotFoundError(source_mesh)
    observed_checksum = provenance.file_sha256(source_mesh)
    if observed_checksum != expected_mesh_checksum:
        raise RuntimeError(f"Source mesh checksum changed: {source_mesh}")

    if cache_anat.exists() or cache_anat.is_symlink():
        _validate_seeded_cache(
            cache_anat=cache_anat,
            subject=subject,
            expected_mesh_checksum=expected_mesh_checksum,
            source_mesh=source_mesh,
        )
        return {
            "seed_status": "ready",
            "subject": subject,
            "source_repeat_tag": source_repeat_tag,
            "source_anat_dir": str(source_anat),
            "source_mesh_path": str(source_mesh),
            "mesh_checksum": expected_mesh_checksum,
            "cache_anat_dir": str(cache_anat),
        }

    cache_anat.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{cache_anat.name}.", dir=cache_anat.parent)
    )
    try:
        for suffix in ("T1w.nii", "T2w.nii", "T1w_ras_1mm_T1andT2_masks.nii"):
            name = f"{subject}_{suffix}"
            source = source_anat / name
            if not source.exists():
                raise FileNotFoundError(source)
            shutil.copy2(source, temporary / name, follow_symlinks=True)
        shutil.copytree(
            source_m2m,
            temporary / f"m2m_{subject}",
            symlinks=False,
        )
        _write_json_atomic(
            temporary / ".mesh_ready.json",
            _mesh_ready_payload(
                subject=subject,
                source_anat=source_anat,
                source_mesh=source_mesh,
                mesh_checksum=expected_mesh_checksum,
                source_repeat_tag=source_repeat_tag,
            ),
        )
        temporary.rename(cache_anat)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    _validate_seeded_cache(
        cache_anat=cache_anat,
        subject=subject,
        expected_mesh_checksum=expected_mesh_checksum,
        source_mesh=source_mesh,
    )
    return {
        "seed_status": "ready",
        "subject": subject,
        "source_repeat_tag": source_repeat_tag,
        "source_anat_dir": str(source_anat),
        "source_mesh_path": str(source_mesh),
        "mesh_checksum": expected_mesh_checksum,
        "cache_anat_dir": str(cache_anat),
    }


CACHE_SEED_FIELDS = [
    "condition",
    "seed_status",
    "subject",
    "source_repeat_tag",
    "source_anat_dir",
    "source_mesh_path",
    "mesh_checksum",
    "cache_anat_dir",
]


def prepare_fixed(
    *,
    source_experiment_root: Path,
    metrics_csv: Path,
    output_root: Path,
    repeat_count: int = DEFAULT_REPEAT_COUNT,
    dry_run: bool = False,
) -> dict[str, Any]:
    source_config, selections = select_spherical_medians(
        source_experiment_root=source_experiment_root,
        metrics_csv=metrics_csv,
        repeat_count=repeat_count,
    )
    analysis = source_config.get("analysis", {})
    target = str(analysis.get("roi_preset", ""))
    changed = sum(selection.selection_changed is True for selection in selections)
    payload: dict[str, Any] = {
        "status": "ready" if dry_run else "prepared",
        "study_type": "spherical_median_fixed_mesh_correction",
        "target": target,
        "subjects": len(selections),
        "conditions": 1,
        "repeats_per_subject": repeat_count,
        "simulation_tasks": len(selections) * repeat_count,
        "expected_ti_outputs": len(selections) * repeat_count,
        "selection_metric": SPHERICAL_SELECTION_METRIC,
        "previous_selections_available": sum(
            bool(selection.previous_selected_repeat_tag) for selection in selections
        ),
        "changed_selections": changed,
        "source_experiment_root": str(source_experiment_root),
        "optimizer_metrics_csv": str(metrics_csv),
        "optimizer_metrics_sha256": provenance.file_sha256(metrics_csv),
        "output_root": str(output_root),
        "selections": _selection_rows(selections),
    }
    if dry_run:
        return payload

    pipeline_root = output_root / "_pipeline"
    config = _fixed_config(
        source_config=source_config,
        output_root=output_root,
        repeat_count=repeat_count,
    )
    config_path = pipeline_root / "configs" / "fixed_mesh_spherical_median.json"
    _write_or_validate_json(config_path, config)
    selection_path = pipeline_root / "spherical_median_selection.csv"
    selection_rows = _selection_rows(selections)
    if selection_path.is_file():
        if _read_csv(selection_path) != [
            {field: str(row.get(field, "")) for field in SELECTION_FIELDS}
            for row in selection_rows
        ]:
            raise RuntimeError(f"Existing spherical selection differs: {selection_path}")
    else:
        _write_csv_atomic(selection_path, selection_rows, SELECTION_FIELDS)

    seed_rows = []
    loaded_config = load_experiment_config(config_path, validate_paths=False)
    for selection in selections:
        cache_anat = subject_condition_mesh_cache_root(
            loaded_config,
            selection.subject,
            FIXED_CONDITION,
        )
        row = _seed_cache(
            cache_anat=cache_anat,
            source_anat=selection.selected_m2m_dir.parent,
            subject=selection.subject,
            source_repeat_tag=selection.selected_repeat_tag,
            expected_mesh_checksum=selection.mesh_checksum,
        )
        seed_rows.append({"condition": FIXED_CONDITION, **row})
    seed_path = pipeline_root / "cache_seed_manifest.csv"
    _write_csv_atomic(seed_path, seed_rows, CACHE_SEED_FIELDS)

    manifest = {
        **{key: value for key, value in payload.items() if key != "selections"},
        "schema_version": SCHEMA_VERSION,
        "config": str(config_path),
        "config_sha256": provenance.file_sha256(config_path),
        "selection_csv": str(selection_path),
        "selection_csv_sha256": provenance.file_sha256(selection_path),
        "cache_seed_manifest": str(seed_path),
        "cache_seed_manifest_sha256": provenance.file_sha256(seed_path),
        "source_outputs_modified": False,
    }
    _write_or_validate_json(pipeline_root / "experiment_manifest.json", manifest)
    payload.update(
        {
            "config": str(config_path),
            "selection_csv": str(selection_path),
            "cache_seed_manifest": str(seed_path),
        }
    )
    return payload


def _eligible_nested_cases(
    *,
    target_sources: dict[str, tuple[Path, Path]],
    repeat_count: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    configs: dict[str, dict[str, Any]] = {}
    cases: list[dict[str, str]] = []
    subject_sets: list[set[str]] = []
    for target in sorted(target_sources):
        source_root, metrics_csv = target_sources[target]
        config = _load_json(_paired_config_path(source_root))
        if str(config.get("analysis", {}).get("roi_preset", "")) != target:
            raise ValueError(f"Target/config mismatch for {source_root}")
        grouped = _validate_optimizer_rows(
            metrics_csv=metrics_csv,
            source_config=config,
            repeat_count=repeat_count,
        )
        subjects = set(grouped)
        subject_sets.append(subjects)
        configs[target] = config
    common_subjects = sorted(set.intersection(*subject_sets))
    if not common_subjects:
        raise ValueError("The target datasets have no common eligible subjects")

    for subject in common_subjects:
        for target in sorted(target_sources):
            source_root, _ = target_sources[target]
            source_config = configs[target]
            if subject not in source_config["subjects"]:
                continue
            mesh_paths = [
                _source_remesh_anat(source_root, subject, _repeat_tag(index))
                / f"m2m_{subject}"
                / f"{subject}.msh"
                for index in range(1, repeat_count + 1)
            ]
            missing = [str(path) for path in mesh_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"{target}/{subject} lacks {len(missing)} remesh meshes; first: {missing[0]}"
                )
            cases.append(
                {
                    "target": target,
                    "subject": subject,
                    "source_experiment_root": str(source_root),
                }
            )
    return configs, cases


def _case_pool_hash(cases: list[dict[str, str]]) -> str:
    return _sha256_bytes(_canonical_json_bytes(cases))


def _choose_nested_case(
    *,
    cases: list[dict[str, str]],
    seed: int,
    target: str,
) -> tuple[dict[str, str], dict[str, int]]:
    target_cases = [case for case in cases if case["target"] == target]
    if not target_cases:
        raise ValueError(f"No eligible nested cases for target {target!r}")
    subjects = sorted({case["subject"] for case in target_cases})
    rng = random.Random(seed)
    subject_index = rng.randrange(len(subjects))
    subject = subjects[subject_index]
    selected = next(
        case
        for case in target_cases
        if case["subject"] == subject and case["target"] == target
    )
    return selected, {"subject_index": subject_index}


def prepare_nested(
    *,
    target_sources: dict[str, tuple[Path, Path]],
    output_root: Path,
    selection_seed: int = DEFAULT_NESTED_SEED,
    nested_target: str = "left-hippocampus",
    outer_repeat_count: int = DEFAULT_REPEAT_COUNT,
    inner_repeat_count: int = DEFAULT_REPEAT_COUNT,
    dry_run: bool = False,
) -> dict[str, Any]:
    configs, cases = _eligible_nested_cases(
        target_sources=target_sources,
        repeat_count=outer_repeat_count,
    )
    if nested_target not in target_sources:
        raise ValueError(f"Unsupported nested target {nested_target!r}")
    eligible_cases = [case for case in cases if case["target"] == nested_target]
    pool_hash = _case_pool_hash(eligible_cases)
    selected, indices = _choose_nested_case(
        cases=eligible_cases,
        seed=selection_seed,
        target=nested_target,
    )
    selection_path = output_root / "_pipeline" / "nested_case_selection.json"
    persisted = False
    if selection_path.is_file():
        stored = _load_json(selection_path)
        if stored.get("candidate_pool_sha256") != pool_hash:
            raise RuntimeError(
                "The persisted nested selection candidate pool changed; refusing to reselect."
            )
        selected = {
            "target": str(stored["selected_target"]),
            "subject": str(stored["selected_subject"]),
            "source_experiment_root": str(stored["source_experiment_root"]),
        }
        if selected not in eligible_cases:
            raise RuntimeError("The persisted nested case is no longer eligible")
        persisted = True

    selection_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "selected",
        "selection_unit": "participant within a prespecified target",
        "selection_algorithm": "uniform subject; Python MT19937",
        "selection_seed": selection_seed,
        "candidate_subjects": sorted({case["subject"] for case in eligible_cases}),
        "prespecified_target": nested_target,
        "candidate_case_count": len(eligible_cases),
        "candidate_pool_sha256": pool_hash,
        "selected_subject": selected["subject"],
        "selected_target": selected["target"],
        "source_experiment_root": selected["source_experiment_root"],
        "selected_indices": indices,
        "created_utc": _utc_now(),
        "selection_policy": (
            "This file is immutable experiment state. Relaunches reuse this case and "
            "must never draw another subject or target."
        ),
    }
    if persisted:
        selection_payload = stored
    if dry_run:
        return {
            "status": "ready",
            "study_type": "nested_mesh_by_solver_repeatability",
            "selection_was_already_persisted": persisted,
            **selection_payload,
            "outer_meshes": outer_repeat_count,
            "inner_repeats_per_mesh": inner_repeat_count,
            "simulation_tasks": outer_repeat_count * inner_repeat_count,
            "expected_ti_outputs": outer_repeat_count * inner_repeat_count,
            "output_root": str(output_root),
        }

    if not persisted:
        observed = _write_json_once(selection_path, selection_payload)
        if (
            observed.get("selected_subject") != selected["subject"]
            or observed.get("selected_target") != selected["target"]
        ):
            raise RuntimeError("A concurrent preparation persisted a different nested case")
    else:
        selection_payload = _load_json(selection_path)

    target = str(selection_payload["selected_target"])
    subject = str(selection_payload["selected_subject"])
    source_root = Path(str(selection_payload["source_experiment_root"]))
    source_config = configs[target]
    config = _nested_config(
        source_config=source_config,
        output_root=output_root,
        subject=subject,
        outer_repeat_count=outer_repeat_count,
        inner_repeat_count=inner_repeat_count,
    )
    config_path = output_root / "_pipeline" / "configs" / "nested_40x40.json"
    _write_or_validate_json(config_path, config)
    loaded_config = load_experiment_config(config_path, validate_paths=False)

    seed_rows: list[dict[str, object]] = []
    for outer_index in range(1, outer_repeat_count + 1):
        source_repeat_tag = _repeat_tag(outer_index)
        source_anat = _source_remesh_anat(source_root, subject, source_repeat_tag)
        source_mesh = source_anat / f"m2m_{subject}" / f"{subject}.msh"
        checksum = provenance.file_sha256(source_mesh)
        condition = f"mesh_{outer_index:03d}"
        cache_anat = subject_condition_mesh_cache_root(
            loaded_config,
            subject,
            condition,
        )
        row = _seed_cache(
            cache_anat=cache_anat,
            source_anat=source_anat,
            subject=subject,
            source_repeat_tag=source_repeat_tag,
            expected_mesh_checksum=checksum,
        )
        seed_rows.append({"condition": condition, **row})
    seed_path = output_root / "_pipeline" / "cache_seed_manifest.csv"
    _write_csv_atomic(seed_path, seed_rows, CACHE_SEED_FIELDS)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared",
        "study_type": "nested_mesh_by_solver_repeatability",
        "selected_subject": subject,
        "selected_target": target,
        "source_experiment_root": str(source_root),
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": provenance.file_sha256(selection_path),
        "config": str(config_path),
        "config_sha256": provenance.file_sha256(config_path),
        "cache_seed_manifest": str(seed_path),
        "cache_seed_manifest_sha256": provenance.file_sha256(seed_path),
        "outer_meshes": outer_repeat_count,
        "inner_repeats_per_mesh": inner_repeat_count,
        "simulation_tasks": outer_repeat_count * inner_repeat_count,
        "expected_ti_outputs": outer_repeat_count * inner_repeat_count,
        "source_outputs_modified": False,
    }
    _write_or_validate_json(output_root / "_pipeline" / "experiment_manifest.json", manifest)
    return {
        **manifest,
        "selection_was_already_persisted": persisted,
    }


def _generated_volume_exists(parent: Path) -> bool:
    return parent.is_dir() and any(parent.glob("TI_Volumetric_*"))


def collect_simulation_status(config_path: Path) -> dict[str, Any]:
    config = load_experiment_config(config_path, validate_paths=False)
    tasks = iter_experiment_tasks(config)
    seed_path = config.experiment_root / "_pipeline" / "cache_seed_manifest.csv"
    if not seed_path.is_file():
        raise FileNotFoundError(seed_path)
    seed_rows = _read_csv(seed_path)
    seed_by_key = {
        (row["subject"], row["condition"]): row
        for row in seed_rows
    }
    cache_issues: list[str] = []
    for condition in config.conditions:
        for subject in config.subjects:
            row = seed_by_key.get((subject, condition.name))
            if row is None:
                cache_issues.append(f"missing seed row {subject}/{condition.name}")
                continue
            cache_anat = subject_condition_mesh_cache_root(config, subject, condition.name)
            try:
                _validate_seeded_cache(
                    cache_anat=cache_anat,
                    subject=subject,
                    expected_mesh_checksum=row["mesh_checksum"],
                    source_mesh=Path(row["source_mesh_path"]),
                )
            except Exception as exc:
                cache_issues.append(f"{subject}/{condition.name}: {exc}")

    complete = 0
    incomplete: list[str] = []
    for task in tasks:
        repeat_root = (
            subject_condition_repeats_root(
                config,
                task.subject,
                task.condition_name,
            )
            / task.repeat_tag
        )
        anat = repeat_root / task.subject / "anat"
        output = anat / "SimNIBS" / "Output" / task.subject
        required = [
            anat / f"m2m_{task.subject}" / f"{task.subject}.msh",
            anat / "SimNIBS" / "ti_brain_only.nii.gz",
            output / "TI.msh",
            repeat_root / task.subject / "task_manifest.json",
        ]
        valid = all(path.is_file() for path in required)
        valid = valid and _generated_volume_exists(output / "Volume_Labels")
        valid = valid and _generated_volume_exists(output / "Volume_Base")
        if valid:
            try:
                manifest = _load_json(required[-1])
                valid = manifest.get("stimulation") == config.stimulation.to_dict()
            except Exception:
                valid = False
        if valid:
            complete += 1
        elif len(incomplete) < 50:
            incomplete.append(
                f"{task.subject}/{task.condition_name}/{task.repeat_tag}"
            )

    counts_by_condition = {
        condition.name: condition.repeat_count * len(config.subjects)
        for condition in config.conditions
    }
    return {
        "status": "complete" if complete == len(tasks) and not cache_issues else "incomplete",
        "config": str(config_path),
        "experiment_root": str(config.experiment_root),
        "subject_count": len(config.subjects),
        "condition_count": len(config.conditions),
        "conditions": counts_by_condition,
        "repeats_per_condition": (
            config.conditions[0].repeat_count if config.conditions else 0
        ),
        "expected_ti_nifti": len(tasks),
        "observed_complete_tasks": complete,
        "incomplete_task_count": len(tasks) - complete,
        "incomplete_task_examples": incomplete,
        "cache_seed_rows": len(seed_rows),
        "cache_issue_count": len(cache_issues),
        "cache_issue_examples": cache_issues[:50],
        "roi": config.analysis.roi_preset,
    }


def finalize(config_path: Path) -> dict[str, Any]:
    status = collect_simulation_status(config_path)
    if status["status"] != "complete":
        raise RuntimeError(
            "Simulation scope is incomplete: "
            f"{status['observed_complete_tasks']}/{status['expected_ti_nifti']} tasks; "
            f"cache issues={status['cache_issue_count']}"
        )
    config = load_experiment_config(config_path, validate_paths=False)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "completed_utc": _utc_now(),
        "scope": status,
        "source_outputs_modified": False,
    }
    completion = config.experiment_root / "_pipeline" / "workflow" / "complete.json"
    if completion.is_file():
        observed = _load_json(completion)
        if observed.get("status") != "complete" or observed.get("scope") != status:
            raise RuntimeError(f"Existing completion receipt differs: {completion}")
        return observed
    _write_json_atomic(completion, receipt)
    return receipt


def _target_sources_from_args(args: argparse.Namespace) -> dict[str, tuple[Path, Path]]:
    return {
        "left-hippocampus": (
            args.left_source_root.expanduser().resolve(),
            args.left_metrics_csv.expanduser().resolve(),
        ),
        "right-m1": (
            args.right_source_root.expanduser().resolve(),
            args.right_metrics_csv.expanduser().resolve(),
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("preflight-fixed", "prepare-fixed"):
        child = subparsers.add_parser(command)
        child.add_argument("--source-experiment-root", type=Path, required=True)
        child.add_argument("--optimizer-metrics-csv", type=Path, required=True)
        child.add_argument("--output-root", type=Path, required=True)
        child.add_argument("--repeat-count", type=int, default=DEFAULT_REPEAT_COUNT)

    for command in ("preflight-nested", "prepare-nested"):
        child = subparsers.add_parser(command)
        child.add_argument("--left-source-root", type=Path, required=True)
        child.add_argument("--right-source-root", type=Path, required=True)
        child.add_argument("--left-metrics-csv", type=Path, required=True)
        child.add_argument("--right-metrics-csv", type=Path, required=True)
        child.add_argument("--output-root", type=Path, required=True)
        child.add_argument("--selection-seed", type=int, default=DEFAULT_NESTED_SEED)
        child.add_argument(
            "--nested-target",
            choices=sorted(TARGET_SPECS),
            default="left-hippocampus",
        )
        child.add_argument("--outer-repeat-count", type=int, default=DEFAULT_REPEAT_COUNT)
        child.add_argument("--inner-repeat-count", type=int, default=DEFAULT_REPEAT_COUNT)

    for command in ("status", "finalize"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"preflight-fixed", "prepare-fixed"}:
        payload = prepare_fixed(
            source_experiment_root=args.source_experiment_root.expanduser().resolve(),
            metrics_csv=args.optimizer_metrics_csv.expanduser().resolve(),
            output_root=args.output_root.expanduser().resolve(),
            repeat_count=args.repeat_count,
            dry_run=args.command == "preflight-fixed",
        )
    elif args.command in {"preflight-nested", "prepare-nested"}:
        payload = prepare_nested(
            target_sources=_target_sources_from_args(args),
            output_root=args.output_root.expanduser().resolve(),
            selection_seed=args.selection_seed,
            nested_target=args.nested_target,
            outer_repeat_count=args.outer_repeat_count,
            inner_repeat_count=args.inner_repeat_count,
            dry_run=args.command == "preflight-nested",
        )
    elif args.command == "status":
        payload = collect_simulation_status(args.config.expanduser().resolve())
    else:
        payload = finalize(args.config.expanduser().resolve())
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
