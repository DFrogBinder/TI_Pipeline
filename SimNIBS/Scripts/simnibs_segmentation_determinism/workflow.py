from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

from .charm_runner import (
    CharmRepeatRequest,
    RepeatRunError as CharmRunError,
    run_charm_repeat,
    validate_charm_repeat,
)
from .headreco_runner import (
    HeadrecoRepeatRequest,
    RepeatRunError as HeadrecoRunError,
    run_headreco_repeat,
    validate_headreco_repeat,
)
from .provenance import (
    ProvenanceError,
    canonical_json_bytes,
    sha256_file,
    validate_sha256_sidecar,
    write_json_atomic,
    write_sha256_sidecar,
)
from .settings import (
    ExperimentConfig,
    preflight,
    resolve_subject_inputs,
    scope_payload,
    validate_output_isolation,
)


class WorkflowError(RuntimeError):
    """Raised when a workflow state transition is unsafe or incomplete."""


ARMS = ("simnibs326_headreco", "simnibs401_charm")
PREFLIGHT_FILES = (
    "simnibs326.json",
    "simnibs326_cat12_probe.json",
    "simnibs401.json",
)


def workflow_preflight(config: ExperimentConfig) -> Dict[str, object]:
    return preflight(config)


def _config_payload(config: ExperimentConfig) -> Dict[str, object]:
    payload = asdict(config)
    for key in (
        "cohort_file",
        "selection_audit_file",
        "source_root",
        "output_root",
    ):
        payload[key] = str(payload[key])
    payload["protected_roots"] = [str(path) for path in config.protected_roots]
    return payload


def _plan_payload(config: ExperimentConfig, ready: Mapping[str, object]) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "workflow": "simnibs_segmentation_determinism",
        "config": _config_payload(config),
        "scope": scope_payload(config),
        "source": {
            "subject": config.subject,
            "cohort": ready["cohort"],
            "selection_audit": ready["selection_audit"],
            "inputs": ready["inputs"],
        },
        "stage_contract": {
            "simnibs326_headreco": "preparevols --noclean",
            "simnibs401_charm": list(CharmRepeatRequest(
                config=config,
                inputs=resolve_subject_inputs(config),
                repeat_index=1,
            ).flags),
            "surface_tasks": 0,
            "meshing_tasks": 0,
            "fem_simulation_tasks": 0,
        },
    }


def _plan_hash(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _read_json(path: Path) -> Dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"Invalid workflow JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise WorkflowError(f"Workflow JSON is not an object: {path}")
    return payload


def initialize(config: ExperimentConfig) -> Dict[str, object]:
    ready = workflow_preflight(config)
    plan = _plan_payload(config, ready)
    plan_sha256 = _plan_hash(plan)
    pipeline = config.output_root / "_pipeline"
    config_path = pipeline / "config.json"
    if config.output_root.exists():
        if not config_path.is_file():
            raise WorkflowError(
                f"Output root already exists without this workflow's config: {config.output_root}"
            )
        existing = _read_json(config_path)
        if existing.get("plan_sha256") != plan_sha256:
            raise WorkflowError(
                "Existing workflow plan differs from the requested plan: "
                f"{existing.get('plan_sha256')} != {plan_sha256}"
            )
        return {
            "status": "prepared",
            "action": "reused",
            "output_root": str(config.output_root),
            "plan_sha256": plan_sha256,
        }

    config.output_root.mkdir(parents=True, exist_ok=False)
    pipeline.mkdir()
    stored_config = {**plan, "status": "prepared", "plan_sha256": plan_sha256}
    write_json_atomic(config_path, stored_config)
    write_sha256_sidecar(config_path)
    source_manifest = {
        "schema_version": 1,
        "status": "pinned",
        "subject": config.subject,
        "selection_seed": config.selection_seed,
        "cohort": ready["cohort"],
        "selection_audit": ready["selection_audit"],
        "inputs": ready["inputs"],
        "source_outputs_modified": False,
    }
    source_path = pipeline / "source_manifest.json"
    write_json_atomic(source_path, source_manifest)
    write_sha256_sidecar(source_path)
    preflight_path = pipeline / "preflight.json"
    write_json_atomic(preflight_path, ready)
    write_sha256_sidecar(preflight_path)
    return {
        "status": "prepared",
        "action": "created",
        "output_root": str(config.output_root),
        "plan_sha256": plan_sha256,
    }


def _ensure_initialized(config: ExperimentConfig) -> Dict[str, object]:
    config_path = config.output_root / "_pipeline/config.json"
    if not config_path.is_file():
        raise WorkflowError("Workflow is not initialized")
    stored = _read_json(config_path)
    ready = workflow_preflight(config)
    expected_hash = _plan_hash(_plan_payload(config, ready))
    if stored.get("plan_sha256") != expected_hash:
        raise WorkflowError("Initialized plan does not match the requested configuration")
    return stored


def _headreco_request(config: ExperimentConfig, repeat_index: int, attempt: int):
    scripts_root = Path(__file__).resolve().parents[1]
    return HeadrecoRepeatRequest(
        config=config,
        inputs=resolve_subject_inputs(config),
        repeat_index=repeat_index,
        attempt=attempt,
        compatibility_script=scripts_root / "simnibs_326_repeatability/cat12_compat.py",
        python_executable=Path(sys.executable),
    )


def _charm_request(config: ExperimentConfig, repeat_index: int, attempt: int):
    return CharmRepeatRequest(
        config=config,
        inputs=resolve_subject_inputs(config),
        repeat_index=repeat_index,
        attempt=attempt,
        charm_bin=os.environ.get("CHARM_BIN", "charm"),
    )


def run_repeat(
    config: ExperimentConfig, *, arm: str, array_index: int, attempt: int = 1
) -> Dict[str, object]:
    _ensure_initialized(config)
    if arm not in ARMS:
        raise WorkflowError(f"Unknown arm: {arm}")
    if not 0 <= array_index < config.repeats_per_arm:
        raise WorkflowError(
            f"Array index {array_index} is outside 0-{config.repeats_per_arm - 1}"
        )
    repeat_index = array_index + 1
    if arm == "simnibs326_headreco":
        return run_headreco_repeat(_headreco_request(config, repeat_index, attempt))
    return run_charm_repeat(_charm_request(config, repeat_index, attempt))


def _validate_receipts(config: ExperimentConfig):
    receipts: Dict[Tuple[str, int], Dict[str, object]] = {}
    failures = []
    for arm in ARMS:
        for repeat_index in range(1, config.repeats_per_arm + 1):
            try:
                if arm == "simnibs326_headreco":
                    receipt = validate_headreco_repeat(
                        _headreco_request(config, repeat_index, 1)
                    )
                else:
                    receipt = validate_charm_repeat(
                        _charm_request(config, repeat_index, 1)
                    )
                receipts[(arm, repeat_index)] = receipt
            except (
                HeadrecoRunError,
                CharmRunError,
                ProvenanceError,
                OSError,
                ValueError,
                KeyError,
                TypeError,
            ) as exc:
                failures.append(
                    {
                        "arm": arm,
                        "repeat_index": repeat_index,
                        "error": str(exc),
                    }
                )
    return receipts, failures


def _validate_workflow_evidence(config: ExperimentConfig) -> Dict[str, object]:
    ready = workflow_preflight(config)
    root = config.output_root / "_pipeline/module_preflight"
    receipts = {}
    for name in PREFLIGHT_FILES:
        path = root / name
        if not path.is_file():
            raise WorkflowError(f"Required module preflight receipt is missing: {path}")
        payload = _read_json(path)
        if payload.get("status") != "ready":
            raise WorkflowError(f"Module preflight did not report ready: {path}")
        receipts[name] = {"path": str(path), "sha256": sha256_file(path)}
        if name != "simnibs326_cat12_probe.json":
            if payload.get("selected_subject") != config.subject:
                raise WorkflowError(f"Module preflight subject mismatch: {path}")
            if payload.get("inputs") != ready["inputs"]:
                raise WorkflowError(f"Module preflight input provenance mismatch: {path}")
            if payload.get("modules") != ready["modules"]:
                raise WorkflowError(f"Module preflight module-version mismatch: {path}")
    probe = _read_json(root / "simnibs326_cat12_probe.json")
    required_probe = {
        "patch_id": "simnibs326-cat12-r2023b-post-init-xml-warning-v2",
        "spm_jobman_initcfg_tested": True,
        "post_init_reassertion_tested": True,
        "mat_report_created": True,
    }
    for key, expected in required_probe.items():
        if probe.get(key) != expected:
            raise WorkflowError(f"CAT12 module preflight {key} mismatch")

    submission_path = config.output_root / "_pipeline/workflow/submission.json"
    if not submission_path.is_file():
        raise WorkflowError(f"Required submission receipt is missing: {submission_path}")
    submission = _read_json(submission_path)
    required_jobs = {
        "module_preflight_326",
        "module_preflight_401",
        "simnibs326_headreco_array",
        "simnibs401_charm_array",
        "collector",
    }
    jobs = submission.get("jobs")
    if submission.get("status") != "submitted" or not isinstance(jobs, dict):
        raise WorkflowError("Submission receipt is not valid")
    if set(jobs) != required_jobs or not all(str(value).isdigit() for value in jobs.values()):
        raise WorkflowError("Submission receipt job graph is incomplete or invalid")
    return {
        "module_preflights": receipts,
        "submission": {
            "path": str(submission_path),
            "sha256": sha256_file(submission_path),
            "jobs": jobs,
        },
    }


def _invalidate_terminal(config: ExperimentConfig) -> None:
    terminal = config.output_root / "_pipeline/complete.json"
    for path in (terminal, terminal.with_name("complete.json.sha256")):
        if path.exists():
            path.unlink()


def _validate_terminal(
    config: ExperimentConfig,
    terminal: Path,
    current_workflow_evidence: Mapping[str, object],
) -> Dict[str, object]:
    try:
        validate_sha256_sidecar(terminal)
    except RuntimeError as exc:
        raise WorkflowError("Terminal completion sidecar is invalid") from exc
    payload = _read_json(terminal)
    stored = _read_json(config.output_root / "_pipeline/config.json")
    if payload.get("plan_sha256") != stored.get("plan_sha256"):
        raise WorkflowError("Terminal plan binding differs from initialized plan")
    if payload.get("source") != stored.get("source"):
        raise WorkflowError("Terminal source binding differs from initialized source")
    if payload.get("workflow_evidence") != current_workflow_evidence:
        raise WorkflowError("Terminal workflow evidence differs from current evidence")
    analysis_hashes = payload.get("analysis_artifacts")
    if payload.get("status") != "complete" or not isinstance(analysis_hashes, dict):
        raise WorkflowError("Terminal completion receipt is invalid")
    for name, expected in analysis_hashes.items():
        path = config.output_root / "analysis" / str(name)
        if not path.is_file() or sha256_file(path) != expected:
            raise WorkflowError(f"Terminal analysis artifact mismatch: {path}")
    repeat_receipts = payload.get("repeat_receipts")
    if not isinstance(repeat_receipts, list) or len(repeat_receipts) != 2 * config.repeats_per_arm:
        raise WorkflowError("Terminal repeat-receipt inventory is incomplete")
    for record in repeat_receipts:
        path = Path(str(record.get("path")))
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise WorkflowError(f"Terminal repeat receipt mismatch: {path}")
    return payload


def _artifact_maps(config: ExperimentConfig, receipts):
    arms: Dict[str, Dict[int, Dict[str, Path]]] = {arm: {} for arm in ARMS}
    for (arm, repeat_index), receipt in receipts.items():
        repeat_root = config.output_root / arm / f"repeat_{repeat_index:03d}"
        if arm == "simnibs326_headreco":
            arms[arm][repeat_index] = {
                entry["relative_path"]: repeat_root / "artifacts" / entry["relative_path"]
                for entry in receipt["artifacts"]
            }
        else:
            entry = receipt["artifact"]
            arms[arm][repeat_index] = {
                entry["relative_path"]: repeat_root / entry["relative_path"]
            }
    return arms


def _write_tsv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_analysis_outputs(
    config: ExperimentConfig,
    receipts: Mapping[Tuple[str, int], Mapping[str, object]],
    arm_results: Mapping[str, Mapping[str, object]],
    retried: List[Dict[str, object]],
) -> Dict[str, str]:
    analysis_root = config.output_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    inventory_rows = []
    comparison_rows = []
    tissue_rows = []
    changed_rows = []
    for (arm, repeat_index), receipt in sorted(receipts.items()):
        entries = receipt.get("artifacts") or [receipt.get("artifact")]
        for entry in entries:
            inventory_rows.append(
                {
                    "arm": arm,
                    "repeat_index": repeat_index,
                    "attempt": receipt["attempt"],
                    "relative_path": entry["relative_path"],
                    "size_bytes": entry["size_bytes"],
                    "sha256": entry["sha256"],
                }
            )
    for arm, result in arm_results.items():
        for relative_path, artifact in result["artifacts"].items():
            for repeat_index, fingerprint in artifact["fingerprints"].items():
                if fingerprint.get("label_counts") is not None:
                    tissue_rows.append(
                        {
                            "arm": arm,
                            "repeat_index": repeat_index,
                            "relative_path": relative_path,
                            "label_counts": json.dumps(fingerprint["label_counts"], sort_keys=True),
                            "label_volumes_mm3": json.dumps(
                                fingerprint["label_volumes_mm3"], sort_keys=True
                            ),
                        }
                    )
            for comparison in artifact["comparisons"]:
                row = {
                    "arm": arm,
                    "relative_path": relative_path,
                    "kind": artifact["kind"],
                    **comparison,
                }
                for key in ("transition_counts", "dice_by_label", "absolute_threshold_counts"):
                    if row.get(key) is not None:
                        row[key] = json.dumps(row[key], sort_keys=True)
                comparison_rows.append(row)
                if int(comparison["changed_voxels"]) > 0:
                    changed_rows.append(row)

    inventory_fields = ["arm", "repeat_index", "attempt", "relative_path", "size_bytes", "sha256"]
    comparison_fields = [
        "arm",
        "relative_path",
        "kind",
        "repeat_a",
        "repeat_b",
        "file_bytes_equal",
        "array_equal",
        "geometry_equal",
        "header_equal",
        "changed_voxels",
        "changed_fraction",
        "max_absolute_difference",
        "mean_absolute_difference",
        "absolute_threshold_counts",
        "transition_counts",
        "dice_by_label",
    ]
    outputs = {
        "artifact_inventory.tsv": inventory_rows,
        "repeat_comparisons.tsv": comparison_rows,
        "tissue_summary.tsv": tissue_rows,
        "changed_voxel_summary.tsv": changed_rows,
    }
    _write_tsv(analysis_root / "artifact_inventory.tsv", inventory_rows, inventory_fields)
    _write_tsv(analysis_root / "repeat_comparisons.tsv", comparison_rows, comparison_fields)
    _write_tsv(
        analysis_root / "tissue_summary.tsv",
        tissue_rows,
        ["arm", "repeat_index", "relative_path", "label_counts", "label_volumes_mm3"],
    )
    _write_tsv(
        analysis_root / "changed_voxel_summary.tsv",
        changed_rows,
        comparison_fields,
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "workflow": "simnibs_segmentation_determinism",
        "scope": scope_payload(config),
        "validated_repeats": len(receipts),
        "retried_repeats": retried,
        "arms": arm_results,
        "source_outputs_modified": False,
    }
    manifest_path = analysis_root / "determinism_manifest.json"
    write_json_atomic(manifest_path, manifest)
    report = [
        "# Single-subject segmentation determinism report",
        "",
        f"Subject: `{config.subject}`",
        "",
        f"Validated repeats: {len(receipts)} / {2 * config.repeats_per_arm}",
        "",
    ]
    for arm in ARMS:
        result = arm_results[arm]
        report.extend(
            [
                f"## {arm}",
                "",
                f"- Geometric determinism: `{result['geometric_deterministic']}`",
                f"- Operational determinism: `{result['operational_deterministic']}`",
                f"- Discrete segmentation determinism: `{result['discrete_segmentation_deterministic']}`",
                f"- Probability-map determinism: `{result['probability_map_deterministic']}`",
                f"- Intensity-image determinism: `{result['intensity_image_deterministic']}`",
                "",
            ]
        )
    report.extend(["## Retries", "", json.dumps(retried, sort_keys=True), ""])
    report_path = analysis_root / "report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    output_paths = [
        analysis_root / name
        for name in (
            "determinism_manifest.json",
            "artifact_inventory.tsv",
            "repeat_comparisons.tsv",
            "tissue_summary.tsv",
            "changed_voxel_summary.tsv",
            "report.md",
        )
    ]
    return {path.name: sha256_file(path) for path in output_paths}


def _git_commit() -> str:
    override = os.environ.get("SEGDET_GIT_COMMIT")
    if override:
        return override
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def collect(config: ExperimentConfig) -> Dict[str, object]:
    validate_output_isolation(config)
    config_path = config.output_root / "_pipeline/config.json"
    if config_path.is_file():
        ownership = _read_json(config_path)
        stored_output = ownership.get("config", {}).get("output_root")
        if (
            ownership.get("workflow") != "simnibs_segmentation_determinism"
            or stored_output != str(config.output_root)
        ):
            raise WorkflowError(
                "Refusing recovery in an output root not owned by this exact workflow"
            )
    try:
        stored = _ensure_initialized(config)
    except (WorkflowError, ProvenanceError, OSError, ValueError) as exc:
        if not (config.output_root / "_pipeline/config.json").is_file():
            raise
        _invalidate_terminal(config)
        analysis_root = config.output_root / "analysis"
        analysis_root.mkdir(parents=True, exist_ok=True)
        audit = {
            "schema_version": 1,
            "status": "incomplete",
            "expected_repeats": 2 * config.repeats_per_arm,
            "validated_repeats": 0,
            "failures": [{"arm": "workflow_provenance", "error": str(exc)}],
            "retried_repeats": [],
            "source_outputs_modified": False,
        }
        write_json_atomic(analysis_root / "incomplete_audit.json", audit)
        return audit
    receipts, failures = _validate_receipts(config)
    workflow_evidence = None
    try:
        workflow_evidence = _validate_workflow_evidence(config)
    except (WorkflowError, ProvenanceError, OSError, ValueError, KeyError, TypeError) as exc:
        failures.append({"arm": "workflow_evidence", "error": str(exc)})
    expected = 2 * config.repeats_per_arm
    analysis_root = config.output_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    retried = [
        {"arm": arm, "repeat_index": repeat, "attempt": int(receipt["attempt"])}
        for (arm, repeat), receipt in sorted(receipts.items())
        if int(receipt.get("attempt", 1)) > 1
    ]
    if failures or len(receipts) != expected:
        _invalidate_terminal(config)
        audit = {
            "schema_version": 1,
            "status": "incomplete",
            "expected_repeats": expected,
            "validated_repeats": len(receipts),
            "failures": failures,
            "retried_repeats": retried,
            "source_outputs_modified": False,
        }
        write_json_atomic(analysis_root / "incomplete_audit.json", audit)
        return audit

    artifacts = _artifact_maps(config, receipts)
    try:
        from .analysis import analyze_arm

        arm_results = {arm: analyze_arm(arm, artifacts[arm]) for arm in ARMS}
        for arm in ARMS:
            arm_results[arm]["operational_deterministic"] = all(
                int(receipt.get("attempt", 1)) == 1
                for (receipt_arm, _), receipt in receipts.items()
                if receipt_arm == arm
            )
    except Exception as exc:
        _invalidate_terminal(config)
        audit = {
            "schema_version": 1,
            "status": "incomplete",
            "expected_repeats": expected,
            "validated_repeats": len(receipts),
            "failures": [{"arm": "analysis", "error": str(exc)}],
            "retried_repeats": retried,
            "source_outputs_modified": False,
        }
        write_json_atomic(analysis_root / "incomplete_audit.json", audit)
        return audit
    hashes = _write_analysis_outputs(config, receipts, arm_results, retried)
    incomplete = analysis_root / "incomplete_audit.json"
    if incomplete.exists():
        incomplete.unlink()
    terminal = {
        "schema_version": 1,
        "status": "complete",
        "workflow": "simnibs_segmentation_determinism",
        "plan_sha256": stored["plan_sha256"],
        "git_commit": _git_commit(),
        "scope": scope_payload(config),
        "validated_repeats": len(receipts),
        "retried_repeats": retried,
        "source": stored["source"],
        "workflow_evidence": workflow_evidence,
        "repeat_receipts": [
            {
                "arm": arm,
                "repeat_index": repeat,
                "path": str(config.output_root / arm / f"repeat_{repeat:03d}/complete.json"),
                "sha256": sha256_file(
                    config.output_root / arm / f"repeat_{repeat:03d}/complete.json"
                ),
            }
            for arm, repeat in sorted(receipts)
        ],
        "analysis_artifacts": hashes,
        "source_outputs_modified": False,
    }
    terminal_path = config.output_root / "_pipeline/complete.json"
    write_json_atomic(terminal_path, terminal)
    write_sha256_sidecar(terminal_path)
    return terminal


def status(config: ExperimentConfig) -> Dict[str, object]:
    pipeline = config.output_root / "_pipeline"
    config_path = pipeline / "config.json"
    if not config.output_root.exists():
        return {"status": "ok", "state": "uninitialized", "validated_repeats": 0}
    if not config_path.is_file():
        return {"status": "error", "state": "foreign_output_root", "validated_repeats": 0}
    terminal = pipeline / "complete.json"
    incomplete = config.output_root / "analysis/incomplete_audit.json"
    if incomplete.is_file():
        return {
            "status": "ok",
            "state": "incomplete",
            "validated_repeats": int(_read_json(incomplete).get("validated_repeats", 0)),
            "incomplete_audit": str(incomplete),
        }
    if terminal.is_file():
        current_receipts, failures = _validate_receipts(config)
        try:
            current_evidence = _validate_workflow_evidence(config)
            terminal_payload = _validate_terminal(config, terminal, current_evidence)
        except (WorkflowError, ProvenanceError, OSError, ValueError, KeyError, TypeError) as exc:
            failures.append({"arm": "workflow_evidence", "error": str(exc)})
        if failures or len(current_receipts) != 2 * config.repeats_per_arm:
            return {
                "status": "error",
                "state": "stale_complete",
                "validated_repeats": len(current_receipts),
                "failures": failures,
                "completion_receipt": str(terminal),
            }
        return {
            "status": "ok",
            "state": "complete",
            "validated_repeats": int(terminal_payload.get("validated_repeats", 0)),
            "completion_receipt": str(terminal),
        }
    receipts = sum(
        1
        for arm in ARMS
        for path in (config.output_root / arm).glob("repeat_*/complete.json")
        if path.is_file()
    )
    return {
        "status": "ok",
        "state": "prepared" if receipts == 0 else "in_progress",
        "validated_repeats": receipts,
        "expected_repeats": 2 * config.repeats_per_arm,
    }
