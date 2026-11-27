"""Strict configuration model for reusable CHARM meshing campaigns."""

from __future__ import annotations

import hashlib
import json
import string
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


SUPPORTED_SCHEMA_VERSIONS = {1}
SUPPORTED_BACKENDS = {"direct_charm_create_mesh"}
SUPPORTED_SIMNIBS_VERSIONS = {"4.0.1", "4.6"}
SUPPORTED_SETTINGS_POLICIES = {"installation_charm_ini", "explicit_file"}
TEMPLATE_FIELDS = {"subject_id", "realization_id", "task_key", "task_id"}


@dataclass(frozen=True)
class InputColumns:
    subject_id: str
    realization_id: str | None
    label_path: str
    label_sha256: str
    label_bytes: str
    status: str


@dataclass(frozen=True)
class InputConfig:
    manifest: Path
    delimiter: str
    ready_status: str
    columns: InputColumns
    segmentation_version: str
    excluded_task_keys: tuple[str, ...]


@dataclass(frozen=True)
class OutputConfig:
    root: Path
    task_manifest: Path
    preflight_summary: Path
    mesh_template: str
    result_template: str


@dataclass(frozen=True)
class SoftwareConfig:
    backend: str
    expected_version: str
    module: str
    python: str
    settings_path: Path | None
    settings_policy: str


@dataclass(frozen=True)
class QualityConfig:
    require_tetrahedra: bool
    required_tissue_tags: tuple[int, ...]


@dataclass(frozen=True)
class ExecutionConfig:
    expected_tasks: int
    partition: str
    cpus_per_task: int
    memory: str
    time_limit: str
    max_concurrent: int
    workers_per_array_task: int
    local_staging: bool
    max_retries: int


@dataclass(frozen=True)
class ResumeConfig:
    reuse_validated: bool
    overwrite: bool


@dataclass(frozen=True)
class CampaignConfig:
    schema_version: int
    campaign_id: str
    input: InputConfig
    output: OutputConfig
    software: SoftwareConfig
    quality: QualityConfig
    execution: ExecutionConfig
    resume: ResumeConfig
    source_path: Path
    sha256: str


def _object(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _exact_fields(
    value: Mapping[str, Any], required: set[str], context: str
) -> None:
    unknown = sorted(set(value) - required)
    missing = sorted(required - set(value))
    if unknown:
        raise ValueError(f"{context} has unknown field(s): {', '.join(unknown)}")
    if missing:
        raise ValueError(f"{context} has missing field(s): {', '.join(missing)}")


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return value


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be true or false")
    return value


def _path(value: object, base: Path, context: str) -> Path:
    raw = Path(_text(value, context))
    return (raw if raw.is_absolute() else base / raw).resolve()


def _relative_output_path(value: object, root: Path, context: str) -> Path:
    raw = PurePosixPath(_text(value, context))
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"{context} must be a safe path relative to output.root")
    return (root / Path(*raw.parts)).resolve()


def _validate_template(value: object, context: str) -> str:
    template = _text(value, context)
    raw = PurePosixPath(template)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"{context} must stay beneath output.root")
    try:
        fields = {
            field_name
            for _, field_name, _, _ in string.Formatter().parse(template)
            if field_name is not None
        }
    except ValueError as exc:
        raise ValueError(f"{context} is not a valid format template: {exc}") from exc
    unknown = fields - TEMPLATE_FIELDS
    if unknown:
        raise ValueError(f"{context} uses unknown field(s): {', '.join(sorted(unknown))}")
    if "task_key" not in fields and not {"subject_id", "realization_id"}.issubset(fields):
        raise ValueError(
            f"{context} must contain task_key or both subject_id and realization_id"
        )
    return template


def _normalized_payload(config: CampaignConfig) -> dict[str, object]:
    payload = asdict(config)
    payload.pop("source_path", None)
    payload.pop("sha256", None)

    def convert(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        return value

    return convert(payload)  # type: ignore[return-value]


def load_campaign_config(path: str | Path) -> CampaignConfig:
    """Load and strictly validate a versioned JSON campaign configuration."""

    source_path = Path(path).resolve(strict=True)
    try:
        raw = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON configuration {source_path}: {exc}") from exc
    top = _object(raw, "configuration")
    top_fields = {
        "schema_version",
        "campaign_id",
        "input",
        "output",
        "software",
        "quality",
        "execution",
        "resume",
    }
    _exact_fields(top, top_fields, "configuration")
    schema_version = _integer(top["schema_version"], "schema_version", minimum=1)
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported schema_version: {schema_version}")
    campaign_id = _text(top["campaign_id"], "campaign_id")
    base = source_path.parent

    input_raw = _object(top["input"], "input")
    _exact_fields(
        input_raw,
        {
            "manifest",
            "delimiter",
            "ready_status",
            "columns",
            "segmentation_version",
            "excluded_task_keys",
        },
        "input",
    )
    columns_raw = _object(input_raw["columns"], "input.columns")
    column_fields = {
        "subject_id",
        "realization_id",
        "label_path",
        "label_sha256",
        "label_bytes",
        "status",
    }
    _exact_fields(columns_raw, column_fields, "input.columns")
    realization_column = columns_raw["realization_id"]
    if realization_column is not None:
        realization_column = _text(realization_column, "input.columns.realization_id")
    columns = InputColumns(
        subject_id=_text(columns_raw["subject_id"], "input.columns.subject_id"),
        realization_id=realization_column,
        label_path=_text(columns_raw["label_path"], "input.columns.label_path"),
        label_sha256=_text(columns_raw["label_sha256"], "input.columns.label_sha256"),
        label_bytes=_text(columns_raw["label_bytes"], "input.columns.label_bytes"),
        status=_text(columns_raw["status"], "input.columns.status"),
    )
    exclusions = input_raw["excluded_task_keys"]
    if not isinstance(exclusions, list) or not all(
        isinstance(item, str) and item for item in exclusions
    ):
        raise ValueError("input.excluded_task_keys must be a list of non-empty strings")
    delimiter = _text(input_raw["delimiter"], "input.delimiter")
    if delimiter not in {"tab", "comma"}:
        raise ValueError("input.delimiter must be 'tab' or 'comma'")
    segmentation_version = _text(
        input_raw["segmentation_version"], "input.segmentation_version"
    )
    input_config = InputConfig(
        manifest=_path(input_raw["manifest"], base, "input.manifest"),
        delimiter=delimiter,
        ready_status=_text(input_raw["ready_status"], "input.ready_status"),
        columns=columns,
        segmentation_version=segmentation_version,
        excluded_task_keys=tuple(exclusions),
    )

    output_raw = _object(top["output"], "output")
    output_fields = {
        "root",
        "task_manifest",
        "preflight_summary",
        "mesh_template",
        "result_template",
    }
    _exact_fields(output_raw, output_fields, "output")
    output_root = _path(output_raw["root"], base, "output.root")
    output_config = OutputConfig(
        root=output_root,
        task_manifest=_relative_output_path(
            output_raw["task_manifest"], output_root, "output.task_manifest"
        ),
        preflight_summary=_relative_output_path(
            output_raw["preflight_summary"], output_root, "output.preflight_summary"
        ),
        mesh_template=_validate_template(
            output_raw["mesh_template"], "output.mesh_template"
        ),
        result_template=_validate_template(
            output_raw["result_template"], "output.result_template"
        ),
    )

    software_raw = _object(top["software"], "software")
    software_fields = {
        "backend",
        "expected_version",
        "module",
        "python",
        "settings_path",
        "settings_policy",
    }
    _exact_fields(software_raw, software_fields, "software")
    backend = _text(software_raw["backend"], "software.backend")
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported software.backend: {backend}")
    expected_version = _text(
        software_raw["expected_version"], "software.expected_version"
    )
    if expected_version not in SUPPORTED_SIMNIBS_VERSIONS:
        raise ValueError(f"unsupported software.expected_version: {expected_version}")
    settings_policy = _text(
        software_raw["settings_policy"], "software.settings_policy"
    )
    if settings_policy not in SUPPORTED_SETTINGS_POLICIES:
        raise ValueError(f"unsupported software.settings_policy: {settings_policy}")
    settings_value = software_raw["settings_path"]
    if settings_policy == "installation_charm_ini" and settings_value is not None:
        raise ValueError(
            "software.settings_path must be null with installation_charm_ini"
        )
    if settings_policy == "explicit_file" and settings_value is None:
        raise ValueError("software.settings_path is required with explicit_file")
    settings_path = (
        _path(settings_value, base, "software.settings_path")
        if settings_value is not None
        else None
    )
    software_config = SoftwareConfig(
        backend=backend,
        expected_version=expected_version,
        module=_text(software_raw["module"], "software.module"),
        python=_text(software_raw["python"], "software.python"),
        settings_path=settings_path,
        settings_policy=settings_policy,
    )
    if segmentation_version != expected_version:
        raise ValueError(
            "input.segmentation_version must equal software.expected_version; "
            "mixed-version processing requires a separate declared adapter"
        )

    quality_raw = _object(top["quality"], "quality")
    _exact_fields(
        quality_raw, {"require_tetrahedra", "required_tissue_tags"}, "quality"
    )
    tags = quality_raw["required_tissue_tags"]
    if not isinstance(tags, list) or not all(
        isinstance(tag, int) and not isinstance(tag, bool) and tag > 0 for tag in tags
    ):
        raise ValueError("quality.required_tissue_tags must be positive integers")
    quality_config = QualityConfig(
        require_tetrahedra=_boolean(
            quality_raw["require_tetrahedra"], "quality.require_tetrahedra"
        ),
        required_tissue_tags=tuple(sorted(set(tags))),
    )

    execution_raw = _object(top["execution"], "execution")
    execution_fields = {
        "expected_tasks",
        "partition",
        "cpus_per_task",
        "memory",
        "time_limit",
        "max_concurrent",
        "workers_per_array_task",
        "local_staging",
        "max_retries",
    }
    _exact_fields(execution_raw, execution_fields, "execution")
    cpus = _integer(execution_raw["cpus_per_task"], "execution.cpus_per_task", minimum=1)
    workers = _integer(
        execution_raw["workers_per_array_task"],
        "execution.workers_per_array_task",
        minimum=1,
    )
    if cpus % workers:
        raise ValueError(
            "execution.cpus_per_task must divide evenly across workers_per_array_task"
        )
    execution_config = ExecutionConfig(
        expected_tasks=_integer(
            execution_raw["expected_tasks"], "execution.expected_tasks", minimum=1
        ),
        partition=_text(execution_raw["partition"], "execution.partition"),
        cpus_per_task=cpus,
        memory=_text(execution_raw["memory"], "execution.memory"),
        time_limit=_text(execution_raw["time_limit"], "execution.time_limit"),
        max_concurrent=_integer(
            execution_raw["max_concurrent"], "execution.max_concurrent", minimum=1
        ),
        workers_per_array_task=workers,
        local_staging=_boolean(
            execution_raw["local_staging"], "execution.local_staging"
        ),
        max_retries=_integer(
            execution_raw["max_retries"], "execution.max_retries", minimum=0
        ),
    )

    resume_raw = _object(top["resume"], "resume")
    _exact_fields(resume_raw, {"reuse_validated", "overwrite"}, "resume")
    resume_config = ResumeConfig(
        reuse_validated=_boolean(resume_raw["reuse_validated"], "resume.reuse_validated"),
        overwrite=_boolean(resume_raw["overwrite"], "resume.overwrite"),
    )
    if resume_config.reuse_validated and resume_config.overwrite:
        raise ValueError("resume.reuse_validated and resume.overwrite cannot both be true")

    provisional = CampaignConfig(
        schema_version=schema_version,
        campaign_id=campaign_id,
        input=input_config,
        output=output_config,
        software=software_config,
        quality=quality_config,
        execution=execution_config,
        resume=resume_config,
        source_path=source_path,
        sha256="",
    )
    normalized = json.dumps(
        _normalized_payload(provisional), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(normalized).hexdigest()
    return CampaignConfig(**{**provisional.__dict__, "sha256": digest})


def normalized_config(config: CampaignConfig) -> dict[str, object]:
    """Return the canonical, JSON-serializable configuration payload."""

    return _normalized_payload(config)
