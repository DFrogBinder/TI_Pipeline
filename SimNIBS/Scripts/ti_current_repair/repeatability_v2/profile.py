"""Operational path profile kept separate from the canonical scientific scope."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .config import StudyConfigV2
from .inventory import sha256_file


def _required_path(payload: Mapping[str, Any], key: str) -> Path:
    value = payload.get(key)
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"Operational profile requires non-empty {key}")
    return Path(str(value)).expanduser().resolve(strict=False)


def _required_nested_path(
    record: Mapping[str, Any], *, subject: str, key: str
) -> Path:
    value = record.get(key)
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"subject_inputs[{subject}].{key} must be a non-empty path")
    return Path(str(value)).expanduser().resolve(strict=False)


@dataclass(frozen=True)
class SubjectInputs:
    t1: Path
    t2: Path
    corrected_segmentation: Path
    atlas: Path


@dataclass(frozen=True)
class OperationalProfile:
    study_config: StudyConfigV2
    output_root: Path
    cache_root: Path
    targets_csv: Path
    montage_presets: dict[str, str]
    subject_inputs: dict[str, SubjectInputs]
    historical_read_only_roots: tuple[Path, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OperationalProfile":
        if int(payload.get("schema_version", -1)) != 2:
            raise ValueError("Operational profile schema_version must be 2")
        config_raw = payload.get("study_config")
        if not isinstance(config_raw, Mapping):
            raise ValueError("Operational profile requires study_config")
        config = StudyConfigV2.from_mapping(config_raw)
        output_root = _required_path(payload, "output_root")
        cache_root = _required_path(payload, "cache_root")
        targets_csv = _required_path(payload, "targets_csv")
        historical = tuple(
            Path(str(value)).expanduser().resolve(strict=False)
            for value in payload.get("historical_read_only_roots", ())
        )
        if not historical:
            raise ValueError("At least one historical read-only root is required")
        for destination, label in (
            (output_root, "output_root"),
            (cache_root, "cache_root"),
        ):
            if any(
                destination == source or destination.is_relative_to(source)
                for source in historical
            ):
                raise ValueError(f"{label} overlaps a historical read-only root")
        if output_root == cache_root or output_root.is_relative_to(cache_root) or cache_root.is_relative_to(output_root):
            raise ValueError("output_root and cache_root must be separate")
        presets_raw = payload.get("montage_presets")
        if not isinstance(presets_raw, Mapping) or set(presets_raw) != set(config.targets):
            raise ValueError("montage_presets must define both canonical targets")
        subject_raw = payload.get("subject_inputs")
        if not isinstance(subject_raw, Mapping) or set(subject_raw) != set(config.subjects):
            raise ValueError("subject_inputs must define the canonical cohort exactly")
        subject_inputs: dict[str, SubjectInputs] = {}
        for subject in config.subjects:
            record = subject_raw[subject]
            if not isinstance(record, Mapping):
                raise ValueError(f"subject_inputs[{subject}] must be a mapping")
            subject_inputs[subject] = SubjectInputs(
                t1=_required_nested_path(record, subject=subject, key="t1"),
                t2=_required_nested_path(record, subject=subject, key="t2"),
                corrected_segmentation=_required_nested_path(
                    record, subject=subject, key="corrected_segmentation"
                ),
                atlas=_required_nested_path(record, subject=subject, key="atlas"),
            )
        return cls(
            study_config=config,
            output_root=output_root,
            cache_root=cache_root,
            targets_csv=targets_csv,
            montage_presets={
                str(key): str(value) for key, value in presets_raw.items()
            },
            subject_inputs=subject_inputs,
            historical_read_only_roots=historical,
        )


def load_profile(path: Path) -> OperationalProfile:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Operational profile must be a JSON object: {path}")
    return OperationalProfile.from_mapping(payload)


def verify_profile_source_inputs(
    profile: OperationalProfile,
) -> dict[str, object]:
    """Verify production inputs directly; never trust a caller-supplied hash flag."""

    required_files = 0
    hashed_inputs = 0
    expected_targets = profile.study_config.targets_csv_sha256
    if not profile.targets_csv.is_file() or profile.targets_csv.is_symlink():
        raise FileNotFoundError(f"Missing or unsafe targets_csv: {profile.targets_csv}")
    observed_targets = sha256_file(profile.targets_csv)
    if observed_targets != expected_targets:
        raise ValueError(
            f"targets_csv hash mismatch: {observed_targets} != {expected_targets}"
        )
    required_files += 1
    hashed_inputs += 1
    for subject in profile.study_config.subjects:
        inputs = profile.subject_inputs[subject]
        for label, path in (("t1", inputs.t1), ("t2", inputs.t2)):
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(
                    f"Missing or unsafe subject_inputs[{subject}].{label}: {path}"
                )
            required_files += 1
        for label, path, expected in (
            (
                "corrected_segmentation",
                inputs.corrected_segmentation,
                profile.study_config.corrected_segmentation_sha256[subject],
            ),
            ("atlas", inputs.atlas, profile.study_config.atlas_sha256[subject]),
        ):
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(
                    f"Missing or unsafe subject_inputs[{subject}].{label}: {path}"
                )
            observed = sha256_file(path)
            if observed != expected:
                raise ValueError(
                    f"subject_inputs[{subject}].{label} hash mismatch: "
                    f"{observed} != {expected}"
                )
            required_files += 1
            hashed_inputs += 1
    return {
        "schema_version": 2,
        "status": "verified",
        "required_files": required_files,
        "hashed_inputs": hashed_inputs,
        "subjects": len(profile.study_config.subjects),
        "source_outputs_modified": False,
    }

