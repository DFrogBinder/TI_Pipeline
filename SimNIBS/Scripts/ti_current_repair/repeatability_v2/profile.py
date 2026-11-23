"""Operational path profile kept separate from the canonical scientific scope."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .config import StudyConfigV2


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
        output_root = Path(str(payload.get("output_root", ""))).expanduser().resolve(
            strict=False
        )
        cache_root = Path(str(payload.get("cache_root", ""))).expanduser().resolve(
            strict=False
        )
        targets_csv = Path(str(payload.get("targets_csv", ""))).expanduser().resolve(
            strict=False
        )
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
                t1=Path(str(record.get("t1", ""))).expanduser().resolve(strict=False),
                t2=Path(str(record.get("t2", ""))).expanduser().resolve(strict=False),
                corrected_segmentation=Path(
                    str(record.get("corrected_segmentation", ""))
                ).expanduser().resolve(strict=False),
                atlas=Path(str(record.get("atlas", ""))).expanduser().resolve(
                    strict=False
                ),
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

