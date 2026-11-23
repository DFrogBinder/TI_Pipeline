"""Canonical, fail-closed configuration for the v2 repeatability campaign."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping


CANONICAL_SUBJECTS = (
    "sub-CC110174",
    "sub-CC121144",
    "sub-CC310407",
    "sub-CC320616",
    "sub-CC420071",
    "sub-CC410432",
    "sub-CC510415",
    "sub-CC520127",
    "sub-CC610631",
    "sub-CC720941",
)
TARGETS = ("left_hippocampus", "right_m1")
CONDITIONS = ("remesh", "fixed_mesh")
OUTCOMES = (
    "ti_target",
    "tdcs_pair1_matched_target",
    "tdcs_pair1_under_anode",
)
EXPECTED_HEAD_POLICY = {
    "remesh": "rebuild_per_repeat",
    "fixed_mesh": "reuse_representative_geometry",
}
EXPECTED_ELECTRODE_POLICY = {
    "remesh": "rebuild_unseeded_per_repeat",
    "fixed_mesh": "reuse_pair_specific_geometry",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: object, label: str) -> str:
    text = str(value)
    if not SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _require_subject_hashes(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(CANONICAL_SUBJECTS):
        raise ValueError(f"{label} must contain the canonical cohort exactly")
    return {
        subject: _require_sha256(value[subject], f"{label}[{subject}]")
        for subject in CANONICAL_SUBJECTS
    }


@dataclass(frozen=True)
class StudyConfigV2:
    """Immutable production contract; exploratory legacy configs cannot launch."""

    simnibs_version: str
    simnibs_module: str
    subjects: tuple[str, ...]
    targets: tuple[str, ...]
    conditions: tuple[str, ...]
    repeat_count: int
    corrected_segmentation_sha256: dict[str, str]
    atlas_sha256: dict[str, str]
    targets_csv_sha256: str
    head_mesh_policy: dict[str, str]
    electrode_mesh_policy: dict[str, str]
    authoritative_field_format: str
    nifti_role: str
    schema_version: int = 2
    outcomes: tuple[str, ...] = OUTCOMES

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StudyConfigV2":
        if "mesh_mode" in payload:
            raise ValueError(
                "Legacy mesh_mode configurations may be inventoried but cannot launch v2"
            )
        if int(payload.get("schema_version", -1)) != 2:
            raise ValueError("schema_version must be 2")
        version = str(payload.get("simnibs_version", ""))
        if version != "4.6.0":
            raise ValueError("Production v2 requires SimNIBS 4.6.0")
        module = str(payload.get("simnibs_module", ""))
        if not module or "4.6" not in module:
            raise ValueError("simnibs_module must identify the SimNIBS 4.6.0 module")
        subjects = tuple(str(value) for value in payload.get("subjects", ()))
        if subjects != CANONICAL_SUBJECTS:
            raise ValueError(
                "subjects must equal the canonical 9+1 cohort with sub-CC510415 "
                "and without sub-CC520083"
            )
        targets = tuple(str(value) for value in payload.get("targets", ()))
        if targets != TARGETS:
            raise ValueError(f"targets must equal {TARGETS}")
        conditions = tuple(str(value) for value in payload.get("conditions", ()))
        if conditions != CONDITIONS:
            raise ValueError(f"conditions must equal {CONDITIONS}")
        repeats = int(payload.get("repeat_count", -1))
        if repeats != 40:
            raise ValueError("repeat_count must be 40")
        head_policy = dict(payload.get("head_mesh_policy", {}))
        if head_policy != EXPECTED_HEAD_POLICY:
            raise ValueError(f"head_mesh_policy must equal {EXPECTED_HEAD_POLICY}")
        electrode_policy = dict(payload.get("electrode_mesh_policy", {}))
        if electrode_policy != EXPECTED_ELECTRODE_POLICY:
            raise ValueError(
                f"electrode_mesh_policy must equal {EXPECTED_ELECTRODE_POLICY}"
            )
        authoritative = str(payload.get("authoritative_field_format", ""))
        if authoritative != "native_msh":
            raise ValueError("authoritative_field_format must be native_msh")
        nifti_role = str(payload.get("nifti_role", ""))
        if nifti_role != "derived_qc":
            raise ValueError("nifti_role must be derived_qc")
        return cls(
            simnibs_version=version,
            simnibs_module=module,
            subjects=subjects,
            targets=targets,
            conditions=conditions,
            repeat_count=repeats,
            corrected_segmentation_sha256=_require_subject_hashes(
                payload.get("corrected_segmentation_sha256"),
                "corrected_segmentation_sha256",
            ),
            atlas_sha256=_require_subject_hashes(
                payload.get("atlas_sha256"), "atlas_sha256"
            ),
            targets_csv_sha256=_require_sha256(
                payload.get("targets_csv_sha256"), "targets_csv_sha256"
            ),
            head_mesh_policy=head_policy,
            electrode_mesh_policy=electrode_policy,
            authoritative_field_format=authoritative,
            nifti_role=nifti_role,
        )

    @property
    def expected_simulations(self) -> int:
        return (
            len(self.subjects)
            * len(self.targets)
            * len(self.conditions)
            * self.repeat_count
        )

    @property
    def expected_component_solves(self) -> int:
        return self.expected_simulations * 2

