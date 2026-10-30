#!/usr/bin/env python3
"""Dependency-light output naming for supported stimulation modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FieldArtifactSpec:
    mode: str
    mesh_filename: str
    brain_nifti_filename: str
    volume_prefix: str
    element_field_name: str
    display_name: str
    fem_solves_per_simulation: int


TI_ARTIFACTS = FieldArtifactSpec(
    mode="ti",
    mesh_filename="TI.msh",
    brain_nifti_filename="ti_brain_only.nii.gz",
    volume_prefix="TI_Volumetric_",
    element_field_name="TImax",
    display_name="TI electric-field magnitude",
    fem_solves_per_simulation=2,
)

TDCS_ARTIFACTS = FieldArtifactSpec(
    mode="tdcs",
    mesh_filename="TDCS.msh",
    brain_nifti_filename="tdcs_brain_only.nii.gz",
    volume_prefix="TDCS_Volumetric_",
    element_field_name="E_magnitude",
    display_name="tDCS electric-field magnitude",
    fem_solves_per_simulation=1,
)


def stimulation_mode(stimulation: object) -> str:
    if isinstance(stimulation, dict):
        mode = stimulation.get("mode", "ti")
    else:
        mode = getattr(stimulation, "mode", "ti")
    if not isinstance(mode, str):
        raise ValueError(f"Invalid stimulation mode: {mode!r}.")
    return mode


def artifacts_for_mode(mode: str) -> FieldArtifactSpec:
    if mode == "ti":
        return TI_ARTIFACTS
    if mode == "tdcs":
        return TDCS_ARTIFACTS
    raise ValueError(f"Unsupported stimulation mode: {mode!r}.")


def artifacts_for_stimulation(stimulation: Any) -> FieldArtifactSpec:
    return artifacts_for_mode(stimulation_mode(stimulation))
