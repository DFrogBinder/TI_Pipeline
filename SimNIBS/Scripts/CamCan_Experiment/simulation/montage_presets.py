"""Simulation montage presets and ROI-to-montage resolution."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from utils.roi_registry import (
    match_fastsurfer_roi_from_directory,
    resolve_fastsurfer_roi_name,
)


@dataclass(frozen=True)
class PairSpec:
    anode: str
    cathode: str
    current_amp: float

    def as_signed_tuple(self) -> tuple[str, float, str, float]:
        return (self.anode, self.current_amp, self.cathode, -self.current_amp)


@dataclass(frozen=True)
class MontageSpec:
    name: str
    roi_name: str
    description: str
    pair1: PairSpec
    pair2: PairSpec
    electrode_radius_mm: float = 10.0
    electrode_thickness_mm: float = 2.0
    electrode_shape: str = "ellipse"
    electrode_conductivity: float = 1.4


MONTAGE_PRESETS: dict[str, MontageSpec] = {
    "left-pallidum": MontageSpec(
        name="left-pallidum",
        roi_name="Left-Pallidum",
        description="Left pallidum montage from the subject-specific runner.",
        pair1=PairSpec("Fpz", "AF8", 2e-3),
        pair2=PairSpec("TP7", "PO9", 1.261915e-3),
    ),
    "right-pallidum": MontageSpec(
        name="right-pallidum",
        roi_name="Right-Pallidum",
        description="Right pallidum montage from the subject-specific runner.",
        pair1=PairSpec("F8", "F10", 2e-3),
        pair2=PairSpec("FT7", "C3", 1.261915e-3),
    ),
    "left-thalamus": MontageSpec(
        name="left-thalamus",
        roi_name="Left-Thalamus-Proper",
        description="Left thalamus montage from the subject-specific runner.",
        pair1=PairSpec("F7", "P7", 1.588656e-3),
        pair2=PairSpec("F8", "P8", 2e-3),
    ),
    "right-thalamus": MontageSpec(
        name="right-thalamus",
        roi_name="Right-Thalamus-Proper",
        description="Right thalamus montage from the subject-specific runner.",
        pair1=PairSpec("AF7", "TP7", 2e-3),
        pair2=PairSpec("T8", "PO8", 2e-3),
    ),
    "left-hippocampus": MontageSpec(
        name="left-hippocampus",
        roi_name="Left-Hippocampus",
        description="Left hippocampus montage from the subject-specific runner.",
        pair1=PairSpec("F10", "P8", 2e-3),
        pair2=PairSpec("T7", "P7", 1.588656e-3),
    ),
    "left-m1": MontageSpec(
        name="left-m1",
        roi_name="ctx-lh-precentral",
        description="Left primary motor cortex montage from the subject-specific runner.",
        pair1=PairSpec("FC1", "FCz", 2e-3),
        pair2=PairSpec("C3", "P5", 0.632456e-3),
    ),
    "right-m1": MontageSpec(
        name="right-m1",
        roi_name="ctx-rh-precentral",
        description="Right primary motor cortex montage from the subject-specific runner.",
        pair1=PairSpec("FC6", "FT8", 2e-3),
        pair2=PairSpec("C2", "C4", 0.796214e-3),
    ),
    "right-dlpfc": MontageSpec(
        name="right-dlpfc",
        roi_name="ctx-rh-dlpfc-dkt",
        description="Right DLPFC/DLPC montage from the subject-specific runner.",
        pair1=PairSpec("AF4", "F4", 0.796214e-3),
        pair2=PairSpec("C2", "CP1", 2e-3),
    ),
}


ROI_TO_MONTAGE_PRESET: dict[str, str] = {
    preset.roi_name: preset.name for preset in MONTAGE_PRESETS.values()
}


def _normalize_preset_name(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def available_montage_names() -> tuple[str, ...]:
    return tuple(sorted(MONTAGE_PRESETS))


def list_montage_presets() -> str:
    lines = ["Available subject-specific montage presets:"]
    for name in available_montage_names():
        preset = MONTAGE_PRESETS[name]
        lines.append(
            f"- {name}: ROI={preset.roi_name}; "
            f"pair1={preset.pair1.anode}->{preset.pair1.cathode} ({preset.pair1.current_amp:.6g} A), "
            f"pair2={preset.pair2.anode}->{preset.pair2.cathode} ({preset.pair2.current_amp:.6g} A)"
        )
        lines.append(f"  {preset.description}")
    return "\n".join(lines)


def preset_for_roi_name(roi_name: str) -> MontageSpec:
    roi_match = resolve_fastsurfer_roi_name(roi_name)
    preset_name = ROI_TO_MONTAGE_PRESET.get(roi_match.canonical_name)
    if preset_name is None:
        supported = ", ".join(sorted(ROI_TO_MONTAGE_PRESET))
        raise ValueError(
            f"No simulation montage preset is configured for ROI "
            f"'{roi_match.canonical_name}'. Supported ROIs: {supported}."
        )
    return MONTAGE_PRESETS[preset_name]


def infer_montage_from_directory(directory_name: str | Path) -> MontageSpec:
    roi_match = match_fastsurfer_roi_from_directory(directory_name)
    return preset_for_roi_name(roi_match.canonical_name)


def resolve_montage_preset(
    value: str | None,
    *,
    dataset_root: str | Path | None = None,
) -> MontageSpec:
    requested = (value or "auto").strip()
    if _normalize_preset_name(requested) == "auto":
        if dataset_root is None:
            raise ValueError("--montage-preset auto requires a dataset root.")
        return infer_montage_from_directory(dataset_root)

    normalized = _normalize_preset_name(requested)
    if normalized in MONTAGE_PRESETS:
        return MONTAGE_PRESETS[normalized]

    return preset_for_roi_name(requested)


def validate_all_dataset_montages(dataset_roots: Iterable[str | Path]) -> dict[str, MontageSpec]:
    return {
        str(Path(dataset_root).expanduser().resolve()): infer_montage_from_directory(dataset_root)
        for dataset_root in dataset_roots
    }
