from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


TARGETS_CSV_PATH = Path(__file__).resolve().parents[2] / "utils" / "targets.csv"

DEFAULT_ELECTRODE_RADIUS_MM = 10.0
DEFAULT_ELECTRODE_THICKNESS_MM = 2.0
DEFAULT_ELECTRODE_SHAPE = "ellipse"
DEFAULT_ELECTRODE_CONDUCTIVITY = 1.4


@dataclass(frozen=True)
class PairSpec:
    anode: str
    cathode: str
    current_a: float

    @property
    def current_amp(self) -> float:
        return self.current_a


@dataclass(frozen=True)
class MontageSpec:
    name: str
    description: str
    roi: str
    e_target: float
    stimulated_volume: float
    configuration: int
    pair1: PairSpec
    pair2: PairSpec
    electrode_radius_mm: float = DEFAULT_ELECTRODE_RADIUS_MM
    electrode_thickness_mm: float = DEFAULT_ELECTRODE_THICKNESS_MM
    electrode_shape: str = DEFAULT_ELECTRODE_SHAPE
    electrode_conductivity: float = DEFAULT_ELECTRODE_CONDUCTIVITY


ROI_PRESET_KEYS = {
    "ctx_lh_G_precentral": "left-m1",
    "ctx_lh_G_front_middle": "left-dlpfc",
    "Left_Hippocampus": "left-hippocampus",
    "Left_Thalamus": "left-thalamus",
    "Left_Pallidum": "left-pallidum",
    "ctx_rh_G_precentral": "right-m1",
    "ctx_rh_G_front_middle": "right-dlpfc",
    "Right_Hippocampus": "right-hippocampus",
    "Right_Thalamus": "right-thalamus",
    "Right_Pallidum": "right-pallidum",
}

MONTAGE_ALIASES = {
    "left-dlpc": "left-dlpfc",
    "right-dlpc": "right-dlpfc",
}


def _normalized_key(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def preset_key_for_roi(roi: str) -> str:
    return ROI_PRESET_KEYS.get(roi, _normalized_key(roi))


def _parse_pair(value: str) -> tuple[str, str]:
    try:
        anode, cathode = value.split("-", maxsplit=1)
    except ValueError as exc:
        raise ValueError(
            f"Expected electrode pair formatted as 'anode-cathode', got '{value}'."
        ) from exc
    return anode, cathode


def _load_montage_presets(csv_path: Path = TARGETS_CSV_PATH) -> dict[str, MontageSpec]:
    presets: dict[str, MontageSpec] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = preset_key_for_roi(row["roi"])
            pair1_anode, pair1_cathode = _parse_pair(row["pair1"])
            pair2_anode, pair2_cathode = _parse_pair(row["pair2"])
            presets[key] = MontageSpec(
                name=key,
                description=f"{row['roi']} montage from {csv_path.name}.",
                roi=row["roi"],
                e_target=float(row["E_target"]),
                stimulated_volume=float(row["stimulated_volume"]),
                configuration=int(row["configuration"]),
                pair1=PairSpec(
                    pair1_anode,
                    pair1_cathode,
                    float(row["current1"]) * 1e-3,
                ),
                pair2=PairSpec(
                    pair2_anode,
                    pair2_cathode,
                    float(row["current2"]) * 1e-3,
                ),
            )
    return presets


MONTAGE_PRESETS = _load_montage_presets()
MONTAGE_CHOICES = sorted(set(MONTAGE_PRESETS) | set(MONTAGE_ALIASES))


def normalize_montage_preset(name: str) -> str:
    key = _normalized_key(name)
    if key in MONTAGE_PRESETS:
        return key
    if key in MONTAGE_ALIASES:
        return MONTAGE_ALIASES[key]
    roi_lookup = {
        _normalized_key(roi): preset
        for roi, preset in ROI_PRESET_KEYS.items()
    }
    return roi_lookup.get(key, key)


def resolve_montage_preset(name: str) -> MontageSpec:
    key = normalize_montage_preset(name)
    try:
        return MONTAGE_PRESETS[key]
    except KeyError as exc:
        available = ", ".join(MONTAGE_CHOICES)
        raise ValueError(f"Unknown montage preset '{name}'. Available presets: {available}.") from exc
