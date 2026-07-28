from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from utils.camcan_dataset import (
    canonical_subject,
    electrode_names_from_target_row,
    load_individualized_target_row,
    sha256_file,
)


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


def targets_csv_sha256(csv_path: Path = TARGETS_CSV_PATH) -> str:
    digest = hashlib.sha256()
    with csv_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def resolve_individualized_montage(
    csv_path: str | Path,
    *,
    subject: str,
    dataset_roi: str,
    expected_targets_roi: str,
    expected_montage_preset: str,
    expected_sha256: str,
) -> tuple[MontageSpec, dict[str, str]]:
    path = Path(csv_path).expanduser().resolve(strict=True)
    actual_sha256 = sha256_file(path)
    if not expected_sha256 or actual_sha256 != expected_sha256:
        raise ValueError(
            "Individualized targets CSV hash mismatch: "
            f"{actual_sha256} != {expected_sha256}"
        )
    row = load_individualized_target_row(
        path,
        subject=subject,
        dataset_roi=dataset_roi,
    )
    if row["roi"].strip() != expected_targets_roi:
        raise ValueError(
            f"Individualized montage for {canonical_subject(subject)}/{dataset_roi} "
            f"uses targets ROI {row['roi']!r}; expected {expected_targets_roi!r}."
        )
    selected_preset = normalize_montage_preset(row["montage_preset"])
    expected_preset = normalize_montage_preset(expected_montage_preset)
    if selected_preset != expected_preset:
        raise ValueError(
            f"Individualized montage preset {row['montage_preset']!r} does not "
            f"match {dataset_roi}; expected {expected_montage_preset!r}."
        )
    if row["pareto_selection"].strip() != "TI_free.Emin":
        raise ValueError(
            "Individualized montage is not the predeclared TI_free.Emin "
            f"Pareto solution: {canonical_subject(subject)}/{dataset_roi}"
        )
    pair1_anode, pair1_cathode, pair2_anode, pair2_cathode = (
        electrode_names_from_target_row(row)
    )
    current1_ma = float(row["current1"])
    current2_ma = float(row["current2"])
    if not 0 < current1_ma <= 2.0 or not 0 < current2_ma <= 2.0:
        raise ValueError(
            "Individualized montage currents must be in (0, 2] mA: "
            f"{current1_ma}, {current2_ma}"
        )
    montage = MontageSpec(
        name=expected_preset,
        description=(
            f"{canonical_subject(subject)} {dataset_roi} individualized "
            f"TI_free.Emin montage from {path.name}."
        ),
        roi=row["roi"].strip(),
        e_target=float(row["E_target"]),
        stimulated_volume=float(row["stimulated_volume"]),
        configuration=int(row["configuration"]),
        pair1=PairSpec(pair1_anode, pair1_cathode, current1_ma * 1e-3),
        pair2=PairSpec(pair2_anode, pair2_cathode, current2_ma * 1e-3),
    )
    return montage, row
