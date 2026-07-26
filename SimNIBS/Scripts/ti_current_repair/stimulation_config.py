#!/usr/bin/env python3
"""Confirmed targets.csv-backed stimulation configuration."""

from __future__ import annotations

import csv
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CURRENT_REPAIR_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = CURRENT_REPAIR_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.append(str(SCRIPTS_ROOT))

from CamCan_Experiment.simulation.target_montages import (  # noqa: E402
    DEFAULT_ELECTRODE_CONDUCTIVITY,
    DEFAULT_ELECTRODE_RADIUS_MM,
    DEFAULT_ELECTRODE_SHAPE,
    DEFAULT_ELECTRODE_THICKNESS_MM,
    MONTAGE_ALIASES,
    ROI_PRESET_KEYS,
    TARGETS_CSV_PATH,
)
from CamCan_Experiment.utils.camcan_dataset import (  # noqa: E402
    CONFIRMED_TARGETS_SHA256,
)


@dataclass(frozen=True)
class StimulationPair:
    anode: str
    cathode: str
    current_a: float

    def to_dict(self) -> dict[str, object]:
        return {
            "anode": self.anode,
            "cathode": self.cathode,
            "current_a": self.current_a,
        }


@dataclass(frozen=True)
class StimulationConfig:
    montage_preset: str
    target_roi: str
    targets_csv: Path
    targets_csv_sha256: str
    e_target: float
    stimulated_volume: float
    configuration: int
    pair1: StimulationPair
    pair2: StimulationPair
    electrode_radius_mm: float
    electrode_thickness_mm: float
    electrode_shape: str
    electrode_conductivity: float

    def to_dict(self) -> dict[str, object]:
        return {
            "montage_preset": self.montage_preset,
            "target_roi": self.target_roi,
            "targets_csv": str(self.targets_csv),
            "targets_csv_sha256": self.targets_csv_sha256,
            "e_target": self.e_target,
            "stimulated_volume": self.stimulated_volume,
            "configuration": self.configuration,
            "pair1": self.pair1.to_dict(),
            "pair2": self.pair2.to_dict(),
            "electrode": {
                "radius_mm": self.electrode_radius_mm,
                "thickness_mm": self.electrode_thickness_mm,
                "shape": self.electrode_shape,
                "conductivity": self.electrode_conductivity,
            },
        }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_key(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def _canonical_montage_preset(value: str) -> str:
    normalized = _normalized_key(value)
    return MONTAGE_ALIASES.get(normalized, normalized)


def _target_roi_for_preset(montage_preset: str) -> tuple[str, str]:
    preset = _canonical_montage_preset(montage_preset)
    by_preset = {
        _canonical_montage_preset(value): roi
        for roi, value in ROI_PRESET_KEYS.items()
    }
    try:
        return preset, by_preset[preset]
    except KeyError as exc:
        available = ", ".join(sorted(by_preset))
        raise ValueError(
            f"Unsupported confirmed montage preset {montage_preset!r}. "
            f"Available presets: {available}."
        ) from exc


def _parse_pair(value: str, *, field_name: str) -> tuple[str, str]:
    parts = [part.strip() for part in value.split("-", maxsplit=1)]
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"Expected {field_name} formatted as 'anode-cathode'; got {value!r}."
        )
    return parts[0], parts[1]


def _finite_float(value: object, *, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected finite numeric {field_name}; got {value!r}.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Expected finite numeric {field_name}; got {value!r}.")
    return parsed


def _load_target_row(targets_csv: Path, target_roi: str) -> dict[str, str]:
    with targets_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            dict(row)
            for row in csv.DictReader(handle)
            if row.get("roi") == target_roi
        ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one targets.csv row for {target_roi!r}; "
            f"found {len(rows)} in {targets_csv}."
        )
    return rows[0]


def resolve_confirmed_stimulation(
    montage_preset: str,
    *,
    targets_csv: str | Path = TARGETS_CSV_PATH,
) -> StimulationConfig:
    """Resolve one montage from the repository's confirmed optimized CSV."""
    csv_path = Path(targets_csv).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Confirmed targets.csv does not exist: {csv_path}")

    observed_hash = file_sha256(csv_path)
    if observed_hash != CONFIRMED_TARGETS_SHA256:
        raise ValueError(
            "targets.csv is not the confirmed optimized file: "
            f"{observed_hash} != {CONFIRMED_TARGETS_SHA256} ({csv_path})"
        )

    preset, target_roi = _target_roi_for_preset(montage_preset)
    row = _load_target_row(csv_path, target_roi)
    pair1_anode, pair1_cathode = _parse_pair(row["pair1"], field_name="pair1")
    pair2_anode, pair2_cathode = _parse_pair(row["pair2"], field_name="pair2")

    try:
        configuration = int(row["configuration"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected integer configuration for {target_roi}; "
            f"got {row.get('configuration')!r}."
        ) from exc

    return StimulationConfig(
        montage_preset=preset,
        target_roi=target_roi,
        targets_csv=csv_path,
        targets_csv_sha256=observed_hash,
        e_target=_finite_float(row["E_target"], field_name="E_target"),
        stimulated_volume=_finite_float(
            row["stimulated_volume"],
            field_name="stimulated_volume",
        ),
        configuration=configuration,
        pair1=StimulationPair(
            anode=pair1_anode,
            cathode=pair1_cathode,
            current_a=_finite_float(row["current1"], field_name="current1") * 1e-3,
        ),
        pair2=StimulationPair(
            anode=pair2_anode,
            cathode=pair2_cathode,
            current_a=_finite_float(row["current2"], field_name="current2") * 1e-3,
        ),
        electrode_radius_mm=DEFAULT_ELECTRODE_RADIUS_MM,
        electrode_thickness_mm=DEFAULT_ELECTRODE_THICKNESS_MM,
        electrode_shape=DEFAULT_ELECTRODE_SHAPE,
        electrode_conductivity=DEFAULT_ELECTRODE_CONDUCTIVITY,
    )


def validate_stimulation_config(raw: object) -> StimulationConfig:
    """Validate that a serialized config exactly matches its confirmed CSV row."""
    if not isinstance(raw, dict):
        raise ValueError("The top-level 'stimulation' section must be a JSON object.")

    montage_preset = raw.get("montage_preset")
    targets_csv = raw.get("targets_csv")
    if not isinstance(montage_preset, str) or not montage_preset.strip():
        raise ValueError("Expected non-empty 'stimulation.montage_preset'.")
    if not isinstance(targets_csv, str) or not targets_csv.strip():
        raise ValueError("Expected non-empty 'stimulation.targets_csv'.")

    resolved = resolve_confirmed_stimulation(
        montage_preset,
        targets_csv=targets_csv,
    )
    expected = resolved.to_dict()
    if raw != expected:
        differences = [
            key
            for key in sorted(set(raw) | set(expected))
            if raw.get(key) != expected.get(key)
        ]
        raise ValueError(
            "Serialized stimulation parameters do not exactly match the confirmed "
            f"targets.csv row; differing fields: {', '.join(differences)}."
        )
    return resolved


def stimulation_summary(stimulation: StimulationConfig) -> dict[str, Any]:
    return {
        "montage_preset": stimulation.montage_preset,
        "target_roi": stimulation.target_roi,
        "targets_csv": str(stimulation.targets_csv),
        "targets_csv_sha256": stimulation.targets_csv_sha256,
        "pair1": (
            f"{stimulation.pair1.anode}-{stimulation.pair1.cathode} "
            f"{stimulation.pair1.current_a * 1e3:.15g} mA"
        ),
        "pair2": (
            f"{stimulation.pair2.anode}-{stimulation.pair2.cathode} "
            f"{stimulation.pair2.current_a * 1e3:.15g} mA"
        ),
    }
