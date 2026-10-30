#!/usr/bin/env python3
"""Validated TI and explicitly specified provisional tDCS configurations."""

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

    @property
    def mode(self) -> str:
        """Keep legacy TI serialization unchanged while exposing a mode."""
        return "ti"

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


PROVISIONAL_TDCS_REVIEW_STATUS = "provisional_pending_supervisor_review"


@dataclass(frozen=True)
class TDCSStimulationConfig:
    """One-pair tDCS configuration that is intentionally not targets.csv-backed."""

    montage_preset: str
    target_roi: str
    pair: StimulationPair
    electrode_radius_mm: float
    electrode_thickness_mm: float
    electrode_shape: str
    electrode_conductivity: float
    review_status: str = PROVISIONAL_TDCS_REVIEW_STATUS

    @property
    def mode(self) -> str:
        return "tdcs"

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "review_status": self.review_status,
            "montage_preset": self.montage_preset,
            "target_roi": self.target_roi,
            "pair": self.pair.to_dict(),
            "electrode": {
                "radius_mm": self.electrode_radius_mm,
                "thickness_mm": self.electrode_thickness_mm,
                "shape": self.electrode_shape,
                "conductivity": self.electrode_conductivity,
            },
        }


AnyStimulationConfig = StimulationConfig | TDCSStimulationConfig


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


def resolve_provisional_tdcs_stimulation(
    *,
    anode: str,
    cathode: str,
    current_ma: float,
    montage_preset: str,
    target_roi: str,
    electrode_radius_mm: float,
    electrode_thickness_mm: float,
    electrode_shape: str,
    electrode_conductivity: float,
    review_status: str = PROVISIONAL_TDCS_REVIEW_STATUS,
) -> TDCSStimulationConfig:
    """Build a provisional tDCS montage without silently supplying study parameters."""
    anode = anode.strip()
    cathode = cathode.strip()
    montage_preset = montage_preset.strip()
    target_roi = target_roi.strip()
    review_status = review_status.strip()
    for value, field_name in (
        (anode, "anode"),
        (cathode, "cathode"),
        (montage_preset, "montage_preset"),
        (target_roi, "target_roi"),
        (review_status, "review_status"),
    ):
        if not value:
            raise ValueError(f"Expected non-empty tDCS {field_name}.")
    if anode.casefold() == cathode.casefold():
        raise ValueError("tDCS anode and cathode must be different electrodes.")

    current_ma_value = _finite_float(current_ma, field_name="current_ma")
    radius = _finite_float(electrode_radius_mm, field_name="electrode_radius_mm")
    thickness = _finite_float(
        electrode_thickness_mm,
        field_name="electrode_thickness_mm",
    )
    conductivity = _finite_float(
        electrode_conductivity,
        field_name="electrode_conductivity",
    )
    if current_ma_value <= 0:
        raise ValueError("tDCS current_ma must be greater than zero.")
    if radius <= 0 or thickness <= 0 or conductivity <= 0:
        raise ValueError("tDCS electrode dimensions and conductivity must be positive.")
    if not electrode_shape.strip():
        raise ValueError("Expected non-empty tDCS electrode_shape.")

    return TDCSStimulationConfig(
        montage_preset=montage_preset,
        target_roi=target_roi,
        pair=StimulationPair(
            anode=anode,
            cathode=cathode,
            current_a=current_ma_value * 1e-3,
        ),
        electrode_radius_mm=radius,
        electrode_thickness_mm=thickness,
        electrode_shape=electrode_shape.strip(),
        electrode_conductivity=conductivity,
        review_status=review_status,
    )


def validate_stimulation_config(raw: object) -> AnyStimulationConfig:
    """Validate a serialized TI or provisional tDCS configuration exactly."""
    if not isinstance(raw, dict):
        raise ValueError("The top-level 'stimulation' section must be a JSON object.")

    mode = raw.get("mode", "ti")
    if mode == "tdcs":
        pair = raw.get("pair")
        electrode = raw.get("electrode")
        if not isinstance(pair, dict):
            raise ValueError("Expected 'stimulation.pair' to be a JSON object.")
        if not isinstance(electrode, dict):
            raise ValueError("Expected 'stimulation.electrode' to be a JSON object.")
        resolved_tdcs = resolve_provisional_tdcs_stimulation(
            anode=str(pair.get("anode", "")),
            cathode=str(pair.get("cathode", "")),
            current_ma=_finite_float(
                pair.get("current_a"),
                field_name="pair.current_a",
            )
            * 1e3,
            montage_preset=str(raw.get("montage_preset", "")),
            target_roi=str(raw.get("target_roi", "")),
            electrode_radius_mm=_finite_float(
                electrode.get("radius_mm"),
                field_name="electrode.radius_mm",
            ),
            electrode_thickness_mm=_finite_float(
                electrode.get("thickness_mm"),
                field_name="electrode.thickness_mm",
            ),
            electrode_shape=str(electrode.get("shape", "")),
            electrode_conductivity=_finite_float(
                electrode.get("conductivity"),
                field_name="electrode.conductivity",
            ),
            review_status=str(raw.get("review_status", "")),
        )
        expected_tdcs = resolved_tdcs.to_dict()
        if raw != expected_tdcs:
            differences = [
                key
                for key in sorted(set(raw) | set(expected_tdcs))
                if raw.get(key) != expected_tdcs.get(key)
            ]
            raise ValueError(
                "Serialized tDCS parameters are not canonical; differing fields: "
                f"{', '.join(differences)}."
            )
        return resolved_tdcs
    if mode != "ti":
        raise ValueError(f"Unsupported stimulation mode: {mode!r}.")

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


def stimulation_summary(stimulation: AnyStimulationConfig) -> dict[str, Any]:
    if isinstance(stimulation, TDCSStimulationConfig):
        return {
            "mode": stimulation.mode,
            "review_status": stimulation.review_status,
            "montage_preset": stimulation.montage_preset,
            "target_roi": stimulation.target_roi,
            "pair": (
                f"{stimulation.pair.anode}-{stimulation.pair.cathode} "
                f"{stimulation.pair.current_a * 1e3:.15g} mA"
            ),
            "electrode": (
                f"{stimulation.electrode_shape}, radius "
                f"{stimulation.electrode_radius_mm:.15g} mm, thickness "
                f"{stimulation.electrode_thickness_mm:.15g} mm, conductivity "
                f"{stimulation.electrode_conductivity:.15g} S/m"
            ),
        }
    return {
        "mode": stimulation.mode,
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
