from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPEAT_DATASET_PATTERN = re.compile(
    r"^(?P<prefix>.+)_Data_(?P<repeat>[0-9]+)$"
)
CONFIRMED_TARGETS_SHA256 = "97a8c7a72faf88d9af9e4facbdf628fba1a130d327da778bcbd00af66f2916e6"


@dataclass(frozen=True)
class CamCanRoiConfig:
    dataset_prefix: str
    targets_roi: str
    montage_preset: str


CAMCAN_ROI_CONFIGS = (
    CamCanRoiConfig("Left_M1", "ctx_lh_G_precentral", "left-m1"),
    CamCanRoiConfig("Left_Hippocampus", "Left_Hippocampus", "left-hippocampus"),
    CamCanRoiConfig("Right_DLPC", "ctx_rh_G_front_middle", "right-dlpfc"),
    CamCanRoiConfig("Right_Thalamus", "Right_Thalamus", "right-thalamus"),
)

_CONFIG_BY_PREFIX = {config.dataset_prefix: config for config in CAMCAN_ROI_CONFIGS}
_PRESET_ALIASES = {
    "right-dlpc": "right-dlpfc",
    "left-dlpc": "left-dlpfc",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_montage_preset(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-").replace(" ", "-")
    return _PRESET_ALIASES.get(normalized, normalized)


def parse_dataset_name(dataset_name: str) -> tuple[CamCanRoiConfig, str]:
    match = REPEAT_DATASET_PATTERN.fullmatch(dataset_name.strip())
    if match is None:
        raise ValueError(
            f"Unexpected CamCan dataset name {dataset_name!r}; expected <ROI>_Data_<repeat>."
        )
    prefix = match.group("prefix")
    try:
        config = _CONFIG_BY_PREFIX[prefix]
    except KeyError as exc:
        expected = ", ".join(sorted(_CONFIG_BY_PREFIX))
        raise ValueError(
            f"Unsupported CamCan ROI prefix {prefix!r}. Expected one of: {expected}."
        ) from exc
    return config, match.group("repeat")


def validate_dataset_montage(dataset_names: Iterable[str], preset: str) -> CamCanRoiConfig:
    names = [name for name in dataset_names if name.strip()]
    if not names:
        raise ValueError("No dataset names were supplied for montage validation.")

    configs = {parse_dataset_name(name)[0] for name in names}
    if len(configs) != 1:
        prefixes = ", ".join(sorted(config.dataset_prefix for config in configs))
        raise ValueError(f"Manifest mixes multiple ROI prefixes: {prefixes}.")

    config = next(iter(configs))
    selected = canonical_montage_preset(preset)
    expected = canonical_montage_preset(config.montage_preset)
    if selected != expected:
        raise ValueError(
            f"Montage preset {preset!r} does not match {config.dataset_prefix}; "
            f"expected {config.montage_preset!r}."
        )
    return config


def load_target_row(targets_csv: str | Path, targets_roi: str) -> dict[str, str]:
    path = Path(targets_csv)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("roi") == targets_roi]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one targets.csv row for {targets_roi!r}; found {len(rows)} in {path}."
        )
    return rows[0]


def electrode_names_for_config(
    config: CamCanRoiConfig,
    targets_csv: str | Path,
) -> tuple[str, str, str, str]:
    row = load_target_row(targets_csv, config.targets_roi)
    pair1 = row["pair1"].split("-")
    pair2 = row["pair2"].split("-")
    if len(pair1) != 2 or len(pair2) != 2 or not all(pair1 + pair2):
        raise ValueError(
            f"Invalid electrode pair(s) for {config.targets_roi}: "
            f"pair1={row['pair1']!r}, pair2={row['pair2']!r}."
        )
    return pair1[0], pair1[1], pair2[0], pair2[1]
