"""Resolve optimized CamCan post-processing electrodes from targets.csv."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from utils.camcan_dataset import (
    CamCanRoiConfig,
    CONFIRMED_TARGETS_SHA256,
    electrode_names_for_config,
    parse_dataset_name,
    sha256_file,
)


SIMNIBS_CAP_TEMPLATE = (
    "{root}/{subject}/anat/m2m_{subject}/eeg_positions/"
    "EEG10-10_UI_Jurak_2007.csv"
)


@dataclass(frozen=True)
class CamCanPostElectrodes:
    roi: CamCanRoiConfig
    names: tuple[str, str, str, str]
    eeg_positions_path_template: str
    targets_csv_sha256: str


def resolve_camcan_post_electrodes(
    batch_root: str | Path,
    targets_csv: str | Path,
    *,
    dataset_glob: str = "*_Data_*",
    expected_targets_sha256: str = CONFIRMED_TARGETS_SHA256,
) -> CamCanPostElectrodes:
    root = Path(batch_root).expanduser().resolve()
    target_path = Path(targets_csv).expanduser().resolve()
    actual_hash = sha256_file(target_path)
    if actual_hash != expected_targets_sha256:
        raise ValueError(
            f"targets.csv hash mismatch: {actual_hash} != {expected_targets_sha256}"
        )

    dataset_names = sorted(path.name for path in root.glob(dataset_glob) if path.is_dir())
    if not dataset_names:
        raise ValueError(f"No repeat datasets matching {dataset_glob!r} under {root}")
    configs = {parse_dataset_name(name)[0] for name in dataset_names}
    if len(configs) != 1:
        prefixes = ", ".join(sorted(config.dataset_prefix for config in configs))
        raise ValueError(f"Post-processing batch mixes ROI prefixes: {prefixes}")
    config = next(iter(configs))
    return CamCanPostElectrodes(
        roi=config,
        names=electrode_names_for_config(config, target_path),
        eeg_positions_path_template=SIMNIBS_CAP_TEMPLATE,
        targets_csv_sha256=actual_hash,
    )
