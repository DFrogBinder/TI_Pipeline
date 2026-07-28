from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPEAT_DATASET_PATTERN = re.compile(
    r"^(?P<prefix>.+)_Data_(?P<repeat>[0-9]+)$"
)
CONFIRMED_TARGETS_SHA256 = "97a8c7a72faf88d9af9e4facbdf628fba1a130d327da778bcbd00af66f2916e6"
INDIVIDUALIZED_TARGET_REQUIRED_FIELDS = (
    "subject",
    "dataset_roi",
    "roi",
    "montage_preset",
    "E_target",
    "stimulated_volume",
    "configuration",
    "pair1",
    "pair2",
    "current1",
    "current2",
    "pareto_selection",
)


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


def canonical_subject(value: str) -> str:
    subject = value.strip()
    if not subject:
        raise ValueError("Subject ID is empty.")
    return subject if subject.startswith("sub-") else f"sub-{subject}"


def load_individualized_target_rows(
    targets_csv: str | Path,
) -> list[dict[str, str]]:
    path = Path(targets_csv)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [
            field
            for field in INDIVIDUALIZED_TARGET_REQUIRED_FIELDS
            if field not in (reader.fieldnames or ())
        ]
        if missing:
            raise ValueError(
                f"Individualized targets CSV lacks required columns "
                f"{missing}: {path}"
            )
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Individualized targets CSV is empty: {path}")
    return rows


def load_individualized_target_row(
    targets_csv: str | Path,
    *,
    subject: str,
    dataset_roi: str,
) -> dict[str, str]:
    selected_subject = canonical_subject(subject)
    rows = [
        row
        for row in load_individualized_target_rows(targets_csv)
        if canonical_subject(row["subject"]) == selected_subject
        and row["dataset_roi"].strip() == dataset_roi
    ]
    if len(rows) != 1:
        raise ValueError(
            "Expected exactly one individualized target row for "
            f"{selected_subject}/{dataset_roi}; found {len(rows)} in "
            f"{Path(targets_csv)}."
        )
    return rows[0]


def electrode_names_from_target_row(
    row: Mapping[str, str],
) -> tuple[str, str, str, str]:
    pair1 = row["pair1"].split("-")
    pair2 = row["pair2"].split("-")
    if len(pair1) != 2 or len(pair2) != 2 or not all(pair1 + pair2):
        raise ValueError(
            f"Invalid electrode pair(s): pair1={row['pair1']!r}, "
            f"pair2={row['pair2']!r}."
        )
    if len(set(pair1 + pair2)) != 4:
        raise ValueError(
            "Individualized TI montage must use four distinct electrodes: "
            f"pair1={row['pair1']!r}, pair2={row['pair2']!r}."
        )
    return pair1[0], pair1[1], pair2[0], pair2[1]


def validate_individualized_target_table(
    targets_csv: str | Path,
    *,
    subjects: Sequence[str],
    configs: Sequence[CamCanRoiConfig] = CAMCAN_ROI_CONFIGS,
) -> dict[tuple[str, str], dict[str, str]]:
    path = Path(targets_csv)
    requested_subjects = tuple(canonical_subject(value) for value in subjects)
    requested_rois = {config.dataset_prefix: config for config in configs}
    rows = load_individualized_target_rows(path)
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        subject = canonical_subject(row["subject"])
        dataset_roi = row["dataset_roi"].strip()
        key = (subject, dataset_roi)
        if key in indexed:
            raise ValueError(
                f"Duplicate individualized target row for {subject}/{dataset_roi}: "
                f"{path}"
            )
        if subject not in requested_subjects:
            raise ValueError(
                f"Unexpected individualized target subject {subject}: {path}"
            )
        try:
            config = requested_rois[dataset_roi]
        except KeyError as exc:
            raise ValueError(
                f"Unexpected individualized target ROI {dataset_roi!r}: {path}"
            ) from exc
        if row["roi"].strip() != config.targets_roi:
            raise ValueError(
                f"Individualized target {subject}/{dataset_roi} uses targets ROI "
                f"{row['roi']!r}; expected {config.targets_roi!r}."
            )
        if canonical_montage_preset(row["montage_preset"]) != (
            canonical_montage_preset(config.montage_preset)
        ):
            raise ValueError(
                f"Individualized target {subject}/{dataset_roi} uses montage preset "
                f"{row['montage_preset']!r}; expected {config.montage_preset!r}."
            )
        if row["pareto_selection"].strip() != "TI_free.Emin":
            raise ValueError(
                f"Individualized target {subject}/{dataset_roi} is not the "
                "predeclared TI_free.Emin Pareto solution."
            )
        electrode_names_from_target_row(row)
        try:
            e_target = float(row["E_target"])
            stimulated_volume = float(row["stimulated_volume"])
            current1 = float(row["current1"])
            current2 = float(row["current2"])
            configuration = int(row["configuration"])
        except ValueError as exc:
            raise ValueError(
                f"Non-numeric individualized target value for "
                f"{subject}/{dataset_roi}."
            ) from exc
        if not 0.18 <= e_target <= 0.25:
            raise ValueError(
                f"Individualized target field is outside the audited near-0.2 V/m "
                f"range for {subject}/{dataset_roi}: {e_target}"
            )
        if stimulated_volume < 0 or configuration < 0:
            raise ValueError(
                f"Invalid Pareto volume/configuration for {subject}/{dataset_roi}."
            )
        if not 0 < current1 <= 2.0 or not 0 < current2 <= 2.0:
            raise ValueError(
                f"Individualized currents must be in (0, 2] mA for "
                f"{subject}/{dataset_roi}: {current1}, {current2}"
            )
        indexed[key] = row

    expected_keys = {
        (subject, config.dataset_prefix)
        for subject in requested_subjects
        for config in configs
    }
    missing = sorted(expected_keys - set(indexed))
    if missing:
        raise ValueError(
            "Individualized target table is incomplete; missing "
            + ", ".join(f"{subject}/{roi}" for subject, roi in missing)
        )
    extras = sorted(set(indexed) - expected_keys)
    if extras:
        raise ValueError(
            "Individualized target table has unexpected rows: "
            + ", ".join(f"{subject}/{roi}" for subject, roi in extras)
        )
    return indexed


def electrode_names_for_config(
    config: CamCanRoiConfig,
    targets_csv: str | Path,
) -> tuple[str, str, str, str]:
    row = load_target_row(targets_csv, config.targets_roi)
    return electrode_names_from_target_row(row)
