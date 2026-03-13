"""
FastSurfer ROI registry and alias resolution helpers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence, Set


FASTSURFER_DKT_LABELS: Dict[int, str] = {
    # Subcortical
    0: "Unknown",
    2: "Left-Cerebral-White-Matter",
    3: "Left-Cerebral-Cortex",
    4: "Left-Lateral-Ventricle",
    5: "Left-Inf-Lat-Vent",
    7: "Left-Cerebellum-White-Matter",
    8: "Left-Cerebellum-Cortex",
    10: "Left-Thalamus-Proper",
    11: "Left-Caudate",
    12: "Left-Putamen",
    13: "Left-Pallidum",
    14: "3rd-Ventricle",
    15: "4th-Ventricle",
    16: "Brain-Stem",
    17: "Left-Hippocampus",
    18: "Left-Amygdala",
    24: "CSF",
    26: "Left-Accumbens-area",
    28: "Left-VentralDC",
    30: "Left-vessel",
    31: "Left-choroid-plexus",
    41: "Right-Cerebral-White-Matter",
    42: "Right-Cerebral-Cortex",
    43: "Right-Lateral-Ventricle",
    44: "Right-Inf-Lat-Vent",
    46: "Right-Cerebellum-White-Matter",
    47: "Right-Cerebellum-Cortex",
    49: "Right-Thalamus-Proper",
    50: "Right-Caudate",
    51: "Right-Putamen",
    52: "Right-Pallidum",
    53: "Right-Hippocampus",
    54: "Right-Amygdala",
    58: "Right-Accumbens-area",
    60: "Right-VentralDC",
    62: "Right-vessel",
    63: "Right-choroid-plexus",
    # Left cortex (DKT)
    1000: "ctx-lh-bankssts",
    1001: "ctx-lh-caudalanteriorcingulate",
    1002: "ctx-lh-caudalmiddlefrontal",
    1003: "ctx-lh-cuneus",
    1004: "ctx-lh-entorhinal",
    1005: "ctx-lh-fusiform",
    1006: "ctx-lh-inferiorparietal",
    1007: "ctx-lh-inferiortemporal",
    1008: "ctx-lh-isthmuscingulate",
    1009: "ctx-lh-lateraloccipital",
    1010: "ctx-lh-lateralorbitofrontal",
    1011: "ctx-lh-lingual",
    1012: "ctx-lh-medialorbitofrontal",
    1013: "ctx-lh-middletemporal",
    1014: "ctx-lh-parahippocampal",
    1015: "ctx-lh-paracentral",
    1016: "ctx-lh-parsopercularis",
    1017: "ctx-lh-parsorbitalis",
    1018: "ctx-lh-parstriangularis",
    1019: "ctx-lh-pericalcarine",
    1020: "ctx-lh-postcentral",
    1021: "ctx-lh-posteriorcingulate",
    1022: "ctx-lh-precentral",
    1023: "ctx-lh-precuneus",
    1024: "ctx-lh-rostralanteriorcingulate",
    1025: "ctx-lh-rostralmiddlefrontal",
    1026: "ctx-lh-superiorfrontal",
    1027: "ctx-lh-superiorparietal",
    1028: "ctx-lh-superiortemporal",
    1029: "ctx-lh-supramarginal",
    1030: "ctx-lh-frontalpole",
    1031: "ctx-lh-temporalpole",
    1032: "ctx-lh-transversetemporal",
    1033: "ctx-lh-insula",
    # Right cortex (DKT)
    2000: "ctx-rh-bankssts",
    2001: "ctx-rh-caudalanteriorcingulate",
    2002: "ctx-rh-caudalmiddlefrontal",
    2003: "ctx-rh-cuneus",
    2004: "ctx-rh-entorhinal",
    2005: "ctx-rh-fusiform",
    2006: "ctx-rh-inferiorparietal",
    2007: "ctx-rh-inferiortemporal",
    2008: "ctx-rh-isthmuscingulate",
    2009: "ctx-rh-lateraloccipital",
    2010: "ctx-rh-lateralorbitofrontal",
    2011: "ctx-rh-lingual",
    2012: "ctx-rh-medialorbitofrontal",
    2013: "ctx-rh-middletemporal",
    2014: "ctx-rh-parahippocampal",
    2015: "ctx-rh-paracentral",
    2016: "ctx-rh-parsopercularis",
    2017: "ctx-rh-parsorbitalis",
    2018: "ctx-rh-parstriangularis",
    2019: "ctx-rh-pericalcarine",
    2020: "ctx-rh-postcentral",
    2021: "ctx-rh-posteriorcingulate",
    2022: "ctx-rh-precentral",
    2023: "ctx-rh-precuneus",
    2024: "ctx-rh-rostralanteriorcingulate",
    2025: "ctx-rh-rostralmiddlefrontal",
    2026: "ctx-rh-superiorfrontal",
    2027: "ctx-rh-superiorparietal",
    2028: "ctx-rh-superiortemporal",
    2029: "ctx-rh-supramarginal",
    2030: "ctx-rh-frontalpole",
    2031: "ctx-rh-temporalpole",
    2032: "ctx-rh-transversetemporal",
    2033: "ctx-rh-insula",
}

_HEMISPHERE_ALIASES = {
    "left": ("left", "lh", "l"),
    "right": ("right", "rigth", "rh", "r"),
}

_SPECIAL_REGION_ALIASES = {
    "precentral": {
        "m1",
        "motor_cortex",
        "primary_motor_cortex",
        "primary_motor",
    },
}


def _snake_case(value: str) -> str:
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower())
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def _region_aliases(region_name: str) -> Set[str]:
    aliases = {region_name}
    aliases.update(_SPECIAL_REGION_ALIASES.get(region_name, set()))
    return aliases


def _canonical_parts(canonical_name: str) -> tuple[str | None, str]:
    if canonical_name.startswith("ctx-lh-"):
        return "left", _snake_case(canonical_name.removeprefix("ctx-lh-"))
    if canonical_name.startswith("ctx-rh-"):
        return "right", _snake_case(canonical_name.removeprefix("ctx-rh-"))
    if canonical_name.startswith("Left-"):
        return "left", _snake_case(canonical_name.removeprefix("Left-"))
    if canonical_name.startswith("Right-"):
        return "right", _snake_case(canonical_name.removeprefix("Right-"))
    return None, _snake_case(canonical_name)


def _generate_aliases(canonical_name: str) -> Set[str]:
    aliases = {_snake_case(canonical_name)}
    hemisphere, region_name = _canonical_parts(canonical_name)
    region_aliases = _region_aliases(region_name)

    aliases.update(region_aliases)
    if hemisphere is None:
        return aliases

    for hemi_alias in _HEMISPHERE_ALIASES[hemisphere]:
        for region_alias in region_aliases:
            aliases.add(f"{hemi_alias}_{region_alias}")
            aliases.add(f"{region_alias}_{hemi_alias}")
    return aliases


FASTSURFER_ROI_ALIASES: Dict[str, tuple[str, ...]] = {
    canonical_name: tuple(sorted(_generate_aliases(canonical_name)))
    for canonical_name in sorted(set(FASTSURFER_DKT_LABELS.values()), key=str.lower)
}

_fastsurfer_alias_index: Dict[str, Set[str]] = {}
for canonical_name, aliases in FASTSURFER_ROI_ALIASES.items():
    for alias in aliases:
        _fastsurfer_alias_index.setdefault(alias, set()).add(canonical_name)
FASTSURFER_ALIAS_INDEX: Dict[str, tuple[str, ...]] = {
    alias: tuple(sorted(canonical_names))
    for alias, canonical_names in _fastsurfer_alias_index.items()
}


@dataclass(frozen=True)
class FastsurferRoiMatch:
    canonical_name: str
    matched_alias: str


def _normalize_tokens(value: str | Path) -> tuple[str, ...]:
    normalized = _snake_case(str(Path(value).name))
    return tuple(token for token in normalized.split("_") if token)


def _contains_alias(tokens: Sequence[str], alias: str) -> bool:
    alias_tokens = tuple(token for token in alias.split("_") if token)
    if not alias_tokens or len(alias_tokens) > len(tokens):
        return False
    width = len(alias_tokens)
    return any(tuple(tokens[idx : idx + width]) == alias_tokens for idx in range(len(tokens) - width + 1))


def _format_ambiguity(alias: str, canonical_names: Iterable[str]) -> str:
    choices = ", ".join(sorted(canonical_names))
    return f"Alias '{alias}' is ambiguous. Use one of: {choices}."


def resolve_fastsurfer_roi_name(name: str) -> FastsurferRoiMatch:
    alias = _snake_case(name)
    canonical_names = FASTSURFER_ALIAS_INDEX.get(alias)
    if not canonical_names:
        raise ValueError(f"Unknown FastSurfer ROI alias '{name}'.")
    if len(canonical_names) != 1:
        raise ValueError(_format_ambiguity(alias, canonical_names))
    return FastsurferRoiMatch(canonical_name=canonical_names[0], matched_alias=alias)


def match_fastsurfer_roi_from_directory(directory_name: str | Path) -> FastsurferRoiMatch:
    tokens = _normalize_tokens(directory_name)
    matches: list[tuple[int, int, str, tuple[str, ...]]] = []
    for alias, canonical_names in FASTSURFER_ALIAS_INDEX.items():
        if _contains_alias(tokens, alias):
            alias_width = len(alias.split("_"))
            matches.append((alias_width, len(alias), alias, canonical_names))

    if not matches:
        sample_aliases = ", ".join(sorted(FASTSURFER_ALIAS_INDEX)[:5])
        raise ValueError(
            f"Could not match any FastSurfer ROI alias in '{Path(directory_name).name}'. "
            f"Examples: {sample_aliases}."
        )

    matches.sort(reverse=True)
    _, _, alias, canonical_names = matches[0]
    if len(canonical_names) != 1:
        raise ValueError(_format_ambiguity(alias, canonical_names))
    return FastsurferRoiMatch(canonical_name=canonical_names[0], matched_alias=alias)


def find_aliases_for_canonical_name(canonical_name: str) -> tuple[str, ...]:
    try:
        return FASTSURFER_ROI_ALIASES[canonical_name]
    except KeyError as exc:
        raise ValueError(f"Unknown FastSurfer ROI '{canonical_name}'.") from exc
