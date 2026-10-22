"""Scientific contract and timing for the blood-label animation.

This module intentionally has no Manim dependency.  It is the testable source
of truth for the tissue meanings and the order of the explanatory beats.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Tissue:
    """A tissue class used by either the corrected or harmonized schematic."""

    identifier: str
    display_name: str
    colour: str
    label: int | None
    conductive: bool = True


@dataclass(frozen=True, slots=True)
class Beat:
    """One ordered animation beat and its target screen time."""

    identifier: str
    caption: str
    duration_seconds: float


_TISSUES = {
    "white_matter": Tissue("white_matter", "White matter", "#F2E9E4", 1),
    "grey_matter": Tissue("grey_matter", "Grey matter", "#8E7DBE", 2),
    "csf": Tissue("csf", "CSF", "#56B4E9", 3),
    "generic_bone": Tissue("generic_bone", "Generic bone", "#C7A76C", 4),
    "scalp": Tissue("scalp", "Scalp / skin", "#E69F00", 5),
    "external_soft_tissue": Tissue(
        "external_soft_tissue", "External soft tissue", "#E69F00", 5
    ),
    "eyes": Tissue("eyes", "Eyes", "#F0E442", 6),
    "compact_bone": Tissue("compact_bone", "Compact bone", "#9A6F42", 7),
    "spongy_bone": Tissue("spongy_bone", "Spongy bone", "#D8B878", 8),
    "blood": Tissue("blood", "Blood", "#D55E00", 9),
    "muscle": Tissue("muscle", "Muscle", "#CC79A7", 10),
    "internal_air": Tissue(
        "internal_air", "Internal air (excluded)", "#20252C", 11, conductive=False
    ),
}

TISSUES: Mapping[str, Tissue] = MappingProxyType(_TISSUES)

HARMONIZED_TISSUES = (
    "white_matter",
    "grey_matter",
    "csf",
    "generic_bone",
    "external_soft_tissue",
    "eyes",
)

_LOCAL_BLOOD_REPLACEMENTS = MappingProxyType(
    {
        "white_matter": "white_matter",
        "grey_matter": "grey_matter",
        "csf": "csf",
        "external_soft_tissue": "external_soft_tissue",
    }
)


def local_blood_replacement(enclosing_tissue: str) -> str:
    """Return the local tissue that fills a removed blood-labelled region.

    The caller supplies the compartment established by the nested envelopes.
    A blood label is never accepted as its own enclosure because the explicit
    blood class is precisely what this projection removes.
    """

    try:
        return _LOCAL_BLOOD_REPLACEMENTS[enclosing_tissue]
    except KeyError as exc:
        supported = ", ".join(_LOCAL_BLOOD_REPLACEMENTS)
        raise ValueError(
            f"Unsupported enclosing tissue {enclosing_tissue!r}; expected one of: "
            f"{supported}"
        ) from exc


def storyboard() -> tuple[Beat, ...]:
    """Return the fixed explanatory sequence for the silent animation."""

    return (
        Beat(
            "corrected_v4",
            "Corrected v4 anatomy keeps blood, muscle and two bone classes explicit.",
            6.0,
        ),
        Beat(
            "reject_global_mapping",
            "Blood is not globally converted to CSF or scalp.",
            6.0,
        ),
        Beat(
            "build_nested_envelopes",
            "Nested cumulative envelopes define the local tissue compartments.",
            8.0,
        ),
        Beat(
            "local_inheritance",
            "Remove the blood class; each former vessel region inherits locally.",
            8.0,
        ),
        Beat(
            "merge_tissues",
            "Fuse both bone classes and absorb muscle into external soft tissue.",
            5.0,
        ),
        Beat(
            "final_comparison",
            "Blood is not assigned globally. Each former blood region inherits "
            "its local enclosing tissue compartment.",
            7.0,
        ),
    )
