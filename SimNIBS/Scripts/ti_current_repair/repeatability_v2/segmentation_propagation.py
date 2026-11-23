"""Separate full propagation of the five completed CHARM 4.5 segmentations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Callable, Iterable, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plan_charm45_propagation(
    *,
    subject: str,
    segmentation_paths: Iterable[Path],
    correction_profile_sha256: str,
    output_root: Path,
) -> dict[str, object]:
    paths = tuple(segmentation_paths)
    if subject != "sub-CC420071":
        raise ValueError("CHARM 4.5 propagation is scoped to sub-CC420071")
    if len(paths) != 5:
        raise ValueError("CHARM 4.5 propagation requires exactly five segmentations")
    if len(set(paths)) != 5:
        raise ValueError("CHARM 4.5 segmentation paths must be unique")
    if re.fullmatch(r"[0-9a-f]{64}", correction_profile_sha256) is None:
        raise ValueError("Correction profile SHA-256 is invalid")
    if output_root.exists() or output_root.is_symlink():
        raise ValueError("Segmentation propagation output root must be new and isolated")
    sources = []
    for index, path in enumerate(paths, start=1):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Missing or unsafe CHARM 4.5 segmentation: {path}")
        sources.append(
            {
                "segmentation_index": index,
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
        )
    return {
        "schema_version": 2,
        "status": "planned",
        "subject": subject,
        "correction_profile_sha256": correction_profile_sha256,
        "output_root": str(output_root.resolve(strict=False)),
        "sources": sources,
        "targets": ["left_hippocampus", "right_m1"],
        "mesh_tasks": 5,
        "ti_simulations": 10,
        "component_solves": 20,
        "interpretation": "full_segmentation_to_field_propagation",
        "pure_segmentation_effect": False,
    }


def run_charm45_propagation(
    *,
    plan: Mapping[str, object],
    correct_segmentation: Callable[[Path, Path], Path],
    build_mesh: Callable[[Path, Path], Path],
    simulate_and_extract: Callable[[Path, str, Path], Mapping[str, object]],
) -> dict[str, object]:
    if plan.get("status") != "planned" or plan.get("mesh_tasks") != 5:
        raise ValueError("CHARM 4.5 propagation plan is not a complete five-map plan")
    output_root = Path(str(plan["output_root"]))
    if output_root.exists() or output_root.is_symlink():
        raise ValueError("Refusing to overwrite segmentation propagation output root")
    for source in plan["sources"]:
        path = Path(source["path"])
        if _sha256(path) != source["sha256"]:
            raise ValueError(f"CHARM 4.5 source changed after planning: {path}")
    output_root.mkdir(parents=True)
    results: list[dict[str, object]] = []
    for source in plan["sources"]:
        index = int(source["segmentation_index"])
        case_root = output_root / f"segmentation_{index:02d}"
        corrected = correct_segmentation(
            Path(source["path"]), case_root / "corrected_segmentation.nii.gz"
        )
        head = build_mesh(corrected, case_root / "head.msh")
        for target in plan["targets"]:
            result = dict(
                simulate_and_extract(head, str(target), case_root / str(target))
            )
            if result.get("status") != "complete" or int(
                result.get("component_solves", -1)
            ) != 2:
                raise ValueError(
                    f"Incomplete CHARM 4.5 propagation for map {index}, {target}"
                )
            results.append(
                {"segmentation_index": index, "target": target, **result}
            )
    receipt = {
        "schema_version": 2,
        "status": "complete",
        "subject": plan["subject"],
        "mesh_tasks": 5,
        "ti_simulations": len(results),
        "component_solves": sum(int(row["component_solves"]) for row in results),
        "interpretation": "full_segmentation_to_field_propagation",
        "pure_segmentation_effect": False,
        "results": results,
        "source_outputs_modified": False,
    }
    (output_root / "propagation_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt
