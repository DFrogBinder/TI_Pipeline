#!/usr/bin/env python3
"""Validate one isolated MNI152 ROI baseline and its scientific provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.target_montages import resolve_montage_preset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, *, min_bytes: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required MNI152 output is missing: {path}")
    size = path.stat().st_size
    if size < min_bytes:
        raise ValueError(
            f"MNI152 output is unexpectedly small: {path} "
            f"({size} < {min_bytes} bytes)."
        )
    return {"path": str(path), "size_bytes": size}


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}.")


def require_close(actual: Any, expected: float, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}.")


def validate(args: argparse.Namespace) -> dict[str, Any]:
    parent = args.output_parent.expanduser().resolve(strict=True)
    baseline_root = parent / args.output_subject
    simnibs_root = baseline_root / "anat" / "SimNIBS"
    output_dir = simnibs_root / "Output" / "MNI152"
    provenance_path = simnibs_root / "mni_baseline_provenance.json"
    ti_brain_path = simnibs_root / "ti_brain_only.nii.gz"
    ti_mesh_path = output_dir / "TI.msh"
    tdcs_paths = [
        output_dir / "MNI152_TDCS_1_scalar.msh",
        output_dir / "MNI152_TDCS_2_scalar.msh",
    ]

    files = {
        "provenance": require_file(provenance_path, min_bytes=500),
        "ti_brain_only": require_file(ti_brain_path, min_bytes=100_000),
        "ti_mesh": require_file(ti_mesh_path, min_bytes=1_000_000),
        "tdcs_1_scalar": require_file(tdcs_paths[0], min_bytes=1_000_000),
        "tdcs_2_scalar": require_file(tdcs_paths[1], min_bytes=1_000_000),
    }
    volume_base = sorted((output_dir / "Volume_Base").glob("TI_Volumetric_*.nii*"))
    volume_labels = sorted(
        (output_dir / "Volume_Labels").glob("TI_Volumetric_*.nii*")
    )
    if not volume_base or not volume_labels:
        raise FileNotFoundError(
            f"Missing TI volume or label export below {output_dir}."
        )
    files["ti_volume"] = require_file(volume_base[0], min_bytes=100_000)
    files["ti_labels"] = require_file(volume_labels[0], min_bytes=100_000)

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    require_equal(provenance.get("status"), "complete", "provenance status")
    require_equal(
        provenance.get("output_subject"),
        args.output_subject,
        "output subject",
    )
    require_equal(provenance.get("preset"), args.preset, "montage preset")
    require_equal(
        provenance.get("software", {}).get("simnibs_version"),
        args.expected_simnibs_version,
        "SimNIBS version",
    )
    require_equal(
        provenance.get("inputs", {}).get("mni_mesh_sha256"),
        args.expected_mni_mesh_sha256,
        "MNI152 mesh SHA-256",
    )
    require_equal(
        provenance.get("inputs", {}).get("reference_t1_sha256"),
        args.expected_reference_t1_sha256,
        "reference T1 SHA-256",
    )
    require_equal(
        provenance.get("inputs", {}).get("targets_csv_sha256"),
        args.expected_targets_sha256,
        "targets.csv SHA-256",
    )

    montage = resolve_montage_preset(args.preset)
    stimulation = provenance["stimulation"]
    for pair_name, pair in (
        ("pair1", montage.pair1),
        ("pair2", montage.pair2),
    ):
        recorded = stimulation[pair_name]
        require_equal(recorded.get("anode"), pair.anode, f"{pair_name} anode")
        require_equal(
            recorded.get("cathode"),
            pair.cathode,
            f"{pair_name} cathode",
        )
        require_close(
            recorded.get("current_a"),
            pair.current_amp,
            f"{pair_name} current",
        )

    img = nib.load(str(ti_brain_path))
    reference_img = nib.load(str(args.reference_t1.expanduser().resolve(strict=True)))
    if img.shape != reference_img.shape:
        raise ValueError(
            f"TI/reference shape mismatch: {img.shape} != {reference_img.shape}."
        )
    if not np.allclose(img.affine, reference_img.affine, atol=1e-3):
        raise ValueError("TI/reference affine mismatch.")
    data = img.get_fdata(dtype=np.float32)
    finite = np.isfinite(data)
    if not finite.any():
        raise ValueError("Brain-only TI NIfTI contains no finite voxels.")
    finite_values = data[finite]
    if np.any(finite_values < 0):
        raise ValueError("Brain-only TI NIfTI contains negative values.")

    recorded_ti_sha = (
        provenance.get("outputs", {})
        .get("ti_brain_only", {})
        .get("sha256")
    )
    actual_ti_sha = sha256_file(ti_brain_path)
    require_equal(actual_ti_sha, recorded_ti_sha, "brain-only TI SHA-256")

    result = {
        "validation_schema_version": 1,
        "status": "complete",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_parent": str(parent),
        "output_subject": args.output_subject,
        "preset": args.preset,
        "simnibs_version": args.expected_simnibs_version,
        "mni_mesh_sha256": args.expected_mni_mesh_sha256,
        "reference_t1_sha256": args.expected_reference_t1_sha256,
        "targets_csv_sha256": args.expected_targets_sha256,
        "files": files,
        "nifti": {
            "shape": list(img.shape),
            "finite_voxels": int(finite.sum()),
            "minimum_v_per_m": float(np.min(finite_values)),
            "maximum_v_per_m": float(np.max(finite_values)),
            "mean_v_per_m": float(np.mean(finite_values, dtype=np.float64)),
            "sha256": actual_ti_sha,
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.summary.with_suffix(args.summary.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.summary)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--output-subject", required=True)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--reference-t1", type=Path, required=True)
    parser.add_argument("--expected-simnibs-version", required=True)
    parser.add_argument("--expected-mni-mesh-sha256", required=True)
    parser.add_argument("--expected-reference-t1-sha256", required=True)
    parser.add_argument("--expected-targets-sha256", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
