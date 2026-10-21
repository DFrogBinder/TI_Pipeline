from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


SCRIPTS_ROOT = Path(__file__).resolve().parents[2]


def _save(path: Path, data: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(np.asarray(data), np.eye(4)), path)


def test_numerical_workflow_import_does_not_require_matplotlib() -> None:
    code = """
import builtins

real_import = builtins.__import__
def import_without_matplotlib(name, *args, **kwargs):
    if name == "matplotlib" or name.startswith("matplotlib."):
        raise ModuleNotFoundError("blocked matplotlib for regression test")
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_matplotlib
import simnibs_326_finite_support_audit.workflow
print("workflow-imported")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SCRIPTS_ROOT)

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=SCRIPTS_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == "workflow-imported"


def test_render_request_writes_two_nonblank_panels_and_hashed_receipt(
    tmp_path: Path,
) -> None:
    from simnibs_326_finite_support_audit.cell_render import render_request
    from simnibs_326_finite_support_audit.io_utils import sha256_file

    shape = (15, 15, 15)
    atlas = np.zeros(shape, dtype=np.int16)
    atlas[2:13, 2:13, 2:13] = 17
    atlas_path = tmp_path / "atlas.nii.gz"
    _save(atlas_path, atlas)
    t1_path = tmp_path / "T1.nii.gz"
    _save(t1_path, np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
    raw_path = tmp_path / "raw.nii.gz"
    raw = np.linspace(0.05, 0.5, np.prod(shape), dtype=np.float32).reshape(shape)
    _save(raw_path, raw)
    labels_path = tmp_path / "labels.nii.gz"
    _save(labels_path, np.ones(shape, dtype=np.int16))
    masked_path = tmp_path / "masked.nii.gz"
    _save(masked_path, raw)

    conditions = {}
    for condition in ("remesh", "fixed_mesh"):
        versions = {}
        for version in ("v3", "v4"):
            versions[version] = {
                "t1": str(t1_path),
                "raw_field": str(raw_path),
                "tissue_labels": str(labels_path),
                "masked_field": str(masked_path),
                "cause": "complete_support",
            }
        conditions[condition] = {
            "output_path": str(tmp_path / "figures" / f"{condition}.png"),
            "versions": versions,
        }
    request = {
        "schema_version": 1,
        "target": "left-hippocampus",
        "target_label": "Left Hippocampus",
        "roi": "Left_Hippocampus",
        "atlas_labels": [17],
        "subject": "sub-test",
        "atlas_path": str(atlas_path),
        "conditions": conditions,
    }
    request_path = tmp_path / "render_request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    receipt = render_request(request_path=request_path)

    assert receipt["status"] == "complete"
    assert receipt["request_sha256"] == sha256_file(request_path)
    assert len(receipt["figures"]) == 2
    assert (tmp_path / "render_complete.json").is_file()
    for record in receipt["figures"]:
        image_path = Path(record["path"])
        assert record["sha256"] == sha256_file(image_path)
        with Image.open(image_path) as image:
            assert image.width > 1000
            assert image.height > 700
            assert np.asarray(image).std() > 0
