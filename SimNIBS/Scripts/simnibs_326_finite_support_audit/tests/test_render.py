from __future__ import annotations

import nibabel as nib
import numpy as np
from PIL import Image

from simnibs_326_finite_support_audit.render import render_paired_diagnostic


def test_render_paired_diagnostic_creates_nonblank_png(tmp_path):
    shape = (7, 7, 7)
    reference = nib.Nifti1Image(np.zeros(shape), np.eye(4))
    t1_path = tmp_path / "t1.nii.gz"
    nib.save(
        nib.Nifti1Image(
            np.arange(np.prod(shape), dtype=np.float32).reshape(shape), np.eye(4)
        ),
        t1_path,
    )
    roi = np.zeros(shape, dtype=bool)
    roi[2:5, 2:5, 2:5] = True

    versions = []
    for name in ("v3", "v4"):
        field = np.full(shape, np.nan)
        field[roi] = np.linspace(0.1, 0.8, np.count_nonzero(roi))
        missing = np.zeros(shape, dtype=bool)
        missing[3, 3, 3] = name == "v3"
        field[missing] = np.nan
        support = {
            "final_finite_voxels": int(np.count_nonzero(roi & np.isfinite(field))),
            "roi_voxels": int(np.count_nonzero(roi)),
        }
        versions.append(
            {
                "name": name,
                "t1_path": str(t1_path),
                "audit": {
                    "support": support,
                    "cause": "brain_mask_exclusion" if name == "v3" else "complete_support",
                    "diagnostic": {
                        "reference_img": reference,
                        "masked_field": field,
                        "roi_mask": roi,
                        "raw_nonfinite_mask": np.zeros(shape, dtype=bool),
                        "mask_excluded_mask": missing,
                        "unexplained_mask": np.zeros(shape, dtype=bool),
                        "inconsistent_mask": np.zeros(shape, dtype=bool),
                    },
                },
            }
        )

    output = tmp_path / "diagnostic.png"
    render_paired_diagnostic(
        output_path=output,
        target_label="left hippocampus",
        subject="sub-test",
        condition="remesh",
        versions=versions,
    )
    with Image.open(output) as image:
        assert image.width > 1000
        assert image.height > 700
        assert np.asarray(image).std() > 0
