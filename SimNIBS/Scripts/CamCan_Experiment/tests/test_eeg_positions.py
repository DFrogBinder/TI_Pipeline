import numpy as np
import pytest

from post.eeg_positions import read_eeg_positions
from post.metric_extensions import resolve_electrode_centers


def test_reads_native_simnibs_cap_csv(tmp_path):
    path = tmp_path / "EEG10-10_UI_Jurak_2007.csv"
    path.write_text(
        "Fiducial,0,1,2,Nz\n"
        "Electrode,10.5,20.5,30.5,F1\n"
        "Electrode,-1,-2,-3,CP3\n",
        encoding="utf-8",
    )

    positions = read_eeg_positions(path)

    np.testing.assert_allclose(positions["F1"], [10.5, 20.5, 30.5])
    np.testing.assert_allclose(positions["CP3"], [-1, -2, -3])


def test_resolver_requires_all_requested_electrodes(tmp_path):
    subject = "sub-01"
    path = (
        tmp_path
        / subject
        / "anat"
        / f"m2m_{subject}"
        / "eeg_positions"
        / "EEG10-10_UI_Jurak_2007.csv"
    )
    path.parent.mkdir(parents=True)
    path.write_text("Electrode,1,2,3,F1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing requested electrode"):
        resolve_electrode_centers(
            root_dir=str(tmp_path),
            subject=subject,
            roi_name="ctx_lh_G_precentral",
            electrode_csv=None,
            electrode_dataset_dir=None,
            electrode_names=["F1", "F2"],
            eeg_positions_path_template=(
                "{root}/{subject}/anat/m2m_{subject}/eeg_positions/"
                "EEG10-10_UI_Jurak_2007.csv"
            ),
        )
