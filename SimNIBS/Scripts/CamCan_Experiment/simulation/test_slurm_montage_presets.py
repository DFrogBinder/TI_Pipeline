from pathlib import Path


HPC_ROOT = Path(__file__).resolve().parents[1] / "HPC_scripts"


def test_my_job_array_passes_explicit_left_m1_montage():
    submitter = (HPC_ROOT / "submit_my_jobArray.sh").read_text()
    array_script = (HPC_ROOT / "my_jobArray.slurm").read_text()

    assert 'MONTAGE_PRESET="${TI_MONTAGE_PRESET:-left-m1}"' in submitter
    assert "TI_MONTAGE_PRESET=\"$MONTAGE_PRESET\"" in submitter
    assert 'MONTAGE_PRESET="${TI_MONTAGE_PRESET:-left-m1}"' in array_script
    assert '--montage-preset "$MONTAGE_PRESET"' in array_script


def test_ti_multi_passes_explicit_left_m1_montage():
    source = (HPC_ROOT / "ti_multi.slurm").read_text()

    assert 'TI_MONTAGE_PRESET="${TI_MONTAGE_PRESET:-left-m1}"' in source
    assert '--montage-preset "$TI_MONTAGE_PRESET"' in source


def test_repair_launcher_mentions_all_csv_presets():
    source = (HPC_ROOT / "launch_simulation_repair.slurm").read_text()

    for preset in [
        "left-dlpfc",
        "left-hippocampus",
        "left-m1",
        "left-pallidum",
        "left-thalamus",
        "right-dlpfc",
        "right-hippocampus",
        "right-m1",
        "right-pallidum",
        "right-thalamus",
    ]:
        assert preset in source
