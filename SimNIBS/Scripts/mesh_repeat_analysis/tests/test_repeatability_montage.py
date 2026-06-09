from simulation_runners import repeatability_experiment


def test_ti_montage_parameters_match_hippocampus_montage():
    params = repeatability_experiment._ti_montage_parameters()

    assert params["electrode_size"] == [10, 1]
    assert params["electrode_shape"] == "ellipse"
    assert params["electrode_conductivity"] == 0.85
    assert params["montage_right"] == ("F10", 2e-3, "P8", -2e-3)
    assert params["montage_left"] == ("T7", 1.588656e-3, "P7", -1.588656e-3)


def test_legacy_ti_runners_do_not_keep_old_hardcoded_montage():
    runner_root = repeatability_experiment.PIPELINE_ROOT / "simulation_runners"
    runner_paths = [
        runner_root / "TI_runner_multi-core_repeat.py",
        runner_root / "TI_runner_batch_reuse_mesh.py",
        runner_root / "TI_runner_multi-core_resolution-repeat.py",
    ]

    for runner_path in runner_paths:
        source = runner_path.read_text(encoding="utf-8")
        assert "Fp2" not in source
        assert "2, 'P8', -2" not in source
        assert '2, "P8", -2' not in source
        assert "F10" in source
        assert "1.588656e-3" in source
