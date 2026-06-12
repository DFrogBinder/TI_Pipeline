from simulation_runners import repeatability_experiment


def test_ti_montage_parameters_match_hippocampus_montage():
    params = repeatability_experiment._ti_montage_parameters()

    assert params["electrode_size"] == [10, 2]
    assert params["electrode_shape"] == "ellipse"
    assert params["electrode_conductivity"] == 1.4
    assert params["custom_conductivities"] == {
        "WM": 0.126,
        "GM": 0.276,
        "CSF": 1.65,
        "Skull": 0.01,
        "Scalp": 0.465,
        "Eye": 0.5,
        "Muscle": 0.16,
        "Saline": 1.4,
    }
    assert params["montage_right"] == ("F10", 2e-3, "P8", -2e-3)
    assert params["montage_left"] == ("T7", 1.588656e-3, "P7", -1.588656e-3)


def test_all_repeatability_runners_assign_left_montage_currents():
    runner_root = repeatability_experiment.PIPELINE_ROOT / "simulation_runners"
    runner_paths = [
        runner_root / "repeatability_experiment.py",
        runner_root / "TI_runner_multi-core_repeat.py",
        runner_root / "TI_runner_batch_reuse_mesh.py",
        runner_root / "TI_runner_multi-core_resolution-repeat.py",
    ]

    for runner_path in runner_paths:
        source = runner_path.read_text(encoding="utf-8")
        assert "tdcs2.currents" in source
        assert "tdcs2.currents = [montage_left[1], montage_left[3]]" in source


def test_all_repeatability_runners_use_camcan_electrode_properties():
    runner_root = repeatability_experiment.PIPELINE_ROOT / "simulation_runners"
    runner_paths = [
        runner_root / "repeatability_experiment.py",
        runner_root / "TI_runner_multi-core_repeat.py",
        runner_root / "TI_runner_batch_reuse_mesh.py",
        runner_root / "TI_runner_multi-core_resolution-repeat.py",
    ]

    for runner_path in runner_paths:
        source = runner_path.read_text(encoding="utf-8")
        assert "[10, 2]" in source
        assert "electrode_conductivity = 1.4" in source
        assert '"Saline": electrode_conductivity' in source
        assert "tdcs1.cond[2].value = electrode_conductivity" not in source


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


def test_fixed_mesh_cache_force_reset_is_once_per_slurm_array_token(tmp_path, monkeypatch):
    subject = "sub-CC000000"
    source_anat = tmp_path / "source" / subject / "anat"
    source_anat.mkdir(parents=True)
    source_paths = repeatability_experiment.SourceSubjectPaths(
        subject=subject,
        subject_root=source_anat.parent,
        anat_dir=source_anat,
        t1_path=source_anat / f"{subject}_T1w.nii",
        t2_path=source_anat / f"{subject}_T2w.nii",
        seg_path=source_anat / f"{subject}_T1w_ras_1mm_T1andT2_masks.nii",
    )
    for path in (source_paths.t1_path, source_paths.t2_path, source_paths.seg_path):
        path.write_text("placeholder", encoding="utf-8")

    cache_anat = tmp_path / "fixed_mesh" / "mesh_cache" / subject / "anat"
    cache_anat.mkdir(parents=True)
    stale_file = cache_anat / "stale.txt"
    stale_file.write_text("old cache", encoding="utf-8")

    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "10339990")

    repeatability_experiment._prepare_mesh_cache_workspace(
        source_paths,
        mesh_cache_anat_dir=cache_anat,
        overwrite=True,
    )
    assert not stale_file.exists()

    built_sentinel = cache_anat / "built_by_first_repeat.txt"
    built_sentinel.write_text("keep me", encoding="utf-8")

    repeatability_experiment._prepare_mesh_cache_workspace(
        source_paths,
        mesh_cache_anat_dir=cache_anat,
        overwrite=True,
    )
    assert built_sentinel.exists()
