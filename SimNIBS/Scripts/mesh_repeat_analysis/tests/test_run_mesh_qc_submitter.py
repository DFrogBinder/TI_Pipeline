from pathlib import Path


def test_submit_mesh_qc_helper_exports_safe_log_paths():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "hpc_scripts" / "submit_mesh_qc.sh"

    assert script.exists(), "submit_mesh_qc.sh should exist"

    text = script.read_text(encoding="utf-8")

    assert "--export=" in text
    assert "--output=" in text
    assert "--error=" in text
    assert "SLURM_ERROR" in text
    assert "MESH_QC_LOG_DIR=" in text
    assert "LOG_DIR=" in text
    assert 'RENDERER_CONFIG="gmsh"' in text
    assert "run_mesh_qc.slurm" in text


def test_mesh_qc_slurm_loads_configured_xvfb_module_for_gmsh():
    repo_root = Path(__file__).resolve().parents[1]
    submitter = repo_root / "hpc_scripts" / "submit_mesh_qc.sh"
    slurm_script = repo_root / "hpc_scripts" / "run_mesh_qc.slurm"

    submitter_text = submitter.read_text(encoding="utf-8")
    slurm_text = slurm_script.read_text(encoding="utf-8")

    assert 'XVFB_MODULE_CONFIG="Xvfb/21.1.6-GCCcore-12.2.0"' in submitter_text
    assert "XVFB_MODULE=" in submitter_text
    assert "XVFB_MODULE=${XVFB_MODULE}" in submitter_text
    assert 'XVFB_MODULE_CONFIG="Xvfb/21.1.6-GCCcore-12.2.0"' in slurm_text
    assert 'if [ -n "${XVFB_MODULE}" ]; then' in slurm_text
    assert 'module load "${XVFB_MODULE}"' in slurm_text
    assert "which Xvfb" in slurm_text
