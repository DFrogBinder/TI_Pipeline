from pathlib import Path


def test_submit_mesh_qc_helper_exports_safe_log_paths():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "hpc_scripts" / "submit_mesh_qc.sh"

    assert script.exists(), "submit_mesh_qc.sh should exist"

    text = script.read_text(encoding="utf-8")

    assert "--export=" in text
    assert "MESH_QC_LOG_DIR=" in text
    assert "LOG_DIR=" in text
    assert 'RENDERER_CONFIG="gmsh"' in text
    assert "run_mesh_qc.slurm" in text
