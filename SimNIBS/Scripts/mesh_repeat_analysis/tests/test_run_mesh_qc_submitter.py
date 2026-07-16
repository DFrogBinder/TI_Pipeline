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
    assert 'TISSUE_WALLS_CONFIG="0"' in text
    assert "TISSUE_WALLS=${TISSUE_WALLS}" in text
    assert "run_mesh_qc.slurm" in text


def test_mesh_qc_slurm_exposes_opt_in_tissue_walls():
    repo_root = Path(__file__).resolve().parents[1]
    slurm_script = repo_root / "hpc_scripts" / "run_mesh_qc.slurm"
    text = slurm_script.read_text(encoding="utf-8")

    assert 'TISSUE_WALLS_CONFIG="0"' in text
    assert 'if [ "${TISSUE_WALLS}" = "1" ]; then' in text
    assert "PYTHON_CMD+=(--tissue-walls)" in text


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
    assert 'if [ -n "${XVFB_MODULE_EFFECTIVE}" ]; then' in slurm_text
    assert 'module load "${XVFB_MODULE_EFFECTIVE}"' in slurm_text
    assert "which Xvfb" in slurm_text


def test_mesh_qc_slurm_fails_on_module_load_errors_before_loading_modules():
    repo_root = Path(__file__).resolve().parents[1]
    slurm_script = repo_root / "hpc_scripts" / "run_mesh_qc.slurm"
    text = slurm_script.read_text(encoding="utf-8")

    assert text.index("set -euo pipefail") < text.index("module purge")


def test_submitter_exports_configurable_gmsh_module_and_binary_override():
    repo_root = Path(__file__).resolve().parents[1]
    submitter = repo_root / "hpc_scripts" / "submit_mesh_qc.sh"
    slurm_script = repo_root / "hpc_scripts" / "run_mesh_qc.slurm"

    submitter_text = submitter.read_text(encoding="utf-8")
    slurm_text = slurm_script.read_text(encoding="utf-8")

    assert "GMSH_MODULE_CONFIG" in submitter_text
    assert "MESH_QC_GMSH_BIN_CONFIG" in submitter_text
    assert "GMSH_MODULE=${GMSH_MODULE}" in submitter_text
    assert "MESH_QC_GMSH_BIN=${MESH_QC_GMSH_BIN}" in submitter_text
    assert 'module load "${GMSH_MODULE_EFFECTIVE}"' in slurm_text
    assert "MESH_QC_GMSH_BIN" in slurm_text
    assert '"${GMSH_CHECK_BIN}" -version' in slurm_text


def test_submitter_exports_configurable_imagemagick_module_for_mosaics():
    repo_root = Path(__file__).resolve().parents[1]
    submitter = repo_root / "hpc_scripts" / "submit_mesh_qc.sh"
    slurm_script = repo_root / "hpc_scripts" / "run_mesh_qc.slurm"

    submitter_text = submitter.read_text(encoding="utf-8")
    slurm_text = slurm_script.read_text(encoding="utf-8")

    assert "IMAGEMAGICK_MODULE_CONFIG" in submitter_text
    assert "IMAGEMAGICK_MODULE=${IMAGEMAGICK_MODULE}" in submitter_text
    assert 'echo "[INFO] ImageMagick module:' in submitter_text
    assert "IMAGEMAGICK_MODULE_CONFIG" in slurm_text
    assert 'module load "${IMAGEMAGICK_MODULE}"' in slurm_text
    assert "which montage" in slurm_text


def test_submitter_exports_label_regex_overrides():
    repo_root = Path(__file__).resolve().parents[1]
    submitter = repo_root / "hpc_scripts" / "submit_mesh_qc.sh"
    slurm_script = repo_root / "hpc_scripts" / "run_mesh_qc.slurm"

    submitter_text = submitter.read_text(encoding="utf-8")
    slurm_text = slurm_script.read_text(encoding="utf-8")

    assert "ROI_REGEX_CONFIG" in submitter_text
    assert "SUBJECT_REGEX_CONFIG" in submitter_text
    assert "REPEAT_REGEX_CONFIG" in submitter_text
    assert "ROI_REGEX=${ROI_REGEX}" in submitter_text
    assert "SUBJECT_REGEX=${SUBJECT_REGEX}" in submitter_text
    assert "REPEAT_REGEX=${REPEAT_REGEX}" in submitter_text
    assert 'PYTHON_CMD+=(--roi-regex "${ROI_REGEX}")' in slurm_text
    assert 'PYTHON_CMD+=(--subject-regex "${SUBJECT_REGEX}")' in slurm_text
    assert 'PYTHON_CMD+=(--repeat-regex "${REPEAT_REGEX}")' in slurm_text


def test_submitter_exports_configurable_simnibs_module():
    repo_root = Path(__file__).resolve().parents[1]
    submitter = repo_root / "hpc_scripts" / "submit_mesh_qc.sh"

    text = submitter.read_text(encoding="utf-8")

    assert "SIMNIBS_MODULE_CONFIG" in text
    assert "SIMNIBS_MODULE=" in text
    assert "SIMNIBS_MODULE=${SIMNIBS_MODULE}" in text
    assert 'echo "[INFO] SimNIBS module:' in text


def test_render_gmsh_slurm_skips_simnibs_module_to_avoid_module_conflicts():
    repo_root = Path(__file__).resolve().parents[1]
    slurm_script = repo_root / "hpc_scripts" / "run_mesh_qc.slurm"

    text = slurm_script.read_text(encoding="utf-8")

    assert "SIMNIBS_MODULE_EFFECTIVE" in text
    assert '[ "${MESH_QC_STAGE}" = "render" ]' in text
    assert '[ "${RENDERER}" = "gmsh" ]' in text
    assert 'SIMNIBS_MODULE_EFFECTIVE=""' in text
    assert 'module load "${SIMNIBS_MODULE_EFFECTIVE}"' in text
    assert 'module load "${SIMNIBS_MODULE}"' not in text
    assert text.index("MESH_QC_STAGE=") < text.index("module purge")


def test_qc_only_slurm_skips_render_modules_to_avoid_module_conflicts():
    repo_root = Path(__file__).resolve().parents[1]
    slurm_script = repo_root / "hpc_scripts" / "run_mesh_qc.slurm"

    text = slurm_script.read_text(encoding="utf-8")

    assert "GMSH_MODULE_EFFECTIVE" in text
    assert "XVFB_MODULE_EFFECTIVE" in text
    assert '[ "${MESH_QC_STAGE}" = "qc" ]' in text
    assert 'GMSH_MODULE_EFFECTIVE=""' in text
    assert 'XVFB_MODULE_EFFECTIVE=""' in text
    assert 'module load "${GMSH_MODULE_EFFECTIVE}"' in text
    assert 'module load "${XVFB_MODULE_EFFECTIVE}"' in text
    assert "NEEDS_GMSH_CHECK" in text
    assert '[ "${MESH_QC_STAGE}" != "qc" ]' in text
