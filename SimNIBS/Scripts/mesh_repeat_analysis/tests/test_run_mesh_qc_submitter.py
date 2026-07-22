import os
import subprocess
import sys
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
    assert 'MESH_QC_PYTHON_CONFIG="python"' in text
    assert "MESH_QC_PYTHON=${MESH_QC_PYTHON}" in text
    assert "run_mesh_qc.slurm" in text


def test_submit_mesh_qc_has_full_collected_charm_tissue_profile():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "hpc_scripts" / "submit_mesh_qc.sh"
    text = script.read_text(encoding="utf-8")

    assert "collected-charm-tissues)" in text
    assert "charm_segmentation_meshes_474/subjects" in text
    assert "charm_segmentation_meshes_474_tissue_views" in text
    assert 'CPUS_PER_TASK_CONFIG="16"' in text
    assert 'MEMORY_CONFIG="64G"' in text
    assert 'TIME_LIMIT_CONFIG="08:00:00"' in text
    assert 'MESH_QC_STAGE_CONFIG="tissue"' in text
    assert 'TISSUE_WALLS_CONFIG="1"' in text
    assert 'ROI_WALLS_CONFIG="0"' in text
    assert 'WORKERS_CONFIG="16"' in text
    assert 'SIMNIBS_MODULE_CONFIG="none"' in text
    assert 'GMSH_MODULE_CONFIG="gmsh/4.11.1-foss-2022b"' in text
    assert 'MESH_QC_GMSH_TIMEOUT_SECONDS_CONFIG="900"' in text
    assert 'XVFB_MODULE_CONFIG="Xvfb/21.1.6-GCCcore-12.2.0"' in text
    assert "${HOME}/.conda/envs/ti-post/bin/python" in text
    assert "--preflight" in text
    assert "--expected-meshes" in text
    assert "full collected cohort; tissue-only; no smoke or reduced tasks" in text
    assert "Existing resumable outputs" in text
    assert "ras_anatomical_orthographic_compact_top_v3" in text
    assert "MESH_QC_EXPECTED_MESHES=${MESH_QC_EXPECTED_MESHES}" in text
    assert "MESH_QC_EXPECTED_SUBJECTS=${MESH_QC_EXPECTED_SUBJECTS}" in text
    assert "MESH_QC_GMSH_TIMEOUT_SECONDS=${MESH_QC_GMSH_TIMEOUT_SECONDS}" in text
    assert "protects MESH_QC_GMSH_TIMEOUT_SECONDS=900" in text
    assert "Protected Gmsh timeout:" in text


def test_submit_mesh_qc_supports_resumable_render_array_and_afterany_collector():
    repo_root = Path(__file__).resolve().parents[1]
    submitter = repo_root / "hpc_scripts" / "submit_mesh_qc.sh"
    runner = repo_root / "hpc_scripts" / "run_mesh_qc.slurm"
    submitter_text = submitter.read_text(encoding="utf-8")
    runner_text = runner.read_text(encoding="utf-8")

    assert "--array-shards" in submitter_text
    assert "MESH_QC_ARRAY_CONCURRENCY" in submitter_text
    assert "MESH_QC_STAGE=tissue-shard" in submitter_text
    assert "MESH_QC_STAGE=tissue-collect" in submitter_text
    assert '--dependency="afterany:${RENDER_JOB_ID}"' in submitter_text
    assert "render-%A_%a.out" in submitter_text
    assert "collector-%j.out" in submitter_text
    assert "Existing compatible tissue tiles are resumed in place" in submitter_text
    assert "tissue-shard)" in runner_text
    assert "tissue-collect)" in runner_text
    assert "--tissue-shard-count" in runner_text
    assert "--tissue-shard-index" in runner_text
    assert "--tissue-collect-only" in runner_text
    assert 'MESH_QC_GMSH_TIMEOUT_SECONDS_CONFIG="900"' in runner_text


def test_accelerated_submitter_emits_render_array_and_dependent_collector(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    submitter = repo_root / "hpc_scripts" / "submit_mesh_qc.sh"
    mesh_root = tmp_path / "subjects"
    mesh_path = mesh_root / "sub-CC1" / "anat" / "m2m_sub-CC1" / "sub-CC1.msh"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text("$MeshFormat\n", encoding="utf-8")
    output_root = tmp_path / "walls"
    log_root = tmp_path / "logs"
    command_log = tmp_path / "sbatch_commands.txt"
    fake_sbatch = tmp_path / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/bash\n"
        f"printf '%s\\n' \"$*\" >> {command_log}\n"
        "if [[ \"$*\" == *'--dependency=afterany:'* ]]; then\n"
        "  echo 222222\n"
        "else\n"
        "  echo 111111\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "MESH_QC_ROOT": str(mesh_root),
            "MESH_QC_OUT": str(output_root),
            "MESH_QC_LOG_DIR": str(log_root),
            "MESH_QC_PYTHON": sys.executable,
            "SBATCH_BIN": str(fake_sbatch),
            "JOB_NAME": "accelerated_test",
        }
    )
    completed = subprocess.run(
        [
            "bash",
            str(submitter),
            "--profile",
            "collected-charm-tissues",
            "--expected-meshes",
            "1",
            "--array-shards",
            "2",
        ],
        cwd=repo_root.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Submitted tissue-render array job: 111111" in completed.stdout
    assert "Submitted afterany collector job: 222222" in completed.stdout
    commands = command_log.read_text(encoding="utf-8").splitlines()
    assert len(commands) == 2
    assert "--array=0-1%2" in commands[0]
    assert "MESH_QC_STAGE=tissue-shard" in commands[0]
    assert "--dependency=afterany:111111" in commands[1]
    assert "MESH_QC_STAGE=tissue-collect" in commands[1]


def test_mesh_qc_slurm_exposes_opt_in_tissue_walls():
    repo_root = Path(__file__).resolve().parents[1]
    slurm_script = repo_root / "hpc_scripts" / "run_mesh_qc.slurm"
    text = slurm_script.read_text(encoding="utf-8")

    assert "#SBATCH --time=08:00:00" in text
    assert 'TISSUE_WALLS_CONFIG="0"' in text
    assert 'if [ "${TISSUE_WALLS}" = "1" ]; then' in text
    assert "PYTHON_CMD+=(--tissue-walls)" in text
    assert 'MESH_QC_PYTHON_CONFIG="python"' in text
    assert '"${MESH_QC_PYTHON}"' in text
    assert '"${MESH_QC_PYTHON}" -E -c' in text
    assert '"${MESH_QC_PYTHON}"\n    -E\n    -u' in text
    assert "tissue)" in text
    assert "PYTHON_CMD+=(--tissue-only)" in text
    assert 'MESH_QC_EXPECTED_MESHES_CONFIG=""' in text
    assert 'MESH_QC_EXPECTED_SUBJECTS_CONFIG=""' in text
    assert 'MESH_QC_GMSH_TIMEOUT_SECONDS_CONFIG="900"' in text
    assert "MESH_QC_GMSH_TIMEOUT_SECONDS must be a positive integer" in text
    assert "Mesh-count gate failed" in text
    assert "Subject-count gate failed" in text


def test_left_hippocampus_tissue_wall_pilot_is_self_contained_slurm_job():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "hpc_scripts" / "run_left_hippocampus_tissue_walls_pilot.slurm"

    assert script.exists(), "the Left Hippocampus pilot launcher should exist"
    text = script.read_text(encoding="utf-8")

    assert "#SBATCH --partition=sheffield" in text
    assert "#SBATCH --cpus-per-task=4" in text
    assert "#SBATCH --mem=64G" in text
    assert "#SBATCH --time=08:00:00" in text
    assert "Left_Hippocampus_Runs/Left_Hippocampus_Data_01" in text
    assert 'MESH_QC_EXPECTED_MESHES_CONFIG="175"' in text
    assert 'MESH_QC_STAGE_CONFIG="full"' in text
    assert 'TISSUE_WALLS_CONFIG="1"' in text
    assert 'ROI_WALLS_CONFIG="0"' in text
    assert 'WORKERS_CONFIG="1"' in text
    assert 'RENDERER_CONFIG="gmsh"' in text
    assert 'SIMNIBS_MODULE_CONFIG="none"' in text
    assert 'GMSH_MODULE_CONFIG="gmsh/4.11.1-foss-2022b"' in text
    assert 'XVFB_MODULE_CONFIG="Xvfb/21.1.6-GCCcore-12.2.0"' in text
    assert "${HOME}/.conda/envs/ti-post/bin/python" in text
    assert 'PIPELINE_DIR_CONFIG="${HOME}/Repos/TI_Pipeline/SimNIBS/Scripts"' in text
    assert 'dirname "${BASH_SOURCE[0]}"' not in text
    assert 'export MESH_QC_ROOT="${MESH_QC_ROOT_CONFIG}"' in text
    assert "LEFT_HIPPOCAMPUS_PILOT_STAGE" in text
    assert "discover_meshes" in text
    assert "requirements-tissue-walls.txt" in text
    assert "import meshio" in text
    assert "iter_tissue_surface_arrays" in text
    assert "TISSUE_PREFLIGHT" in text
    assert 'exec bash "${GENERIC_RUNNER}"' in text


def test_left_hippocampus_full_tissue_array_covers_all_repeats_in_parallel():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "hpc_scripts" / "run_left_hippocampus_tissue_walls_array.slurm"

    assert script.exists(), (
        "the full Left Hippocampus tissue-wall array launcher should exist"
    )
    text = script.read_text(encoding="utf-8")

    assert "#SBATCH --partition=sheffield" in text
    assert "#SBATCH --cpus-per-task=16" in text
    assert "#SBATCH --mem=64G" in text
    assert "#SBATCH --time=08:00:00" in text
    assert "#SBATCH --array=1-10%10" in text
    assert 'EXPECTED_REPEATS_CONFIG="10"' in text
    assert 'EXPECTED_SUBJECTS_PER_REPEAT_CONFIG="175"' in text
    assert 'EXPECTED_TOTAL_MESHES_CONFIG="1750"' in text
    assert 'EXPECTED_TOTAL_TILES_CONFIG="33250"' in text
    assert 'EXPECTED_TOTAL_WALLS_CONFIG="190"' in text
    assert 'MESH_QC_STAGE_CONFIG="tissue"' in text
    assert 'WORKERS_CONFIG="16"' in text
    assert 'TISSUE_WALLS="1"' in text
    assert 'ROI_WALLS="0"' in text
    assert 'RENDERER_CONFIG="gmsh"' in text
    assert "Left_Hippocampus_tissue_front_back_orthographic_all_repeats" in text
    assert (
        "RAS +Y face / -Y posterior; compact bone also +Z superior; orthographic"
        in text
    )
    assert 'SIMNIBS_MODULE_CONFIG="none"' in text
    assert 'GMSH_MODULE_CONFIG="gmsh/4.11.1-foss-2022b"' in text
    assert 'XVFB_MODULE_CONFIG="Xvfb/21.1.6-GCCcore-12.2.0"' in text
    assert 'PIPELINE_DIR_CONFIG="${HOME}/Repos/TI_Pipeline/SimNIBS/Scripts"' in text
    assert 'REPEAT_INDEX="${SLURM_ARRAY_TASK_ID:-}"' in text
    assert 'REPEAT_NAME="Left_Hippocampus_Data_${REPEAT_PADDED}"' in text
    assert "Execution scope:      full" in text
    assert "Whole-mesh renders:   disabled" in text
    assert 'exec bash "${GENERIC_RUNNER}"' in text


def test_tissue_wall_requirements_pin_locally_tested_meshio():
    repo_root = Path(__file__).resolve().parents[1]
    requirements = repo_root / "post" / "mesh_qc" / "requirements-tissue-walls.txt"

    assert requirements.exists()
    assert "meshio==5.3.5" in requirements.read_text(encoding="utf-8")


def test_gmsh_tissue_visibility_smoke_uses_one_mesh_load_for_both_views():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "hpc_scripts" / "run_gmsh_tissue_visibility_smoke.slurm"

    assert script.exists()
    text = script.read_text(encoding="utf-8")

    assert "#SBATCH --time=08:00:00" in text
    assert 'GMSH_MODULE="gmsh/4.11.1-foss-2022b"' in text
    assert 'XVFB_MODULE="Xvfb/21.1.6-GCCcore-12.2.0"' in text
    assert text.count('Merge "${MESH}";') == 1
    assert "Recursive Show { Physical Volume{${TISSUE_TAG}}; }" in text
    assert "General.RotationZ = 0;" in text
    assert "General.RotationZ = 180;" in text
    assert 'Print "${FRONT_PNG}";' in text
    assert 'Print "${BACK_PNG}";' in text
    assert "timeout 600" not in text
    assert "requirements-tissue-walls" not in text


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
