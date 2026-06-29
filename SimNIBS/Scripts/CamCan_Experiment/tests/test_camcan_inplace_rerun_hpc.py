import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_multi_core_runner_exposes_reuse_existing_mesh_flag():
    source = (ROOT / "simulation" / "TI_runner_multi-core.py").read_text(encoding="utf-8")

    assert "--reuse-existing-mesh" in source
    assert "resolve_existing_mesh" in source
    assert "reuse_existing_mesh" in source
    assert "mesh_reuse_enabled" in source


def test_inplace_rerun_slurm_scripts_have_expected_entrypoints():
    array_script = ROOT / "HPC_scripts" / "camcan_inplace_rerun_array.slurm"
    submit_script = ROOT / "HPC_scripts" / "submit_camcan_inplace_rerun.sh"

    assert array_script.is_file()
    assert submit_script.is_file()


def test_inplace_rerun_shell_scripts_pass_bash_syntax_check():
    for relative in (
        "HPC_scripts/camcan_inplace_rerun_array.slurm",
        "HPC_scripts/submit_camcan_inplace_rerun.sh",
    ):
        result = subprocess.run(
            ["bash", "-n", str(ROOT / relative)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


def test_inplace_rerun_scripts_use_manifest_and_reuse_mesh_flag():
    array_source = (
        ROOT / "HPC_scripts" / "camcan_inplace_rerun_array.slurm"
    ).read_text(encoding="utf-8")
    submit_source = (
        ROOT / "HPC_scripts" / "submit_camcan_inplace_rerun.sh"
    ).read_text(encoding="utf-8")

    assert "TI_INPLACE_RERUN_MANIFEST" in array_source
    assert "--reuse-existing-mesh" in array_source
    assert "TASK_OFFSET" in array_source
    assert "MONTAGE_PRESET" in submit_source
    assert "MAX_ARRAY_TASKS" in submit_source


def test_submit_wrapper_supports_chunk_resume_controls():
    submit_source = (
        ROOT / "HPC_scripts" / "submit_camcan_inplace_rerun.sh"
    ).read_text(encoding="utf-8")

    assert "START_TASK_OFFSET" in submit_source
    assert "MAX_SUBMITTED_CHUNKS" in submit_source
    assert "Next resume command" in submit_source
    assert "SBATCH_OUTPUT=" in submit_source
