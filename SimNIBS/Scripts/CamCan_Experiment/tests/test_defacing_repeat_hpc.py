from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
HPC_ROOT = ROOT / "HPC_scripts"


def test_multi_core_runner_supports_explicit_pure_charm_generation():
    source = (ROOT / "simulation" / "TI_runner_multi-core.py").read_text(encoding="utf-8")

    assert "resolve_subject_input_paths" in source
    assert "subject_inputs.custom_segmentation is not None" in source
    assert "pure_charm_generation_complete" in source
    assert "merge_segmentation_maps" not in source


def test_defacing_repeat_shell_scripts_have_expected_entrypoints():
    submit_source = (HPC_ROOT / "submit_defacing_repeat_batch.sh").read_text(encoding="utf-8")
    array_source = (HPC_ROOT / "defacing_repeat_array.slurm").read_text(encoding="utf-8")

    assert "TI_DEFACING_REPEAT_MANIFEST" in submit_source
    assert "TI_DEFACING_REPEAT_MANIFEST" in array_source
    assert '--montage-preset "$MONTAGE_PRESET"' in array_source
    assert "--generate-charm-mesh" in array_source
    assert "--reuse-existing-mesh" not in array_source


def test_defacing_repeat_shell_scripts_pass_bash_syntax_check():
    for path in [
        HPC_ROOT / "defacing_repeat_array.slurm",
        HPC_ROOT / "submit_defacing_repeat_batch.sh",
    ]:
        subprocess.run(["bash", "-n", str(path)], check=True)
