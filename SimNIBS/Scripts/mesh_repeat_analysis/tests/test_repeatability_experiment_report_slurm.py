import re
from pathlib import Path


def test_report_slurm_uses_json_atlas_dir_by_default():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "hpc_scripts" / "repeatability_experiment_report.slurm"

    text = script.read_text(encoding="utf-8")

    assert re.search(r'^ATLAS_DIR_CONFIG=""$', text, re.MULTILINE)

