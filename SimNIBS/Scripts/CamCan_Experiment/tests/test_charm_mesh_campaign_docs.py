import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "charm_mesh_campaign"
sys.path.insert(0, str(ROOT))


def test_uncorrected_v401_example_is_strict_and_separate_from_corrected_outputs():
    from charm_mesh_campaign.config import load_campaign_config

    config = load_campaign_config(PACKAGE / "configs/uncorrected_v401.example.json")

    assert config.campaign_id == "uncorrected-charm-v401-meshes"
    assert config.input.segmentation_version == "4.0.1"
    assert config.software.expected_version == "4.0.1"
    assert config.software.module == "SimNIBS/4.0.1-foss-2023a"
    assert config.execution.expected_tasks == 652
    assert "uncorrected" in str(config.output.root)
    assert "corrected" not in str(config.output.root).replace("uncorrected", "")
    assert config.output.root != config.input.manifest.parent


def test_documentation_keeps_submission_explicitly_gated_and_scope_provisional():
    readme = (PACKAGE / "README.md").read_text(encoding="utf-8")

    assert "pending live preflight" in readme.lower()
    assert "submit --config" in readme
    assert "submit --config configs/uncorrected_v401.example.json --submit" in readme
    assert "does not run segmentation" in readme.lower()
    assert "one mesh per" in readme.lower()
