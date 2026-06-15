import csv
import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "post"
    / "median_mesh_dataset"
    / "hpc_seed_fixed_median_mesh_dataset.py"
)
spec = importlib.util.spec_from_file_location("hpc_seed_fixed_median_mesh_dataset", SCRIPT_PATH)
assert spec is not None
seed_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = seed_module
spec.loader.exec_module(seed_module)


def _write_selection_csv(path: Path, selected_m2m_dir: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "subject",
                "selection_status",
                "selected_repeat_tag",
                "selected_m2m_dir",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "subject": "sub-01",
                "selection_status": "selected",
                "selected_repeat_tag": "repeat_007",
                "selected_m2m_dir": str(selected_m2m_dir),
            }
        )


def _create_selected_repeat_anat(tmp_path: Path) -> Path:
    anat_dir = tmp_path / "scaled" / "sub-01_repeatability" / "remesh" / "repeats" / "repeat_007" / "sub-01" / "anat"
    m2m_dir = anat_dir / "m2m_sub-01"
    m2m_dir.mkdir(parents=True)
    (m2m_dir / "sub-01.msh").write_text("mesh\n", encoding="utf-8")
    for name in [
        "sub-01_T1w.nii",
        "sub-01_T2w.nii",
        "sub-01_T1w_ras_1mm_T1andT2_masks.nii",
    ]:
        (anat_dir / name).write_text(name + "\n", encoding="utf-8")
    return anat_dir


def test_seed_dataset_can_seed_fixed_repeat_workspaces_from_selected_repeat(tmp_path):
    selected_anat = _create_selected_repeat_anat(tmp_path)
    selection_csv = tmp_path / "selection.csv"
    _write_selection_csv(selection_csv, selected_anat / "m2m_sub-01")
    new_root = tmp_path / "fixed_median"
    manifest = tmp_path / "manifest.csv"

    seed_module.seed_dataset(
        selection_csv=selection_csv,
        new_root=new_root,
        manifest=manifest,
        overwrite=False,
        dry_run=False,
        seed_repeat_workspaces=True,
        repeat_count=2,
        common_input_mode="symlink",
    )

    cache_anat = new_root / "sub-01_repeatability" / "fixed_mesh" / "mesh_cache" / "sub-01" / "anat"
    cache_m2m = cache_anat / "m2m_sub-01"
    assert (cache_m2m / "sub-01.msh").read_text(encoding="utf-8") == "mesh\n"
    assert (cache_anat / "sub-01_T1w.nii").resolve() == selected_anat / "sub-01_T1w.nii"
    assert (cache_anat / ".mesh_ready.json").is_file()

    for repeat_tag in ("repeat_001", "repeat_002"):
        repeat_anat = (
            new_root
            / "sub-01_repeatability"
            / "fixed_mesh"
            / "repeats"
            / repeat_tag
            / "sub-01"
            / "anat"
        )
        assert (repeat_anat / "sub-01_T1w.nii").resolve() == selected_anat / "sub-01_T1w.nii"
        assert (repeat_anat / "m2m_sub-01").resolve() == cache_m2m

    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8", newline="")))
    assert rows[0]["status"] == "seeded"
    assert rows[0]["seeded_repeat_workspaces"] == "2"
    assert rows[0]["common_input_mode"] == "symlink"
