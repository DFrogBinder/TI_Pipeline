import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "charm_segmentation_batch"))

import mesh_collected_segmentations as legacy  # noqa: E402


class FakeMesh:
    def __init__(self, tags=(1, 2, 5)):
        self.elm = SimpleNamespace(
            elm_type=np.array([2] + [4] * len(tags)),
            tag1=np.array([1005, *tags]),
        )


def fake_api(tmp_path: Path, *, version: str = "4.0.1", tags=(1, 2, 5)):
    mesh = FakeMesh(tags)
    settings = tmp_path / "simnibs" / "charm.ini"
    settings.parent.mkdir(exist_ok=True)
    settings.write_text("[mesh]\n", encoding="utf-8")

    class Nib:
        loads = 0

        @classmethod
        def load(cls, path):
            cls.loads += 1
            return SimpleNamespace(
                shape=(2, 2, 2),
                dataobj=np.ones((2, 2, 2), dtype=np.uint16),
                affine=np.eye(4),
            )

    class MeshIO:
        @staticmethod
        def write_msh(value, path):
            Path(path).write_bytes(b"mesh")

        @staticmethod
        def read_msh(path):
            return mesh

    class Settings:
        @staticmethod
        def read_ini(path):
            return {
                "mesh": {
                    "elem_sizes": {},
                    "smooth_size_field": 2,
                    "skin_facet_size": 2.0,
                    "facet_distances": {},
                    "optimize": False,
                    "remove_spikes": True,
                    "skin_tag": 1005,
                    "hierarchy": None,
                    "smooth_steps": 5,
                    "skin_care": 20,
                }
            }

    return {
        "nib": Nib,
        "np": np,
        "simnibs": SimpleNamespace(__version__=version),
        "SIMNIBSDIR": str(settings.parent),
        "mesh_io": MeshIO,
        "create_mesh": lambda label, affine, **kwargs: mesh,
        "settings_reader": Settings,
        "crop_vol": lambda volume, affine, mask, thickness_boundary: (
            volume,
            affine,
            None,
        ),
    }


def primitive_paths(tmp_path: Path):
    label = tmp_path / "input.nii.gz"
    label.write_bytes(b"label")
    mesh = tmp_path / "outputs" / "subjects" / "sub-1" / "seg_001" / "mesh.msh"
    result = tmp_path / "outputs" / "results" / "task.json"
    return label, mesh, result


def test_version_mismatch_stops_before_loading_label_or_meshing(tmp_path, monkeypatch):
    label, mesh, result = primitive_paths(tmp_path)
    api = fake_api(tmp_path, version="4.6")
    monkeypatch.setattr(legacy, "_load_meshing_api", lambda: api)

    with pytest.raises(RuntimeError, match="version mismatch"):
        legacy.create_mesh_from_label(
            subject="sub-1",
            label_path=label,
            expected_label_hash=hashlib.sha256(b"label").hexdigest(),
            mesh_path=mesh,
            result_path=result,
            task_index=0,
            expected_simnibs_version="4.0.1",
        )

    assert api["nib"].loads == 0
    assert not mesh.exists()


def test_audited_primitive_records_version_settings_and_config_hash(tmp_path, monkeypatch):
    label, mesh, result = primitive_paths(tmp_path)
    api = fake_api(tmp_path)
    monkeypatch.setattr(legacy, "_load_meshing_api", lambda: api)

    payload = legacy.create_mesh_from_label(
        subject="sub-1",
        label_path=label,
        expected_label_hash=hashlib.sha256(b"label").hexdigest(),
        mesh_path=mesh,
        result_path=result,
        task_index=0,
        expected_simnibs_version="4.0.1",
        required_tissue_tags=(1, 2),
        provenance={"campaign_id": "test", "config_sha256": "a" * 64},
        staging_root=tmp_path,
    )

    assert payload["expected_simnibs_version"] == "4.0.1"
    assert payload["simnibs_version"] == "4.0.1"
    assert payload["settings_sha256"] == legacy.sha256_file(
        Path(payload["settings_path"])
    )
    assert payload["config_sha256"] == "a" * 64
    assert payload["required_tissue_tags"] == [1, 2]


def test_changed_settings_invalidates_reuse_and_refuses_overwrite(tmp_path, monkeypatch):
    label, mesh, result = primitive_paths(tmp_path)
    api = fake_api(tmp_path)
    monkeypatch.setattr(legacy, "_load_meshing_api", lambda: api)
    kwargs = dict(
        subject="sub-1",
        label_path=label,
        expected_label_hash=hashlib.sha256(b"label").hexdigest(),
        mesh_path=mesh,
        result_path=result,
        task_index=0,
        expected_simnibs_version="4.0.1",
        provenance={"config_sha256": "a" * 64},
        staging_root=tmp_path,
        overwrite=False,
    )
    first = legacy.create_mesh_from_label(**kwargs)
    second = legacy.create_mesh_from_label(**kwargs)
    assert second["mesh_sha256"] == first["mesh_sha256"]

    Path(first["settings_path"]).write_text("[changed]\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite is disabled"):
        legacy.create_mesh_from_label(**kwargs)


def test_required_tissue_tags_are_enforced(tmp_path, monkeypatch):
    label, mesh, result = primitive_paths(tmp_path)
    monkeypatch.setattr(legacy, "_load_meshing_api", lambda: fake_api(tmp_path, tags=(1, 2)))

    with pytest.raises(ValueError, match="required tissue tags"):
        legacy.create_mesh_from_label(
            subject="sub-1",
            label_path=label,
            expected_label_hash=hashlib.sha256(b"label").hexdigest(),
            mesh_path=mesh,
            result_path=result,
            task_index=0,
            expected_simnibs_version="4.0.1",
            required_tissue_tags=(1, 2, 5),
            staging_root=tmp_path,
        )


def test_campaign_runner_passes_explicit_task_identity_and_provenance(tmp_path, monkeypatch):
    from charm_mesh_campaign import runner
    from charm_mesh_campaign.tasks import TaskRecord

    label = tmp_path / "input.nii.gz"
    label.write_bytes(b"label-data")
    output_root = tmp_path / "out"
    task = TaskRecord(
        task_id=0,
        subject_id="sub-1",
        realization_id="seg_002",
        task_key="sub-1__seg_002",
        label_path=label,
        label_sha256=hashlib.sha256(b"label-data").hexdigest(),
        label_bytes=10,
        mesh_path=output_root / "mesh.msh",
        result_path=output_root / "result.json",
        config_sha256="c" * 64,
        status="ready",
        message="ready",
    )
    config = SimpleNamespace(
        campaign_id="campaign",
        sha256="c" * 64,
        output=SimpleNamespace(task_manifest=Path("tasks.tsv"), root=output_root),
        input=SimpleNamespace(segmentation_version="4.0.1"),
        software=SimpleNamespace(
            expected_version="4.0.1", module="module", python="python", settings_path=None
        ),
        quality=SimpleNamespace(required_tissue_tags=(1, 2)),
        execution=SimpleNamespace(local_staging=False),
        resume=SimpleNamespace(reuse_validated=True, overwrite=False),
    )
    monkeypatch.setattr(runner, "read_task_manifest", lambda path: [task])
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(runner.legacy, "create_mesh_from_label", create)

    assert runner.run_task(config, 0) == {"status": "complete"}
    assert captured["subject"] == "sub-1"
    assert captured["expected_simnibs_version"] == "4.0.1"
    assert captured["provenance"]["realization_id"] == "seg_002"
    assert captured["provenance"]["task_key"] == "sub-1__seg_002"
    assert captured["provenance"]["config_sha256"] == "c" * 64
