import csv
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "charm_segmentation_batch"
sys.path.insert(0, str(WORKFLOW_DIR))

import mesh_collected_segmentations as workflow  # noqa: E402


def write_collection(tmp_path: Path, subjects: tuple[str, ...]) -> Path:
    maps = tmp_path / "charm_segmentations" / "maps"
    maps.mkdir(parents=True)
    rows = []
    for subject in subjects:
        map_path = maps / f"{subject}{workflow.MAP_SUFFIX}"
        payload = f"label-{subject}".encode()
        map_path.write_bytes(payload)
        rows.append(
            {
                "subject": subject,
                "status": "complete",
                "source_map": str(map_path),
                "collected_map": str(map_path),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "message": "ok",
            }
        )
    manifest = tmp_path / "charm_segmentations" / "collection" / "charm_segmentation_manifest.tsv"
    manifest.parent.mkdir(parents=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def build_manifest(tmp_path: Path, subjects: tuple[str, ...]) -> tuple[Path, Path]:
    collection = write_collection(tmp_path, subjects)
    mesh_root = tmp_path / "meshes"
    manifest = mesh_root / "campaign" / "mesh_manifest.tsv"
    summary = mesh_root / "campaign" / "preflight.json"
    payload = workflow.build_preflight_manifest(
        collection_manifest=collection,
        mesh_root=mesh_root,
        manifest=manifest,
        summary=summary,
        expected_subjects=len(subjects),
    )
    assert payload["status"] == "ready"
    return manifest, mesh_root


def test_preflight_verifies_full_collection_and_builds_mesh_wall_layout(tmp_path):
    subjects = ("sub-CC000001", "sub-CC000002")
    manifest, mesh_root = build_manifest(tmp_path, subjects)

    rows = workflow.read_tsv(manifest)
    assert len(rows) == 2
    assert all(row["status"] == "ready" for row in rows)
    assert rows[0]["mesh_path"] == str(
        mesh_root
        / "subjects"
        / subjects[0]
        / "anat"
        / f"m2m_{subjects[0]}"
        / f"{subjects[0]}.msh"
    )
    summary = json.loads((mesh_root / "campaign" / "preflight.json").read_text())
    assert summary["subjects_expected"] == 2
    assert summary["ready"] == 2
    assert summary["segmentation_rerun"] is False


def test_preflight_blocks_count_mismatch(tmp_path):
    collection = write_collection(tmp_path, ("sub-CC000001", "sub-CC000002"))
    mesh_root = tmp_path / "meshes"
    manifest = mesh_root / "mesh_manifest.tsv"
    payload = workflow.build_preflight_manifest(
        collection_manifest=collection,
        mesh_root=mesh_root,
        manifest=manifest,
        summary=mesh_root / "preflight.json",
        expected_subjects=474,
    )

    assert payload["status"] == "blocked"
    assert payload["ready"] == 0
    assert all(row["status"] == "blocked" for row in workflow.read_tsv(manifest))


def test_preflight_explicitly_excludes_one_incomplete_subject(tmp_path):
    kept = "sub-CC000001"
    excluded = "sub-CC000002"
    collection = write_collection(tmp_path, (kept, excluded))
    rows = workflow.read_tsv(collection)
    Path(rows[1]["collected_map"]).unlink()
    rows[1]["status"] = "incomplete"
    rows[1]["sha256"] = ""
    rows[1]["bytes"] = ""
    rows[1]["message"] = "missing map"
    workflow.write_tsv(collection, tuple(rows[0]), rows)
    mesh_root = tmp_path / "meshes"
    manifest = mesh_root / "mesh_manifest.tsv"

    payload = workflow.build_preflight_manifest(
        collection_manifest=collection,
        mesh_root=mesh_root,
        manifest=manifest,
        summary=mesh_root / "preflight.json",
        expected_subjects=1,
        excluded_subjects=(excluded,),
    )

    assert payload["status"] == "ready"
    assert payload["collection_rows_total"] == 2
    assert payload["subjects_found"] == 1
    assert payload["excluded_subjects"] == [excluded]
    assert [row["subject"] for row in workflow.read_tsv(manifest)] == [kept]


class FakeMesh:
    def __init__(self):
        self.elm = SimpleNamespace(
            elm_type=np.array([2, 4, 4, 4]),
            tag1=np.array([1005, 1, 2, 5]),
        )


def test_mesh_settings_match_simnibs_401_api_without_newer_keys(monkeypatch):
    settings = {
        "mesh": {
            "elem_sizes": {},
            "smooth_size_field": 2,
            "skin_facet_size": 2.0,
            "facet_distances": {},
            "optimize": True,
            "remove_spikes": True,
            "skin_tag": 1005,
            "hierarchy": None,
            "smooth_steps": 5,
            "skin_care": 20,
        }
    }

    def create_mesh_401(
        label_img,
        affine,
        elem_sizes=None,
        smooth_size_field=2,
        skin_facet_size=2.0,
        facet_distances=None,
        optimize=True,
        remove_spikes=True,
        skin_tag=1005,
        hierarchy=None,
        smooth_steps=5,
        skin_care=20,
        sizing_field=None,
        DEBUG_FN=None,
    ):
        return None

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    supported = set(inspect.signature(create_mesh_401).parameters)
    options = workflow._mesh_settings(settings, supported)

    assert options["DEBUG_FN"] is None
    assert options["optimize"] is True
    assert "apply_cream" not in options
    assert "mmg_noinsert" not in options
    assert "num_threads" not in options
    assert "debug" not in options
    assert "debug_path" not in options


def test_run_task_directly_meshes_map_and_writes_provenance(tmp_path, monkeypatch):
    subject = "sub-CC000001"
    manifest, mesh_root = build_manifest(tmp_path, (subject,))
    fake_mesh = FakeMesh()

    class FakeNib:
        @staticmethod
        def load(path):
            return SimpleNamespace(
                shape=(3, 3, 3),
                dataobj=np.ones((3, 3, 3), dtype=np.uint16),
                affine=np.eye(4),
            )

    class FakeSettingsReader:
        @staticmethod
        def read_ini(path):
            return {
                "mesh": {
                    "elem_sizes": {},
                    "smooth_size_field": 2,
                    "skin_facet_size": 2.0,
                    "facet_distances": {},
                    "optimize": False,
                    "apply_cream": True,
                    "remove_spikes": True,
                    "skin_tag": 1005,
                    "hierarchy": None,
                    "smooth_steps": 5,
                    "skin_care": 20,
                    "mmg_noinsert": False,
                }
            }

    class FakeMeshIO:
        @staticmethod
        def write_msh(mesh, path):
            Path(path).write_bytes(b"loadable-mesh")

        @staticmethod
        def read_msh(path):
            assert Path(path).read_bytes() == b"loadable-mesh"
            return fake_mesh

    def create_mesh(label, affine, **kwargs):
        assert label.shape == (3, 3, 3)
        assert kwargs["num_threads"] == 8
        return fake_mesh

    settings_path = tmp_path / "simnibs" / "charm.ini"
    settings_path.parent.mkdir()
    settings_path.write_text("[fake]\n", encoding="utf-8")
    monkeypatch.setattr(
        workflow,
        "_load_meshing_api",
        lambda: {
            "nib": FakeNib,
            "np": np,
            "simnibs": SimpleNamespace(__version__="4.0.1"),
            "SIMNIBSDIR": str(settings_path.parent),
            "mesh_io": FakeMeshIO,
            "create_mesh": create_mesh,
            "settings_reader": FakeSettingsReader,
            "crop_vol": lambda volume, affine, mask, thickness_boundary: (
                volume,
                affine,
                None,
            ),
        },
    )
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")

    payload = workflow.run_mesh_task(manifest=manifest, task_index=0)

    mesh_path = Path(payload["mesh_path"])
    assert mesh_path == workflow.mesh_path_for_subject(mesh_root, subject)
    assert mesh_path.read_bytes() == b"loadable-mesh"
    assert payload["tetrahedra"] == 3
    assert payload["tissue_tags"] == [1, 2, 5]
    assert payload["label_sha256_before"] == payload["label_sha256_after"]
    assert payload["segmentation_rerun"] is False
    assert json.loads(workflow.result_path_for_subject(mesh_root, subject).read_text())["status"] == "complete"
    assert not list((mesh_root / ".staging").glob(f"{subject}-*"))


def test_submitter_enforces_scope_and_submits_one_477_task_array(tmp_path):
    subjects = tuple(f"sub-CC{index:06d}" for index in range(1, 478))
    manifest, mesh_root = build_manifest(tmp_path, subjects)
    sbatch_log = tmp_path / "sbatch.log"
    fake_sbatch = tmp_path / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f"printf '%s\\n' \"$*\" > '{sbatch_log}'\n"
        "echo '12345;testcluster'\n",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)
    environment = {
        **os.environ,
        "MANIFEST": str(manifest),
        "LOG_DIR": str(mesh_root / "logs"),
        "EXPECTED_TASKS": "477",
        "SBATCH_BIN": str(fake_sbatch),
    }

    result = subprocess.run(
        ["bash", str(WORKFLOW_DIR / "submit_collected_meshes.sh")],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert "subjects: 477" in result.stdout
    assert "tasks: 477" in result.stdout
    assert "array: 0-476%50" in result.stdout
    assert "execution: full requested 477-subject" in result.stdout
    assert "Submitted full collected-CHARM mesh array: 12345" in result.stdout
    call = sbatch_log.read_text(encoding="utf-8")
    assert "--array=0-476%50" in call
    assert "--job-name=mesh_charm_maps_477" in call
    assert "--cpus-per-task=8" in call
    assert "--mem=32G" in call
    assert "--time=08:00:00" in call


def test_new_shell_launchers_pass_syntax_check():
    for name in (
        "mesh_collected_segmentations_array.slurm",
        "submit_collected_meshes.sh",
    ):
        subprocess.run(["bash", "-n", str(WORKFLOW_DIR / name)], check=True)
