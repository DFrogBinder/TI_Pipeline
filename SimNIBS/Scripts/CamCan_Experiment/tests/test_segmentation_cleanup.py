import csv
import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from segmentation_cleanup import workflow


def _synthetic_labels() -> np.ndarray:
    labels = np.zeros((41, 41, 41), dtype=np.uint16)
    labels[3:38, 3:38, 3:38] = 5
    labels[6:35, 6:35, 6:35] = 7
    labels[9:32, 9:32, 9:32] = 3
    labels[12:29, 12:29, 12:29] = 2
    labels[15:26, 15:26, 15:26] = 1

    # Preserve representative non-target tissues.
    labels[7:9, 18:20, 18:20] = 8
    labels[6:8, 12:14, 12:14] = 6
    labels[10:12, 25:27, 25:27] = 9
    labels[4:6, 25:28, 25:28] = 10
    labels[6:8, 20:22, 20:22] = 11

    # A narrow CSF/brain-envelope cleft and a through-head hole.
    labels[9:22, 20, 20] = 7
    labels[3:10, 8, 8] = 0

    # A WM island embedded in main GM becomes GM after WM filtering.
    labels[13, 13, 20] = 1
    # A fully detached GM/WM fragment becomes CSF after cumulative filtering.
    labels[34, 34, 34] = 2
    labels[34, 34, 35] = 1
    return labels


def _write_map(path: Path, labels: np.ndarray | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(
        _synthetic_labels() if labels is None else labels,
        np.diag([1.0, 1.0, 1.0, 1.0]),
    )
    image.set_qform(image.affine, 1)
    image.set_sform(image.affine, 1)
    nib.save(image, str(path))
    return path


def test_edt_ball_closing_matches_scipy_ball_structure():
    mask = np.zeros((25, 25, 25), dtype=bool)
    mask[4:21, 4:21, 4:21] = True
    mask[4:15, 12, 12] = False
    radius = 2
    coordinates = np.indices((2 * radius + 1,) * 3) - radius
    ball = np.sum(coordinates * coordinates, axis=0) <= radius * radius
    expected = ndimage.binary_closing(mask, structure=ball, border_value=0)

    actual = workflow._ball_closing(mask, radius)

    assert np.array_equal(actual, expected)


def test_edt_ball_opening_matches_scipy_ball_structure():
    mask = np.zeros((25, 25, 25), dtype=bool)
    mask[4:21, 4:21, 4:21] = True
    mask[3:12, 12, 12] = True
    radius = 2
    coordinates = np.indices((2 * radius + 1,) * 3) - radius
    ball = np.sum(coordinates * coordinates, axis=0) <= radius * radius
    expected = ndimage.binary_opening(mask, structure=ball, border_value=0)

    actual = workflow._ball_opening(mask, radius)

    assert np.array_equal(actual, expected)


def test_cumulative_cleanup_closes_holes_filters_fragments_and_preserves_labels():
    source = _synthetic_labels()
    untouched = source.copy()

    corrected, metrics = workflow.correct_label_array(
        source,
        csf_radius=2,
        skin_radius=2,
        component_policy="largest",
        connectivity=26,
    )

    assert np.array_equal(source, untouched)
    assert metrics["changed_voxels"] > 0
    assert (
        metrics["parameters"]["algorithm"]
        == workflow.CSF_EXCLUDES_BLOOD_ALGORITHM
    )
    assert metrics["parameters"]["csf_source_labels"] == [1, 2, 3]
    assert metrics["parameters"]["blood_restored_after_csf"] is True
    assert "csf_component_policy" not in metrics["parameters"]
    assert "csf_cumulative_components" not in metrics
    assert metrics["wm_components"]["components_removed"] >= 1
    assert metrics["gm_cumulative_components"]["components_removed"] >= 1
    assert corrected[13, 13, 20] == 2
    assert corrected[34, 34, 34] == 3
    assert corrected[34, 34, 35] == 3
    assert corrected[8, 8, 8] == 5
    assert corrected[14, 20, 20] == 3
    for label in (6, 8, 9, 10, 11):
        assert np.any(corrected == label)


def test_csf_largest_component_filter_removes_detached_cumulative_island():
    source = _synthetic_labels()
    detached = (36, 36, 36)
    source[detached] = 3

    unfiltered, _ = workflow.correct_label_array(
        source,
        csf_radius=0,
        skin_radius=0,
        csf_component_policy="none",
    )
    filtered, metrics = workflow.correct_label_array(
        source,
        csf_radius=0,
        skin_radius=0,
        csf_component_policy="largest",
    )

    assert unfiltered[detached] == 3
    assert filtered[detached] == 7
    assert (
        metrics["parameters"]["algorithm"]
        == workflow.CSF_EXCLUDES_BLOOD_ALGORITHM
    )
    assert metrics["parameters"]["csf_component_policy"] == "largest"
    assert metrics["csf_cumulative_components"]["components_removed"] >= 1
    assert metrics["csf_cumulative_voxels_after_component_filter"] < metrics[
        "csf_cumulative_voxels_before"
    ]
    assert np.count_nonzero(filtered == 9) == np.count_nonzero(source == 9)


def test_csf_excludes_blood_before_closing_and_restores_blood_afterward():
    source = _synthetic_labels()
    source[6:8, 18:23, 18:23] = 9

    corrected, metrics = workflow.correct_label_array(
        source,
        csf_radius=2,
        skin_radius=0,
    )
    legacy, legacy_metrics = workflow.correct_label_array(
        source,
        csf_radius=2,
        skin_radius=0,
        include_blood_in_csf=True,
    )

    # Legacy inclusion grows a CSF bridge toward the nearby blood mask.
    assert corrected[8, 19, 19] == source[8, 19, 19]
    assert legacy[8, 19, 19] == 3
    assert np.count_nonzero((legacy == 3) & (corrected != 3)) > 0

    # Both modes restore every original blood voxel after CSF reconstruction.
    original_blood = source == 9
    assert np.all(corrected[original_blood] == 9)
    assert np.all(legacy[original_blood] == 9)
    assert metrics["parameters"]["csf_source_labels"] == [1, 2, 3]
    assert "csf_source_labels" not in legacy_metrics["parameters"]
    assert legacy_metrics["parameters"]["algorithm"] == workflow.ALGORITHM


def test_csf_opening_removes_attached_protrusion_without_removing_blood():
    source = _synthetic_labels()
    source[6:9, 22, 22] = 3

    closed_only, _ = workflow.correct_label_array(
        source,
        csf_radius=2,
        csf_opening_radius=0,
        skin_radius=0,
    )
    smoothed, metrics = workflow.correct_label_array(
        source,
        csf_radius=2,
        csf_opening_radius=2,
        skin_radius=0,
        csf_component_policy="largest",
    )

    assert closed_only[7, 22, 22] == 3
    assert smoothed[7, 22, 22] != 3
    assert metrics["parameters"]["algorithm"] == workflow.CSF_CLOSE_OPEN_ALGORITHM
    assert metrics["parameters"]["csf_opening_radius_voxels"] == 2
    assert metrics["csf_opening_removed_voxels"] > 0
    assert "csf_cumulative_components_after_opening" in metrics
    assert metrics["csf_post_opening_component_filter"]["components_kept"] == 1
    assert metrics["csf_cumulative_components_final"]["components"] == 1
    original_blood = source == 9
    assert np.all(smoothed[original_blood] == 9)


def test_csf_opening_rejects_legacy_blood_inclusion():
    try:
        workflow.correct_label_array(
            _synthetic_labels(),
            csf_opening_radius=2,
            include_blood_in_csf=True,
        )
    except ValueError as exc:
        assert "cannot be combined" in str(exc)
    else:
        raise AssertionError("CSF opening accepted legacy blood inclusion")


def test_min_size_policy_requires_explicit_positive_thresholds():
    labels = _synthetic_labels()
    try:
        workflow.correct_label_array(labels, component_policy="min-size")
    except ValueError as exc:
        assert "positive WM and GM component thresholds" in str(exc)
    else:
        raise AssertionError("min-size cleanup accepted missing thresholds")


def test_completed_output_cannot_be_silently_rewritten_with_new_parameters(tmp_path):
    subject = "sub-CC000001"
    maps = tmp_path / "source" / "maps"
    _write_map(maps / f"{subject}{workflow.MAP_SUFFIX}")
    output = tmp_path / "corrected"
    manifest = output / "campaign" / "cleanup.tsv"
    workflow.build_preflight_manifest(
        maps_root=maps,
        output_root=output,
        manifest=manifest,
        summary=output / "campaign" / "preflight.json",
        expected_subjects=1,
    )
    workflow.run_task(manifest=manifest, task_index=0, csf_radius=1, skin_radius=1)

    try:
        workflow.run_task(manifest=manifest, task_index=0, csf_radius=2, skin_radius=1)
    except ValueError as exc:
        assert "new versioned output root" in str(exc)
    else:
        raise AssertionError("cleanup silently replaced a different parameter version")


def test_end_to_end_cleanup_writes_a_mesh_compatible_collection(tmp_path):
    subject = "sub-CC000001"
    maps = tmp_path / "source" / "maps"
    source = _write_map(maps / f"{subject}{workflow.MAP_SUFFIX}")
    original_hash = workflow.sha256_file(source)
    output = tmp_path / "corrected"
    manifest = output / "campaign" / "cleanup.tsv"
    preflight = workflow.build_preflight_manifest(
        maps_root=maps,
        output_root=output,
        manifest=manifest,
        summary=output / "campaign" / "preflight.json",
        expected_subjects=1,
    )
    assert preflight["status"] == "ready"

    result = workflow.run_task(
        manifest=manifest,
        task_index=0,
        csf_radius=2,
        skin_radius=2,
    )
    assert result["status"] == "complete"
    assert result["source_modified"] is False
    assert workflow.sha256_file(source) == original_hash
    assert result["corrected_sha256"] != original_hash

    collection = output / "collection" / "charm_segmentation_manifest.tsv"
    validation = workflow.validate_results(
        manifest=manifest,
        validation=output / "campaign" / "validation.tsv",
        summary=output / "campaign" / "validation.json",
        collection_manifest=collection,
        checksums=output / "collection" / "sha256sums.txt",
    )
    assert validation["status"] == "complete"
    assert validation["complete"] == 1
    collection_rows = workflow.read_tsv(collection)
    assert collection_rows[0]["status"] == "complete"
    assert collection_rows[0]["source_map"] == str(source)
    assert Path(collection_rows[0]["collected_map"]).is_file()

    # The existing direct-CHARM mesh preflight can consume the corrected
    # collection without conversion or manual renaming.
    from charm_segmentation_batch import mesh_collected_segmentations

    mesh_payload = mesh_collected_segmentations.build_preflight_manifest(
        collection_manifest=collection,
        mesh_root=tmp_path / "meshes",
        manifest=tmp_path / "meshes" / "campaign" / "manifest.tsv",
        summary=tmp_path / "meshes" / "campaign" / "preflight.json",
        expected_subjects=1,
    )
    assert mesh_payload["status"] == "ready"


def _write_executable(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_full_submitter_uses_discovered_scope_and_afterany_collector(tmp_path):
    maps = tmp_path / "source" / "maps"
    _write_map(maps / f"sub-CC000001{workflow.MAP_SUFFIX}")
    sbatch_log = tmp_path / "sbatch.log"
    sbatch_count = tmp_path / "sbatch.count"
    fake_sbatch = _write_executable(
        tmp_path / "sbatch",
        "#!/bin/bash\n"
        "printf '%s\\n' \"$*\" >> \"$FAKE_SBATCH_LOG\"\n"
        "count=0\n"
        "if [ -f \"$FAKE_SBATCH_COUNT\" ]; then count=$(cat \"$FAKE_SBATCH_COUNT\"); fi\n"
        "count=$((count + 1))\n"
        "printf '%s\\n' \"$count\" > \"$FAKE_SBATCH_COUNT\"\n"
        "echo $((15000 + count))\n",
    )
    fake_scontrol = _write_executable(
        tmp_path / "scontrol",
        "#!/bin/bash\necho 'MaxArraySize = 1001'\n",
    )
    script = (
        Path(__file__).resolve().parents[1]
        / "segmentation_cleanup"
        / "submit_charm_segmentation_cleanup.sh"
    )
    env = os.environ.copy()
    env.update(
        {
            "MAPS_ROOT": str(maps),
            "OUTPUT_ROOT": str(tmp_path / "corrected"),
            "EXPECTED_SUBJECTS": "1",
            "MAX_CONCURRENT_TASKS": "1",
            "TI_CHARM_CLEANUP_CSF_RADIUS": "7",
            "TI_CHARM_CLEANUP_CSF_OPENING_RADIUS": "3",
            "TI_CHARM_CLEANUP_CSF_COMPONENT_POLICY": "largest",
            "SBATCH_BIN": str(fake_sbatch),
            "SCONTROL_BIN": str(fake_scontrol),
            "LOAD_SIMNIBS_MODULE": "0",
            "FAKE_SBATCH_LOG": str(sbatch_log),
            "FAKE_SBATCH_COUNT": str(sbatch_count),
        }
    )

    completed = subprocess.run(
        ["bash", str(script)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "subjects: 1" in completed.stdout
    assert "correction tasks: 1" in completed.stdout
    assert "array: 0-0%1" in completed.stdout
    assert "CSF closing: 7 voxels" in completed.stdout
    assert "CSF opening: 3 voxels" in completed.stdout
    assert (
        "CSF source labels: 1,2,3 (blood excluded and restored afterward)"
        in completed.stdout
    )
    assert "CSF components: largest, connectivity 26" in completed.stdout
    assert "skin closing: 10 voxels" in completed.stdout
    assert "Preflight Python:" in completed.stdout
    assert "Module bootstrap:  0" in completed.stdout
    assert "Submitted cleanup array job: 15001" in completed.stdout
    assert "Submitted afterany collector job: 15002" in completed.stdout
    submissions = sbatch_log.read_text(encoding="utf-8").splitlines()
    assert len(submissions) == 2
    assert "--array=0-0%1" in submissions[0]
    assert "TI_CHARM_CLEANUP_CSF_RADIUS=7" in submissions[0]
    assert "TI_CHARM_CLEANUP_CSF_OPENING_RADIUS=3" in submissions[0]
    assert "TI_CHARM_CLEANUP_CSF_INCLUDE_BLOOD=0" in submissions[0]
    assert "TI_CHARM_CLEANUP_CSF_COMPONENT_POLICY=largest" in submissions[0]
    assert "TI_CHARM_CLEANUP_COMPONENT_POLICY=largest" in submissions[0]
    assert "--time=08:00:00" in submissions[0]
    assert "--dependency=afterany:15001" in submissions[1]
    assert "--mem=16G" in submissions[1]
    assert "--time=08:00:00" in submissions[1]


def test_submitter_can_pack_two_subject_workers_per_array_element(tmp_path):
    maps = tmp_path / "source" / "maps"
    for index in range(3):
        _write_map(maps / f"sub-CC00000{index + 1}{workflow.MAP_SUFFIX}")
    sbatch_log = tmp_path / "sbatch.log"
    sbatch_count = tmp_path / "sbatch.count"
    fake_sbatch = _write_executable(
        tmp_path / "sbatch",
        "#!/bin/bash\n"
        "printf '%s\\n' \"$*\" >> \"$FAKE_SBATCH_LOG\"\n"
        "count=0\n"
        "if [ -f \"$FAKE_SBATCH_COUNT\" ]; then count=$(cat \"$FAKE_SBATCH_COUNT\"); fi\n"
        "count=$((count + 1))\n"
        "printf '%s\\n' \"$count\" > \"$FAKE_SBATCH_COUNT\"\n"
        "echo $((16000 + count))\n",
    )
    fake_scontrol = _write_executable(
        tmp_path / "scontrol",
        "#!/bin/bash\necho 'MaxArraySize = 1001'\n",
    )
    script = (
        Path(__file__).resolve().parents[1]
        / "segmentation_cleanup"
        / "submit_charm_segmentation_cleanup.sh"
    )
    env = os.environ.copy()
    env.update(
        {
            "MAPS_ROOT": str(maps),
            "OUTPUT_ROOT": str(tmp_path / "corrected"),
            "EXPECTED_SUBJECTS": "3",
            "MAX_CONCURRENT_TASKS": "50",
            "TI_CHARM_CLEANUP_WORKERS_PER_ARRAY_TASK": "2",
            "CPUS_PER_TASK": "2",
            "MEMORY": "24G",
            "SBATCH_BIN": str(fake_sbatch),
            "SCONTROL_BIN": str(fake_scontrol),
            "LOAD_SIMNIBS_MODULE": "0",
            "FAKE_SBATCH_LOG": str(sbatch_log),
            "FAKE_SBATCH_COUNT": str(sbatch_count),
        }
    )

    completed = subprocess.run(
        ["bash", str(script)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "subjects: 3" in completed.stdout
    assert "correction tasks: 3" in completed.stdout
    assert "array elements: 2" in completed.stdout
    assert "array: 0-1%50" in completed.stdout
    assert "workers per element: 2" in completed.stdout
    assert "maximum simultaneous subjects: 100" in completed.stdout
    submissions = sbatch_log.read_text(encoding="utf-8").splitlines()
    assert len(submissions) == 2
    assert "--array=0-1%50" in submissions[0]
    assert "--cpus-per-task=2" in submissions[0]
    assert "--mem=24G" in submissions[0]
    assert "TI_CHARM_CLEANUP_WORKERS_PER_ARRAY_TASK=2" in submissions[0]


def test_array_runner_executes_a_two_subject_shard(tmp_path):
    manifest = tmp_path / "manifest.tsv"
    rows = []
    for index in range(3):
        subject = f"sub-CC00000{index + 1}"
        rows.append(
            {
                "task_id": index,
                "subject": subject,
                "source_map": str(tmp_path / f"{subject}.nii.gz"),
                "source_sha256": f"hash-{index}",
                "source_bytes": 1,
                "corrected_map": str(tmp_path / "corrected" / f"{subject}.nii.gz"),
                "result_path": str(tmp_path / "results" / f"{subject}.json"),
                "status": "ready",
                "message": "ready",
            }
        )
    workflow.write_tsv(manifest, workflow.MANIFEST_FIELDS, rows)

    marker_root = tmp_path / "markers"
    marker_root.mkdir()
    fake_workflow = tmp_path / "fake_workflow.py"
    fake_workflow.write_text(
        "import os, pathlib, sys, time\n"
        "index = sys.argv[sys.argv.index('--task-index') + 1]\n"
        "root = pathlib.Path(os.environ['FAKE_MARKER_ROOT'])\n"
        "(root / f'started-{index}').write_text(str(time.time_ns()))\n"
        "time.sleep(0.2)\n"
        "(root / f'finished-{index}').write_text('complete')\n",
        encoding="utf-8",
    )
    _write_executable(tmp_path / "module", "#!/bin/bash\nexit 0\n")
    runner = (
        Path(__file__).resolve().parents[1]
        / "segmentation_cleanup"
        / "cleanup_charm_segmentations_array.slurm"
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{tmp_path}:{env['PATH']}",
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_ARRAY_JOB_ID": "17000",
            "SLURM_JOB_ID": "17001",
            "SLURM_CPUS_PER_TASK": "2",
            "TI_CHARM_CLEANUP_MANIFEST": str(manifest),
            "TI_CHARM_CLEANUP_WORKFLOW_PY": str(fake_workflow),
            "TI_CHARM_CLEANUP_LOG_DIR": str(tmp_path / "logs"),
            "TI_CHARM_CLEANUP_WORKERS_PER_ARRAY_TASK": "2",
            "FAKE_MARKER_ROOT": str(marker_root),
        }
    )

    completed = subprocess.run(
        ["bash", str(runner)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Manifest tasks:     0-1" in completed.stdout
    assert "Worker processes:   2 maximum" in completed.stdout
    assert (marker_root / "finished-0").is_file()
    assert (marker_root / "finished-1").is_file()
    assert not (marker_root / "started-2").exists()

    env["SLURM_ARRAY_TASK_ID"] = "1"
    env["SLURM_JOB_ID"] = "17002"
    final_shard = subprocess.run(
        ["bash", str(runner)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Manifest tasks:     2-2" in final_shard.stdout
    assert (marker_root / "finished-2").is_file()


def test_packed_runner_requeues_and_skips_completed_companion(tmp_path):
    manifest = tmp_path / "manifest.tsv"
    rows = []
    for index in range(2):
        subject = f"sub-CC00000{index + 1}"
        rows.append(
            {
                "task_id": index,
                "subject": subject,
                "source_map": str(tmp_path / f"{subject}.nii.gz"),
                "source_sha256": f"hash-{index}",
                "source_bytes": 1,
                "corrected_map": str(tmp_path / "corrected" / f"{subject}.nii.gz"),
                "result_path": str(tmp_path / "results" / f"{subject}.json"),
                "status": "ready",
                "message": "ready",
            }
        )
    workflow.write_tsv(manifest, workflow.MANIFEST_FIELDS, rows)

    marker_root = tmp_path / "markers"
    marker_root.mkdir()
    fake_workflow = tmp_path / "fake_retry_workflow.py"
    fake_workflow.write_text(
        "import os, pathlib, sys\n"
        "index = sys.argv[sys.argv.index('--task-index') + 1]\n"
        "root = pathlib.Path(os.environ['FAKE_MARKER_ROOT'])\n"
        "complete = root / f'complete-{index}'\n"
        "if complete.exists():\n"
        "    print(f'already-current-{index}')\n"
        "elif index == '1' and not (root / 'failed-once-1').exists():\n"
        "    (root / 'failed-once-1').write_text('failed')\n"
        "    raise SystemExit(7)\n"
        "else:\n"
        "    complete.write_text('complete')\n",
        encoding="utf-8",
    )
    _write_executable(tmp_path / "module", "#!/bin/bash\nexit 0\n")
    scontrol_log = tmp_path / "scontrol.log"
    _write_executable(
        tmp_path / "scontrol",
        "#!/bin/bash\nprintf '%s\\n' \"$*\" >> \"$FAKE_SCONTROL_LOG\"\n",
    )
    runner = (
        Path(__file__).resolve().parents[1]
        / "segmentation_cleanup"
        / "cleanup_charm_segmentations_array.slurm"
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{tmp_path}:{env['PATH']}",
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_ARRAY_JOB_ID": "18000",
            "SLURM_JOB_ID": "18001",
            "SLURM_CPUS_PER_TASK": "2",
            "TI_CHARM_CLEANUP_MANIFEST": str(manifest),
            "TI_CHARM_CLEANUP_WORKFLOW_PY": str(fake_workflow),
            "TI_CHARM_CLEANUP_LOG_DIR": str(tmp_path / "logs"),
            "TI_CHARM_CLEANUP_WORKERS_PER_ARRAY_TASK": "2",
            "TI_CHARM_CLEANUP_MAX_RETRIES": "2",
            "FAKE_MARKER_ROOT": str(marker_root),
            "FAKE_SCONTROL_LOG": str(scontrol_log),
        }
    )

    first = subprocess.run(
        ["bash", str(runner)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "requeueing (1/2)" in first.stdout
    assert (marker_root / "complete-0").is_file()
    assert not (marker_root / "complete-1").exists()
    assert scontrol_log.read_text(encoding="utf-8").strip() == "requeue 18001"

    env["SLURM_JOB_ID"] = "18002"
    second = subprocess.run(
        ["bash", str(runner)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "already-current-0" in second.stdout
    assert (marker_root / "complete-1").is_file()
    retry_file = tmp_path / "logs" / "retry_state" / "18000_0.retry"
    assert not retry_file.exists()
    assert scontrol_log.read_text(encoding="utf-8").strip() == "requeue 18001"


def test_preflight_blocks_source_output_alias(tmp_path):
    maps = tmp_path / "source" / "maps"
    _write_map(maps / f"sub-CC000001{workflow.MAP_SUFFIX}")
    try:
        workflow.build_preflight_manifest(
            maps_root=maps,
            output_root=maps.parent,
            manifest=tmp_path / "manifest.tsv",
            summary=tmp_path / "summary.json",
            expected_subjects=1,
        )
    except ValueError as exc:
        assert "must differ" in str(exc)
    else:
        raise AssertionError("preflight accepted source/output alias")


def test_collection_manifest_has_expected_header(tmp_path):
    rows = [
        {
            "subject": "sub-CC000001",
            "status": "complete",
            "source_map": "source",
            "collected_map": "corrected",
            "sha256": "hash",
            "bytes": 1,
            "message": "corrected",
        }
    ]
    path = tmp_path / "collection.tsv"
    workflow.write_tsv(path, workflow.COLLECTION_FIELDS, rows)
    with path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle, delimiter="\t"))
    assert tuple(header) == workflow.COLLECTION_FIELDS
