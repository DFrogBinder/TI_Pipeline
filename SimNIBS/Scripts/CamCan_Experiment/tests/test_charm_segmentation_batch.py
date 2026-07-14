import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "charm_segmentation_batch"
sys.path.insert(0, str(WORKFLOW))

from collect_charm_segmentations import collect  # noqa: E402
from run_charm_segmentation import (  # noqa: E402
    INCOMPLETE_EXIT_CODE,
    completed_output_is_valid,
    discover_subjects,
    map_destination,
    metadata_path,
    read_subjects_file,
    run_subject,
)


def make_subject(source_root: Path, subject: str) -> None:
    anat = source_root / subject / "anat"
    anat.mkdir(parents=True)
    (anat / f"{subject}_T1w.nii.gz").write_bytes(b"t1")
    (anat / f"{subject}_T2w.nii.gz").write_bytes(b"t2")


def make_fake_charm(path: Path, calls: Path) -> None:
    path.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f"printf '%s\\n' \"$*\" >> '{calls}'\n"
        "subject=$1\n"
        'mkdir -p "m2m_${subject}/label_prep"\n'
        "printf 'raw-charm-map-%s' \"$subject\" > "
        '"m2m_${subject}/label_prep/tissue_labeling_upsampled.nii.gz"\n',
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_discovery_includes_all_complete_subjects_and_reports_incomplete(tmp_path):
    source_root = tmp_path / "source"
    out_root = tmp_path / "out"
    make_subject(source_root, "sub-03")
    make_subject(source_root, "sub-01")
    incomplete_anat = source_root / "sub-02" / "anat"
    incomplete_anat.mkdir(parents=True)
    (incomplete_anat / "sub-02_T1w.nii.gz").write_bytes(b"t1")
    (source_root / "not-a-subject").mkdir()
    subjects_file = out_root / "submission" / "subjects.txt"
    report = out_root / "submission" / "preflight.tsv"

    result = discover_subjects(
        source_root=source_root,
        out_root=out_root,
        subjects_file=subjects_file,
        report_path=report,
    )

    assert result == 0
    assert read_subjects_file(subjects_file) == ("sub-01", "sub-03")
    report_text = report.read_text(encoding="utf-8")
    assert "sub-01\t" in report_text
    assert "sub-02\t\t\tblocked\t" in report_text


def test_runner_uses_segmentation_stages_only_and_preserves_bytes(tmp_path):
    source_root = tmp_path / "source"
    out_root = tmp_path / "out"
    subject = "sub-CC000001"
    make_subject(source_root, subject)
    calls = tmp_path / "calls.txt"
    fake_charm = tmp_path / "charm"
    make_fake_charm(fake_charm, calls)

    destination = run_subject(
        source_root=source_root,
        out_root=out_root,
        subject=subject,
        timeout_hours=1,
        charm_bin=str(fake_charm),
        force=False,
    )

    command = calls.read_text(encoding="utf-8")
    assert "--registerT2" in command
    assert "--initatlas" in command
    assert "--segment" in command
    assert "--surfaces" not in command
    assert "--mesh" not in command
    assert "--forcerun" not in command
    assert destination.read_bytes() == b"raw-charm-map-sub-CC000001"
    assert completed_output_is_valid(out_root, subject)
    metadata = json.loads(metadata_path(out_root, subject).read_text(encoding="utf-8"))
    assert metadata["mesh_requested"] is False
    assert metadata["surfaces_requested"] is False
    assert metadata["simulation_requested"] is False
    assert metadata["temporary_charm_outputs_deleted"] is True
    assert metadata["generated_map_sha256"] == metadata["archived_map_sha256"]
    assert not list((out_root / ".tmp").glob("charm_sub-CC000001_*"))


def test_completed_subject_is_reused_without_running_charm_again(tmp_path):
    source_root = tmp_path / "source"
    out_root = tmp_path / "out"
    subject = "sub-CC000001"
    make_subject(source_root, subject)
    calls = tmp_path / "calls.txt"
    fake_charm = tmp_path / "charm"
    make_fake_charm(fake_charm, calls)

    for _ in range(2):
        run_subject(
            source_root=source_root,
            out_root=out_root,
            subject=subject,
            timeout_hours=1,
            charm_bin=str(fake_charm),
            force=False,
        )

    assert calls.read_text(encoding="utf-8").count("--segment") == 1


def write_complete_map(out_root: Path, subject: str, payload: bytes) -> None:
    map_path = map_destination(out_root, subject)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    marker = metadata_path(out_root, subject)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "status": "complete",
                "subject": subject,
                "archived_map_sha256": digest,
            }
        ),
        encoding="utf-8",
    )


def test_collector_copies_valid_maps_and_reports_missing_subjects(tmp_path):
    out_root = tmp_path / "out"
    subjects_file = tmp_path / "subjects.txt"
    subjects_file.write_text("sub-01\nsub-02\n", encoding="utf-8")
    write_complete_map(out_root, "sub-01", b"one")

    result = collect(
        out_root=out_root,
        subjects_file=subjects_file,
        collection_dir=out_root / "collection",
    )

    assert result == INCOMPLETE_EXIT_CODE
    collected = map_destination(out_root, "sub-01")
    assert collected.read_bytes() == b"one"
    manifest = (out_root / "collection" / "charm_segmentation_manifest.tsv").read_text(
        encoding="utf-8"
    )
    assert "sub-01\tcomplete" in manifest
    assert "sub-02\tincomplete" in manifest


def test_shell_launchers_pass_syntax_check_and_submitter_throttles_to_50():
    for name in (
        "charm_segmentation_array.slurm",
        "collect_charm_segmentations.slurm",
        "submit_charm_segmentations.sh",
    ):
        subprocess.run(["bash", "-n", str(WORKFLOW / name)], check=True)

    submit_source = (WORKFLOW / "submit_charm_segmentations.sh").read_text(
        encoding="utf-8"
    )
    array_source = (WORKFLOW / "charm_segmentation_array.slurm").read_text(
        encoding="utf-8"
    )
    assert 'MAX_CONCURRENT="${TI_CHARM_MAX_CONCURRENT:-50}"' in submit_source
    assert 'ARRAY_SPEC="0-${ARRAY_END}%${MAX_CONCURRENT}"' in submit_source
    assert "/mnt/parscratch/users/cop23bi/ti_dataset" in submit_source
    assert "#SBATCH --array=" not in array_source
    assert "SimNIBS/4.0.1-foss-2023a" in array_source


def test_submitter_discovers_subjects_and_submits_array_then_collector(tmp_path):
    source_root = tmp_path / "source"
    for subject in ("sub-01", "sub-02", "sub-03"):
        make_subject(source_root, subject)
    incomplete_anat = source_root / "sub-04" / "anat"
    incomplete_anat.mkdir(parents=True)
    (incomplete_anat / "sub-04_T1w.nii.gz").write_bytes(b"t1")

    sbatch_log = tmp_path / "sbatch.log"
    fake_sbatch = tmp_path / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f"printf '%s\\n' \"$*\" >> '{sbatch_log}'\n"
        "if [[ \"$*\" == *'--dependency=afterany:'* ]]; then\n"
        "    echo '22222;testcluster'\n"
        "else\n"
        "    echo '11111;testcluster'\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)
    out_root = tmp_path / "out"
    environment = {
        **os.environ,
        "CAMCAN_DIR": str(ROOT),
        "TI_CHARM_SOURCE_ROOT": str(source_root),
        "TI_CHARM_OUTPUT_ROOT": str(out_root),
        "SBATCH_BIN": str(fake_sbatch),
    }

    result = subprocess.run(
        ["bash", str(WORKFLOW / "submit_charm_segmentations.sh")],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert "Submitted CHARM array job: 11111" in result.stdout
    assert "Submitted dependent collector job: 22222" in result.stdout
    calls = sbatch_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert "--array=0-2%50" in calls[0]
    assert "--dependency=afterany:11111" in calls[1]
    subjects_file = out_root / "submission" / "subjects.txt"
    assert read_subjects_file(subjects_file) == ("sub-01", "sub-02", "sub-03")
    preflight = (out_root / "submission" / "preflight.tsv").read_text(encoding="utf-8")
    assert preflight.count("\n") == 5
    assert "sub-04\t\t\tblocked\t" in preflight
