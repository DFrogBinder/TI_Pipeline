import os
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "migrate_parscratch_to_shared.sh"


def manifest_entries(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_reviewed_dependency_scope_is_complete_and_disjoint():
    inventory = manifest_entries(HERE / "parscratch_inventory.txt")
    active = manifest_entries(HERE / "active_repeatability_exclusions.txt")
    core = manifest_entries(HERE / "core_data_exclusions.txt")
    combined = manifest_entries(HERE / "repeatability_migration_exclusions.txt")

    assert len(inventory) == 24
    assert len(active) == 6
    assert len(core) == 9
    assert active.isdisjoint(core)
    assert combined == active | core
    assert combined <= inventory
    assert len(inventory - combined) == 9

    scope_rows = [
        line.split("\t", 3)
        for line in (HERE / "migration_scope.tsv").read_text().splitlines()[1:]
        if line.strip()
    ]
    assert len(scope_rows) == 24
    assert {row[0] for row in scope_rows} == inventory
    kept = {row[0] for row in scope_rows if row[2] == "keep_parscratch"}
    migrated = {row[0] for row in scope_rows if row[2] == "migrate"}
    assert kept == combined
    assert migrated == inventory - combined


def make_inventory(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    manifest = tmp_path / "manifest.txt"
    source.mkdir()
    destination.mkdir()
    entries = [f"entry_{index:02d}" for index in range(24)]
    for entry in entries:
        entry_dir = source / entry
        entry_dir.mkdir()
        (entry_dir / "payload.txt").write_text(f"payload for {entry}\n")
    manifest.write_text("\n".join(entries) + "\n")
    return source, destination, manifest, entries[-1]


def run_script(
    source: Path,
    destination: Path,
    manifest: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "TI_MIGRATION_SOURCE_ROOT": str(source),
            "TI_MIGRATION_DEST_ROOT": str(destination),
            "TI_MIGRATION_MANIFEST": str(manifest),
        }
    )
    return subprocess.run(
        ["bash", str(SCRIPT), *arguments],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_copy_verify_checksum_and_exclusion(tmp_path):
    source, destination, manifest, excluded = make_inventory(tmp_path)

    copied = run_script(
        source,
        destination,
        manifest,
        "copy",
        "--exclude",
        excluded,
        "--confirm-destination",
        str(destination),
    )
    assert copied.returncode == 0, copied.stderr
    assert not (destination / excluded).exists()
    assert (destination / "entry_00" / "payload.txt").is_file()

    verified = run_script(
        source, destination, manifest, "verify", "--exclude", excluded
    )
    assert verified.returncode == 0, verified.stderr
    assert verified.stdout.count("\tmatch\tverify") == 23

    checksummed = run_script(
        source, destination, manifest, "checksum", "--exclude", excluded
    )
    assert checksummed.returncode == 0, checksummed.stderr
    assert checksummed.stdout.count("\tmatch\tchecksum") == 23


def test_inventory_drift_blocks_copy(tmp_path):
    source, destination, manifest, excluded = make_inventory(tmp_path)
    (source / "unreviewed").mkdir()

    completed = run_script(
        source,
        destination,
        manifest,
        "copy",
        "--exclude",
        excluded,
        "--confirm-destination",
        str(destination),
    )

    assert completed.returncode == 2
    assert "Unreviewed source entries" in completed.stderr
    assert not (destination / "entry_00").exists()


def test_exclusion_file_supports_multiple_active_roots(tmp_path):
    source, destination, manifest, _ = make_inventory(tmp_path)
    exclusion_file = tmp_path / "excluded.txt"
    exclusion_file.write_text("# active roots\nentry_22\nentry_23\n")

    copied = run_script(
        source,
        destination,
        manifest,
        "copy",
        "--exclude-file",
        str(exclusion_file),
        "--confirm-destination",
        str(destination),
    )

    assert copied.returncode == 0, copied.stderr
    assert "top-level entries included: 22" in copied.stdout
    assert not (destination / "entry_22").exists()
    assert not (destination / "entry_23").exists()
    assert (destination / "entry_21" / "payload.txt").is_file()


def test_copy_requires_exact_destination_confirmation(tmp_path):
    source, destination, manifest, excluded = make_inventory(tmp_path)

    completed = run_script(
        source,
        destination,
        manifest,
        "copy",
        "--exclude",
        excluded,
        "--confirm-destination",
        str(destination) + "-typo",
    )

    assert completed.returncode == 2
    assert "copy requires --confirm-destination" in completed.stderr
    assert not (destination / "entry_00").exists()


def test_verification_detects_changed_destination(tmp_path):
    source, destination, manifest, excluded = make_inventory(tmp_path)
    copied = run_script(
        source,
        destination,
        manifest,
        "copy",
        "--exclude",
        excluded,
        "--confirm-destination",
        str(destination),
    )
    assert copied.returncode == 0, copied.stderr
    (destination / "entry_00" / "payload.txt").write_text("changed\n")

    verified = run_script(
        source, destination, manifest, "verify", "--exclude", excluded
    )

    assert verified.returncode == 2
    assert "entry_00\tdifferent\tverify" in verified.stdout


def test_copy_allows_neuroimaging_outputs_in_migratable_root(tmp_path):
    source, destination, manifest, excluded = make_inventory(tmp_path)
    (source / "entry_00" / "subject_T1w.nii.gz").write_bytes(b"not-a-real-nifti")

    completed = run_script(
        source,
        destination,
        manifest,
        "copy",
        "--exclude",
        excluded,
        "--confirm-destination",
        str(destination),
    )

    assert completed.returncode == 0, completed.stderr
    assert (destination / "entry_00" / "subject_T1w.nii.gz").read_bytes() == b"not-a-real-nifti"
