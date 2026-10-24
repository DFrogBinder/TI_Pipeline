"""Receipt-bound source audit and immutable subject selection."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import tarfile
import tempfile
from typing import Callable, Optional

from .settings import BASE9_SUBJECTS, EXPECTED_ARTIFACTS, ReviewPaths


SCHEMA_VERSION = 1
WORKFLOW = "simnibs_headmodel_supervisor_review"


class ReviewPackageError(RuntimeError):
    """Raised when source or package provenance cannot be proved."""


@dataclass(frozen=True)
class ArtifactSource:
    key: str
    destination_name: str
    source_path: Path
    source_sha256: str
    expected_sha256: str
    mode: str
    description: str
    software_version: str
    provenance_paths: tuple[Path, ...]
    reference_path: Path | None = None
    reference_sha256: str = ""
    metadata: dict[str, object] | None = None

    def payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["source_path"] = str(self.source_path)
        payload["provenance"] = [
            {"path": str(path), "sha256": _sha256(path)}
            for path in self.provenance_paths
        ]
        payload.pop("provenance_paths", None)
        payload["reference_path"] = (
            str(self.reference_path) if self.reference_path is not None else None
        )
        return payload


@dataclass(frozen=True)
class SubjectSources:
    subject: str
    artifacts: tuple[ArtifactSource, ...]

    def payload(self) -> dict[str, object]:
        return {
            "subject": self.subject,
            "artifact_count": len(self.artifacts),
            "artifacts": [artifact.payload() for artifact in self.artifacts],
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, *, root: Path, label: str) -> dict[str, object]:
    _regular_file(path, root=root, label=label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewPackageError(f"Unreadable {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReviewPackageError(f"{label} is not a JSON object: {path}")
    return payload


def _inside(path: Path, root: Path) -> bool:
    resolved = path.resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    return resolved == resolved_root or resolved_root in resolved.parents


def _regular_file(path: Path, *, root: Path, label: str) -> Path:
    if not _inside(path, root):
        raise ReviewPackageError(f"{label} is outside protected root {root}: {path}")
    if path.is_symlink():
        raise ReviewPackageError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ReviewPackageError(f"Missing {label}: {path}") from exc
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ReviewPackageError(f"{label} is not a non-empty regular file: {resolved}")
    if not _inside(resolved, root):
        raise ReviewPackageError(
            f"{label} resolves outside protected root {root}: {resolved}"
        )
    return resolved


def _receipt_header(
    payload: dict[str, object], *, subject: str, status: str, label: str
) -> None:
    if payload.get("status") != status or payload.get("subject") != subject:
        raise ReviewPackageError(
            f"{label} is not bound to {subject} with status {status!r}"
        )


def _hash_value(value: object, *, label: str) -> str:
    result = str(value or "").lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ReviewPackageError(f"{label} has no valid SHA-256")
    return result


def _exact_recorded_file(
    raw_path: object,
    *,
    expected_path: Path,
    root: Path,
    label: str,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ReviewPackageError(f"{label} receipt path is missing")
    recorded = Path(raw_path).expanduser()
    resolved = _regular_file(recorded, root=root, label=label)
    expected = _regular_file(expected_path, root=root, label=label)
    if resolved != expected:
        raise ReviewPackageError(
            f"{label} receipt names {resolved}, expected canonical path {expected}"
        )
    return resolved


def _verified_hash(path: Path, expected: str, *, label: str) -> str:
    actual = _sha256(path)
    if actual != expected:
        raise ReviewPackageError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {actual}"
        )
    return actual


def _artifact(
    *,
    key: str,
    destination_name: str,
    source_path: Path,
    expected_sha256: str,
    mode: str,
    description: str,
    software_version: str,
    provenance_paths: tuple[Path, ...],
    reference_path: Path | None = None,
    reference_sha256: str = "",
    metadata: dict[str, object] | None = None,
) -> ArtifactSource:
    actual = _verified_hash(source_path, expected_sha256, label=key)
    return ArtifactSource(
        key=key,
        destination_name=destination_name,
        source_path=source_path,
        source_sha256=actual,
        expected_sha256=expected_sha256,
        mode=mode,
        description=description,
        software_version=software_version,
        provenance_paths=provenance_paths,
        reference_path=reference_path,
        reference_sha256=reference_sha256,
        metadata=metadata or {},
    )


def _v4_selection(
    paths: ReviewPaths, subject: str
) -> tuple[Path, str, Path, Path, dict[str, object]]:
    selection_csv = (
        paths.completed_v4_left_root
        / "_pipeline"
        / "median_mesh_selection"
        / "median_representative_remesh_repeats.csv"
    )
    selection_csv = _regular_file(
        selection_csv,
        root=paths.completed_v4_left_root,
        label="v4 median-mesh selection receipt",
    )
    completion = paths.completed_v4_left_root / "_pipeline" / "workflow" / "complete.json"
    completion_payload = _read_json(
        completion,
        root=paths.completed_v4_left_root,
        label="v4 completion receipt",
    )
    if completion_payload.get("status") != "complete":
        raise ReviewPackageError("v4 left campaign completion receipt is not complete")
    with selection_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("subject") == subject]
    if len(rows) != 1 or rows[0].get("selection_status") != "selected":
        raise ReviewPackageError(f"v4 median selection is not unique and selected for {subject}")
    row = rows[0]
    mesh = _regular_file(
        Path(row.get("selected_mesh_path", "")),
        root=paths.completed_v4_left_root,
        label="v4 median mesh",
    )
    checksum = _hash_value(row.get("mesh_checksum"), label="v4 median mesh receipt")
    completion = completion.resolve()
    return mesh, checksum, selection_csv, completion, {
        "selected_repeat_tag": row.get("selected_repeat_tag", ""),
        "selection_receipt": str(selection_csv),
        "completion_receipt": str(completion),
    }


def audit_subject(paths: ReviewPaths, subject: str) -> SubjectSources:
    """Resolve and hash the eight approved artifacts for one base-nine subject."""
    if subject not in BASE9_SUBJECTS:
        raise ReviewPackageError(f"Subject is not in the canonical base-nine cohort: {subject}")

    original_map_expected = (
        paths.original_charm_root
        / "maps"
        / f"{subject}_CHARM_tissue_labeling_upsampled.nii.gz"
    )
    original_receipt_path = paths.original_charm_root / "metadata" / f"{subject}.json"
    original_receipt = _read_json(
        original_receipt_path,
        root=paths.original_charm_root,
        label="original CHARM receipt",
    )
    _receipt_header(
        original_receipt, subject=subject, status="complete", label="original CHARM receipt"
    )
    original_map = _exact_recorded_file(
        original_receipt.get("archived_map"),
        expected_path=original_map_expected,
        root=paths.original_charm_root,
        label="original CHARM map",
    )
    original_hash = _hash_value(
        original_receipt.get("archived_map_sha256"), label="original CHARM receipt"
    )

    corrected_receipt_path = paths.corrected_v4_scaffold_root / "results" / f"{subject}.json"
    corrected_receipt = _read_json(
        corrected_receipt_path,
        root=paths.corrected_v4_scaffold_root,
        label="corrected CHARM scaffold receipt",
    )
    _receipt_header(
        corrected_receipt,
        subject=subject,
        status="complete",
        label="corrected CHARM scaffold receipt",
    )
    corrected_expected = (
        paths.corrected_v4_scaffold_root
        / "subjects"
        / subject
        / "anat"
        / f"m2m_{subject}"
        / "label_prep"
        / "tissue_labeling_upsampled.nii.gz"
    )
    corrected_map = _exact_recorded_file(
        corrected_receipt.get("installed_label"),
        expected_path=corrected_expected,
        root=paths.corrected_v4_scaffold_root,
        label="corrected CHARM map",
    )
    corrected_hash = _hash_value(
        corrected_receipt.get("installed_label_sha256"),
        label="corrected CHARM scaffold receipt",
    )

    legacy_anat = paths.legacy_v3_root / subject / "anat"
    legacy_marker_path = legacy_anat / ".simnibs326_scaffold_ready.json"
    legacy_marker = _read_json(
        legacy_marker_path,
        root=paths.legacy_v3_root,
        label="original headreco ready marker",
    )
    _receipt_header(
        legacy_marker,
        subject=subject,
        status="ready",
        label="original headreco ready marker",
    )
    if legacy_marker.get("simnibs_module") != "SimNIBS/3.2.6-foss-2023a":
        raise ReviewPackageError("original headreco marker is not pinned to SimNIBS 3.2.6")
    legacy_mesh_expected = legacy_anat / f"{subject}.msh"
    legacy_mesh = _exact_recorded_file(
        legacy_marker.get("native_mesh"),
        expected_path=legacy_mesh_expected,
        root=paths.legacy_v3_root,
        label="original headreco mesh",
    )
    legacy_mesh_hash = _hash_value(
        legacy_marker.get("native_mesh_sha256"), label="original headreco marker"
    )
    legacy_reference = _regular_file(
        legacy_anat / f"m2m_{subject}" / "T1fs_conform.nii.gz",
        root=paths.legacy_v3_root,
        label="original headreco T1fs reference",
    )
    legacy_reference_hash = _sha256(legacy_reference)

    harmonized_map_expected = (
        paths.harmonized_v3_root
        / "harmonized_labels"
        / f"{subject}_v3_harmonized_t1fs_grid.nii.gz"
    )
    harmonization_qc_path = (
        paths.harmonized_v3_root / "harmonized_labels" / "qc" / f"{subject}.json"
    )
    headmodel_receipt_path = (
        paths.harmonized_v3_root
        / "_pipeline"
        / "subjects"
        / subject
        / "headmodel_complete.json"
    )
    headmodel_receipt = _read_json(
        headmodel_receipt_path,
        root=paths.harmonized_v3_root,
        label="harmonized v3 head-model receipt",
    )
    _receipt_header(
        headmodel_receipt,
        subject=subject,
        status="complete",
        label="harmonized v3 head-model receipt",
    )
    api_probe = headmodel_receipt.get("api_probe")
    if not isinstance(api_probe, dict) or api_probe.get("simnibs_version") != "3.2.6":
        raise ReviewPackageError("harmonized head-model receipt is not SimNIBS 3.2.6")
    harmonized_record = headmodel_receipt.get("harmonized_map")
    mesh_record = headmodel_receipt.get("mesh")
    qc_record = headmodel_receipt.get("harmonization_receipt")
    if not all(isinstance(record, dict) for record in (harmonized_record, mesh_record, qc_record)):
        raise ReviewPackageError("harmonized v3 receipt lacks map, mesh, or QC records")
    assert isinstance(harmonized_record, dict)
    assert isinstance(mesh_record, dict)
    assert isinstance(qc_record, dict)
    harmonized_map = _exact_recorded_file(
        harmonized_record.get("path"),
        expected_path=harmonized_map_expected,
        root=paths.harmonized_v3_root,
        label="harmonized v3 map",
    )
    harmonized_hash = _hash_value(
        harmonized_record.get("sha256"), label="harmonized v3 map receipt"
    )
    harmonization_qc = _exact_recorded_file(
        qc_record.get("path"),
        expected_path=harmonization_qc_path,
        root=paths.harmonized_v3_root,
        label="harmonization QC receipt",
    )
    _verified_hash(
        harmonization_qc,
        _hash_value(qc_record.get("sha256"), label="harmonization QC record"),
        label="harmonization QC receipt",
    )
    harmonized_mesh_expected = (
        paths.harmonized_v3_root / subject / "anat" / f"{subject}.msh"
    )
    harmonized_mesh = _exact_recorded_file(
        mesh_record.get("path"),
        expected_path=harmonized_mesh_expected,
        root=paths.harmonized_v3_root,
        label="harmonized v3 mesh",
    )
    harmonized_mesh_hash = _hash_value(
        mesh_record.get("sha256"), label="harmonized v3 mesh receipt"
    )

    atlas = _regular_file(
        paths.atlas_root / f"{subject}.nii.gz",
        root=paths.atlas_root,
        label="subject atlas",
    )
    atlas_hash = _sha256(atlas)
    (
        v4_mesh,
        v4_hash,
        v4_selection_receipt,
        v4_completion_receipt,
        v4_metadata,
    ) = _v4_selection(paths, subject)

    artifacts = (
        _artifact(
            key="original_charm_labels",
            destination_name="01_original_charm_tissue_labels.nii.gz",
            source_path=original_map,
            expected_sha256=original_hash,
            mode="copy",
            description="Untouched CHARM-only tissue-label map.",
            software_version="SimNIBS 4.0.1",
            provenance_paths=(original_receipt_path.resolve(),),
        ),
        _artifact(
            key="corrected_charm_labels",
            destination_name="02_corrected_charm_tissue_labels.nii.gz",
            source_path=corrected_map,
            expected_sha256=corrected_hash,
            mode="copy",
            description="Corrected CHARM map installed in the corrected-v4 scaffold.",
            software_version="SimNIBS 4.0.1 corrected CHARM branch",
            provenance_paths=(corrected_receipt_path.resolve(),),
        ),
        _artifact(
            key="harmonized_v3_labels",
            destination_name="03_harmonized_v3_tissue_labels.nii.gz",
            source_path=harmonized_map,
            expected_sha256=harmonized_hash,
            mode="copy",
            description="Corrected label map harmonised onto the v3 T1fs grid.",
            software_version="SimNIBS 3.2.6 harmonisation workflow",
            provenance_paths=(harmonization_qc.resolve(), headmodel_receipt_path.resolve()),
        ),
        _artifact(
            key="original_headreco_labels",
            destination_name="04_original_headreco_mesh_tissue_labels.nii.gz",
            source_path=legacy_mesh,
            expected_sha256=legacy_mesh_hash,
            mode="mesh_tags_to_nifti",
            description=(
                "Derived combined tissue-label volume exported from original "
                "headreco mesh tetrahedral tags onto T1fs_conform."
            ),
            software_version="SimNIBS 3.2.6",
            provenance_paths=(legacy_marker_path.resolve(),),
            reference_path=legacy_reference,
            reference_sha256=legacy_reference_hash,
        ),
        _artifact(
            key="subject_atlas",
            destination_name="05_subject_atlas.nii.gz",
            source_path=atlas,
            expected_sha256=atlas_hash,
            mode="copy",
            description="Subject-specific anatomical atlas used by the campaigns.",
            software_version="source atlas",
            provenance_paths=(),
        ),
        _artifact(
            key="v3_original_headreco_mesh",
            destination_name="06_v3_original_headreco_mesh.msh",
            source_path=legacy_mesh,
            expected_sha256=legacy_mesh_hash,
            mode="copy",
            description="Native v3 mesh generated from original headreco segmentation.",
            software_version="SimNIBS 3.2.6",
            provenance_paths=(legacy_marker_path.resolve(),),
        ),
        _artifact(
            key="v3_harmonized_mesh",
            destination_name="07_v3_harmonized_mesh.msh",
            source_path=harmonized_mesh,
            expected_sha256=harmonized_mesh_hash,
            mode="copy",
            description="Native v3 mesh generated from the retained harmonised map.",
            software_version="SimNIBS 3.2.6",
            provenance_paths=(headmodel_receipt_path.resolve(),),
        ),
        _artifact(
            key="v4_corrected_charm_median_mesh",
            destination_name="08_v4_corrected_charm_median_mesh.msh",
            source_path=v4_mesh,
            expected_sha256=v4_hash,
            mode="copy",
            description=(
                "Median representative corrected-CHARM mesh from the completed "
                "left-hippocampus v4 remesh campaign."
            ),
            software_version="SimNIBS 4.0.1",
            provenance_paths=(
                v4_selection_receipt.resolve(),
                v4_completion_receipt,
            ),
            metadata=v4_metadata,
        ),
    )
    if len(artifacts) != EXPECTED_ARTIFACTS:
        raise ReviewPackageError(
            f"Internal artifact count is {len(artifacts)}, expected {EXPECTED_ARTIFACTS}"
        )
    return SubjectSources(subject=subject, artifacts=artifacts)


def _validate_root_isolation(paths: ReviewPaths) -> None:
    for source_root in paths.protected_roots:
        if _inside(paths.output_root, source_root) or _inside(source_root, paths.output_root):
            raise ReviewPackageError(
                f"Writable output root overlaps protected source root: "
                f"output={paths.output_root}; source={source_root}"
            )


def preflight(paths: ReviewPaths) -> dict[str, object]:
    """Hash-audit all nine subjects without writing output state."""
    _validate_root_isolation(paths)
    subjects: list[dict[str, object]] = []
    issues: list[str] = []
    for subject in BASE9_SUBJECTS:
        try:
            subjects.append(audit_subject(paths, subject).payload())
        except ReviewPackageError as exc:
            issues.append(f"{subject}: {exc}")
    if issues:
        raise ReviewPackageError(
            "All nine subjects must be complete before random selection: "
            + "; ".join(issues)
        )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "status": "ready",
        "candidate_subjects": list(BASE9_SUBJECTS),
        "candidate_count": len(BASE9_SUBJECTS),
        "artifact_count_per_subject": EXPECTED_ARTIFACTS,
        "subjects": subjects,
        "source_outputs_modified": False,
    }
    payload["audit_sha256"] = _json_sha256(payload)
    return payload


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _exclusive_json(path: Path, payload: dict[str, object]) -> bool:
    """Atomically install JSON without replacing a concurrent immutable owner."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def load_selection(
    paths: ReviewPaths, *, verify_current: bool = False
) -> dict[str, object]:
    receipt = _read_json(
        paths.selection_receipt,
        root=paths.output_root,
        label="selection receipt",
    )
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("workflow") != WORKFLOW
        or receipt.get("status") != "selected"
        or receipt.get("candidate_subjects") != list(BASE9_SUBJECTS)
        or receipt.get("candidate_count") != len(BASE9_SUBJECTS)
        or receipt.get("artifact_count") != EXPECTED_ARTIFACTS
        or receipt.get("selected_subject") not in BASE9_SUBJECTS
    ):
        raise ReviewPackageError("selection receipt has an invalid or stale binding")
    if verify_current:
        current = preflight(paths)
        if receipt.get("audit_sha256") != current.get("audit_sha256"):
            raise ReviewPackageError("selection receipt is stale relative to current sources")
    return receipt


def select_subject(
    paths: ReviewPaths,
    *,
    chooser: Callable[[tuple[str, ...]], str] | None = None,
) -> dict[str, object]:
    """Audit all candidates, then create or reuse one immutable random selection."""
    if paths.selection_receipt.exists():
        return load_selection(paths, verify_current=True)
    if paths.output_root.exists() and any(paths.output_root.iterdir()):
        raise ReviewPackageError(
            f"Output root is non-empty and has no owned selection receipt: {paths.output_root}"
        )
    audit = preflight(paths)
    select = chooser or secrets.choice
    selected = select(BASE9_SUBJECTS)
    if selected not in BASE9_SUBJECTS:
        raise ReviewPackageError(f"Random selector returned an ineligible subject: {selected}")
    receipt: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "status": "selected",
        "created_utc": _utc_now(),
        "selection_method": (
            "secrets.choice over canonical ordered BASE9_SUBJECTS after full hash audit"
        ),
        "selected_subject": selected,
        "candidate_subjects": list(BASE9_SUBJECTS),
        "candidate_count": len(BASE9_SUBJECTS),
        "artifact_count": EXPECTED_ARTIFACTS,
        "audit_sha256": audit["audit_sha256"],
        "source_audit": audit,
        "source_outputs_modified": False,
    }
    if not _exclusive_json(paths.selection_receipt, receipt):
        return load_selection(paths, verify_current=True)
    return load_selection(paths, verify_current=True)


LabelExporter = Callable[[Path, Path, Path], Optional[Path]]


def _default_label_exporter(
    mesh_path: Path, reference_path: Path, output_path: Path
) -> Path:
    """Load the SimNIBS-dependent exporter only inside the HPC build process."""
    from simnibs_326_repeatability.simulation_runner import (  # noqa: PLC0415
        _write_label_volume_326,
    )

    _write_label_volume_326(mesh_path, reference_path, output_path)
    return output_path


def _readme(subject: str) -> str:
    return f"""# SimNIBS head-model supervisor review package

Selected subject: `{subject}`

This package contains eight receipt-bound scientific artifacts from one subject
randomly selected after all nine retained subjects passed the same source audit.
The numbered filenames correspond to the records in `manifest.json`.

The original headreco tissue-label map (item 04) is a derived combined label
volume: SimNIBS 3.2.6 tetrahedral tissue tags from the original headreco mesh
were assigned onto that subject's `T1fs_conform.nii.gz` grid. It is not a raw
CAT12 segmentation. Items 06 and 07 are distinct SimNIBS 3.2.6 meshes: item 06
uses the original headreco segmentation and item 07 uses the retained harmonised
map. Item 08 is the receipt-selected median representative mesh from the
completed corrected-CHARM SimNIBS 4 left-hippocampus remesh campaign.

`checksums.sha256` verifies every packaged file other than the checksum list
itself. `manifest.json` records source paths, source hashes, derivation mode,
software versions, references, and upstream provenance receipts. The material
is intended for methods/supervisor review and is not a clinical result.
"""


def _package_file(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise ReviewPackageError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ReviewPackageError(f"Missing {label}: {path}") from exc
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ReviewPackageError(f"{label} is not a non-empty regular file: {path}")
    return resolved


def _manifest_artifact(
    artifact: ArtifactSource, *, output_path: Path, output_sha256: str
) -> dict[str, object]:
    payload = artifact.payload()
    payload.update(
        {
            "destination": artifact.destination_name,
            "output_size_bytes": output_path.stat().st_size,
            "output_sha256": output_sha256,
        }
    )
    payload.pop("destination_name", None)
    return payload


def _write_checksums(package_dir: Path, filenames: tuple[str, ...]) -> str:
    lines = [f"{_sha256(package_dir / name)}  {name}" for name in filenames]
    content = "\n".join(lines) + "\n"
    (package_dir / "checksums.sha256").write_text(content, encoding="utf-8")
    return content


def _build_archive(source_dir: Path, archive_path: Path) -> None:
    with tarfile.open(archive_path, mode="w:gz") as archive:
        for path in sorted(source_dir.iterdir(), key=lambda item: item.name):
            archive.add(path, arcname=f"package/{path.name}", recursive=False)


def build_package(
    paths: ReviewPaths, *, exporter: LabelExporter | None = None
) -> dict[str, object]:
    """Build the selected eight-artifact package without mutating any source."""
    _validate_root_isolation(paths)
    if paths.completion_receipt.exists():
        validated = validate_package(paths)
        return {
            "action": "reused",
            "status": validated["status"],
            "subject": validated["subject"],
            "package_dir": str(paths.package_dir),
            "archive_path": str(paths.archive_path),
        }
    if paths.package_dir.exists() or paths.archive_path.exists():
        raise ReviewPackageError(
            "Refusing to overwrite a partial or unowned package output"
        )

    selection = load_selection(paths, verify_current=True)
    subject = str(selection["selected_subject"])
    sources = audit_subject(paths, subject)
    known_root_entries = {paths.selection_receipt.name}
    unexpected = [
        path.name
        for path in paths.output_root.iterdir()
        if path.name not in known_root_entries
    ]
    if unexpected:
        raise ReviewPackageError(
            "Refusing to overwrite a partial or unowned output root: "
            + ", ".join(sorted(unexpected))
        )

    paths.output_root.mkdir(parents=True, exist_ok=True)
    build_root = Path(
        tempfile.mkdtemp(prefix=".review-package-build-", dir=paths.output_root)
    )
    temporary_package = build_root / "package"
    temporary_archive = build_root / paths.archive_path.name
    temporary_package.mkdir()
    label_exporter = exporter or _default_label_exporter
    try:
        manifest_artifacts: list[dict[str, object]] = []
        for artifact in sources.artifacts:
            destination = temporary_package / artifact.destination_name
            if artifact.mode == "copy":
                shutil.copy2(artifact.source_path, destination)
            elif artifact.mode == "mesh_tags_to_nifti":
                if artifact.reference_path is None:
                    raise ReviewPackageError(
                        f"Derived artifact lacks a reference: {artifact.key}"
                    )
                label_exporter(
                    artifact.source_path, artifact.reference_path, destination
                )
            else:
                raise ReviewPackageError(
                    f"Unsupported materialization mode for {artifact.key}: "
                    f"{artifact.mode}"
                )
            output = _package_file(destination, label=artifact.key)
            output_hash = _sha256(output)
            if artifact.mode == "copy" and output_hash != artifact.source_sha256:
                raise ReviewPackageError(
                    f"Copied artifact checksum mismatch for {artifact.key}"
                )
            manifest_artifacts.append(
                _manifest_artifact(
                    artifact, output_path=output, output_sha256=output_hash
                )
            )

        readme_path = temporary_package / "README.md"
        readme_path.write_text(_readme(subject), encoding="utf-8")
        selection_sha256 = _sha256(paths.selection_receipt)
        manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "workflow": WORKFLOW,
            "status": "complete",
            "created_utc": _utc_now(),
            "subject": subject,
            "scientific_artifact_count": EXPECTED_ARTIFACTS,
            "selection_receipt": str(paths.selection_receipt),
            "selection_receipt_sha256": selection_sha256,
            "source_outputs_modified": False,
            "artifacts": manifest_artifacts,
        }
        manifest_path = temporary_package / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checksum_targets = tuple(
            [artifact.destination_name for artifact in sources.artifacts]
            + ["README.md", "manifest.json"]
        )
        _write_checksums(temporary_package, checksum_targets)
        expected_package_names = set(checksum_targets) | {"checksums.sha256"}
        observed_names = {path.name for path in temporary_package.iterdir()}
        if observed_names != expected_package_names:
            raise ReviewPackageError(
                "Temporary package does not contain the exact approved file set"
            )
        _build_archive(temporary_package, temporary_archive)
        _package_file(temporary_archive, label="package archive")

        os.replace(temporary_package, paths.package_dir)
        os.replace(temporary_archive, paths.archive_path)
        completion: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "workflow": WORKFLOW,
            "status": "complete",
            "created_utc": _utc_now(),
            "subject": subject,
            "scientific_artifact_count": EXPECTED_ARTIFACTS,
            "selection_receipt_sha256": selection_sha256,
            "manifest_sha256": _sha256(paths.package_dir / "manifest.json"),
            "checksums_sha256": _sha256(paths.package_dir / "checksums.sha256"),
            "archive_sha256": _sha256(paths.archive_path),
            "package_dir": str(paths.package_dir),
            "archive_path": str(paths.archive_path),
            "source_outputs_modified": False,
        }
        _atomic_json(paths.completion_receipt, completion)
        validated = validate_package(paths)
        return {
            "action": "built",
            "status": validated["status"],
            "subject": subject,
            "package_dir": str(paths.package_dir),
            "archive_path": str(paths.archive_path),
            "completion_receipt": str(paths.completion_receipt),
        }
    finally:
        shutil.rmtree(build_root, ignore_errors=True)


def _checksum_rows(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or parts[1] in rows:
            raise ReviewPackageError("checksums.sha256 has an invalid row")
        rows[parts[1]] = _hash_value(parts[0], label="packaged checksum")
    return rows


def validate_package(paths: ReviewPaths) -> dict[str, object]:
    """Validate selection binding, exact contents, hashes, and archive members."""
    _validate_root_isolation(paths)
    completion = _read_json(
        paths.completion_receipt,
        root=paths.output_root,
        label="completion receipt",
    )
    selection = load_selection(paths, verify_current=False)
    subject = selection.get("selected_subject")
    if (
        completion.get("schema_version") != SCHEMA_VERSION
        or completion.get("workflow") != WORKFLOW
        or completion.get("status") != "complete"
        or completion.get("subject") != subject
        or completion.get("scientific_artifact_count") != EXPECTED_ARTIFACTS
        or completion.get("source_outputs_modified") is not False
    ):
        raise ReviewPackageError("completion receipt has an invalid package binding")
    if paths.package_dir.is_symlink() or not paths.package_dir.is_dir():
        raise ReviewPackageError(f"Package directory is missing: {paths.package_dir}")
    archive = _package_file(paths.archive_path, label="package archive")

    expected_artifacts = {
        "01_original_charm_tissue_labels.nii.gz",
        "02_corrected_charm_tissue_labels.nii.gz",
        "03_harmonized_v3_tissue_labels.nii.gz",
        "04_original_headreco_mesh_tissue_labels.nii.gz",
        "05_subject_atlas.nii.gz",
        "06_v3_original_headreco_mesh.msh",
        "07_v3_harmonized_mesh.msh",
        "08_v4_corrected_charm_median_mesh.msh",
    }
    expected_names = expected_artifacts | {
        "README.md",
        "manifest.json",
        "checksums.sha256",
    }
    observed_names = {path.name for path in paths.package_dir.iterdir()}
    if observed_names != expected_names:
        raise ReviewPackageError("Package does not contain the exact approved file set")
    for name in expected_names:
        _package_file(paths.package_dir / name, label=f"packaged file {name}")

    manifest_path = paths.package_dir / "manifest.json"
    manifest = _read_json(
        manifest_path, root=paths.package_dir, label="package manifest"
    )
    artifacts = manifest.get("artifacts")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("workflow") != WORKFLOW
        or manifest.get("status") != "complete"
        or manifest.get("subject") != subject
        or manifest.get("scientific_artifact_count") != EXPECTED_ARTIFACTS
        or manifest.get("source_outputs_modified") is not False
        or not isinstance(artifacts, list)
        or len(artifacts) != EXPECTED_ARTIFACTS
    ):
        raise ReviewPackageError("manifest has an invalid package binding")
    artifact_names = {
        str(row.get("destination"))
        for row in artifacts
        if isinstance(row, dict)
    }
    if artifact_names != expected_artifacts:
        raise ReviewPackageError("manifest artifact set is invalid")

    checksum_path = paths.package_dir / "checksums.sha256"
    checksum_rows = _checksum_rows(checksum_path)
    checksum_targets = expected_names - {"checksums.sha256"}
    if set(checksum_rows) != checksum_targets:
        raise ReviewPackageError("checksums.sha256 does not cover the exact package")
    for name, expected_hash in checksum_rows.items():
        observed_hash = _sha256(paths.package_dir / name)
        if observed_hash != expected_hash:
            raise ReviewPackageError(f"Packaged file checksum mismatch: {name}")
    for row in artifacts:
        assert isinstance(row, dict)
        destination = str(row["destination"])
        if row.get("output_sha256") != checksum_rows[destination]:
            raise ReviewPackageError(
                f"Manifest checksum mismatch for artifact: {destination}"
            )

    selection_hash = _sha256(paths.selection_receipt)
    receipt_hashes = {
        "selection_receipt_sha256": selection_hash,
        "manifest_sha256": _sha256(manifest_path),
        "checksums_sha256": _sha256(checksum_path),
        "archive_sha256": _sha256(archive),
    }
    for key, observed in receipt_hashes.items():
        if completion.get(key) != observed:
            raise ReviewPackageError(f"Completion receipt checksum mismatch: {key}")
    if manifest.get("selection_receipt_sha256") != selection_hash:
        raise ReviewPackageError("Manifest selection receipt checksum mismatch")

    with tarfile.open(archive, mode="r:gz") as handle:
        members = handle.getmembers()
        archive_names = {member.name for member in members}
        expected_archive_names = {f"package/{name}" for name in expected_names}
        if archive_names != expected_archive_names or not all(
            member.isfile() for member in members
        ):
            raise ReviewPackageError("Archive does not contain the exact package files")
        for member in members:
            extracted = handle.extractfile(member)
            if extracted is None:
                raise ReviewPackageError(f"Archive member cannot be read: {member.name}")
            digest = hashlib.sha256()
            for block in iter(lambda: extracted.read(1024 * 1024), b""):
                digest.update(block)
            package_name = member.name.removeprefix("package/")
            if digest.hexdigest() != _sha256(paths.package_dir / package_name):
                raise ReviewPackageError(
                    f"Archive member checksum mismatch: {member.name}"
                )
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "status": "complete",
        "subject": subject,
        "scientific_artifact_count": EXPECTED_ARTIFACTS,
        "package_dir": str(paths.package_dir),
        "archive_path": str(paths.archive_path),
        "archive_sha256": receipt_hashes["archive_sha256"],
        "source_outputs_modified": False,
    }


def status(paths: ReviewPaths) -> dict[str, object]:
    """Report immutable workflow state without creating or changing files."""
    _validate_root_isolation(paths)
    if paths.completion_receipt.exists():
        payload = validate_package(paths)
        payload["state"] = "complete"
        return payload
    if paths.package_dir.exists() or paths.archive_path.exists():
        raise ReviewPackageError(
            "Package output is partial: scientific output exists without a "
            "valid completion receipt"
        )
    if paths.selection_receipt.exists():
        selection = load_selection(paths, verify_current=False)
        return {
            "schema_version": SCHEMA_VERSION,
            "workflow": WORKFLOW,
            "status": "selected",
            "state": "selected",
            "subject": selection["selected_subject"],
            "candidate_count": len(BASE9_SUBJECTS),
            "scientific_artifact_count": EXPECTED_ARTIFACTS,
            "selection_receipt": str(paths.selection_receipt),
            "source_outputs_modified": False,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "status": "absent",
        "state": "absent",
        "candidate_count": len(BASE9_SUBJECTS),
        "scientific_artifact_count": EXPECTED_ARTIFACTS,
        "source_outputs_modified": False,
    }
