from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import remove_camcan_subjects as cleanup


class RemoveCamcanSubjectsTests(unittest.TestCase):
    def make_subject(self, root: Path, subject_id: str) -> Path:
        subject = root / subject_id
        (subject / "anat").mkdir(parents=True)
        (subject / "anat" / f"{subject_id}_T1w.nii").write_bytes(b"t1")
        return subject

    def make_roots(self) -> tuple[tempfile.TemporaryDirectory[str], Path, Path]:
        temporary = tempfile.TemporaryDirectory()
        base = Path(temporary.name)
        reference = base / "reference"
        full = base / "full"
        reference.mkdir()
        full.mkdir()
        return temporary, reference, full

    def test_audit_plan_matches_only_reference_subjects(self) -> None:
        temporary, reference, full = self.make_roots()
        with temporary:
            for subject_id in ("sub-CC110033", "sub-CC110044"):
                self.make_subject(reference, subject_id)
                self.make_subject(full, subject_id)
            self.make_subject(full, "sub-CC110055")

            plan = cleanup.build_plan(reference, full, None, 3, False)

            self.assertEqual(
                [target.name for target in plan.targets],
                ["sub-CC110033", "sub-CC110044"],
            )
            self.assertEqual(plan.remaining_subjects, 1)
            self.assertTrue((full / "sub-CC110033").is_dir())

    def test_default_cli_mode_is_read_only_for_dynamic_cohort(self) -> None:
        temporary, reference, full = self.make_roots()
        with temporary:
            reference_subject = self.make_subject(reference, "sub-CC110033")
            full_subject = self.make_subject(full, "sub-CC110033")
            output = io.StringIO()

            with redirect_stdout(output):
                return_code = cleanup.main([str(reference), str(full)])

            self.assertEqual(return_code, 0)
            self.assertTrue(reference_subject.is_dir())
            self.assertTrue(full_subject.is_dir())
            self.assertIn("Reference subjects:     1", output.getvalue())
            self.assertIn("AUDIT ONLY: nothing was deleted", output.getvalue())

    def test_apply_deletes_matches_and_writes_manifest(self) -> None:
        temporary, reference, full = self.make_roots()
        with temporary:
            for subject_id in ("sub-CC110033", "sub-CC110044"):
                self.make_subject(reference, subject_id)
                self.make_subject(full, subject_id)
            untouched = self.make_subject(full, "sub-CC110055")
            plan = cleanup.build_plan(reference, full, None, None, False)
            manifest = Path(temporary.name) / "cleanup.tsv"

            return_code = cleanup.apply_plan(plan, manifest)

            self.assertEqual(return_code, 0)
            self.assertFalse((full / "sub-CC110033").exists())
            self.assertFalse((full / "sub-CC110044").exists())
            self.assertTrue(untouched.is_dir())
            manifest_text = manifest.read_text(encoding="utf-8")
            self.assertEqual(manifest_text.count("\tplanned\t"), 2)
            self.assertEqual(manifest_text.count("\tdeleted\t"), 2)

    def test_missing_target_blocks_partial_cleanup(self) -> None:
        temporary, reference, full = self.make_roots()
        with temporary:
            self.make_subject(reference, "sub-CC110033")
            self.make_subject(reference, "sub-CC110044")
            present = self.make_subject(full, "sub-CC110033")

            with self.assertRaisesRegex(ValueError, "absent from the full dataset"):
                cleanup.build_plan(reference, full, None, None, False)

            self.assertTrue(present.is_dir())

    def test_allow_missing_targets_supports_resume(self) -> None:
        temporary, reference, full = self.make_roots()
        with temporary:
            self.make_subject(reference, "sub-CC110033")
            self.make_subject(reference, "sub-CC110044")
            self.make_subject(full, "sub-CC110033")

            plan = cleanup.build_plan(reference, full, None, None, True)

            self.assertEqual([target.name for target in plan.targets], ["sub-CC110033"])
            self.assertEqual(plan.missing_ids, ("sub-CC110044",))

    def test_optional_reference_count_guard_blocks_wrong_cohort(self) -> None:
        temporary, reference, full = self.make_roots()
        with temporary:
            self.make_subject(reference, "sub-CC110033")
            self.make_subject(full, "sub-CC110033")

            with self.assertRaisesRegex(ValueError, "expected exactly 175"):
                cleanup.build_plan(reference, full, 175, None, False)

    def test_reference_count_is_dynamic_by_default(self) -> None:
        temporary, reference, full = self.make_roots()
        with temporary:
            self.make_subject(reference, "sub-CC110033")
            self.make_subject(full, "sub-CC110033")

            plan = cleanup.build_plan(reference, full, None, None, False)

            self.assertEqual(len(plan.reference_ids), 1)
            self.assertEqual(len(plan.targets), 1)

    def test_nested_roots_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            full = Path(temporary) / "full"
            reference = full / "reference"
            reference.mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "must not be inside"):
                cleanup.build_plan(reference, full, None, None, False)

    def test_subject_symlink_is_rejected(self) -> None:
        temporary, reference, full = self.make_roots()
        with temporary:
            self.make_subject(reference, "sub-CC110033")
            external = Path(temporary.name) / "external"
            external.mkdir()
            (full / "sub-CC110033").symlink_to(external, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "symbolic link not allowed"):
                cleanup.build_plan(reference, full, None, None, False)


if __name__ == "__main__":
    unittest.main()
