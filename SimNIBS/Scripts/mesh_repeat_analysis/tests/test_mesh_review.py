from __future__ import annotations

import csv
from pathlib import Path

from mesh_repeat_analysis.post.mesh_review.discovery import discover_images
from mesh_repeat_analysis.post.mesh_review.store import ReviewStore


def _image(root: Path, relative_path: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")
    return path


def test_discovery_infers_existing_layout_and_flat_filenames(tmp_path):
    root = tmp_path / "images"
    _image(
        root,
        "Left_Hippocampus_Data_01/renders/tissues_back/tag_07_compact_bone/"
        "00001__sub-CC120001__unknown_repeat__mesh.png",
    )
    _image(root, "flat/sub-CC120002__tag_05_scalp__front.png")
    _image(
        root,
        "Left_Hippocampus_Data_01/renders/tissues_top/tag_07_compact_bone/"
        "00002__sub-CC120003__unknown_repeat__mesh.png",
    )
    _image(root, "flat/no_subject__tag_05_scalp.png")

    result = discover_images(root)

    assert len(result.images) == 3
    assert len(result.skipped) == 1
    by_subject = {image.subject_id: image for image in result.images}
    compact = by_subject["sub-CC120001"]
    assert compact.tissue == "tag_07_compact_bone"
    assert compact.view == "back"
    scalp = by_subject["sub-CC120002"]
    assert scalp.tissue == "tag_05_scalp"
    assert scalp.view == "front"
    compact_top = by_subject["sub-CC120003"]
    assert compact_top.tissue == "tag_07_compact_bone"
    assert compact_top.view == "top"


def test_decline_removes_subject_from_all_remaining_queue_images(tmp_path):
    root = tmp_path / "images"
    _image(root, "tissues/tag_04_bone/001__sub-CC1__front.png")
    _image(root, "tissues_back/tag_04_bone/001__sub-CC1__back.png")
    _image(root, "tissues/tag_05_scalp/002__sub-CC2__front.png")
    state_dir = tmp_path / "state"
    store = ReviewStore(root, state_dir, target=2)
    try:
        store.rescan()
        queue = store.queue()
        subject_one = next(item for item in queue if item["subject_id"] == "sub-CC1")

        result = store.record_decision(subject_one["id"], "decline", "broken skull")

        assert result["subject"]["status"] == "declined"
        assert {item["subject_id"] for item in store.queue()} == {"sub-CC2"}
        assert store.stats()["declined_subjects"] == 1
        assert (
            state_dir / "exports" / "declined_subjects.txt"
        ).read_text().strip() == "sub-CC1"
    finally:
        store.close()


def test_subject_is_accepted_only_after_every_active_image_is_accepted(tmp_path):
    root = tmp_path / "images"
    _image(root, "tag_04_bone/001__sub-CC1__front.png")
    _image(root, "tag_05_scalp/001__sub-CC1__front.png")
    store = ReviewStore(root, tmp_path / "state", target=1)
    try:
        store.rescan()
        queue = store.queue()

        first = store.record_decision(queue[0]["id"], "accept")
        assert first["subject"]["status"] == "in_review"
        assert first["stats"]["accepted_subjects"] == 0
        assert first["target_just_reached"] is False

        second = store.record_decision(queue[1]["id"], "accept")
        assert second["subject"]["status"] == "accepted"
        assert second["stats"]["accepted_subjects"] == 1
        assert second["target_just_reached"] is True
    finally:
        store.close()


def test_queue_can_prioritize_a_selected_tissue_in_both_order_modes(tmp_path):
    root = tmp_path / "images"
    for subject in ("sub-CC1", "sub-CC2"):
        _image(root, f"tag_01_white_matter/{subject}__front.png")
        _image(root, f"tag_02_gray_matter/{subject}__front.png")
        _image(root, f"tag_07_compact_bone/{subject}__front.png")
    store = ReviewStore(root, tmp_path / "state", target=1)
    try:
        store.rescan()

        assert store.queue(order="subject")[0]["tissue"] == "tag_01_white_matter"

        subject_first = store.queue(
            order="subject", first_tissue="tag_07_compact_bone"
        )
        assert [item["tissue"] for item in subject_first[:3]] == [
            "tag_07_compact_bone",
            "tag_01_white_matter",
            "tag_02_gray_matter",
        ]
        assert {item["subject_id"] for item in subject_first[:3]} == {"sub-CC1"}

        tissue_first = store.queue(
            order="tissue", first_tissue="tag_07_compact_bone"
        )
        assert [item["tissue"] for item in tissue_first[:2]] == [
            "tag_07_compact_bone",
            "tag_07_compact_bone",
        ]
        assert [item["subject_id"] for item in tissue_first[:2]] == [
            "sub-CC1",
            "sub-CC2",
        ]

        assert store.tissues() == [
            {"slug": "tag_01_white_matter", "display": "White Matter"},
            {"slug": "tag_02_gray_matter", "display": "Gray Matter"},
            {"slug": "tag_07_compact_bone", "display": "Compact Bone"},
        ]
    finally:
        store.close()


def test_maybe_queue_and_undo_restore_review_state(tmp_path):
    root = tmp_path / "images"
    _image(root, "tag_04_bone/001__sub-CC1__front.png")
    store = ReviewStore(root, tmp_path / "state", target=1)
    try:
        store.rescan()
        image_id = store.queue()[0]["id"]
        store.record_decision(image_id, "maybe", "inspect later")

        assert store.queue() == []
        maybe_queue = store.queue(mode="maybe")
        assert len(maybe_queue) == 1
        assert maybe_queue[0]["note"] == "inspect later"
        assert store.stats()["maybe_subjects"] == 1

        restored = store.undo_last()
        assert restored["restored_decision"] is None
        assert len(store.queue()) == 1
        assert store.stats()["maybe_subjects"] == 0
    finally:
        store.close()


def test_rescan_preserves_decisions_and_reopens_accepted_subject(tmp_path):
    root = tmp_path / "images"
    _image(root, "tag_05_scalp/001__sub-CC1__front.png")
    state_dir = tmp_path / "state"
    store = ReviewStore(root, state_dir, target=1)
    try:
        store.rescan()
        store.record_decision(store.queue()[0]["id"], "accept")
        assert store.stats()["accepted_subjects"] == 1

        _image(root, "tag_04_bone/001__sub-CC1__front.png")
        scan = store.rescan()

        assert scan["images"] == 2
        assert store.stats()["accepted_subjects"] == 0
        assert store.subject_summaries()[0]["status"] == "in_review"
        assert len(store.queue()) == 1
    finally:
        store.close()

    reopened = ReviewStore(root, state_dir, target=1)
    try:
        reopened.rescan()
        summary = reopened.subject_summaries()[0]
        assert summary["accepted_images"] == 1
        assert summary["unreviewed_images"] == 1
    finally:
        reopened.close()


def test_exports_contain_subject_and_image_decision_tables(tmp_path):
    root = tmp_path / "images"
    _image(root, "tag_05_scalp/001__sub-CC1__front.png")
    store = ReviewStore(root, tmp_path / "state", target=1)
    try:
        store.rescan()
        store.record_decision(store.queue()[0]["id"], "accept", "good surface")
        paths = store.export_all()

        assert Path(paths["accepted_subjects"]).read_text().strip() == "sub-CC1"
        with Path(paths["subject_summary"]).open(
            newline="", encoding="utf-8"
        ) as handle:
            subject_rows = list(csv.DictReader(handle))
        assert subject_rows[0]["status"] == "accepted"
        with Path(paths["image_decisions"]).open(
            newline="", encoding="utf-8"
        ) as handle:
            image_rows = list(csv.DictReader(handle))
        assert image_rows[0]["decision"] == "accept"
        assert image_rows[0]["note"] == "good surface"
    finally:
        store.close()


def test_standalone_bundle_runtime_matches_canonical_tool():
    scripts_root = Path(__file__).resolve().parents[2]
    canonical = scripts_root / "mesh_repeat_analysis" / "post" / "mesh_review"
    bundled = scripts_root / "mesh_review_qc_bundle" / "mesh_review"
    runtime_files = (
        "__init__.py",
        "discovery.py",
        "server.py",
        "store.py",
        "static/app.css",
        "static/app.js",
        "static/index.html",
    )

    for relative_path in runtime_files:
        assert (bundled / relative_path).read_bytes() == (
            canonical / relative_path
        ).read_bytes(), f"standalone bundle is stale: {relative_path}"
