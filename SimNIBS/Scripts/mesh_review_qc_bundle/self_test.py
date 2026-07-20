#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import urlopen

from mesh_review.discovery import discover_images
from mesh_review.server import MeshReviewHandler, MeshReviewServer
from mesh_review.store import ReviewStore


def _image(root: Path, relative_path: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"self-test image")


class BundleSelfTest(unittest.TestCase):
    def test_discovery_decisions_exports_and_local_server(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mesh_review_self_test_") as tmp_name:
            root = Path(tmp_name)
            images = root / "images"
            state = root / "state"
            _image(images, "tissues/tag_07_compact_bone/01__sub-CC1__front.png")
            _image(images, "tissues_back/tag_07_compact_bone/01__sub-CC1__back.png")
            _image(images, "tissues_top/tag_07_compact_bone/01__sub-CC1__top.png")
            _image(images, "tissues/tag_05_scalp/02__sub-CC2__front.png")

            discovered = discover_images(images)
            self.assertEqual(len(discovered.images), 4)
            self.assertEqual(
                {item.view for item in discovered.images if item.subject_id == "sub-CC1"},
                {"front", "back", "top"},
            )

            store = ReviewStore(images, state, target=1)
            server = None
            server_thread = None
            try:
                store.rescan()
                subject_one = next(
                    item for item in store.queue() if item["subject_id"] == "sub-CC1"
                )
                store.record_decision(subject_one["id"], "decline", "self-test")
                remaining = store.queue()
                self.assertEqual({item["subject_id"] for item in remaining}, {"sub-CC2"})
                store.record_decision(remaining[0]["id"], "accept")
                self.assertEqual(store.stats()["accepted_subjects"], 1)
                exports = store.export_all()
                self.assertEqual(
                    Path(exports["accepted_subjects"]).read_text(encoding="utf-8").strip(),
                    "sub-CC2",
                )

                server = MeshReviewServer(("127.0.0.1", 0), MeshReviewHandler, store)
                server_thread = threading.Thread(target=server.serve_forever, daemon=True)
                server_thread.start()
                port = int(server.server_address[1])
                with urlopen(f"http://127.0.0.1:{port}/api/bootstrap", timeout=5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self.assertEqual(payload["stats"]["accepted_subjects"], 1)
                self.assertEqual(
                    {tissue["slug"] for tissue in payload["tissues"]},
                    {"tag_05_scalp", "tag_07_compact_bone"},
                )
                with urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
                    page = response.read().decode("utf-8")
                self.assertIn("<title>Mesh Review</title>", page)
                self.assertIn('id="firstTissue"', page)
            finally:
                if server is not None:
                    server.shutdown()
                    server.server_close()
                if server_thread is not None:
                    server_thread.join(timeout=5)
                store.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
