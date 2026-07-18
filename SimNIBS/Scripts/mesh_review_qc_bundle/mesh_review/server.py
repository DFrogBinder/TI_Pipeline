#!/usr/bin/env python3
from __future__ import annotations

import argparse
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import shutil
import sys
import threading
from typing import Any
from urllib.parse import parse_qs, urlparse
import webbrowser

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mesh_review.discovery import DEFAULT_SUBJECT_REGEX
    from mesh_review.store import ReviewStore
else:
    from .discovery import DEFAULT_SUBJECT_REGEX
    from .store import ReviewStore


STATIC_DIR = Path(__file__).resolve().parent / "static"
EXPORT_FILES = {
    "accepted_subjects.txt",
    "maybe_subjects.txt",
    "declined_subjects.txt",
    "subject_summary.csv",
    "image_decisions.csv",
    "review_manifest.json",
}


class MeshReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, handler_class, store: ReviewStore):
        super().__init__(server_address, handler_class)
        self.store = store


class MeshReviewHandler(BaseHTTPRequestHandler):
    server: MeshReviewServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/bootstrap":
                self._send_json(
                    {
                        "stats": self.server.store.stats(),
                        "image_root": str(self.server.store.image_root),
                        "state_dir": str(self.server.store.state_dir),
                        "export_dir": str(self.server.store.export_dir),
                    }
                )
                return
            if parsed.path == "/api/queue":
                query = parse_qs(parsed.query)
                mode = query.get("mode", ["new"])[0]
                order = query.get("order", ["subject"])[0]
                self._send_json(
                    {
                        "mode": mode,
                        "order": order,
                        "items": self.server.store.queue(mode=mode, order=order),
                    }
                )
                return
            if parsed.path == "/api/subjects":
                self._send_json({"subjects": self.server.store.subject_summaries()})
                return
            if parsed.path.startswith("/api/image/"):
                image_id = int(parsed.path.rsplit("/", 1)[1])
                self._send_file(self.server.store.image_path(image_id), inline=True)
                return
            if parsed.path.startswith("/api/export/"):
                filename = parsed.path.rsplit("/", 1)[1]
                if filename not in EXPORT_FILES:
                    self._send_error_json(HTTPStatus.NOT_FOUND, "Unknown export file.")
                    return
                path = self.server.store.export_dir / filename
                if not path.is_file():
                    self._send_error_json(
                        HTTPStatus.NOT_FOUND, "Export has not been generated yet."
                    )
                    return
                self._send_file(path, inline=False)
                return
            if parsed.path == "/":
                self._send_static("index.html")
                return
            if parsed.path.startswith("/static/"):
                self._send_static(parsed.path.removeprefix("/static/"))
                return
            self._send_error_json(HTTPStatus.NOT_FOUND, "Not found.")
        except (KeyError, ValueError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/decision":
                result = self.server.store.record_decision(
                    int(payload["image_id"]),
                    str(payload["decision"]),
                    str(payload.get("note", "")),
                )
                self._send_json(result)
                return
            if parsed.path == "/api/undo":
                self._send_json(self.server.store.undo_last())
                return
            if parsed.path == "/api/rescan":
                scan = self.server.store.rescan()
                self._send_json({"scan": scan, "stats": self.server.store.stats()})
                return
            if parsed.path == "/api/export":
                files = self.server.store.export_all()
                self._send_json({"files": files, "stats": self.server.store.stats()})
                return
            self._send_error_json(HTTPStatus.NOT_FOUND, "Not found.")
        except KeyError as exc:
            self._send_error_json(
                HTTPStatus.BAD_REQUEST, f"Missing request field: {exc.args[0]}"
            )
        except (LookupError, ValueError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header.") from exc
        if length > 1_000_000:
            raise ValueError("Request body is too large.")
        if length == 0:
            return {}
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON request body must be an object.")
        return payload

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def _send_static(self, relative_name: str) -> None:
        path = (STATIC_DIR / relative_name).resolve()
        if not path.is_relative_to(STATIC_DIR) or not path.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, "Static file not found.")
            return
        self._send_file(path, inline=True, cache=False)

    def _send_file(self, path: Path, *, inline: bool, cache: bool = True) -> None:
        path = Path(path)
        if not path.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, "File not found.")
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        stat = path.stat()
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(stat.st_size))
        self.send_header(
            "Content-Disposition",
            f"{'inline' if inline else 'attachment'}; filename={json.dumps(path.name)}",
        )
        self.send_header(
            "Cache-Control", "private, max-age=3600" if cache else "no-store"
        )
        self.end_headers()
        with path.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile)

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"[HTTP] {self.address_string()} {format_string % args}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Review tissue-render images one at a time and persist accept, maybe, "
            "or decline decisions by subject."
        )
    )
    parser.add_argument(
        "--images",
        required=True,
        type=Path,
        help="Root directory scanned recursively for images.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        help="SQLite and export directory (default: <images>/.mesh-review).",
    )
    parser.add_argument(
        "--target", type=int, default=200, help="Accepted-subject alert threshold."
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address.")
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="HTTP port; use 0 to choose an available port.",
    )
    parser.add_argument("--subject-regex", default=DEFAULT_SUBJECT_REGEX)
    parser.add_argument("--tissue-regex", default=None)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the local browser automatically.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    image_root = args.images.expanduser().resolve()
    state_dir = (
        args.state_dir.expanduser().resolve()
        if args.state_dir is not None
        else image_root / ".mesh-review"
    )
    store = ReviewStore(
        image_root,
        state_dir,
        target=args.target,
        subject_regex=args.subject_regex,
        tissue_regex=args.tissue_regex,
    )
    try:
        scan = store.rescan()
        handler = partial(MeshReviewHandler)
        server = MeshReviewServer((args.host, args.port), handler, store)
        host, port = server.server_address[:2]
        browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        url = f"http://{browser_host}:{port}/"
        print("[INFO] Mesh render review", flush=True)
        print(f"[INFO] Images:       {image_root}", flush=True)
        print(f"[INFO] State:        {state_dir}", flush=True)
        print(
            f"[INFO] Discovered:   {scan['images']} images across {scan['subjects']} subjects",
            flush=True,
        )
        print(f"[INFO] Skipped:      {scan['skipped']}", flush=True)
        print(f"[INFO] Target:       {args.target} accepted subjects", flush=True)
        print(f"[INFO] Open:         {url}", flush=True)
        if not args.no_browser:
            threading.Timer(0.4, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[INFO] Stopping mesh review server", flush=True)
        finally:
            server.server_close()
            store.export_all()
    finally:
        store.close()


if __name__ == "__main__":
    main()
