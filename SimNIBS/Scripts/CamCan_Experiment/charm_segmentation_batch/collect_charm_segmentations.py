#!/usr/bin/env python3
"""Collect untouched CHARM tissue maps into a flat, transfer-ready folder."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from run_charm_segmentation import (
    INCOMPLETE_EXIT_CODE,
    map_destination,
    metadata_path,
    read_subjects_file,
    sha256_file,
    write_json_atomic,
)


def collect(
    *,
    out_root: Path,
    subjects_file: Path,
    collection_dir: Path,
) -> int:
    subjects = read_subjects_file(subjects_file)
    expected_count = len(subjects)
    if expected_count <= 0:
        print(f"[ERROR] No subjects found in {subjects_file}.", file=sys.stderr)
        return INCOMPLETE_EXIT_CODE

    collection_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    valid_count = 0
    for subject in subjects:
        source_map = map_destination(out_root, subject)
        marker = metadata_path(out_root, subject)
        status = "complete"
        message = "ok"
        expected_hash = ""
        actual_hash = ""
        try:
            if not marker.is_file():
                raise FileNotFoundError(f"missing metadata: {marker}")
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if payload.get("status") != "complete" or payload.get("subject") != subject:
                raise ValueError(f"invalid completion metadata: {marker}")
            if not source_map.is_file() or source_map.stat().st_size <= 0:
                raise FileNotFoundError(f"missing or empty map: {source_map}")
            expected_hash = str(payload.get("archived_map_sha256", ""))
            actual_hash = sha256_file(source_map)
            if not expected_hash or actual_hash != expected_hash:
                raise ValueError(
                    f"map hash does not match metadata: {actual_hash} != {expected_hash}"
                )
            valid_count += 1
        except (OSError, ValueError, TypeError) as exc:
            status = "incomplete"
            message = str(exc)
        rows.append(
            {
                "subject": subject,
                "status": status,
                "source_map": str(source_map),
                "collected_map": str(source_map) if status == "complete" else "",
                "sha256": actual_hash,
                "bytes": source_map.stat().st_size if source_map.is_file() else "",
                "message": message,
            }
        )

    manifest = collection_dir / "charm_segmentation_manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "subject",
            "status",
            "source_map",
            "collected_map",
            "sha256",
            "bytes",
            "message",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    checksums = collection_dir / "sha256sums.txt"
    checksums.write_text(
        "".join(
            f"{row['sha256']}  {row['collected_map']}\n"
            for row in rows
            if row["status"] == "complete"
        ),
        encoding="utf-8",
    )
    complete = valid_count == expected_count
    summary = {
        "status": "complete" if complete else "incomplete",
        "expected_subjects": expected_count,
        "valid_maps": valid_count,
        "missing_or_invalid_maps": expected_count - valid_count,
        "collection_dir": str(collection_dir),
        "maps_dir": str(out_root / "maps"),
        "manifest": str(manifest),
        "checksums": str(checksums),
        "maps_are_byte_for_byte_copies": True,
        "created_at_epoch": time.time(),
    }
    write_json_atomic(collection_dir / "collection_summary.json", summary)

    print(
        f"[INFO] CHARM collection: valid={valid_count}/{expected_count} "
        f"collection={collection_dir}"
    )
    if not complete:
        print(f"[ERROR] Collection is incomplete; inspect {manifest}.", file=sys.stderr)
        return INCOMPLETE_EXIT_CODE
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--subjects-file", required=True)
    parser.add_argument("--collection-dir")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    out_root = Path(args.out_root).expanduser().resolve()
    collection_dir = (
        Path(args.collection_dir).expanduser().resolve()
        if args.collection_dir
        else out_root / "collection"
    )
    try:
        return collect(
            out_root=out_root,
            subjects_file=Path(args.subjects_file).expanduser().resolve(),
            collection_dir=collection_dir,
        )
    except (OSError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return INCOMPLETE_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
