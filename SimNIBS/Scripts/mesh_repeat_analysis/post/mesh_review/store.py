from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any

from .discovery import DEFAULT_SUBJECT_REGEX, DiscoveryResult, discover_images


DECISIONS = {"accept", "maybe", "decline"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _display_tissue(slug: str) -> str:
    value = slug
    if value.startswith("tag_"):
        parts = value.split("_", 2)
        if len(parts) == 3:
            value = parts[2]
    words = value.replace("_", " ").strip()
    return "CSF" if words.lower() == "csf" else words.title()


class ReviewStore:
    def __init__(
        self,
        image_root: Path,
        state_dir: Path,
        *,
        target: int = 200,
        subject_regex: str = DEFAULT_SUBJECT_REGEX,
        tissue_regex: str | None = None,
    ) -> None:
        if target < 1:
            raise ValueError("Acceptance target must be at least 1.")
        self.image_root = Path(image_root).expanduser().resolve()
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir = self.state_dir / "exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.state_dir / "mesh_review.sqlite3"
        self.target = int(target)
        self.subject_regex = subject_regex
        self.tissue_regex = tissue_regex
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._initialize_schema()
        self._validate_configuration()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _initialize_schema(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY,
                    relative_path TEXT NOT NULL UNIQUE,
                    subject_id TEXT NOT NULL,
                    tissue TEXT NOT NULL,
                    view TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    discovered_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS images_subject_idx
                    ON images(subject_id, active);
                CREATE INDEX IF NOT EXISTS images_tissue_idx
                    ON images(tissue, active);

                CREATE TABLE IF NOT EXISTS decisions (
                    image_id INTEGER PRIMARY KEY REFERENCES images(id),
                    decision TEXT NOT NULL CHECK(decision IN ('accept', 'maybe', 'decline')),
                    note TEXT NOT NULL DEFAULT '',
                    decided_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS decision_history (
                    id INTEGER PRIMARY KEY,
                    image_id INTEGER NOT NULL REFERENCES images(id),
                    previous_decision TEXT,
                    previous_note TEXT,
                    previous_decided_at TEXT,
                    previous_updated_at TEXT,
                    new_decision TEXT NOT NULL,
                    new_note TEXT NOT NULL,
                    changed_at TEXT NOT NULL,
                    undone INTEGER NOT NULL DEFAULT 0
                );
                """
            )

    def _setting(self, key: str) -> str | None:
        row = self._connection.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else str(row["value"])

    def _set_setting(self, key: str, value: str) -> None:
        self._connection.execute(
            """
            INSERT INTO settings(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def _validate_configuration(self) -> None:
        with self._lock, self._connection:
            configured_root = self._setting("image_root")
            current_root = str(self.image_root)
            if configured_root and configured_root != current_root:
                raise RuntimeError(
                    f"Review database {self.db_path} belongs to {configured_root}, not "
                    f"{current_root}. Use a different --state-dir."
                )
            self._set_setting("image_root", current_root)
            self._set_setting("target", str(self.target))
            self._set_setting("subject_regex", self.subject_regex)
            self._set_setting("tissue_regex", self.tissue_regex or "")

    def rescan(self) -> dict[str, Any]:
        result = discover_images(
            self.image_root,
            subject_regex=self.subject_regex,
            tissue_regex=self.tissue_regex,
        )
        scan_time = _utc_now()
        with self._lock, self._connection:
            self._connection.execute("UPDATE images SET active = 0")
            for image in result.images:
                self._connection.execute(
                    """
                    INSERT INTO images(
                        relative_path, subject_id, tissue, view, file_size,
                        mtime_ns, active, discovered_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                    ON CONFLICT(relative_path) DO UPDATE SET
                        subject_id = excluded.subject_id,
                        tissue = excluded.tissue,
                        view = excluded.view,
                        file_size = excluded.file_size,
                        mtime_ns = excluded.mtime_ns,
                        active = 1,
                        discovered_at = excluded.discovered_at
                    """,
                    (
                        image.relative_path,
                        image.subject_id,
                        image.tissue,
                        image.view,
                        image.file_size,
                        image.mtime_ns,
                        scan_time,
                    ),
                )
            self._set_setting("last_scan", scan_time)
            self._set_setting("last_scan_skipped", str(len(result.skipped)))
        self.export_all()
        return self._scan_payload(result)

    def _scan_payload(self, result: DiscoveryResult) -> dict[str, Any]:
        subjects = len({image.subject_id for image in result.images})
        tissues = sorted({image.tissue for image in result.images})
        return {
            "images": len(result.images),
            "subjects": subjects,
            "tissues": tissues,
            "skipped": len(result.skipped),
            "skipped_examples": [
                {"relative_path": issue.relative_path, "reason": issue.reason}
                for issue in result.skipped[:20]
            ],
        }

    def _subject_rows(self) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            """
            SELECT
                i.subject_id,
                COUNT(*) AS total_images,
                SUM(CASE WHEN d.decision = 'accept' THEN 1 ELSE 0 END) AS accepted_images,
                SUM(CASE WHEN d.decision = 'maybe' THEN 1 ELSE 0 END) AS maybe_images,
                SUM(CASE WHEN d.decision = 'decline' THEN 1 ELSE 0 END) AS declined_images
            FROM images i
            LEFT JOIN decisions d ON d.image_id = i.id
            WHERE i.active = 1
            GROUP BY i.subject_id
            ORDER BY LOWER(i.subject_id), i.subject_id
            """
        ).fetchall()
        summaries: list[dict[str, Any]] = []
        for row in rows:
            total = int(row["total_images"])
            accepted = int(row["accepted_images"] or 0)
            maybe = int(row["maybe_images"] or 0)
            declined = int(row["declined_images"] or 0)
            unreviewed = total - accepted - maybe - declined
            if declined:
                status = "declined"
            elif accepted == total and total:
                status = "accepted"
            elif maybe:
                status = "maybe"
            elif accepted:
                status = "in_review"
            else:
                status = "unreviewed"
            summaries.append(
                {
                    "subject_id": str(row["subject_id"]),
                    "status": status,
                    "total_images": total,
                    "accepted_images": accepted,
                    "maybe_images": maybe,
                    "declined_images": declined,
                    "unreviewed_images": unreviewed,
                }
            )
        return summaries

    def subject_summaries(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._subject_rows()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            summaries = self._subject_rows()
            status_counts = {
                status: sum(1 for row in summaries if row["status"] == status)
                for status in (
                    "accepted",
                    "declined",
                    "maybe",
                    "in_review",
                    "unreviewed",
                )
            }
            image_row = self._connection.execute(
                """
                SELECT
                    COUNT(*) AS total_images,
                    SUM(CASE WHEN d.decision = 'accept' THEN 1 ELSE 0 END) AS accepted_images,
                    SUM(CASE WHEN d.decision = 'maybe' THEN 1 ELSE 0 END) AS maybe_images,
                    SUM(CASE WHEN d.decision = 'decline' THEN 1 ELSE 0 END) AS declined_images
                FROM images i
                LEFT JOIN decisions d ON d.image_id = i.id
                WHERE i.active = 1
                """
            ).fetchone()
            queue_counts = self._connection.execute(
                """
                SELECT
                    SUM(CASE WHEN d.decision IS NULL THEN 1 ELSE 0 END) AS new_images,
                    SUM(CASE WHEN d.decision = 'maybe' THEN 1 ELSE 0 END) AS maybe_images
                FROM images i
                LEFT JOIN decisions d ON d.image_id = i.id
                WHERE i.active = 1
                  AND NOT EXISTS (
                      SELECT 1
                      FROM images declined_image
                      JOIN decisions declined_decision
                        ON declined_decision.image_id = declined_image.id
                      WHERE declined_image.active = 1
                        AND declined_image.subject_id = i.subject_id
                        AND declined_decision.decision = 'decline'
                  )
                """
            ).fetchone()
            accepted_subjects = status_counts["accepted"]
            return {
                "target": self.target,
                "target_reached": accepted_subjects >= self.target,
                "total_subjects": len(summaries),
                "accepted_subjects": accepted_subjects,
                "declined_subjects": status_counts["declined"],
                "maybe_subjects": status_counts["maybe"],
                "in_review_subjects": status_counts["in_review"],
                "unreviewed_subjects": status_counts["unreviewed"],
                "total_images": int(image_row["total_images"] or 0),
                "accepted_images": int(image_row["accepted_images"] or 0),
                "maybe_images": int(image_row["maybe_images"] or 0),
                "declined_images": int(image_row["declined_images"] or 0),
                "remaining_new_images": int(queue_counts["new_images"] or 0),
                "remaining_maybe_images": int(queue_counts["maybe_images"] or 0),
            }

    def queue(
        self, *, mode: str = "new", order: str = "subject"
    ) -> list[dict[str, Any]]:
        if mode not in {"new", "maybe"}:
            raise ValueError(f"Unsupported queue mode: {mode}")
        if order not in {"subject", "tissue"}:
            raise ValueError(f"Unsupported queue order: {order}")
        decision_clause = (
            "d.decision IS NULL" if mode == "new" else "d.decision = 'maybe'"
        )
        if order == "subject":
            order_clause = (
                "LOWER(i.subject_id), i.subject_id, LOWER(i.tissue), "
                "CASE i.view WHEN 'front' THEN 0 WHEN 'back' THEN 1 ELSE 2 END, i.relative_path"
            )
        else:
            order_clause = (
                "LOWER(i.tissue), CASE i.view WHEN 'front' THEN 0 WHEN 'back' THEN 1 ELSE 2 END, "
                "LOWER(i.subject_id), i.subject_id, i.relative_path"
            )
        sql = f"""
            SELECT i.id, i.relative_path, i.subject_id, i.tissue, i.view,
                   d.decision, d.note
            FROM images i
            LEFT JOIN decisions d ON d.image_id = i.id
            WHERE i.active = 1
              AND {decision_clause}
              AND NOT EXISTS (
                  SELECT 1
                  FROM images declined_image
                  JOIN decisions declined_decision
                    ON declined_decision.image_id = declined_image.id
                  WHERE declined_image.active = 1
                    AND declined_image.subject_id = i.subject_id
                    AND declined_decision.decision = 'decline'
              )
            ORDER BY {order_clause}
        """
        with self._lock:
            rows = self._connection.execute(sql).fetchall()
        return [
            {
                "id": int(row["id"]),
                "relative_path": str(row["relative_path"]),
                "subject_id": str(row["subject_id"]),
                "tissue": str(row["tissue"]),
                "tissue_display": _display_tissue(str(row["tissue"])),
                "view": str(row["view"]),
                "decision": row["decision"],
                "note": str(row["note"] or ""),
                "image_url": f"/api/image/{int(row['id'])}",
            }
            for row in rows
        ]

    def image_path(self, image_id: int) -> Path:
        with self._lock:
            row = self._connection.execute(
                "SELECT relative_path FROM images WHERE id = ? AND active = 1",
                (int(image_id),),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown active image ID: {image_id}")
        path = (self.image_root / str(row["relative_path"])).resolve()
        if not path.is_relative_to(self.image_root):
            raise RuntimeError(
                "Indexed image resolved outside the configured image root."
            )
        return path

    def record_decision(
        self, image_id: int, decision: str, note: str = ""
    ) -> dict[str, Any]:
        decision = decision.lower().strip()
        if decision not in DECISIONS:
            raise ValueError(f"Decision must be one of: {', '.join(sorted(DECISIONS))}")
        note = note.strip()
        before = self.stats()["accepted_subjects"]
        changed_at = _utc_now()
        with self._lock, self._connection:
            image = self._connection.execute(
                "SELECT subject_id FROM images WHERE id = ? AND active = 1",
                (int(image_id),),
            ).fetchone()
            if image is None:
                raise KeyError(f"Unknown active image ID: {image_id}")
            previous = self._connection.execute(
                "SELECT * FROM decisions WHERE image_id = ?", (int(image_id),)
            ).fetchone()
            self._connection.execute(
                """
                INSERT INTO decision_history(
                    image_id, previous_decision, previous_note,
                    previous_decided_at, previous_updated_at,
                    new_decision, new_note, changed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(image_id),
                    None if previous is None else previous["decision"],
                    None if previous is None else previous["note"],
                    None if previous is None else previous["decided_at"],
                    None if previous is None else previous["updated_at"],
                    decision,
                    note,
                    changed_at,
                ),
            )
            decided_at = changed_at if previous is None else str(previous["decided_at"])
            self._connection.execute(
                """
                INSERT INTO decisions(image_id, decision, note, decided_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    decision = excluded.decision,
                    note = excluded.note,
                    updated_at = excluded.updated_at
                """,
                (int(image_id), decision, note, decided_at, changed_at),
            )
        self.export_all(include_images=False)
        after_stats = self.stats()
        subject = next(
            row
            for row in self.subject_summaries()
            if row["subject_id"] == image["subject_id"]
        )
        return {
            "stats": after_stats,
            "subject": subject,
            "target_just_reached": before
            < self.target
            <= after_stats["accepted_subjects"],
        }

    def undo_last(self) -> dict[str, Any]:
        with self._lock, self._connection:
            history = self._connection.execute(
                """
                SELECT * FROM decision_history
                WHERE undone = 0
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
            if history is None:
                raise LookupError("There is no decision to undo.")
            if history["previous_decision"] is None:
                self._connection.execute(
                    "DELETE FROM decisions WHERE image_id = ?",
                    (int(history["image_id"]),),
                )
            else:
                self._connection.execute(
                    """
                    INSERT INTO decisions(image_id, decision, note, decided_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(image_id) DO UPDATE SET
                        decision = excluded.decision,
                        note = excluded.note,
                        decided_at = excluded.decided_at,
                        updated_at = excluded.updated_at
                    """,
                    (
                        int(history["image_id"]),
                        str(history["previous_decision"]),
                        str(history["previous_note"] or ""),
                        str(history["previous_decided_at"]),
                        str(history["previous_updated_at"]),
                    ),
                )
            self._connection.execute(
                "UPDATE decision_history SET undone = 1 WHERE id = ?",
                (int(history["id"]),),
            )
            image = self._connection.execute(
                "SELECT subject_id FROM images WHERE id = ?",
                (int(history["image_id"]),),
            ).fetchone()
        self.export_all(include_images=False)
        return {
            "image_id": int(history["image_id"]),
            "subject_id": None if image is None else str(image["subject_id"]),
            "restored_decision": history["previous_decision"],
            "stats": self.stats(),
        }

    def export_all(self, *, include_images: bool = True) -> dict[str, str]:
        with self._lock:
            summaries = self._subject_rows()
            status_by_subject = {row["subject_id"]: row["status"] for row in summaries}

            summary_fields = (
                "subject_id",
                "status",
                "total_images",
                "accepted_images",
                "maybe_images",
                "declined_images",
                "unreviewed_images",
            )
            self._write_csv_atomic(
                self.export_dir / "subject_summary.csv", summaries, summary_fields
            )

            if include_images:
                decision_rows = self._connection.execute(
                    """
                    SELECT i.subject_id, i.tissue, i.view, i.relative_path,
                           d.decision, d.note, d.decided_at, d.updated_at
                    FROM images i
                    LEFT JOIN decisions d ON d.image_id = i.id
                    WHERE i.active = 1
                    ORDER BY LOWER(i.subject_id), i.subject_id, LOWER(i.tissue),
                             i.view, i.relative_path
                    """
                ).fetchall()
                image_fields = (
                    "subject_id",
                    "subject_status",
                    "tissue",
                    "view",
                    "relative_path",
                    "decision",
                    "note",
                    "decided_at",
                    "updated_at",
                )
                image_export = [
                    {
                        "subject_id": str(row["subject_id"]),
                        "subject_status": status_by_subject[str(row["subject_id"])],
                        "tissue": str(row["tissue"]),
                        "view": str(row["view"]),
                        "relative_path": str(row["relative_path"]),
                        "decision": str(row["decision"] or ""),
                        "note": str(row["note"] or ""),
                        "decided_at": str(row["decided_at"] or ""),
                        "updated_at": str(row["updated_at"] or ""),
                    }
                    for row in decision_rows
                ]
                self._write_csv_atomic(
                    self.export_dir / "image_decisions.csv", image_export, image_fields
                )

            for status, filename in (
                ("accepted", "accepted_subjects.txt"),
                ("maybe", "maybe_subjects.txt"),
                ("declined", "declined_subjects.txt"),
            ):
                subject_ids = [
                    row["subject_id"] for row in summaries if row["status"] == status
                ]
                self._write_text_atomic(
                    self.export_dir / filename,
                    "".join(f"{subject_id}\n" for subject_id in subject_ids),
                )

            stats = self.stats()
            manifest = {
                "exported_at": _utc_now(),
                "image_root": str(self.image_root),
                "database": str(self.db_path),
                "target": self.target,
                "stats": stats,
            }
            self._write_text_atomic(
                self.export_dir / "review_manifest.json",
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            )
        return {
            "directory": str(self.export_dir),
            "accepted_subjects": str(self.export_dir / "accepted_subjects.txt"),
            "subject_summary": str(self.export_dir / "subject_summary.csv"),
            "image_decisions": str(self.export_dir / "image_decisions.csv"),
        }

    @staticmethod
    def _write_text_atomic(path: Path, text: str) -> None:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _write_csv_atomic(
        path: Path,
        rows: list[dict[str, Any]],
        fieldnames: tuple[str, ...],
    ) -> None:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
