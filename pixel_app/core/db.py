from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS photos (
  id TEXT PRIMARY KEY,
  original_name TEXT NOT NULL,
  mime TEXT NOT NULL,
  sha256 TEXT NOT NULL UNIQUE,
  added_at TEXT NOT NULL,
  taken_at TEXT,
  width INTEGER,
  height INTEGER,
  enc_path TEXT NOT NULL,
  caption TEXT,
  tags_json TEXT NOT NULL DEFAULT '[]',
  auto_caption TEXT,
  auto_tags_json TEXT NOT NULL DEFAULT '[]',
  auto_debug_json TEXT
);

CREATE TABLE IF NOT EXISTS persons (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS faces (
  id TEXT PRIMARY KEY,
  photo_id TEXT NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  person_id TEXT REFERENCES persons(id) ON DELETE SET NULL,
  bbox_json TEXT NOT NULL,
  embedding BLOB NOT NULL,
  confidence REAL,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id);
CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);

CREATE TABLE IF NOT EXISTS shares (
  token TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  note TEXT,
  payload_path TEXT NOT NULL
);
"""


@dataclass
class DB:
    path: Path

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """
        Lightweight, additive migrations for hackathon builds.
        SQLite does not support many ALTERs, so keep to ADD COLUMN only.
        """
        try:
            cols = conn.execute("PRAGMA table_info(photos)").fetchall()
            have = {str(r[1]) for r in cols}  # row[1] = name
        except Exception:
            return

        def add_col(name: str, ddl: str) -> None:
            if name in have:
                return
            conn.execute(f"ALTER TABLE photos ADD COLUMN {ddl}")
            have.add(name)

        # Added in hackathon v2: background-aware auto caption/tags stored separately
        add_col("auto_caption", "auto_caption TEXT")
        add_col("auto_tags_json", "auto_tags_json TEXT NOT NULL DEFAULT '[]'")
        add_col("auto_debug_json", "auto_debug_json TEXT")

    def get_meta(self, key: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            return None if row is None else str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(conn.execute(sql, tuple(params)).fetchall())

    def execute(self, sql: str, params: Iterable[Any] = ()) -> None:
        with self.connect() as conn:
            conn.execute(sql, tuple(params))

    def execute_many(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
        with self.connect() as conn:
            conn.executemany(sql, rows)


def dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def loads_json(value: str) -> Any:
    return json.loads(value)

