from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from pixel_app.core.db import DB, dumps_json, loads_json


def _iso_date(s: str | None) -> str | None:
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    # accept YYYY-MM-DD
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    return None


def _decode_embedding(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="float32")


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype("float32")
    b = b.astype("float32")
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


@dataclass
class PeopleService:
    db: DB

    def list_people(self) -> list[dict]:
        rows = self.db.query("SELECT * FROM persons ORDER BY name ASC")
        return [dict(r) for r in rows]

    def rename_person(self, person_id: str, new_name: str) -> None:
        new_name = new_name.strip()
        if not new_name:
            return
        self.db.execute("UPDATE persons SET name=? WHERE id=?", (new_name, person_id))

    def create_person(self, name: str) -> str:
        pid = str(uuid.uuid4())
        self.db.execute(
            "INSERT INTO persons(id, name, created_at) VALUES(?, ?, ?)",
            (pid, name.strip(), datetime.utcnow().isoformat() + "Z"),
        )
        return pid

    def assign_face_to_person(self, face_id: str, person_id: str | None) -> None:
        self.db.execute("UPDATE faces SET person_id=? WHERE id=?", (person_id, face_id))

    def auto_cluster_unknown_faces(self, sim_threshold: float = 0.78, max_new_people: int = 50) -> dict[str, Any]:
        """
        Simple incremental clustering:
        - For each unassigned face embedding, match to best existing person centroid.
        - If no match above threshold, create a new 'Unknown N' person.
        """
        unknown_faces = self.db.query(
            "SELECT id, embedding FROM faces WHERE person_id IS NULL ORDER BY created_at ASC"
        )
        if not unknown_faces:
            return {"created_people": 0, "assigned_faces": 0}

        # Build centroids for existing persons
        persons = self.db.query("SELECT id, name FROM persons")
        centroids: dict[str, np.ndarray] = {}
        for p in persons:
            emb_rows = self.db.query("SELECT embedding FROM faces WHERE person_id=?", (p["id"],))
            if not emb_rows:
                continue
            embs = np.stack([_decode_embedding(r["embedding"]) for r in emb_rows]).astype("float32")
            c = embs.mean(axis=0)
            centroids[str(p["id"])] = c

        created = 0
        assigned = 0

        def next_unknown_name() -> str:
            base = "Unknown"
            existing = self.db.query("SELECT name FROM persons WHERE name LIKE 'Unknown %'")
            nums = []
            for r in existing:
                m = re.fullmatch(r"Unknown (\d+)", str(r["name"]))
                if m:
                    nums.append(int(m.group(1)))
            n = (max(nums) + 1) if nums else 1
            return f"{base} {n}"

        for row in unknown_faces:
            face_id = str(row["id"])
            emb = _decode_embedding(row["embedding"])

            best_pid = None
            best_sim = -1.0
            for pid, c in centroids.items():
                sim = _cosine_sim(emb, c)
                if sim > best_sim:
                    best_sim = sim
                    best_pid = pid

            if best_pid is not None and best_sim >= sim_threshold:
                self.assign_face_to_person(face_id, best_pid)
                assigned += 1
                # Update centroid incrementally (cheap)
                centroids[best_pid] = (centroids[best_pid] + emb) / 2.0
                continue

            if created >= max_new_people:
                continue

            pid = self.create_person(next_unknown_name())
            centroids[pid] = emb
            self.assign_face_to_person(face_id, pid)
            created += 1
            assigned += 1

        return {"created_people": created, "assigned_faces": assigned}


@dataclass
class SearchService:
    db: DB

    def keyword_search(self, q: str, limit: int = 50) -> list[dict]:
        q = (q or "").strip()
        if not q:
            rows = self.db.query("SELECT * FROM photos ORDER BY added_at DESC LIMIT ?", (limit,))
            return [dict(r) for r in rows]

        terms = [t for t in re.split(r"\s+", q) if t]
        like = "%" + "%".join(terms) + "%"

        # tags_json is JSON string; do naive LIKE match (hackathon-friendly)
        rows = self.db.query(
            "SELECT DISTINCT photos.* "
            "FROM photos "
            "LEFT JOIN faces ON faces.photo_id = photos.id "
            "LEFT JOIN persons ON persons.id = faces.person_id "
            "WHERE photos.original_name LIKE ? "
            "   OR COALESCE(photos.caption,'') LIKE ? "
            "   OR photos.tags_json LIKE ? "
            "   OR COALESCE(persons.name,'') LIKE ? "
            "ORDER BY photos.added_at DESC "
            "LIMIT ?",
            (like, like, like, like, limit),
        )
        return [dict(r) for r in rows]

    def structured_search(self, parsed: Any) -> list[dict]:
        # parsed is ParsedQuery from llm.py, but we keep this module decoupled
        limit = int(getattr(parsed, "limit", 50) or 50)
        limit = max(1, min(200, limit))

        people = [p.strip() for p in (getattr(parsed, "people", None) or []) if p.strip()]
        tags = [t.strip() for t in (getattr(parsed, "tags", None) or []) if t.strip()]
        text = (getattr(parsed, "text", "") or "").strip()
        date_from = _iso_date(getattr(parsed, "date_from", None))
        date_to = _iso_date(getattr(parsed, "date_to", None))

        clauses: list[str] = []
        params: list[Any] = []

        if text:
            clauses.append("(photos.original_name LIKE ? OR COALESCE(photos.caption,'') LIKE ?)")
            like = f"%{text}%"
            params += [like, like]

        if tags:
            # naive: tags_json contains substrings
            for t in tags:
                clauses.append("photos.tags_json LIKE ?")
                params.append(f"%{t}%")

        if people:
            # Match any requested person names via join
            placeholders = ",".join("?" for _ in people)
            clauses.append(f"persons.name IN ({placeholders})")
            params += people

        if date_from:
            clauses.append("substr(photos.added_at,1,10) >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("substr(photos.added_at,1,10) <= ?")
            params.append(date_to)

        where = " AND ".join(clauses) if clauses else "1=1"

        rows = self.db.query(
            "SELECT DISTINCT photos.* "
            "FROM photos "
            "LEFT JOIN faces ON faces.photo_id = photos.id "
            "LEFT JOIN persons ON persons.id = faces.person_id "
            f"WHERE {where} "
            "ORDER BY photos.added_at DESC "
            "LIMIT ?",
            tuple(params + [limit]),
        )
        return [dict(r) for r in rows]

    def set_photo_caption(self, photo_id: str, caption: str) -> None:
        self.db.execute("UPDATE photos SET caption=? WHERE id=?", (caption.strip(), photo_id))

    def set_photo_tags(self, photo_id: str, tags: list[str]) -> None:
        clean = sorted({t.strip() for t in tags if t.strip()})
        self.db.execute("UPDATE photos SET tags_json=? WHERE id=?", (dumps_json(clean), photo_id))

    def get_photo_tags(self, photo_row: dict) -> list[str]:
        try:
            return list(loads_json(photo_row.get("tags_json") or "[]"))
        except Exception:
            return []

