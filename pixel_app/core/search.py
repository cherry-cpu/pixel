from __future__ import annotations

import math
import re
import struct
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

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


def _decode_embedding(b: bytes) -> list[float]:
    if not b:
        return []
    n = len(b) // 4
    return list(struct.unpack("<" + "f" * n, b[: n * 4]))


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return -1.0
    n = min(len(a), len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        x = float(a[i])
        y = float(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    denom = (math.sqrt(na) * math.sqrt(nb)) + 1e-12
    return dot / denom


def _mean(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    count = 0
    for v in vectors:
        if len(v) != dim:
            continue
        for i in range(dim):
            acc[i] += float(v[i])
        count += 1
    if count == 0:
        return []
    return [x / count for x in acc]


def _avg2(a: list[float], b: list[float]) -> list[float]:
    if not a:
        return b
    if not b:
        return a
    n = min(len(a), len(b))
    out = [(float(a[i]) + float(b[i])) / 2.0 for i in range(n)]
    # keep trailing dims if any (shouldn't happen)
    if len(a) > n:
        out += [float(x) for x in a[n:]]
    elif len(b) > n:
        out += [float(x) for x in b[n:]]
    return out


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
        centroids: dict[str, list[float]] = {}
        for p in persons:
            emb_rows = self.db.query("SELECT embedding FROM faces WHERE person_id=?", (p["id"],))
            if not emb_rows:
                continue
            embs = [_decode_embedding(r["embedding"]) for r in emb_rows]
            c = _mean([e for e in embs if e])
            if c:
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
            if not emb:
                continue

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
                centroids[best_pid] = _avg2(centroids[best_pid], emb)
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
            "   OR COALESCE(photos.auto_caption,'') LIKE ? "
            "   OR photos.tags_json LIKE ? "
            "   OR photos.auto_tags_json LIKE ? "
            "   OR COALESCE(persons.name,'') LIKE ? "
            "ORDER BY photos.added_at DESC "
            "LIMIT ?",
            (like, like, like, like, like, like, limit),
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
            clauses.append(
                "(photos.original_name LIKE ? OR COALESCE(photos.caption,'') LIKE ? OR COALESCE(photos.auto_caption,'') LIKE ?)"
            )
            like = f"%{text}%"
            params += [like, like, like]

        if tags:
            # naive: tags_json contains substrings
            for t in tags:
                clauses.append("(photos.tags_json LIKE ? OR photos.auto_tags_json LIKE ?)")
                like = f"%{t}%"
                params += [like, like]

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

    def get_photo_auto_tags(self, photo_row: dict) -> list[str]:
        try:
            return list(loads_json(photo_row.get("auto_tags_json") or "[]"))
        except Exception:
            return []

    def get_photo_auto_caption(self, photo_row: dict) -> str:
        return str(photo_row.get("auto_caption") or "").strip()

    # --- Event & timeline intelligence + quality/duplicates ---

    def _parse_structured_tags(self, tags: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for t in tags:
            if ":" in t:
                k, v = t.split(":", 1)
                out[k.strip()] = v.strip()
        return out

    def build_events(self, gap_hours: float = 3.0) -> list[dict]:
        """
        Group photos into events based on taken_at (or added_at) time gaps.
        Returns list of {id, start, end, count, photo_ids}.
        """
        rows = self.db.query(
            "SELECT id, added_at, taken_at, tags_json, original_name FROM photos ORDER BY COALESCE(taken_at, added_at) ASC"
        )
        if not rows:
            return []

        events: list[dict] = []
        current: dict | None = None
        last_ts: datetime | None = None
        gap_sec = gap_hours * 3600.0

        def parse_ts(s: str | None) -> datetime | None:
            if not s:
                return None
            try:
                return datetime.fromisoformat(s)
            except Exception:
                return None

        for r in rows:
            pid = str(r["id"])
            ts = parse_ts(r["taken_at"] or r["added_at"])
            if ts is None:
                continue

            if current is None or last_ts is None:
                current = {
                    "id": f"event-{len(events)+1}",
                    "start": ts,
                    "end": ts,
                    "photo_ids": [pid],
                }
                last_ts = ts
                continue

            delta = (ts - last_ts).total_seconds()
            if delta > gap_sec:
                # new event
                current["count"] = len(current["photo_ids"])
                events.append(current)
                current = {
                    "id": f"event-{len(events)+1}",
                    "start": ts,
                    "end": ts,
                    "photo_ids": [pid],
                }
            else:
                current["end"] = ts
                current["photo_ids"].append(pid)
            last_ts = ts

        if current:
            current["count"] = len(current["photo_ids"])
            events.append(current)

        # convert datetimes to iso for UI
        for ev in events:
            ev["start_iso"] = ev["start"].isoformat()
            ev["end_iso"] = ev["end"].isoformat()
        return events

    def get_top_quality(self, limit: int = 20) -> list[dict]:
        rows = self.db.query("SELECT * FROM photos")
        scored: list[tuple[float, dict]] = []
        for r in rows:
            # Quality score is stored as structured auto tag.
            tags = self.get_photo_auto_tags(dict(r))
            meta = self._parse_structured_tags(tags)
            q = float(meta.get("quality", "0") or "0")
            scored.append((q, dict(r)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [row for _q, row in scored[:limit] if _q > 0]

    def _hamming(self, a: str, b: str) -> int:
        if len(a) != len(b):
            return 64
        return sum(1 for x, y in zip(a, b) if x != y)

    def find_duplicate_groups(self, max_hamming: int = 4, min_group_size: int = 2) -> list[dict]:
        """
        Find near-duplicate groups based on phash:* tags.
        Returns groups of {phash, photo_ids}.
        """
        rows = self.db.query("SELECT id, tags_json, auto_tags_json FROM photos")
        items: list[tuple[str, str]] = []  # (photo_id, phash)
        for r in rows:
            # pHash is stored as structured auto tag.
            row = dict(r)
            tags = self.get_photo_auto_tags(row) or self.get_photo_tags(row)
            meta = self._parse_structured_tags(tags)
            ph = meta.get("phash")
            if ph:
                items.append((str(r["id"]), ph))
        if not items:
            return []

        used: set[str] = set()
        groups: list[dict] = []

        for i, (pid, ph) in enumerate(items):
            if pid in used:
                continue
            group = [pid]
            used.add(pid)
            for j in range(i + 1, len(items)):
                pid2, ph2 = items[j]
                if pid2 in used:
                    continue
                if self._hamming(ph, ph2) <= max_hamming:
                    group.append(pid2)
                    used.add(pid2)
            if len(group) >= min_group_size:
                groups.append({"phash": ph, "photo_ids": group})

        return groups

