from __future__ import annotations

import hashlib
import io
import mimetypes
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import struct

from PIL import Image, ImageOps

from pixel_app.core.auth import Auth
from pixel_app.core.db import DB, dumps_json
from pixel_app.core.faces import FaceEmbedder
from pixel_app.core.llm import analyze_image


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_mime(name: str, fallback: str = "application/octet-stream") -> str:
    mt, _ = mimetypes.guess_type(name)
    return mt or fallback


def _encode_embedding(emb: list[float]) -> bytes:
    # Store float32 to keep payload compact and consistent.
    return struct.pack("<" + "f" * len(emb), *[float(x) for x in emb])


@dataclass
class Library:
    db: DB
    auth: Auth
    photos_dir: Path
    thumbs_dir: Path
    embedder: FaceEmbedder

    def init_dirs(self) -> None:
        self.photos_dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)

    def list_photos(self, limit: int = 200, offset: int = 0) -> list[dict]:
        rows = self.db.query(
            "SELECT * FROM photos ORDER BY added_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [dict(r) for r in rows]

    def get_photo(self, photo_id: str) -> dict | None:
        rows = self.db.query("SELECT * FROM photos WHERE id = ?", (photo_id,))
        return None if not rows else dict(rows[0])

    def get_photo_faces(self, photo_id: str) -> list[dict]:
        rows = self.db.query(
            "SELECT faces.*, persons.name as person_name "
            "FROM faces LEFT JOIN persons ON persons.id = faces.person_id "
            "WHERE faces.photo_id = ?",
            (photo_id,),
        )
        return [dict(r) for r in rows]

    def read_photo_bytes(self, photo_row: dict) -> bytes:
        enc_path = Path(photo_row["enc_path"])
        return enc_path.read_bytes()

    def read_thumbnail_bytes(self, photo_row: dict, max_size: int = 512) -> bytes:
        thumb_path = self.thumbs_dir / f"{photo_row['id']}.jpg"
        if thumb_path.exists():
            return thumb_path.read_bytes()

        raw = self.read_photo_bytes(photo_row)
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)
        img.thumbnail((max_size, max_size))
        out = io.BytesIO()
        img.convert("RGB").save(out, format="JPEG", quality=85, optimize=True)
        thumb_bytes = out.getvalue()
        thumb_path.write_bytes(thumb_bytes)
        return thumb_bytes

    def ingest(self, original_name: str, file_bytes: bytes) -> tuple[bool, str]:
        self.init_dirs()

        sha = _sha256(file_bytes)
        existing = self.db.query("SELECT id FROM photos WHERE sha256 = ?", (sha,))
        if existing:
            return False, "Duplicate image (already in library)."

        photo_id = str(uuid.uuid4())
        mime = _safe_mime(original_name, "image/jpeg")
        added_at = _utc_now_iso()

        # Parse image for width/height and normalize orientation
        img = Image.open(io.BytesIO(file_bytes))
        img = ImageOps.exif_transpose(img)
        width, height = img.size

        # Store original bytes (unencrypted)
        enc_path = self.photos_dir / f"{photo_id}.bin"
        enc_path.write_bytes(file_bytes)

        # Call Vision LLM for background analysis and tagging
        caption = None
        tags = []
        try:
            analysis = analyze_image(file_bytes, mime)
            if analysis:
                caption = analysis.get("caption")
                tags = analysis.get("tags", [])
        except Exception as e:
            print(f"Skipping LLM analysis: {e}")

        self.db.execute(
            "INSERT INTO photos(id, original_name, mime, sha256, added_at, width, height, enc_path, caption, tags_json) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                photo_id,
                original_name,
                mime,
                sha,
                added_at,
                int(width),
                int(height),
                str(enc_path),
                caption,
                dumps_json(tags),
            ),
        )

        # Face detection + embedding (optional if local face model installed)
        try:
            detections = self.embedder.detect_and_embed(img)
        except Exception:
            detections = []

        if detections:
            face_rows: list[tuple] = []
            for det in detections:
                face_rows.append(
                    (
                        str(uuid.uuid4()),
                        photo_id,
                        None,
                        dumps_json({"xyxy": det.bbox_xyxy}),
                        _encode_embedding(det.embedding),
                        det.confidence,
                        added_at,
                    )
                )
            self.db.execute_many(
                "INSERT INTO faces(id, photo_id, person_id, bbox_json, embedding, confidence, created_at) "
                "VALUES(?, ?, ?, ?, ?, ?, ?)",
                face_rows,
            )

        # Add a cached thumbnail (unencrypted; can be deleted/regenerated)
        try:
            self.read_thumbnail_bytes({"id": photo_id, "enc_path": str(enc_path)}, max_size=512)
        except Exception:
            # Thumbnail is an optimization, not a correctness requirement.
            pass

        return True, "Added to library."

