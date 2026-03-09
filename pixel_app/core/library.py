from __future__ import annotations

import hashlib
import io
import mimetypes
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import struct

from PIL import Image, ImageOps, ExifTags

from pixel_app.core.auth import Auth
from pixel_app.core.db import DB, dumps_json
from pixel_app.core.background import analyze_background
from pixel_app.core.faces import FaceEmbedder


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_mime(name: str, fallback: str = "application/octet-stream") -> str:
    mt, _ = mimetypes.guess_type(name)
    return mt or fallback


def _parse_exif_datetime(dt_str: str | None) -> str | None:
    if not dt_str:
        return None
    dt_str = str(dt_str).strip()
    # Typical EXIF format: "YYYY:MM:DD HH:MM:SS"
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(dt_str, fmt)
            return dt.replace(tzinfo=timezone.utc).isoformat()
        except Exception:
            continue
    return None


def _dms_to_deg(values, ref) -> float | None:
    try:
        d, m, s = values
        d = float(d[0]) / float(d[1])
        m = float(m[0]) / float(m[1])
        s = float(s[0]) / float(s[1])
        deg = d + m / 60.0 + s / 3600.0
        if ref in ("S", "W"):
            deg = -deg
        return deg
    except Exception:
        return None


def _extract_exif_info(img: Image.Image) -> tuple[str | None, float | None, float | None]:
    taken_iso: str | None = None
    lat: float | None = None
    lon: float | None = None
    try:
        exif_raw = img._getexif() or {}
    except Exception:
        exif_raw = {}

    tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif_raw.items()}
    # Date
    taken_iso = _parse_exif_datetime(tag_map.get("DateTimeOriginal") or tag_map.get("DateTime"))

    # GPS
    gps = tag_map.get("GPSInfo")
    if isinstance(gps, dict):
        # GPSInfo keys are numeric in EXIF
        gps_decoded = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps.items()}
        lat_vals = gps_decoded.get("GPSLatitude")
        lat_ref = gps_decoded.get("GPSLatitudeRef")
        lon_vals = gps_decoded.get("GPSLongitude")
        lon_ref = gps_decoded.get("GPSLongitudeRef")
        if lat_vals and lat_ref:
            lat = _dms_to_deg(lat_vals, lat_ref)
        if lon_vals and lon_ref:
            lon = _dms_to_deg(lon_vals, lon_ref)

    return taken_iso, lat, lon


def _compute_quality_and_phash(img: Image.Image) -> tuple[float, str]:
    """
    Lightweight quality + perceptual hash.
    - quality: based on contrast and edge energy
    - phash: 64-bit aHash (8x8)
    """
    try:
        import numpy as np  # provided transitively by streamlit
    except Exception:
        return 0.5, ""

    # Quality
    g = img.convert("L").resize((256, 256))
    arr = np.asarray(g, dtype=np.float32) / 255.0
    contrast = float(arr.std())
    # edge approximation via simple finite differences
    dx = np.abs(arr[:, 1:] - arr[:, :-1])
    dy = np.abs(arr[1:, :] - arr[:-1, :])
    edges = float((dx.mean() + dy.mean()) / 2.0)
    q = 0.6 * contrast + 0.4 * edges
    q = max(0.0, min(1.0, q * 2.0))  # simple normalization

    # aHash
    ah = img.convert("L").resize((8, 8), Image.BILINEAR)
    a_arr = np.asarray(ah, dtype=np.float32)
    avg = float(a_arr.mean())
    bits = (a_arr > avg).astype("uint8").flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    phash = f"{val:016x}"
    return q, phash


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
        crypto = self.auth.require_crypto()
        enc_path = Path(photo_row["enc_path"])
        token = enc_path.read_bytes()
        return crypto.decrypt(token)

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
        crypto = self.auth.require_crypto()

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

        # EXIF: taken_at + GPS
        taken_at_iso, lat, lon = _extract_exif_info(img)

        # Auto quality + perceptual hash (for curation/duplicates)
        quality_score, phash = _compute_quality_and_phash(img)

        # Background-aware auto caption + tags (stored separately from user tags)
        try:
            bg = analyze_background(img)
            auto_tags = list(bg.tags or [])
            auto_caption = (bg.caption or "").strip() or None
            auto_debug_json = dumps_json(bg.debug or {})
        except Exception:
            auto_tags = []
            auto_caption = None
            auto_debug_json = dumps_json({})

        # Attach structured auto tags for later analytics/search
        auto_tags = list(sorted(set(auto_tags)))
        auto_tags.append(f"quality:{quality_score:.3f}")
        if phash:
            auto_tags.append(f"phash:{phash}")
        if taken_at_iso:
            # date-only + full timestamp
            auto_tags.append(f"taken_date:{taken_at_iso[:10]}")
            auto_tags.append(f"taken_at:{taken_at_iso}")
        if lat is not None and lon is not None:
            auto_tags.append(f"loc_lat:{lat:.5f}")
            auto_tags.append(f"loc_lon:{lon:.5f}")

        # Store encrypted original bytes (as uploaded)
        enc_bytes = crypto.encrypt(file_bytes)
        enc_path = self.photos_dir / f"{photo_id}.bin"
        enc_path.write_bytes(enc_bytes)

        self.db.execute(
            "INSERT INTO photos(id, original_name, mime, sha256, added_at, taken_at, width, height, enc_path, tags_json, auto_caption, auto_tags_json, auto_debug_json) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                photo_id,
                original_name,
                mime,
                sha,
                added_at,
                taken_at_iso,
                int(width),
                int(height),
                str(enc_path),
                dumps_json([]),  # user tags (editable)
                auto_caption,
                dumps_json(sorted(set(auto_tags))),
                auto_debug_json,
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

        if auto_tags:
            # Show only human-friendly, non-structured tags
            pretty = [t for t in auto_tags if ":" not in t][:8]
            if pretty:
                return True, f"Added to library (auto tags: {', '.join(pretty)})."
        return True, "Added to library."

