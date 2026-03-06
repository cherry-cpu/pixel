from __future__ import annotations

import datetime as dt
import os
import shutil
import uuid
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image, ExifTags

from .config import get_settings
from .db import (
    Photo,
    SessionLocal,
    deserialize_embedding,
    init_db,
    serialize_embedding,
)
from .hf_client import HuggingFaceClient


def _ensure_dirs() -> None:
    settings = get_settings()
    Path(settings.photos_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.shared_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(settings.db_path)).mkdir(parents=True, exist_ok=True)


def init_app_storage() -> None:
    _ensure_dirs()
    init_db()


def _extract_taken_at(image: Image.Image) -> Optional[dt.datetime]:
    try:
        exif = image._getexif()  # type: ignore[attr-defined]
        if not exif:
            return None
        exif_data = {
            ExifTags.TAGS.get(tag, tag): value for tag, value in exif.items()
        }
        date_str = exif_data.get("DateTimeOriginal") or exif_data.get("DateTime")
        if not date_str:
            return None
        # Common EXIF format: "YYYY:MM:DD HH:MM:SS"
        return dt.datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None


def save_photo(
    file_bytes: bytes,
    original_name: str,
    person_label: Optional[str] = None,
    tags: Optional[list[str]] = None,
    caption: Optional[str] = None,
    is_private: bool = True,
) -> Photo:
    settings = get_settings()
    _ensure_dirs()

    suffix = Path(original_name).suffix or ".jpg"
    safe_name = f"{uuid.uuid4().hex}{suffix}"
    dest_path = Path(settings.photos_dir) / safe_name

    with open(dest_path, "wb") as f:
        f.write(file_bytes)

    img = Image.open(dest_path)
    taken_at = _extract_taken_at(img)

    hf = HuggingFaceClient()
    embedding = hf.image_embedding(img)

    with SessionLocal() as db:
        photo = Photo(
            filename=safe_name,
            original_name=original_name,
            person_label=person_label,
            tags=",".join(tags) if tags else None,
            caption=caption,
            taken_at=taken_at,
            is_private=1 if is_private else 0,
            embedding_model=hf.image_model,
            embedding_dim=len(embedding),
            embedding=serialize_embedding(embedding),
        )
        db.add(photo)
        db.commit()
        db.refresh(photo)
        return photo


def list_photos(person: Optional[str] = None) -> list[Photo]:
    with SessionLocal() as db:
        query = db.query(Photo)
        if person:
            query = query.filter(Photo.person_label == person)
        query = query.order_by(Photo.created_at.desc())
        return list(query.all())


def all_people() -> list[str]:
    with SessionLocal() as db:
        rows = db.query(Photo.person_label).filter(Photo.person_label.isnot(None)).distinct().all()
        return sorted({r[0] for r in rows if r[0]})


def update_photo_labels(
    photo_ids: Iterable[int],
    person_label: Optional[str] = None,
    tags: Optional[list[str]] = None,
    is_private: Optional[bool] = None,
) -> None:
    with SessionLocal() as db:
        for pid in photo_ids:
            photo = db.query(Photo).get(pid)
            if not photo:
                continue
            if person_label is not None:
                photo.person_label = person_label
            if tags is not None:
                photo.tags = ",".join(tags)
            if is_private is not None:
                photo.is_private = 1 if is_private else 0
        db.commit()


def get_photo_path(photo: Photo) -> Path:
    settings = get_settings()
    return Path(settings.photos_dir) / photo.filename


def search_photos_by_text(query: str, top_k: int = 12) -> list[Tuple[Photo, float]]:
    hf = HuggingFaceClient()
    q_emb = hf.text_embedding(query)

    from .hf_client import HuggingFaceClient as HFClient  # avoid circular

    with SessionLocal() as db:
        photos = db.query(Photo).all()

    results: List[Tuple[Photo, float]] = []
    for p in photos:
        emb = deserialize_embedding(p.embedding)
        score = HFClient.cosine_similarity(q_emb, emb)
        results.append((p, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def share_photos(photo_ids: Iterable[int]) -> Path:
    """Create a simple shared folder with copies of selected photos and return its path."""
    settings = get_settings()
    shared_root = Path(settings.shared_dir)
    shared_root.mkdir(parents=True, exist_ok=True)

    share_id = uuid.uuid4().hex[:8]
    target_dir = shared_root / f"share_{share_id}"
    target_dir.mkdir(parents=True, exist_ok=True)

    with SessionLocal() as db:
        for pid in photo_ids:
            photo = db.query(Photo).get(pid)
            if not photo:
                continue
            if photo.is_private:
                # Skip private photos to respect secure storage
                continue
            src = get_photo_path(photo)
            if src.exists():
                shutil.copy2(src, target_dir / photo.original_name)

    return target_dir


