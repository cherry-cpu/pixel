from __future__ import annotations

import datetime as dt
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import get_settings

Base = declarative_base()


class Photo(Base):
    __tablename__ = "photos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False, unique=True)
    original_name = Column(String(255), nullable=False)
    person_label = Column(String(255), nullable=True)  # e.g. "John", "Family"
    tags = Column(Text, nullable=True)  # comma-separated tags
    caption = Column(Text, nullable=True)
    taken_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    is_private = Column(Integer, default=1, nullable=False)  # 1 = private, 0 = shareable

    # Embedding metadata
    embedding_model = Column(String(255), nullable=True)
    embedding_dim = Column(Integer, nullable=True)
    # Stored as comma-separated floats
    embedding = Column(Text, nullable=True)


def _create_engine():
    settings = get_settings()
    db_path = Path(settings.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", echo=False, future=True)
    return engine


engine = _create_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def serialize_embedding(vec: list[float]) -> str:
    return ",".join(f"{x:.6f}" for x in vec)


def deserialize_embedding(value: str | None) -> list[float]:
    if not value:
        return []
    return [float(x) for x in value.split(",") if x]


