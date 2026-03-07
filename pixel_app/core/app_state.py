from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from pixel_app.core.auth import Auth
from pixel_app.core.db import DB
from pixel_app.core.faces import FaceEmbedder
from pixel_app.core.library import Library
from pixel_app.core.paths import AppPaths
from pixel_app.core.search import PeopleService, SearchService
from pixel_app.core.sharing import ShareService


@dataclass
class PixelApp:
    paths: AppPaths
    db: DB
    auth: Auth
    library: Library
    people: PeopleService
    search: SearchService
    sharing: ShareService


@st.cache_resource
def get_app() -> PixelApp:
    root = Path(__file__).resolve().parents[2]
    paths = AppPaths(root=root)

    db = DB(paths.db_path)
    db.init()

    auth = Auth(db=db)
    auth.ensure_initialized()
    auth.ensure_unlocked("default_passphrase_for_silent_encryption")

    # Face model is optional; app runs without it (no face clustering).
    try:
        embedder = FaceEmbedder()
    except Exception:
        class _NoFaceEmbedder:
            def detect_and_embed(self, _img):
                return []

        embedder = _NoFaceEmbedder()  # type: ignore[assignment]
    library = Library(
        db=db,
        auth=auth,
        photos_dir=paths.photos_dir,
        thumbs_dir=paths.thumbs_dir,
        embedder=embedder,
    )
    people = PeopleService(db=db)
    search = SearchService(db=db)
    sharing = ShareService(db=db, auth=auth, library=library, shares_dir=paths.shares_dir)

    return PixelApp(
        paths=paths,
        db=db,
        auth=auth,
        library=library,
        people=people,
        search=search,
        sharing=sharing,
    )

