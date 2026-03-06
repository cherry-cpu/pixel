import os
from functools import lru_cache


class Settings:
    photos_dir: str = os.path.join(os.getcwd(), "photos")
    shared_dir: str = os.path.join(os.getcwd(), "shared")
    db_path: str = os.path.join(os.getcwd(), "data", "photos.db")

    # HuggingFace Inference API
    hf_api_token: str | None = None
    hf_image_embedding_model: str = "sentence-transformers/clip-ViT-B-32"
    hf_text_embedding_model: str = "sentence-transformers/clip-ViT-B-32"

    def __init__(self) -> None:
        # Prefer environment variable, then Streamlit secrets (if available at runtime)
        token = os.getenv("HF_API_TOKEN")
        if not token:
            try:
                import streamlit as st

                token = st.secrets.get("HF_API_TOKEN")  # type: ignore[attr-defined]
            except Exception:
                token = None
        self.hf_api_token = token


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


