from __future__ import annotations

import logging
from typing import List

import numpy as np
import requests
from PIL import Image

from .config import get_settings

logger = logging.getLogger(__name__)


class HuggingFaceClient:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.hf_api_token:
            raise RuntimeError(
                "HF_API_TOKEN is not configured. Set it as an environment variable or in Streamlit secrets."
            )
        self.token = settings.hf_api_token
        self.image_model = settings.hf_image_embedding_model
        self.text_model = settings.hf_text_embedding_model
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {self.token}"})

    def _post_feature_extraction(self, model: str, inputs) -> List[float]:
        url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
        response = self._session.post(url, json={"inputs": inputs})
        if response.status_code != 200:
            logger.error("HF API error %s: %s", response.status_code, response.text)
            raise RuntimeError(f"HuggingFace API error: {response.status_code}")
        data = response.json()
        # CLIP returns [1, dim]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            return [float(x) for x in data[0]]
        return [float(x) for x in data]

    def image_embedding(self, image: Image.Image) -> list[float]:
        # Convert PIL image to RGB and then to nested list for HF API
        img = image.convert("RGB")
        arr = np.array(img).tolist()
        return self._post_feature_extraction(self.image_model, arr)

    def text_embedding(self, text: str) -> list[float]:
        return self._post_feature_extraction(self.text_model, text)

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        va = np.array(a, dtype="float32")
        vb = np.array(b, dtype="float32")
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)


