"""
Vision APIs: Hugging Face and Gemini for image caption, tags, and description.
Used when HF_TOKEN is set; falls back to CNN/heuristics otherwise.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VisionAPIResult:
    tags: list[str]
    caption: str | None
    debug: dict[str, Any]


def _analyze_with_hf(img: Any) -> VisionAPIResult | None:
    """Use Hugging Face Inference API for caption + image classification."""
    import os

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        return None

    try:
        from huggingface_hub import InferenceClient

        tags: set[str] = set()
        caption: str | None = None
        debug: dict[str, Any] = {}

        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=90)

        # 1) Image-to-text caption (BLIP)
        cap_model = os.getenv("HF_CAPTION_MODEL", "Salesforce/blip-image-captioning-base")
        try:
            cap_client = InferenceClient(model=cap_model, token=hf_token)
            cap = cap_client.image_to_text(buf.getvalue())
            if isinstance(cap, str):
                caption = cap.strip() or None
            elif isinstance(cap, dict) and cap.get("generated_text"):
                caption = str(cap["generated_text"]).strip() or None
            debug["hf_caption_model"] = cap_model
        except Exception as e:
            debug["hf_caption_error"] = str(e)

        # 2) Image classification for tags
        cls_model = os.getenv("HF_VISION_MODEL", "google/vit-base-patch16-224")
        try:
            cls_client = InferenceClient(model=cls_model, token=hf_token)
            buf2 = io.BytesIO()
            img.convert("RGB").save(buf2, format="JPEG", quality=85)
            preds = cls_client.image_classification(buf2.getvalue())
            labels = [p.label for p in (preds or [])] if preds else []
            text = " ".join(str(x).lower() for x in labels[:12])

            def add_if(words: list[str], tag: str) -> None:
                for w in words:
                    if w in text:
                        tags.add(tag)
                        return

            add_if(["beach", "seashore", "coast"], "beach")
            add_if(["mountain", "alp", "cliff", "peak"], "mountain")
            add_if(["forest", "woodland", "jungle"], "forest")
            add_if(["city", "street", "downtown", "building", "skyscraper"], "city")
            add_if(["snow", "ice", "glacier"], "snow")
            add_if(["lake", "river", "ocean", "sea", "water"], "water")
            add_if(["desert", "sand", "dune"], "sand")
            add_if(["sunset", "sunrise"], "sunset")
            add_if(["night"], "night")
            for lbl in labels[:8]:
                tag = lbl.replace(" ", "_").lower()
                if tag and len(tag) > 2:
                    tags.add(tag)
            debug["hf_vision_model"] = cls_model
            debug["hf_top_labels"] = labels[:8]
        except Exception as e:
            debug["hf_classification_error"] = str(e)

        return VisionAPIResult(tags=sorted(tags), caption=caption, debug=debug)
    except Exception as e:
        return VisionAPIResult(tags=[], caption=None, debug={"hf_error": str(e)})


def _analyze_with_gemini(img: Any) -> VisionAPIResult | None:
    """Use Google Gemini API for image description and tags."""
    import os

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    try:
        from google import genai
        from google.genai import types

        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=90)
        image_bytes = buf.getvalue()

        client = genai.Client(api_key=api_key)
        model = os.getenv("GEMINI_VISION_MODEL", "gemini-2.0-flash")

        prompt = (
            "Describe this image in one short sentence (caption). "
            "Then list 5-15 comma-separated tags: background (indoor/outdoor, sky, water, etc.), "
            "objects (person, car, dog, etc.), and scene type. "
            "Reply in this exact format:\nCAPTION: <one sentence>\nTAGS: tag1, tag2, tag3, ..."
        )

        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text=prompt),
            ],
            config=types.GenerateContentConfig(temperature=0.2),
        )

        caption: str | None = None
        tags: list[str] = []
        debug: dict[str, Any] = {"gemini_model": model}

        text = ""
        if response and response.candidates:
            for c in response.candidates:
                if c.content and c.content.parts:
                    for p in c.content.parts:
                        if hasattr(p, "text") and p.text:
                            text += str(p.text)

        if text:
            cap_match = re.search(r"CAPTION:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
            if cap_match:
                caption = cap_match.group(1).strip()
            if not caption and "CAPTION:" not in text.upper():
                caption = text.strip()[:200]

            tags_match = re.search(r"TAGS:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
            if tags_match:
                raw = tags_match.group(1).strip()
                tags = [t.strip().replace(" ", "_").lower() for t in raw.split(",") if t.strip()]
            debug["gemini_raw"] = text[:500]

        return VisionAPIResult(tags=sorted(set(tags)), caption=caption, debug=debug)
    except Exception as e:
        return VisionAPIResult(tags=[], caption=None, debug={"gemini_error": str(e)})


def analyze_with_apis(img: Any) -> VisionAPIResult | None:
    """
    Try Hugging Face and Gemini APIs (when configured).
    Returns merged tags and best caption, or None if neither API is configured.
    """
    hf_result = _analyze_with_hf(img)
    gemini_result = _analyze_with_gemini(img)

    tags: set[str] = set()
    caption: str | None = None
    debug: dict[str, Any] = {}

    if hf_result:
        tags.update(hf_result.tags or [])
        if hf_result.caption:
            caption = hf_result.caption
        debug["hf"] = hf_result.debug

    if gemini_result:
        tags.update(gemini_result.tags or [])
        if gemini_result.caption and not caption:
            caption = gemini_result.caption
        elif gemini_result.caption and caption:
            caption = gemini_result.caption
        debug["gemini"] = gemini_result.debug

    if not tags and not caption:
        return None

    return VisionAPIResult(tags=sorted(tags), caption=caption, debug=debug)
