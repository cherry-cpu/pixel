from __future__ import annotations

import os
import io
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BackgroundResult:
    tags: list[str]
    caption: str | None
    debug: dict[str, Any]


def _rgb_to_hsv(r: float, g: float, b: float) -> tuple[float, float, float]:
    # r,g,b in 0..1; returns h in 0..360, s in 0..1, v in 0..1
    mx = max(r, g, b)
    mn = min(r, g, b)
    d = mx - mn
    if d == 0:
        h = 0.0
    elif mx == r:
        h = (60.0 * ((g - b) / d) + 360.0) % 360.0
    elif mx == g:
        h = (60.0 * ((b - r) / d) + 120.0) % 360.0
    else:
        h = (60.0 * ((r - g) / d) + 240.0) % 360.0
    s = 0.0 if mx == 0 else d / mx
    v = mx
    return h, s, v


def _sample_border_rgb(img) -> tuple[list[tuple[int, int, int]], dict[str, float]]:
    # uses PIL only
    im = img.convert("RGB")
    w, h = im.size
    # downscale for speed
    max_dim = 160
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        w, h = im.size

    px = im.load()
    border: list[tuple[int, int, int]] = []
    # thickness proportional
    t = max(2, min(8, int(min(w, h) * 0.04)))
    # top/bottom
    for y in range(t):
        for x in range(w):
            border.append(px[x, y])
            border.append(px[x, h - 1 - y])
    # left/right
    for x in range(t):
        for y in range(h):
            border.append(px[x, y])
            border.append(px[w - 1 - x, y])

    # separate top/bottom for sky/water heuristics
    top: list[tuple[int, int, int]] = []
    bottom: list[tuple[int, int, int]] = []
    for y in range(t):
        for x in range(w):
            top.append(px[x, y])
            bottom.append(px[x, h - 1 - y])

    stats = {
        "w": float(w),
        "h": float(h),
        "border_pixels": float(len(border)),
        "top_pixels": float(len(top)),
        "bottom_pixels": float(len(bottom)),
    }
    return border, {"_t": float(t), **stats, "_top": top, "_bottom": bottom}


def _color_votes(pixels: list[tuple[int, int, int]]) -> dict[str, float]:
    votes = {
        "blue": 0,
        "green": 0,
        "white": 0,
        "yellow": 0,
        "gray": 0,
        "dark": 0,
        "bright": 0,
        "high_sat": 0,
    }
    n = max(1, len(pixels))
    for (R, G, B) in pixels:
        r, g, b = R / 255.0, G / 255.0, B / 255.0
        h, s, v = _rgb_to_hsv(r, g, b)

        if v < 0.18:
            votes["dark"] += 1
        if v > 0.75:
            votes["bright"] += 1
        if s > 0.45:
            votes["high_sat"] += 1

        if v > 0.80 and s < 0.18:
            votes["white"] += 1
        if s < 0.18 and 0.25 < v < 0.85:
            votes["gray"] += 1
        if 35 <= h <= 70 and s > 0.20 and v > 0.35:
            votes["yellow"] += 1
        if 80 <= h <= 160 and s > 0.20 and v > 0.22:
            votes["green"] += 1
        if 190 <= h <= 250 and s > 0.18 and v > 0.22:
            votes["blue"] += 1

    return {k: float(v) / float(n) for k, v in votes.items()}


def infer_background(img) -> BackgroundResult:
    """
    Heuristic-only background inference.
    Returns tags like: outdoor/indoor, sky, water, greenery, snow, sand, city, night
    """
    border, extra = _sample_border_rgb(img)
    top = extra.pop("_top")
    bottom = extra.pop("_bottom")

    v_all = _color_votes(border)
    v_top = _color_votes(top)
    v_bot = _color_votes(bottom)

    tags: list[str] = []

    # indoor/outdoor heuristic:
    # - outdoor tends to have more high-saturation / sky/green/yellow on borders
    # - indoor tends to be darker/grayer on borders
    outdoor_score = (
        1.3 * v_all["high_sat"]
        + 1.1 * v_all["blue"]
        + 1.1 * v_all["green"]
        + 0.8 * v_all["yellow"]
        + 0.2 * v_all["bright"]
    )
    indoor_score = 1.2 * v_all["gray"] + 1.0 * v_all["dark"] + 0.3 * (1.0 - v_all["high_sat"])
    if outdoor_score >= indoor_score:
        tags.append("outdoor")
    else:
        tags.append("indoor")

    # sky / water split by where blue dominates
    blue_top = v_top["blue"]
    blue_bot = v_bot["blue"]
    if max(blue_top, blue_bot) >= 0.22:
        if blue_top >= blue_bot + 0.05:
            tags.append("sky")
        elif blue_bot >= blue_top + 0.05:
            tags.append("water")
        else:
            tags.append("blue_background")

    if v_all["green"] >= 0.18:
        tags.append("greenery")
    if v_all["yellow"] >= 0.18:
        tags.append("sand")
    if v_all["white"] >= 0.18:
        tags.append("snow")
    if v_all["gray"] >= 0.28 and v_all["high_sat"] < 0.25:
        tags.append("city")
    if v_all["dark"] >= 0.35:
        tags.append("night")

    # light cleanup
    tags = sorted(set(tags))
    debug = {
        "border_votes": v_all,
        "top_votes": v_top,
        "bottom_votes": v_bot,
        "meta": extra,
        "outdoor_score": outdoor_score,
        "indoor_score": indoor_score,
    }
    return BackgroundResult(tags=tags, caption=None, debug=debug)


def _tags_to_caption(tags: list[str]) -> str:
    """
    Create a short, background-focused caption when no vision caption model is available.
    """
    t = set(tags or [])
    parts: list[str] = []
    if "indoor" in t:
        parts.append("Indoors")
    elif "outdoor" in t:
        parts.append("Outdoors")

    # Prefer specific scene tags when present
    scene = None
    for s in ("beach", "mountain", "forest", "city", "snow", "water", "sky", "greenery", "sand", "night"):
        if s in t:
            scene = s
            break
    if scene:
        parts.append(f"({scene} background)")
    if not parts:
        parts.append("Photo")
    return " ".join(parts)


def infer_background_tags(img) -> list[str]:
    """
    Returns tags. If HF_TOKEN is set, attempts to enhance with HF image classification.
    Always returns something (heuristics at least).
    """
    base = infer_background(img)
    tags = set(base.tags)

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import InferenceClient
            import io

            # Broad scene classifier; output labels vary, so we map keywords.
            model = os.getenv("HF_VISION_MODEL", "google/vit-base-patch16-224")
            client = InferenceClient(model=model, token=hf_token)
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=85)
            preds = client.image_classification(buf.getvalue())
            text = " ".join([p.label.lower() for p in preds[:8]])

            def add_if(words: list[str], tag: str) -> None:
                for w in words:
                    if w in text:
                        tags.add(tag)
                        return

            add_if(["beach", "seashore", "coast"], "beach")
            add_if(["mountain", "alp", "cliff"], "mountain")
            add_if(["forest", "woodland", "jungle"], "forest")
            add_if(["city", "street", "downtown", "building", "skyscraper"], "city")
            add_if(["snow", "ice", "glacier"], "snow")
            add_if(["lake", "river", "ocean", "sea", "water"], "water")
            add_if(["desert", "sand"], "sand")
        except Exception:
            pass

    return sorted(tags)


def analyze_background(img) -> BackgroundResult:
    """
    Background-aware analysis using Hugging Face and Gemini APIs (when configured),
    with CNN fallback for local image identification and object detection.
    - HF_TOKEN: Hugging Face caption (BLIP) + classification (ViT)
    - GEMINI_API_KEY: Gemini vision for caption + tags
    - CNN (torch): ResNet50 + Faster R-CNN when APIs not configured
    Returns: tags + caption + debug blob (safe to store).
    """
    base = infer_background(img)
    tags = set(base.tags)
    debug: dict[str, Any] = {"heuristics": base.debug}
    caption: str | None = None

    # 1) Hugging Face + Gemini APIs (primary when configured)
    try:
        from pixel_app.core.vision_api import analyze_with_apis
        api_result = analyze_with_apis(img)
        if api_result:
            for t in api_result.tags or []:
                if t:
                    tags.add(t.strip())
            if api_result.caption:
                caption = api_result.caption
            debug["vision_api"] = api_result.debug
    except Exception as e:
        debug["vision_api_error"] = str(e)

    # 2) CNN fallback: when no API caption/tags, or to enrich
    if not caption or len(tags) < 3:
        try:
            from pixel_app.core.vision_cnn import analyze_with_cnn, CNNResult
            cnn_result = analyze_with_cnn(img)
            if isinstance(cnn_result, CNNResult):
                for t in (cnn_result.background_tags or []) + (cnn_result.object_tags or []):
                    if t:
                        tags.add(t.strip())
                if not caption and cnn_result.caption:
                    caption = cnn_result.caption
                debug["cnn"] = cnn_result.debug
        except Exception as e:
            debug["cnn_error"] = str(e)

    if not caption:
        caption = _tags_to_caption(sorted(tags))

    return BackgroundResult(tags=sorted(tags), caption=caption, debug=debug)

