"""
CNN-based image identification and object detection for background/scene tags,
object tags, and image description (caption).

- Object detection: Faster R-CNN (COCO) → detected objects as tags.
- Image classification: ResNet50 (ImageNet) → top scene/object labels as tags.
- Background/scene tags are merged with heuristics in background.py.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# COCO instance category names (91 entries: __background__ + 80 classes; indices 81-90 reserved).
# Indices match torchvision's Faster R-CNN COCO model output (0 = background, 1-90 object ids).
COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
] + [""] * (91 - 81)  # pad to 91 for safe index access


@dataclass(frozen=True)
class CNNResult:
    """Result of CNN-based image analysis."""
    background_tags: list[str]   # scene/background from classification + detection context
    object_tags: list[str]       # detected objects (COCO) + top ImageNet labels
    caption: str | None         # short description from objects + scene
    debug: dict[str, Any]


def _get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_imagenet_labels() -> list[str]:
    """Load ImageNet 1000 class labels (one per line, index = line number)."""
    path = Path(__file__).resolve().parent / "imagenet_classes.txt"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def _pil_to_tensor(pil_image: Any, size: tuple[int, int] = (224, 224)) -> Any:
    """Convert PIL Image to normalized tensor (C, H, W) for torchvision models."""
    import torch
    from torchvision import transforms

    t = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = pil_image.convert("RGB")
    return t(img).unsqueeze(0)


def _run_object_detection(pil_image: Any, device: str) -> list[tuple[str, float]]:
    """Run Faster R-CNN COCO object detection. Returns list of (class_name, score)."""
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    model.eval().to(device)

    # Preprocess: model expects list of tensors [C, H, W] in 0-1 range
    from torchvision import transforms
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(pil_image.convert("RGB")).to(device)

    with torch.no_grad():
        preds = model([img_tensor])

    out: list[tuple[str, float]] = []
    conf_threshold = 0.4
    boxes = preds[0]["boxes"].cpu()
    labels = preds[0]["labels"].cpu()
    scores = preds[0]["scores"].cpu()

    for box, label_idx, score in zip(boxes, labels, scores):
        if score.item() < conf_threshold:
            continue
        idx = int(label_idx.item())
        if idx <= 0 or idx >= len(COCO_INSTANCE_CATEGORY_NAMES):
            continue
        name = (COCO_INSTANCE_CATEGORY_NAMES[idx] or "").strip()
        if name and name != "__background__":
            out.append((name, float(score.item())))
    return out


def _run_classification(pil_image: Any, device: str, top_k: int = 5) -> list[tuple[str, float]]:
    """Run ResNet50 ImageNet classification. Returns top-k (label, score)."""
    import torch
    from torchvision.models import resnet50
    from torchvision.models import ResNet50_Weights

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.eval().to(device)

    x = _pil_to_tensor(pil_image).to(device)
    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits[0], dim=0).cpu()
    top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))

    labels = _load_imagenet_labels()
    out: list[tuple[str, float]] = []
    for idx, prob in zip(top_indices.tolist(), top_probs.tolist()):
        if idx < len(labels) and labels[idx]:
            label = labels[idx].replace(" ", "_").lower()
            out.append((label, float(prob)))
    return out


def _scene_like(label: str) -> bool:
    """Heuristic: label looks like a scene/background (for caption)."""
    scene_words = {
        "beach", "seashore", "lakeside", "cliff", "valley", "volcano", "alp",
        "coral_reef", "geyser", "promontory", "sandbar", "outdoor", "indoor",
        "church", "palace", "castle", "restaurant", "library", "grocery_store",
        "bakery", "barbershop", "barn", "boathouse", "cinema", "greenhouse",
        "lumbermill", "monastery", "mosque", "stupa", "theater", "tower",
    }
    return label.lower() in scene_words or any(w in label.lower() for w in ("background", "landscape", "scene"))


def analyze_with_cnn(pil_image: Any) -> CNNResult:
    """
    Run CNN-based image identification and object detection.
    - Object detection (Faster R-CNN) → object_tags.
    - Image classification (ResNet50) → background_tags (scene-like) + object_tags.
    - Build a short caption from detected objects and top scene/object labels.
    """
    debug: dict[str, Any] = {}
    object_tags: list[str] = []
    background_tags: list[str] = []
    caption_parts: list[str] = []

    try:
        import torch
        from PIL import Image
    except Exception as e:
        debug["error"] = f"CNN dependencies missing: {e}"
        return CNNResult(
            background_tags=[],
            object_tags=[],
            caption=None,
            debug=debug,
        )

    device = _get_device()
    debug["device"] = device

    # 1) Object detection
    try:
        detections = _run_object_detection(pil_image, device)
        for name, score in detections[:15]:
            tag = name.replace(" ", "_").lower()
            if tag and tag not in {t.split(":")[0] for t in object_tags}:
                object_tags.append(tag)
            if len(caption_parts) < 5:
                caption_parts.append(name)
        debug["detections"] = [(n, round(s, 2)) for n, s in detections[:10]]
    except Exception as e:
        debug["detection_error"] = str(e)

    # 2) Image classification (ResNet) for scene/object labels
    try:
        top_labels = _run_classification(pil_image, device, top_k=5)
        for label, score in top_labels:
            tag = label.replace(" ", "_").lower()
            if _scene_like(tag) or score >= 0.15:
                if tag not in background_tags and tag not in object_tags:
                    background_tags.append(tag)
            if tag not in object_tags and score >= 0.1:
                object_tags.append(tag)
        debug["classification_top5"] = [(l, round(s, 2)) for l, s in top_labels]
    except Exception as e:
        debug["classification_error"] = str(e)

    # Deduplicate and limit
    object_tags = list(dict.fromkeys(object_tags))[:25]
    background_tags = list(dict.fromkeys(background_tags))[:15]

    # Build caption
    if caption_parts:
        caption = "Photo with " + ", ".join(caption_parts[:5])
        if len(caption_parts) > 5:
            caption += " and more."
    else:
        caption = None
        if background_tags:
            caption = "Scene: " + ", ".join(background_tags[:5]).replace("_", " ")

    return CNNResult(
        background_tags=background_tags,
        object_tags=object_tags,
        caption=caption,
        debug=debug,
    )
