from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FaceDetection:
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float | None
    embedding: list[float]  # 512-dim, L2-normalized


class FaceEmbedder:
    """
    Local face detector + embedder based on facenet-pytorch.
    - Detector: MTCNN
    - Embedder: InceptionResnetV1 (vggface2)
    """

    def __init__(self, device: str | None = None) -> None:
        # Optional dependency: works when requirements-face.txt is installed.
        try:
            import torch  # type: ignore
            from facenet_pytorch import InceptionResnetV1, MTCNN  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Local face model not available. Install requirements-face.txt (Python 3.11/3.12 recommended)."
            ) from e

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device=device, select_largest=False, post_process=True)
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        self._torch = torch

    def detect_and_embed(self, pil_image: Any) -> list[FaceDetection]:
        torch = self._torch

        boxes, probs = self.mtcnn.detect(pil_image)
        if boxes is None or len(boxes) == 0:
            return []

        # Extract aligned face tensors for each detected box
        face_tensors = self.mtcnn.extract(pil_image, boxes, None)
        if face_tensors is None or len(face_tensors) == 0:
            return []

        face_tensors = face_tensors.to(self.device)
        with torch.no_grad():
            emb = self.model(face_tensors).detach().cpu()

        out: list[FaceDetection] = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
            conf = None if probs is None else float(probs[i])
            vec = emb[i].tolist()
            out.append(FaceDetection(bbox_xyxy=(x1, y1, x2, y2), confidence=conf, embedding=l2_normalize(vec)))
        return out


def l2_normalize(v: list[float]) -> list[float]:
    s = 0.0
    for x in v:
        s += float(x) * float(x)
    n = math.sqrt(s) + 1e-12
    return [float(x) / n for x in v]

