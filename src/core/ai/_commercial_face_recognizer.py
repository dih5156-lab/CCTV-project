"""Adapter from the commercial TensorRT pipeline to the existing face contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class CommercialFaceRecognitionResult:
    bbox: dict[str, int]
    label: str
    confidence: float
    matched: bool
    decision: str
    person_id: str | None
    category: str | None
    second_best_similarity: float
    margin: float
    model_id: str
    age: float | None = None
    gender: str | None = None


class CommercialFaceRecognizer:
    """Expose YuNet/SFace/gallery results through detect_and_recognize()."""

    enabled = True
    backend_name = "opencv_yunet_sface_tensorrt"

    def __init__(
        self,
        embedding_pipeline: object,
        gallery: object,
        *,
        similarity_threshold: float = 0.5,
        similarity_margin: float = 0.1,
        top_k: int = 5,
    ) -> None:
        self.embedding_pipeline = embedding_pipeline
        self.gallery = gallery
        self.similarity_threshold = float(similarity_threshold)
        self.similarity_margin = float(similarity_margin)
        self.top_k = int(top_k)

    def detect_and_recognize(
        self,
        frame: np.ndarray,
        person_bbox: Mapping[str, int],
    ) -> list[CommercialFaceRecognitionResult]:
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return []
        x = max(int(person_bbox.get("x", 0)), 0)
        y = max(int(person_bbox.get("y", 0)), 0)
        width = max(int(person_bbox.get("width", 0)), 0)
        height = max(int(person_bbox.get("height", 0)), 0)
        if width <= 0 or height <= 0:
            return []
        frame_height, frame_width = frame.shape[:2]
        x2 = min(x + width, frame_width)
        y2 = min(y + max(int(height * 0.6), 1), frame_height)
        if x >= frame_width or y >= frame_height or x2 <= x or y2 <= y:
            return []

        embedded_faces = self.embedding_pipeline.extract_embeddings(
            frame, (x, y, x2 - x, y2 - y)
        )
        results: list[CommercialFaceRecognitionResult] = []
        for embedded_face in embedded_faces:
            search = self.gallery.search(
                embedded_face.embedding,
                top_k=self.top_k,
                threshold=self.similarity_threshold,
                margin=self.similarity_margin,
            )
            best = search.best
            face_x, face_y, face_width, face_height = embedded_face.face.bbox
            results.append(
                CommercialFaceRecognitionResult(
                    bbox={
                        "x": int(round(face_x)),
                        "y": int(round(face_y)),
                        "width": int(round(face_width)),
                        "height": int(round(face_height)),
                    },
                    label=best.name if search.matched and best is not None else "unknown",
                    confidence=float(best.similarity if best is not None else 0.0),
                    matched=bool(search.matched),
                    decision=str(search.decision),
                    person_id=best.person_id if search.matched and best is not None else None,
                    category=best.category if search.matched and best is not None else None,
                    second_best_similarity=float(search.second_best_similarity),
                    margin=float(search.margin),
                    model_id=str(search.model_id),
                )
            )
        return results

    def extract_enrollment_embedding(self, image: np.ndarray) -> np.ndarray:
        if (
            not isinstance(image, np.ndarray)
            or image.size == 0
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            raise ValueError("enrollment image must be a non-empty BGR image")
        results = self.embedding_pipeline.extract_embeddings(
            image, (0, 0, image.shape[1], image.shape[0])
        )
        if len(results) != 1:
            raise ValueError(
                "enrollment image must contain exactly one face, "
                f"detected {len(results)}"
            )
        return np.ascontiguousarray(results[0].embedding, dtype=np.float32)
