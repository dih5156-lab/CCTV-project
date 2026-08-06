"""Persistent CRUD and recognition service for YuNet/SFace TensorRT."""

from __future__ import annotations

import base64
import os
import re
import uuid
from pathlib import Path

import cv2
import numpy as np

from ._commercial_face_recognizer import CommercialFaceRecognizer
from ._commercial_face_tensorrt import (
    CommercialFaceEmbeddingPipeline,
    TensorRTSFaceEmbedder,
    TensorRTYuNetDetector,
)
from ._sqlite_face_gallery import SQLiteFaceGallery


class CommercialTensorRTFaceService:
    backend_name = "opencv_yunet_sface_tensorrt"
    enabled = True

    def __init__(
        self,
        *,
        model_dir: Path = Path("models/commercial_face"),
        database_path: Path = Path("data/runtime/commercial_faces.db"),
        faces_dir: Path = Path("known_faces"),
        gallery: SQLiteFaceGallery | None = None,
        recognizer: CommercialFaceRecognizer | None = None,
    ) -> None:
        self.faces_dir = Path(faces_dir)
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.gallery = gallery or SQLiteFaceGallery(
            database_path, model_id="opencv-sface-tensorrt-v1"
        )
        if recognizer is None:
            embedding_pipeline = CommercialFaceEmbeddingPipeline(
                TensorRTYuNetDetector(Path(model_dir) / "yunet_fp16.engine"),
                TensorRTSFaceEmbedder(Path(model_dir) / "sface_fp16.engine"),
            )
            recognizer = CommercialFaceRecognizer(
                embedding_pipeline,
                self.gallery,
                similarity_threshold=float(os.getenv("FACE_SIMILARITY_THRESHOLD", "0.6")),
                similarity_margin=float(os.getenv("FACE_SIMILARITY_MARGIN", "0.12")),
            )
        self.recognizer = recognizer

    def detect_and_recognize(self, frame, person_bbox):
        return self.recognizer.detect_and_recognize(frame, person_bbox)

    def list_faces(self) -> list[dict[str, object]]:
        return self.gallery.list_people()

    def reload_gallery(self) -> list[dict[str, object]]:
        self.gallery.reload()
        return self.list_faces()

    def register_face(
        self,
        name: str,
        phone: str,
        image_base64: str,
        filename: str | None = None,
        category: str | None = None,
        **metadata: object,
    ) -> dict[str, object]:
        clean_name = str(name).strip()
        clean_phone = str(phone).strip()
        if not clean_name or not clean_phone or not image_base64:
            raise ValueError("name, phone, and image_base64 are required")
        image_bytes, extension = self._decode_image(image_base64, filename)
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("registered image could not be decoded")
        embedding = self.recognizer.extract_enrollment_embedding(image)

        stored_name = f"{uuid.uuid4().hex[:12]}{extension}"
        stored_path = self.faces_dir / stored_name
        stored_path.write_bytes(image_bytes)
        relative_path = f"known_faces/{stored_name}"
        existing = next(
            (person for person in self.list_faces() if person["phone"] == clean_phone),
            None,
        )
        try:
            if existing is not None:
                return self.gallery.add_samples(
                    str(existing["person_id"]), [embedding], [relative_path]
                )
            return self.gallery.enroll_person(
                {
                    "person_id": uuid.uuid4().hex[:8],
                    "name": clean_name,
                    "phone": clean_phone,
                    "category": category or "employee",
                },
                [embedding],
                [relative_path],
            )
        except Exception:
            stored_path.unlink(missing_ok=True)
            raise

    def delete_face(self, face_id: str) -> bool:
        try:
            person = self.gallery.get_person(face_id)
        except KeyError:
            return False
        if not self.gallery.delete(face_id):
            return False
        for image_path in person.get("images", []):
            filename = Path(str(image_path)).name
            (self.faces_dir / filename).unlink(missing_ok=True)
        return True

    @staticmethod
    def _decode_image(payload: str, filename: str | None) -> tuple[bytes, str]:
        encoded = str(payload).strip()
        extension = Path(filename or "face.jpg").suffix.lower()
        if extension not in {".jpg", ".jpeg", ".png", ".bmp"}:
            extension = ".jpg"
        match = re.match(r"^data:image/([a-zA-Z0-9.+-]+);base64,(.+)$", encoded)
        if match:
            encoded = match.group(2)
            extension = ".jpg" if match.group(1).lower() in {"jpg", "jpeg"} else f".{match.group(1).lower()}"
        try:
            data = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise ValueError("invalid image_base64 payload") from exc
        if not data:
            raise ValueError("decoded image is empty")
        return data, extension


def create_face_recognition_engine(*, device: str = "cpu"):
    backend = os.getenv("FACE_RECOGNITION_BACKEND", "auto").strip().lower()
    if backend in {"commercial_tensorrt", "yunet_sface_tensorrt"}:
        return CommercialTensorRTFaceService(
            model_dir=Path(os.getenv("COMMERCIAL_FACE_MODEL_DIR", "models/commercial_face")),
            database_path=Path(
                os.getenv("COMMERCIAL_FACE_DATABASE", "data/runtime/commercial_faces.db")
            ),
        )
    from ...utils.face_recognition import FaceRecognitionEngine

    return FaceRecognitionEngine(device=device)
