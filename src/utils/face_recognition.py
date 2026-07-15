"""얼굴 검출/인식 유틸리티.

우선순위:
1. InsightFace + ONNX Runtime 기반 실사용 얼굴 인식
2. 패키지가 없을 때만 OpenCV 베이스라인으로 폴백
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..time_utils import now_kst_iso

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FACE_GALLERY = _PROJECT_ROOT / "known_faces.json"
_FACE_VECTOR_SIZE = (32, 32)
# det_size: CPU=160×160 (속도 우선), GPU=320×320 (정확도 우선)
# ctx_id:   CPU=-1, GPU=0 (cuda:N → N)


@dataclass
class FaceRecognitionResult:
    bbox: Dict[str, int]
    label: str
    confidence: float
    matched: bool
    age: Optional[float] = None
    gender: Optional[str] = None


class FaceRecognitionEngine:
    """등록 얼굴 갤러리 기반 얼굴 검출/인식 엔진."""

    def __init__(
        self,
        gallery_path: Optional[str] = None,
        similarity_threshold: float = 0.40,
        min_face_size: int = 40,
        device: str = "cpu",
    ) -> None:
        self.gallery_path = Path(gallery_path) if gallery_path else _DEFAULT_FACE_GALLERY
        self.similarity_threshold = similarity_threshold
        self.min_face_size = int(min_face_size)
        self._gallery_mtime: Optional[float] = None

        # device-aware InsightFace 설정
        # GPU(cuda): ctx_id ≥ 0, det_size 320 (정확도 우선)
        # CPU:       ctx_id = -1, det_size 160 (속도 우선)
        _is_gpu = device.lower().startswith("cuda")
        try:
            _gpu_idx = int(device.split(":")[1]) if ":" in device else 0
        except (ValueError, IndexError):
            _gpu_idx = 0
        self._ctx_id: int   = _gpu_idx if _is_gpu else -1
        self._det_size      = (320, 320) if _is_gpu else (160, 160)
        self._backend_setting = os.environ.get("FACE_RECOGNITION_BACKEND", "auto").strip().lower()
        self._recognition_disabled = self._backend_setting in {"disabled", "none", "off"}

        self.insight_app = self._load_insightface()
        self.detector = self._load_detector() if self.insight_app is None else None
        self.gallery = self._load_gallery()
        self._gallery_mtime = self._get_gallery_mtime()

    @property
    def enabled(self) -> bool:
        return self.insight_app is not None or self.detector is not None

    @property
    def backend_name(self) -> str:
        return "insightface_arcface" if self.insight_app is not None else "opencv_haar_baseline"

    def reload_gallery(self) -> List[Dict[str, str]]:
        self.gallery = self._load_gallery()
        self._gallery_mtime = self._get_gallery_mtime()
        return self.list_faces()

    def list_faces(self) -> List[Dict[str, str]]:
        return self._load_entries()

    def register_face(
        self,
        name: str,
        phone: str,
        image_base64: str,
        filename: Optional[str] = None,
        department: Optional[str] = None,
        position: Optional[str] = None,
        employee_id: Optional[str] = None,
        hired_at: Optional[str] = None,
        note: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Dict[str, str]:
        clean_name = str(name).strip()
        clean_phone = str(phone).strip()
        if not clean_name:
            raise ValueError("'name' is required")
        if not clean_phone:
            raise ValueError("'phone' is required")
        if not image_base64:
            raise ValueError("'image_base64' is required")

        # 전화번호 중복 체크
        existing = self._load_entries()
        for e in existing:
            if e.get("phone", "") == clean_phone:
                raise ValueError(f"이미 등록된 전화번호입니다: {clean_phone}")

        image_bytes, ext = self._decode_image_base64(image_base64, filename)
        faces_dir = self.gallery_path.parent / "known_faces"
        faces_dir.mkdir(parents=True, exist_ok=True)
        stored_name = f"{uuid.uuid4().hex[:12]}{ext}"
        stored_path = faces_dir / stored_name
        stored_path.write_bytes(image_bytes)

        entry: Dict[str, str] = {
            "id": uuid.uuid4().hex[:8],
            "name": clean_name,
            "phone": clean_phone,
            "image": f"known_faces/{stored_name}",
            "registered_at": now_kst_iso(),
        }
        if department is not None:
            entry["department"] = str(department).strip()
        if position is not None:
            entry["position"] = str(position).strip()
        if employee_id is not None:
            entry["employee_id"] = str(employee_id).strip()
        if hired_at is not None:
            entry["hired_at"] = str(hired_at).strip()
        if note is not None:
            entry["note"] = str(note).strip()

        existing.append(entry)
        self._write_entries(existing)
        self.reload_gallery()
        logger.info("얼굴 등록 완료: %s / %s (%s)", clean_name, clean_phone, entry["image"])
        return entry

    def delete_face(self, face_id: str) -> bool:
        entries = self._load_entries()
        remaining: List[Dict[str, str]] = []
        deleted_entry: Optional[Dict[str, str]] = None

        for entry in entries:
            if str(entry.get("id")) == face_id:
                deleted_entry = entry
            else:
                remaining.append(entry)

        if deleted_entry is None:
            return False

        image_rel = deleted_entry.get("image", "")
        image_path = (self.gallery_path.parent / image_rel).resolve()
        try:
            if image_path.exists():
                image_path.unlink()
        except Exception as exc:
            logger.warning("등록 얼굴 이미지 삭제 실패: %s", exc)

        self._write_entries(remaining)
        self.reload_gallery()
        logger.info("얼굴 등록 삭제 완료: %s", face_id)
        return True

    def detect_and_recognize(
        self,
        frame: np.ndarray,
        person_bbox: Dict[str, int],
    ) -> List[FaceRecognitionResult]:
        if frame is None or not self.enabled:
            return []

        self._refresh_gallery_if_needed()
        x = max(int(person_bbox.get("x", 0)), 0)
        y = max(int(person_bbox.get("y", 0)), 0)
        w = max(int(person_bbox.get("width", 0)), 0)
        h = max(int(person_bbox.get("height", 0)), 0)
        if w <= 0 or h <= 0:
            return []

        frame_h, frame_w = frame.shape[:2]
        x2 = min(x + w, frame_w)
        y2 = min(y + max(int(h * 0.6), self.min_face_size), frame_h)
        if x2 <= x or y2 <= y:
            return []

        roi = frame[y:y2, x:x2]
        if roi.size == 0:
            return []

        if self.insight_app is not None:
            return self._detect_with_insightface(roi, x, y)
        return self._detect_with_opencv(roi, x, y)

    def _detect_with_insightface(
        self,
        roi: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> List[FaceRecognitionResult]:
        try:
            faces = self.insight_app.get(roi)
        except Exception as exc:
            logger.warning("InsightFace 추론 실패: %s", exc)
            return []

        results: List[FaceRecognitionResult] = []
        for face in faces:
            bbox = getattr(face, "bbox", None)
            embedding = getattr(face, "embedding", None)
            if bbox is None or embedding is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            if width < self.min_face_size or height < self.min_face_size:
                continue

            label, score = self._match_embedding(np.asarray(embedding, dtype=np.float32))

            raw_age = getattr(face, "age", None)
            raw_gender = getattr(face, "gender", None)
            age: Optional[float] = float(raw_age) if raw_age is not None else None
            gender_str: Optional[str] = None
            if raw_gender is not None:
                gender_str = "female" if int(raw_gender) == 0 else "male"

            results.append(
                FaceRecognitionResult(
                    bbox={
                        "x": offset_x + x1,
                        "y": offset_y + y1,
                        "width": width,
                        "height": height,
                    },
                    label=label,
                    confidence=score,
                    matched=label != "unknown",
                    age=age,
                    gender=gender_str,
                )
            )
        return results

    def _detect_with_opencv(
        self,
        roi: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> List[FaceRecognitionResult]:
        if self.detector is None:
            return []

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )

        results: List[FaceRecognitionResult] = []
        for fx, fy, fw, fh in faces:
            face_crop = roi[fy:fy + fh, fx:fx + fw]
            vector = self._encode_fallback(face_crop)
            label, score = self._match_embedding(vector)
            results.append(
                FaceRecognitionResult(
                    bbox={
                        "x": int(offset_x + fx),
                        "y": int(offset_y + fy),
                        "width": int(fw),
                        "height": int(fh),
                    },
                    label=label,
                    confidence=float(score),
                    matched=label != "unknown",
                )
            )
        return results

    def _load_insightface(self):
        backend = getattr(
            self,
            "_backend_setting",
            os.environ.get("FACE_RECOGNITION_BACKEND", "auto").strip().lower(),
        )
        if backend in {"opencv", "haar", "disabled", "none", "off"}:
            logger.info("InsightFace 비활성화됨 (FACE_RECOGNITION_BACKEND=%s)", backend)
            return None

        insightface_required = backend == "insightface"
        cuda_required = self._ctx_id >= 0

        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:
            if insightface_required:
                raise RuntimeError("InsightFace 백엔드를 요청했지만 패키지를 불러올 수 없습니다") from exc
            logger.info("InsightFace 사용 불가, OpenCV 폴백 사용: %s", exc)
            return None

        try:
            providers = None
            if cuda_required:
                import onnxruntime as ort

                available_providers = ort.get_available_providers()
                if "CUDAExecutionProvider" not in available_providers:
                    raise RuntimeError(
                        "FACE_DEVICE가 CUDA를 요청했지만 ONNX Runtime에서 "
                        f"CUDAExecutionProvider를 사용할 수 없습니다: {available_providers}"
                    )
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

            app = FaceAnalysis(name="buffalo_l", providers=providers)
            app.prepare(ctx_id=self._ctx_id, det_size=self._det_size)
            logger.info(
                "InsightFace 얼굴 인식 활성화됨 (ctx_id=%d, det_size=%s, providers=%s)",
                self._ctx_id, self._det_size, providers or "runtime-default",
            )
            return app
        except Exception as exc:
            if insightface_required or cuda_required:
                raise RuntimeError(
                    f"요청한 InsightFace 실행 환경을 초기화하지 못했습니다 (ctx_id={self._ctx_id})"
                ) from exc
            logger.warning("InsightFace 초기화 실패, OpenCV 폴백 사용: %s", exc)
            return None

    def _load_detector(self):
        if getattr(self, "_recognition_disabled", False):
            logger.info("얼굴 인식 비활성화됨 (FACE_RECOGNITION_BACKEND=%s)", self._backend_setting)
            return None
        try:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(str(cascade_path))
            if detector.empty():
                logger.warning("얼굴 검출기 로드 실패: %s", cascade_path)
                return None
            return detector
        except Exception as exc:
            logger.warning("얼굴 검출기 초기화 실패: %s", exc)
            return None

    def _load_gallery(self) -> Dict[str, List[np.ndarray]]:
        gallery: Dict[str, List[np.ndarray]] = {}
        if getattr(self, "_recognition_disabled", False):
            return gallery
        for item in self._load_entries():
            name = str(item.get("name", "")).strip()
            image_path = str(item.get("image", "")).strip()
            if not name or not image_path:
                continue
            img = cv2.imread(str((self.gallery_path.parent / image_path).resolve()))
            if img is None:
                logger.warning("등록 얼굴 이미지 로드 실패: %s", image_path)
                continue
            embedding = self._extract_gallery_embedding(img)
            if embedding is None:
                logger.warning("등록 얼굴 특징 추출 실패: %s", image_path)
                continue
            gallery.setdefault(name, []).append(embedding)

        if gallery:
            logger.info("등록 얼굴 %d명 로드됨 (%s)", len(gallery), self.backend_name)
        return gallery

    def _extract_gallery_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        if self.insight_app is not None:
            try:
                faces = self.insight_app.get(image)
            except Exception as exc:
                logger.warning("등록 얼굴 InsightFace 분석 실패: %s", exc)
                return None
            if not faces:
                return None
            best = max(
                faces,
                key=lambda face: (float(face.bbox[2] - face.bbox[0]) * float(face.bbox[3] - face.bbox[1])),
            )
            embedding = getattr(best, "embedding", None)
            if embedding is None:
                return None
            return self._normalize_embedding(np.asarray(embedding, dtype=np.float32))
        return self._encode_fallback(image)

    def _encode_fallback(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        resized = cv2.resize(gray, _FACE_VECTOR_SIZE, interpolation=cv2.INTER_AREA)
        equalized = cv2.equalizeHist(resized)
        vector = equalized.astype(np.float32).reshape(-1)
        return self._normalize_embedding(vector)

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def _match_embedding(self, embedding: np.ndarray) -> Tuple[str, float]:
        if not self.gallery:
            return "unknown", 0.0

        embedding = self._normalize_embedding(embedding.astype(np.float32))
        best_name = "unknown"
        best_score = -1.0

        for name, vectors in self.gallery.items():
            if not vectors:
                continue
            score = max(float(np.dot(embedding, known_vector)) for known_vector in vectors)
            if score > best_score:
                best_name = name
                best_score = score

        if best_score < self.similarity_threshold:
            return "unknown", max(0.0, best_score)
        return best_name, best_score

    def _refresh_gallery_if_needed(self) -> None:
        current_mtime = self._get_gallery_mtime()
        if current_mtime is None:
            return
        if self._gallery_mtime is None or current_mtime > self._gallery_mtime:
            self.reload_gallery()

    def _get_gallery_mtime(self) -> Optional[float]:
        try:
            return self.gallery_path.stat().st_mtime
        except FileNotFoundError:
            return None

    def _load_entries(self) -> List[Dict[str, str]]:
        if not self.gallery_path.exists():
            return []

        try:
            payload = json.loads(self.gallery_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("known_faces.json 로드 실패: %s", exc)
            return []

        if not isinstance(payload, list):
            logger.warning("known_faces.json 형식이 올바르지 않습니다. 리스트가 필요합니다.")
            return []

        _OPTIONAL_FIELDS = (
            "phone", "department", "position", "employee_id",
            "hired_at", "registered_at", "note",
        )
        entries: List[Dict[str, str]] = []
        changed = False
        for item in payload:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            image = str(item.get("image", "")).strip()
            if not name or not image:
                continue
            entry: Dict[str, str] = {
                "id": str(item.get("id") or uuid.uuid4().hex[:8]),
                "name": name,
                "image": image,
            }
            if "id" not in item:
                changed = True
            for field in _OPTIONAL_FIELDS:
                if field in item and item[field] is not None:
                    entry[field] = str(item[field])
            entries.append(entry)

        if changed:
            self._write_entries(entries)
        return entries

    def _write_entries(self, entries: List[Dict[str, str]]) -> None:
        self.gallery_path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _decode_image_base64(
        self,
        image_base64: str,
        filename: Optional[str],
    ) -> Tuple[bytes, str]:
        payload = image_base64.strip()
        ext = self._infer_extension(filename)
        match = re.match(r"^data:image/([a-zA-Z0-9.+-]+);base64,(.+)$", payload)
        if match:
            ext = self._extension_from_mime(match.group(1))
            payload = match.group(2)
        try:
            data = base64.b64decode(payload, validate=True)
        except Exception as exc:
            raise ValueError("invalid image_base64 payload") from exc
        if not data:
            raise ValueError("decoded image is empty")
        return data, ext

    def _infer_extension(self, filename: Optional[str]) -> str:
        if not filename:
            return ".jpg"
        suffix = Path(filename).suffix.lower()
        return suffix if suffix in {".jpg", ".jpeg", ".png", ".bmp"} else ".jpg"

    def _extension_from_mime(self, mime_subtype: str) -> str:
        subtype = mime_subtype.lower()
        if subtype in {"jpeg", "jpg"}:
            return ".jpg"
        if subtype == "png":
            return ".png"
        if subtype == "bmp":
            return ".bmp"
        return ".jpg"


__all__ = ["FaceRecognitionEngine", "FaceRecognitionResult"]
