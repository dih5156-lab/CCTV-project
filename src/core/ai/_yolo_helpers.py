"""YOLO 추론 결과 추출 유틸리티.

AIAnalyzer 클래스에서 사용하던 정적(static) 헬퍼 함수들을 모듈 레벨로 분리.
단독 테스트 가능하며 다른 검출기 클래스에서도 재사용할 수 있다.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── YOLO 결과 추출 ────────────────────────────────────────────────────


def extract_bbox(box) -> Optional[Tuple[int, int, int, int]]:
    """YOLO box에서 bbox 좌표 추출 → (x1, y1, x2, y2)."""
    try:
        xyxy_tensor = box.xyxy[0]
        if hasattr(xyxy_tensor, "cpu"):
            xyxy = xyxy_tensor.cpu().numpy().astype(int)
        else:
            xyxy = np.array(xyxy_tensor).astype(int)
        return int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
    except (ValueError, TypeError, IndexError) as exc:
        logger.debug("bbox 추출 실패: %s", exc)
        return None


def extract_confidence(box) -> float:
    """YOLO box에서 신뢰도 추출."""
    try:
        conf_tensor = box.conf[0]
        if hasattr(conf_tensor, "cpu"):
            return float(conf_tensor.cpu().numpy())
        return float(conf_tensor)
    except (ValueError, TypeError, IndexError):
        return 0.0


def extract_keypoints(keypoints, idx: int) -> Optional[np.ndarray]:
    """YOLO pose 결과에서 키포인트 배열 추출 → shape (N, 3): [x, y, confidence]."""
    try:
        if hasattr(keypoints, "data"):
            kpts = keypoints.data[idx]
            if hasattr(kpts, "cpu"):
                return kpts.cpu().numpy()
            return kpts
        if hasattr(keypoints, "xy"):
            kpts_xy   = keypoints.xy[idx]
            kpts_conf = keypoints.conf[idx]
            if hasattr(kpts_xy, "cpu"):
                kpts_xy   = kpts_xy.cpu().numpy()
                kpts_conf = kpts_conf.cpu().numpy()
            return np.column_stack([kpts_xy, kpts_conf])
        return None
    except Exception as exc:
        logger.debug("포인트 추출 실패: %s", exc)
        return None


def extract_track_id(box) -> Optional[int]:
    """YOLOv8 track() 결과에서 추적 ID 추출."""
    if not hasattr(box, "id") or box.id is None:
        return None
    try:
        track_id = box.id[0]
        if hasattr(track_id, "cpu"):
            return int(track_id.cpu().numpy())
        return int(track_id)
    except (ValueError, TypeError, IndexError, AttributeError) as exc:
        logger.debug("추적 ID 추출 실패: %s", exc)
        return None


# ── Bbox 계산 유틸리티 ────────────────────────────────────────────────


def bbox_iou_from_coords(
    bbox1: Tuple[int, int, int, int],
    bbox2: Tuple[int, int, int, int],
) -> float:
    """두 bbox (x1, y1, w, h) 간 IoU 계산."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
    inter_w    = max(0, x2 - x1)
    inter_h    = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area1      = max(0, bbox1[2]) * max(0, bbox1[3])
    area2      = max(0, bbox2[2]) * max(0, bbox2[3])
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def center_distance_ratio(
    bbox1: Tuple[int, int, int, int],
    bbox2: Tuple[int, int, int, int],
) -> float:
    """두 bbox 중심 간 거리를 bbox 크기로 정규화한 비율."""
    c1x = bbox1[0] + (bbox1[2] / 2.0)
    c1y = bbox1[1] + (bbox1[3] / 2.0)
    c2x = bbox2[0] + (bbox2[2] / 2.0)
    c2y = bbox2[1] + (bbox2[3] / 2.0)
    dist  = ((c1x - c2x) ** 2 + (c1y - c2y) ** 2) ** 0.5
    scale = max(1.0, bbox1[2], bbox1[3], bbox2[2], bbox2[3])
    return dist / scale


def generate_temp_id(x: int, y: int, w: int, h: int) -> int:
    """위치 기반 결정론적 임시 객체 ID 생성.

    Returns:
        1_500_000_000 ~ 1_999_999_999 범위의 정수
    """
    cx = x + max(w, 0) // 2
    cy = y + max(h, 0) // 2
    return 1_500_000_000 + (abs(cx * 10_000 + cy * 17) % 500_000_000)


# ── 엔진 imgsz 자동 감지 ─────────────────────────────────────────────


def detect_engine_imgsz(model, fallback: int) -> int:
    """로드된 YOLO 모델에서 실제 입력 이미지 크기를 자동 감지한다.

    TensorRT .engine 파일은 컴파일 시 입력 shape이 고정되므로
    ultralytics가 노출하는 메타데이터에서 imgsz를 읽어 인적 오류를 방지한다.
    .pt 파일이거나 감지 실패 시 fallback 값을 반환한다.

    탐색 우선순위:
      1. model.model.imgsz  (ultralytics TensorRT 래퍼)
      2. model.overrides["imgsz"]  (저장된 학습 설정)
      3. ICudaEngine 첫 번째 바인딩 입력 shape  (tensorrt 직접 접근)
    """
    if model is None:
        return fallback
    try:
        inner      = getattr(model, "model", None)
        imgsz_attr = getattr(inner, "imgsz", None)
        if imgsz_attr is not None:
            size = int(imgsz_attr[0]) if isinstance(imgsz_attr, (list, tuple)) else int(imgsz_attr)
            if size > 0:
                logger.info("엔진 imgsz 자동 감지 (model.model.imgsz): %d", size)
                return size
    except Exception:
        pass
    try:
        overrides = getattr(model, "overrides", {}) or {}
        val = overrides.get("imgsz")
        if val is not None:
            size = int(val[0]) if isinstance(val, (list, tuple)) else int(val)
            if size > 0:
                logger.info("엔진 imgsz 자동 감지 (overrides): %d", size)
                return size
    except Exception:
        pass
    try:
        import tensorrt as trt  # type: ignore
        engine = getattr(getattr(model, "model", None), "engine", None)
        if engine is not None and isinstance(engine, trt.ICudaEngine):
            binding_name = engine.get_binding_name(0)
            shape = engine.get_binding_shape(binding_name)
            size  = int(shape[-1])
            if size > 0:
                logger.info("엔진 imgsz 자동 감지 (TRT binding[0]): %d", size)
                return size
    except Exception:
        pass
    logger.debug("엔진 imgsz 자동 감지 실패 → 폴백 값 사용: %d", fallback)
    return fallback


# ── 기타 헬퍼 ────────────────────────────────────────────────────────


def age_to_group(age: float) -> str:
    """추정 나이(float) → 연령대 문자열 변환."""
    a = int(age)
    if a < 10:
        return "10대 미만"
    decade = (a // 10) * 10
    return f"{decade}대"
