"""DeepStream camera source 상태 관리 유틸리티."""

from __future__ import annotations

from typing import Any, Dict, Optional


def clear_source_state(info: Dict[str, Any]) -> None:
    """카메라 source element와 streammux pad 상태를 초기화한다."""
    info["src_element"] = None
    info["sinkpad"] = None
    info["pad_id"] = None


def set_source_state(
    info: Dict[str, Any],
    *,
    src_element: Any,
    sinkpad: Any,
    pad_id: int,
) -> None:
    """카메라 source element와 streammux pad 상태를 저장한다."""
    info["src_element"] = src_element
    info["sinkpad"] = sinkpad
    info["pad_id"] = pad_id


def next_available_pad_id(cameras: Dict[str, Dict[str, Any]]) -> int:
    """현재 사용 중인 source pad를 피해서 다음 pad id를 반환한다."""
    used = {
        int(info["pad_id"])
        for info in cameras.values()
        if info.get("pad_id") is not None
    }
    pad_id = 0
    while pad_id in used:
        pad_id += 1
    return pad_id


def count_attached_sources(cameras: Dict[str, Dict[str, Any]]) -> int:
    """현재 파이프라인에 붙어 있는 source 수를 반환한다."""
    return sum(1 for info in cameras.values() if info.get("src_element") is not None)


def rebuild_pad_to_camera(cameras: Dict[str, Dict[str, Any]]) -> Dict[int, str]:
    """카메라 상태에서 pad_id -> camera_id 캐시를 다시 만든다."""
    return {
        int(info["pad_id"]): camera_id
        for camera_id, info in cameras.items()
        if info.get("pad_id") is not None
    }


def remove_pad_mapping(
    pad_to_camera: Dict[int, str],
    pad_id: Optional[int],
) -> None:
    """pad id가 있으면 pad_id -> camera_id 캐시에서 제거한다."""
    if pad_id is not None:
        pad_to_camera.pop(int(pad_id), None)
