"""DeepStream 카메라 기능 플래그와 추론 토폴로지 계산."""

from __future__ import annotations

from typing import Dict, Tuple


def feature_flags_for_camera(
    *,
    camera_ai_flags: Dict[str, Dict[str, bool]],
    camera_name: str,
    helmet_enabled: bool,
    face_enabled_default: bool,
    appearance_enabled_default: bool,
) -> Dict[str, bool]:
    flags = camera_ai_flags.get(camera_name)
    if flags is not None:
        return flags
    return {
        "use_helmet": helmet_enabled,
        "use_pose": True,
        "use_person": False,
        "use_face": face_enabled_default,
        "use_appearance": appearance_enabled_default,
    }


def any_camera_flag(
    camera_ai_flags: Dict[str, Dict[str, bool]],
    *flag_names: str,
) -> bool:
    return any(
        bool(flags.get(flag_name))
        for flags in camera_ai_flags.values()
        for flag_name in flag_names
    )


def inference_topology_signature(
    *,
    camera_ai_flags: Dict[str, Dict[str, bool]],
    helmet_enabled: bool,
    pphuman_sgie_enabled: bool,
    helmet_config_exists: bool,
    pphuman_config_exists: bool,
) -> Tuple[bool, bool, bool]:
    """현재 모델 플래그로 필요한 DeepStream nvinfer 구성을 계산한다."""
    primary_enabled = any_camera_flag(
        camera_ai_flags,
        "use_pose",
        "use_person",
        "use_face",
        "use_appearance",
    )
    helmet_topology_enabled = (
        helmet_enabled
        and any_camera_flag(camera_ai_flags, "use_helmet")
        and helmet_config_exists
    )
    pphuman_topology_enabled = (
        pphuman_sgie_enabled
        and any_camera_flag(camera_ai_flags, "use_appearance")
        and pphuman_config_exists
    )
    return primary_enabled, helmet_topology_enabled, pphuman_topology_enabled
