"""DeepStream 카메라별 모델 활성화 플래그 변환."""

from __future__ import annotations

from typing import Dict, List


def normalize_model_flags(flags: Dict[str, object]) -> Dict[str, bool]:
    use_pose = bool(flags.get("use_pose", flags.get("pose", False)))
    use_helmet = bool(flags.get("use_helmet", flags.get("helmet", False)))
    use_person = bool(flags.get("use_person", flags.get("person", False)))
    use_face = bool(flags.get("use_face", flags.get("face", False)))
    use_appearance = bool(flags.get("use_appearance", flags.get("appearance", False)))

    return {
        "use_helmet": use_helmet,
        "use_pose": use_pose,
        "use_person": use_person,
        "use_face": use_face,
        "use_appearance": use_appearance,
    }


def flags_to_detection_modes(flags: Dict[str, object]) -> List[str]:
    normalized = normalize_model_flags(flags)
    modes: List[str] = []
    if normalized["use_pose"]:
        modes.extend(["fall", "person"])
    if normalized["use_helmet"]:
        modes.append("helmet")
    if normalized["use_face"]:
        modes.append("face")
    if normalized["use_person"] and "person" not in modes:
        modes.append("person")
    if normalized["use_appearance"]:
        modes.append("appearance")
    return modes
