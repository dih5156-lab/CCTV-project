"""ActionBridge MQTT 토픽 정의와 기본 구독 세트."""

from __future__ import annotations

from typing import Set

CMD_TOPIC_MODE = "cctv/commands/mode"  # {"site_id"?, "mode": "auto|manual"}
CMD_TOPIC_APPROVE = "cctv/commands/approve"  # {"event_id": "..."}
CMD_TOPIC_REJECT = "cctv/commands/reject"  # {"event_id": "..."}
STATUS_TOPIC_PREFIX = "cctv/status/action"

ZONE_TOPICS = {
    "cctv/ai/events/+/zone_entered",
    "cctv/ai/events/+/zone_dwelling",
    "cctv/ai/events/+/zone_object_detected",
    "cctv/ai/events/+/crowd_warning",
}
DETECTION_TOPICS = {
    "cctv/ai/events/+/person",
    "cctv/ai/events/+/fall_detected",
    "cctv/ai/events/+/unsafe_behavior",
    "cctv/ai/events/+/helmet",
    "cctv/ai/events/+/head",
    "cctv/ai/events/+/face_unknown",
    "cctv/ai/events/+/face_recognized",
}
INTRUSION_TOPICS = {
    "cctv/rules/intrusion/filtered",
    "cctv/rules/intrusion/persisted",
    "cctv/rules/intrusion/critical",
}
SENSOR_TOPICS = {
    "aiot/rules/sensor/tilt",
    "aiot/rules/sensor/temperature",
    "aiot/rules/sensor/vibration",
}

DEFAULT_SUBSCRIBE_TOPICS = (
    INTRUSION_TOPICS | ZONE_TOPICS | DETECTION_TOPICS | SENSOR_TOPICS
)

DEFAULT_ALARM_TOPICS = (
    {"cctv/rules/intrusion/persisted", "cctv/rules/intrusion/critical"}
    | ZONE_TOPICS
    | {
        "cctv/ai/events/+/helmet",
        "cctv/ai/events/+/head",
        "cctv/ai/events/+/fall_detected",
        "cctv/ai/events/+/unsafe_behavior",
    }
    | SENSOR_TOPICS
)


def default_subscribe_topics() -> Set[str]:
    """ActionBridge가 기본으로 저장 구독할 MQTT 토픽 목록을 반환한다."""
    return set(DEFAULT_SUBSCRIBE_TOPICS)


def default_alarm_topics() -> Set[str]:
    """ActionBridge가 기본으로 알람 처리할 MQTT 토픽 목록을 반환한다."""
    return set(DEFAULT_ALARM_TOPICS)
