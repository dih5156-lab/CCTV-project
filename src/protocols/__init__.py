"""protocols 패키지 - 외부 통신 프로토콜 클라이언트.

무거운 하위 모듈은 필요할 때 지연 로드한다.
"""

from typing import Any

__all__ = [
    "MqttEventPublisher",
    "MqttTopicSubscriber",
    "HttpEventForwarder",
    "HttpEventTarget",
    "RestEventReceiver",
    "GoTLVDecoderClient",
]


def __getattr__(name: str) -> Any:
    if name == "MqttEventPublisher":
        from .mqtt_publisher import MqttEventPublisher

        return MqttEventPublisher
    if name == "MqttTopicSubscriber":
        from .mqtt_subscriber import MqttTopicSubscriber

        return MqttTopicSubscriber
    if name in {"HttpEventForwarder", "HttpEventTarget"}:
        from .http import HttpEventForwarder, HttpEventTarget

        return {
            "HttpEventForwarder": HttpEventForwarder,
            "HttpEventTarget": HttpEventTarget,
        }[name]
    if name == "RestEventReceiver":
        from .rest import RestEventReceiver

        return RestEventReceiver
    if name == "GoTLVDecoderClient":
        from .tlv_decoder import GoTLVDecoderClient

        return GoTLVDecoderClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
