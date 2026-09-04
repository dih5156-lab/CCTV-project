"""EdgeX 관련 모듈

redis / paho-mqtt 가 설치되지 않은 환경에서도 패키지를 임포트할 수 있도록
직접 임포트 대신 지연(lazy) 임포트 패턴을 사용한다.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .adapter_service import EdgeXDeviceAdapterService
    from .device_service import CCTVDeviceService
    from .siren_device_service import SirenDeviceService
    from .speaker_device_service import SpeakerDeviceService

__all__ = [
    "CCTVDeviceService",
    "EdgeXDeviceAdapterService",
    "SpeakerDeviceService",
    "SirenDeviceService",
]


def __getattr__(name: str):
    """EdgeX 관련 클래스를 실제 사용 시점에 지연 임포트한다."""
    if name == "CCTVDeviceService":
        from .device_service import CCTVDeviceService
        return CCTVDeviceService
    if name == "EdgeXDeviceAdapterService":
        from .adapter_service import EdgeXDeviceAdapterService
        return EdgeXDeviceAdapterService
    if name == "SpeakerDeviceService":
        from .speaker_device_service import SpeakerDeviceService
        return SpeakerDeviceService
    if name == "SirenDeviceService":
        from .siren_device_service import SirenDeviceService
        return SirenDeviceService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
