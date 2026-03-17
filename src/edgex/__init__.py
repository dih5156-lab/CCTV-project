"""EdgeX 관련 모듈

redis / paho-mqtt 가 설치되지 않은 환경에서도 패키지를 임포트할 수 있도록
직접 임포트 대신 지연(lazy) 임포트 패턴을 사용한다.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .device_service import CCTVDeviceService
    from .adapter_service import EdgeXDeviceAdapterService

__all__ = ["CCTVDeviceService", "EdgeXDeviceAdapterService"]


def __getattr__(name: str):
    if name == "CCTVDeviceService":
        from .device_service import CCTVDeviceService
        return CCTVDeviceService
    if name == "EdgeXDeviceAdapterService":
        from .adapter_service import EdgeXDeviceAdapterService
        return EdgeXDeviceAdapterService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
