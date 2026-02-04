"""EdgeX 관련 모듈"""

from .device_service import CCTVDeviceService
from .edgex_wrapper import EdgeXCCTVProcessor

__all__ = ["CCTVDeviceService", "EdgeXCCTVProcessor"]
