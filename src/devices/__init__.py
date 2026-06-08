"""디바이스 컨트롤러 패키지

각 물리 디바이스(스피커, 전광판, 센서/경광등)를 독립 모듈로 관리한다.
"""

from .sensor_device import SensorReading
from .signboard import SignboardConfig, SignboardDevice
from .siren import SensorConfig, SirenDevice
from .speaker import SpeakerConfig, SpeakerDevice

__all__ = [
    "SpeakerDevice", "SpeakerConfig",
    "SignboardDevice", "SignboardConfig",
    "SirenDevice", "SensorConfig",
    "SensorReading",
]
