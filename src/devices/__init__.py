"""디바이스 컨트롤러 패키지

각 물리 디바이스(스피커, 전광판, 센서/경광등)를 독립 모듈로 관리한다.
"""

from .speaker import SpeakerDevice, SpeakerConfig
from .signboard import SignboardDevice, SignboardConfig
from .siren import SirenDevice, SensorConfig
from .sensor_device import SensorReading

__all__ = [
    "SpeakerDevice", "SpeakerConfig",
    "SignboardDevice", "SignboardConfig",
    "SirenDevice", "SensorConfig",
    "SensorReading",
]
