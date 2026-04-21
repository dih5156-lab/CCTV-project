"""핵심 모듈 - 분석 및 처리 로직"""

from .events import DetectionEvent, EventType
from .sensor_detection import SensorAlertEvent, SensorEventDetector, SensorRuleConfig

try:
    from .ai.analyzer import AIAnalyzer
    from .base_processor import BaseProcessor
    from .processor import VideoProcessor
    from .deepstream_processor import DEEPSTREAM_AVAILABLE, DeepStreamProcessor
except ImportError:
    AIAnalyzer = None  # type: ignore[assignment,misc]
    BaseProcessor = None  # type: ignore[assignment,misc]
    VideoProcessor = None  # type: ignore[assignment,misc]
    DEEPSTREAM_AVAILABLE = False
    DeepStreamProcessor = None  # type: ignore[assignment,misc]

__all__ = [
    'DetectionEvent',
    'EventType',
    'AIAnalyzer',
    'BaseProcessor',
    'VideoProcessor',
    'DeepStreamProcessor',
    'DEEPSTREAM_AVAILABLE',
    'SensorAlertEvent',
    'SensorEventDetector',
    'SensorRuleConfig',
]
