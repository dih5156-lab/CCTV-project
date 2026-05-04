"""핵심 모듈 - 분석 및 처리 로직.

API 스키마나 가벼운 유틸리티가 `src.core.events` 만 필요로 할 때도
이 패키지의 `__init__` 이 먼저 실행된다. 여기서 AI/DeepStream 모듈을
eager import 하면 테스트와 API import 시점에 불필요하게 무거운 초기화가
발생하므로, 필요한 심볼만 lazy import 하도록 유지한다.
"""

from .events import DetectionEvent, EventType
from .sensor_detection import SensorAlertEvent, SensorEventDetector, SensorRuleConfig

__all__ = [
    "DetectionEvent",
    "EventType",
    "AIAnalyzer",
    "BaseProcessor",
    "VideoProcessor",
    "DeepStreamProcessor",
    "DEEPSTREAM_AVAILABLE",
    "SensorAlertEvent",
    "SensorEventDetector",
    "SensorRuleConfig",
]


def __getattr__(name: str):
    if name in {"AIAnalyzer", "BaseProcessor", "VideoProcessor"}:
        from .ai.analyzer import AIAnalyzer
        from .base_processor import BaseProcessor
        from .processor import VideoProcessor

        mapping = {
            "AIAnalyzer": AIAnalyzer,
            "BaseProcessor": BaseProcessor,
            "VideoProcessor": VideoProcessor,
        }
        return mapping[name]

    if name in {"DeepStreamProcessor", "DEEPSTREAM_AVAILABLE"}:
        try:
            from .deepstream_processor import DEEPSTREAM_AVAILABLE, DeepStreamProcessor
        except ImportError:
            if name == "DEEPSTREAM_AVAILABLE":
                return False
            return None

        return {
            "DeepStreamProcessor": DeepStreamProcessor,
            "DEEPSTREAM_AVAILABLE": DEEPSTREAM_AVAILABLE,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
