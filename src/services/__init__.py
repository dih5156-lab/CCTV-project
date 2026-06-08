"""서비스 모듈 - 외부 서비스 연동.

서비스 간 순환 import를 피하기 위해 공개 객체는 지연 로드한다.
"""

from typing import Any

__all__ = [
    'ActionBridge',
    'ExternalIngestService',
    'SensorBridgeService',
    'SensorRuleBridgeService',
]


def __getattr__(name: str) -> Any:
    if name == "ActionBridge":
        from .action_bridge import ActionBridge

        return ActionBridge
    if name == "ExternalIngestService":
        from .external_ingest import ExternalIngestService

        return ExternalIngestService
    if name == "SensorBridgeService":
        from .sensor_bridge import SensorBridgeService

        return SensorBridgeService
    if name == "SensorRuleBridgeService":
        from .sensor_rule_bridge import SensorRuleBridgeService

        return SensorRuleBridgeService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
