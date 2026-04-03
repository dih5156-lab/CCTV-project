"""센서 telemetry를 운영 이벤트로 판정하는 규칙 엔진."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Dict, List, Optional

from ..devices.sensor_device import SensorReading


@dataclass
class SensorRuleConfig:
    """센서 이벤트 판정 임계치."""

    tilt_warning_angle: float = 30.0
    tilt_critical_angle: float = 45.0
    temperature_warning: float = 50.0
    temperature_critical: float = 70.0


@dataclass
class SensorAlertEvent:
    """Action Layer와 EdgeX가 소비할 센서 운영 이벤트."""

    camera_id: str
    event_type: str
    severity: str
    message: str
    timestamp: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "type": self.event_type,
            "severity": self.severity,
            "message": self.message,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class SensorEventDetector:
    """TLV decode 결과를 센서 운영 이벤트로 변환한다."""

    def __init__(self, rules: Optional[SensorRuleConfig] = None) -> None:
        self.rules = rules or SensorRuleConfig()

    def detect_events(self, reading: SensorReading) -> List[SensorAlertEvent]:
        events: List[SensorAlertEvent] = []

        tilt_event = self._detect_tilt_alert(reading)
        if tilt_event:
            events.append(tilt_event)

        temperature_event = self._detect_temperature_alert(reading)
        if temperature_event:
            events.append(temperature_event)

        return events

    def _detect_tilt_alert(self, reading: SensorReading) -> Optional[SensorAlertEvent]:
        angle_x = self._coerce_float(reading.telemetry.get("angle_x"), max_abs=180.0)
        angle_y = self._coerce_float(reading.telemetry.get("angle_y"), max_abs=180.0)
        if angle_x is None and angle_y is None:
            return None

        peak = max(abs(angle_x or 0.0), abs(angle_y or 0.0))
        if peak < self.rules.tilt_warning_angle:
            return None

        severity = "critical" if peak >= self.rules.tilt_critical_angle else "warning"
        return SensorAlertEvent(
            camera_id=reading.device_id,
            event_type="tilt_alert",
            severity=severity,
            message="기울기 이상 감지",
            timestamp=reading.received_at,
            metadata=self._build_metadata(
                reading,
                telemetry={
                    "angle_x": angle_x,
                    "angle_y": angle_y,
                },
            ),
        )

    def _detect_temperature_alert(self, reading: SensorReading) -> Optional[SensorAlertEvent]:
        temperature = self._coerce_float(
            reading.telemetry.get("temperature"),
            min_value=-50.0,
            max_value=150.0,
        )
        if temperature is None or temperature < self.rules.temperature_warning:
            return None

        severity = (
            "critical"
            if temperature >= self.rules.temperature_critical
            else "warning"
        )
        return SensorAlertEvent(
            camera_id=reading.device_id,
            event_type="temperature_alert",
            severity=severity,
            message="온도 이상 감지",
            timestamp=reading.received_at,
            metadata=self._build_metadata(
                reading,
                telemetry={"temperature": temperature},
            ),
        )

    @staticmethod
    def _coerce_float(
        value: Any,
        *,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        max_abs: Optional[float] = None,
    ) -> Optional[float]:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        if not isfinite(number):
            return None
        if min_value is not None and number < min_value:
            return None
        if max_value is not None and number > max_value:
            return None
        if max_abs is not None and abs(number) > max_abs:
            return None
        return number

    @staticmethod
    def _build_metadata(
        reading: SensorReading,
        *,
        telemetry: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "source": reading.source,
            "table": reading.table_name,
            "app_eui": reading.app_eui,
            "dev_eui": reading.dev_eui,
            "telemetry": telemetry,
            **reading.metadata,
        }
