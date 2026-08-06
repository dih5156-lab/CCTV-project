"""센서 telemetry를 운영 이벤트로 판정하는 규칙 엔진."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from typing import Any, Callable, Dict, List, Optional

from ..canonical_event import build_canonical_event
from ..devices.sensor_device import SensorReading


@dataclass
class SensorRuleConfig:
    """센서 이벤트 판정 임계치."""

    tilt_warning_angle: float = 30.0
    tilt_critical_angle: float = 45.0
    temperature_warning: float = 50.0
    temperature_critical: float = 70.0
    # T34958 가속도 크기가 정지 상태의 중력(1g)에서 벗어나는 정도.
    # 전용 진동 센서의 주파수 분석값이 아니라 IMU 기반 충격 감지 임계치다.
    vibration_warning_delta_g: float = 0.5
    vibration_critical_delta_g: float = 1.0
    vibration_reference_g: float = 1.0


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
        payload = {
            "camera_id": self.camera_id,
            "type": self.event_type,
            "severity": self.severity,
            "message": self.message,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
        payload.update(
            build_canonical_event(
                camera_id=self.camera_id,
                event_type=self.event_type,
                message_type="sensor_event",
                occurred_at=self.timestamp,
                message_id=self.metadata.get("message_id"),
                source=str(self.metadata.get("source", "lora_tlv")),
                source_type="sensor",
                severity=self.severity,
                confidence=self.confidence,
                message=self.message,
                display_message=self.message,
                tts_message=self._default_tts_message(),
                device={
                    "camera_id": self.camera_id,
                    "device_id": self.camera_id,
                    "app_eui": self.metadata.get("app_eui"),
                    "dev_eui": self.metadata.get("dev_eui"),
                    "f_port": self.metadata.get("f_port"),
                    "f_cnt_up": self.metadata.get("f_cnt_up"),
                },
                gateway={
                    "channel": self.metadata.get("channel"),
                    "frequency": self.metadata.get("frequency"),
                    "rssi": self.metadata.get("rssi"),
                    "snr": self.metadata.get("snr"),
                },
                decoded=self.metadata.get("telemetry") or {},
                raw={"metadata": self.metadata},
            )
        )
        return payload

    def _default_tts_message(self) -> str:
        if self.severity == "critical":
            return f"{self.message}. 즉시 현장을 확인 바랍니다."
        if self.severity == "warning":
            return f"{self.message}. 현장을 확인 바랍니다."
        return self.message


class SensorEventDetector:
    """TLV decode 결과를 센서 운영 이벤트로 변환한다."""

    def __init__(self, rules: Optional[SensorRuleConfig] = None) -> None:
        self.rules = rules or SensorRuleConfig()
        self._detectors: List[Callable[[SensorReading], Optional[SensorAlertEvent]]] = [
            self._detect_tilt_alert,
            self._detect_temperature_alert,
            self._detect_vibration_alert,
        ]

    def register_detector(
        self,
        detector: Callable[[SensorReading], Optional[SensorAlertEvent]],
    ) -> None:
        """외부 센서 규칙 detector를 실행 순서의 끝에 추가한다."""
        self._detectors.append(detector)

    def detect_events(self, reading: SensorReading) -> List[SensorAlertEvent]:
        events: List[SensorAlertEvent] = []
        for detector in self._detectors:
            event = detector(reading)
            if event:
                events.append(event)
        return events

    def _detect_tilt_alert(self, reading: SensorReading) -> Optional[SensorAlertEvent]:
        angle_x = self._coerce_float(reading.telemetry.get("angle_x"), max_abs=180.0)
        angle_y = self._coerce_float(reading.telemetry.get("angle_y"), max_abs=180.0)
        if angle_x is None and angle_y is None:
            return None

        peak = max(abs(angle_x or 0.0), abs(angle_y or 0.0))
        severity = self._threshold_severity(
            peak,
            warning_threshold=self.rules.tilt_warning_angle,
            critical_threshold=self.rules.tilt_critical_angle,
        )
        if severity is None:
            return None

        return self._build_alert_event(
            reading,
            event_type="tilt_alert",
            severity=severity,
            message="기울기 이상 감지",
            telemetry={
                "angle_x_deg": angle_x,
                "angle_y_deg": angle_y,
            },
        )

    def _detect_temperature_alert(self, reading: SensorReading) -> Optional[SensorAlertEvent]:
        temperature = self._coerce_float(
            reading.telemetry.get("temperature"),
            min_value=-50.0,
            max_value=150.0,
        )
        if temperature is None:
            return None

        severity = self._threshold_severity(
            temperature,
            warning_threshold=self.rules.temperature_warning,
            critical_threshold=self.rules.temperature_critical,
        )
        if severity is None:
            return None

        return self._build_alert_event(
            reading,
            event_type="temperature_alert",
            severity=severity,
            message="온도 이상 감지",
            telemetry={"temperature_c": temperature},
        )

    def _detect_vibration_alert(self, reading: SensorReading) -> Optional[SensorAlertEvent]:
        """IMU 3축 가속도 크기의 1g 대비 편차를 충격/진동으로 판정한다.

        샘플링 주파수와 전용 진동 센서값이 없는 현재 T34958 payload에서는
        주파수 영역 진동 분석을 할 수 없으므로, 순간 가속도 편차만 다룬다.
        3축이 모두 유효하지 않으면 오탐 방지를 위해 판정하지 않는다.
        """
        axis_values = []
        for axis in ("x", "y", "z"):
            value = reading.telemetry.get(f"acc_{axis}_g")
            if value is None:
                # 일부 레거시 입력은 단위를 생략한 acc_x 형식을 사용한다.
                value = reading.telemetry.get(f"acc_{axis}")
            parsed = self._coerce_float(value, max_abs=32.0)
            if parsed is None:
                return None
            axis_values.append(parsed)

        magnitude_g = sqrt(sum(value * value for value in axis_values))
        delta_g = abs(magnitude_g - self.rules.vibration_reference_g)
        severity = self._threshold_severity(
            delta_g,
            warning_threshold=self.rules.vibration_warning_delta_g,
            critical_threshold=self.rules.vibration_critical_delta_g,
        )
        if severity is None:
            return None

        telemetry = {
            "acc_x_g": axis_values[0],
            "acc_y_g": axis_values[1],
            "acc_z_g": axis_values[2],
            "acceleration_magnitude_g": round(magnitude_g, 4),
            "vibration_delta_g": round(delta_g, 4),
            "detection_method": "imu_acceleration_delta_from_1g",
        }
        return self._build_alert_event(
            reading,
            event_type="vibration_alert",
            severity=severity,
            message="진동 충격 감지",
            telemetry=telemetry,
        )

    def _build_alert_event(
        self,
        reading: SensorReading,
        *,
        event_type: str,
        severity: str,
        message: str,
        telemetry: Dict[str, Any],
    ) -> SensorAlertEvent:
        """센서 rule 결과를 공통 AlertEvent 형식으로 만든다."""
        return SensorAlertEvent(
            camera_id=reading.device_id,
            event_type=event_type,
            severity=severity,
            message=message,
            timestamp=reading.received_at,
            metadata=self._build_metadata(reading, telemetry=telemetry),
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
    def _threshold_severity(
        value: float,
        *,
        warning_threshold: float,
        critical_threshold: float,
    ) -> Optional[str]:
        if value < warning_threshold:
            return None
        return "critical" if value >= critical_threshold else "warning"

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
