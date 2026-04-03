"""
mqtt/interfaces.py
==================
Go 원본: aiot-tlv-parser/pkg/mqtt/interfaces.go

MQTT 처리기 인터페이스 정의 모듈입니다.
Go의 interface → Python의 ABC(Abstract Base Class) + Protocol 로 변환되었습니다.
"""

from abc import ABC, abstractmethod


class SensorDataProcessor(ABC):
    """
    센서 데이터 처리기 인터페이스
    Go: type SensorDataProcessor interface { ProcessSensorData(...) error }

    MQTT로 수신된 센서 페이로드를 처리하는 컴포넌트가 구현해야 합니다.
    실제 구현체: service.SensorService
    """

    @abstractmethod
    def process_sensor_data(
        self,
        app_id: str,
        dev_eui: str,
        payload: str,
        channel: int,
        frequency: int,
        received_at: int,
    ) -> None:
        """
        MQTT 수신 센서 데이터 처리
        Go: ProcessSensorData(appID, devEUI, payload string, channel, frequency int, receivedAt int64) error

        Args:
            app_id      : 애플리케이션 EUI/ID
            dev_eui     : 디바이스 EUI
            payload     : Base64 인코딩된 TLV 페이로드
            channel     : LoRa 채널 번호
            frequency   : LoRa 주파수 (Hz)
            received_at : 게이트웨이 수신 Unix 밀리초 타임스탬프
        """
        ...


class EventDataProcessor(ABC):
    """
    이벤트 데이터 처리기 인터페이스
    Go: type EventDataProcessor interface { ProcessEventData(...) error; StartApplicationIdsUpdate() }
    """

    @abstractmethod
    def process_event_data(self, data: object) -> None:
        """
        이벤트 데이터 처리
        Go: ProcessEventData(data interface{}) error
        """
        ...

    @abstractmethod
    def start_application_ids_update(self) -> None:
        """
        애플리케이션 ID 목록 주기적 업데이트 시작
        Go: StartApplicationIdsUpdate()
        """
        ...
