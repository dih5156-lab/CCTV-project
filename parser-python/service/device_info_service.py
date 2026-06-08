"""
service/device_info_service.py
===============================
Go 원본: aiot-tlv-parser/pkg/service/device_info_service.go

디바이스 정보(DevEUI → DeviceID 매핑)를 원자적으로 관리하는 서비스입니다.
Go의 sync/atomic.Pointer → Python의 threading.Lock + 참조 교체 방식으로 변환되었습니다.

동작 방식:
  - batch 작업이 API에서 가져온 디바이스 목록으로 내부 dict 를 교체
  - MQTT 메시지 처리 시 DevEUI로 DeviceID 조회
"""

import logging
import threading
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class DeviceInfoService:
    """
    디바이스 정보 원자적 관리 서비스
    Go: type DeviceInfoService struct { devices atomic.Pointer[map[string]string] }

    Go의 atomic.Pointer 는 참조 자체를 원자적으로 교체합니다.
    Python에서는 threading.Lock으로 dict 참조 교체를 보호합니다.
    """

    def __init__(self):
        """
        Go: func NewDeviceInfoService() *DeviceInfoService
        빈 딕셔너리로 초기화합니다.
        """
        # Go: emptyDevices := make(map[string]string); service.devices.Store(&emptyDevices)
        self._devices: Dict[str, str] = {}
        self._lock = threading.RLock()  # Go의 atomic.Pointer 보호

    def update_devices_from_batch(self, device_mappings: Dict[str, str]) -> None:
        """
        배치 API 결과로 디바이스 매핑 갱신
        Go: func (d *DeviceInfoService) UpdateDevicesFromBatch(deviceMappings map[string]string) error

        빈 EUI 또는 이름은 무시합니다.

        Args:
            device_mappings: {devEUI: deviceID} 딕셔너리
        """
        logger.info(f"Updating device mappings from batch data ({len(device_mappings)} devices)")

        # 새 딕셔너리 생성 후 참조 교체 (Go: atomic.Pointer.Store)
        new_devices = {
            eui: name
            for eui, name in device_mappings.items()
            if eui and name
        }

        with self._lock:
            self._devices = new_devices

    def get_device_id(self, dev_eui: str) -> str:
        """
        DevEUI로 DeviceID 조회
        Go: func (d *DeviceInfoService) GetDeviceID(devEUI string) string

        Args:
            dev_eui: 조회할 디바이스 EUI

        Returns:
            DeviceID 문자열 (없으면 빈 문자열)
        """
        with self._lock:
            devices = self._devices

        if not devices:
            logger.debug(f"GetDeviceID: devices is nil for EUI {dev_eui}")
            return ""

        return devices.get(dev_eui, "")

    def get_device_id_safe(self, dev_eui: str) -> Tuple[str, bool]:
        """
        DevEUI로 DeviceID 조회 (존재 여부 함께 반환)
        Go: func (d *DeviceInfoService) GetDeviceIDSafe(devEUI string) (string, bool)

        Returns:
            (device_id, exists) 튜플
        """
        with self._lock:
            devices = self._devices

        if not devices:
            return "", False

        device_id = devices.get(dev_eui)
        if device_id is None:
            return "", False
        return device_id, True

    def list_device_euis(self) -> List[str]:
        """
        등록된 모든 DevEUI 목록 반환
        Go: func (d *DeviceInfoService) ListDeviceEUIs() []string
        """
        with self._lock:
            return list(self._devices.keys())

    def list_allowed_devices(self) -> List[str]:
        """
        허용된 모든 DeviceID 목록 반환
        Go: func (d *DeviceInfoService) ListAllowedDevices() []string
        """
        with self._lock:
            return list(self._devices.values())

    def get_device_count(self) -> int:
        """
        등록된 디바이스 수 반환
        Go: func (d *DeviceInfoService) GetDeviceCount() int
        """
        with self._lock:
            return len(self._devices)
