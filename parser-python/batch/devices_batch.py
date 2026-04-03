"""
batch/devices_batch.py
======================
Go 원본: aiot-tlv-parser/pkg/batch/devices_batch.go

외부 API에서 디바이스 목록을 주기적으로 가져오는 배치 작업 모듈입니다.
Go의 goroutine 기반 병렬 API 호출 → Python의 concurrent.futures.ThreadPoolExecutor 로 변환되었습니다.

동작 원리:
  1. SchedulerConfig 의 application_ids 목록을 병렬로 API 호출
  2. 각 응답에서 {DeviceID: Name} 매핑 추출
  3. 모든 결과 통합 후 DeviceUpdateCallback 호출
"""

import logging
import ssl
import threading
import time
import urllib.request
import urllib.error
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# DeviceUpdateCallback 타입 정의
# Go: type DeviceUpdateCallback func(deviceMappings map[string]string) error
DeviceUpdateCallback = Callable[[Dict[str, str]], None]


# ──────────────────────────────────────────────
# API 응답 데이터 구조
# Go: type APIDevice, APIDeviceIDs, APIDeviceResponse 등
# ──────────────────────────────────────────────

@dataclass
class DeviceInfo:
    """
    내부 디바이스 정보
    Go: type DeviceInfo struct { ID, EUI, Name string; LastSeen time.Time }
    """
    id: str = ""
    eui: str = ""
    name: str = ""
    last_seen: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SchedulerConfig:
    """
    디바이스 스케줄러 설정
    Go: type SchedulerConfig struct { APIURL, Token string; Interval time.Duration; MaxRetries int; ... }
    """
    api_url: str = ""
    interval: timedelta = timedelta(hours=1)
    max_retries: int = 3
    enabled: bool = True
    application_ids: List[str] = field(default_factory=list)  # 여러 앱 ID
    token: str = ""
    skip_tls_verify: bool = False


# ──────────────────────────────────────────────
# API 호출 함수들
# ──────────────────────────────────────────────

def get_devices_batch(api_url: str, token: str, skip_tls_verify: bool = False) -> Dict[str, str]:
    """
    HTTP GET으로 디바이스 목록 가져오기
    Go: func GetDevicesBatch(apiURL string, token string, skipTLSVerify bool) (map[string]string, error)

    Args:
        api_url        : 디바이스 목록 API URL
        token          : Bearer 인증 토큰
        skip_tls_verify: TLS 인증서 검증 건너뛰기 여부

    Returns:
        {DeviceID: Name} 딕셔너리
    """
    logger.info(f"API URL: {api_url}")

    req = urllib.request.Request(api_url)
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("Authorization", f"Bearer {token}")

    # TLS 검증 건너뛰기 (Go: &tls.Config{InsecureSkipVerify: true})
    ssl_context = None
    if skip_tls_verify:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(req, timeout=30, context=ssl_context) as resp:
            if resp.status != 200:
                raise RuntimeError(f"API request failed with status: {resp.status}")
            body = resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"API request failed with status: {e.code}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"failed to execute request: {e.reason}")

    # JSON 파싱
    # Go: json.Unmarshal(body, &apiResponse)
    try:
        api_response = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"failed to parse JSON response: {e}")

    end_devices = api_response.get("end_devices", [])
    return _convert_device_info_to_mapping(end_devices)


def get_devices_batch_with_retry(
    api_url: str,
    token: str,
    max_retries: int,
    skip_tls_verify: bool = False,
) -> Dict[str, str]:
    """
    재시도 로직이 포함된 디바이스 배치 조회
    Go: func GetDevicesBatchWithRetry(apiURL string, token string, maxRetries int, skipTLSVerify bool) (map[string]string, error)

    지수 백오프: i+1 초 대기 (Go와 동일)
    """
    last_err = None
    for i in range(max_retries):
        try:
            devices = get_devices_batch(api_url, token, skip_tls_verify)
            return devices
        except Exception as e:
            last_err = e
            logger.warning(f"Attempt {i + 1} failed: {e}")
            if i < max_retries - 1:
                time.sleep(i + 1)  # Go: time.Sleep(time.Duration(i+1) * time.Second)

    raise RuntimeError(f"failed after {max_retries} retries: {last_err}")


def process_devices_batch(device_mapping: Dict[str, str], callback: DeviceUpdateCallback) -> None:
    """
    디바이스 매핑 처리 후 콜백 호출
    Go: func ProcessDevicesBatch(deviceMapping map[string]string, callback DeviceUpdateCallback) error
    """
    if callback is None:
        raise ValueError("callback function is nil")
    callback(device_mapping)


def _convert_device_info_to_mapping(devices: list) -> Dict[str, str]:
    """
    API 응답 디바이스 목록을 {DeviceID: Name} 딕셔너리로 변환
    Go: func convertDeviceInfoToMapping(devices []APIDevice) map[string]string

    Name 이 없으면 DeviceID 를 이름으로 사용합니다.
    """
    mapping: Dict[str, str] = {}
    for device in devices:
        ids = device.get("ids", {})
        device_id = ids.get("device_id", "")
        if device_id:
            name = device.get("name", "") or device_id
            mapping[device_id] = name
    return mapping


# ──────────────────────────────────────────────
# DeviceScheduler
# Go: type DeviceScheduler struct { ... }
# ──────────────────────────────────────────────

class DeviceScheduler:
    """
    디바이스 정보 주기적 갱신 스케줄러
    Go: type DeviceScheduler struct { config SchedulerConfig; ticker *time.Ticker; ... }

    BatchJob 인터페이스 구현:
      - start()      : 스케줄러 시작
      - stop()       : 스케줄러 중지
      - get_name()   : 스케줄러 이름 반환
      - is_running() : 실행 여부 반환
    """

    def __init__(self, config: SchedulerConfig, callback: DeviceUpdateCallback):
        """
        Go: func NewDeviceScheduler(config SchedulerConfig, callback DeviceUpdateCallback) *DeviceScheduler
        """
        self._config = config
        self._callback = callback
        self._running = False
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def get_name(self) -> str:
        """
        Go: func (ds *DeviceScheduler) GetName() string
        """
        return f"Device(id,name) Scheduler-{self._config.application_ids}"

    def is_running(self) -> bool:
        """
        Go: func (ds *DeviceScheduler) IsRunning() bool
        """
        with self._lock:
            return self._running

    def start(self) -> None:
        """
        스케줄러 시작 (즉시 한 번 실행 후 주기 실행)
        Go: func (ds *DeviceScheduler) Start() error
        """
        with self._lock:
            if not self._config.enabled:
                logger.info("Device scheduler is disabled")
                return
            if self._running:
                raise RuntimeError("scheduler is already running")

            logger.info(f"Starting device scheduler with interval: {self._config.interval}")
            self._running = True
            self._stop_event.clear()

        # 즉시 한 번 실행 (Go: go ds.fetchDevices())
        fetch_thread = threading.Thread(target=self._fetch_devices, daemon=True)
        fetch_thread.start()

        # 주기 실행 루프
        def scheduler_loop():
            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=self._config.interval.total_seconds())
                if not self._stop_event.is_set():
                    self._fetch_devices()
            with self._lock:
                self._running = False
            logger.info("Device scheduler stopped")

        self._thread = threading.Thread(
            target=scheduler_loop,
            daemon=True,
            name=f"DeviceScheduler-{self._config.application_ids}",
        )
        self._thread.start()

    def stop(self) -> None:
        """
        스케줄러 중지
        Go: func (ds *DeviceScheduler) Stop()
        """
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        with self._lock:
            self._running = False
        logger.info("Device scheduler stopped")

    def _fetch_devices(self) -> None:
        """
        모든 애플리케이션 ID에 대해 병렬 API 호출 후 결과 통합
        Go: func (ds *DeviceScheduler) fetchDevices()

        Go의 channel 기반 병렬 처리 → ThreadPoolExecutor 로 변환
        """
        logger.info(f"Fetching device information from {len(self._config.application_ids)} applications...")

        all_devices: Dict[str, str] = {}
        success_count = 0

        # Go: Promise.all과 같은 병렬 처리
        def fetch_for_app(app_id: str):
            url = f"{self._config.api_url}/applications/{app_id}/devices?field_mask=name"
            return get_devices_batch_with_retry(
                url,
                self._config.token,
                self._config.max_retries,
                self._config.skip_tls_verify,
            )

        with ThreadPoolExecutor(max_workers=len(self._config.application_ids) or 1) as executor:
            futures = {executor.submit(fetch_for_app, app_id): app_id
                      for app_id in self._config.application_ids}

            for future in as_completed(futures):
                app_id = futures[future]
                try:
                    device_mapping = future.result()
                    all_devices.update(device_mapping)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Error fetching devices for application {app_id}: {e}")

        logger.info(
            f"Successfully fetched devices from {success_count}/{len(self._config.application_ids)} "
            f"applications. Total devices: {len(all_devices)}"
        )

        # 통합 결과로 콜백 호출
        try:
            process_devices_batch(all_devices, self._callback)
        except Exception as e:
            logger.error(f"Failed to process devices: {e}")
            return

        logger.info("Device information processing completed")
