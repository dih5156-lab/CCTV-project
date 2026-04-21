# CCTV 프로젝트 코드 리뷰 보고서

> 작성일: 2026-03-26  
> 리뷰 범위: 전체 소스코드 (src/, main.py, Dockerfile, docker-compose.yml, requirements.txt)

---

## 목차

1. [전체 평가](#1-전체-평가)
2. [긍정적인 점](#2-긍정적인-점)
3. [문제점 및 수정사항](#3-문제점-및-수정사항)
4. [표준화 제안](#4-표준화-제안)
5. [파일별 세부 리뷰](#5-파일별-세부-리뷰)
6. [우선순위별 수정 목록](#6-우선순위별-수정-목록)

---

## 1. 전체 평가

| 항목 | 점수 | 비고 |
|------|------|------|
| 코드 구조 | ⭐⭐⭐⭐⭐ | 레이어 분리, 책임 분리 매우 우수 |
| 가독성 | ⭐⭐⭐⭐ | 주석·docstring 충실, 일부 불일치 |
| 안정성 | ⭐⭐⭐⭐ | 예외 처리 전반적으로 양호 |
| 테스트 | ⭐⭐⭐ | 테스트 파일 존재하나 커버리지 불명확 |
| 표준화 | ⭐⭐⭐ | 일부 파일 인코딩 문제, 스타일 불일치 |
| 배포 | ⭐⭐⭐⭐ | Docker 멀티스테이지 빌드 적용 |

---

## 2. 긍정적인 점

### ✅ 아키텍처 설계
- **레이어 분리 우수**: `src/core/`, `src/services/`, `src/protocols/`, `src/devices/`, `src/edgex/`, `src/utils/` 로 명확하게 분리
- **단일 책임 원칙(SRP)** 잘 준수: `_EventDebouncer`, `_DisplayGrid`, `_CameraRegistry` 등 내부 헬퍼 클래스로 분리
- **의존성 역전**: `AppConfig`를 통한 중앙화된 설정 관리, 환경변수 오버라이드 패턴 우수

### ✅ 안정성
- **재연결 로직**: RTSP 카메라 지수 백오프 재연결, MQTT 재연결 백오프 모두 구현
- **스레드 안전성**: `Lock` 사용 일관성 있음
- **예외 처리**: 대부분의 외부 I/O에 try/except 적용
- **낙상 감지 알고리즘**: 4가지 방법 조합으로 오탐 최소화 설계 우수

### ✅ 운영 편의성
- **로그 로테이션**: `RotatingFileHandler` 적용 (10MB × 5개)
- **통계 추적**: `ProcessorStats` DTO로 성능 모니터링
- **Zone API**: REST API로 런타임 구역 설정 변경 가능
- **데이터셋 수집**: 자동 수집 기능 내장

### ✅ EdgeX 통합
- **버전 폴백**: v3 → v2 → v1 순서로 API 버전 자동 폴백
- **Redis/MQTT 이중화**: 메시지 버스 이중화 구현

---

## 3. 문제점 및 수정사항

### 🔴 심각 (즉시 수정 필요)

#### 3.1 `camera_input.py` 파일 인코딩 깨짐
**파일**: `src/utils/camera_input.py`  
**문제**: 파일 내 한글 주석/문자열이 모두 `?`로 깨져 있음 (인코딩 손상)
```python
# 현재 (깨진 상태)
"""???/??? ?? ??.
RTSP ?? ???? ???? `RTSPCamera`?
```
**수정**: 파일을 UTF-8로 재저장하거나 한글 주석을 영문으로 교체

#### 3.2 `visualizer.py` - EventType 변환 오류 가능성
**파일**: `src/utils/visualizer.py`  
**문제**: `EventType(event_type_str.upper())` 호출 시 `EventType.OTHER` 등 소문자 값을 대문자로 변환하면 `ValueError` 발생
```python
# 현재 (버그)
"color": EVENT_COLORS.get(EventType(event_type_str.upper()), EVENT_COLORS[EventType.OTHER]),

# 수정
try:
    event_type_enum = EventType(event_type_str.lower())
except ValueError:
    event_type_enum = EventType.OTHER
"color": EVENT_COLORS.get(event_type_enum, EVENT_COLORS[EventType.OTHER]),
```

#### 3.3 `action_bridge.py` - `_forwarder._targets` 내부 속성 직접 접근
**파일**: `src/services/action_bridge.py`  
**문제**: `http_sent = bool(self._forwarder._targets)` — 내부 속성(`_targets`) 직접 접근은 캡슐화 위반
```python
# 현재 (캡슐화 위반)
http_sent = bool(self._forwarder._targets)

# 수정: HttpEventForwarder에 공개 프로퍼티 추가
@property
def has_targets(self) -> bool:
    return bool(self._targets)

# action_bridge.py에서
http_sent = self._forwarder.has_targets
```

#### 3.4 `device_service.py` - 클래스 변수로 공유되는 연결 상태
**파일**: `src/edgex/device_service.py`  
**문제**: `_redis_last_fail_time`, `_mqtt_fail_count` 등이 클래스 변수로 선언되어 **모든 인스턴스가 공유**됨. 다중 인스턴스 사용 시 의도치 않은 동작 발생
```python
# 현재 (클래스 변수 - 모든 인스턴스 공유)
class CCTVDeviceService:
    _redis_last_fail_time: float = 0
    _mqtt_fail_count: int = 0

# 수정: 인스턴스 변수로 변경
def __init__(self, config: Dict):
    self._redis_last_fail_time: float = 0
    self._mqtt_fail_count: int = 0
    self._redis_fail_count: int = 0
    self._mqtt_last_fail_time: float = 0
```

---

### 🟡 중요 (가능한 빨리 수정)

#### 3.5 `processor.py` - `stop()` 메서드에서 내부 속성 직접 접근
**파일**: `src/core/processor.py`  
**문제**: `for flag in self._cams._stop_flags.values()` — `_CameraRegistry`의 내부 속성 직접 접근
```python
# 현재
for flag in self._cams._stop_flags.values():
    flag.set()

# 수정: _CameraRegistry에 메서드 추가
def set_all_stop_flags(self) -> None:
    for flag in self._stop_flags.values():
        flag.set()

# processor.py에서
self._cams.set_all_stop_flags()
```

#### 3.6 `main.py` - `_release_all()` 내부 구조 직접 접근
**파일**: `main.py`  
**문제**: `_proc_ref[0]._cams.cameras.values()` — 프로세서 내부 구조에 직접 접근
```python
# 현재
for cam in _proc_ref[0]._cams.cameras.values():
    cam.release()

# 수정: VideoProcessor에 release_all_cameras() 공개 메서드 추가
def release_all_cameras(self) -> None:
    for cam in self._cams.cameras.values():
        try:
            cam.release()
        except Exception:
            pass
```

#### 3.7 `config.py` - `AppConfig` 필드 타입 힌트 불일치
**파일**: `src/config/config.py`  
**문제**: `Optional[ModelPaths] = None`으로 선언되어 있지만 `__post_init__`에서 항상 초기화되므로 실제로는 `None`이 될 수 없음. 타입 힌트가 오해를 유발
```python
# 현재 (혼란스러운 타입 힌트)
@dataclass
class AppConfig:
    models: Optional[ModelPaths] = None  # 실제로는 항상 ModelPaths

# 수정: field(default_factory=...) 패턴 사용
from dataclasses import field

@dataclass
class AppConfig:
    models: ModelPaths = field(default_factory=ModelPaths)
    mqtt: MqttConfig = field(default_factory=MqttConfig)
    # ...
```

#### 3.8 AI 분석 모듈 - `_generate_temp_id` 충돌 가능성
**파일**: `src/core/ai/analyzer.py`  
**문제**: blake2b 4바이트 해시 기반 임시 ID는 500,000,000 범위 내에서 충돌 가능성 존재. 동일 프레임에서 여러 객체가 같은 임시 ID를 가질 수 있음
```python
# 현재: 50픽셀 그리드 기반 → 같은 그리드 셀의 다른 객체 충돌 가능
center_x = (x1 + width // 2) // 50

# 개선: 더 세밀한 그리드 또는 카운터 기반 ID 사용
```

#### 3.9 `event_filters.py` - `CumulativeViolationFilter.filter()` 락 범위 문제
**파일**: `src/core/event_filters.py`  
**문제**: `violation_summary` 딕셔너리를 락 내부에서 채우고 락 외부에서 읽는 패턴은 안전하지만, 락 해제 후 `events` 순회 중 다른 스레드가 `_history`를 수정할 수 있음
```python
# 현재: 락 해제 후 violation_summary 사용 (안전하지만 주석 필요)
with self._lock:
    # ... violation_summary 채우기
# 락 해제 후 violation_summary 읽기 (로컬 변수라 안전)
for event in events:
    violation_count, history_size = violation_summary.get(key, (0, 0))
```
→ 현재 구현은 `violation_summary`가 로컬 변수이므로 실제로는 안전하나, 코드 의도를 명확히 하는 주석 추가 권장

#### 3.10 `device_service.py` - `asyncio.to_thread` Python 3.9+ 전용
**파일**: `src/edgex/device_service.py`  
**문제**: `asyncio.to_thread()`는 Python 3.9+에서만 사용 가능. `Dockerfile`은 Python 3.10 사용이지만 명시적 버전 체크 없음
```python
# 현재
result = await asyncio.to_thread(requests.get, endpoint, timeout=timeout)

# 수정: requirements.txt 또는 Dockerfile에 Python 버전 명시
# Dockerfile에 추가:
# RUN python --version | grep -E "3\.(9|10|11|12)" || exit 1
```

---

### 🟢 개선 권장 (품질 향상)

#### 3.11 `mqtt.py` - `paho-mqtt` v2 API 호환성
**파일**: `src/protocols/mqtt.py`  
**문제**: `paho-mqtt>=1.6.1,<3.0.0`을 지원하지만 v2에서 `mqtt.Client()` 생성자 시그니처가 변경됨 (`callback_api_version` 필수)
```python
# 현재 (paho-mqtt v2에서 DeprecationWarning)
self._client = mqtt.Client(
    client_id=f"{self.client_id_prefix}-{uuid.uuid4().hex[:8]}",
    clean_session=True,
)

# 수정: 버전 호환 처리
try:
    from paho.mqtt.client import CallbackAPIVersion
    self._client = mqtt.Client(
        callback_api_version=CallbackAPIVersion.VERSION1,
        client_id=f"{self.client_id_prefix}-{uuid.uuid4().hex[:8]}",
        clean_session=True,
    )
except ImportError:
    # paho-mqtt v1
    self._client = mqtt.Client(
        client_id=f"{self.client_id_prefix}-{uuid.uuid4().hex[:8]}",
        clean_session=True,
    )
```

#### 3.12 `requirements.txt` - 불필요한 패키지 포함
**파일**: `requirements.txt`  
**문제**: `pandas`, `colorlog`, `tqdm`, `psutil`은 핵심 기능에 불필요. 개발 도구(`pytest`, `black`, `flake8`, `mypy`)가 프로덕션 requirements에 포함됨
```
# 수정: requirements.txt와 requirements-dev.txt 분리
# requirements.txt (프로덕션)
ultralytics>=8.0.0,<9.0.0
torch>=2.0.0,<3.0.0
# ...

# requirements-dev.txt (개발)
-r requirements.txt
pytest>=7.4.0,<9.0.0
black>=23.0.0,<25.0.0
flake8>=6.0.0,<8.0.0
mypy>=1.5.0,<2.0.0
```

#### 3.13 `Dockerfile` - 보안 설정 불완전
**파일**: `Dockerfile`  
**문제**: `cctv` 사용자를 생성했지만 Python 패키지가 `/root/.local`에 설치되어 `cctv` 사용자가 실행 시 접근 불가 가능성
```dockerfile
# 현재 (불일치)
RUN groupadd -r cctv && useradd -r -g cctv cctv
COPY --from=builder /root/.local /root/.local  # root 소유
# USER cctv 없음 → root로 실행됨

# 수정
COPY --from=builder /root/.local /home/cctv/.local
RUN chown -R cctv:cctv /home/cctv/.local
ENV PATH=/home/cctv/.local/bin:$PATH
USER cctv
```

#### 3.14 `zone_api.py` - CORS 헤더 없음
**파일**: `src/services/zone_api.py`  
**문제**: 브라우저 기반 프론트엔드에서 API 호출 시 CORS 오류 발생 가능
```python
# 수정: _respond 메서드에 CORS 헤더 추가
def _respond(self, code: int, body) -> None:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    self.send_response(code)
    self.send_header("Content-Type", "application/json; charset=utf-8")
    self.send_header("Content-Length", str(len(data)))
    self.send_header("Access-Control-Allow-Origin", "*")  # 추가
    self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")  # 추가
    self.end_headers()
    self.wfile.write(data)

# OPTIONS 메서드 핸들러 추가
def do_OPTIONS(self):
    self.send_response(204)
    self.send_header("Access-Control-Allow-Origin", "*")
    self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
    self.send_header("Access-Control-Allow-Headers", "Content-Type")
    self.end_headers()
```

#### 3.15 `geometry.py` - 순환 참조 위험
**파일**: `src/utils/geometry.py`  
**문제**: `geometry.py`가 `src/core/events.py`를 import하고, `events.py`는 독립적이지만 향후 `core` 모듈이 `geometry`를 import하면 순환 참조 발생 가능
```python
# 현재
from ..core.events import DetectionEvent

# 개선: Protocol 또는 TypeVar 사용으로 결합도 낮추기
from typing import Protocol

class BBoxProtocol(Protocol):
    x: int
    y: int
    width: int
    height: int

def calculate_iou(box1: BBoxProtocol, box2: BBoxProtocol) -> float:
    ...
```

---

## 4. 표준화 제안

### 4.1 타입 힌트 표준화
현재 일부 파일에서 `Optional[str]`, `Union[str, int]` 등 구형 스타일 사용. Python 3.10+ 기준으로 통일 권장:
```python
# 구형 스타일 (현재 혼용)
from typing import Optional, Union, List, Dict, Tuple
def func(x: Optional[str]) -> Union[str, int]: ...

# 신형 스타일 (Python 3.10+)
def func(x: str | None) -> str | int: ...
# List, Dict, Tuple → list, dict, tuple 소문자 사용
```

### 4.2 로깅 표준화
일부 파일에서 f-string 로깅 사용 (성능 저하 가능):
```python
# 비권장 (f-string - 로그 레벨 무관하게 문자열 생성)
logger.info(f"카메라 등록: {camera_id}")

# 권장 (lazy formatting - 로그 레벨 통과 시에만 문자열 생성)
logger.info("카메라 등록: %s", camera_id)
```
`device_service.py`에서 f-string 로깅이 다수 발견됨.

### 4.3 상수 관리 표준화
현재 상수가 여러 파일에 분산:
- `src/core/ai/_constants.py`: `MAX_HELMET_WIDTH`, `FALL_ANGLE_HORIZONTAL` 등
- `processor.py`: `CONNECT_TIMEOUT = 30` (함수 내 지역 상수)
- `visualizer.py`: `LABEL_FONT`, `BBOX_THICKNESS` 등

**제안**: `src/config/constants.py` 파일 생성하여 전역 상수 중앙화

```python
# src/config/constants.py
"""프로젝트 전역 상수"""

# 헬멧 감지
MAX_HELMET_WIDTH = 300
MAX_HELMET_HEIGHT = 300
MIN_HELMET_SIZE = 15

# 낙상 감지
FALL_ANGLE_HORIZONTAL = 40
FALL_ANGLE_INVERTED = 140

# 시각화
LABEL_FONT_SCALE = 0.5
BBOX_THICKNESS = 2
```

### 4.4 에러 코드 표준화
현재 에러 메시지가 한국어/영어 혼용:
```python
# 혼용 예시
logger.error("카메라 연결 실패: %s", camera_id)  # 한국어
logger.error("POST 실패 (%s): %s", endpoint, error)  # 영어
```
**제안**: 로그 메시지 언어를 한국어로 통일 (또는 영어로 통일)

### 4.5 `__all__` 표준화
일부 파일만 `__all__` 정의:
- `zone_api.py`: `__all__ = ["ZoneApiHandler", "start_zone_api_server"]` ✅
- `camera_input.py`: `__all__ = ["RTSPCamera"]` ✅
- `src/core/ai/analyzer.py`: 공개 API 성격이 강하므로 `__all__` 또는 패키지 레벨 재노출 기준 명확화 필요
- `processor.py`: `__all__` 없음 ❌

**제안**: 공개 API가 있는 모든 모듈에 `__all__` 정의

### 4.6 docstring 스타일 표준화
현재 Google Style, NumPy Style, 자체 스타일 혼용:
```python
# 현재 혼용
def func(x: str) -> bool:
    """설명
    
    매개변수:
        x: 입력값
    반환값:
        성공 여부
    """

# 표준화 제안: Google Style 통일
def func(x: str) -> bool:
    """설명.

    Args:
        x: 입력값.

    Returns:
        성공 여부.
    """
```

---

## 5. 파일별 세부 리뷰

### `main.py` ⭐⭐⭐⭐⭐
- 전반적으로 매우 잘 작성됨
- `_release_all()` 내부에서 `_proc_ref[0]._cams.cameras` 직접 접근 → 캡슐화 위반 (3.6 참조)
- `CONNECT_TIMEOUT = 30` 함수 내 지역 상수 → 모듈 레벨 상수로 이동 권장

### `src/config/config.py` ⭐⭐⭐⭐
- `ENV_OVERRIDES` 패턴 우수
- `AppConfig` 필드 타입 힌트 `Optional` 불필요 (3.7 참조)
- `default_config = AppConfig()` 모듈 레벨 인스턴스 → 사이드 이펙트 주의

### `src/core/ai/analyzer.py` ⭐⭐⭐⭐⭐
- 낙상 감지 4가지 방법 조합 설계 우수
- 포즈 모델 우선 → person 모델 fallback 구조 명확
- 임시 ID 충돌 가능성 (3.8 참조)
- `_CLASS_MAP` 딕셔너리 중복 키 없음 확인 필요

### `src/core/processor.py` ⭐⭐⭐⭐
- 내부 헬퍼 클래스 분리 우수
- `_cams._stop_flags` 직접 접근 (3.5 참조)
- `_process_inference`의 지수 백오프 로직 우수

### `src/core/events.py` ⭐⭐⭐⭐⭐
- 간결하고 명확한 데이터 모델
- `severity` 필드가 `to_dict()`에서만 생성됨 → `DetectionEvent` 필드로 승격 고려

### `src/core/event_filters.py` ⭐⭐⭐⭐
- `TrackManager`, `CumulativeViolationFilter` 분리 우수
- 락 범위 주석 보강 필요 (3.9 참조)

### `src/utils/camera_input.py` ⭐⭐ (인코딩 손상)
- **긴급**: 파일 인코딩 복구 필요 (3.1 참조)
- 로직 자체는 재연결 백오프, 프레임 재시도 등 잘 구현됨

### `src/utils/geometry.py` ⭐⭐⭐⭐
- 함수 분리 명확
- `core.events` 의존성 → 순환 참조 위험 (3.15 참조)

### `src/utils/visualizer.py` ⭐⭐⭐
- `EventType(event_type_str.upper())` 버그 (3.2 참조)
- 그리기 우선순위 `_DRAW_PRIORITY` 패턴 우수

### `src/protocols/mqtt.py` ⭐⭐⭐⭐
- 재연결 백오프 구현 우수
- paho-mqtt v2 호환성 문제 (3.11 참조)

### `src/protocols/http.py` ⭐⭐⭐⭐
- 재시도 큐 + 지수 백오프 패턴 우수
- `_send()` 메서드가 동기 호출이지만 `forward()`에서 블로킹 → 비동기 처리 고려

### `src/services/action_bridge.py` ⭐⭐⭐⭐
- 내부 헬퍼 클래스 분리 우수
- `_forwarder._targets` 직접 접근 (3.3 참조)
- SQLite WAL 모드 적용 우수

### `src/services/zone_api.py` ⭐⭐⭐⭐
- 라우트 패턴 컴파일 최적화 우수
- CORS 헤더 없음 (3.14 참조)
- `OPTIONS` 메서드 미처리

### `src/edgex/device_service.py` ⭐⭐⭐
- 클래스 변수 공유 문제 (3.4 참조)
- f-string 로깅 다수 (4.2 참조)
- 파일이 매우 길어 분리 고려 (Redis 클라이언트, MQTT 클라이언트, HTTP 클라이언트 분리)

### `src/devices/speaker.py` ⭐⭐⭐⭐⭐
- 인수인계 코드 통합 매우 우수
- `_IntermClient` 분리로 테스트 용이성 확보
- `SpeakerNetworkError` 커스텀 예외 적절

### `Dockerfile` ⭐⭐⭐
- 멀티스테이지 빌드 적용 우수
- `cctv` 사용자 생성 후 `USER cctv` 미적용 (3.13 참조)
- `pydantic` 별도 설치 → `requirements.txt`에 포함 권장

### `docker-compose.yml` ⭐⭐⭐⭐
- 서비스 의존성 명확
- `cctv-action-layer`에서 `./src` 볼륨 마운트 → 프로덕션에서 코드 변경 가능 (보안 위험)
- 헬스체크 일부 서비스에만 적용

---

## 6. 우선순위별 수정 목록

### 🔴 즉시 수정 (P0)

| # | 파일 | 문제 | 수정 방법 |
|---|------|------|-----------|
| 1 | `src/utils/camera_input.py` | 파일 인코딩 손상 | UTF-8로 재저장 |
| 2 | `src/utils/visualizer.py` | `EventType.upper()` 버그 | `.lower()` 사용 |
| 3 | `src/services/action_bridge.py` | `_targets` 내부 속성 접근 | `has_targets` 프로퍼티 추가 |
| 4 | `src/edgex/device_service.py` | 클래스 변수 공유 | 인스턴스 변수로 변경 |

### 🟡 단기 수정 (P1, 1주 이내)

| # | 파일 | 문제 | 수정 방법 |
|---|------|------|-----------|
| 5 | `src/core/processor.py` | `_stop_flags` 직접 접근 | `set_all_stop_flags()` 메서드 추가 |
| 6 | `main.py` | `_cams.cameras` 직접 접근 | `release_all_cameras()` 메서드 추가 |
| 7 | `src/config/config.py` | `Optional` 타입 힌트 불필요 | `field(default_factory=...)` 사용 |
| 8 | `src/protocols/mqtt.py` | paho-mqtt v2 호환성 | 버전 분기 처리 |
| 9 | `Dockerfile` | `USER cctv` 미적용 | 사용자 전환 추가 |
| 10 | `requirements.txt` | 개발 도구 혼재 | `requirements-dev.txt` 분리 |

### 🟢 중기 개선 (P2, 1개월 이내)

| # | 파일 | 문제 | 수정 방법 |
|---|------|------|-----------|
| 11 | `src/services/zone_api.py` | CORS 미지원 | CORS 헤더 추가 |
| 12 | `src/utils/geometry.py` | 순환 참조 위험 | Protocol 패턴 적용 |
| 13 | `src/edgex/device_service.py` | f-string 로깅 | `%s` 포맷 통일 |
| 14 | 전체 | 상수 분산 | `constants.py` 중앙화 |
| 15 | 전체 | `__all__` 미정의 | 공개 모듈에 `__all__` 추가 |
| 16 | 전체 | docstring 스타일 혼용 | Google Style 통일 |
| 17 | `src/core/ai/analyzer.py` | 임시 ID 충돌 | 더 세밀한 ID 생성 |

---

## 부록: 즉시 적용 가능한 수정 코드

### visualizer.py 버그 수정
```python
# 수정 전
"color": EVENT_COLORS.get(EventType(event_type_str.upper()), EVENT_COLORS[EventType.OTHER]),

# 수정 후
try:
    _et = EventType(event_type_str.lower())
except ValueError:
    _et = EventType.OTHER
"color": EVENT_COLORS.get(_et, EVENT_COLORS[EventType.OTHER]),
```

### http.py - has_targets 프로퍼티 추가
```python
class HttpEventForwarder:
    @property
    def has_targets(self) -> bool:
        """전송 대상이 하나 이상 등록되어 있으면 True."""
        return bool(self._targets)
```

### mqtt.py - paho-mqtt v2 호환
```python
def _ensure_connected(self) -> bool:
    if self._client is None:
        try:
            from paho.mqtt.client import CallbackAPIVersion
            self._client = mqtt.Client(
                callback_api_version=CallbackAPIVersion.VERSION1,
                client_id=f"{self.client_id_prefix}-{uuid.uuid4().hex[:8]}",
                clean_session=True,
            )
        except (ImportError, AttributeError):
            self._client = mqtt.Client(
                client_id=f"{self.client_id_prefix}-{uuid.uuid4().hex[:8]}",
                clean_session=True,
            )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
```
