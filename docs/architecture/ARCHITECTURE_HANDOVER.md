# 시스템 아키텍처 기준 인수인계 설명서

최종 확인 기준: 2026-09-03

## 1. 이 문서의 목적

이 문서는 [시스템 아키텍처 v4](https://www.figma.com/board/uickZg65te7UqNZuoD2yVd)를 기준으로 작성했다. 처음 담당하는 사람이 왼쪽의 입력 장치부터 오른쪽의 출력 장치까지 흐름을 따라가며 다음을 이해할 수 있도록 한다.

- 어떤 디바이스가 시스템에 연결되는가
- 각 디바이스의 데이터 형식과 진입 서비스는 무엇인가
- AI 분석·센서 규칙·EdgeX·eKuiper가 어디에서 동작하는가
- 이벤트가 어떻게 DB와 장치 출력으로 전달되는가
- 장애가 발생했을 때 어느 구간을 확인해야 하는가

## 2. 전체 흐름 한눈에 보기

```text
디바이스 입력
  → 수집 서비스
  → 이벤트·분석 처리
  → MQTT / EdgeX / eKuiper
  → Action Layer
  → 저장 + 실제 출력 장치
```

아키텍처를 읽을 때는 모든 화살표를 한꺼번에 이해할 필요가 없다. 아래 순서로 읽는다.

1. 입력 장치의 종류와 통신 방식을 확인한다.
2. 입력이 들어가는 수집 서비스를 확인한다.
3. AI 또는 센서 규칙으로 위험 이벤트가 만들어지는 지점을 확인한다.
4. MQTT와 EdgeX 중 어느 경로를 타는지 구분한다.
5. Action Layer가 출력 장치별 명령으로 변환하는 과정을 확인한다.

## 3. 입력단: 디바이스별 연결 방식

### 3.1 CCTV 카메라

| 항목 | 내용 |
|---|---|
| 입력 형태 | RTSP 영상 스트림 |
| 진입 서비스 | `cctv-ai-engine` |
| 주요 식별자 | `camera_id` |
| 주요 설정 | `cameras.json`, `POSE_MODEL_PATH`, `HELMET_MODEL_PATH` |
| 처리 결과 | 사람·pose·헬멧·낙상·구역 이벤트 |

카메라는 `cameras.json`의 `id`와 `source`로 등록한다. `camera_id`는 영상 로그, MQTT topic, 이벤트 DB, 검색 API에서 동일하게 사용해야 한다.

```json
{
  "id": "camera_1",
  "name": "현장_카메라_1",
  "source": "rtsp://CAMERA_USER:CAMERA_PASSWORD@CAMERA_IP:554/stream1",
  "enabled": true,
  "detections": ["helmet", "fall", "intrusion"]
}
```

PC는 OpenCV, Jetson은 DeepStream/TensorRT 경로를 사용한다. Jetson에서는 `nvurisrcbin` → `nvstreammux` → `nvinfer` → tracker/OSD → 이벤트 publisher 순서로 동작한다.

### 3.2 기울기 센서

| 항목 | 내용 |
|---|---|
| 입력 형태 | LoRa uplink의 Base64 TLV |
| 원본 topic | `{app_eui}/{dev_eui}/up` |
| 파싱 후 topic | `aiot/sensors/{dev_eui}/t34955` 또는 `t34957` |
| 진입 서비스 | `aiot-parser` |
| 주요 값 | `angle_x`, `angle_y` |
| 규칙 출력 | `aiot/rules/sensor/tilt` |

파서는 Base64를 byte buffer로 바꾸고 table/TLV 타입을 해석한다. 기본 eKuiper 기준 `TILT_THRESHOLD=10.0`도 단위는 degree다. 센서 자체의 물리 측정 범위와 소프트웨어 threshold는 별도 값이다.

### 3.3 온도 센서

| 항목 | 내용 |
|---|---|
| 입력 형태 | LoRa uplink의 Base64 TLV |
| 파싱 후 topic | `aiot/sensors/{dev_eui}/t34957` |
| 주요 값 | `temperature`, `angle_x`, `angle_y`, `event_code` |
| 규칙 출력 | `aiot/rules/sensor/temperature` |
| 기본 기준 | `TEMP_HIGH_THRESHOLD=60.0` |

온도는 `aiot-parser`가 값을 추출하고 eKuiper가 threshold를 비교한다. 파서가 값을 저장했다고 해서 온도 경보가 발생한 것은 아니다. 저장과 위험 판정은 분리되어 있다.

### 3.4 진동·IMU 센서

| 항목 | 내용 |
|---|---|
| 입력 형태 | LoRa uplink의 Base64 TLV |
| 파싱 후 topic | `aiot/sensors/{dev_eui}/t34958` |
| 주요 값 | `acc_x`, `acc_y`, `acc_z`, `gyro_x`, `gyro_y`, `gyro_z`, `event_code` |
| 규칙 출력 | `aiot/rules/sensor/vibration` |

현재 eKuiper 룰은 `event_code=1`인 경우를 진동/충격 이벤트로 전달한다. Python `SensorRuleBridge`에는 3축 가속도의 기준 중력 대비 편차를 계산하는 별도 경로도 있으므로 두 로직을 같은 판정 방식으로 설명하지 않는다.

### 3.5 관제 사용자·API 입력

관제 사용자는 Public API 또는 Alert API로 상태 조회, 자동/수동 모드 변경, 이벤트 승인·거부, 장치 상태 조회를 수행한다. 이 입력은 영상·센서 데이터가 아니라 운영 제어 명령이다.

```text
관제 UI
  → Public API :9000 또는 Alert API :8000
  → Action Layer :8080
  → 장치 제어 또는 상태 변경
```

운영 환경에서는 Public API key와 내부 서비스 token을 사용한다. URL query에 API key를 넣으면 로그에 남을 수 있으므로 헤더 사용을 우선한다.

## 4. 수집·분석 계층

### CCTV AI Engine

입력 frame을 모델에 넣고 사람·pose·헬멧·외형을 분석한다. 낙상은 pose 결과를 `src/core/ai/_fall_detector.py`의 규칙과 temporal 상태에 넣어 최종 후보를 만든다. 모델 score와 운영 이벤트 확정은 같은 개념이 아니다.

### AIoT Parser

외부 MQTT uplink를 수신하고 Base64/TLV를 해석한다. 파서의 주된 책임은 정확한 원시값과 metadata 저장·재발행이다. threshold를 적용해 실제 알람을 결정하는 책임은 eKuiper와 Sensor Rule Bridge에 있다.

### Event Router

AI 이벤트와 센서 이벤트를 공통 형식으로 맞춘다. 대표 필드는 `event_id`, `camera_id` 또는 `device_id`, `type`, `severity`, `confidence`, `occurred_at`, `metadata`다. 기존 top-level 필드 호환성을 유지하면서 canonical event 구조를 추가한다.

## 5. 메시지·규칙 계층

### Mosquitto MQTT

프로젝트 서비스 사이의 비동기 전달 경로다.

| 목적 | topic |
|---|---|
| AI 원본 이벤트 | `cctv/ai/events/{camera_id}/{event_type}` |
| 센서 파싱 결과 | `aiot/sensors/{dev_eui}/{table}` |
| CCTV eKuiper 결과 | `cctv/rules/intrusion/filtered`, `persisted`, `critical` |
| 센서 eKuiper 결과 | `aiot/rules/sensor/tilt`, `temperature`, `vibration` |
| Action 상태 | `cctv/status/action/...` |

### eKuiper

eKuiper는 MQTT stream을 만들고 SQL 조건을 실행한 뒤 결과를 MQTT로 발행한다.

```text
MQTT input
  → Stream schema
  → SQL WHERE / window / HAVING
  → MQTT output
```

주요 파일은 `kuiper/rules/cctv_intrusion_rules.json`, `kuiper/rules/aiot_sensor_rules.json`이다. CCTV 룰은 confidence filter와 5초 지속 조건을 사용하고, 센서 룰은 angle·temperature·event_code를 사용한다.

### EdgeX MessageBus와의 구분

EdgeX Redis MessageBus는 EdgeX 내부 서비스가 사용하는 버스다. 외부 MQTT topic을 직접 구독하는 것과 동일하지 않다. AI/센서 이벤트를 EdgeX Reading으로 투영할 때는 `cctv-edgex-adapter` 또는 parser forwarder가 중간 변환을 수행한다.

## 6. 저장 계층

| 저장 대상 | 목적 |
|---|---|
| 이벤트·센서 DB | 원본 이벤트, 센서 측정값, 검색 |
| EdgeX Core Data | 표준 Reading 저장 |
| Action DB | 장치 명령·실행 결과·실패 이력 |
| Outbox | MQTT/EdgeX 발행 실패 재처리 |

저장 성공과 장치 출력 성공은 별개다. 이벤트 DB에 저장되었어도 스피커가 꺼져 있을 수 있고, 장치 요청이 성공해도 사람이 실제 출력물을 확인한 것은 아닐 수 있다.

## 7. Action Layer와 출력 장치

Action Layer는 이벤트를 받으면 다음 정책을 순서대로 적용한다.

1. 이벤트 canonicalization
2. 사이트별 auto/manual 모드 확인
3. confidence threshold 확인
4. 알람 대상 장치 확인
5. cooldown 및 중복 이벤트 확인
6. 장치별 명령 실행
7. 결과 저장·상태 MQTT 발행

### 스피커

`tts_message` 또는 기본 문구를 사용한다. InterM HTTP Digest로 TTS 생성·BGM 변환·재생을 수행한다.

### 전광판

`display_message` 또는 기본 문구를 사용한다. Dabit TCP/EUC-KR로 표시하거나 Dabit Device Service를 거쳐 Command를 전달한다.

### 사이렌·경광등

문구를 사용하지 않고 이벤트가 알람 조건을 통과하면 ON한다. `SIREN_AUTO_STOP` 후 OFF한다.

실행 결과 예시:

```json
{
  "status": "executed",
  "alarm_played": true,
  "device_results": [
    {"device": "speaker", "status": "acknowledged"},
    {"device": "signboard", "status": "failed"},
    {"device": "siren", "status": "acknowledged"}
  ]
}
```

`acknowledged`는 요청 성공 응답이며 실제 출력 확인과 구분한다.

## 8. 입력에서 출력까지 예시

### 낙상 이벤트

```text
CCTV RTSP
  → AI Engine
  → YOLO-Pose
  → 낙상 규칙·temporal 상태
  → cctv/ai/events/camera_1/fall_detected
  → Event Router / eKuiper
  → Action Layer
  → 스피커·전광판·사이렌
```

### 기울기 센서 이벤트

```text
기울기 센서
  → LoRa Network Server
  → {app_eui}/{dev_eui}/up
  → aiot-parser Base64/TLV
  → aiot/sensors/{dev_eui}/t34955
  → eKuiper threshold
  → aiot/rules/sensor/tilt
  → Action Layer
  → 출력 장치
```

## 9. 수정 시 영향 범위

| 바꾸려는 것 | 반드시 함께 확인할 것 |
|---|---|
| 카메라 ID/topic | cameras.json, MQTT, DB, API, WebRTC path |
| 센서 table/필드 | TLV parser, stream schema, SQL, fixture |
| eKuiper threshold | 정상·경계·위험 payload, Action 알람 빈도 |
| EdgeX resource | profile, metadata 등록, Core Data 조회 |
| 장치 문구 | event_type_map, `display_message`, `tts_message` |
| 장치 protocol | device client, timeout, 결과 저장, UAT |
| MQTT broker | AI Engine, parser, eKuiper, adapter, Action Layer |

## 10. 장애 확인 순서

### 입력이 안 들어오는 경우

1. CCTV RTSP 또는 센서 원본 MQTT가 실제로 살아 있는지 확인한다.
2. `camera_id`, `app_eui`, `dev_eui`, topic을 확인한다.
3. 수집 서비스 로그와 연결 설정을 확인한다.

### 입력은 있지만 이벤트가 없는 경우

1. 모델 경로·runtime·label map을 확인한다.
2. 후처리 threshold와 eKuiper SQL 조건을 확인한다.
3. confidence threshold, 시간 window, cooldown을 확인한다.

### 이벤트는 있지만 장치가 동작하지 않는 경우

1. Action Layer가 해당 topic을 구독하는지 확인한다.
2. 자동/수동 모드와 pending 상태를 확인한다.
3. `/api/v1/control/devices`의 configured/reachable을 확인한다.
4. 장치별 HTTP/TCP 로그와 `device_results`를 확인한다.

### EdgeX에만 문제가 있는 경우

1. Core Metadata의 device/profile/resource 등록 여부를 확인한다.
2. Core Data와 Redis health를 확인한다.
3. adapter/forwarder outbox pending과 retry를 확인한다.
4. MQTT 이벤트는 정상인지 EdgeX Reading만 누락인지 분리한다.

## 11. 인수인계 실습 순서

1. Figma 아키텍처 v4를 열고 입력 장치 5종을 설명한다.
2. `camera_1`의 RTSP 입력과 AI Engine 로그를 확인한다.
3. 과거 fall 영상 replay 결과와 실제 MQTT 이벤트를 확인한다.
4. 센서 fixture 또는 실제 uplink를 parser에 넣는다.
5. `aiot/sensors/#`와 `aiot/rules/sensor/#`를 비교한다.
6. EdgeX Metadata/Core Data에서 device와 Reading을 확인한다.
7. Action Layer의 auto/manual을 각각 실행한다.
8. 각 장치의 acknowledged/failed 결과를 확인한다.
9. UAT 문서에 결과와 증거 경로를 기록한다.

## 관련 문서

- [Figma 시스템 아키텍처 v4](https://www.figma.com/board/uickZg65te7UqNZuoD2yVd)
- [프로젝트 구조·파이프라인](PROJECT_STRUCTURE_AND_PIPELINES.md)
- [EdgeX·eKuiper 상세 인수인계서](../integrations/EDGEX_KUIPER_HANDOVER.md)
- [센서 연동](SENSOR_INTEGRATION.md)
- [디바이스 이벤트 Payload](../devices/EVENT_PAYLOADS.md)
- [현장 UAT 체크리스트](../handover/FIELD_UAT_CHECKLIST.md)
