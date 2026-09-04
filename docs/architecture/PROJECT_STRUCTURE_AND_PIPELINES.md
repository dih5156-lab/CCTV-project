# 프로젝트 구조와 처리 파이프라인

## 1. 한 문장으로 이해하기

카메라 영상은 AI Engine에서 사람·자세·위험 상황으로 분석되고, LoRa 센서는 `aiot-parser`에서 TLV를 해석한다. 두 입력은 MQTT 이벤트로 합쳐져 Alert/EdgeX/Action Layer를 거쳐 DB, 관제 API, 스피커·전광판·사이렌으로 전달된다.

## 2. 주요 디렉터리

| 경로 | 역할 | 처음 수정할 때 볼 파일 |
|---|---|---|
| `src/core/ai` | 낙상·헬멧·외관 등 AI 판정 | `analyzer.py`, `_fall_detector.py` |
| `src/core` | 영상 프레임·DeepStream 처리 | `deepstream_processor.py`, `_yolo_postprocess.py` |
| `src/api` | Public/Alert API | `src/api/v1/` |
| `src/services` | MQTT, 센서, Action, 외부 입력 연결 | `action_bridge.py`, `sensor_rule_bridge.py` |
| `src/devices` | 스피커·전광판·사이렌·센서 모델 | `speaker.py`, `signboard.py`, `siren.py` |
| `src/protocols` | 내부 REST 및 통신 계약 | `rest.py` |
| `parser-python` | LoRa MQTT 수신, Base64/TLV 파싱, DB 저장 | `main.py`, `tlv/parser.py` |
| `edgex` | EdgeX 장치 프로파일·등록 스크립트·ASC 설정 | `register_*.py`, `device-profiles/` |
| `kuiper` | 센서·AI 이벤트 규칙 | `rules/`, `etc/` |
| `runners` | 장치 서비스와 보조 프로세스 실행 진입점 | `run_*_device_service.py` |
| `docker-compose*.yml` | PC/Jetson 서비스 조립 | `docker-compose.yml`, `docker-compose.jetson.yml` |
| `docs` | 운영·통합·인수인계 문서 | `guides/`, `integrations/` |

## 3. 영상 파이프라인

```text
RTSP/파일
  → OpenCV(PC) 또는 DeepStream( Jetson )
  → YOLO / YOLO-Pose 추론
  → bbox·keypoint 후처리
  → 낙상·안전 규칙과 시간적 상태 판정
  → MQTT cctv/ai/events/{camera_id}/{event_type}
  → Alert 저장 / EdgeX 투영 / Action Layer 장치 제어
```

Jetson 경로는 `nvinfer`의 raw tensor를 Python pad-probe에서 후처리한다. letterbox 좌표를 원본 영상으로 되돌린 뒤 사람별 keypoint를 만들고, 낙상 규칙·NMS·OSD 순서로 처리한다. 후처리 통계는 `yolo_postprocess`, `avg_ms`, `max_ms`, `frame_dropped`, `failed` 로그로 확인한다.

### 낙상 판정에서 중요한 값

기본 규칙은 단일 자세만 보지 않고 bbox 비율, 머리 위치, 몸통 기울기, keypoint span, 시간 누적 상태를 함께 본다. 기본 threshold와 운영용 환경 변수는 [배포 환경 변수 문서](../guides/DEPLOYMENT_ENVIRONMENT_VARIABLES.md)의 Pose 표를 기준으로 한다.

실제 리플레이 검증에서는 `data/fall_demo/20260902_142824/overlay.mp4`를 사용해 `fall_detected` 이벤트가 확인됐다. 이는 해당 영상과 현재 설정에 대한 검증이며, 모든 카메라·조명·거리 조건의 정확도를 보장하는 결과는 아니다.

## 4. 센서 파이프라인

```text
LoRa Network Server
  → 외부 MQTT {app_eui}/{dev_eui}/up
  → payload Base64 디코드
  → TLV table/type 해석
  → aiot/sensors/{dev_eui}/{table_name}
  → SensorReading 표준화 및 DB 저장
  → aiot/rules/sensor/{sensor_type}
  → eKuiper / Sensor Rule Bridge
  → cctv/rules 또는 장치 명령
```

상세 payload와 예시는 [센서 연동 형식과 방법](SENSOR_INTEGRATION.md)에 있다.

## 5. 이벤트와 장치 파이프라인

AI·센서 이벤트는 Mosquitto MQTT에서 비동기로 전달된다. Alert 계층은 이벤트·알림을 저장하고, Action Layer는 자동/수동 정책, 대상 장치, cooldown, 재시도, 결과 이력을 관리한다. 전광판은 Dabit TCP, 스피커와 사이렌은 InterM HTTP Digest 방식이다.

