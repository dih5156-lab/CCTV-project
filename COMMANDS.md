# CCTV Project — 실행 명령어 가이드

## 목차
1. [환경 설정](#1-환경-설정)
2. [AI 엔진 (메인 CCTV)](#2-ai-엔진-메인-cctv)
3. [액션 레이어](#3-액션-레이어)
4. [Alert API 서버](#4-alert-api-서버)
5. [EdgeX 어댑터](#5-edgex-어댑터)
6. [Kuiper 룰 배포](#6-kuiper-룰-배포)
7. [외부 MQTT 수신 (External Ingest)](#7-외부-mqtt-수신-external-ingest)
8. [AIoT TLV 파서 서버](#8-aiot-tlv-파서-서버)
9. [테스트](#9-테스트)
10. [Docker Compose](#10-docker-compose)

---

## 1. 환경 설정

### 가상환경 생성 및 활성화

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 의존성 설치

```bash
# AI 엔진 전체 (YOLO, OpenCV, torch 포함)
pip install -r requirements.txt

# 액션 레이어 전용 (torch 제외, 경량)
pip install -r requirements-action.txt

# 개발 도구 (pytest, black, mypy 포함)
pip install -r requirements-dev.txt

# AIoT TLV 파서 서버
pip install -r parser-python/requirements.txt
```

---

## 2. AI 엔진 (메인 CCTV)

진입점: `main.py`

### 기본 실행 (웹캠)

```bash
python main.py
```

### 웹캠 + 화면 표시

```bash
python main.py --display
```

### 비디오 파일 테스트

```bash
python main.py --video sample.mp4 --display
```

### 다중 카메라 (cameras.json)

```bash
python main.py --cameras cameras.json
```

### CUDA 가속 + 다중 카메라

```bash
python main.py --cameras cameras.json --device cuda --display
```

### 헬멧/낙상 감지 + MQTT 전송

```bash
python main.py \
  --cameras cameras.json \
  --device cuda \
  --confidence 0.5 \
  --pose-confidence 0.3 \
  --mqtt-broker localhost \
  --mqtt-port 1883 \
  --mqtt-topic-prefix cctv/ai/events
```

### 위험 구역 탐지 + Zone API 활성화

```bash
python main.py \
  --cameras cameras.json \
  --zone-detection \
  --zones-config zones_config.json \
  --api-port 8765
```

### 데이터셋 수집 모드

```bash
python main.py \
  --cameras cameras.json \
  --collect-dataset \
  --dataset-dir ./collected_data
```

### 주요 인자 요약

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--cameras` | (없음) | 카메라 목록 JSON 파일 |
| `--video` | (없음) | 단일 비디오 파일 |
| `--device` | `cpu` | `cpu` 또는 `cuda` |
| `--confidence` | `0.5` | 헬멧 감지 임계값 |
| `--pose-confidence` | `0.3` | 사람 감지 임계값 |
| `--fps` | `30` | 목표 FPS |
| `--frame-skip` | `3` | AI 추론 간격 (N프레임마다) |
| `--display` | off | 화면 표시 |
| `--mqtt-broker` | `localhost` | MQTT 브로커 호스트 |
| `--mqtt-port` | `1883` | MQTT 브로커 포트 |
| `--api-port` | `0` | Zone API 포트 (0=비활성) |
| `--zone-detection` | off | 위험 구역 탐지 활성화 |
| `--no-debounce` | off | 이벤트 디바운싱 비활성화 |
| `--debounce` | `3.0` | 디바운싱 간격 (초) |

---

## 3. 액션 레이어

진입점: `runners/run_action_bridge.py`

스피커 / 전광판 / 경광등 조치 실행, 외부 플랫폼 HTTP 전송, SQLite 이벤트 저장.

### 기본 실행

```bash
python runners/run_action_bridge.py
```

### 환경 변수로 실행 (권장)

```bash
# .env 또는 shell export
export MQTT_BROKER=localhost
export MQTT_PORT=1883
export DB_PATH=/app/action_events.db

python runners/run_action_bridge.py
```

### CLI 인자로 실행

```bash
python runners/run_action_bridge.py \
  --mqtt-broker localhost \
  --mqtt-port 1883 \
  --db-path ./action_events.db \
  --subscribe-topics "cctv/rules/intrusion/filtered,cctv/rules/intrusion/critical" \
  --alarm-topics "cctv/rules/intrusion/critical"
```

---

## 4. Alert API 서버

진입점: `runners/run_alert_api.py`

내부 HTTP Alert 수신 서버. 외부 플랫폼 → CCTV 시스템으로 알림 수신.

### 기본 실행

```bash
python runners/run_alert_api.py
```

### 포트 및 로그 경로 지정

```bash
python runners/run_alert_api.py \
  --host 0.0.0.0 \
  --port 8000 \
  --log-path ./alert_api_events.jsonl
```

### 헬스 체크 (서버 실행 후)

```bash
curl http://localhost:8000/health
```

### 알림 전송 테스트

```bash
curl -X POST http://localhost:8000/api/alerts \
  -H "Content-Type: application/json" \
  -d '{"camera_id": "cam1", "event": "intrusion"}'
```

---

## 5. EdgeX 어댑터

진입점: `runners/run_edgex_adapter.py`

AI 엔진 MQTT 이벤트 → EdgeX Core Data/MessageBus 브릿지.

### 기본 실행

```bash
python runners/run_edgex_adapter.py
```

### EdgeX와 연동

```bash
python runners/run_edgex_adapter.py \
  --ai-mqtt-broker localhost \
  --ai-mqtt-port 1883 \
  --ai-topic-prefix cctv/ai/events \
  --edgex-metadata-url http://localhost:59881 \
  --edgex-data-url http://localhost:59880 \
  --edgex-mqtt-broker localhost \
  --edgex-mqtt-port 1883 \
  --service-name cctv-device-service
```

---

## 6. Kuiper 룰 배포

진입점: `runners/run_kuiper_rules.py`

eKuiper Rules Engine에 침입 감지 룰(스트림 + 규칙)을 배포.

### 기본 실행

```bash
python runners/run_kuiper_rules.py
```

### 브로커 및 Kuiper API 지정

```bash
python runners/run_kuiper_rules.py \
  --kuiper-api http://localhost:59720 \
  --mqtt-broker localhost \
  --mqtt-port 1883 \
  --rules-file kuiper/rules/cctv_intrusion_rules.json
```

### 신뢰도 및 재시도 설정

```bash
python runners/run_kuiper_rules.py \
  --intrusion-confidence 0.7 \
  --critical-confidence 0.9 \
  --persist-hit-count 3 \
  --retry-count 5 \
  --retry-delay 5
```

---

## 7. 외부 MQTT 수신 (External Ingest)

진입점: `run_external_ingest.py`

외부 MQTT 브로커 구독 → 내부 이벤트 정규화 → (선택) 내부 MQTT 재발행.

### 기본 실행

```bash
python run_external_ingest.py \
  --mqtt-broker external-broker.example.com \
  --mqtt-port 1883 \
  --topic "sensors/#" \
  --topic "alerts/#"
```

### 인증 + 재발행

```bash
python run_external_ingest.py \
  --mqtt-broker external-broker.example.com \
  --mqtt-port 8883 \
  --mqtt-username myuser \
  --mqtt-password mypass \
  --topic "sensors/#" \
  --republish \
  --republish-broker localhost \
  --republish-port 1883 \
  --republish-topic-prefix cctv/external
```

### DB 저장 경로 지정

```bash
python run_external_ingest.py \
  --mqtt-broker localhost \
  --topic "test/#" \
  --db-path ./ingest_raw.db
```

---

## 8. AIoT TLV 파서 서버

진입점: `parser-python/main.py`

LoRaWAN TLV 센서 데이터 수신 → 파싱 → PostgreSQL 저장 + EdgeX 발행.

### 환경 변수 준비

```bash
# parser-python/.env 파일 생성
cp parser-python/.env.example parser-python/.env  # 없으면 직접 작성
```

`.env` 예시:
```ini
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PW=yourpassword
DB_NAME=aiot_sensor

NS_PARK_MQTT_HOST=ns.example.com
NS_PARK_MQTT_PORT=1883
NS_PARK_MQTT_ID=user

LAB_MQTT_HOST=lab-broker.example.com
LAB_MQTT_PORT=1883

EDGEX_MQTT_HOST=localhost
EDGEX_MQTT_PORT=1883

ROUTER=3500
NC_APPLICATION_IDS=app1,app2
NC_API_RUI=http://localhost:3000/api/v1/devices
```

### 서버 실행

```bash
cd parser-python
python main.py
```

### TLV 파서 단독 테스트

```bash
cd parser-python
pytest tests/test_tlv_parser.py -v
```

### 실시간 MQTT 수신 모니터링 (개발용)

```bash
cd parser-python
python live_receiver.py

# 특정 브로커만
python live_receiver.py --broker ns_park

# 다른 .env 경로 지정
python live_receiver.py --env ../.env
```

---

## 9. 테스트

### 전체 테스트 실행

```bash
python -m pytest tests/
```

### 상세 출력

```bash
python -m pytest tests/ -v
```

### 특정 파일만

```bash
python -m pytest tests/test_zone_detection.py -v
python -m pytest tests/test_ai_analysis.py -v
python -m pytest tests/test_action_bridge.py -v
```

### 커버리지 측정

```bash
python -m pytest tests/ --cov=src --cov-report=term-missing
```

### TLV 파서 테스트 (별도 경로)

```bash
python -m pytest parser-python/tests/ -v
```

---

## 10. Docker Compose

### 전체 스택 시작

```bash
docker compose up -d
```

### 특정 서비스만 시작

```bash
# AI 엔진만
docker compose up -d cctv-ai-engine

# 액션 레이어만
docker compose up -d cctv-action-layer
```

### 로그 확인

```bash
docker compose logs -f cctv-ai-engine
docker compose logs -f cctv-action-layer
docker compose logs -f app-rules-engine
```

### 재시작

```bash
docker compose restart cctv-action-layer
```

### 서비스 재빌드 후 시작

```bash
docker compose up -d --build cctv-ai-engine
```

### 전체 중지 및 제거

```bash
docker compose down
```

### Jetson Orin 전용 Compose

```bash
docker compose -f docker-compose.jetson.yml up -d
```
