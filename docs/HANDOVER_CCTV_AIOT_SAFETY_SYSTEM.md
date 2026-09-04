# CCTV AIoT 안전관리 시스템 인수인계서

> CCTV 영상 AI와 AIoT 센서 연동 프로젝트를 처음 접하는 담당자가 개발, 운영, 장애 대응을 할 수 있도록 작성한 인수인계 초안입니다.

## 0. 문서 정보

| 항목 | 내용 |
|---|---|
| 프로젝트명 | CCTV AIoT 안전관리 시스템 |
| 저장소 | `CCTV-project` |
| 작성일 | 2026-09-03 |
| 인계자 | `[이름 입력]` |
| 인수자 | `[이름 입력]` |
| 퇴사일/인계 완료일 | `[날짜 입력]` |
| 운영 장비 | NVIDIA Jetson Orin `[장비명/IP 입력]` |
| 운영 장소 | `[현장명 입력]` |

### 처음 읽는 사람을 위한 순서

1. [프로젝트 한눈에 보기](#1-프로젝트-한눈에-보기)
2. [실행과 상태 확인](#3-실행과-상태-확인)
3. [주요 폴더와 파일](#4-주요-폴더와-파일)
4. [낙상 검출 업무](#6-낙상-검출-업무)
5. [장애 대응](#10-장애-대응)
6. [인계 전 확인 목록](#13-인계-전-확인-목록)

---

## 1. 프로젝트 한눈에 보기

### 1.1 프로젝트가 하는 일

CCTV 영상과 센서 데이터를 받아 현장의 안전 관련 상황을 감지하고, 필요한 경우 현장 장비를 제어합니다.

- 사람 검출
- 헬멧 착용 여부 검출
- 낙상 검출
- 얼굴/외형 속성 분석
- 위험구역 침입 및 객체 감시
- 센서 기반 위험 이벤트 처리
- 스피커, 전광판, 사이렌 제어
- Public API, 웹 화면, Grafana를 통한 조회와 모니터링

### 1.2 전체 흐름

```text
카메라/RTSP/비디오
        ↓
AI Engine
(YOLO, YOLO-Pose, DeepStream/TensorRT)
        ↓
표준 AI 이벤트(JSON)
        ↓ MQTT / HTTP
Alert API · EdgeX Adapter · eKuiper Rule Engine
        ↓
Action Layer
        ↓
스피커 · 전광판 · 사이렌 · 외부 시스템 · SQLite/JSONL
        ↓
Public API · 웹 Demo UI · Grafana
```

### 1.3 운영 시 가장 중요한 원칙

- PC 개발/PoC와 Jetson 운영 경로는 다릅니다.
- Jetson 운영은 `docker-compose.jetson.yml`과 `.env.jetson`을 기준으로 합니다.
- 비밀번호, RTSP 계정, API key, 내부 token은 문서와 Git에 기록하지 않습니다.
- 낙상 보조 모델은 충분히 검증하기 전까지 `shadow` 모드로 운영합니다.
- AI Engine, MQTT, Alert API, Action Layer 중 한 계층이라도 중단되면 최종 장비 알람이 나오지 않을 수 있습니다.

---

## 2. 시스템 구성

| 서비스 | 기본 포트 | 역할 |
|---|---:|---|
| `cctv-ai-engine` | 8769 | 영상 입력, AI 추론, 이벤트 생성 |
| `cctv-alert-api` | 8000 | 이벤트/센서 입력 수신 및 저장 |
| `cctv-action-layer` | 8080 | 알람, 외부 전송, 장비 제어 |
| `cctv-public-api` | 9000 | 이벤트, 카메라, 상태 조회 API |
| `cctv-media-server` | 8554 등 | RTSP/MediaMTX 영상 중계 |
| `edgex-mqtt-broker` | 1883 | AI 이벤트 MQTT 전달 |
| `cctv-edgex-adapter` | 내부 | MQTT 이벤트를 EdgeX 형식으로 변환 |
| `cctv-kuiper-rule-loader` | 내부 | eKuiper 룰 등록 |
| `cctv-sensor-rule-bridge` | 내부 | 센서 룰 결과 연결 |
| `aiot-parser` | 내부 | AIoT TLV 센서 데이터 파싱 |
| `cctv-device-dabit` | 내부 | 전광판/장비 연동 |
| `public-demo-ui` | 7000 | 웹 시연 화면 |
| Prometheus/Grafana | 9090/3001 | 선택 실행 모니터링 |

### 실행 경로

```text
PC 개발:    main.py → OpenCV + Ultralytics → VideoProcessor
Jetson 운영: Compose → DeepStream + TensorRT → DeepStreamProcessor
```

---

## 3. 실행과 상태 확인

### 3.1 PC/서버 실행

```bash
cp .env.example .env
docker compose --env-file .env config --quiet
docker compose --env-file .env up -d --build
docker compose --env-file .env ps
```

### 3.2 Jetson 실행

```bash
cp .env.jetson.example .env.jetson
docker compose --env-file .env.jetson -f docker-compose.jetson.yml config --quiet
docker compose --env-file .env.jetson -f docker-compose.jetson.yml up -d --build
docker compose --env-file .env.jetson -f docker-compose.jetson.yml ps
```

### 3.3 상태 확인

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:9000/api/v1/health
curl -fsS http://127.0.0.1:9000/api/v1/readiness
curl -fsS http://127.0.0.1:8769/health
```

운영 API 인증이 켜져 있으면 `-H "X-API-Key: ${PUBLIC_API_KEY}"`를 추가합니다.

주요 화면:

- Swagger: `http://127.0.0.1:9000/docs`
- Demo UI: `http://127.0.0.1:7000/public-demo.html`
- Stream API: `http://127.0.0.1:8769/health`
- Grafana: `http://127.0.0.1:3001` (모니터링 실행 시)

### 3.4 종료

```bash
docker compose --env-file .env.jetson -f docker-compose.jetson.yml down
```

`down`은 external volume의 모델, DB, 로그를 삭제하지 않습니다. volume 삭제는 데이터 손실 가능성이 있으므로 별도 승인 없이 실행하지 않습니다.

---

## 4. 주요 폴더와 파일

| 경로 | 설명 |
|---|---|
| `main.py` | AI Engine 실행 진입점 |
| `app/` | CLI 구현과 외부 ingest |
| `src/bootstrap/` | 런타임 구성, 프로세서 선택 |
| `src/core/` | 영상 처리, 추론, 이벤트 생성 |
| `src/core/ai/` | OpenCV 경로 AI 분석/낙상 판정 |
| `src/core/deepstream_processor.py` | Jetson DeepStream 처리 |
| `src/core/_yolo_postprocess.py` | YOLO-Pose tensor 후처리 |
| `src/core/_deepstream_osd.py` | bbox, skeleton, 상태 표시 |
| `src/core/_fall_shadow_review.py` | 낙상 shadow/검수 클립 |
| `src/api/` | FastAPI Public API |
| `src/services/` | Action Layer와 외부 전송 |
| `src/devices/` | 스피커/전광판/사이렌 클라이언트 |
| `src/storage/` | SQLite 및 outbox |
| `parser-python/` | AIoT TLV 센서 파서 |
| `edgex/` | EdgeX 프로파일/등록 스크립트 |
| `web/` | 시연/관제 화면 |
| `config/` | 이벤트, 라벨, DeepStream 설정 |
| `models/` | YOLO, pose, 헬멧, 외형, TensorRT 모델 |
| `data/` | DB, 로그, crop, 검수 영상 |
| `scripts/` | 학습, 평가, smoke, 운영 점검 |
| `tests/` | pytest 테스트 |
| `docs/` | 설계/운영/API/배포 문서 |

### 업무별 우선 확인 파일

| 업무 | 파일 |
|---|---|
| 낙상 규칙 | `src/core/ai/_fall_detector.py`, `src/core/ai/analyzer.py` |
| DeepStream 낙상 | `src/core/deepstream_processor.py`, `src/core/_yolo_postprocess.py` |
| 이벤트 형식 | `src/core/events.py`, `src/canonical_event.py` |
| 이벤트 우선순위 | `config/event_type_map.json`, `src/event_priority.py` |
| 카메라 | `cameras.json`, `src/utils/camera_input.py` |
| 환경변수 | `.env.example`, `.env.jetson.example`, `docker-compose*.yml` |

---

## 5. 카메라와 환경 설정

### 설정 파일

- `cameras.example.json`: 카메라 설정 예시
- `cameras.json`: 실제 카메라 설정
- `.env.example`: PC/서버용 예시
- `.env.jetson.example`: Jetson용 예시
- `.env`, `.env.jetson`: 실제 운영값. 외부 공유 금지

변경 후 확인:

```bash
python -m json.tool cameras.json >/dev/null
docker compose --env-file .env.jetson -f docker-compose.jetson.yml config --quiet
```

### 자주 조정하는 환경변수

| 변수 | 설명 |
|---|---|
| `USE_DEEPSTREAM`, `USE_GSTREAMER` | 영상 처리 경로 선택 |
| `POSE_MODEL_PATH`, `PERSON_MODEL_PATH`, `HELMET_MODEL_PATH` | 모델 경로 |
| `MQTT_BROKER`, `MQTT_PORT` | MQTT 연결 정보 |
| `PUBLIC_API_KEY` | Public API 인증 키 |
| `INTERNAL_SERVICE_TOKEN` | 내부 서비스 인증 token |
| `DS_YOLO_POSTPROCESS_MODE` | `vectorized` 기본, 문제 시 `legacy` |
| `DS_FALL_SCORE_THRESHOLD` | DeepStream 낙상 점수 기준 |
| `DS_FALL_ENABLE_FOLDED_POSE` | 접힌 바닥 자세 사용 여부 |
| `DS_FALL_FOLDED_POSE_MAX_SPAN_RATIO` | 접힌 자세 후보 품질 제한 |
| `FALLDATA_AUX_ENABLED`, `FALLDATA_AUX_MODE` | 보조 검증과 `shadow/confirm` 모드 |
| `SPEAKER_HOST`, `SIGNBOARD_HOST`, `SIREN_HOST` | 현장 장비 주소 |

---

## 6. 낙상 검출 업무

### 6.1 판정 흐름

```text
YOLO-Pose tensor
  → keypoint/confidence 추출
  → 좌표 복원/NMS
  → 몸통 각도, bbox 비율, keypoint 분산 계산
  → fall_score/fall_reasons 생성
  → fall_detected 또는 fall_near_miss
```

### 6.2 주요 규칙

- `torso_horizontal`: 몸통이 수평에 가까운지 확인
- `torso_flattened`: 몸통이 바닥에 눕는 형태인지 확인
- `wide_bbox_low_head`: bbox 기준으로 머리가 낮고 가로로 긴지 확인
- `low_vertical_span`: keypoint 수직 분산이 작은지 확인
- `leg_above_head`: 다리 관절 위치를 이용하는 보조 조건
- `folded_floor_pose`: 접힌 바닥 자세를 near-miss로 기록
- `missing_leg`: 다리 keypoint 부족 상태를 사유로 기록

머리 위치는 이미지 전체의 절대 좌표가 아니라 사람 bbox의 `y`를 기준으로 계산합니다. 화면 아래쪽이나 가장자리의 사람도 동일 기준으로 판단하기 위한 처리입니다.

### 6.3 최근 pose 업무

| 변경 | 목적 |
|---|---|
| bbox 기준 `bbox_y` 전달 | 화면 좌표와 사람별 좌표 혼동 방지 |
| `wide_bbox_low_head` 보정 | 사람 bbox 내부 기준으로 머리 높이 계산 |
| 다리 신뢰도 기준 분리 | 다리 관절 누락/저신뢰 오판 완화 |
| `folded_floor_pose` span 제한 | 앉은 자세와 바닥 자세 구분 |
| `DS_FALL_FOLDED_POSE_MAX_SPAN_RATIO` 추가 | 운영 중 후보 품질 조정 |
| unit test 추가 | 규칙 회귀 방지 |

### 6.4 실제 영상 replay 결과

예전 영상의 180~230초 부근에 사람이 넘어져 바닥에 누운 장면이 있었고, 현재 코드로 replay한 결과는 다음과 같습니다.

- 결과: `TP`
- `fall_detected`: 5건
- shadow record: 112건
- fall candidate: 7건
- near-miss: 44건
- 최대 fall score: `6.0`
- 최대 확률: 약 `0.784`

주요 판정 근거는 `torso_horizontal`, `torso_flattened`, `wide_bbox_low_head`, `low_vertical_span`입니다.

검증 산출물:

- `data/fall_demo/20260902_142824/overlay.mp4`
- `data/fall_eval/manual_old_overlay.jsonl`
- `data/fall_eval/sample_deepstream_results.jsonl`

주의: 원본이 아닌 OSD 포함 overlay 영상을 입력으로 replay한 결과입니다. 운영 승격 전에는 원본 CCTV 또는 현장 영상으로 재검증해야 합니다.

### 6.5 규칙 수정 원칙

1. 낙상 영상과 정상 행동 영상을 함께 확보합니다.
2. `fall_score`, `fall_reasons`, `near_miss`를 함께 봅니다.
3. 단순히 threshold를 낮춰 `fall_detected`만 늘리지 않습니다.
4. 앉기, 눕기, 물건 줍기, 화면 가장자리 이동의 오탐을 확인합니다.
5. unit test와 DeepStream replay를 모두 실행합니다.

---

## 7. 모델과 학습

| 경로 | 용도 |
|---|---|
| `models/fall/` | 낙상/pose 모델 |
| `models/person/` | 사람 모델 |
| `models/head/` | 머리 모델 |
| `models/appearance/` | 외형/색상 모델 |
| `models/legacy/` | 이전 모델 |
| `models/model_manifest.json` | 모델 목록/경로 |
| `models/experiments/` | 학습/비교 산출물 |

Jetson은 일반적으로 TensorRT `.engine`, PC 개발은 `.pt` 또는 ONNX 모델을 사용합니다. 모델 변경 후 해당 런타임에서 파일 존재 여부와 추론 가능 여부를 확인합니다.

### 주요 명령

```bash
python scripts/run_fall_training_pipeline.py \
  --dataset-root "/path/to/Training" \
  --output-dir data/fall_eval/auto \
  --train --train-direction --decision-threshold 0.7

python scripts/ops/evaluate_sample_deepstream_replay.py \
  --source-mode file --max-videos 20 \
  --results-jsonl data/fall_eval/test_replay_results.jsonl \
  --results-csv data/fall_eval/test_replay_results.csv

python scripts/quality_gate_fall_replay.py \
  --results-jsonl data/fall_eval/test_replay_results.jsonl \
  --min-precision 0.90 --min-recall 0.80
```

`falldata`는 현재 YOLO keypoint와 입력 형식이 다르므로 직접 대체 모델로 사용하지 않습니다. 운영 전에는 `shadow`로 현장 데이터를 수집하고, 정탐/오탐 검수 후 `confirm` 전환 여부를 결정합니다.

---

## 8. 이벤트와 API

### 주요 이벤트

| 이벤트 | 의미 |
|---|---|
| `person` | 사람 검출 |
| `helmet` / `head` | 헬멧 착용/미착용 관련 검출 |
| `fall_detected` | 확정 낙상 이벤트 |
| `fall_near_miss` | 낙상 후보이나 미확정 |
| `danger_zone` / `intrusion` | 위험구역/침입 이벤트 |
| `crowd_warning` | 인원 임계치 초과 |
| `face_recognized` / `face_unknown` | 얼굴 인식 결과 |
| `appearance_match` | 외형 조건 일치 |
| `unsafe_behavior` | 위험행동 |

이벤트 우선순위와 장비 출력 문구는 `config/event_type_map.json`에서 관리합니다.

### 자주 사용하는 API

```text
GET  /api/v1/health
GET  /api/v1/readiness
GET  /api/v1/metrics
GET  /api/v1/events
GET  /api/v1/cameras
GET  /api/v1/sensor-readings
POST /api/v1/alerts
POST /api/v1/event-reviews
GET  /api/v1/event-reviews/summary
GET  /api/v1/appearances
GET  /api/v1/search
```

낙상 이벤트 조회:

```bash
curl -G http://127.0.0.1:9000/api/v1/events \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  --data-urlencode "event_type=fall_detected" \
  --data-urlencode "camera_id=<실제 camera_id>"
```

---

## 9. 장비 알람 흐름

```text
AI Engine
  → MQTT: cctv/ai/events/{camera_id}/<event_type>
  → Alert API / EdgeX Adapter
  → eKuiper 조건 필터
  → Action Layer
  → 스피커/전광판/사이렌/외부 HTTP
```

장비가 동작하지 않을 때는 다음 순서로 확인합니다.

1. AI Engine 로그에 이벤트가 있는가
2. MQTT broker가 살아 있는가
3. Alert API가 이벤트를 받았는가
4. eKuiper confidence/지속시간 조건을 통과했는가
5. Action Layer에 pending/action event가 있는가
6. 장비 IP, 포트, 인증, 네트워크가 정상인가
7. cooldown/중복 억제에 걸리지 않았는가

---

## 10. 장애 대응

### 컨테이너 종료

```bash
docker compose --env-file .env.jetson -f docker-compose.jetson.yml ps
docker compose --env-file .env.jetson -f docker-compose.jetson.yml logs --tail 150 cctv-ai-engine
```

확인 항목:

- 올바른 `.env.jetson` 사용 여부
- 모델 파일과 external volume 존재 여부
- NVIDIA runtime/GPU 접근 여부
- 필요한 장치와 X display 접근 여부

### 영상 미수신

가능성 높은 원인은 RTSP 주소/계정 오류, 네트워크 단절, camera ID 불일치, MediaMTX path 불일치, H.264 timestamp 문제입니다.

```bash
curl -fsS http://127.0.0.1:8769/health
docker compose --env-file .env.jetson -f docker-compose.jetson.yml logs --tail 150 cctv-ai-engine
```

### 낙상 이벤트 없음

1. 영상에 사람이 실제로 보이는가
2. 사람 bbox와 pose keypoint가 표시되는가
3. `fall_score`, `fall_reasons`, `fall_near_miss`가 있는가
4. `DS_FALL_SCORE_THRESHOLD`가 너무 높지 않은가
5. 다른 `camera_id`로 저장되지 않았는가
6. MQTT/Alert API까지 전달되었는가

### 오탐 증가

앉기, 눕기, 물건 줍기 영상을 정상 데이터로 확보하고 near-miss를 정탐/오탐으로 라벨링합니다. threshold 완화보다 pose 누락, bbox 잘림, 카메라 각도와 조명을 먼저 확인합니다.

### 테스트 로그가 비어 있음

영상의 `overlay_source`와 API 조회의 `camera_id`가 같은지 확인합니다. 서로 다르면 영상에는 낙상이 있어도 이벤트 로그가 빈 것처럼 보일 수 있습니다. 기존 이벤트와 새 이벤트를 구분하지 않는 수집 도구에서는 과거 이벤트가 새 이벤트처럼 보일 수도 있습니다.

---

## 11. 테스트와 검증

```bash
.venv/bin/python -m pytest -q
RUNTIME_ENV_FILE=.env.jetson ./scripts/ops/run_operation_check.sh
./scripts/ops/run_deepstream_stability_watch.sh 60 60
python scripts/validate_event_contracts.py --samples
```

각 테스트에는 실행일시, Git commit, camera ID, 영상 구간, 모델/환경변수, TP/FN/FP/TN, score/reasons, frame drop, 제한사항을 함께 기록합니다.

---

## 12. 현재 제한사항과 남은 작업

### 확인된 제한사항

- 예전 실제 녹화 테스트에서 영상과 API `camera_id`가 달라 이벤트 비교가 어려웠습니다.
- 최신 5분 녹화에는 사람이 나타나지 않아 실제 성능 평가로 사용할 수 없었습니다.
- 예전 영상 replay는 `TP`였지만 원본이 아닌 overlay 입력이었습니다.
- `falldata` 모델은 현재 YOLO pose 입력과 직접 호환되지 않습니다.
- 현장 장비와 네트워크는 실제 Jetson에서 별도 확인해야 합니다.

### 우선순위 작업

1. 원본 CCTV 낙상 영상 3개 이상, 정상 행동 영상 5개 이상 확보
2. 녹화 도구의 기존/신규 이벤트 구분 보완
3. fall/non-fall DeepStream replay 평가
4. 카메라별 false positive/false negative 검수
5. Jetson 1/2/4카메라 장시간 안정성 점검
6. 스피커/전광판/사이렌 실제 출력 확인
7. 운영 계정, 환경변수, 모델 파일을 보안 방식으로 인계

---

## 13. 인계 전 확인 목록

### 접근권한

- [ ] Git 저장소/브랜치 권한
- [ ] Jetson SSH 접속 정보
- [ ] 카메라 계정
- [ ] Docker/NVIDIA runtime 권한
- [ ] MQTT 계정
- [ ] Public API key 전달 방법
- [ ] 내부 서비스 token 전달 방법
- [ ] 스피커/전광판/사이렌 계정과 네트워크 정보
- [ ] 운영 서버/Grafana 권한

### 운영자료

- [ ] 실제 카메라 목록과 `camera_id`
- [ ] 카메라별 RTSP 주소와 설치 위치
- [ ] 카메라별 활성 기능
- [ ] 모델 파일과 버전
- [ ] `.env.jetson` 보안 전달
- [ ] Docker volume 목록
- [ ] 백업/복구 방법
- [ ] 최근 장애 이력

### 실습 인계

- [ ] 인수자가 Jetson stack을 기동함
- [ ] health/readiness를 확인함
- [ ] 테스트 영상을 재생함
- [ ] 낙상 이벤트를 API에서 조회함
- [ ] Action Layer 장비 동작을 확인함
- [ ] 로그와 shadow 검수 클립을 찾음
- [ ] 컨테이너 재시작과 롤백을 수행함

### 인계 완료 기준

인수자가 다음 질문에 답할 수 있으면 기본 인계가 완료된 것으로 봅니다.

1. 영상이 안 나올 때 어디부터 확인하는가?
2. 사람이 보이는데 낙상 이벤트가 없으면 어떤 로그를 보는가?
3. Public API와 Action Layer의 차이는 무엇인가?
4. 운영에서 `.env`와 `.env.jetson` 중 어떤 파일을 사용하는가?
5. 모델 교체 전에 어떤 평가를 해야 하는가?
6. 스피커가 울리지 않을 때 AI Engine과 장비를 어떻게 분리해서 확인하는가?

---

## 14. 참고 문서

| 문서 | 용도 |
|---|---|
| `README.md` | 프로젝트 요약/기본 명령 |
| `docs/modules/PROJECT_OVERVIEW.md` | 아키텍처와 설계 의도 |
| `docs/modules/PROJECT_STRUCTURE.md` | 폴더와 실행 진입점 |
| `docs/guides/QUICK_START.md` | PC/Jetson 시작 방법 |
| `docs/guides/OPERATIONS_RUNBOOK.md` | 상태 점검/장애 대응 |
| `docs/guides/OPERATION_CHECKLIST.md` | 운영 투입 전 점검 |
| `docs/guides/DEPLOYMENT_ENVIRONMENT_VARIABLES.md` | 운영 환경변수 |
| `docs/guides/DEEPSTREAM_POSE_POSTPROCESS_DEPLOYMENT.md` | DeepStream pose 후처리 |
| `docs/features/FALLDATA_INTEGRATION.md` | falldata 보조 검증 |
| `docs/features/PUBLIC_API_GUIDE.md` | Public API 사용법 |
| `docs/features/EVENT_SCHEMA_STANDARD.md` | 이벤트 스키마 |
| `docs/operations/pose_training_baseline_20260902.md` | pose 학습 기준선 |

---

## 15. 추가 인계 메모

### 회사 내부 정보

```text
[담당자 연락처, 현장 특이사항, 비공개 계정 전달 방법, 고객별 약속사항 등을 기록]
```

### 질문/답변

| 날짜 | 질문자 | 질문 | 답변 | 후속 작업 |
|---|---|---|---|---|
|  |  |  |  |  |
|  |  |  |  |  |

### 서명

| 구분 | 이름 | 서명/확인일 |
|---|---|---|
| 인계자 |  |  |
| 인수자 |  |  |
| 확인자/팀장 |  |  |
