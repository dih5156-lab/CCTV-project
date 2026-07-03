# CCTV Public API 사용 가이드

## 결론

Public API는 서버팀, 프론트엔드 대시보드, 외부 시스템이 CCTV 플랫폼 상태와 이벤트를 조회하거나 제어할 때 사용하는 FastAPI 기반 API입니다.

기본 실행 주소는 다음과 같습니다.

```text
http://localhost:9000
```

브라우저에서 `http://localhost:9000/`을 열면 주요 경로 안내 JSON이 표시됩니다.
Swagger 문서는 `http://localhost:9000/docs`에서 확인합니다.

모든 주요 API는 `/api/v1` prefix를 사용합니다.

새로 생성되는 ISO 8601 시각 문자열은 한국 표준시 오프셋 `+09:00`을 포함합니다.
Unix epoch 숫자는 시간대와 무관한 절대시각이므로 기존 형식을 유지합니다.

```text
GET    /api/v1/health
GET    /api/v1/readiness
GET    /api/v1/metrics
GET    /api/v1/events
POST   /api/v1/alerts
GET    /api/v1/sensor-readings
POST   /api/v1/sensor-readings
GET    /api/v1/cameras
GET    /api/v1/sites
POST   /api/v1/sites
DELETE /api/v1/sites/{site_id}
GET    /api/v1/control/mode
POST   /api/v1/control/mode
GET    /api/v1/control/pending
GET    /api/v1/control/devices
GET    /api/v1/control/action-events
POST   /api/v1/control/approve/{event_id}
POST   /api/v1/control/reject/{event_id}
POST   /api/v1/event-reviews
GET    /api/v1/event-reviews/summary
GET    /api/v1/appearances
GET    /api/v1/appearances/status
POST   /api/v1/appearances
DELETE /api/v1/appearances/{condition_id}
GET    /api/v1/search
GET    /api/v1/search/crops/{filename}
```

## 실행 방법

로컬에서 단독 실행:

```bash
.venv/bin/python runners/run_public_api.py --host 0.0.0.0 --port 9000
```

Docker Compose 기준 실행:

```bash
docker compose up -d cctv-public-api
```

API 문서는 서버 실행 후 아래 주소에서 확인할 수 있습니다.

```text
http://localhost:9000/docs
http://localhost:9000/redoc
```

복붙용 `curl` 샘플은 [PUBLIC_API_EXAMPLES.md](PUBLIC_API_EXAMPLES.md)를 참고하세요.

## 인증

운영 환경에서는 `PUBLIC_API_KEY`를 설정하고, 클라이언트는 `X-API-Key` 헤더를 보내야 합니다.

```bash
curl -H "X-API-Key: ${PUBLIC_API_KEY}" \
  http://localhost:9000/api/v1/health
```

개발 환경에서 `PUBLIC_API_KEY`가 설정되지 않으면 인증은 통과하지만 경고 로그가 남습니다. 운영에서는 반드시 설정해야 합니다.

쿼리 파라미터 인증은 기본적으로 비활성화되어 있습니다. 꼭 필요한 임시 상황에서만 아래 값을 켭니다.

```bash
PUBLIC_API_ALLOW_QUERY_KEY=1
```

이 경우 `?api_key=...`도 허용됩니다. 단, URL 로그에 키가 남을 수 있으므로 운영 상시 사용은 권장하지 않습니다.

## 공통 응답 형식

단건/명령형 API는 `BaseResponse` 형식을 사용합니다.

```json
{
  "success": true,
  "data": {},
  "error": null,
  "timestamp": "2026-05-04T00:00:00Z"
}
```

목록 API 중 페이지네이션이 있는 API는 `PaginatedResponse` 형식을 사용합니다.

```json
{
  "success": true,
  "items": [],
  "total": 0,
  "limit": 50,
  "offset": 0,
  "timestamp": "2026-05-04T00:00:00Z"
}
```

오류 응답은 가능한 한 아래 형식을 유지합니다.

```json
{
  "success": false,
  "data": null,
  "error": "오류 메시지",
  "timestamp": "2026-05-04T00:00:00Z"
}
```

## 주요 환경변수

| 환경변수 | 기본값 | 설명 |
|---|---:|---|
| `PUBLIC_API_KEY` | 없음 | Public API 인증 키 |
| `PUBLIC_API_ALLOW_QUERY_KEY` | `0` | `?api_key=` 인증 허용 여부 |
| `ACTION_LAYER_URL` | `http://cctv-action-layer:8080` | 사이트/제어 API가 프록시할 Action Layer 주소 |
| `ALERT_API_URL` | `http://cctv-alert-api:8000` | Alert 수신 API 주소 |
| `ALERT_LOG_PATH` | `/app/data/logs/alert_api_events.jsonl` | 이벤트 조회 API가 읽는 JSONL 로그. `.1`부터 `.5`까지 회전 로그도 함께 조회 |
| `ALERT_FALLBACK_LOG` | `/app/data/logs/public_api_fallback.jsonl` | Alert API 중계 실패 시 fallback 로그 |
| `CAMERAS_JSON` | `/app/cameras.json` | 카메라 목록 조회에 사용하는 설정 파일 |
| `APPEARANCES_DB` | `/app/data/runtime/appearances.db` | 외형 조건/검색에 사용하는 SQLite DB |
| `INTERNAL_SERVICE_TOKEN` | 없음 | 내부 서비스 간 `X-Internal-Token` 공유 토큰 |
| `CORS_ORIGINS` | `*` | 허용할 CORS origin 목록, comma-separated |

## 상태와 메트릭

### `GET /api/v1/health`

Public API 프로세스 상태를 확인합니다.

```bash
curl -H "X-API-Key: ${PUBLIC_API_KEY}" \
  http://localhost:9000/api/v1/health
```

응답 예시:

```json
{
  "success": true,
  "data": {
    "status": "up",
    "service": "cctv-public-api",
    "version": "1.0.0",
    "checked_at": "2026-05-04T00:00:00Z",
    "action_layer_url": "http://cctv-action-layer:8080",
    "alert_api_url": "http://cctv-alert-api:8000"
  },
  "error": null,
  "timestamp": "2026-05-04T00:00:00Z"
}
```

### `GET /api/v1/readiness`

Public API가 의존하는 Action Layer와 Alert API까지 함께 확인합니다.
운영 배포나 대시보드 초기 연결 확인에는 `/health`보다 `/readiness`를 보는 것이 더 안전합니다.

```bash
curl -H "X-API-Key: ${PUBLIC_API_KEY}" \
  http://localhost:9000/api/v1/readiness
```

정상 응답은 HTTP 200과 `status=ready`입니다.

```json
{
  "success": true,
  "data": {
    "status": "ready",
    "service": "cctv-public-api",
    "checked_at": "2026-05-04T00:00:00Z",
    "dependencies": [
      {
        "name": "action-layer",
        "url": "http://cctv-action-layer:8080/health",
        "status": "up",
        "status_code": 200
      },
      {
        "name": "alert-api",
        "url": "http://cctv-alert-api:8000/health",
        "status": "up",
        "status_code": 200
      }
    ]
  },
  "error": null,
  "timestamp": "2026-05-04T00:00:00Z"
}
```

하위 서비스가 내려가면 HTTP 503과 `status=degraded`를 반환합니다.

### `GET /api/v1/metrics`

Prometheus scrape용 엔드포인트입니다. 인증 의존성은 붙어 있지 않고, Prometheus text format을 반환합니다.
`cctv_public_api_http_requests_total` counter는 HTTP method, 정규화된 path prefix, status code별 요청 수를 기록합니다.

```bash
curl http://localhost:9000/api/v1/metrics
```

Prometheus 설정 예시:

```yaml
- job_name: cctv-public-api
  static_configs:
    - targets: ["cctv-public-api:9000"]
  metrics_path: /api/v1/metrics
```

## 이벤트 API

### `GET /api/v1/events`

Alert API가 저장한 JSONL 로그를 최신순으로 조회합니다.

Query parameter:

| 이름 | 설명 |
|---|---|
| `limit` | 페이지 크기, 기본 50, 최대 500 |
| `offset` | 시작 offset |
| `camera_id` | 특정 카메라 ID 필터 |
| `event_type` | 이벤트 타입 필터 |
| `time_from` | 시작 Unix timestamp, 초 단위 |
| `time_to` | 종료 Unix timestamp, 초 단위 |

```bash
curl -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "http://localhost:9000/api/v1/events?limit=20&event_type=fall_detected"
```

응답 아이템 예시:

```json
{
  "camera_id": "cam-01",
  "event_type": "fall_detected",
  "severity": "critical",
  "confidence": 0.91,
  "timestamp": 1777891200.0,
  "bbox": {
    "x": 100,
    "y": 120,
    "width": 80,
    "height": 160
  },
  "object_id": 12,
  "metadata": {},
  "received_at": "2026-05-04T00:00:00Z"
}
```

### `POST /api/v1/alerts`

외부 시스템이나 AI 엔진이 탐지 이벤트를 Public API로 push할 때 사용합니다. Public API는 내부 `cctv-alert-api`로 중계하고, 중계 실패 시 fallback JSONL에 저장합니다.

```bash
curl -X POST "http://localhost:9000/api/v1/alerts" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  -d '{
    "camera_id": "cam-01",
    "event_type": "fall_detected",
    "severity": "critical",
    "confidence": 0.91,
    "timestamp": 1777891200.0,
    "bbox": {"x": 100, "y": 120, "width": 80, "height": 160},
    "object_id": 12,
    "metadata": {"zone_id": "zone-a"}
  }'
```

지원 이벤트 타입:

```text
helmet
head
face_recognized
face_unknown
danger_zone
fall_detected
not_fall
unsafe_behavior
person
other
crowd_warning
zone_object
appearance_match
```

## 카메라 API

### `GET /api/v1/cameras`

`CAMERAS_JSON` 파일을 읽어 카메라 목록을 반환합니다. RTSP URL의 사용자명/비밀번호는 제거됩니다.

```bash
curl -H "X-API-Key: ${PUBLIC_API_KEY}" \
  http://localhost:9000/api/v1/cameras
```

응답 아이템 예시:

```json
{
  "id": "cam-01",
  "name": "현장 1번 카메라",
  "url": "rtsp://192.168.0.10",
  "zones": []
}
```

### `GET /api/v1/cameras/{camera_id}`

카메라 1건을 조회합니다. 없는 ID면 404를 반환합니다.

## 사이트 API

사이트 API는 내부 Action Layer REST 서버로 요청을 프록시합니다. 따라서 `ACTION_LAYER_URL`이 올바르게 설정되어 있어야 합니다.

### `GET /api/v1/sites`

등록된 사이트 목록을 조회합니다.

```bash
curl -H "X-API-Key: ${PUBLIC_API_KEY}" \
  http://localhost:9000/api/v1/sites
```

응답 아이템 예시:

```json
{
  "site_id": "site-01",
  "site_name": "A 공장",
  "site_nickname": "1라인",
  "camera_ids": ["cam-01", "cam-02"],
  "control_mode": "auto",
  "alarm_devices": ["speaker", "signboard"]
}
```

### `POST /api/v1/sites`

사이트를 등록합니다.

```bash
curl -X POST "http://localhost:9000/api/v1/sites" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  -d '{
    "site_id": "site-01",
    "site_name": "A 공장",
    "site_nickname": "1라인",
    "camera_ids": ["cam-01", "cam-02"],
    "control_mode": "auto",
    "alarm_devices": ["speaker", "signboard"]
  }'
```

### `DELETE /api/v1/sites/{site_id}`

사이트를 삭제합니다.

```bash
curl -X DELETE -H "X-API-Key: ${PUBLIC_API_KEY}" \
  http://localhost:9000/api/v1/sites/site-01
```

## 제어 API

제어 API도 Action Layer로 프록시됩니다.

### `GET /api/v1/control/mode`

현재 전역 제어 모드를 조회합니다.

```bash
curl -H "X-API-Key: ${PUBLIC_API_KEY}" \
  http://localhost:9000/api/v1/control/mode
```

### `POST /api/v1/control/mode`

전역 또는 사이트 단위 제어 모드를 변경합니다.

```bash
curl -X POST "http://localhost:9000/api/v1/control/mode" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  -d '{"mode": "manual", "site_id": "site-01"}'
```

`site_id`를 생략하면 전체 모드 변경으로 전달됩니다.

### `GET /api/v1/control/pending`

수동 승인 대기 이벤트 목록을 조회합니다. 프론트에서는 최소한 `event_id`, `camera_id`, `event_type`, `queued_at`을 기준으로 화면을 구성하는 것을 권장합니다.

### `POST /api/v1/control/approve/{event_id}`

수동 승인 대기 이벤트를 승인합니다.

```bash
curl -X POST -H "X-API-Key: ${PUBLIC_API_KEY}" \
  http://localhost:9000/api/v1/control/approve/event-123
```

### `POST /api/v1/control/reject/{event_id}`

수동 승인 대기 이벤트를 거부합니다.

```bash
curl -X POST -H "X-API-Key: ${PUBLIC_API_KEY}" \
  http://localhost:9000/api/v1/control/reject/event-123
```

## 외형 조건 API

외형 조건은 “찾고 싶은 사람의 속성 조건”을 등록하는 API입니다. 예를 들어 상의 색상, 하의 색상, 헬멧 착용 여부, 가방 소지 여부를 조건으로 등록할 수 있습니다.

### `GET /api/v1/appearances`

등록된 외형 조건 목록을 조회합니다.

### `GET /api/v1/appearances/status`

외형 검색 기능의 준비 상태를 조회합니다. 대시보드는 이 응답으로 “검색 필터를 활성화할 수 있는지”를 판단하면 됩니다.

상세 계약은 [APPEARANCES_STATUS_API.md](APPEARANCES_STATUS_API.md)를 참고하세요.

### `POST /api/v1/appearances`

외형 조건을 등록합니다.

```bash
curl -X POST "http://localhost:9000/api/v1/appearances" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  -d '{
    "name": "검은 상의 작업자",
    "upper_color": "black",
    "has_helmet": true,
    "threshold": 0.8,
    "cameras": ["cam-01"],
    "enabled": true
  }'
```

사용 가능한 색상 값:

```text
red
orange
yellow
green
blue
purple
white
black
gray
```

### `DELETE /api/v1/appearances/{condition_id}`

외형 조건을 삭제합니다.

## 외형 기록 검색 API

### `GET /api/v1/search`

SQLite에 저장된 외형 기록을 조건으로 검색합니다.

Query parameter:

| 이름 | 설명 |
|---|---|
| `camera_id` | 카메라 ID |
| `upper_color` | 상의 색상 |
| `lower_color` | 하의 색상 |
| `has_helmet` | 헬멧 착용 여부 |
| `helmet_color` | 헬멧 색상 |
| `has_backpack` | 백팩 소지 |
| `has_handbag` | 핸드백 소지 |
| `has_suitcase` | 캐리어 소지 |
| `gender` | 성별 |
| `age_group` | 나이대 |
| `face_name` | 얼굴 이름 부분 일치 |
| `time_from` | 시작 시각, `YYYY-MM-DD`, `YYYY-MM-DD HH:MM:SS`, `YYYY-MM-DDTHH:MM:SS` |
| `time_to` | 종료 시각 |
| `limit` | 페이지 크기, 기본 50, 최대 500 |
| `offset` | 시작 offset |

```bash
curl -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "http://localhost:9000/api/v1/search?upper_color=black&has_helmet=true&limit=20"
```

응답 아이템에는 crop 이미지 URL이 포함될 수 있습니다.

```json
{
  "id": 1,
  "timestamp": 1777891200.0,
  "datetime_str": "2026-05-04 00:00:00",
  "camera_id": "cam-01",
  "track_id": 12,
  "upper_color": "black",
  "lower_color": "gray",
  "has_helmet": true,
  "helmet_color": "white",
  "has_backpack": false,
  "has_handbag": false,
  "has_suitcase": false,
  "gender": null,
  "age_group": null,
  "face_name": null,
  "attribute_backend": "hsv",
  "crop_url": "/api/v1/search/crops/cam-01_12.jpg"
}
```

### `GET /api/v1/search/crops/{filename}`

저장된 JPEG crop 이미지를 반환합니다.

보안상 파일명은 안전한 `.jpg` 패턴만 허용합니다.
보존 기간이 지나 crop 파일이 삭제된 외형 기록은 검색 목록에 남지만 `crop_url`은 `null`로 반환됩니다.

## 대시보드 연동 권장 순서

대시보드 초기 로딩 시:

1. `GET /api/v1/health`로 Public API 상태 확인
2. `GET /api/v1/cameras`로 카메라 목록과 zone 표시
3. `GET /api/v1/sites`로 사이트/카메라/알람 장치 매핑 표시
4. `GET /api/v1/control/mode`로 현재 제어 모드 표시
5. `GET /api/v1/events?limit=50`로 최근 이벤트 표시
6. `GET /api/v1/appearances/status`로 외형 검색 UI 활성 여부 결정
7. 필요하면 `GET /api/v1/control/devices`와 `GET /api/v1/control/action-events`로 장비 설정 및 최근 실행 결과 표시

수동 승인 화면:

1. `GET /api/v1/control/pending` 주기적 조회
2. 사용자가 승인하면 `POST /api/v1/control/approve/{event_id}`
3. 사용자가 거부하면 `POST /api/v1/control/reject/{event_id}`

외형 검색 화면:

1. `GET /api/v1/appearances/status`
2. 준비 상태가 `ready=true`이면 필터 UI 활성화
3. `GET /api/v1/search?...`로 검색
4. `crop_url`이 있으면 `GET /api/v1/search/crops/{filename}`로 이미지 표시

운영 검수와 센서 화면:

1. `POST /api/v1/event-reviews`로 이벤트의 정탐·오탐·애매함 판정 저장
2. `GET /api/v1/event-reviews/summary`로 누적 검수 결과 조회
3. `GET /api/v1/sensor-readings`로 최신 센서 입력 조회
4. 시연 입력이 필요할 때만 `POST /api/v1/sensor-readings` 사용

## 운영 체크리스트

- `PUBLIC_API_KEY`를 반드시 설정합니다.
- 외부에 노출되는 환경에서는 `CORS_ORIGINS`를 실제 프론트 도메인으로 제한합니다.
- `ACTION_LAYER_URL`, `ALERT_API_URL`이 Docker 네트워크/호스트 환경에 맞는지 확인합니다.
- `ALERT_LOG_PATH`와 `APPEARANCES_DB`가 컨테이너 볼륨으로 유지되는지 확인합니다.
- Prometheus가 `/api/v1/metrics`를 scrape하는지 확인합니다.
- 대시보드에서는 `GET /api/v1/health`만 보고 전체 시스템 정상으로 판단하지 말고, 필요 시 Action Layer와 Alert API health도 함께 확인합니다.

## 관련 문서

- [../modules/PROJECT_STRUCTURE.md](../modules/PROJECT_STRUCTURE.md): 전체 프로젝트 구조와 데이터 흐름
- [PUBLIC_API_EXAMPLES.md](PUBLIC_API_EXAMPLES.md): Public API 복붙용 요청 예시
- [APPEARANCES_STATUS_API.md](APPEARANCES_STATUS_API.md): 외형 검색 상태 API 상세 계약
- [../modules/ACTION_LAYER_SPEAKER_BRIDGE.md](../modules/ACTION_LAYER_SPEAKER_BRIDGE.md): Action Layer와 알람 장치 연동
- [../modules/KUIPER_RULE_ENGINE.md](../modules/KUIPER_RULE_ENGINE.md): eKuiper 룰 엔진 구성
