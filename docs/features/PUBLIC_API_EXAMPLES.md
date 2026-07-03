# Public API 샘플 요청 모음

## 결론

이 문서는 Public API를 빠르게 테스트하기 위한 복붙용 `curl` 예시 모음입니다.

기본 전제:

```bash
export API_BASE="http://localhost:9000"
export PUBLIC_API_KEY="change-me"
```

운영 환경에서는 실제 API key를 사용하고, 개발 환경에서 `PUBLIC_API_KEY`를 설정하지 않은 상태라면 `-H "X-API-Key: ..."` 헤더 없이도 호출될 수 있습니다.

## 상태 확인

### Public API root 안내

```bash
curl -sS "${API_BASE}/"
```

예상 응답:

```json
{
  "service": "cctv-public-api",
  "description": "CCTV Platform Public API",
  "docs": "/docs",
  "health": "/api/v1/health",
  "events": "/api/v1/events",
  "sensor_readings": "/api/v1/sensor-readings",
  "cameras": "/api/v1/cameras",
  "sites": "/api/v1/sites",
  "search": "/api/v1/search"
}
```

### Public API health

```bash
curl -sS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/health"
```

예상 응답:

```json
{
  "success": true,
  "data": {
    "status": "up",
    "service": "cctv-public-api",
    "version": "1.0.0",
    "checked_at": "2026-05-04T00:00:00.000000+00:00",
    "action_layer_url": "http://cctv-action-layer:8080",
    "alert_api_url": "http://cctv-alert-api:8000"
  },
  "error": null,
  "timestamp": "2026-05-04T00:00:00.000000+00:00"
}
```

### Public API readiness

```bash
curl -sS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/readiness"
```

정상일 때는 `status=ready`, 하위 서비스 연결에 문제가 있으면 HTTP 503과 `status=degraded`가 나옵니다.

### Prometheus metrics

```bash
curl -sS "${API_BASE}/api/v1/metrics" | head
```

요청 카운터 확인:

```bash
curl -sS "${API_BASE}/api/v1/metrics" | grep cctv_public_api_http_requests_total
```

## 이벤트 조회

### 최신 이벤트 20건

```bash
curl -sS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/events?limit=20&offset=0"
```

### 특정 카메라 이벤트

```bash
curl -sS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/events?camera_id=cam-01&limit=20"
```

### 낙상 이벤트만 조회

```bash
curl -sS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/events?event_type=fall_detected&limit=20"
```

### 시간 범위 필터

```bash
curl -sS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/events?time_from=1777891200&time_to=1777894800&limit=50"
```

## 이벤트 수신 테스트

### 낙상 이벤트 push

```bash
curl -sS -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/alerts" \
  -d '{
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
    "metadata": {
      "zone_id": "zone-a",
      "source": "manual-api-test"
    }
  }'
```

예상 응답:

```json
{
  "success": true,
  "data": {
    "accepted": true,
    "event_type": "fall_detected",
    "camera_id": "cam-01"
  },
  "error": null,
  "timestamp": "2026-05-04T00:00:00.000000+00:00"
}
```

### 헬멧 미착용 이벤트 push

```bash
curl -sS -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/alerts" \
  -d '{
    "camera_id": "cam-01",
    "event_type": "head",
    "severity": "normal",
    "confidence": 0.87,
    "timestamp": 1777891200.0,
    "bbox": {
      "x": 250,
      "y": 80,
      "width": 48,
      "height": 54
    },
    "object_id": 21,
    "metadata": {
      "source": "manual-api-test"
    }
  }'
```

## 카메라 조회

### 카메라 목록

```bash
curl -sS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/cameras"
```

예상 응답:

```json
{
  "success": true,
  "data": [
    {
      "id": "cam-01",
      "name": "현장 1번 카메라",
      "url": "rtsp://192.168.0.10",
      "zones": []
    }
  ],
  "error": null,
  "timestamp": "2026-05-04T00:00:00.000000+00:00"
}
```

### 카메라 단건

```bash
curl -sS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/cameras/cam-01"
```

## 사이트 관리

사이트 API는 내부 Action Layer가 떠 있어야 정상 동작합니다.

### 사이트 목록

```bash
curl -sS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/sites"
```

### 사이트 등록

```bash
curl -sS -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/sites" \
  -d '{
    "site_id": "site-01",
    "site_name": "A 공장",
    "site_nickname": "1라인",
    "camera_ids": ["cam-01", "cam-02"],
    "control_mode": "auto",
    "alarm_devices": ["speaker", "signboard"]
  }'
```

### 사이트 삭제

```bash
curl -sS -X DELETE \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/sites/site-01"
```

## 제어 모드와 수동 승인

### 현재 제어 모드

```bash
curl -sS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/control/mode"
```

### 전체 수동 모드 전환

```bash
curl -sS -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/control/mode" \
  -d '{
    "mode": "manual"
  }'
```

### 특정 사이트 자동 모드 전환

```bash
curl -sS -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/control/mode" \
  -d '{
    "mode": "auto",
    "site_id": "site-01"
  }'
```

### 승인 대기 이벤트 조회

```bash
curl -sS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/control/pending"
```

### 이벤트 승인

```bash
curl -sS -X POST \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/control/approve/event-123"
```

### 이벤트 거부

```bash
curl -sS -X POST \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/control/reject/event-123"
```

## 외형 조건

### 외형 검색 준비 상태

```bash
curl -sS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/appearances/status"
```

### 외형 조건 목록

```bash
curl -sS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/appearances"
```

### 외형 조건 등록

```bash
curl -sS -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/appearances" \
  -d '{
    "name": "검은 상의 + 헬멧 착용",
    "upper_color": "black",
    "has_helmet": true,
    "threshold": 0.8,
    "cameras": ["cam-01"],
    "enabled": true
  }'
```

### 외형 조건 삭제

```bash
curl -sS -X DELETE \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/appearances/condition-id"
```

## 외형 기록 검색

### 검은 상의 기록 검색

```bash
curl -sS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/search?upper_color=black&limit=20"
```

### 헬멧 착용 기록 검색

```bash
curl -sS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/search?has_helmet=true&limit=20"
```

### 시간 범위 외형 검색

```bash
curl -sS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/search?time_from=2026-05-04T09:00:00&time_to=2026-05-04T18:00:00&limit=50"
```

### crop 이미지 다운로드 확인

검색 응답의 `crop_url` 값을 사용합니다.
보존 기간이 지난 기록은 `crop_url=null`일 수 있으므로 값이 있는 경우에만 이미지를 요청합니다.

```bash
curl -sS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/search/crops/cam-01_12.jpg" \
  --output /tmp/cctv-crop.jpg
```

## 출력 장비와 Action Layer 이력

```bash
curl -sS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/control/devices"

curl -sS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/control/action-events?limit=20"
```

## 이벤트 검수

```bash
curl -sS -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/event-reviews" \
  -d '{
    "event_id": "evt_demo_001",
    "status": "false_positive",
    "reviewer": "operator-01",
    "category": "sitting",
    "note": "의자에 앉는 동작을 낙상으로 판정"
  }'

curl -sS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/event-reviews/summary?recent_limit=20"
```

`status`는 `true_positive`, `false_positive`, `uncertain` 중 하나입니다.

## 센서 로그 조회와 시연 입력

```bash
curl -sS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/sensor-readings?limit=20"

curl -sS -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/sensor-readings" \
  -d '{
    "device_id": "sensor-demo-01",
    "temperature": 72.0,
    "angle_x": 0.0,
    "angle_y": 0.0,
    "event_code": 0
  }'
```

`POST /sensor-readings`는 시연용 입력이며 내부 Alert API로 전달됩니다. 운영 센서 데이터는 실제 parser/ingest 경로를 사용합니다.

## 에러 케이스 확인

### API key 누락

`PUBLIC_API_KEY`가 서버에 설정된 상태에서 아래처럼 호출하면 401이 나와야 합니다.

```bash
curl -i -sS "${API_BASE}/api/v1/events"
```

예상 응답:

```json
{
  "success": false,
  "data": null,
  "error": "API Key가 필요합니다. X-API-Key 헤더를 제공하세요.",
  "timestamp": "2026-05-04T00:00:00.000000+00:00"
}
```

### 잘못된 API key

```bash
curl -i -sS \
  -H "X-API-Key: wrong-key" \
  "${API_BASE}/api/v1/events"
```

예상 응답은 403입니다.

### 잘못된 이벤트 payload

`confidence`는 `0.0`에서 `1.0` 사이여야 합니다.

```bash
curl -i -sS -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/alerts" \
  -d '{
    "camera_id": "cam-01",
    "event_type": "fall_detected",
    "severity": "critical",
    "confidence": 2.0,
    "timestamp": 1777891200.0
  }'
```

예상 응답은 422입니다.

### Alert API POST 전용 경로를 브라우저로 열었을 때

Alert API의 `/api/alerts`는 `POST` 전용입니다. 최신 코드에서는 브라우저 `GET`으로 열면 405와 함께 허용 method를 알려줍니다.

```bash
curl -i -sS http://localhost:8000/api/alerts
```

예상 응답:

```json
{
  "error": "method not allowed",
  "path": "/api/alerts",
  "allowed": "POST",
  "hint": "/api/alerts는 POST 요청으로 호출해야 합니다."
}
```

## 빠른 smoke 확인 순서

로컬 서비스들이 떠 있는 상태에서 아래 순서로 보면 됩니다.

```bash
curl -sS "${API_BASE}/api/v1/metrics" >/dev/null

curl -sS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/health"

curl -sS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/cameras"

curl -sS \
  -H "X-API-Key: ${PUBLIC_API_KEY}" \
  "${API_BASE}/api/v1/events?limit=5"
```

자동화된 smoke test는 아래 스크립트를 사용합니다.

```bash
.venv/bin/python scripts/smoke/smoke_test_deployment.py
.venv/bin/python scripts/smoke/smoke_test_data_flow.py
```
