# Public API 빠른 참조

기본 주소는 `http://<호스트>:9000`이며 모든 경로는 `/api/v1` 아래에 있습니다. 운영 환경에서는 `X-API-Key` 헤더가 필요합니다.

## 공통 규칙

### 인증

```http
X-API-Key: <PUBLIC_API_KEY>
```

### 목록 응답

```json
{
  "items": [],
  "total": 0,
  "limit": 50,
  "offset": 0
}
```

유효하지 않은 쿼리 값은 `422`, 인증 실패는 `401/403`으로 반환됩니다. 상세 스키마는 `/docs`의 OpenAPI 문서를 기준으로 합니다.

## 이벤트 조회

```http
GET /api/v1/events
```

주요 쿼리:

| 파라미터 | 설명 | 예시 |
|---|---|---|
| `limit` | 페이지 크기(1~500) | `50` |
| `offset` | 시작 위치 | `0` |
| `camera_id` | 카메라 필터 | `sample_eval` |
| `event_type` | 이벤트 타입 | `fall_detected` |
| `fall_direction` | `front`, `side`, `back`, `unclassified` | `back` |
| `time_from`, `time_to` | Unix timestamp 범위 | `1770000000` |

예시:

```bash
curl -H "X-API-Key: $PUBLIC_API_KEY" \
  "http://localhost:9000/api/v1/events?event_type=fall_detected&fall_direction=back&limit=20"
```

낙상 이벤트의 `metadata`에는 모델이 제공한 경우 다음 값이 포함됩니다.

```json
{
  "fall_direction": "back",
  "fall_type": "뒤로 넘어짐",
  "scene_cat_name": "후면낙상",
  "fall_detail_status": "classified"
}
```

방향을 확정하지 못하면 `fall_detail_status`는 `unclassified`입니다. 이 상세 값은 DB/웹 조회용이며, 스피커·전광판 출력은 항상 `fall_detected` 통합 문구를 사용합니다.

## 이벤트 수신

```http
POST /api/v1/alerts
Content-Type: application/json
```

```json
{
  "camera_id": "cam01",
  "event_type": "fall_detected",
  "severity": "critical",
  "confidence": 0.86,
  "timestamp": 1770000000.0,
  "metadata": {
    "fall_direction": "back",
    "fall_type": "뒤로 넘어짐"
  }
}
```

`event_type`은 경보 라우팅에 사용되는 통합 타입이고, 상세 유형은 `metadata`에 넣습니다.

## 자주 사용하는 조회 경로

| 기능 | 경로 |
|---|---|
| 상태 확인 | `GET /api/v1/health` |
| 카메라 목록 | `GET /api/v1/cameras` |
| 외형 기록 | `GET /api/v1/appearances` |
| 이벤트 검수 | `POST /api/v1/event-reviews` |
| 센서 조회 | `GET /api/v1/sensor-readings` |
| API 문서 | `GET /docs` |

## 운영 시 주의사항

- API 키·비밀번호·RTSP 주소를 문서나 Git에 기록하지 않습니다.
- `fall_direction`은 방향 분류 모델이 확정한 결과만 사용합니다.
- API 응답의 `metadata`는 선택 필드이므로 값이 없을 수 있습니다.
- 페이지네이션을 사용해 대량 이벤트를 한 번에 조회하지 않습니다.
