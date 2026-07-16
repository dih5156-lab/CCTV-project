# EdgeX 양방향 AIoT Jetson Pilot

## 결론

기존 MQTT 이벤트 경로를 유지한 상태에서 카메라 1대의 검색과 요청 미디어 업로드만 단계적으로 활성화한다. 오류, 중요 이벤트 누락, Outbox 증가, DeepStream FPS 기준 미달이 발생하면 `AIOT_COMMANDS_ENABLED=false`로 즉시 롤백한다.

## 1. 사전 설정

`.env.jetson`에 다음 값을 설정한다. 업로드 호스트는 서버가 발급하는 HTTPS 임시 URL의 정확한 hostname만 쉼표로 구분한다.

```dotenv
AIOT_COMMANDS_ENABLED=false
AIOT_JETSON_ID=jetson-01
AIOT_COMMAND_TOPIC_PREFIX=edgex/commands/cctv
AIOT_ALLOWED_UPLOAD_HOSTS=uploads.example.com
AIOT_QUERY_MAX_RESULTS=20
AIOT_LIVE_WINDOW_SECONDS=30
AIOT_METRICS_PORT=9105
```

Compose 설정을 먼저 검증한다.

```bash
rtk docker compose --env-file .env.jetson -f docker-compose.jetson.yml config -q
```

## 2. 단계 0: 기준선

기능을 끈 상태에서 운영 점검과 최소 30분 DeepStream 기준선을 수집한다.

```bash
rtk ./scripts/ops/run_operation_check.sh --with-deepstream 30 30
```

기준선으로 FPS, frame drop, 중요 이벤트 수, MQTT 전달 지연을 기록한다. Pilot 통과 FPS 하한은 기준선의 90%다.

## 3. 단계 1: 명령 경로 활성화

`.env.jetson`에서 `AIOT_COMMANDS_ENABLED=true`로 변경하고 Adapter만 재생성한다.

```bash
rtk docker compose --env-file .env.jetson -f docker-compose.jetson.yml up -d --no-deps --force-recreate cctv-edgex-adapter
rtk docker logs cctv-edgex-adapter
```

로그에서 `AIoT 명령 구독 시작`과 Jetson 전용 topic을 확인한다. 기존 AI 이벤트와 알림 경로는 변경하지 않는다.

## 4. 단계 2: Query Pilot

서버에서 `history`, `live`, `both` 요청을 각각 한 번씩 발행한다. 모든 요청은 고유 `request_id`, 5분 이내 `expires_at`, 카메라 1대, `limit <= 20`을 사용한다.

예시 payload:

```json
{
  "schema_version": "1.0",
  "message_type": "ai_query_request",
  "request_id": "pilot-query-001",
  "target": {"jetson_id": "jetson-01", "camera_ids": ["camera-1"]},
  "search_mode": "both",
  "filters": {"gender": "female", "has_handbag": true, "upper_color": "red"},
  "limit": 10,
  "expires_at": "2099-01-01T00:00:00Z"
}
```

확인 기준:

- 상태 순서가 `accepted → running → completed`다.
- 동일 `request_id` 재발행 시 검색을 다시 실행하지 않고 저장 결과를 반환한다.
- 결과에는 `match_id`가 있지만 절대 crop 경로나 이미지 바이트는 없다.

## 5. 단계 3: Media Pilot

검색 결과 하나에 대해 요청별 HTTPS PUT URL 하나를 발급한다. 요청 하나에는 `match_id`를 정확히 하나만 넣는다.

```json
{
  "schema_version": "1.0",
  "message_type": "fetch_media_request",
  "request_id": "pilot-media-001",
  "parent_request_id": "pilot-query-001",
  "match_ids": ["event-1"],
  "media_kind": "snapshot",
  "upload_url": "https://uploads.example.com/presigned-object",
  "max_bytes": 5242880,
  "expires_at": "2099-01-01T00:00:00Z"
}
```

업로드 완료 결과의 `sha256`과 바이트 수를 서버 객체와 비교한다. URL 전체와 인증 query는 로그에 남지 않아야 한다.

## 6. 장애 및 성능 점검

```bash
AIOT_PILOT_CHECK=1 rtk ./scripts/ops/run_operation_check.sh --with-deepstream 30 30
```

EdgeX broker를 계획된 점검 시간에만 일시 중단했다 복구하여 결과 Outbox가 적체 후 감소하는지 확인한다. Pilot 중 FPS는 기준선의 90% 이상이어야 하며 중요 이벤트 누락은 0건이어야 한다.

## 7. 즉시 롤백

다음 중 하나라도 발생하면 롤백한다.

- 낙상·침입 등 중요 이벤트 누락
- Outbox 지속 증가 또는 디스크 위험
- DeepStream FPS가 기준선의 90% 미만
- 허용하지 않은 host로 업로드 시도
- 서로 다른 `request_id` 결과가 섞임

```bash
rtk sed -i 's/^AIOT_COMMANDS_ENABLED=.*/AIOT_COMMANDS_ENABLED=false/' .env.jetson
rtk docker compose --env-file .env.jetson -f docker-compose.jetson.yml up -d --no-deps --force-recreate cctv-edgex-adapter
```

롤백은 AIoT 명령 subscriber만 비활성화하며 기존 MQTT/Kuiper/알림 경로는 유지한다.
