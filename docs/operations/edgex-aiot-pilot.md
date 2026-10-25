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

### 지표 수집과 명령 기능의 구분

2026-10-16 I64 조사에서 운영 어댑터는 `healthy`였지만, 명령 기능이 꺼지면 지표 서버도 시작하지 않아 Prometheus 수집은 `down`이었다. 지표 서버를 명령 기능과 분리한 수정 후보를 로컬·격리 환경에서 검증했다. 운영에는 아직 반영하지 않았으므로 수집 복구 완료로 해석하지 않는다.

수정 후보에서는 `AIOT_COMMANDS_ENABLED=false`를 유지해도 `/metrics`를 제공하며, `cctv_edgex_adapter_aiot_commands_enabled`가 `0`이면 명령 기능 비활성, `1`이면 활성이다. `up{job="cctv-edgex-adapter"}`는 수집 가능 여부이며 MQTT 접속·이벤트 전달 성공을 보장하지 않는다. 프로세스 지표와 명령 기능 상태는 별도로 확인한다.

수집 실패를 해결하기 위해 명령 기능을 켜지 않는다. 운영 반영 시에는 수정한 실행 파일의 이미지 포함 여부, 기존 내부 포트9105와 수집 대상의 일치, `up=1` 및 명령 상태 `0`, 기존 MQTT 이벤트 전달·Outbox를 확인한다. 사용자 지정 포트는 현재 Compose의 내부 포트·Prometheus 대상도 함께 맞춰야 하며, 이번 수정은 포트 설정을 바꾸지 않는다. [원인·수정·검증 근거](../planning/evidence/adapter_metrics_20261016.json)

2026-10-16 배포 준비에서는 기존 이미지 `b146c351`에 검증한 runner 1개만 반영한 후보 `ab366dd3`를 만들었다. 후보 이미지 자체의 격리 HTTP 시험에서 200·명령 상태 0을 확인했고, 후보/복귀 Compose의 환경·명령·마운트·포트가 운영과 같음을 대조했다. 실행안과 해시는 프로젝트 내부 `data/runtime/p63_adapter_metrics_20261016/release/manifest.json`, `commands.json`에 보관한다. 아직 전환 명령은 실행하지 않았다.

Outbox와 명령 DB는 온라인 스냅샷을 만들어 복사본을 읽기 전용으로 다시 열고 `quick_check=ok`를 확인했다. 원본 삭제·스키마 변경·DB 덮어쓰기는 없다. 백업은 접근 제한 폴더에 보관하며 공개 진행판에 DB 내용을 게시하지 않는다. 현재 부팅 유닛은 어댑터를 의존 서비스로 다루므로 후보 적용 때 부팅용 Compose 지정도 필요하다. `20-i64-adapter.conf.draft`는 문법 검사를 마친 초안이며 설치하지 않았다.

실제 적용 전 현재 구성 해시·이벤트 유입 및 복구 시간을 다시 확인하고, 적용 후 `up=1`·명령 상태 0·MQTT/Outbox 전달·실제 알림을 확인한다. 복귀 프로필은 구성 대조까지 완료했으나 복귀 모의 실행은 자동 승인 검토가 차단했다. 실제 전환·복귀·재부팅과 장기 시계열은 아직 검증하지 않았다. DB 이관이 없는 이번 변경의 이미지 복귀를 위해 원본 DB를 과거 백업으로 덮어쓰지 않는다.

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
