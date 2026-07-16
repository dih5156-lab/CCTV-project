# EdgeX 기반 Jetson 양방향 AIoT 설계

## 1. 결론

Jetson과 서버 사이의 AIoT 통신을 단순 이벤트 업로드가 아닌 양방향 명령·이벤트 순환 구조로 만든다. EdgeX는 명령, 상태, 경량 이벤트의 제어 경로를 담당하고 이미지·영상 바이트는 서버가 요청한 경우에만 별도 데이터 경로로 업로드한다.

기존 DeepStream 추론 경로, MQTT 이벤트 경로, EdgeX Adapter, CanonicalEvent, Outbox를 유지·확장한다. 기존 경로를 즉시 교체하지 않고 Shadow 방식으로 검증한 뒤 카메라별로 점진 확대한다.

## 2. 목표와 범위

### 목표

- 서버가 EdgeX를 통해 Jetson에 실시간·과거 AI 검색을 요청할 수 있다.
- Jetson은 검색 결과의 경량 메타데이터만 먼저 반환한다.
- 서버가 특정 결과 이미지를 요청하면 Jetson이 해당 파일만 요청별 임시 URL로 업로드한다.
- EdgeX 또는 서버가 일시 중단되어도 명령 상태와 결과 이벤트를 잃지 않는다.
- 검색과 업로드가 DeepStream 실시간 추론 안정성을 침해하지 않는다.
- 기존 MQTT/Kuiper 소비자와 이벤트 호환성을 유지한다.

### 포함 범위

- Jetson `docker-compose.jetson.yml` 운영 환경
- CCTV 입력, DeepStream 추론, AI 이벤트 정규화
- EdgeX Command/MessageBus 기반 요청·응답
- `live | history | both` 검색 모드
- 요청별 미디어 업로드
- Command Inbox와 결과 Outbox
- Shadow 계측과 점진 배포

### 제외 범위

- 모든 runner를 하나의 프로세스로 통합하는 작업
- 이미지·영상을 Base64로 EdgeX 이벤트에 포함하는 방식
- 전체 DeepStream 파이프라인 재작성
- Shadow 검증 전 기존 MQTT 이벤트 경로 제거
- 서버 검색 UI 구현

## 3. 설계 원칙

1. EdgeX는 제어 경로이고 대용량 미디어 저장소가 아니다.
2. Jetson은 원본·크롭·클립과 상세 AI 메타데이터를 로컬에 유지한다.
3. 서버는 필요한 결과와 미디어만 요청한다.
4. 모든 비동기 작업은 `request_id`로 상관관계를 유지한다.
5. 재전송해도 같은 `event_id` 또는 `request_id`를 유지한다.
6. 실시간 추론이 검색과 미디어 업로드보다 높은 우선순위를 가진다.
7. 기존 구현을 재사용하고 새로운 계층은 명확한 경계에만 추가한다.

## 4. 전체 아키텍처

```text
Camera / RTSP / Sensor
        |
        v
DeepStreamProcessor + AI 분석기
        |
        v
CanonicalEvent Boundary
        |
        v
EventDispatcher / Publish Loop
   |              |               |
   v              v               v
Legacy MQTT    EdgeX Adapter    Local Media Store
   |              |               |
   |              +---- Result Outbox
   |                              |
   +--------- Shadow Metrics -----+

Server
   |
   | ai_query_request / fetch_media_request
   v
EdgeX Command / MessageBus
   |
   v
Jetson AI Query Command Handler
   |-- Live Search
   |-- AppearanceLog History Search
   |-- On-demand Media Uploader
   +-- Command Inbox / Result Outbox
```

### 기존 코드 재사용 대상

- `src/canonical_event.py`: 레거시 이벤트 보강과 CanonicalEvent 생성
- `src/core/event_dispatcher.py`: 이벤트 큐잉과 계측
- `src/core/_event_publish.py`: 정규화 및 publish loop
- `src/protocols/mqtt_publisher.py`: 기존 MQTT 기준 경로
- `src/edgex/adapter_service.py`: EdgeX Adapter 실행 경계
- `src/edgex/_publisher_mixin.py`: EdgeX 장치 이벤트 게시
- `src/edgex/_outbox_mixin.py`: EdgeX 결과 영속화와 재전송
- `src/services/action_bridge.py`: 명령 dispatch/result 패턴
- `src/services/appearance_log.py`: 과거 외형 이벤트 검색
- `src/core/ai/_appearance_pipeline.py`: 실시간 속성 매칭

새 AI 검색 명령을 물리 장치 동작과 혼합하지 않는다. `ActionBridge`의 명령 상태·결과 패턴은 재사용하되 AI 검색 실행기는 별도 책임으로 둔다.

## 5. CanonicalEvent와 EdgeX Projection

### Jetson 내부 이벤트

기존 `schema_version: "1.0"`과 레거시 top-level 필드를 유지한다. 다음 값은 반드시 보장한다.

- `event_id`: 재전송해도 변하지 않는 이벤트 식별자
- `message_type`: 이벤트 종류
- `occurred_at`: 실제 발생 시각을 나타내는 ISO-8601 UTC 값
- `device.camera_id`, `device.device_type`
- `event.event_type`, `event.source`
- 선택 필드: `event.confidence`, `event.severity`
- 선택 필드: `media.snapshot_url`, `media.clip_url`, `media.expires_at`

`raw`에는 bbox, keypoints, 모델별 상세 메타데이터를 유지할 수 있다. 이 필드는 EdgeX 경량 이벤트로 그대로 전달하지 않는다.

### EdgeX 경량 Projection

EdgeX Adapter가 CanonicalEvent에서 다음 필드만 추출한다.

- `event_id`, `schema_version`
- `type`, `resource`
- `device`, `device_type`
- `confidence`, `severity`, `occurred_at`
- 필요한 경우 `snapshot_url` 또는 미디어 존재 여부

EdgeX에는 Base64 이미지, 영상, 전체 keypoints, 대형 bbox 목록, 모델 디버그 값을 넣지 않는다. EdgeX `resource` 이름은 허용된 매핑표를 통해 생성한다.

## 6. 양방향 메시지 계약

### `ai_query_request`

서버가 EdgeX를 통해 Jetson에 검색을 요청한다.

필수 필드:

- `schema_version`
- `message_type: "ai_query_request"`
- `request_id`
- `target.jetson_id`
- `search_mode: "live" | "history" | "both"`
- `expires_at`

선택 필드:

- `target.camera_ids`
- `query_text`
- `filters`: `gender`, `bag_type`, `bag_color` 등 구조화된 조건
- `time_range.from`, `time_range.to`
- `limit`

서버는 자연어와 구조화 필터를 함께 보낼 수 있다. Jetson은 지원하는 구조화 필터를 우선 적용하고, 현재 `AppearanceLog`의 자연어 검색 호환 경로를 보조적으로 사용한다. 알 수 없는 필드는 무시하지 않고 명시적인 검증 오류로 반환한다.

### 검색 상태·결과 이벤트

검색은 비동기로 처리하며 다음 상태를 게시한다.

- `accepted`: 스키마와 만료 검증 후 접수
- `running`: 검색 수행 중
- `completed`: 결과 생성 완료
- `failed`: 실행 오류
- `expired`: 명령 만료
- `rate_limited`: Jetson 보호 정책으로 실행 거절 또는 지연

모든 상태는 원본 `request_id`를 포함한다. 결과 항목은 `match_id`, 카메라, 발생 시각, 신뢰도, 검색된 속성, 로컬 미디어 존재 여부만 포함한다. 원본 이미지 바이트는 포함하지 않는다.

### `fetch_media_request`

서버가 검색 결과 중 필요한 파일만 요청한다.

필수 필드:

- `request_id`
- `parent_request_id`
- `match_ids`
- `media_kind: "snapshot" | "clip"`
- 요청별 `upload_url`
- `expires_at`
- `max_bytes`

Jetson은 URL 만료, HTTPS, 허용된 호스트, 파일 종류와 최대 용량을 검증한다. 선택되지 않은 파일은 업로드하지 않는다. 업로드 완료 결과에는 checksum, 실제 크기, 상태를 포함하되 임시 URL 전체를 로그나 결과 이벤트에 기록하지 않는다.

## 7. 신뢰성과 장애 처리

### Command Inbox

수신한 `request_id`, 명령 종류, 상태, 만료 시각, 마지막 결과 참조를 영속 저장한다. 동일한 `request_id`가 재수신되면 검색이나 업로드를 다시 수행하지 않고 마지막 상태 또는 결과를 다시 게시한다.

### Result Outbox

EdgeX 전송 실패 시 상태·결과 이벤트를 기존 영속 Outbox 패턴으로 저장한다. 재전송은 지수 backoff와 최대 간격을 적용한다. `request_id + message_type + sequence`를 멱등 키로 사용한다.

### 업로드 URL 만료

만료된 URL은 재사용하지 않는다. 업로드 중 URL이 만료되면 `upload_url_expired` 상태를 반환하고 서버가 동일 `match_id`에 새 URL을 발급하도록 한다.

### Jetson 부하 보호

- 동시 검색 수 제한
- 요청별 결과 수 제한
- 업로드 동시성 및 대역폭 제한
- 검색 시간 제한
- DeepStream FPS, frame drop, GPU/CPU/메모리 임계치 기반 admission control
- 낙상·침입 등 실시간 중요 이벤트가 서버 검색 요청보다 높은 우선순위

## 8. Shadow 전환

### 단계 0: 기준선 계측

기존 MQTT 이벤트의 수량, 지연, 누락, 중복, DeepStream FPS를 수집한다.

### 단계 1: Mirror

기존 이벤트를 유지하면서 경량 EdgeX 이벤트를 병행 게시한다. 신규 결과가 외부 알림이나 장치 동작을 발생시키지 않도록 한다.

### 단계 2: Query Pilot

허용 목록에 등록된 Jetson과 카메라에서 `live | history | both` 검색과 경량 결과 반환만 활성화한다.

### 단계 3: Media Pilot

요청별 임시 URL 업로드를 활성화한다. 카메라 1대와 제한된 파일 크기로 시작해 점진 확대한다.

기존 MQTT 경로 제거는 이 설계의 초기 범위에 포함하지 않는다. Shadow 결과가 안정적이어도 별도 운영 승인 후 결정한다.

## 9. 테스트 전략

### 계약 테스트

- 각 메시지의 필수 필드, 버전, 만료 검증
- 알 수 없는 `message_type`, `search_mode`, 필터 거절
- CanonicalEvent 1.0과 레거시 MQTT 필드 호환

### 컴포넌트 테스트

- `live`, `history`, `both` 검색
- `request_id` 중복 억제
- 결과 개수와 시간 범위 제한
- `match_id`와 로컬 미디어 참조 연결
- 업로드 URL의 scheme, host, 만료, 최대 크기 검증

### 장애 통합 테스트

- EdgeX/MQTT 중단 후 Outbox 재전송
- Jetson 프로세스 재시작 후 Inbox/Outbox 복구
- 업로드 실패와 URL 만료
- 동일 명령 반복 전달
- Outbox 용량과 TTL 정리

### Jetson 운영 테스트

- DeepStream과 검색·업로드 동시 실행
- FPS, frame drop, GPU/CPU, 메모리, 디스크, 네트워크 측정
- 샘플 영상 replay 후 카메라 1대 Shadow 검증

## 10. 관측성

필수 메트릭:

- `aiot_commands_received_total`
- `aiot_commands_duplicate_total`
- `aiot_commands_rejected_total`
- `aiot_query_duration_seconds`
- `aiot_query_matches_total`
- `aiot_query_inflight`
- `aiot_result_outbox_pending`
- `aiot_result_retry_total`
- `aiot_result_delivery_seconds`
- `aiot_media_upload_bytes_total`
- `aiot_media_upload_failures_total`
- `aiot_media_url_expired_total`
- `shadow_event_missing_total`
- `shadow_event_duplicate_total`
- `shadow_latency_delta_seconds`
- 기존 `deepstream_fps`, frame drop, GPU 사용률

모든 관련 로그에는 `request_id`, `parent_request_id`, `event_id`, `jetson_id`, `camera_id`를 포함한다. 이미지 내용, 임시 URL 전체, 인증정보는 로그에 기록하지 않는다.

## 11. 승인과 롤백 기준

Pilot 승격 조건:

- 검색 명령과 결과 이벤트 유실 0건
- 중복 `request_id`에 대한 검색 재실행 0건
- EdgeX 복구 후 Outbox 적체 정상 해소
- 선택한 `match_id` 이외 미디어 업로드 0건
- DeepStream FPS 저하가 구현 계획에서 정한 허용 임계치 이내

즉시 롤백 조건:

- 낙상·침입 등 중요 이벤트 누락
- Outbox 무제한 증가 또는 디스크 위험
- DeepStream 불안정이나 frame drop 급증
- 허용되지 않은 URL로 미디어 전송 시도
- 서로 다른 `request_id`의 결과가 잘못 연결됨

## 12. 예상 변경 영향

- API/이벤트 계약: 새 `ai_query_request`, 검색 상태·결과, `fetch_media_request` 추가
- DB: Command Inbox 저장 구조 추가 가능성이 높으며, 기존 Outbox는 목적지·멱등 키 요구에 맞춰 최소 확장
- 설정: EdgeX command topic/resource, 허용 업로드 host, 요청/검색/업로드 제한값 추가
- 배포: Jetson Compose에서 기능 플래그와 Shadow 설정 추가
- 운영: Prometheus 메트릭과 상태 점검 항목 추가

정확한 파일 목록, DB 변경 방식, 기본 제한값과 성능 임계치는 구현 계획 단계에서 현재 코드와 배포 설정을 추적한 뒤 확정한다.
