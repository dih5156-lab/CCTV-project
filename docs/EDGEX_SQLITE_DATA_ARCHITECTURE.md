# EdgeX App Service + SQLite 데이터 아키텍처 권장안

## 목적

이 문서는 현재 CCTV 프로젝트에서 `EdgeX Application Service` 와 `SQLite` 를 함께 사용할 때,
어떤 역할을 어디에 두는 것이 맞는지 운영 관점에서 정리한 설계 메모입니다.

핵심 목표:

- 엣지 노드 장애 시 이벤트 유실 최소화
- 재전송 가능한 store-and-forward 구조 확보
- 중복 저장과 중복 알람 방지
- 장기 저장과 운영 저장의 책임 분리
- 나중에 중앙 DB 확장 시 무리 없는 구조 유지

## 결론

권장 방향은 아래와 같습니다.

1. `SQLite` 는 엣지 로컬 영속 계층으로 사용한다.
2. `EdgeX App Service` 는 라우팅, 전송, 재시도, 외부 연계 계층으로 사용한다.
3. 장기 보관과 통합 조회는 나중에 중앙 DB로 분리한다.

즉, `App Service + SQLite` 조합은 맞는 방향이지만,
`SQLite` 를 최종 메인 데이터 플랫폼으로 키우는 것은 권장하지 않습니다.

## 권장 계층 구조

```text
AI Engine / Sensor Ingest
  -> EdgeX Adapter / Device Service
  -> Kuiper / App Service
  -> SQLite (edge-local bronze + outbox + ops log)
  -> External API / MQTT / Alert
  -> Central DB (optional, later)
```

역할 분리:

- AI Engine
  - 원본 탐지 이벤트 생성
  - `event_id`, `camera_id`, `object_id`, `occurred_at` 포함
- EdgeX Adapter / Device Service
  - EdgeX 표준 이벤트 포맷으로 정규화
  - Device/Profile/Resource 정합 유지
- App Service
  - 정책 라우팅
  - 외부 HTTP/MQTT 전송
  - 필요시 재시도
- SQLite
  - 원시 이벤트 보존
  - pending/sent/failed 상태 추적
  - 로컬 검색/복구용 로그 저장
- Central DB
  - 장기 저장
  - 통합 조회
  - 리포트/통계/모델 평가 데이터

## 데이터 레이어 권장안

### 1. Edge Bronze

SQLite 에 append 위주로 저장하는 원시 이벤트 계층입니다.

보관 대상:

- 원본 MQTT/HTTP payload
- EdgeX 정규화 payload
- 전송 대상 payload
- 이벤트 수신 시각, 처리 시각, 상태

특징:

- 최소 변환
- 재처리 가능
- 재전송 가능
- 장애 복구 기준 데이터

현재 프로젝트에서 가까운 예:

- `src/services/external_ingest.py`
- `src/edgex/_outbox_mixin.py`
- `parser-python/database/edgex_outbox.py`

### 2. Edge Silver

운영에 필요한 정제 계층입니다.

예시:

- appearance search table
- dedupe 된 intrusion event log
- 알람 이력
- camera health event

특징:

- 중복 제거
- 상태 컬럼 정리
- 필수 조회 인덱스 구성
- 운영 API 검색 최적화

현재 프로젝트에서 가까운 예:

- `src/services/appearance_log.py`
- action layer 의 이벤트 저장소

### 3. Central Gold

중앙 DB가 필요해질 때만 도입하는 소비자 계층입니다.

예시:

- 일/주/월 통계
- 카메라별 경보 빈도
- 시간대별 침입 추세
- 모델 품질 평가 리포트

## SQLite 에 저장해야 하는 것과 저장하면 안 되는 것

SQLite 에 저장하기 좋은 것:

- store-and-forward outbox
- 최근 1일~30일 운영 로그
- 현장 장애 복구용 이벤트
- 로컬 API 검색용 테이블
- 외부 전송 성공/실패 상태

SQLite 에 장기적으로 몰아넣지 말아야 할 것:

- 수개월~수년 단위 장기 적재
- 다중 서비스 동시 쓰기 중심 워크로드
- 무거운 집계성 분석
- 다중 사용자 동시 조회가 잦은 운영 포털 메인 DB

## 테이블 설계 권장안

### A. raw_event_ingest

외부 입력과 내부 표준화를 함께 보관하는 원시 이벤트 테이블입니다.

권장 컬럼:

```sql
id INTEGER PRIMARY KEY AUTOINCREMENT
event_id TEXT NOT NULL
source_type TEXT NOT NULL
source_topic TEXT
camera_id TEXT
device_id TEXT
event_type TEXT NOT NULL
occurred_at TEXT NOT NULL
received_at TEXT NOT NULL
raw_payload_json TEXT NOT NULL
normalized_payload_json TEXT NOT NULL
schema_version TEXT NOT NULL DEFAULT 'v1'
```

권장 인덱스:

- `(event_id)` unique
- `(camera_id, occurred_at)`
- `(event_type, occurred_at)`

### B. event_outbox

외부 전송, EdgeX 전송, 재발행 전송 상태를 관리하는 핵심 테이블입니다.

권장 컬럼:

```sql
id INTEGER PRIMARY KEY AUTOINCREMENT
event_id TEXT NOT NULL
destination_type TEXT NOT NULL
destination_name TEXT NOT NULL
payload_json TEXT NOT NULL
status TEXT NOT NULL
retry_count INTEGER NOT NULL DEFAULT 0
created_at TEXT NOT NULL
last_attempt_at TEXT
sent_at TEXT
expire_at TEXT
last_error TEXT
```

`status` 값:

- `pending`
- `sent`
- `failed`
- `expired`

권장 인덱스:

- `(status, id)`
- `(event_id, destination_name)` unique
- `(created_at)`

### C. operations_event_log

운영 검색용 정제 테이블입니다.

권장 컬럼:

```sql
id INTEGER PRIMARY KEY AUTOINCREMENT
event_id TEXT NOT NULL
camera_id TEXT NOT NULL
event_type TEXT NOT NULL
severity TEXT
object_id INTEGER
confidence REAL
occurred_at TEXT NOT NULL
message TEXT
metadata_json TEXT
```

권장 인덱스:

- `(camera_id, occurred_at DESC)`
- `(event_type, occurred_at DESC)`
- `(severity, occurred_at DESC)`

### D. appearance_log

현재처럼 별도 검색 목적이 있으면 유지하는 것이 맞습니다.

추가 권장 사항:

- `event_id` 컬럼 추가
- `schema_version` 컬럼 추가
- `attribute_backend` 컬럼 추가

이유:

- 중복 기록 추적
- 속성 모델 교체 이력 추적
- HSV / PP-Human 결과 비교 가능

## replay-safe 저장 전략

핵심 규칙은 아래 4개입니다.

### 1. event_id 를 반드시 만든다

가능하면 이벤트 생성 시점에 `event_id` 를 고정 생성합니다.

권장 구성:

```text
{camera_id}:{event_type}:{object_id}:{occurred_at_ms}
```

더 안전하게 가려면 payload 일부를 포함한 hash 를 추가합니다.

### 2. insert 는 append, 상태는 별도 테이블 또는 상태 컬럼으로 관리한다

원시 이벤트를 update 위주로 관리하면 재처리 시점에 원본 보존성이 약해집니다.

권장 방식:

- raw ingest 는 append-only
- outbox 는 상태 갱신 허용
- search table 은 upsert 가능

### 3. dedupe key 를 명시한다

중복 저장 방지 규칙은 코드에 암묵적으로 두지 말고 테이블 제약으로도 둡니다.

예:

```sql
CREATE UNIQUE INDEX uq_event_outbox_event_dest
ON event_outbox(event_id, destination_name);
```

### 4. TTL 과 archive 기준을 둔다

SQLite 는 보존정책 없이 계속 쌓으면 결국 운영 문제가 납니다.

권장 예:

- raw ingest: 7~30일
- outbox sent: 3~7일
- failed: 운영 확인 전까지 또는 30일
- appearance/op log: 30~90일

## App Service 에 기대할 역할

App Service 는 아래 역할에 집중하는 것이 좋습니다.

- 입력 이벤트 수신
- 라우팅
- 정책별 분기
- 외부 HTTP Export
- MQTT 재발행
- 재시도/실패 추적

App Service 에 과도하게 넣지 말아야 할 것:

- 장기 분석 로직
- 복잡한 join 기반 조회
- 대시보드 메인 DB 역할
- 운영 포털용 임의 검색 전부

## 중앙 DB 가 필요한 시점

아래 중 3개 이상 해당하면 SQLite 단독 운영을 넘길 시점입니다.

- 카메라 수가 빠르게 늘어남
- 다중 서비스가 같은 이벤트를 함께 참조함
- 월 단위 이상 보관이 필요함
- 분석 리포트 요구가 생김
- 운영자가 웹에서 자유 검색을 많이 함
- 이벤트를 다른 시스템과 정식 연동해야 함

그 시점의 권장안:

- Edge SQLite 는 유지
- 중앙 PostgreSQL 추가
- Edge -> Central 비동기 적재 추가

## 현재 프로젝트 기준 추천 구현 순서

1. 현재 SQLite 사용처를 `raw / outbox / search` 3종으로 명확히 나눈다.
2. 모든 저장 경로에 `event_id` 를 넣는다.
3. outbox 테이블에 unique dedupe key 를 넣는다.
4. App Service 는 라우팅과 재전송만 담당하게 유지한다.
5. 운영 API 검색은 SQLite search table 기준으로 제공한다.
6. 장기 저장 요구가 생기면 PostgreSQL sink 를 추가한다.

## 지금 바로 권장하는 실제 변경 포인트

### 우선순위 1

- `external_ingest` 에 `event_id` 추가
- `appearance_log` 에 `event_id` 추가
- action 저장소에도 `event_id` 추가

### 우선순위 2

- outbox 테이블에 `destination_name` + `event_id` unique 제약
- `schema_version` 컬럼 추가
- `expire_at` 또는 TTL 기준 추가

### 우선순위 3

- 중앙 적재용 exporter 인터페이스 추가
- PostgreSQL sink 는 나중에 옵션으로 붙이기

## 최종 판단

`EdgeX Application Service + SQLite` 는 이 프로젝트에서 맞는 방향입니다.

단, 전제는 명확합니다.

- SQLite 는 엣지 로컬 영속 버퍼이자 운영 로그 계층이어야 합니다.
- App Service 는 데이터 플랫폼이 아니라 라우팅/전송 계층이어야 합니다.
- 장기 분석과 통합 조회는 나중에 중앙 DB로 분리해야 합니다.

이 원칙만 지키면 지금 방향은 안정적이고 확장성도 좋습니다.
