# P1.3 — 우선순위·만료·중복 및 물리 실행 계약

기준: 2026-09-22 KST · 계획 v0.28 · **P1.3 DONE: 계약 산출물**

높은 위험의 이벤트부터 **같은 물리 장치의 대기 작업**을 선택한다. 낮은 위험의 기록은 지우지 않는다.
출력 유효시간이 끝난 경보는 만료 사유와 함께 관제에 남기며, 오래된 경보를 복구 과정에서 무조건 재생하지 않는다.
이번 완료는 규칙·상태·예제를 확정한 범위다. 영속 장치 큐와 실행 격리의 구현·배포는 P3.1~P3.3에 남긴다.

## 1. 현재 구현과의 차이

| 현재 코드에서 확인한 동작 | 목표 계약 / 후속 작업 |
|---|---|
| [ActionPolicy](../../src/services/action_policy.py)는 bool로 허용/스킵 반환 | 정책 판정과 이유를 기록. 스킵·대기·만료·중복을 구분 / P3.1 |
| [_AlarmCoordinator](../../src/services/_action_bridge_alarm.py)는 카메라별 메모리 슬롯, 높은 숫자 우선순위 스킵 | 물리 장치별 영속 대기, 고위험 우선·동순위 FIFO. 같은 장치를 공유하는 카메라도 경합 / I02·P3.1 |
| 선점 로그와 슬롯 변경은 장치 중단 완료 확인과 다름 | 중단/교체 능력과 결과가 확인된 장치에서만 선점 / P3.1·P3.3 |
| [_ActionExecutor.execute](../../src/services/_action_bridge_executor.py)는 스피커→전광판→사이렌 처리 후 기존 HTTP 전달/이벤트 저장 | 접수·기록 전달과 장치 worker 분리, 장치별 장애 격리 / I02·P3.2 |
| 한 장치 acknowledged만으로 alarm_played=True, 마지막 이벤트 상태 executed 발행 | 장치별 결과 벡터·집계와 물리 확인 수준을 별도로 표시 / I03·P3.3 |
| [_EventRepo](../../src/services/_action_bridge_repo.py)의 저장 오류는 로그 후 반환 | commit 실패를 호출자에게 전달, 확정되지 않은 명령 실행 금지 / I18·P3.3 |
| [우선순위 계산](../../src/config/event_type_map.py)은 위험 등급 기본값+유형별 값을 더함 | 위험 등급을 먼저 비교하는 정렬 키. 아래 I23은 이번에 재현한 구현 차이 |

**I23 FIXED_LOCAL — 2026-10-13:** 위험 등급 기본값을 critical0 / warning10 / normal20 / info30 / low40으로 분리했다.
유형 순서0~9를 더해도 다음 위험 등급을 넘지 않으며, 겹치거나 역전된 설정은 기본 구간으로 복귀하고 경고를 남긴다.
emergency→critical, warn/error→warning과 공백/대소문자를 정규화한다.
기존 head warning2→10, temperature warning4→12; zone_dwelling critical3은 유지한다.
공개 API `priority`는 계속 정수·작은 값 우선이지만 warning의 구체적인 숫자는 바뀐다. 임의 숫자를 위험 등급으로 하드코딩한 외부 소비자는 전환 전에 대조한다.
현재 정책의 severity 승격/기본값 규칙은 유지하며 이 숫자를 영속 큐의 정책 버전·risk_rank/FIFO로 대신하지 않는다.
실제 ActionPolicy의 warning→critical 허용·반대 차단, 전체 설정 유형 교차, 별칭, 누락/오류 설정 및 조회 경로를 검증했다.
수정 전11 FAIL→관련140 PASS. [과거 재현](evidence/action_policy_priority_gap_20260922.json) · [이번 검증](evidence/tools_priority_20261013.json).
운영과 기존 준비 이미지는 미반영이며 영속 장치 큐·FIFO·물리 선점은 잔여다. 만료·안전 재개의 후속 로컬 구현은 아래 4절을 참조한다.

## 2. 입력과 저장 책임

정책 입력은 공통 저장이 성공한 event 참조와 검증된 장치/구역 설정이다.
정상 관측(예: 온도 측정값·의상 색상)은 경보 조건이 없으면 기록만 전달한다.
미지원 유형, 판정 불가 품질, 대상 미등록은 성공한 장치 실행으로 처리하지 않는다.
최초 접수 저장 실패는 접수 실패이며, 이 계약이 없는 원본을 복원하지는 않는다.

후속 명령 저장소에 필요한 **논리 필드**는 다음과 같다. 이번에 테이블·공개 API 필드를 추가한 것은 아니다.

| 필드 묶음 | 규칙 |
|---|---|
| source_namespace / event_id / ingest_id | 공통 기록 참조. UUID만 단독으로 비교하지 않고 source 범위를 함께 확인 |
| command_id / request_id / device_id | [기존 명령 ID 계약](../architecture/COMMAND_CORRELATION_CONTRACT.md) 유지. command_id=event_id:device, request_id=command_id:device_id |
| site_id / zone_id / resource_key | resource_key는 전역에서 식별 가능한 실제 장치 리소스. 구역은 라우팅 조건이며 카메라별 큐로 같은 장치를 중복 점유하지 않음 |
| policy_version / risk_rank / within_risk_rank | 서버가 검증한 정책 버전과 정렬 값을 명령 생성 시 고정. 외부 payload의 임의 우선순위 숫자를 신뢰하지 않음 |
| accepted_sequence / queued_at | 저장소가 부여한 단조 접수 순서·시각. FIFO는 수신 시각 동률이나 장치 시계에 의존하지 않음 |
| occurred_at / time_quality / expires_at / max_wait_at | 신뢰할 수 있는 사건 시각과 수신 시각 구분. 물리 출력 가능 기한과 장기 대기 관측 기한 구분 |
| payload_sha256 / target_snapshot | 명령 내용과 선택 대상 고정. 정책·레지스트리 변경으로 기존 대기 명령을 몰래 바꾸지 않음 |
| execution_state / reason / attempt_id / lease | 영속 상태와 시도별 이력. 모든 변경은 기존 상태·소유권을 확인한 transaction으로 처리 |
| transport_evidence / device_evidence / physical_evidence | 전송 성공, 장치 응답, 실제 출력 확인을 분리 |

이벤트 보존/관제 outbox 확정과 명령 생성은 복구 가능한 공통 저장 경계로 연결한다.
장치 명령 등록 실패가 이미 확정한 공통 관제 기록을 취소하지 않는다. pending-action 상태/이유를 기록하고 재생성한다.
명령·장치 작업·대기 상태는 같은 저장 책임 아래 원자적으로 확정한 후 장치 worker가 가져간다.

## 3. 장치별 순서와 선점

**10-13 P3.1 영속 물리 큐 로컬 구현:** [검증 근거](evidence/action_queue_20261013.json).
`ACTION_PHYSICAL_QUEUE_ENABLED=true` 또는 `--physical-queue`로 선택한다. 만료 정책·영속 MQTT 입력·EdgeX 모드가 전제이며 기본은 비활성이다.
기존 수신 DB의 `action_device_queue`에 명령/대상/만료 정책과 위험·유형 순서를 고정하고,
`(risk_rank, within_risk_rank, accepted_sequence, request_id)`로 선택한다. sequence는 큐 저장소의 AUTOINCREMENT 값으로 장치 시각·ID 사전순에 의존하지 않는다.
서버의 event_type_map을 사용하며 payload의 priority/risk_rank는 정렬 권한으로 사용하지 않는다. 정책 변경·미등록 유형/위험은 보류한다.

같은 물리 device_id는 카메라·장치 유형을 넘어 하나의 resource_key로 경합한다. active 상태의 partial unique index와 조건부 갱신으로 선점하며,
점유 중인 장치를 건너뛰어 다른 물리 대상은 진행한다. 같은 명령의 여러 대상은 한 트랜잭션으로 큐에 등록한다.
큐 사용 시 카메라 메모리 슬롯의 쿨다운은 대기 경보를 버리는 데 사용하지 않는다. person 등 기존 출력 제외 유형은 유지한다.
큐 대기는 `queued`, 실제 발행 선점부터 응답 제한시간을 시작한다. 입력 처리 complete 이후에만 큐 발행을 허용하며, 중간 종료로 일부만 등록된 입력은 보류한다.

| 큐 상태/입력 | 처리 |
|---|---|
| queued | 위험/유형/FIFO 선택, 선점/발행 직전 만료·대상·모드·정책 재검사 |
| dispatching / sent / acknowledged / uncertain | 물리 리소스 점유. 시각 경과·TTL·더 높은 위험 유입만으로 해제하지 않음 |
| 정상 ACK만 수신 | 명령 ACK는 보존하지만 물리 점유는 유지 |
| 같은 request_id/event_id/device_id의 acknowledged + `execution_completed: true` + `resource_idle: true` | 종료 근거 저장 후 complete·점유 해제. 문자열 true/한 필드만/실패·simulated는 해제 불가 |
| 발행 예외·반환 false·발행 후 저장 실패·재시작 | uncertain 또는 dispatching 선점 보존, 자동 재출력/점유 해제 금지 |
| 미발행 대기의 TTL 경과 | expired로 보존. 이미 실행 중인 장치는 중단됐다고 간주하지 않음 |
| 같은 DB에 미완료 큐가 있는데 큐 옵션 해제 | 기동 거부. 다른 DB/구버전 이미지까지 차단하는 분산 잠금은 아님 |

**10-13 P3.3 후속 로컬 구현:** [검증 근거](evidence/action_completion_20261013.json). 회귀389+실제MQTT9 PASS, 신규36개.
전광판은 원래 경보의 모든 프레임과 이후 유휴 화면의 밝기/제목/본문 응답이 확인된 경우에만 동일 요청을 완료 처리한다.
10-08 실장비 캡처의 정상 응답 `![00500!]`(밝기), `![0000!]`(제목/본문)를 정확히 비교하며, 빈 응답·일반 OK·미등록 프레임·시간 경과만으로는 완료하지 않는다.
기존 통신 ACK 호환 동작은 유지하며 위 긍정 증거가 없는 ACK는 계속 점유한다. 다른 display/clear/power가 개입하면 이전 완료 콜백을 버린다.
완료는 장치 DB의 기존 result_json에 `execution_completed`, `resource_idle`, `completion_evidence=dabit_idle_frame_ack`로 먼저 commit하고 결과를 발행한다.
최초 ACK/완료 저장 실패 시 점유를 풀지 않는다. 완료 저장 후 MQTT 유실/서비스 재시작은 보존된 이력 조회로 복원한다. 완료 저장 전 프로세스 종료는 여전히 미확정이다.

물리 큐 옵션을 켠 Action 작업자는 5초 간격으로 active 요청을 대조한다(로컬 초기값, 현장 SLA 아님).
먼저 Action DB에 저장된 완료 응답을 확인해 큐 갱신 실패를 복구하고, 나머지는 기존 대상 토픽에 `action=request_status`, 빈 payload로 이력만 요청한다.
장치3종 러너는 이 요청을 execute_request에 전달하지 않고 lookup으로 처리한다. HTTP GET 조회도 동일 완료 필드를 반환한다.
정상 ID/이벤트/장치 상관관계·명시적 두 true 필드가 필요하며, 조회 실패/이력 없음/ACK만/모의 실행은 해제 근거가 아니다. 재시도 명령이나 자동 stop을 발행하지 않는다.

전광판 근거는 **컨트롤러의 유휴 화면 수신 응답**이며 실제 LED 가시성·제조사 전체 규격·다른 제어 주체의 외부 출력까지 검증한 것이 아니다.
스피커/사이렌은 현재 실제 종료를 입증할 신호가 없어 ACK만 보존한다. 해당 장치를 큐에 연결하면 종료 확인 전 계속 점유한다.
DB 스키마와 새 의존성은 추가하지 않았다. Action/장치3종의 request_status 계약을 함께 반영해야 하며 준비 이미지/운영에는 아직 미반영이다.
구버전 결과 dataclass는 새 result_json 필드를 읽지 못해 uncertain으로 보류할 수 있으므로 이전 이미지 복귀의 이력 호환 검증은 배포 전 별도로 수행한다.
장치별 독립 작업자/시간 제한은 아래 P3.2 MQTT 로컬 구현에 연결했다. REST 영속 접수는 후속 로컬 검증했고 큐 계측, 장기 대기 overdue·큐 예산·부하(P2.4/P3.1), 물리 stop/replace·운영/실물 인수는 남는다.
이번 범위는 하나의 수신 DB와 단일 Action 소유자다. 다중 Jetson/서로 다른 DB 간 장치 분산 잠금을 보장하지 않는다.

1. 경합 단위는 물리 `resource_key`다. 같은 스피커를 공유하는 두 카메라는 같은 큐를 사용한다.
   서로 다른 장치는 독립 진행한다. 구역에 여러 장치가 있으면 대상별 작업과 결과를 따로 저장한다.
2. 정렬 키는 `(risk_rank, within_risk_rank, accepted_sequence, request_id)`다. 작은 값이 먼저다.
   위험 순서는 critical → warning → normal → info → low다. emergency→critical, warn/error→warning을 정규화한다.
   미등록/품질 불명은 임의로 critical로 올리지 않고 held 사유를 남긴다. 정상 관측은 물리 큐에 넣지 않는다.
3. 같은 등급 내 유형 순서는 정책 버전으로 관리한다. 현재 config를 이행 입력으로 쓰되,
   `risk_base + level` 한 숫자 때문에 등급 사이 순서가 뒤집히지 않도록 분리한다. 표시용 risk_score와 정렬 키는 다르다.
4. 동순위는 FIFO다. 고위험이 반복되어 오래 기다린 낮은 위험을 지우거나 임의로 critical로 올리지 않는다.
   `max_wait_at` 이후 overdue 상태를 관제에 표시하고, 출력 기한 내에서 대기를 계속하거나 명시적 운영 결정을 기록한다.
5. 이미 실행 중인 장치를 기본적으로 강제 중단하지 않는다. 높은 위험은 다음 실행 후보의 맨 앞에 둔다.
   선점은 장치별 stop/replace 지원과 완료 확인 방식이 검증됐을 때만 사용한다.
6. 선점 흐름은 중단 요청→중단 완료 확인→현재 작업 interrupted 기록→새 작업 실행이다.
   timeout/응답 유실이면 현재 작업 uncertain, 장치 리소스 점유를 유지하고 상태를 조회한다. 새 경보를 겹쳐 보내지 않는다.
7. 전광판의 화면 교체도 이전 요청의 지연 도착이 최신 화면을 덮지 않는지 검증해야 한다.
   스피커·사이렌은 중단 능력을 추정하지 않는다. lease 만료만으로 물리 장치가 멈췄다고 판단하지 않는다.

## 4. 만료·장기 대기·수동 승인

**10-13 P3.1 만료·안전 재개 로컬 구현:** [검증 근거](evidence/action_expiry_20261013.json).
기존 MQTT 수신 DB에 `action_execution_windows` 테이블을 추가해 최초 기한·정책 hash·물리 대상과 실행 선점을 보존한다.
원본과 정책 스냅샷을 모두 commit한 뒤 ACK하며, 접수·실행 선점·장치별 발행 직전에 같은 기한을 확인한다.
동일 `event_id` 재수신은 기한을 늘리지 않는다. 같은 ID의 다른 내용은 원래 계획을 보존하며 `event_content_conflict`로 보류한다.
재시작 시 미실행 queued만 유효 기한·동일 만료 정책/대상·AUTO 모드를 확인해 재개한다.
실행 중 종료·완료 저장 실패는 uncertain으로 남겨 재수신도 차단한다. complete는 입력 처리 반환이며 물리 성공 ACK가 아니다.

활성화는 `ACTION_EXPIRY_POLICY_PATH` 또는 `--action-expiry-policy`로 명시한다. 영속 수신 DB·고정 client ID·EdgeX 장치 모드가 필요하다. REST 병행은 아래10-13 영속 접수 구현에서 같은 DB/복구 규칙으로 연결했다.
기본 제공 [정책 파일](../../config/action_expiry_policy.json)은 `clock_verified=false`, `rules=[]`다. 운영에 자동 연결하지 않았고 승인되지 않은 TTL 기본값도 없다.

| 정책 필드 | 계약 |
|---|---|
| `schema_version` / `policy_version` | 1 / 비어 있지 않은 운영 정책 버전. 파일 내용 변경 시 대기 입력은 policy_changed로 보류 |
| `clock_verified` | 운영자가 수신 서버 시계 검증 후 설정. 자동 NTP 검증을 대신하지 않음 |
| `rules[].topic` / `event_type` | 정확히 일치하는 토픽·이벤트 유형. wildcard 규칙 없음 |
| `time_basis` | occurred_at 또는 명시적으로 선택한 received_at. 누락 시 자동 대체 없음 |
| `time_field` / `source_time_verified` | 발생 기준일 때 occurred_at(기본) 또는 timestamp / 사건 시계 검증 true 필요 |
| `ttl_seconds` | speaker/signboard/siren별 유한한 양수 초. 시험의 10초는 현장 승인값이 아님 |

입력은 명시적인 문자열 `event_id`를 가진 JSON 객체 또는 단일 원소 배열이다. 시각은 timezone이 있는 ISO8601 또는 양수 epoch 초이며, 미래·누락·비정상 시각은 보류한다.
실행 중 벽시계와 단조시계 차이가 1초를 초과하면 time_unverified를 유지한다. 재시작 전후 시계 신뢰는 운영자 설정에 의존하므로 현장 동기화/재부팅 검증이 남는다.
빈/다중 이벤트 배열은 분리 구현 전 batch_requires_split로 보존한다. 만료/보류 원본과 사유는 수신 DB에 남으며 통합 관제 조회·해제 UI는 잔여다.
수동 승인 입력의 재시작 자동 승인은 차단했다. 영속 승인·재승인 경로, 발행 후 브로커 체류 중 만료·장치 중단/교체는 남는다. 위험/FIFO 물리 큐의 후속 로컬 구현은 위 3절을 참조한다.
새 의존성0, 제품 Python8파일/설정1파일 변경. 운영/준비 이미지 반영·실물 시험은 미실행이다.

- 출력을 위한 TTL과 최대 대기 기준은 **유형/장치별 명시적 설정**이다. D04 수치를 이번 문서에서 임의로 확정하지 않는다.
  설정이 없으면 `held(policy_unset)`, 시계/사건 시각을 신뢰할 수 없으면 `held(time_unverified)`로 기록한다.
- 신뢰 가능한 사건 시각이 있으면 `expires_at = occurred_at + TTL`을 사용한다.
  수신 시각으로 대체하려면 해당 입력 프로필의 명시적 정책과 time_basis를 남긴다. 재시도·재시작으로 기한을 연장하지 않는다.
- `now >= expires_at`이면 아직 보내지 않은 작업은 expired다. 경계 시각도 만료에 포함한다.
  큐 등록 시와 실제 발행 직전에 다시 검사한다. 실행 중에는 출력 종료/중단 정책을 별도로 적용하며 상태를 단순 expired로 덮지 않는다.
- manual은 승인까지 waiting_approval이다. 승인 시에도 기한·대상·현재 장치 상태를 다시 확인한다.
  승인 대기는 영속 저장하며 재시작 후 자동 승인하지 않는다. 승인해도 기본 위험 순서는 바뀌지 않는다.
- expired/cancelled/held/filtered/duplicate-linked는 **물리 출력의 처리 결과**다. 원본·이벤트·관제 전달 기록을 삭제하는 상태가 아니다.
- 센서값의 fresh/stale/unknown 조회는 [기존 관제 freshness](MONITORING_FRESHNESS.md)와 구분한다.
  오래된 측정값의 단순 재수신을 새로운 위험 사건으로 만들어 반복 출력하지 않도록 D02 규격과 P4 정책으로 연결한다.

## 5. 중복과 재시작

10-13 과거 요청 이관·재수신 차단을 로컬/후보 검증했다. [근거](evidence/action_legacy_guard_20261013.json)
기존 `device_requests`의 빈 `request_hash`/NULL 결과는 **내용·실행 여부 미확정인 과거 ID**를 뜻한다.
정상 SHA256와 구별하며 과거 ACK/실패/모의 상태로 성공을 꾸미거나 재실행을 허용하지 않는다.
요청의 실제 ID·논리 ID·이벤트+장치 ID를 대조하고 보류된 수신 ID도 commit한 뒤
`uncertain/legacy_execution_unconfirmed`, 조회 `retry_allowed=false`로 반환한다.
기존 정상 실행 기록/내용 충돌 처리는 유지하고 새 이벤트는 정상 실행한다.

이관 도구는 `scripts/ops/seed_legacy_device_requests.py`이며 `--source-db`, `--destination-db`,
`--device signboard|speaker|siren`을 받는다. 원본은 읽기 전용으로 열고 해당 장치 명령/결과 토픽의
논리·물리·이벤트 별칭만 새 DB로 옮긴다. 기존 대상은 덮지 않으며 손상 기록이 있으면 완료본을 만들지 않는다.
명령 원본5775행 중 출력5774행/식별자5815개는 모의 재출력0, 신규 요청은 재시작 포함1회 실행이었다.
기록에 없는 요청/모든 ID가 바뀐 사건은 판별할 수 없다. 아직 수신하지 않은 물리 요청 조회는 unknown일 수
있지만 재실행 허가는 아니다. 실제 전환은 입력·발행·재전달 통제 후 최종 원본으로 다시 이관한다.
시험 저널은 새 모의 실행도 들어 있으므로 운영에 복사하지 않는다. 실제 전환·복귀·보류 해제는 미검증/잔여다.


10-12 후속: 응답 제한시간·늦은 응답·읽기 전용 요청 조회·재시도 금지 사유를 로컬 구현했다. 관련282 PASS. 30초는 운영 조정용 초기값이며 실물 미출력 증명/사람 승인에 따른 재시도 자동화는 구현하지 않았다. [근거](evidence/action_response_20261012.json)

10-12 구현 현황: 장치3종 `execute_request`의 영속 선점·내용 hash 충돌·종료 경계 보류를 로컬 검증했다(A07~A10 해당 경계). 249 PASS·운영 미반영. 장치별 우선순위 큐·실제 상태 조회/재시도 자격·과거 실행 이력 이관까지 완료한 것은 아니다. [근거](evidence/action_durability_20261012.json)

| 조건 | 처리 |
|---|---|
| 같은 source/event/request ID와 같은 고정 내용 | 기존 논리 작업·결과를 반환. 수신/조회 이력은 보존하되 새 물리 실행을 만들지 않음 |
| 같은 ID, 다른 명령 내용·대상/hash | conflict로 격리. 이전 내용을 덮어쓰거나 실행하지 않음 |
| 값이 같지만 서로 다른 정상 접수 ID | 기록 둘 다 보존. 안정적인 생산자 ID 없이 값만 비교해 중복 삭제하지 않음 / I19 |
| 반복 관측을 하나의 사건으로 묶음 | 검증된 incident_key와 정책 버전/기간으로 명시적 묶음. 모든 원본 참조를 보존. 쿨다운은 영속 중복 제거의 대체가 아님 |
| 미발행 queued 작업의 재시작 | 소유권 회수 후 기한·정책·대상을 확인해 재개 가능 |
| dispatching/sent/timeout/ACK 미저장 중 재시작 | uncertain으로 상태 조회. 발행되지 않았음이 확인되거나 장치가 같은 요청의 영속 중복 제거를 보장할 때만 제한 재시도 |
| ACK/출력 완료 저장 후 재시작 | 재생하지 않고 이력을 조회. 결과 저장 실패는 수신 결과를 다시 대조 |
| 장치 상태 조회·중복 제거 불가 | uncertain 유지·관제 확인. 사람이 위험을 평가한 명시적 replay만 별도 감사 기록으로 허용 |

같은 논리 요청의 재전송은 **같은 request_id**, 새 `attempt_id`를 쓴다. 재시도마다 새 request_id를 만들어
장치 중복 제거를 우회하지 않는다. 사람이 새 출력 자체를 요청한 replay는 새 event_id를 발급하고 원 이벤트·사유·승인자를 연결한다.
ID 연결만으로 물리 exactly-once를 보장하지 않는다. broker PUBACK·관제 저장 ACK도 물리 출력 ACK가 아니다.

## 6. 실행 상태와 결과 집계

명령/장치 실행 상태는 queued, waiting_approval, held, dispatching, accepted, succeeded,
failed, uncertain, expired, cancelled, interrupted로 나눈다. held 사유와 재판정 조건을 함께 저장한다.
`accepted`는 어댑터/장치 접수 확인이며, `succeeded`는 해당 장치 계약에 정의된 완료 증거가 있을 때만 사용한다.
물리 확인이 없는 경우 physical_confirmation=unknown을 유지한다. 기존 acknowledged를 곧바로 succeeded로 치환하지 않는다.

| 대상별 결과 | 논리 명령/이벤트 집계 예제 |
|---|---|
| 전광판 succeeded, 스피커 succeeded | succeeded. 실제 출력 확인 수준은 각 대상의 증거와 함께 표시 |
| 전광판 succeeded, 스피커 failed | partial_failure. 성공 대상 재실행 없이 실패 대상의 재시도 자격만 평가 |
| 전광판 succeeded, 스피커 uncertain | uncertain. 완료된 전광판까지 다시 보내지 않음 |
| 전광판 succeeded, 스피커 accepted | in_progress. 접수 확인을 물리 완료로 합산하지 않음 |
| 모든 대상 expired/cancelled/held/미등록 | not_executed 또는 held. 성공으로 표시하지 않고 대상별 이유 보존 |
| 대상 없음 | not_executed(no_target). 유형 공통 토픽으로 대신 발행하지 않음 |
| 실패와 uncertain이 함께 있음 | 전체는 uncertain, 실패 수와 대상별 결과도 보존. 단일 문자열로 부분 결과를 덮지 않음 |

집계는 고정한 대상 집합에 대해 계산한다. 결과 도착 순서와 무관해야 하며,
늦은 이전 attempt의 응답은 감사 이력에 남기되 다른 요청이나 현재 시도를 덮지 않는다.
`request_id`·물리 device_id·attempt 상관관계와 terminal 상태 전이를 검증한다.

## 7. 인수 예제와 구현 인계

아래 숫자/시각은 **계약 설명용 예제**이며 운영 TTL·지연 합격 기준이 아니다.

| ID | 입력/상황 | 기대 결과 | 후속 검증 |
|---|---|---|---|
| A01 | 같은 스피커에 warning head와 critical zone_dwelling 대기 | critical 먼저. I23 숫자/슬롯 로컬 수정; 영속 물리 큐 잔여 | P3.1 전체 위험 등급 조합 |
| A02 | 동일 위험/유형, sequence 10과 11 | 10→11. 시각/객체 ID가 같아도 정상 접수는 보존 | P3.1 FIFO·동시 접수 |
| A03 | 다른 카메라 두 대가 같은 스피커 사용 | 물리 리소스 한 큐, 동시 실행 금지 | P3.1 대상/구역 경합 |
| A04 | 스피커 timeout, 전광판 사용 가능 | 전광판과 관제는 독립 진행, 스피커만 uncertain | P3.2 장애 격리 |
| A05 | expiry 12:00:10, 발행 직전 12:00:10 | expired, 출력 0회·기록/관제 유지 | P3.1 만료 경계 |
| A06 | 최초 접수 TTL 없음/시계 미확인 | held, policy_unset/time_unverified 이유 | P3.1 정책·시계 검증 |
| A07 | 같은 request/hash 동시 재수신 2회 | 영속 unique/transaction으로 물리 작업 1개 | P3.3 동시 중복 |
| A08 | 같은 request ID, 다른 hash | conflict, 원본 작업 불변 | P3.3 충돌 |
| A09 | 장치 실행 후 ACK 저장 직전 종료 | uncertain→상태 대조. 근거 없는 재발행 금지 | P3.3 종료 경계 |
| A10 | ACK 완료 뒤 재시작 | 기록만 복구, 출력 0회 | P3.3 완료 복구 |
| A11 | 오래 기다린 warning, 새 critical 계속 유입 | warning 기록 유지·overdue 표시. 임의 위험 승격 금지 | P3.1 장기 대기 |
| A12 | 선점 요청에 stop ACK 없음 | 현재 리소스 uncertain 점유, 신규 물리 발행 보류 | P3.1/P3.3 선점 |
| A13 | 수동 승인 전 만료 후 승인 요청 | expired 유지, 승인으로 기한 갱신 금지 | P3.1 manual |
| A14 | 같은 값, 다른 센서 접수 ID | 둘 다 저장/관제 전달. 생산자 ID 없는 중복 제거 금지 | I19·P4.1 |

출력 운영 모드는 EdgeX를 기본 경로로 이행한다. 같은 물리 장치에 Direct와 EdgeX를 동시에 쓰지 않는다.
Direct는 별도 검증된 복구 경로일 때만 사용하고, 전환 시 단일 실행 소유자·미완료 요청 대조·대기 배수를 확인한다.
shadow는 실제 출력하지 않는 계약이어야 한다. I06/I09의 기존 배포 정책·검사 불일치는 이 문서로 해결되지 않는다.

## 8. 완료 판정과 제한

- 장치 경합·선점 규칙, 만료/장기 대기/중복 규칙, 대상별 결과 집계 예제를 확정했다.
  A01~A14는 명세 검토 예제이며 새 운영 구현의 실행 시험을 통과했다는 뜻이 아니다.
- 현재 코드의 관련 회귀 **87 PASS / 0 FAIL / 0 SKIP**를 확인했다.
  [JUnit](evidence/action_policy_baseline_20260922.xml). 기존 시험이 커버하지 않던 위험 순서 차이는 I23으로 별도 재현했다.
- [검증 범위·해시·인계](evidence/action_policy_contract_validation_20260922.json).
  09-22 계약 수립 당시 변경은 문서·진행판에 한정됐다. 이후 제품/DB/설정 변경 범위는 각 날짜의 구현 근거를 따른다.
- D04: 장치별 TTL/max_wait/timeout/시도 상한·큐 예산·완료 증거 수치, D02: 센서 생산자 ID/시각 규격,
  D06: 보관/관제 조건은 미정이다. 실제 장치 능력·중단·중복 제거 확인은 P3 실물 시험에서 채운다.
- 이슈 상태는 [STATUS 원본](STATUS.md)을 따른다. I23은10-13 FIXED_LOCAL이며 운영 확인은 잔여다. 계약과 운영 동작을 구분한다.

P1.3은 **DONE(계약)**, P3.1/P3.2는 미완료다. P3.3은 대상·중복/영속 실행·결과 수집 일부 로컬 보완을 마쳤으나 전체는 PLANNED다. P3.1은 P2.4 완료 조건도 충족해야 한다.
10-13 장치별 작업자·느린 장치 격리와 REST 영속 접수를 로컬 검증했으며 다음 세부 작업은 **P3.2 큐 길이·최장 대기·지연 계측**이다. 전광판 완료/상태 대조는 로컬 검증했고 스피커/사이렌 종료 확인·실물 검증은 P3.3에 남긴다. P3.1 물리 큐는 로컬 검증했고, 전체 판정에는 P2.4 선행 조건·장기 대기/선점·운영/실물 검증이 필요하다.


## 10-13 Action MQTT 영속 수신·실행 분리 (로컬/후보 검증)

- `ACTION_MQTT_INTAKE_DB`를 명시할 때만 기존 `IngestStore`와 지속 MQTT 구독을 사용한다. 표준 경로는 `/app/data/runtime/action_mqtt_intake.db`이며 기존 Action DB와 분리한다. 새 DB만 생성하고 기존 접수/단계 테이블을 재사용한다. 신규 파일 권한은600, 표준 백업 대상은12경로다. 다른 경로를 선택하면 백업 경로도 별도로 맞춰야 한다.
- 고정 `ACTION_MQTT_CLIENT_ID`가 필수이며 장비·서비스별로 유일해야 한다. 수신은 QoS1 구독·수동 ACK를 사용해 원본 commit 후 ACK한다. 저장 실패 시 ACK하지 않고 같은 세션으로 재접속한다. 생산자/규칙도 QoS1이어야 하며 AI3규칙 후보의 sink QoS를1로 맞췄다. 운영 규칙은 아직 변경하지 않았다.
- 네트워크 콜백은 원본 저장만 한다. 작업자는 실행 전 ready를 commit하고 기존 Action 처리를 호출한다. complete는 입력 처리 반환이며 장치 ACK/실물 성공이 아니다. 실행 예외/완료 저장 실패의 ready는 `execution_unconfirmed`로 보류하며 자동 재호출하지 않는다. DB 소유 잠금으로 작업자1개만 허용한다.
- `ACTION_MQTT_CAPTURE_ONLY=true`에서는 새 입력을 저장하며 Action 작업자·HTTP 전달 작업자를 시작하지 않는다. 후속 REST 영속 접수 구현부터 REST도 원본만 저장하고 수동 승인은 거부한다. 영속 수신은 요청 ID가 유지되는 EdgeX 장치 모드로 제한한다. 이 선택 설정을 지정하지 않은 기존 운영 경로는 유지한다.
- 정책 옵션이 없는 기존 모드는 재시작 때 pending을 `recovery_requires_review`, ready를 `execution_unconfirmed`로 보류한다. 원본은 삭제하지 않는다. **위 4절의 만료 정책을 명시한 로컬 후보에서만 유효한 미실행 입력 재개를 허용한다.** 브로커에서 늦게 도착한 입력의 발생 시각 기준 만료는 로컬 검증했고, FIFO·물리 장치 대기/운영 전환은 아직 합격이 아니다.
- 검증 범위는 MQTT 수신 저장·ACK·콜백/실행 분리·복구 보류다. 후속 REST 영속화·장치별 독립 실행은 아래 로컬 검증을 참조한다. 큐/대기 지연 계측, 브로커 보존 용량/만료·부하·실물/운영 전환은 잔여다. 최초 세션 등록 전의 메시지나 오프라인 QoS0를 복구한다고 주장하지 않는다.

검증 원자료와 후보 식별자는 [10-13 수신 검증](evidence/action_intake_20261013.json)에 기록한다.

## 10-13 P3.2 장치별 작업자·느린 장치 격리

[로컬 검증 근거](evidence/device_workers_20261013.json): 회귀413+격리 실제MQTT11 PASS, 신규26시험. 운영/준비 이미지에는 미반영이다.

- Action 물리 큐 사용 시 각 resource_key에 작업자 하나를 두어 발행/상태 대조를 수행한다. 느린 네트워크 호출이 다른 물리 대상과 입력 저장을 막지 않는다. 기존 위험/FIFO·만료·완료 증거 조건은 유지한다.
- 전광판/스피커/사이렌 러너의 `DEVICE_WORKERS_ENABLED=true`가 장치 수신 분리를 켠다(기본 OFF). 기존 요청 이력 DB에 `device_mqtt_inbox` 테이블1개/인덱스2개를 추가한다. DB 경로/새 의존성/Compose 기본값은 추가하지 않는다.
- 고정 MQTT client ID, 지속 세션/QoS1과 수동 ACK를 사용한다. 원본 commit 후 ACK하며 저장 실패 시 ACK하지 않고 연결 복구 후 재전달을 받는다. 잘못된 입력도 원본을 held로 보존한다.
- 등록된 물리 device_id별 실행 작업자와 읽기 전용 request_status 작업자를 분리한다. 동일 장치 호출은 직렬화하고 결과 유실은 기존 실행 이력 조회로 대조한다. 실제 출력 지속시간의 점유는 Action 큐/완료 증거가 관리한다.
- `DEVICE_WORKER_TIMEOUT_SECONDS` 기본30초는 로컬 초기값이며 현장 SLA가 아니다. 단조 시계로 지연을 감지하고 `uncertain/device_worker_timeout`을 한 번 기록/발행한다. Python 스레드나 물리 출력을 강제 취소하지 않으며 작업자를 버리고 대체하지 않는다.
- 프로세스 재시작 전 pending/running 출력은 `held/worker_recovery_requires_review`로 보존하고 같은 요청의 재전달도 보류한다. 읽기 전용 상태 조회만 재개한다. 종료 대기 후에도 호출이 남으면 소유 잠금을 유지해 중복 작업자를 막는다.
- 기본 client ID는 `cctv-{EDGEX_JETSON_ID}-{장치종류}`이며 `DEVICE_MQTT_CLIENT_ID`로 지정할 수 있다. 동시 실행 서비스 간 유일해야 하고 재시작에서는 같아야 한다. 기존 요청 DB 전체 백업에 새 수신함도 포함된다.

이번 격리는 MQTT 경로와 같은 DB/단일 소유자 범위다. 직접 HTTP 장치 명령·별도 DB/외부 제어 주체의 경합, 큐 계측/보관 한계, 운영/실물 검증은 남는다. 스피커/사이렌의 실제 종료 신호도 별도 후속이다.


## 10-13 P3.2 REST 영속 접수·실행 분리

[검증 근거](evidence/action_rest_intake_20261013.json): 회귀392+실제 loopback HTTP2 PASS(신규33개), Ruff PASS. 핵심2파일 mypy PASS, 변경8파일 전체는 수정 전후 같은16오류로 미통과다. 운영/준비 이미지 미반영.

기존 `REST_ENABLED=true`와 `ACTION_MQTT_INTAKE_DB`가 함께 설정되면 POST /events도 기존 수신 DB에 보존한다. 고정 `ACTION_MQTT_CLIENT_ID`·EdgeX 모드 조건은 유지한다. 새 DB/테이블/의존성/Compose 설정을 추가하지 않았다. REST 경로는 `action_rest`, MQTT는 `action_mqtt` namespace로 구분하며 두 경로를 실제 접수 순서대로 같은 작업자가 소비한다. 기존 바이트 보존 형식의 payload_base64에 HTTP 본문을 그대로 저장한다.

| 요청/상태 | 계약 |
|---|---|
| POST /events | event_id가 있는 단일 JSON 객체. 재시도는 같은 event_id·내용·topic 사용. Content-Length 필수, 본문 최대1MiB, chunked 미지원 |
| 정상 접수 | 원본·접수 상태·설정된 만료 계획 commit 뒤 HTTP202, status=accepted, queued=true, durable=true, ingest_id/event_id 반환. 장치 실행 성공이 아님 |
| 입력 오류/저장 실패 | 잘못된 JSON/ID/topic400, 크기 초과413, DB 저장 실패·기동 전/종료 중503. 부분 commit 때 원본이 남을 수 있으므로 새 이벤트 ID로 무조건 재시도하지 않음 |
| GET /receipts/{ingest_id} | 같은 내부 인증 적용. received_at/stage/reason 조회, REST 외 namespace/없는 ID404, 저장소 장애503. 원본 본문은 응답하지 않음 |
| 접수 상태 complete | 입력 처리 함수 반환 의미. 장치 ACK/실물 완료는 기존 /commands와 완료 증거로 별도 판단 |
| 저장 전용 | REST도 접수만 하고 메모리 소비자를 시작하지 않음. 수동 승인으로 실행 우회 불가 |
| 재시작 | 유효한 만료 계획을 가진 미실행 입력만 기존 규칙으로 재개. 처리 중 종료·만료/정책 불명·계획 저장 미완료는 원본/사유 보존 후 보류 |

REST topic 기본값은 rest/inbound이며 MQTT 제어·장치 결과 topic은 거부한다. 허용 이벤트 topic의 명시적 만료 규칙이 없으면 보류한다. REST의 topic 필드는 원본에 남기고 실제 이벤트/내용 hash 계산에서는 분리하므로 MQTT와 같은 사건이 최초 기한·내용 충돌/중복 방지를 공유한다.
영속 수신을 설정하지 않은 기존 REST 메모리 큐/HTTP200 계약은 유지한다. 영속 모드 클라이언트는202를 접수로 처리하고 event_id를 유지해야 한다. 새 접수 조회는 내부 Action REST API에만 추가했으며 외부 API 프록시/운영자 화면의 통합 조회는 P3.3 후속이다.
큐 계측·보관 한계·영속 승인·운영 반영/전환·복귀·실물 검증은 잔여다.

공개 API의 경보 중계는 선택 입력 event_id를 보존하며 없으면 기존 ID 생성 함수로 원래 timestamp·내용에서 계산한다. 센서 중계도 생산자 ID 또는 원래 occurred_at/timestamp/received_at으로 같은 ID를 만든다(밀리초→초 정규화). 센서의 ID·시각이 모두 없으면 현재 시각을 보강하지 않으며 영속 REST에서400으로 거부된다. 원본 센서 저장은 기존 공개 API 경로를 따른다. 두 중계 경로의202 처리·ID/시각 전달·재전달 동일 ID를 로컬 검증했다.
