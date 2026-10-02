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

**I23 OPEN:** 현재 설정에서 head=warning/2, zone_dwelling=critical/3이다.
head 슬롯이 활성인 동안 zone_dwelling을 제출하면 False로 스킵된다. smoke_test_alert도 critical/4라
warning/2보다 뒤로 밀릴 수 있다. 낙상=critical/0 시험만으로 전체 위험 등급의 순서를 보장하지 못한다.
[실제 계산·슬롯 재현](evidence/action_policy_priority_gap_20260922.json).
이번에는 앱/설정의 숫자를 바꾸지 않았다. P3.1 적용 전에 전체 유형·별칭·사용처 회귀로 해결해야 한다.

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
| A01 | 같은 스피커에 warning head와 critical zone_dwelling 대기 | critical 먼저. 현재 I23은 이 규칙 미충족 | P3.1 전체 위험 등급 조합 |
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
  변경은 계약·진행 문서·온라인 진행판이며 앱·DB schema·공개 API·설정·운영 서비스는 바꾸지 않았다.
- D04: 장치별 TTL/max_wait/timeout/시도 상한·큐 예산·완료 증거 수치, D02: 센서 생산자 ID/시각 규격,
  D06: 보관/관제 조건은 미정이다. 실제 장치 능력·중단·중복 제거 확인은 P3 실물 시험에서 채운다.
- I02/I03/I06/I09/I18/I19/I23은 구현·운영 확인 전 OPEN을 유지한다. 새 계약을 현행 동작으로 표시하지 않는다.

P1.3은 **DONE(계약)**, P3.1~P3.3 구현은 미착수다. P3.1은 P2.4 완료 조건도 충족해야 한다.
다음 WBS는 선행 조건이 갖춰진 **P5.1 단일 영상 입력·추론 계약/계측**이며,
첫 단위로 실제 로딩 모델·전처리·인물/시각·계측 위치를 조사한다. 운영 재시작이나 전체 부하 시험을 자동으로 시작하지 않는다.
