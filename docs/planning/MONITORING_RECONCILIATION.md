# P2.3 — 관제 누락 대조·적체·구간 지연 계측

2026-09-21 · 계획 v0.16 · **P2.3 IN_PROGRESS** · R04,R10

> 후속 v0.17: [서비스·로컬 MQTT·commit 전체 지연](MONITORING_SERVICE.md)을 연결·검증해
> P2.3 내부 모형을 DONE으로 닫았다. 아래는 대조/계측 실행 단위 완료 당시의 기록이다.

송신·수신 장부를 고정한 snapshot으로 ID/업무 키/본문 hash를 대조한다.
조회 도중 새로운 기록이 추가돼도 기존 목록은 바뀌지 않으며, 과거 시각의 늦은
정규화 결과는 다음 전체 대조에 포함한다. 대기 건수·시간 품질과 worker 구간 지연도 조회한다.

이번 실행 단위는 로컬 모형 구현·시험 완료다. 서비스 주기 실행/시작·종료와 인증된
통신 어댑터, 접수→ACK commit 전체 지연 계측은 남아 P2.3 전체는 진행 중이다.
운영 DB·설정·서비스·GPU·실장비에는 적용하지 않았다.

## 1. 변경과 영향

| 파일 | 책임 |
|---|---|
| [monitoring_snapshot.py](../../src/monitoring_snapshot.py) | 별도 SQLite manifest, 범위·TTL·페이지/업무 키 조회, 원본/수신 DB 읽기 |
| [monitoring_reconciliation.py](../../src/monitoring_reconciliation.py) | 고정 집합의 누락·충돌·예상 밖 기록·동시 전송·대조 불가 분류 |
| [monitoring_metrics.py](../../src/monitoring_metrics.py) | 영속 장부에서 적체/시도 집계, bounded monotonic 구간 표본 |
| [monitoring_worker.py](../../src/monitoring_worker.py) | 선택 인자 `timings`를 연결해 조회·전송·ACK 저장 호출 구간 측정 |
| [snapshot 시험](../../tests/test_monitoring_snapshot.py), [대조 시험](../../tests/test_monitoring_reconciliation.py), [계측 시험](../../tests/test_monitoring_metrics.py) | 고정 목록·복구·누락 판정·센서/영상 연계·시간 품질·오류 전파 |

앱 3파일 추가·기존 worker 1파일 수정, 시험 3파일 추가다. 라이브러리 추가와
공통 source DB v4/receiver DB v1 스키마 변경은 없다. worker의 `timings=None` 기본값은
기존 호출 동작을 유지한다. 장치 출력·경보 재실행은 연결하지 않았다.

## 2. Snapshot 저장·조회

`MonitoringSnapshots(path)`는 호출자가 지정한 **별도 DB**를 사용한다.
application_id=1129598258, user_version=1, DELETE journal/EXTRA/FK ON이다.
기존 다른 DB를 채택하지 않으며 source·receiver와 같은 경로도 거절한다.
snapshot header와 모든 item을 한 transaction으로 저장해 중간 오류/종료 시 부분 목록이 남지 않는다.

source/receiver는 `query_only` read transaction에서 읽는다. 목록 복사를 마치면 닫고
나중의 페이지/대조는 manifest를 읽는다. 복사 중에는 SQLite read lock이 유지되므로
장기 부하·잠금 영향은 P2.4에서 확인해야 한다. MQTT/HTTP 대기를 DB transaction 안에 두지 않는다.

| 호출 | 결과·제한 |
|---|---|
| create_source(store, binding, namespace=None, ttl=300, max_items=100000) | source의 고정 전달 목록, 정규화/전달 생성 대기·정규화 격리 건수 |
| create_consumer(store, binding, namespace=None, ttl=300, max_items=100000) | commit된 수신 장부 목록. source 대기 상태는 알 수 없어 null |
| list_snapshot(binding, snapshot_id, cursor=None, limit=100) | 1~1000건·다음 opaque cursor, scope/TTL/해당 snapshot 소속 검증 |
| get_records(binding, snapshot_id, keys) | namespace/record_type/record_id 키 1~100개, 원래 record·명시적인 missing 키 |

item에는 delivery_id·업무 키·payload_sha256·원래 record와 해당 side의 연결/상태가 있다.
source는 ingest_id/원래 received_at/broker_status/monitoring_status를 포함한다.
consumer의 received_at은 **수신처가 받은 시각**이며 source 접수 시각으로 바꾸지 않는다.
0/null/missing/invalid와 원래 observed_at은 record 안에서 보존한다.

현재 선택 필터는 namespace이며 나머지 범위는 binding으로 고정한다. P1.4에 정의한
device/업무 ID/received_at 기간의 일반 이력 필터는 아직 제공하지 않는다.
누락 대조는 보관 중인 해당 namespace 전체를 대상으로 하며 이전 최대 timestamp 이후만
비교하는 증분 방식은 사용하지 않는다. 업무 키 상세 조회와 기간 이력 필터는 별개다.

TTL 기본 300초가 지나면 snapshot_expired다. 다른 snapshot의 cursor도 사용할 수 없다.
건수 상한 초과는 snapshot_capacity로 실패하고 부분 결과를 보관하지 않는다.
TTL은 자동 삭제 정책이 아니다. manifest 보관/정리·디스크 예산·장기 용량은 D06/P2.4에서 정한다.

## 3. 누락 대조와 완료 판정

`reconcile(snapshots, binding, source_id, consumer_id, lookup_current=...)`는
동일 binding·동일 namespace 필터의 source/consumer snapshot을 비교한다.
전체 ordinal 집합을 순회하며 상세 목록은 기본 100건/최대1000건으로 제한한다.
집계는 전체 대상이며 생략된 상세 건수는 `details_truncated`에 기록한다.

| 결과 | 의미 |
|---|---|
| matched | 전달 ID·namespace·record_type·업무 ID·hash 일치 |
| missing | source 고정 목록에 있고 consumer 고정 장부에 없음 |
| conflict | 동일 ID에 업무 키 또는 hash가 다름 |
| unexpected | consumer에만 있고 현재 source 장부에서도 없음 |
| out_of_snapshot | 원래 source 목록 밖이지만 현재 source 장부와 일치하는 정상 추가 기록 |
| unverifiable | 본문/장부 불일치, 현재 장부 조회 실패, DB 실패 또는 snapshot 만료 등 |

consumer에만 있는 ID는 `lookup_current(binding, delivery_id)`로 현재 source 장부를
다시 확인한다. 이번 구현은 `current_delivery()`의 로컬 DB 조회이며 실제 네트워크 어댑터가 아니다.
불통을 unexpected로 단정하지 않고 unverifiable로 남긴다. 대조 마지막에도 TTL을 확인한다.

- `complete`: 필요한 대조가 끝나고 unverifiable이 없음. missing/conflict가 없어야 한다는 뜻은 아니다.
- `source_set_matches`: 고정 source 집합의 누락·충돌이 없고 대조가 완료됨.
- `sets_equal`: 여기에 unexpected/out_of_snapshot도 없어 두 고정 집합이 같음.
- `all_inputs_accounted`: 위 조건에 정규화 대기·전달 생성 대기·정규화 격리도 없음.

빈 전달 목록이나 같은 건수만으로 전체 처리를 완료 판정하지 않는다. 추가 기록이 있으면
다음 전체 회차로 다시 대조한다. 결과는 **고정 목록 시점의 판정**이며 현재 전체 시스템이나
관제 화면 합격을 뜻하지 않는다. 만료/DB 오류의 부분 결과도 합격으로 바꾸지 않는다.
대조는 읽기 작업이며 자동 삭제·ACK 변경·격리 해제·저장 완료 기록 재발행을 하지 않는다.
아직 미확인인 기록의 재전송은 기존 worker가 담당한다. 수신처 장부 복원/초기화는
새 consumer/destination 식별 범위와 통제된 복구 절차가 필요하다.

## 4. 계측 범위와 시간 품질

`source_metrics()`는 namespace/destination별 관제 상태 수, 정규화/생성 대기·격리,
활성 lease, 발행 재시도, 잘못된 ACK/충돌 이력, DB 크기·파일시스템 여유 공간을 조회한다.
unconfirmed age는 blocked/quarantined를 포함하고, DB 읽기 실패는 0건으로 바꾸지 않고 전파한다.

`clock_sync='unknown'`이 기본이다. 확인되지 않은 wall clock 대기 나이/시간당 처리량은
null·unknown으로 남긴다. 잘못된 시각·미래 접수 시각은 invalid로 별도 집계한다.
`confirmed`는 호출자가 동기화를 확인한 경우만 지정하며, 조회 함수가 동기화를 검증하지는 않는다.
시간당 유입은 해당 namespace의 원본 접수 기준이다. 관제 처리량은 기존 `acked_at`의
**transaction 내부 기록 시각** 기준이며 commit 완료 시각이나 화면 표시 시각이 아니다.

`MonitoringTimings`를 worker에 주면 다음 구간을 monotonic clock으로 측정한다.

| 구간 | 측정 경계 |
|---|---|
| consumer_lookup | reader 호출 시작→반환/예외 |
| mqtt_publish | sender 호출 시작→반환/예외; 실제 transport이면 연결·PUBACK 대기·정리 포함 |
| consumer_ack_commit | source ACK 저장 호출 시작→commit 후 반환/예외 |

구간별 기본 최근 1000개(설정 최대10000개)의 p50/p95/max·유효/무효/예외 표본 수,
누적 표본 수·오래돼 제외된 수·측정 창을 제공한다. 표본이 없으면 지연은 null이다.
음수/무한값은 invalid로 세고 0ms로 보정하지 않는다. 표본은 현재 프로세스 메모리에만 있고
재시작 시 초기화된다. 기존 영속 시도 이력과 적체 집계는 DB에 남는다.
구간은 고정 3개이며 event_id를 metric label로 쓰지 않는다.

접수/정규화 commit 완료 시각, ACK commit 완료 절대 시각, 접수→ACK commit 전체 지연,
UI displayed_at·수신 duplicate 시도 수는 아직 미계측이다. 보고서 `not_measured`에 명시한다.
서비스 연결 시 해당 writer/consumer의 계측 경계를 추가해야 한다. D04 값이 없어 성능 PASS는 판정하지 않는다.

## 5. 검증과 다음 작업

신규 시험 23개: 고정 페이지·재시작·늦은 과거 접수·범위/TTL/cursor/용량 제한,
copy 실패와 **실제 프로세스 종료 후 rollback**, 원본/수신 파일 불변,
건수만 같은 다른 ID·hash 충돌·동시 정상 추가·불통/만료·센서/영상 대조,
시간 품질·표본 한도·ACK 저장 실패 전파를 포함한다.

[최종 회귀 JUnit](evidence/monitoring_reconciliation_final_20260921.xml)과
[검증 요약·파일 해시](evidence/monitoring_reconciliation_validation_20260921.json)에 결과를 기록한다.
**202 PASS / 실패·오류·SKIP 0, 135.81초**. Ruff·Black check·py_compile 변경 7파일 PASS.
[20건 실행 예제](evidence/monitoring_reconciliation_sample_20260921.json)는 영상 10건·센서 10건을
임시 DB와 **모형 전송/PUBACK**으로 처리해 missing 20→matched 20·stored 20을 대조하고
실제 monotonic 표본을 남겼다. 네트워크 MQTT나 처리량 벤치마크가 아니다.

```bash
rtk proxy .venv/bin/python -m pytest \
  tests/test_monitoring_snapshot.py tests/test_monitoring_reconciliation.py \
  tests/test_monitoring_metrics.py -q -o addopts=--tb=short
```

현재 변경의 테스트는 임시 SQLite/모형 전송만 사용한다. 실제 MQTT 검증은 앞선
[송신 모듈 결과](MONITORING_SENDER.md)이며 이번 실행 결과로 합산하지 않는다.
그래프는 신규 코드의 함수/흐름 0개를 반환해 현재 소스·DB 시험으로 검토했다.
다음은 선택형 서비스 주기 실행·시작/종료·로컬 수신/조회 어댑터와 남은 계측 연결이다.
D04/D06은 미정, I10 BLOCKED·I13/I17 OPEN을 유지한다.
