# P2.4 — 영상 최초 저장 잠금 재시도·활성 쓰기 경합 검증

<a id="connection-refactor-v112"></a>

## v0.112 — DB 연결 경합 완화·정리 리팩토링 (2026-10-06)

**저장부 로컬 변경 단위 완료·I45 OPEN·운영 미배포.** 사용자 요청대로 버그 대응과
동작을 유지하는 리팩토링을 분리했다. 제품은 `src/ingest_store.py` 한 파일만 변경했고
`tests/test_ingest_concurrency.py`에 실제 SQLite·서로 다른 저장소·프로세스 경계5조건을 추가했다.

### 근거와 수정 범위

수정 전 SQL 관측은3 PASS 뒤1 FAIL이었다.250ms 내외 BUSY13건 동안 다른 짧은 쓰기가
계속 완료됐다. 영상 최초 INSERT도 BUSY를 보였지만 재시도는 성공했고, 최종 실패는 센서
재시도 상태를 남긴 뒤 시험이 생산을 끝내48건에 도달하지 못한 timeout이었다.
**v0.111의 영상 최초 저장 재시도 소진과 같은 실패로 계산하지 않는다.**

기존 `_connect()`에서 같은 프로세스·정규 경로의 저장소가 최대250ms 동안 공통 RLock을
사용하도록 했다. 확보하면 연결 생성·설정·commit/rollback·close까지 유지한다. 확보 실패 시에는
기존 SQLite 경로로 진행하므로 실제 DB가 저장 가능한데 새 BUSY를 만들지 않는다.
별도 DB는 독립적이며 symlink 별칭은 같은 경로로 묶고, 미사용 lock은 weakref로 회수한다.
fork 자식은 registry와 mutex를 초기화한다. SQLite250ms·영상1회 재시도·DELETE/EXTRA는 유지한다.

수정 후44 PASS를 확인한 다음 표준 `closing()`으로 연결 종료 코드를 정리했다.
중간 엄격한 admission 거절안은 관련130 PASS/1 FAIL(동시 claim)을 냈으며, DB가 사용 가능한데도
거절하는 조건을 추가 시험으로1 FAIL/4 PASS 재현했다. 해당 안은 버리고 native fallback을 적용했다.
기존 atomic claim·혼합 저장 시험의 성공 조건은 바꾸지 않았다.

| 최종 검증 | 결과 |
|---|---|
| 관련 저장/관제 회귀 |132 PASS|
| 전체 회귀 |2795 PASS·52 SKIP·0 FAIL|
| 실제 SSD 혼합 쓰기6회 |각48건 저장·접수/전달 ID·hash·ACK 대조, SQL 오류0|
| 경계 |native 대기0의 로컬 경합/별칭, DB 사용 가능 시 fallback·별도 DB 진행, fork2조건|
| 기존 오류 계약 |외부 쓰기·commit 잠금, 재시도 소진·FULL·rollback·복구 회귀 통과|
| 검토 |독립 검토: 최종 Critical/Important 없음|

### 한계와 다음 작업

이 조정은 엄격한 직렬화나 공정한 순서를 보장하지 않는다.250ms 뒤의 fallback, 별도 프로세스,
hard-link 별칭은 SQLite 잠금 정책에 따른다. 긴 snapshot 조회와 실제 부하는 추가 확인 대상이다.
진입 대기와 native 대기가 합쳐질 수 있으므로 전체500ms 상한·현장 처리량 합격을 뜻하지 않는다.
모든 과거 잠금 실패의 원인을 확정하지 않았으므로 **I45 OPEN**을 유지한다.
새 GPU 후보 이미지·운영 반영은 없고, v0.111 이미지는 이 저장부 변경을 포함하지 않는다.
다음은 낙상 후처리의 검증된 분기를 작은 범위로 정리한다.

[원 실패·중간안·최종 검증](evidence/ingest_connection_refactor_20261006.json).
SQLite의 실제 잠금·대기 동작은 [잠금 설명](https://www.sqlite.org/lockingv3.html)과
[busy timeout](https://www.sqlite.org/c3ref/busy_timeout.html)을 대조했다.

---

<a id="sensor-lock-v105"></a>

## v0.105 — 센서 중복 투영의 불필요한 쓰기 제거 (2026-10-06)

**이번 최소 수정은 로컬 완료, I45 전체는 OPEN·P2.4 운영 기준 BLOCKED 유지.**
`project_sensor`가 이미 처리한 접수에도 `INSERT OR IGNORE`를 실행하여 불필요한 쓰기 잠금을
요구했다. 실제 별도 연결이 `BEGIN IMMEDIATE`를 유지하면 완료/invalid 접수 모두 실패했다.
기존 조회에 공통 처리 상태를 함께 읽어 종료 상태는 바로 반환하고, pending 상태도 준비 INSERT를
생략하도록 수정했다. 최초 동시 준비와 canonical/revision 원자적 상태 재검사는 유지한다.

| 검증 | 결과 |
|---|---|
| 수정 전 원래 2스레드 시험 계측 | 40회 통과·SQL3972건·최장 관측39.9ms, 간헐 실패 미재현 |
| 수정 전 전체 회귀·해당 시험 SQL 관측 | 2746 PASS·52 SKIP·252.75초, 해당 시험96 SQL·오류0 |
| 완료/invalid 접수 + 다른 연결의 쓰기 예약 | 수정 전2 FAIL → 수정 후2 PASS |
| 상태 준비/정규화 각각 쓰기/커밋의 실제 잠금 | 4조건 PASS. 수정 전부터 원본 보존·rollback·새 저장소 복구 후1건 저장 확인 |
| 수정 후 관련 회귀 | 62 PASS·5 SKIP·34.64초. SKIP은 임시 MQTT 브로커 지정이 필요한 통합 시험 |
| 수정 후 전체 회귀 | 2752 PASS·52 SKIP·0 FAIL·298.73초 |

변경 파일은 `src/sensor_record.py`, `tests/test_sensor_record.py`다. 기존 동시 처리 시험의
예외/성공 조건은 완화하지 않았다. API·스키마·DELETE/EXTRA·250ms 대기·재시도 횟수·SDK를
바꾸지 않았고 운영 컨테이너를 재시작하지 않았다. 관련 회귀의 경고2건은 JUnit 출력 형식 설정이며,
최종 전체 실행은 xunit1 형식을 사용한다. 소스/시험7파일의 검증 사본 해시와 작업 폴더가 일치한다.

**한계:** 처음 상태를 만드는 두 호출은 여전히 쓰기 잠금이 필요하다. v0.104의 최초 동시 투영
실패를 이번 최적화로 해결했다고 단정하지 않는다. 과거 영상 접수 재시도 소진·상태 조회 실패도
미해결이며, 아래 v0.104의 실패 이력을 보존한다. 새 실패 단서 없는 동일 조건 반복은 여기서
멈추고 다음 로컬 작업은 I47 낙상 ID 연계 시험의 시간 조건을 제어하여 제품 결함과 시험 의존성을
구분하는 것이다. 운영 부하 기준과 원 native I40는 별도 잔여다.

[실행 근거](evidence/sensor_projection_lock_20261006.json), 원자료
`data/runtime/i45_sensor_projection_20261006_v105/`. v0.104의 4스레드 표기는 실제 코드의
저장소2개·스레드2개로 정정했다.

---

<a id="db-contention-v104"></a>

## v0.104 — 실제 SSD에서 I45 SQL 대기 구간 관측 (2026-10-06)

**I45 OPEN·P2.4 운영 기준 BLOCKED 유지. 제품·DB 스키마·DELETE/EXTRA·대기/재시도 설정 변경 없음.**
인코더 조사 단위 후 사용자 요청대로 다음 업무를 착수했다. 자격정보를 제거한 기존 검증 사본과 실제 SSD
임시 DB에서 기존 혼합 시험을 실행하고, SQL 진입/반환·연결별 transaction 유지 구간만 관측했다.
파라미터·원본 데이터는 기록하지 않았으며 관제 전송은 기존 프로세스 내부 모형이다.

| 검증 | 결과 |
|---|---|
| 혼합 입력 단독 SQL 관측 | 1 PASS·6.08초, 48건 저장/ID/hash/ACK 일치 |
| 정상 진입 보호를 적용한 전체 회귀 | 2744 PASS·52 SKIP·2 FAIL·227.51초 |
| 전체 중 동일 혼합 시험 | 48건 일치·영상20건 접수, 영상 내부 BUSY 재시도1회·서비스 실패 주기1회 후 복구 |
| 전체 실패2건 개별 재확인 | 2 PASS·5.41초, 간헐 결함 종료 판정 아님 |

단독/전체 대표 시험의 BUSY 대기는251~257ms였다. 이때 다른 센서·영상·관제의 성공한 쓰기
transaction16~22개가 대기 구간과 겹쳤다. 개별 관측 구간의 최대값은 단독27.3ms·전체42.2ms다.
측정은 첫 쓰기/BEGIN 반환부터 context 종료까지이며 OS 잠금 소유권의 정확한 취득 시각은 아니다.
**긴 단일 transaction보다 연속 쓰기의 경쟁을 우선 확인할 근거**이며, 과거 영상 재시도 소진 및
상태 SELECT 실패가 이번 대표 시험에서 재현된 것은 아니다. 원인 해결로 처리하지 않는다.

전체 회귀의 실패는 다음처럼 구분한다.

정정(10-06 v0.105): 기존 본문의 4스레드 표기는 오류다. 실제 시험은 저장소2개·스레드2개다.

- `test_sensor_record.py::test_concurrent_projection_commits_one_observation`:
  같은 센서 접수를2스레드가 투영하는 중 `project_sensor`의 `INSERT OR IGNORE`에서
  `database is locked`. 이번 SQL 계측 대상 밖이므로 기존 I45와 동일 원인을 단정하지 않는다.
- `test_processor_stats.py::TestEventDebouncer::test_fall_detected_reassociates_changed_object_id_by_bbox`:
  ID 변경 후 전송 기대가 False. 실시간 `sleep(0.06)`과0.2초 gap 기준의 스케줄 지연은 원인 후보다.
  실제 호출 간격은 기록되지 않아 I47 OPEN으로 분리하고 시간 제어 재현을 남긴다.

첫 전체 관측 실행은 진단 진입기의 `__main__` 보호 누락으로 spawn 자식이 pytest를 다시 시작하여
적체 재시작 시험이 시간 초과했다. **제품 결함 근거에서 제외**하고 실행기를 보완했다.
해당 재시작 시험1 PASS(56.85초) 후 전체를 다시 실행한 결과가 위의 최종2 FAIL이다.
관련 소스/시험13파일은 작업 폴더와 검증 사본 해시가 일치한다. 이 단위에서 제품 수정은 없다.

당시 다음 한 가지는 센서2중 투영의 실패 구간 계측이다. 성공해 보이도록 대기시간·재시도 횟수·journal을
늘리기 전에 쓰기/커밋의 잠금 경계를 확인한다. 외부 부하 기준 대기와 구분하며 운영 서비스는 재시작하지 않았다.
[실행 근거·한계](evidence/db_contention_trace_20261006.json), 원자료 `data/runtime/i45_contention_20261006_v104/`.

---


검증: 2026-09-22 KST · 계획 v0.25 · **이번 내부 단위 완료 / P2.4 IN_PROGRESS / I22 OPEN**

영상의 최초 원본 저장(T1)이 일시적인 DB 잠금으로 실패하면 같은 접수를 **1회 재시도**한다.
쓰기와 커밋 단계에서 각각 실패를 재현한 뒤 수정했다. 신규 14개를 포함한 관련 회귀는
**200 PASS / 0 FAIL / 0 SKIP**다. 저장 전 종료·지속 잠금·용량 부족까지 무손실을 보장하는 변경은 아니다.

## 변경과 영향 범위

| 파일 | 변경 | 영향 |
|---|---|---|
| [event_ingest_store.py](../../src/event_ingest_store.py) | `accept_event_json(..., busy_retries=0)` 선택 인수 추가. 0/1만 허용하고 BUSY일 때 rollback·연결 종료 후 전체 접수 transaction 재시도 | 기본 호출은 기존 1회 시도. 원본 bytes·접수 ID·수신 시각·hash는 재시도 전 고정, commit observer는 성공 후 한 번 |
| [video_event_ingest.py](../../src/core/video_event_ingest.py) | 영상 영속 접수에서만 `busy_retries=1` 사용 | `AIOT_INGEST_DB`로 선택한 DeepStream/OpenCV 공통 접수 경로 |
| [test_video_intake_busy.py](../../tests/test_video_intake_busy.py) | 실제 쓰기/커밋 잠금, 소진, FULL, 오류 분류 시험 | 큐·접수·실패 통계·observer와 원본 불변 검증 |
| [test_monitoring_active_contention.py](../../tests/test_monitoring_active_contention.py) | 센서/영상 생산자와 관제 worker를 함께 실행 | 영상 시험 생산자 추가 재시도 없이 ACK·누락/중복 대조 |

DB schema·wire·공개 HTTP API·SQLite journal/timeout·환경 설정은 바꾸지 않았다.
센서/MQTT 접수, 관제 worker, 기존 큐 wrapper와 실패 통계 구현은 유지했다.
운영/노트북 배포·컨테이너 재시작·물리 출력 명령은 수행하지 않았다.

SQLite BUSY(기본 코드 5)는 다른 연결과의 경합이며 LOCKED·FULL·IOERR와 구분한다.
오류 코드가 있으면 확장 코드의 기본 값을 확인하고, Python 3.10처럼 코드가 없는 경우에만
정확한 `database is locked` 문구로 판별한다. [SQLite 결과 코드](https://sqlite.org/rescode.html)

연결 context manager는 본문 또는 commit 실패 시 rollback한다. 기존 `_connect`가 연결까지
닫은 뒤 재시도하므로 커밋 실패 후의 부분 접수도 남기지 않는다.
[Python 3.10 SQLite transaction 동작](https://docs.python.org/3.10/library/sqlite3.html#using-the-connection-as-a-context-manager)

기존 연결의 잠금 대기 설정은 250ms이며 전체 시도는 최대 2회다. **전체 함수 실행시간의
500ms 상한을 보장하지 않는다.** 연결/SQL/디스크 동기화 시간과 여러 잠금 지점이 더해질 수 있다.
동기 호출이므로 영상 처리 지연 영향은 실제 GPU/영상 부하에서 별도 측정해야 한다.

## 실패 경계 검증

| 조건 | 결과 |
|---|---|
| INSERT 또는 COMMIT 첫 시도의 실제 잠금, 이후 해제 | 수정 전 2 FAIL → 수정 후 PASS. 두 INSERT의 bytes·ID·시각 동일, 접수/상태/큐 각 1건, commit observer 1회 |
| 두 시도 모두 잠금 유지 | DeepStream/OpenCV 모두 예외 전달, `events_failed` 1회, 큐·원본·상태 0건. 잠금 해제 후 복구 worker만 실행해도 미접수 입력은 생기지 않음 |
| 위 실패 후 호출자가 명시적으로 다시 제출 | 접수/큐 각 1건. 자동 영구 재전송이나 종료 후 복원으로 해석하지 않음 |
| 실제 페이지 제한 SQLITE_FULL | 1회 시도 후 실패, 원본/상태/commit 알림 없음. 호스트 파일시스템을 채우지 않음 |
| LOCKED·IOERR·코드와 문구 불일치 | BUSY와 구분, 임의의 OperationalError를 재시도하지 않음 |
| 일반 event store 호출 | 기본 1회 시도 유지 |

최초 저장을 끝내 못 한 입력은 아직 DB에 없다. 호출자는 이를 실패로 취급해야 하며,
영상 재취득/역압/영구 재전송 보장은 이번 범위에 포함되지 않는다. 이미 접수한 영상의
기록 전용 복구와 물리 경보 재실행 금지는 기존 정책을 유지한다.

## 활성 쓰기 표본 결과

임시 source/consumer SQLite와 프로세스 내부 전달 모형이다. 미리 준비한 영상 8건에
영상 20건·센서 20건을 두 생산자에서 추가하고, 20ms 주기의 관제 worker를 함께 실행했다.
관제 INSERT/UPDATE가 생산자 실행 구간에 발생한 것을 SQL trace로 확인했다.

| 항목 | 첫 단독 시험 | 최종 회귀 내 시험 |
|---|---:|---:|
| 영상 신규 시도/접수 | 20 / 20 | 20 / 20 |
| 영상 내부 BUSY 재시도 로그 | 2회 | 1회 |
| 영상 시험 생산자의 추가 재제출 | 0회 | 0회 |
| 센서 시험 생산자의 명시적 잠금 재시도 | 1회 | 0회 |
| 관제 서비스 실패 주기 | 1회 | 2회 |
| 생산자 실행 중 관제 쓰기 문장 | 47회 | 35회 |
| 최종 저장·ID/hash/ACK 대조 | 48 / 48 | 48 / 48 |
| 영상 접수 최대 호출시간 | 0.445초 | 0.266초 |
| 전체 경과시간 | 7.629초 | 6.343초 |

관제 주기의 잠금 실패는 후속 실행으로 복구됐다. 센서 시험 생산자는 명시적 재전달을
허용하며 영상의 내부 1회 재시도와 구분한다. source 원본·canonical 각 48건,
기록 대조 48건 일치, 영상 물리 출력 작업 28건은 held다. DB 재오픈 후 전달 상태도 보존했다.
관제 성공 ACK를 얻기 전에 성공으로 표시하지 않는 기존 경로를 검증했다.

이 작은 표본은 처리량 상한·장기 안정성·실장비 병목 합격 기준이 아니다. 관제 실패 주기도
실제로 관측됐으므로 I22를 전체 해결로 닫지 않는다. 기존 1초/20ms 주기 회귀는 각각
24건 저장, 완료 후 빈 worker 40회에서 쓰기 transaction 0회를 유지했다.

## 증거와 재실행

- [수정 전: 2 FAIL](evidence/video_intake_busy_before_20260922.xml)
- [활성 쓰기 단독 시험](evidence/monitoring_active_contention_20260922.xml)
- [최종 200 PASS](evidence/video_intake_busy_final_20260922.xml)
- [측정값·파일 SHA-256·영향 범위](evidence/video_intake_busy_validation_20260922.json)
- [이번 변경 patch](evidence/video_intake_busy_change_20260922.patch)

Ruff·Black check·Python compile은 변경 앱/신규 테스트 4파일 모두 PASS다.
두 그래프에서 관련 미추적 모듈의 노드를 찾지 못해 실제 호출부·작업 시작 시 저장한 diff와
회귀를 대조했다. 그래프의 risk 0을 영향 없음으로 사용하지 않았다.

```bash
rtk proxy .venv/bin/python -m pytest \
  tests/test_video_intake_busy.py tests/test_monitoring_active_contention.py \
  tests/test_event_ingest_store.py tests/test_ingest_store.py \
  tests/test_video_event_ingest.py tests/test_event_dispatcher.py \
  tests/test_external_ingest_durable.py tests/test_sensor_record.py \
  tests/test_record_delivery.py tests/test_monitoring_delivery.py \
  tests/test_monitoring_worker.py tests/test_monitoring_sender_crash.py \
  tests/test_monitoring_capacity.py tests/test_monitoring_contention.py \
  tests/test_monitoring_contention_load.py tests/test_monitoring_commit_observations.py \
  tests/test_monitoring_reconciliation.py tests/test_monitoring_snapshot.py \
  tests/test_monitoring_freshness.py tests/test_monitoring_freshness_snapshot.py \
  -q -o addopts=--tb=short -o junit_family=xunit1 \
  --junitxml=/tmp/video-intake-busy-rerun.xml
```

이번에는 실제 MQTT/LAN·GPU 추론·실물 출력 시험을 추가 수행하지 않았다.
다음 내부 단위는 **혼합 입력 적체 누적·재시작 복구와 DB 증가량/배수시간 계측**이다.
D02/D04 실센서 유효시간/부하, D04/D06 최대 대기량·보관/장애 시간의 합의와 장시간
실환경 시험은 미완료다. P2.4 상태와 WBS DONE 8/33(작업 수 기준 24.2%)은 유지한다.
