# P2.4 — 영상 최초 저장 잠금 재시도·활성 쓰기 경합 검증

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
