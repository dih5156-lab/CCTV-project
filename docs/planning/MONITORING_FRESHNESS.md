# P2.4 — 오래된 센서값·지연 도착 상태 조회

2026-09-21 · 계획 v0.24 · **내부 조회 실행 단위 완료, P2.4 전체 IN_PROGRESS**.

센서값의 측정 시각으로 현재 데이터 나이를 계산하고, 수신 당시 이미 오래된 값이었는지도 구분한다. 신규 30개 포함 관련 회귀 **131 PASS / 0 FAIL / 0 SKIP**다. 기준 시간과 시계 동기화는 기본값으로 가정하지 않는다. 노트북 수신기·관제 화면·실제 센서에는 이번 변경을 배포하지 않았다.

## 판단 기준

| 조회 상태 | 의미 |
|---|---|
| `fresh` | 확인된 시계와 설정된 유효시간을 기준으로 측정 후 경과시간이 기준 이내 |
| `stale` | 측정 후 경과시간이 기준을 초과 |
| `unknown` | 기준 미설정, 시계 동기화 미확인, 측정 시각 누락/오류/미래, 무결성 오류 등으로 판정 불가 |
| `not_applicable` | 이번 센서 관측 정책의 대상이 아닌 레코드 |

- 직접 `sensor_observation`과 이벤트 내부의 `observation.message_type=sensor_observation`을 지원한다.
- `observed_at_quality=valid`는 시각 형식이 유효하다는 뜻이다. 센서와 조회 서버의 시계 동기화까지 확인됐다는 뜻은 아니다.
- `SensorFreshness()` 기본값은 유효시간 없음·`clock_sync=unknown`이다. 운영 센서의 D02/D04 결정 전 임의의 정상/오래됨 판정을 하지 않는다.
- 시험의 **60초는 합성 입력 검증값**이다. 센서별 운영 기준이 아니다. `clock_sync=confirmed`도 시험 시각에만 명시한다.
- 기준과 정확히 같은 나이는 `fresh`, 초과하면 `stale`이다. `fresh`는 가장 최신 레코드 선정이나 물리 센서 정상/위험도 정상 판정이 아니다.
- 측정 시각을 수신·처리·저장 시각으로 대체하지 않는다. 0·누락·잘못된 측정값의 품질은 원본대로 유지한다.

## 어디까지 연결했는가

[상태 계산](../../src/monitoring_freshness.py)의 `SensorFreshness`를 [기존 스냅샷 조회](../../src/monitoring_snapshot.py)의 `list_snapshot`·`get_records`에 연결했다. 조회 결과의 각 item에 `freshness` 필드가 추가되고, 생성자에는 선택 인자 `freshness`가 추가된다. HTTP API 경로·DB schema·전송 payload·해시·ACK 형식은 바꾸지 않았다.

현재 조회의 `status`, `age_seconds`, `reason`, `evaluated_at`과 수신 당시의 `on_arrival`을 함께 반환한다. source snapshot의 수신 시각은 Jetson 최초 접수, consumer snapshot의 수신 시각은 수신 서버의 해당 전달 접수다. 같은 데이터라도 source에서 받은 때는 기준 이내이고 서버에 도착할 때는 기준 초과일 수 있다.

조회 때마다 상태를 다시 계산한다. 스냅샷 대상 목록과 원본은 고정되지만, 시간 경과에 따른 freshness는 고정하지 않는다. 계산 결과를 DB/wire에 저장하지 않아 재조회·재시작으로 측정 시각이나 ID가 바뀌지 않는다. 기존 binding/cursor/TTL/해시 검사도 유지한다.

예: 측정 100초, Jetson 접수 110초, 수신 서버 도착 200초, 조회 210초, 시험 기준 60초일 때:

| 항목 | Jetson 조회 | 수신 서버 조회 |
|---|---|---|
| 현재 나이 | 110초 | 110초 |
| 현재 상태 | stale | stale |
| 해당 지점 수신 당시 | fresh, 10초 | stale, 100초 |
| 저장 ACK | stored 유지 | 저장된 수신 이력 유지 |

저장 완료와 값의 신선도는 독립적이다. 오래된 값도 이력으로 보존한다. 이번 작업은 오래된 값을 삭제하거나 물리 경보를 재실행/억제하는 정책을 추가하지 않았다.

## 검증·변경 파일

- 앱: `src/monitoring_freshness.py` 추가, `src/monitoring_snapshot.py` 조회 확장.
- 테스트: [순수 판정](../../tests/test_monitoring_freshness.py), [저장·전달·스냅샷 연결](../../tests/test_monitoring_freshness_snapshot.py) 추가.
- 근거: [JUnit](evidence/monitoring_freshness_20260921.xml), [파일 해시·검증 범위](evidence/monitoring_freshness_20260921.json).
- 정상/기준 경계/KST·UTC/epoch 0/누락/오류/미래 시각/시계 미확인/기준 미설정/이벤트 내 센서 관측을 시험했다.
- 같은 스냅샷의 재조회·재오픈에서 fresh→stale 전환, 지연/역순 도착 보존, source·consumer 수신 시각 차이, 0/결측 품질 보존을 확인했다.
- 조회 전후 source/consumer/snapshot DB bytes 동일, 원본 레코드/해시·저장 ACK 유지, 무결성 실패 시 정상 판정 금지를 확인했다.
- 131 PASS와 Ruff·Black·문법 검사 PASS. JUnit 생성 시 기존 용량 시험의 `record_property`와 기본 xunit2 형식 호환 경고 6개가 있었다. 시험 실패는 아니며 재현 명령은 xunit1을 지정한다.

```bash
rtk proxy .venv/bin/python -m pytest tests/test_monitoring_freshness.py tests/test_monitoring_freshness_snapshot.py tests/test_monitoring_snapshot.py tests/test_monitoring_reconciliation.py tests/test_monitoring_metrics.py tests/test_monitoring_commit_observations.py tests/test_monitoring_service.py tests/test_monitoring_worker.py tests/test_monitoring_receiver.py tests/test_sensor_observation.py tests/test_sensor_record.py tests/test_monitoring_capacity.py tests/test_monitoring_poc_bundle.py -q -o addopts=--tb=short -o junit_family=xunit1
rtk proxy .venv/bin/python -m ruff check src/monitoring_freshness.py src/monitoring_snapshot.py tests/test_monitoring_freshness.py tests/test_monitoring_freshness_snapshot.py
```

## 남은 범위와 다음 작업

1. 09-22 후속 [영상 최초 잠금 재시도·활성 쓰기](VIDEO_INTAKE_CONTENTION.md)는 200 PASS로 내부 검증했다. 다음은 혼합 적체 누적·재시작 복구/저장량 계측이며 I22 운영 부하는 남는다.
2. D02/D04: 센서별 송신 주기·유효시간·실제 시계 동기화 확인. 현재 정책은 한 조회 객체에 하나의 기준을 적용하며 센서별 운영 정책 등록은 후속이다.
3. D04/D06: 최대 대기량·디스크 예산·보관시간·장시간 장애 기준의 합의 및 시험.
4. 실제 관제 화면/API 연결·최신값 선정·경보 만료/실행 정책은 P4/P6 및 P1.3/P3에서 연결한다.

정식 서버 이전은 사용자 지시에 따라 관제 대상이 결정되는 시점으로 미룬다. 현재 노트북 서버로 내부 전달/장애 시험을 계속한다. 이전 시 DB/저장 ACK·수신자 신원·주소/인증과 미전송 작업 처리 검증은 D06에 유지하며 완료 처리하지 않는다.
