# P2.4 — 혼합 적체·실제 프로세스 재시작·저장량 검증

검증: 2026-09-22 KST · 계획 v0.27 · **이번 내부 단위 완료 / P2.4 IN_PROGRESS**

영상 150건·센서 150건을 누적하고, 서로 다른 두 저장 경계에서 프로세스를 강제 종료했다.
새 프로세스가 같은 DB를 열어 **300건 모두 저장 ACK까지 복구**했다. 원본과 전달 bytes·ID·hash를
보존했고, 누락·논리 중복·충돌은 0건이다. 신규 시나리오 1개를 포함한 이번 관련 회귀는
**108 PASS / 0 FAIL / 0 SKIP**다. 이전 단위의 200 PASS와 시험 파일 선택 범위가 다르다.

## 구조적으로 확인한 범위

| 경로 | 이번 확인 | 남은 범위 |
|---|---|---|
| 영상 접수 → 원본 → 정규화 → 공통 기록 | 실제 `VideoEventIngest`와 SQLite 사용. 복구 경로의 물리 출력 작업은 held | CCTV/DeepStream 추론·운영 부하·새 저장 경로 운영 전환 |
| 센서 접수 → 공통 관측 | 합성 입력과 미리 해석한 온습도 projection을 `SensorRecordWorker`로 변환 | 실제 AP/MQTT·TLV bytes 해석·타 부서 규격 대조 |
| 공통 기록 → 관제 전달 → 수신 DB → 저장 ACK | 실제 store/worker·DB, 프로세스 종료와 재오픈, 고정 목록 대조 | 통신은 프로세스 내부 `Broker`/`Reader` 모형. 이번에 실제 MQTT/TLS/LAN을 재시험하지 않음 |
| 이벤트 → EdgeX → 출력 장치 | 이번 실행 범위 밖. 영상 물리 작업 150건은 held·시도 0회 확인 | P1.3/P3 우선순위·만료·출력 ACK·실물 대조. I15의 t34954 Core Data 매핑도 OPEN |

구조도는 연결 관계를 설명한다. 이번 통과가 전체 운영 경로의 정상 동작을 보증하지는 않는다.
공통 저장과 관제 복구를 먼저 확인하고, 출력 정책과 EdgeX 장치 검증으로 이어갈 수 있는 상태다.

## 시험 조건과 종료 지점

- 입력은 50쌍씩 3회 누적했다. 각 입력의 원본에 2,048bytes의 합성 padding을 넣었다.
  영상은 metadata에도 포함되며 센서 공통 관측 크기와 다르다. 영상 프레임/파일을 저장한 시험은 아니다.
- 순차 접수로 적체와 재시작을 분리해 확인했다. 시험 생산자의 추가 접수 재시도는 없다.
  동시 쓰기 경합은 [직전 별도 시험](VIDEO_INTAKE_CONTENTION.md)의 근거를 유지한다.
- 수신 상태 조회를 TimeoutError로 실패시켰다. 100→200→300건 모두 uncertain·ACK 없음이며,
  확인되지 않은 건을 완료 처리하거나 이 단계에서 발행하지 않았다.
- `spawn`으로 별도 Python 프로세스를 실행했다. 매번 store/worker/receiver를 DB에서 새로 구성했다.
  강제 종료는 `os._exit(42)`이며 호스트 전원 차단 시험은 아니다.
- SQLite DELETE journal·synchronous EXTRA를 유지했다. 재시도/lease 시각은 주입한 시계로 전진시켰다.
  실제 장시간 단절을 기다리거나 운영 재시도 설정을 변경하지 않았다.

| 실행 | 종료 지점 | 해당 프로세스 발행 | 해당 프로세스 저장 ACK 반영 | 종료 코드 | 프로세스 내부 실측 |
|---|---|---:|---:|---:|---:|
| 첫 실행 | 수신 DB 37번째 commit 후, 송신 PUBACK 기록 전 | 37 | 0 | 42 | 2.685초 |
| 첫 재시작 | 송신 DB 51번째 저장 ACK commit 후 | 263 | 51 | 42 | 20.665초 |
| 두 번째 재시작 | 남은 ACK 조회·반영 후 정상 종료 | 0 | 249 | 0 | 8.372초 |

세 프로세스 PID는 시험 실행 내부에서 14/15/16으로 서로 달랐다. 총 발행은 300회이며
이미 수신된 건은 저장 상태를 재조회해 추가 발행 없이 복구했다. 송신측 만료 lease 1건도 회수했다.
관제 조회·처리 시도 장부는 장애 중 600회와 복구 중 600회를 합쳐 1,200회다.
이 장부의 보관 비용도 DB 증가량에 포함된다.

## 시간·DB 증가량

| 시점 | 송신 DB bytes | 수신 DB bytes |
|---|---:|---:|
| 빈 DB·binding 준비 | 139,264 | 49,152 |
| 100건 적체 | 1,585,152 | 49,152 |
| 200건 적체 | 3,026,944 | 49,152 |
| 300건 적체 | 4,493,312 | 49,152 |
| 300건 저장 ACK 완료 | 5,132,288 | 2,023,424 |

적체 형성은 **35.377초**, 이후 첫 프로세스 시작부터 두 차례 종료/재시작과 최종 송신 상태
재조회까지의 경과시간은 **33.748초**다. 이 복구 시간에는 프로세스 시작 비용이 포함되지만
실제 장애 대기·네트워크 지연·설정된 재시도 대기시간은 포함되지 않는다. 현장 처리량이나 지연 SLO로 쓰지 않는다.

송신 DB는 약 **4.29MiB→4.89MiB**, 최종 수신 DB는 **1.93MiB**다.
원본 UTF-8 데이터 합계는 658,240bytes, wire 합계는 853,573bytes이며 DB에는 정규화 기록,
전달/시도 장부·ACK·인덱스가 함께 들어간다. 별도 대조 snapshot DB는 2,207,744bytes다.
송신·수신 DB의 최종 free page는 0이며 양쪽 `integrity_check`는 ok다.

모든 크기는 트랜잭션이 끝난 뒤의 표본이다. 임시 journal 최대 크기·파일시스템 여유량·
장기 보관/삭제/VACUUM 효과는 측정하지 않았다. 이 300건의 평균을 일/월 보관 용량으로 곧바로 환산하지 않는다.

## 검증과 변경 파일

- 신규 [test_monitoring_backlog_restart.py](../../tests/test_monitoring_backlog_restart.py):
  원본 300건 불변, namespace별 canonical/수신 기록 각 150건, 전체 wire bytes/hash와 ACK의 ID·수신 기록 ID 대조.
  고정 목록 matched=300, missing/conflict/unexpected/unverifiable=0, 미정규화/전달 미생성=0.
  물리 경보 작업 150건은 held·attempts=0이며 자동 재실행하지 않았다.
- [JUnit 결과](evidence/monitoring_backlog_restart_20260922.xml): 관련 108개, 108.403초.
- [측정값·환경·파일 해시](evidence/monitoring_backlog_restart_validation_20260922.json):
  프로세스별 결과, DB 페이지/크기, ACK 대조, 명시적 시험 한계.
- Ruff·Black check·Python compile 모두 PASS. `src`의 Python 204파일은 작업 시작 시 해시와 동일하다.
  앱·API·DB schema·운영 설정을 바꾸지 않았다. 노트북 배포·서비스 재시작·실물 명령도 없다.
- 그래프에서 신규 시험과 관제 worker 노드를 찾지 못해 실제 호출 경계·테스트·파일 해시를 대조했다.
  그래프 risk 0을 영향 없음의 근거로 사용하지 않았다.

재실행:

```bash
rtk proxy .venv/bin/python -m pytest \
  tests/test_monitoring_backlog_restart.py tests/test_monitoring_sender_crash.py \
  tests/test_monitoring_worker.py tests/test_monitoring_delivery.py \
  tests/test_monitoring_receiver.py tests/test_monitoring_reconciliation.py \
  tests/test_monitoring_snapshot.py tests/test_monitoring_capacity.py \
  tests/test_sensor_record.py tests/test_video_event_ingest.py \
  tests/test_video_intake_busy.py \
  -q -o addopts=--tb=short -o junit_family=xunit1 \
  --junitxml=/tmp/monitoring-backlog-rerun.xml
```

## 완료 판정과 다음 작업

혼합 적체·두 차례 실제 종료 복구·DB 증가량 계측의 내부 실행 단위는 완료다.
P2.4 전체는 **IN_PROGRESS**, WBS **DONE 8/33(작업 수 기준 24.2%)**를 유지한다.
D02 입력률/패킷, D04 지연/대기량, D06 보관량/장애시간과 실제 환경의 장시간 검증은 남아 있다.
최초 원본 commit에 실패해 접수되지 않은 입력을 복구하는 보장도 이번 범위 밖이다.

**다음 개발은 READY인 P1.3 우선순위·만료·중복/재실행 계약**으로 전환한다.
P2.4의 미정 운영 수치를 임의로 확정하거나 같은 모형 시험을 계속 늘리지 않는다.
고위험 이벤트의 장치 실행 순서, 오래된 이벤트의 기록 보존과 출력 금지, 재시작 시
물리 경보 재실행 여부를 계약으로 정리한다. 실제 P3.1 구현 진입에는 P1.3와 P2.4 완료 조건을 다시 확인한다.
