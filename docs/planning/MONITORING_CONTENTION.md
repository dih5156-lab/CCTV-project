# P2.4 — 관제 대기 조회의 쓰기 경합 완화

검증: 2026-09-21 KST · 계획 v0.19 · **로컬 수정·회귀 완료 / P2.4 IN_PROGRESS / I22 OPEN**

09-22 후속: [영상 최초 BUSY 1회 재시도·활성 혼합 쓰기 검증](VIDEO_INTAKE_CONTENTION.md)을 완료했다(200 PASS).
아래는 당시의 검증 기록이다. 현재 미접수 정책·잔여 운영 부하는 후속 보고서와 [진행 기록](PROGRESS.md)을 따른다.

처리할 기록이 없는 관제 worker가 반복해서 SQLite 쓰기 잠금을 요청하는 동작을
수정했다. 빈 worker 호출 40회에서 쓰기 트랜잭션이 **80회 → 0회**로 줄었다.
관련 회귀는 **197 PASS / 0 FAIL / 0 SKIP**, 신규 시험은 9개다.
실제 부하의 모든 쓰기 경합이나 영상 최초 접수 실패를 해결했다는 뜻은 아니다.

## 1. 원인·수정 범위

기존 `enqueue_monitoring`과 `claim_monitoring`은 대상 유무를 확인하기 전에
`BEGIN IMMEDIATE`로 쓰기 잠금을 요청했다. 이미 등록된 목적지도
`INSERT OR IGNORE`를 반복했다. 입력 writer가 잠금을 가지고 있으면 빈 관제 조회도
250ms 후 `database is locked`로 실패하는 것을 두 시험에서 재현했다.

[수정 파일](../../src/monitoring_delivery.py)은 다음 순서로 동작한다.

1. 기존 목적지·origin·consumer·송신/조회 endpoint를 읽기로 검증한다.
2. 등록된 목적지에 처리 대상이 없으면 쓰기 트랜잭션 없이 반환한다.
3. 신규 목적지 또는 처리 대상이 있으면 기존 쓰기 잠금을 획득하고 다시 조회한다.
   그 뒤에만 등록·전달 작업 생성·lease 할당을 수행한다.

동시 worker가 같은 기록을 생성하거나 같은 lease를 소유하지 않게 재확인한다.
첫 빈 조회 직후 도착한 데이터는 다음 주기에 발견한다. 활성 작업에는 사전 읽기가
추가되므로 전체 처리 속도 향상을 자동으로 보장하지 않는다.

공개 API·DB 스키마·SQLite timeout/journal·주기 기본값·의존성은 유지했다.
영향은 관제 목적지 등록·대기 생성·lease 획득 경로다. 운영 서비스 재시작/배포는 하지 않았다.
[이번 변경 patch](evidence/monitoring_contention_change_20260921.patch)를 보존했다.

## 2. 검증 결과와 한계

| 조건 | 수정 전 | 수정 후 최종 실행 |
|---|---|---|
| 입력 writer가 잠근 동안 빈 enqueue/claim | 2개 모두 잠금 오류 | 2개 모두 읽기 반환 |
| 1초 주기, 센서 12건·영상 12건 | 24건 저장, 시험 입력자 잠금 재시도 2회 | 24건 저장, 시험 입력자 잠금 재시도 0회 |
| 20ms 주기, 센서 12건·영상 12건 | 24건 저장, 시험 입력자 잠금 재시도 0회 | 24건 저장, 시험 입력자 잠금 재시도 0회 |
| 처리 완료 후 빈 worker 호출 40회 | 두 주기 모두 쓰기 트랜잭션 80회 | 두 주기 모두 0회 |
| 실제 격리 MQTT·수신 DB 잠금 복구 | [이전 검증](MONITORING_OUTAGE.md) | 두 시나리오 각각 ID/hash·저장 ACK 24건 일치, 누락/논리 중복 0건 |

주기 비교는 **실제 SQLite·주기 서비스 + 프로세스 내부 전달 모형**이다.
최종 회귀에는 별도 실제 MQTT 시험도 포함했다. 위 24건 표본은 처리량·장기 안정성·
현장 부하 합격 시험이 아니다. 최종 1초/20ms 조건의 전체 경과시간은 각각
5.592초/3.310초였고, 한 쌍 입력의 최대 호출시간은 0.475초/0.289초였다.
개별 호출 지연 변동이 있어 잠금 제거를 전체 지연 개선으로 확대하지 않는다.

추가 회귀는 동시 enqueue 중복 방지, 기존 concurrent claim, 새 기록 발견,
lease 만료 후 동일 ID/bytes 복구, 잘못된 binding 차단을 확인했다.
Ruff·Black check·Python compile은 변경한 앱/테스트 3개 파일 모두 통과했다.

## 3. 입력별 저장 실패 책임

| 입력 경계 | 실제로 확인한 동작 | 남은 조건 |
|---|---|---|
| 영속 MQTT 입력의 최초 저장 실패 | DB 잠금 시 ACK 생략·연결 종료·재연결 요청, 잠금 해제 후 같은 handler로 재전달하면 접수 1건·ACK 1회 | 이번 추가 시험의 재전달은 모형 client다. 실제 QoS 1 지속 세션 복구는 기존 격리 시험의 범위이며 broker 보관/생산자 조건 필요 |
| 영상 최초 접수 실패 | 예외가 호출자에게 전파되고 큐·원본 DB에 저장되지 않음. 복구 worker만 실행해도 이 입력은 생기지 않음 | 호출자의 동일 이벤트 재전송이 필요하다. 시험은 직접 재호출했고, 자동 재전송을 새로 구현하지 않았다 |
| 이미 접수된 영상의 정규화 실패 | 원본·접수 ID를 보존하고 기존 복구 경로로 처리 | [이전 잠금 시험](MONITORING_OUTAGE.md)과 구분 |

입력별 경계는 [추가 시험](../../tests/test_monitoring_contention.py)과 기존
`MqttTopicSubscriber`, TLV processor, `VideoEventIngest` 경로를 대조했다.
I22는 불필요한 관제 잠금만 로컬 개선한 상태다. 실제 입력률·활성 쓰기 경쟁,
영상 미접수 이벤트의 제한된 재시도/실패 정책은 후속 검증으로 남긴다.

## 4. 증거·후속

- [수정 전 결정적 잠금 재현: 2 FAIL/3 PASS](evidence/monitoring_contention_before_20260921.xml)
- [수정 전 주기 비교: 빈 조회 쓰기 80회로 2 FAIL](evidence/monitoring_contention_load_before_20260921.xml)
- [수정 후 선별 검증: 23 PASS](evidence/monitoring_contention_targeted_20260921.xml)
- [최종 회귀: 197 PASS](evidence/monitoring_contention_final_20260921.xml)
- [검증 요약·측정·파일 해시](evidence/monitoring_contention_validation_20260921.json)
- [주기 비교 시험](../../tests/test_monitoring_contention_load.py)

그래프 도구는 이 미추적 모듈의 노드를 찾지 못했다. 영향 없음으로 해석하지 않고
실제 변경 diff와 관련 회귀를 확인했다. 시험 브로커는 운영망과 분리한 Unix 소켓,
64MiB/0.5 CPU 상한으로 실행했고 검증 후 종료했다.

재실행은 별도 격리 MQTT 소켓을 준비하고 수행한다. 실제 MQTT 시험의 SKIP을 통과로
계산하지 않으며 새 결과는 기존 증거를 덮어쓰지 않는다.

```bash
rtk proxy env CCTV_TEST_MQTT_SOCKET=/tmp/cctv-p22-mqtt-rerun/broker.sock \
  .venv/bin/python -m pytest tests/test_monitoring_*.py \
  tests/test_sensor_record.py tests/test_record_delivery.py tests/test_record_delivery_mqtt.py \
  tests/test_video_event_ingest.py tests/test_mqtt_durable_subscriber.py \
  parser-python/tests/test_tlv_mqtt_durable.py -q -o addopts=--tb=short \
  -o junit_family=xunit1 --junitxml=/tmp/monitoring-contention-rerun.xml
```

진행 중 사용자 제보로 노트북 HTTP 연결이 회복돼, 다음 작은 단위는
[원격 수신·ACK 준비](WINDOWS_MONITORING_POC.md)로 연결한다.
I22 잔여 입력 정책·대기 용량/저장 실패·오래된 데이터 표시는 P2.4 잔여로 유지한다.
D04/D06 수치 합의·실제 관제·운영 적용은 완료하지 않았다.
