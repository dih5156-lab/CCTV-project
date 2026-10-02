# P2.3 — 관제 송신 상태·저장 ACK·재전송 구현

2026-09-21 · 계획 v0.15 · **P2.3 IN_PROGRESS** · R04,R10

> 후속 v0.16: [고정 목록 대조·적체/구간 지연](MONITORING_RECONCILIATION.md)을 구현·검증했다.
> 아래는 송신 실행 단위 완료 시점의 기록이며, 현재 잔여 범위는 후속 보고서/진행 기록을 따른다.

[관제 수신 모듈](MONITORING_RECEIVER.md)에 이어 송신측 저장·복구 실행 단위를 구현했다.
최초 전송 전에 전달 ID·전체 payload bytes·SHA-256을 고정하고, broker PUBACK와
관제 DB 저장 ACK를 각각 기록한다. PUBACK 이후에도 관제 저장 여부를 조회하며,
미저장이면 같은 ID와 bytes로 재전송한다. 기존 기록 내용이 변경돼도 전송 본문은 바뀌지 않는다.

이 실행 단위의 로컬 구현·격리 검증은 완료했다. snapshot 누락 대조·지연/적체 계측·
서비스 시작/종료 연결·통신 어댑터가 남아 **P2.3 전체는 IN_PROGRESS**다.
운영 DB·운영 서비스·실장비에는 적용하지 않았다.

## 1. 변경과 영향 범위

| 파일 | 책임·변경 이유 |
|---|---|
| [monitoring_delivery.py](../../src/monitoring_delivery.py) | source DB v4, origin/consumer/endpoint 고정, 전송 bytes·consumer ACK·시도 이력 영속화 |
| [monitoring_worker.py](../../src/monitoring_worker.py) | 상태 조회→필요 시 재전송, ACK 검증/저장, lease·재시도·오류 격리 |
| [ingest_store.py](../../src/ingest_store.py) | 기존 접수 코드가 v4 공통 DB도 열 수 있도록 지원 버전 확장 |
| [record_delivery.py](../../src/record_delivery.py) | 관제 worker에 등록된 destination은 기존 broker 전용 worker가 claim하지 않도록 보호 |
| [송신 DB 시험](../../tests/test_monitoring_delivery.py), [worker 시험](../../tests/test_monitoring_worker.py) | v4 이행/쓰기 실패 rollback·경합·고정 본문·ACK 오류·조회 장애·다른 기록 진행 |
| [종료 복구 시험](../../tests/test_monitoring_sender_crash.py), [실제 MQTT 시험](../../tests/test_monitoring_delivery_mqtt.py) | 프로세스 종료 5경계, 격리 브로커 PUBACK 후 늦은 관제 수신·저장 ACK |

앱 2파일 추가·기존 2파일 최소 수정, 시험 4파일 추가다. 새 라이브러리·운영 설정·
공개 API는 추가하지 않았다. `drain_once()`를 호출해야 동작하며 자동 실행 서비스는 아직 없다.
물리 경보 경로는 변경하지 않았고 복구 시험에서 기존 경보 job의 `held` 상태를 확인했다.

## 2. source 저장과 이행 조건

`MonitoringDeliveryStore(path)`를 **명시적으로 생성할 때** 공통 source SQLite를 v4로
확장한다. v1→v2→v3의 기존 이행 이후, v3→v4는 별도 한 transaction이다.
v4 생성 실패 시 v3 상태를 유지한다. 이미 v1에서 진행한 기존 이행까지 되돌리는 것은 아니다.
별도 모형 수신 DB 형식(application_id=1129598257, v1)은 변경하지 않았다.

| 테이블 | 보존하는 내용 |
|---|---|
| monitoring_origin | source DB의 고정 origin_id |
| monitoring_targets | destination의 consumer_id·조회 endpoint, broker endpoint는 기존 target 테이블 |
| monitoring_delivery | 최초 wire bytes/hash, 관제 대기/완료 상태·lease·다음 시도·저장 ACK |
| monitoring_attempts | 조회/전송 회차별 결과·오류·잘못된 ACK 응답 |

canonical record에서 기록 job·전송 bytes/hash를 준비하는 작업은 한 transaction이다.
준비 전 종료하면 다음 scan이 다시 찾고, 준비 후에는 저장된 bytes를 사용한다.
기존 `record_delivery_state/attempts`의 PUBACK 이력은 관제 ACK와 독립적으로 유지한다.
source DB는 기존 DELETE journal·EXTRA·FK ON·잠금 대기 0.25초를 따른다.

**향후 적용 전 조건:** 공통 DB에 접근하는 모든 writer를 v4 호환 코드로 배포하고,
기존 broker worker의 동시 실행 여부를 확인해야 한다. 이미 실행 중인 구버전 프로세스가
새 claim 보호 코드를 자동으로 갖게 되는 것은 아니다. 백업·기존 큐와 실행 버전 확인 후
서비스 단위로 전환한다. 구버전 코드로의 단순 rollback은 v4를 지원하지 않을 수 있다.
이번 작업에서는 운영 파일을 열어 이행하거나 서비스를 재시작하지 않았다.

이미 전송 시도가 있었던 destination은 최초 bytes를 증명할 수 없어 새 관제 경로로
자동 채택하지 않는다. 새 destination 버전으로 등록해야 한다. consumer·조회 endpoint·
broker endpoint의 재지정도 거절한다. origin DB 복제/초기화나 consumer 장부가 되돌아가는
복원은 새 식별 범위·대조가 필요하며 이 라이브러리가 자동 처리하지 않는다.

## 3. 처리 순서와 호출 계약

1. `enqueue_monitoring()`이 canonical record를 읽어 전송 내용을 고정한다.
2. `claim_monitoring()`이 기한 있는 lease를 잡는다. 같은 job의 동시 소유자는 하나다.
3. reader로 저장 여부를 먼저 조회한다. ACK가 있으면 범위·종류·ID·hash·시각을 검사하고
   source DB에 ACK/완료 상태를 함께 commit한다. 조회 실패를 미저장으로 취급하지 않는다.
4. 명시적으로 미저장(`None`)이면 고정 bytes를 MQTT QoS 1·retain=false로 전송한다.
5. PUBACK는 `awaiting_ack`로 기록한다. 기본 10초 뒤 조회하고 여전히 미저장이면 재전송한다.
6. 실패는 1,2,4,8,16,32,60초 상한 backoff다. 잘못된 ACK는 응답을 보관하고 완료시키지 않는다.
   잘못된 본문/충돌은 격리, 권한/등록 오류는 blocked다. 수동 해제 API는 아직 없다.

worker의 `sender`는 기존 `MqttRecordTransport`이며, `reader`는 아래 인터페이스를 제공한다.

```python
reader.endpoint  # 비밀값 없는 고정 조회 대상 식별자
reader.get_delivery_status(binding, delivery_id, timeout=5)
# 저장 장부가 있으면 ACK dict, 미저장이면 None, 조회/통신 실패는 예외
```

timeout 5초·lease 30초가 기본이다. reader도 timeout을 지키는 어댑터여야 한다.
DB 오류는 호출자에게 전파하며 lease 만료 후 재개한다. 지난 lease나 늦은 실패 응답이
이미 저장 완료된 상태를 덮지 못한다. `drain_once()`의 반환값은 관제 저장 ACK를
새로 확정한 건수이며 MQTT 발행 건수가 아니다. 자동 thread/스케줄러는 생성하지 않는다.
한 destination은 여러 namespace의 준비된 기록을 처리할 수 있지만 scan은 worker에
설정한 namespace를 대상으로 한다. 센서/영상 양쪽 scan의 서비스 연결은 후속 작업이다.

현재 검증의 reader는 같은 호스트의 모형 수신 DB 조회다. 실제 관제의 인증된 네트워크
조회 어댑터와 수신 callback 서비스는 D06/후속 실행 단위다. 관제 화면 표시 확인도 별도다.

## 4. 검증 근거

이번 신규 시험은 **23개**다. 실제 자식 프로세스를 `os._exit(42)`로 종료한 경계는
전송 준비 commit 후, claim 후, 수신 DB commit 후 source PUBACK 저장 전,
source PUBACK 저장 후, source consumer ACK 저장 후의 5곳이다.
재시작 후 ID·bytes·hash 유지, 수신 record/장부 각 1건, 기존 물리 경보 held를 대조했다.
이 5개 시험의 전송은 모형이며 source/receiver DB와 종료는 실제다.

별도로 실제 Mosquitto에서 영상 이벤트·센서 관측 2건을 시험했다. 구독자가 없을 때
PUBACK만 받고, 관제 구독 후 같은 bytes를 다시 보내 저장 ACK를 확정했다.
센서 0값·결측 품질·원본 접수 ID도 보존됐다. 네트워크는 `--network none`·포트 미노출,
임시 Unix 소켓만 사용했고 broker는 64MiB/0.5CPU로 제한했다. ACK 조회는 로컬 DB다.

최종 [회귀 JUnit](evidence/monitoring_sender_verified_20260921.xml)과
[검증 요약·해시](evidence/monitoring_sender_validation_20260921.json)에 결과를 기록한다.
**187 PASS / 실패·오류·SKIP 0, 116.48초**. 기존 수신/접수/센서/기록 worker·계약 모형·
실제 MQTT 회귀를 포함한다. Ruff·Black check·py_compile은 변경 8파일 모두 PASS다.
초기 MQTT 접근 실패는 샌드박스 및 임시 소켓 권한 문제였고 각각의 실패 XML도 보존했다.
코드 그래프는 변경 파일 8개에서 함수/흐름 0개를 반환해 현재 소스·DB 시험으로 검토했다.

브로커 없이 실행 가능한 21개 시험:

```bash
rtk proxy .venv/bin/python -m pytest \
  tests/test_monitoring_delivery.py tests/test_monitoring_worker.py \
  tests/test_monitoring_sender_crash.py -q -o addopts=--tb=short
```

실제 MQTT 시험은 테스트 전용 `/tmp/cctv-p22-mqtt-*/broker.sock`을 준비한 뒤
`CCTV_TEST_MQTT_SOCKET`으로 지정한다. 지정하지 않으면 해당 두 건은 SKIP한다.
운영 broker를 이 시험에 사용하지 않는다. 전원 차단·디스크 소진·장기 부하·p95 성능을
검증한 것은 아니며 종단 exactly-once 보장으로 확대하지 않는다.

## 5. 다음 실행 단위

- source/consumer snapshot 조회·ID/hash 누락 대조, 지연/적체 계측.
- 서비스 시작/종료·주기 실행·인증된 통신 어댑터 연결 및 전체 최소 흐름 완료 판정.
- D04 지연/부하 수치, D06 관제 제품·인증·보관 정책은 미정. I10 BLOCKED,
  I13/I17 OPEN을 유지한다. P2.4 장애/용량·현장 POC는 이후 단계다.
