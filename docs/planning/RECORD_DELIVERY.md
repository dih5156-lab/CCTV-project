# P2.2 — 기록 전달 worker·lease·MQTT 수신 확인

2026-09-18 · **기록 전달 모듈 로컬/격리 검증 완료 / P2.2 IN_PROGRESS / 운영 미적용**

> 2026-09-21 후속: [직접 TLV 공통 관측·기록 전달 및 P2.2 DONE 판정](TLV_RECORD_DELIVERY.md).
> 센서 envelope는 observation_id를 사용하며 기존 이벤트 event_id는 유지한다.
> 아래 미연결/잔여 설명과 시험 수는 09-18 당시 이력이다. 기록 전송 worker의 자동 실행은 여전히 별도다.

공통 이벤트를 목적지별 작업으로 만들고, 작업을 짧게 점유한 뒤 DB 밖에서 MQTT로 전달한다.
QoS 1 PUBACK를 확인해야 수신 확인 시각을 기록한다. ACK 유실·중간 종료 때는 같은 전달 ID와
같은 본문으로 재시도한다. 기존 물리 경보 작업은 재생하지 않는다.

## 1. 변경 파일과 적용 범위

| 파일 | 변경과 이유 |
|---|---|
| [record_delivery.py](../../src/record_delivery.py) | 기록 목적지 등록, 기존 canonical 이벤트의 전달 작업 생성, 원자적 lease 점유, 시도/ACK/재시도 저장, worker |
| [record_mqtt.py](../../src/protocols/record_mqtt.py) | Paho 2·MQTT 3.1.1 QoS 1 발행 및 PUBACK 대기. 기존 outbox와 별도 연결 |
| [ingest_store.py](../../src/ingest_store.py) | 센서 입력 저장소가 추가형 schema v3를 읽고 계속 기록할 수 있도록 허용 |
| [event_ingest_store.py](../../src/event_ingest_store.py) | 기존 실시간 발행기가 기록 전용 작업의 점유/완료 상태를 바꾸지 못하도록 구분 |
| [단위/DB 시험](../../tests/test_record_delivery.py) | 18개: 이행/rollback·동시 점유·늦은 응답·목적지 격리·영상 복구 연결·시각 설정 검증 |
| [실제 MQTT 시험](../../tests/test_record_delivery_mqtt.py) | 6개: 정상 PUBACK, ACK 처리 직전 단절, 실제 프로세스 종료 4지점 |

새 라이브러리·외부 API·Compose·환경설정은 추가/변경하지 않았다. worker를 기존 서비스의 시작
경로에 자동 연결하지 않았다. 호출자가 명시적으로 만든 worker의 `drain_once()` 단위로 검증했다.
현재 대상은 **canonical_records에 이미 저장된 영상/외부 MQTT 이벤트**다. 직접 TLV의 PostgreSQL
관측을 읽어 이 경로로 자동 전환하지 않는다. 해당 공통 관측 연결은 다음 P2.2 작업이다.

## 2. 저장과 전달 계약

`RecordDeliveryStore(path)`를 만들 때만 공통 DB를 v2→v3로 추가 이행한다. v1은 기존 v2 이행 뒤에
v3로 올라간다. 기존 원본·센서 상태·canonical·revision·delivery_jobs 테이블을 재작성/삭제하지 않는다.

| 추가 테이블 | 책임 |
|---|---|
| record_delivery_targets | destination에 broker/port/transport/TLS 사용 여부를 고정. 변경 시 새 destination 버전 요구 |
| record_delivery_state | 기존 job_id 참조, lease token/만료·다음 시도·ack_kind/acked_at |
| record_delivery_attempts | 매 점유의 token·회차·시작/종료·결과·오류 종류 이력 |

스키마 변경은 한 transaction이다. 중간 실패 시 v2와 기존 데이터를 유지한다. 새 코드의 센서/이벤트
저장소는 v3에서 동작하지만, **종전 v2까지만 허용한 코드로 되돌리면 v3 DB를 열 수 없다**.
운영 전에는 공유 DB를 쓰는 모든 writer 버전, 백업/복원과 대기 데이터 인계를 함께 확인한다.
`PRAGMA user_version`만 낮춰 롤백하지 않는다.

1. 지정한 namespace의 canonical 중 해당 목적지 작업이 없는 행을 최대 100개 찾는다(호출 상한 1,000).
   새 job과 전달 상태를 함께 commit한다. canonical 저장과는 별도 transaction이므로 이 사이에
   종료되면 다음 scan이 누락된 작업을 만든다. 같은 namespace/업무 ID/destination에는 작업 하나다.
2. `BEGIN IMMEDIATE` 안에서 만기가 된 작업 하나를 선택하고 새 token과 점유 시한을 기록한다.
   다른 worker는 만료 전 같은 작업을 점유하지 못한다. 전달 네트워크 대기는 DB transaction 밖이다.
3. PUBACK 후 현재 token과 만료 시각을 대조해 완료를 기록한다. 이전 worker의 늦은 결과는 적용하지 않는다.
4. 불확실한 결과는 원본/작업을 남기고 1·2·4·8·16·32·60초 간격으로 재시도한다(이후 60초).
   lease 만료 후 재점유는 이전 시도에 `lease_expired`를 기록한다. 재시도 횟수만으로 삭제하지 않는다.

| 상태 | 해석 |
|---|---|
| 기존 held/pending/sending/submitted/uncertain 경보 작업 | 이 worker의 대상이 아님. 기존 의미 유지 |
| 기록 작업 pending | 기록 전달 대기 |
| 기록 작업 sending | lease 점유 중, 아직 확정 안 됨 |
| 기록 작업 uncertain | 실패/시간 초과/잘못된 응답, 다음 시도 대기 |
| 기록 작업 submitted + ack_kind=broker_puback + acked_at | MQTT PUBACK까지 확인. 관제 DB 저장/화면/장치 성공은 아님 |

`submitted` 하나만 보고 ACK가 확인됐다고 판단하지 않는다. 기존 발행기의 submitted는 함수 반환 기준이다.
Paho의 발행 결과 rc=0만 사용하지 않고 `wait_for_publish()` 뒤 `is_published()`를 확인한다.
QoS 1의 완료 범위는 PUBACK다. [Paho 공식 client 문서](https://eclipse.dev/paho/files/paho.mqtt.python/html/client.html#paho.mqtt.client.Client.on_publish).
시험 브로커 persistence는 꺼져 있으므로 broker 정전 후 데이터 보존을 입증한 결과가 아니다.

## 3. 목적지·본문·중복 범위

목적지는 `record-mqtt:cctv/records/v1/<sink>`만 허용한다. 실제 토픽은 `cctv/records/v1/<sink>`다.
기존 `cctv/ai/events`와 wildcard 목적지는 거부한다. 운영에서는 이 토픽의 소비자/ACL이
기록 전용인지 확인해야 한다. 토픽 이름만으로 임의 외부 구독자의 동작까지 막을 수는 없다.

전달 본문은 다음 내부 모형이다. 실제 외부 관제 계약 확정은 P1.4/P2.3 범위다.

```json
{
  "schema_version": "1.0",
  "delivery_id": "job_...",
  "source_namespace": "video_detection",
  "event_id": "evt_...",
  "record": {"event_id": "evt_...", "type": "fall"}
}
```

전송 후 ACK 저장 전 종료되면 메시지가 두 번 도착할 수 있다. 같은 작업의 재전송은 delivery_id와
본문이 같으며, 수신측의 멱등 저장 또는 결과 재조회가 필요하다. 이번 모듈은 **최소 한 번 전달 시도**를
구현하며 관제에 정확히 한 번 저장되는 것을 보장하지 않는다. 목적지 버전이 바뀌면 새 작업/ID가 된다.
broker 주소 등 endpoint 변경은 같은 destination으로 허용하지 않으며 비밀번호는 DB에 저장하지 않는다.

## 4. 검증 결과

[최종 JUnit](evidence/record_delivery_final_20260918.xml): **427 PASS / 실패·오류 0 / SKIP 5, 30.90초**.
총 432개 중 신규 24개와 기존 영상·센서 저장소·MQTT·공통 이벤트·processor 인접 회귀를 실행했다.
제외 5개는 pyds가 없는 환경의 기존 Jetson 전용 시험이다.
Ruff 6파일·Black 6파일·앱 4파일 py_compile PASS.
[검증 요약·소스 해시](evidence/record_delivery_validation_20260918.json).

| 실제 자식 종료 위치 | 복구 확인 |
|---|---|
| 기록 작업 생성 commit 직후 | canonical/job 보존, 재시작 뒤 전달 |
| lease claim commit 직후 | 실제 lease 만료를 기다린 뒤 재점유·전달 |
| PUBACK 수신 직후, SQLite ACK 저장 전 | 같은 delivery_id/본문 재전달, 이전 시도 만료 이력 보존 |
| SQLite ACK 저장 직후 | 재시작 후 이미 확인한 작업을 다시 전달하지 않음 |

네 지점 모두 실제 `os._exit(42)`를 사용했다. ACK 유실 시험은 실제 브로커로 전달한 뒤
Paho의 PUBACK 처리 직전에 시험용으로 연결을 끊었다. 최초/재수신 본문과 DB 시도를 대조했다.
스레드 2개의 경쟁 점유·이전 token 차단, 완료 transaction 실패 시 ACK/lease rollback,
목적지별 진행, 영상 기록 복구→기록 worker 연결도 확인했다.

시험 환경은 임시 SQLite·기존 설치 Paho 2.1.0·`eclipse-mosquitto:2.0` 이미지다.
브로커는 network none·호스트 포트 없음·전용 Unix socket·64MiB/0.5CPU로 실행했다.
운영 MQTT/DB·GPU·물리 장치에 연결하지 않았고 시험 컨테이너는 종료/자동 제거했다.
처음 시험 import 경로를 수정했고, 샌드박스 연결 차단과 소켓 파일 권한 문제를 해결했다.
[소켓 권한 실패](evidence/record_transport_environment_failure_20260918.xml)를 구현 결함과 구분해 보존한다.
[저장소 구현 전 8개 실패](evidence/record_delivery_red_20260918.xml),
[worker 구현 전 2개 실패/기존 1개 통과](evidence/record_worker_red_20260918.xml),
[transport 구현 전 2개 실패](evidence/record_transport_red_20260918.xml),
[경계 시험 24개 통과](evidence/record_delivery_boundaries_20260918.xml)도 보존했다.

## 5. 호출·적용·남은 작업

서비스 자동 실행 설정은 추가하지 않았다. 연결이 허용된 **기록 전용 시험 환경**에서 호출하는 예다.
DB는 실제 운영 경로 대신 준비한 시험 DB를 사용한다. 입력은 먼저 canonical에 저장돼 있어야 한다.

```python
import os
from src.record_delivery import RecordDeliveryStore, RecordDeliveryWorker
from src.protocols.record_mqtt import MqttRecordTransport

store = RecordDeliveryStore(os.environ["RECORD_TEST_DB"])
sender = MqttRecordTransport(os.environ["RECORD_TEST_MQTT_SOCKET"], transport="unix")
worker = RecordDeliveryWorker(
    store, sender, namespace="video_detection",
    destination="record-mqtt:cctv/records/v1/test-receiver",
)
confirmed = worker.drain_once(limit=100)
```

호출자 소유 반복문이 다시 호출해야 재시도와 신규 이벤트가 진행된다. 기존 서비스에 붙이는 작업,
정상 종료·운영 지표·이미지 패키징은 아직 수행하지 않았다. 전송 시도마다 독립 Paho 연결을 만들고
정리해 이전 연결의 메모리 발행 대기가 재시도 정책과 겹치지 않게 했다. 연결 비용과 canonical scan의
규모별 비용은 P2.4에서 측정한다. 네트워크 지연/OS DNS 지연·시계 변경에 따른 lease 영향,
장기 용량/보관·공정성·실제 TCP/TLS/ACL·broker 재시작/정전·현장 처리량은 미검증이다.

다음은 **직접 TLV 센서 관측을 공통 레코드·기록 전달 작업에 연결**하는 P2.2 범위다.
생산자 ID 미확정 D02/I19는 별도 재수신의 업무 중복 문제로 계속 추적한다. 관제 최종 수신/저장 ACK와
조회는 P1.4/P2.3, 물리 경보 유효시간·선점·재실행 정책은 P1.3/P3에서 검증한다.
따라서 이번 기록 전달 모듈은 로컬 완료, P2.2 전체는 IN_PROGRESS로 유지한다.
