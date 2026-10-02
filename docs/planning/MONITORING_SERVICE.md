# P2.3 — 로컬 관제 서비스 실행·종료·전체 지연 계측

2026-09-21 · 계획 v0.17 · **P2.3 DONE / 내부 모형** · R04,R10

기존 송신·수신 모듈을 명시적으로 시작하는 로컬 서비스로 연결했다. 등록한 namespace를
주기적으로 scan하고, 실제 MQTT QoS 1 전송과 별도 수신 DB 저장·ACK 조회를 이어간다.
종료 때 새 claim을 멈추고 현재 작업을 마무리하며, 재시작하면 DB의 미확인 작업을 다시 처리한다.
입력/정규화/관제 ACK의 commit 완료 후 시각을 관측해 접수→ACK commit 지연도 측정한다.

운영 자동 시작·배포 설정은 변경하지 않았다. CLI 또는 Python `start()`를 호출해야 실행된다.
외부 관제 제품·인증·화면은 D06 선정 후 연결한다. 이번 로컬 어댑터는 **Unix MQTT 소켓과
같은 호스트의 수신 SQLite 조회**만 제공하며 TCP/HTTP 관제 공개 endpoint가 아니다.

## 1. 변경 파일과 연결 경계

| 파일 | 변경·이유 |
|---|---|
| [monitoring_service.py](../../src/monitoring_service.py) | 주기 실행·오류 상태·중복 시작 방지·종료/재시작, 로컬 조합 생성 |
| [monitoring_local.py](../../src/protocols/monitoring_local.py) | MQTT 구독→수신 DB commit/격리 후 수신 transport ACK, scope별 로컬 저장 ACK 조회 |
| [run_monitoring_local.py](../../runners/run_monitoring_local.py) | 명시적 CLI, SIGINT/SIGTERM 종료와 JSON 상태/계측 출력 |
| [ingest_store.py](../../src/ingest_store.py), [event_ingest_store.py](../../src/event_ingest_store.py) | 선택형 `commit_observer`, 실제 commit 성공 후 입력/정규화 시각 관측 |
| [monitoring_delivery.py](../../src/monitoring_delivery.py) | claim 결과에 원래 접수 ID/시각을 연결. 기존 schema/전송 bytes 유지 |
| [monitoring_worker.py](../../src/monitoring_worker.py) | 종료 flag 확인, ACK commit 성공 후 관측 콜백 |
| [monitoring_metrics.py](../../src/monitoring_metrics.py) | bounded commit 표본·전체 지연 p50/p95/max·시각 품질 |

앱/runner 3파일 추가·기존 5파일 최소 연결 수정, 시험 4파일 추가다. 기존 코드의 관련
경계만 수정했고 새 라이브러리·운영 환경변수·컨테이너·systemd 설정은 추가하지 않았다.
source v4/receiver v1 스키마와 실제 wire 형식은 유지한다.
기존 caller는 `commit_observer=None`, `stop_event=None` 기본값으로 이전 동작을 유지한다.
등록된 관제 destination은 앞선 v4의 기존 broker-worker claim 보호를 그대로 따른다.

## 2. 서비스 실행과 실패 복구

- `MonitoringService.start()`는 수신 구독 준비를 확인한 뒤 전송 thread를 시작한다.
  이미 실행 중이면 중복 thread를 만들지 않는다. 수신 시작 실패는 호출자에게 전파하고 정리한다.
- 기본 주기 1초·namespace별 batch 10개다. worker 사이에서 멈춤을 확인하고,
  `Event.wait()`로 대기하므로 긴 polling sleep 때문에 종료가 지연되지 않는다.
- DB/worker 오류는 실패 cycle·오류 종류로 남기고 다음 주기에 재시도한다.
  개별 메시지의 uncertain/blocked/quarantined는 기존 영속 상태·계측으로 확인한다.
  서비스 thread가 살아 있다는 사실은 전달 완료를 뜻하지 않는다.
- `close()`는 새 작업을 막고 현재 조회/전송/commit을 마무리한다. 기본 worker join 15초를
  넘으면 TimeoutError와 `stopping` 상태를 유지하며 재시작을 거절한다. 진행 작업이 끝난 후
  다시 close해야 한다. 성공적으로 멈춘 뒤에만 수신 MQTT thread를 닫는다.
- 같은 객체/새 프로세스 모두 저장된 origin·binding·ID·bytes를 재사용한다.
  비정상 종료 시 미확인 lease는 기존 만료 규칙으로 회수한다. 기존 성공 ACK를 취소하지 않는다.
- 입력·센서 정규화·AI 추론 thread는 이 서비스가 소유하지 않는다. P2.1/P2.2의 각 writer가
  canonical 기록을 만들고, 이 서비스가 이후 기록 전달을 맡는다. 물리 장치 job은 실행하지 않는다.

로컬 수신기는 등록 topic의 QoS 1·retain=false만 받는다. 레코드 commit 또는 invalid 원본의
격리 commit이 성공한 뒤 MQTT 수신 transport ACK를 보낸다. DB 실패는 ACK 없이 연결을 닫고
서비스가 재연결한다. clean session을 쓰므로 broker 구독 세션만으로 복구를 보장하지 않는다.
source의 미확인 장부·동일 ID 재전송이 복구 기준이며 PUBACK를 consumer ACK로 대체하지 않는다.

ACK reader는 로컬 SQLite의 실제 저장 장부를 조회한다. DB 실패와 미저장(None)을 구분한다.
SQLite 잠금 대기 0.25초보다 작은 reader timeout은 지원하지 않아 조합 생성 때 거절한다.
namespace 목록·로컬 reader endpoint/DB 경로·consumer/destination 변경 조건은
[송신 모듈의 고정 binding 규칙](MONITORING_SENDER.md#2-source-저장과-이행-조건)을 따른다.

## 3. commit 관측과 전체 지연

`CommitObservations.committed`를 입력 writer의 `commit_observer`에 전달하면 같은 ID의
`ingest_commit_completed_at`, `normalize_completed_at`을 실제 commit 후 관측한다.
관제 worker는 `consumer_ack_persisted_at`을 source ACK commit이 성공한 다음 관측한다.
commit 실패 시 해당 성공 marker를 만들지 않는다. 콜백 오류는 별도 오류 수로 남기고
이미 commit된 접수를 실패로 되돌리거나 새 ID로 다시 접수하지 않는다.

```python
from src.monitoring_metrics import CommitObservations
from src.sensor_record import SensorRecordStore

observations = CommitObservations(clock_sync="unknown")
inputs = SensorRecordStore(source_db_path, commit_observer=observations.committed)
# 같은 DB의 영상 입력은 EventIngestStore(..., commit_observer=...)로 연결할 수 있다.
```

관측 시각은 callback 진입 때의 wall clock이며 commit 완료 직후의 관측값이다.
DB transaction 안의 기존 `acked_at`을 commit 완료 시각으로 이름만 바꾸지 않는다.
전체 지연은 `ACK commit 관측 시각 - 최초 원본 received_at`이다. 재시도·재시작 대기도 포함한다.
`clock_sync='confirmed'`일 때만 유효 p50/p95/max에 포함하고 기본 unknown·음수/잘못된 시각은
별도 집계한다. 실제 시각 동기화 확인은 호출자 책임이며 CLI flag가 동기화를 수행하지 않는다.

기본 최근 1000개·최대10000개 commit 표본을 현재 프로세스 메모리에만 둔다.
ID는 구조화 표본 필드이며 metric label이 아니다. 오래돼 제외된 표본 수와 측정 창을 보고한다.
재시작 전 commit marker, hook이 연결되지 않은 기존 writer의 marker, commit 직후 관측 전에
프로세스가 종료된 marker는 복원하지 않는다. 과거 성공 commit 시각은 unknown이며
이미 완료한 기록에 새 지연을 꾸며 넣지 않는다. 데이터/ACK 장부의 내구성과 표본의 보존은 별개다.
제공된 콜백은 짧은 메모리 작업이며 사용자 정의 콜백도 입력을 오래 막지 않아야 한다.

source DB의 `source_metrics().not_measured`는 **DB 필드만으로 확정할 수 없는 항목**이다.
현재 프로세스의 새 관측은 `commits.report()`에서 별도로 확인한다. 수신 중복 횟수는 로컬
receiver 상태에서 확인하며 프로세스 재시작 시 초기화된다. UI displayed_at은 여전히 미계측이다.

## 4. 실행 방법과 적용 조건

개발용 DB·독립된 Unix MQTT 소켓을 준비한 뒤 다음과 같이 명시적으로 실행한다.
실제 데이터와 reader 경로를 바꾸면 binding이 달라지므로 임시 경로를 운영 경로로 재사용하지 않는다.

```bash
rtk proxy .venv/bin/python runners/run_monitoring_local.py \
  --source-db /tmp/cctv-local-source.db \
  --receiver-db /tmp/cctv-local-receiver.db \
  --consumer-id local-poc-v1 \
  --destination record-mqtt:cctv/records/v1/local-poc-v1 \
  --mqtt-socket /tmp/cctv-p22-mqtt-service/broker.sock \
  --namespace video_detection --namespace tlv_mqtt
```

`--run-seconds 30`은 제한된 시간 실행, 기본값은 SIGINT/SIGTERM까지 실행이다.
시작/종료 시 JSON 상태와 표본을 stdout에 남긴다. 주기·ACK 대기·timeout·lease는
명시적 옵션이며 현장 지연 합격 기준을 뜻하지 않는다. 수신 DB를 source와 같은 파일로
지정하면 거절한다. 단순 `--help`는 DB를 열거나 MQTT에 연결하지 않는다.

source store 생성은 앞선 v4 이행을 수행할 수 있다. 운영 적용 전 모든 공통 DB writer의
v4 호환 코드, 기존 worker 전환, DB mount/백업·binding 보존·소켓 권한·모니터링 수집을
확인해야 한다. 이번 실행은 임시 DB·격리 브로커만 사용하며 운영 이행/자동 시작은 하지 않았다.
이 CLI는 인증된 외부 관제 API나 운영용 프로세스 관리자 설정을 대신하지 않는다.

## 5. 검증과 P2.3 완료 판정

신규 시험: [commit 경계](../../tests/test_monitoring_commit_observations.py),
[서비스 lifecycle](../../tests/test_monitoring_service.py),
[로컬 수신/조회](../../tests/test_monitoring_local.py),
[실제 MQTT·CLI](../../tests/test_monitoring_service_mqtt.py).
commit 후 다른 connection의 조회, 관측 실패 격리, 중복 시작/종료 timeout/재시작,
DB 실패 복구·수신 격리 실패·취소된 worker·잘못된 설정의 DB 미생성을 검증한다.

실제 Mosquitto(네트워크 none·포트 미노출·64MiB/0.5CPU·임시 Unix 소켓)에서 다음을 확인했다.

1. 센서/영상 입력→정규화→주기 전송→수신 DB→ACK commit, 0/결측 보존 및 snapshot 일치.
2. 수신 DB 실패 후 재연결·같은 ID/bytes 재전송, 저장 완료까지 복구.
3. 실제 CLI 프로세스의 SIGTERM 종료와 새 프로세스 재시작·미전달 기록 처리.

최종 [JUnit](evidence/monitoring_service_final_20260921.xml)과
[검증 요약·해시](evidence/monitoring_service_validation_20260921.json)에 결과를 기록한다.
**277 PASS / 실패·오류 0 / 5 SKIP, 90.24초**. 신규 19건 모두 통과했고 Ruff·Black check·
py_compile 변경 12파일 PASS다. SKIP은 `CCTV_TEST_PG_DSN`이 필요한 기존 PostgreSQL 센서 시험으로,
이번에 격리 PostgreSQL을 실행하지 않아 제외됐다. SQLite/로컬 MQTT 신규 기능 시험은 SKIP하지 않았다.
[실제 MQTT 10건 예제](evidence/monitoring_service_sample_20260921.json)에도 sensor/video 각 5건의
commit 표본·전체 지연·누락 대조를 남겼다. 동기화 조건은 같은 호스트의 해당 시험 구간에 한정한다.
실제 입력 장비나 운영 모델 추론 품질, TCP/TLS/관제 인증·화면, 전원 차단·디스크 소진·
장기 장애/부하·현장 지연 합격은 이 시험 범위에 포함하지 않는다.

P2.3의 내부 완료 기준은 센서/AI 기록→모형 관제 저장/조회, 재전송·중복 식별,
출력 장비와 독립적인 구간/전체 지연 측정이다. 수신·송신·snapshot·서비스의 검증 근거를
합쳐 로컬 범위를 닫고, 외부 선정/적용 조건은 D06·I10/I13/I17·P6.1에서 계속 추적한다.
기간/device 일반 이력 필터, 표본/manifest 장기 보관·정리와 용량 예산은 후속 범위다.
다음 작업은 P2.4 저장/관제 장애·용량 시험이며 D04 수치 없이 성능 PASS를 판정하지 않는다.
