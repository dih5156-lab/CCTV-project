# P2.3 — 모형 관제 수신 저장·ACK·재조회 1차 구현

2026-09-21 · 계획 v0.14 · **P2.3 IN_PROGRESS** · R04,R10

> 후속 v0.15: [송신 ACK 영속화·고정 bytes·조회/재전송](MONITORING_SENDER.md)을 추가 구현·
> 검증했다. 아래는 수신 모듈 완료 시점의 범위/이력이다. source DB v4 이행 조건과
> 현재 잔여 작업은 후속 문서와 [진행 기록](PROGRESS.md)을 따른다.

[P1.4 내부 계약](MONITORING_DELIVERY_CONTRACT.md)의 수신 모듈을 앱 코드로 구현했다.
기존 센서/AI record envelope를 별도 모형 관제 SQLite에 저장하고, 레코드와 중복
장부가 commit된 뒤 `consumer_committed` ACK를 반환한다. 같은 전달 ID의 저장 여부와
내용을 재조회할 수 있다. 새 코드의 로컬 검증이며 실제 관제 제품/운영 적용은 아니다.

**이번 실행 단위는 관제 수신 모듈이다.** 송신측 ACK 영속화·PUBACK 이후 자동 재전송,
전체 snapshot 조회/누락 대조·지연 계측·서비스 실행 연결은 다음 실행 단위에 남는다.
따라서 P2.3 전체를 DONE으로 옮기지 않는다.

## 1. 변경 파일과 책임

| 파일 | 변경한 내용·이유 |
|---|---|
| [monitoring_contract.py](../../src/monitoring_contract.py) | 등록 범위/record 식별, UTF-8 JSON·버전·ID 일치·원본 bytes 해시, 저장 ACK 검증. 잘못된 성공 확인을 차단 |
| [monitoring_receiver.py](../../src/monitoring_receiver.py) | 별도 SQLite의 레코드+전달 장부 transaction, 충돌/invalid 원본 격리, 범위별 ACK/내용 조회 |
| [계약 시험](../../tests/test_monitoring_contract.py) | 센서 observation_id와 source_record_id 구분, ACK 필수 필드/수신처/해시/시각, malformed/중복 키/비정상 숫자·문자 거절 |
| [수신 DB 시험](../../tests/test_monitoring_receiver.py) | commit·재조회·동시 중복·충돌·등록 범위·DB/격리 실패, 다른 DB 보호 |
| [입력/종료 경계 시험](../../tests/test_monitoring_receiver_integration.py) | 기존 센서/AI 기록 연결, 원본 접수→관측의 0/결측/시각 보존, 실제 자식 프로세스 종료 2지점 |

신규 앱 2파일·시험 3파일이다. 기존 입력/전송/장치 실행 코드는 수정하지 않았다.
새 라이브러리·환경변수·컨테이너·공개 HTTP/MQTT endpoint는 추가하지 않았다.
Python API를 명시적으로 호출해야 동작하며 기존 서비스 시작 시 자동 실행되지 않는다.

## 2. 저장 경계와 호환

`MonitoringReceiverStore(path, consumer_id=...)`는 **모형 수신처 전용 DB**를 사용한다.
공통 source DB/센서 PostgreSQL을 이 경로로 주지 않는다. 파일은 호출자가 지정하며
코드에 운영 경로·인증 정보가 들어 있지 않다. 파일 형식은 application_id=1129598257,
user_version=1로 식별하고 다른 DB/미지원 버전을 채택하지 않는다.

| 테이블 | 의미 |
|---|---|
| monitoring_identity | DB 데이터셋의 고정 consumer_id |
| monitoring_bindings | destination별 고정 origin_id, 수신 데이터로 자동 등록 불가 |
| monitoring_records | origin/namespace/record_type/record_id별 업무 레코드 1건 |
| monitoring_receipts | destination/delivery_id별 장부, 정확한 수신 bytes/hash와 레코드 참조 |
| monitoring_rejected | 잘못된/충돌 입력의 범위·원본 bytes·오류 코드·수신 시각 |

consumer_id는 DB 전체에 고정되므로 각 레코드 행에 중복 저장하지 않는다. 같은 sink의
origin 변경이나 기존 파일을 다른 consumer_id로 여는 요청은 거절한다. 원본 DB를 초기화하거나
장부가 되돌아가는 백업 복원은 새 consumer/destination 버전과 대조가 필요하다.
그 운영 절차를 라이브러리가 자동 수행하지 않는다.

- DELETE journal·synchronous EXTRA·foreign_keys ON·잠금 대기 0.25초를 사용한다.
  쓰기 잠금/commit/디스크 오류는 호출자에게 전파되며 성공 ACK로 바꾸지 않는다.
- 업무 레코드와 전달 장부는 `BEGIN IMMEDIATE` 한 transaction으로 저장한다.
  `with connection`의 commit까지 성공한 다음 ACK를 생성한다.
- 동일 전달 ID/bytes는 duplicate ACK, 다른 전달 ID/같은 업무 레코드는 레코드 재사용+
  새 전달 장부다. 동일 전달 ID/다른 bytes와 동일 업무 ID/다른 내용은 격리한다.
- invalid/충돌 transaction은 rollback한 뒤 별도 transaction으로 원본을 격리한다.
  격리 저장도 실패하면 SQLite 오류를 전파한다. 격리 commit 전 프로세스 종료에는 ACK가
  없으므로 송신 재시도가 필요하다. 이 재시도의 자동 실행은 다음 송신 모듈 작업이다.
- 관측의 null/0/missing/invalid와 원래 시각은 그대로 저장한다. 전달 envelope/ID 경계를
  검증하는 코드이며 모델 품질·센서 업무값 검증의 기존 책임을 대체하지 않는다.
- ACK hash는 MQTT application payload bytes 전체다. JSON 재직렬화/원본 ingest hash와
  혼동하지 않는다. record 내용 비교는 P1.4의 Python sorted JSON 표현을 따른다.

`register_binding()`은 명시적인 로컬 설정 작업이다. `accept()`의 binding은 인증된
어댑터가 제공해야 한다. 현재 네트워크 adapter/인증 채널은 구현하지 않았다.
수신 scope 검사는 다른 origin/destination의 데이터를 잘못 반환하지 않기 위한 DB 경계다.

## 3. 호출과 오류

| Python API | 결과 |
|---|---|
| register_binding(binding) | 재등록은 멱등, consumer/sink/origin 불일치는 binding_mismatch |
| accept(binding, wire_bytes) | commit 후 stored/duplicate ACK; invalid/충돌은 MonitoringContractError, DB 실패는 sqlite3.Error |
| get_delivery_status(binding, delivery_id) | 기존 장부이면 duplicate ACK, 장부가 없으면 None(status_unknown 의미). DB 읽기 실패는 예외 |
| get_record(binding, receiver_record_id) | 허용 범위에서 저장한 원래 record와 ID/시각, 없으면 None |
| validate_ack(ack, binding, parsed_record) | 필수 값·종류·수신처·해시·timezone 시각을 검증한 dict. sender DB 저장/인증은 별도 책임 |

저장 ACK 유실 뒤 수신처를 재시작해도 같은 delivery_id로 ACK를 조회한다. 재전송해도
receiver_record_id/stored_at은 유지한다. 새로운 ack_emitted_at은 조회/응답 생성 시각이다.
이 ACK는 UI 표시·물리 출력 확인이나 source의 ACK 영속화 완료를 뜻하지 않는다.

재현 명령(운영 DB·장비 사용 없음):

```bash
rtk proxy .venv/bin/python -m pytest \
  tests/test_monitoring_contract.py tests/test_monitoring_receiver.py \
  tests/test_monitoring_receiver_integration.py \
  -q -o addopts=--tb=short --junitxml=/tmp/monitoring_receiver_recheck.xml
```

## 4. 검증 근거와 한계

신규 **28 PASS**에는 실제 source store/센서 worker가 만든 관측·AI 기록의 전달과
조회가 포함된다. 테스트가 수신 ACK를 broker ACK로 기록하지 않고, 기존 물리 경보
job이 `held`로 유지되는지도 확인했다. 실제 MQTT/HTTP 통신을 했다는 뜻은 아니다.

실제 자식 프로세스를 `os._exit(42)`로 종료한 두 경계:

1. 레코드/장부 insert 후 commit 전: 다시 열면 둘 다 rollback, 상태는 미저장,
   동일 bytes 재시도 후 각각 1건이다.
2. commit 후 ACK 생성 전: 다시 열면 장부 조회 가능, 재전송은 duplicate,
   레코드/장부 각각 1건이다.

추가로 deferred FK를 이용한 **commit 자체 실패**, 장부 insert 실패, 격리 저장 실패,
수신 DB 읽기 실패, 동시 중복 저장을 검증했다. 전원 차단/파일시스템 내구성이나
장기 잠금 경합·부하·디스크 소진 시험은 아니다.

최종 인접 회귀 결과는 [JUnit](evidence/monitoring_receiver_final_20260921.xml)과
[검증 요약·해시](evidence/monitoring_receiver_validation_20260921.json)에 기록한다.
**158 PASS / 실패·오류·SKIP 0, 62.88초**. Ruff·Black check·py_compile 각 5파일 PASS다.
신규 모듈 미구현 실패와 테스트용 잘못된 Unicode 생성으로 인한 초기 수집 오류를
구분해 보존했다. 변경 그래프는 신규 파일의 함수 0개를 반환해 실제 소스·DB 시험으로 검토했다.

## 5. P2.3 잔여 실행 단위

- **다음:** source의 origin/consumer 등록, 최초 전송 bytes/hash 고정, 관제 ACK 상태 영속화.
  PUBACK 이후에도 미확인 작업을 조회/동일 ID 재전송하며 기존 broker 이력은 유지한다.
- 이후: source/consumer snapshot 조회·누락 대조, 단계별 지연/적체 계측, 선택형 서비스
  시작/종료·재시작 복구와 통신 어댑터 연결. 전체 P2.3 완료 조건으로 판정한다.
- D04 지연 수치·D06 실제 관제/인증/보관 조건은 미정이다. 외부 화면·실장비·운영 배포
  완료로 표시하지 않는다. I10 BLOCKED, I13/I17 OPEN 상태를 유지한다.

기존 source DB v1~v3·전송 상태·운영 파일은 변경하지 않았다. 신규 모듈을 아직 호출하지
않는 현 상태에서는 서비스 동작이 달라지지 않는다. 추후 적용 시 수신 DB·등록 정보·권한과
백업을 보존하고, 보관 정책을 확정하기 전 장부/미확인 데이터를 자동 삭제하지 않는다.
