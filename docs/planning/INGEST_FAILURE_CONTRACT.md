# P1.1 — 입력 대응·실패 보관·재처리 계약

2026-09-18 · **내부 계약 v1.0 확정 / 전체 경로의 준수 구현은 미완료** · R01,R03,R06,R10

입력이 잘못됐거나 DB/전송이 실패했을 때 원본을 어디에 남기고 무엇을 재시도할지
정한다. [공통 필드](COMMON_DATA_CONTRACT.md), [관측 값·시간](OBSERVATION_CONTRACT.md),
[명령 ID](../architecture/COMMAND_CORRELATION_CONTRACT.md)와 함께 P1.1의 계약 산출물이다.
아래 **목표 계약**과 **현재 구현**은 구분한다. 새 DB/ACK/재처리 기능을 배포한 문서가 아니다.

## 1. 진입점별 현재 대응

| 진입점·코드 | 현재 원본/공통 형식 | 현재 보존·실패 동작 | 구현을 이어갈 작업 |
|---|---|---|---|
| DeepStream DetectionEvent / `src/core/video_event_ingest.py` | 필터/디바운싱 후 dataclass 원본→T1→접수 ID 큐→canonical/observation | 선택 모드는 큐 이전 commit, 오류 격리·실패 원본 보존·기록 전용 복구. 기존 모드는 이전 동작 | [영상 접수 검증](VIDEO_INGEST_RECOVERY.md). 전달 worker/종단 ACK·운영 부하는 남음 |
| OpenCV DetectionEvent·구역 dict / `event_dispatcher.py`, `processor.py` | 선택 모드에서 DetectionEvent to_dict 이전 원본 접수; 구역은 생산자가 만든 dict 접수 | 동일 공통 저장·복구 모듈 사용. dict에 출처 계약 없이 observation을 추측하지 않음 | [영상 접수 검증](VIDEO_INGEST_RECOVERY.md). 모델 후보 생성/필터 이전·실물 성능은 별도 |
| 외부 MQTT JSON / `src/services/external_ingest.py` | 장치/시간/버전 검증; sensor observation은 t34954/55/57만 추가 | P2.2 선택 경로는 bytes 선저장·이벤트/전달 대기 commit 후 handler 정상 반환 시 수동 ACK. 실패는 같은 세션 재접속으로 재전송 | [저장](EVENT_PERSISTENCE.md)·[수신 ACK 검증](MQTT_ACK_RECOVERY.md). 전달 worker·운영 적용은 남음 |
| 직접 TLV MQTT / `parser-python/mqtt/manager.py` | 선택 모드에서 원본 topic/bytes를 JSON 해석 전에 접수 | 고정 ID·수동 ACK·오류 격리·같은 접수 ID의 PG 복구 로컬 검증 | [직접 TLV 근거](TLV_MQTT_RECOVERY.md). 기존 경로·운영 적용·센서 재수신 ID 계약은 남음 |
| 직접 TLV 해석 / `parser-python/service/sensor_service.py` | Base64→TLV→종류별 행 + SensorData; 선택 경로는 해석 전 envelope 접수 | P2.1 새 경로는 원본 영속 보존·오류 격리·입력별 PG transaction. 기존 경로는 메모리 배치 | [센서 복구](SENSOR_RECOVERY.md) 로컬 완료; [외부 JSON/ACK](TLV_MQTT_RECOVERY.md)도 로컬 완료; P4.1 프로필/실물 규격은 남음 |
| TLV DB 배치 / `parser-python/database/processor.py` | 기존 경로는 t34954 등과 sensor_data를 별도 배치 INSERT | 실패 배치 메모리 보존은 수정됨. 새 센서 경로는 이 worker 대신 영속 projection worker 사용 | P2.1 로컬 완료; 기존 메모리 경로의 재시작 유실과 운영 적용은 별도 |
| EdgeX 투영·장치 결과 | 경량 Reading; command_id/request_id 결과 | 경량 투영에는 observation 없음. 출력/전달 결과는 수신 원본 보존을 대신하지 않음 | P1.3/P1.4/P2.3/P3.3: 관제/명령의 별도 상태·ID 연결 |

직접 TLV의 dev_eui fallback은 현행 AIOT_ALLOW_UNKNOWN_DEVICES 설정을 따른다.
일률적으로 모든 센서를 unknown으로 차단하는 변경은 하지 않는다. **식별자가 없거나
형식이 잘못된 입력**과 **식별자는 있으나 등록되지 않은 장치**는 구분하며 등록/권한
정책은 P1.3에서 정한다. 단위·프로필의 현장 규격 확인 D02는 계속 대기다.

## 2. 접수 성공의 목표 계약

1. 입력 경계는 해석 전 원본과 출처·수신 시각을 저장할 책임을 가진다. 영상은 분석
   결과와 기존 미디어 참조를 대상에 포함하며 모든 프레임을 DB에 넣는 뜻은 아니다.
2. **접수 완료는 원본/참조를 영속 저장하고 복구할 수 있음을 확인한 이후**다.
   함수 return, 큐 적재, MQTT publish 성공, EdgeX ACK, 장치 출력은 접수 완료와 다르다.
3. 정규화 결과와 전달 대기는 가능하면 같은 저장소의 트랜잭션으로 확정한다. 서로
   다른 DB에 쓰는 단계는 개별 상태·재시도로 연결한다. 두 DB commit을 원자적이라고
   표현하지 않는다. 구체 테이블·쓰기 주체·연결 방법은 P1.2에서 결정한다.
4. DB 쓰기 실패/디스크 부족 시 성공으로 응답하거나 처리 중 원본을 삭제하지 않는다.
   영속 접수 전 실패는 호출자에 드러내고, 이후 실패는 보존된 기록으로 재시도한다.
   용량/장애시간 상한은 D04/D06에서 정하며 무제한 메모리·무손실을 약속하지 않는다.
5. MQTT에서는 브로커 ACK와 앱 접수 결과를 별도로 검증한다. 외부 MQTT의 새 선택 경로는
   고정 ID·QoS 1·지속 세션·수동 ACK와 실제 재전송/프로세스 종료 시험을 완료했다.
   직접 TLV 선택 경로도 같은 수신 경계를 검증했다. 기존 비영속 경로는 같은 보장을 갖지 않는다.
   QoS 0·브로커 자체 장애·용량 한계의 보장은 별도다.

### ID·원본·버전

| 논리 필드 | 확정 규칙 | 현재 적용 |
|---|---|---|
| ingest_id | 수신 기록의 ID, `ing_` + UUID4 hex. 영속 기록 재처리에는 같은 ID 사용. 별도 네트워크 수신은 별도 기록이며 업무 중복 여부와 구분 | P2.1 센서·P2.2 외부 MQTT/영상 선택 경로 구현. 기존 데이터 이행·운영 적용은 남음 |
| source_namespace | 입력 경로 구분. 현재 관측은 video_detection / external_mqtt; 직접 TLV 접수는 tlv_mqtt로 구분 | tlv_mqtt / external_mqtt / video_detection 공통 접수 선택 경로 구현 |
| source_record_id/event_id | 기존 생산자·정규화 ID를 보존하고 접수와 연결. 원본 ID와 앱 ID가 다르면 둘 다 추적 | 외부 원본 ID는 raw, 현재 observation은 기존 앱 event_id 사용 |
| observation_id | 기존 `[namespace, device_id, profile, source_record_id]` 해시 유지 | 센서·DetectionEvent 일부 구현 |
| command_id/request_id | 기존 논리/물리 장치 관계 유지. 오류 접수 ID를 자동 명령 ID로 사용하지 않음 | 기존 계약 재사용; 재시도/실행 불명 정책은 P1.3 |
| received_at | 입력 경계에서 기록, 재처리 때 원래 값 유지. 처리 시각은 별도 기록 | 일부 구현, 영속 접수 전체 적용 미완료 |
| raw | JSON이면 원본 객체/스칼라, JSON/UTF-8 해석 실패면 원본 bytes의 base64. TLV 바깥 MQTT 원본도 포함 | 직접 TLV/외부 MQTT bytes와 영상 JSON/오류 표현의 큐 전 접수 구현. 기존 비영속 경로는 별도 |
| version | canonical/관측의 내부 1.0과 TLV 프로토콜 버전을 혼동하지 않음. 재처리에는 사용한 해석기/정규화 버전과 회차를 기록 | sensor-v1 고정 저장·external-v1/video-v1 첫 정규화 회차 구현. 변환기 교체 후 추가 회차는 남음 |

ingest_id는 장치가 준 ID나 동일 사건 ID가 아니다. 기존 ID 충돌/다른 본문은 원본을
덮어쓰거나 조용히 무시하지 말고 충돌로 추적한다. byte hash만으로 실제로 반복된
측정값을 중복으로 버리지 않는다. 기존 INSERT OR IGNORE 이행 영향은 P1.2에서 다룬다.

영상 객체의 NaN/Infinity처럼 표준 JSON에 들어갈 수 없는 값은 오류 원본의 별도
표현으로 남긴다. 예를 들어 timestamp의 float/nan이라는 타입·필드·토큰을 함께
기록하며 정상값 0이나 평범한 결측 null로 덮지 않는다. 원본 bytes가 있는 수신은
그 bytes의 base64를 우선 보존한다. 새 영상 선택 경로에서 오류 표현 보관/격리를 검증했다. 비표준 객체 repr은 진단용이며 완전 복원을 보장하지 않는다.

## 3. 실패 분류와 재처리

아래는 어댑터가 공통 저장/판단 모듈에 넘겨야 할 의미다. 모든 진입점이 현재 같은
상태 JSON을 생성한다는 뜻은 아니다. 구체 저장 열/조회 API는 P1.2/P1.4의 범위다.

| 상황 | 목표 처리 | 자동 재시도·출력 |
|---|---|---|
| 잘못된 JSON/Base64/TLV, 필수 식별/버전/시각 오류 | 원본을 격리하고 stage·field·code 기록. 오류 격리도 저장 성공 여부를 따로 확인 | 같은 잘못된 입력의 무한 재시도 금지; 정상 경보 경로로 보내지 않음 |
| 모르는 TLV 버전/프로필 | 원본과 unsupported 이유 보존, 규격 확인 대상으로 구분 | 단위·정상값을 추정하지 않음; 어댑터 추가 후 명시적 재처리 |
| 선택 속성·측정의 결측/unknown/invalid | 전체 원본 보존, 필드별 품질 표시. 값 0/false로 덮지 않음 | 품질 valid도 현장 정확도 합격이 아님; 경보 판단은 P1.3/P4/P5의 근거 품질 규칙 필요 |
| 원본 저장 실패 | 접수 미완료를 호출자/운영 상태에 표시 | 원본 책임을 넘기지 않음; 재시도/역압·용량 정책은 P2 |
| 정상화/전달 중 일시 장애 | 영속 원본·실패 단계·시도 횟수·마지막 오류·다음 시도 시각 보존 | 저장 기록 기반으로 재시도. 저장/관제 경로와 장치 실행 대기를 분리 |
| 장치 ACK 유실/실행 불명 | 해당 request_id의 불명 상태 보존 | 원본 재처리나 ACK 유실만으로 실제 출력 재실행 금지; 장치 능력·중복 정책 확인 |

**재처리 기본값은 기록/분석만 수행**이다. 원본·ingest_id·기존 업무 ID·원래 시각을
유지하고 새 처리 회차/버전을 연결한다. 수정된 정규화 결과가 이전 결과를 지우지
않게 한다. 새 경보·명령이 필요하면 재처리 기록과 별도의 실행 판단을 연결한다.
오래된 이벤트가 DB 복구 직후 스피커에서 재생되지 않도록 유효시간·우선순위·현재
장치 상태를 다시 판단해야 한다(P1.3/D05). 현행 DB 행을 읽어 즉시 라이브 토픽에
발행하는 도구는 이 계약의 재처리 구현으로 인정하지 않는다.

## 4. 현재 실패·정상 대조 재현

[재현 도구](evidence/ingest_boundary_probe_20260918.py),
[JSON 결과·주요 소스 해시 6개](evidence/ingest_boundary_probe_20260918.json).
9개 사례의 **정상 대조 2개 충족 / 실패 보관 기준 7개 미충족**, 종료 코드 2다.
이는 결함 재현 결과이며 전체 시스템 테스트 7개가 새로 회귀했다는 뜻은 아니다.

| 사례 | 관측 결과 | 판단·이슈 |
|---|---|---|
| B01 영상 NaN timestamp | 발행 0회, 실패 통계 1, 원래 큐 잔여 0 | 정규화 오류 원본 보관 없음 — I12 |
| B02 영상 발행 예외 | 발행 1회, 실패 통계 1, 큐 잔여 0 | 재시도 원본 보관 없음 — I17 |
| B03 잘못된 Base64 | 데이터 큐/전달 0, 호출자 예외 없음 | 오류 원본 격리 없음 — I12 |
| B04 길이가 부족한 TLV | 데이터 큐/전달 0, 호출자 예외 없음 | 오류 원본 격리 없음 — I12 |
| B05 기존 정상 온습도 예제 | T34954 + SensorData 큐 2건, 모형 전달 1회 | 정상 파싱 대조 충족. DB commit 증거는 아님 |
| B06 센서 DB 장애→복구 | INSERT 2회 실패, 큐 잔여 0, 복구 후 재시도 0건 | 실패 배치 유실 — I01 |
| B07 외부 입력 DB 장애 | publish→save, 저장 0건, 상위 콜백 예외 없음 | 발행/저장 불일치·접수 복구 미완료 — I13 |
| B08 외부 입력 잘못된 시각 | save만 1회, SQLite 원본 1건, republished=0 | 구현된 오류 격리 정상 대조 충족 |
| B09 외부 입력 발행 예외 | publish만 호출, 저장 0건, 상위 콜백 예외 없음 | 발행 예외로 저장도 건너뜀 — I13 |

실행은 CPU·임시 SQLite·모형 발행/DB 예외를 사용한다. SensorService/DataProcessor는
생성자 대신 시험용 필드를 주입해 DB 연결·worker를 시작하지 않는다. MQTT 브로커,
GPU, 실장비, 운영 DB/서비스에 접속하지 않는다. 실패 후 큐/저장 상태를 관찰하는
진단이며 프로세스 재시작·전원 손실·ACK·전체 파이프라인 무손실 시험을 대신하지 않는다.
정상 TLV는 기존 tests/test_tlv_parser.py의 예제이며 실제 보유 장비의 최신 패킷은 아니다.

```bash
rtk proxy .venv/bin/python docs/planning/evidence/ingest_boundary_probe_20260918.py \
  --output /tmp/cctv-ingest-boundary-recheck.json
```

미충족 항목이 남아 있으면 코드 2, 전부 충족하면 0, 도구 자체 실패는 traceback으로
구분한다. 이 도구의 큐 잔여 검사 통과만으로 영속 복구를 완료 처리하지 않는다.
후속 구현이 보관 위치를 바꾸면 실제 저장소를 확인하는 회귀 시험으로 교체/확장한다.

## 5. P1.1 완료 판정과 인계

| P1.1 완료 기준 | 근거 | 판정 |
|---|---|---|
| ID·시간·단위·버전 확정 | 공통/관측/이 문서·기존 명령 ID. 측정/처리/수신, 결측·미지원과 접수/업무 ID 구분 | 계약 산출물 완료 |
| 기존 payload 대응 | 1절의 입력별 어댑터·현행 보존 경계·미적용 범위 | 계약 산출물 완료 |
| 정상/오류 예제로 검증 | 앞선 로컬 회귀 147 PASS, 센서/영상 합성 예제 각 3개, 이번 9개 경계 재현과 목표/현행 차이 | 예제·차이 검토 완료 |

따라서 **P1.1은 계약 작업으로 DONE**이다. 실패 보관·무손실 구현/운영 적용을 완료로
표시하지 않는다. I01/I12/I13/I17은 OPEN이며 P1.2의 저장 책임/이행 설계와 P2의
구현·장애 회귀에서 해결한다. 계약을 만족하지 않는 현재 동작을 정상 사양으로
고정하는 테스트를 추가하지 않았다. P1.3 중요도/재실행 정책, P1.4 실제 관제 계약,
P4 실물 TLV 규격, P5 모델 품질·시간창, 최종 POC 요구는 그대로 남아 있다.
