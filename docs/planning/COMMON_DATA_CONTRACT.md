# P1.1 — 관측·이벤트·명령 공통 계약

2026-09-18 · **DONE / 내부 계약 v1.0 확정, 전체 준수 구현은 미완료** · R01,R02,R06

관제 대상은 아직 미정이다(사용자 확인). 내부 데이터의 식별·보존 규칙을 먼저
정리하고 관제 제품별 API/ACK/화면 대응은 P1.4에서 결정한다. 이 문서는 기존 코드의
확인 결과와 후속 구현을 구분한다. [입력·실패·재처리 계약](INGEST_FAILURE_CONTRACT.md)까지
확정했다. 새 DB 스키마나 외부 API를 적용한 문서가 아니다.

2026-09-21 후속: [P2.2 완료 판정](TLV_RECORD_DELIVERY.md)까지 선택 경로의 원본 접수·공통 저장·
기록 전달 복구를 검증했다. 직접 TLV는 `sensor_observation/observation_id`, 영상·외부 이벤트는
기존 event_id로 구분한다. 아래는 P1.1 확정 당시의 조사/계약이며 현재 구현 범위는 후속 보고서를
따른다. 운영 및 모든 기존 경로의 payload가 일괄 전환된 상태는 아니다.

## 1. 현재 구현과 책임

| 경로 | 현재 코드에서 확인한 형태 | 다음 연결 경계 |
|---|---|---|
| 영상 결과 | `normalize_event_queue_item` → `canonicalize_event_payload`; 식별 가능한 DetectionEvent에는 영상 observation 추가 | dict/구역/OpenCV 대응·오류 보관, 장치 실행과 별개로 저장·관제 전달 보장 필요 |
| 외부 MQTT | `normalize_external_event`의 식별/시간 검증 → 정상 변환·기존 발행. 오류는 수신 DB에 격리 | 로컬 검증 완료. 기존 자동 ID 유지; 영속 접수·MQTT 재시도는 미완료 |
| 센서 TLV | `SensorService.process_sensor_data`; 종류별 테이블과 통합 `SensorData`, EdgeX 전달 | 아직 모든 관측이 영상과 동일 envelope/ID 계약을 쓰는 구조는 아님 |
| 장치 명령/결과 | [기존 명령 상관관계 계약](../architecture/COMMAND_CORRELATION_CONTRACT.md) | 논리 명령과 물리 장치 요청·결과를 별도 식별 |

## 2. 재사용할 필드와 보존 규칙

| 필드 | 기존 의미·이번 기준 | 미완료 범위 |
|---|---|---|
| `schema_version` | 기존 canonical은 `1.0`. 외부 MQTT는 미지원 버전을 오류 보관 | 영상 큐·직접 TLV 진입점의 검증 적용 |
| `message_type` | 영상 `ai_detection_event`, 외부 입력 `external_event`; 제공된 종류 보존 | 정상 센서 관측과 경보를 구분할 종류 확정 |
| `message_id` | 입력 생산자의 메시지 식별자 | 센서 패킷 ID/재전송·장치 재부팅 범위 |
| `event_id` | 영상 보강은 제공 ID 유지. 외부 MQTT는 기존 자동 생성 유지, 원본 ID는 raw에 보존; 명시한 빈 ID는 오류 보관 | 공통 관측 ID, 수신 진입점 간 ID 정책·기존 데이터 이행 |
| `occurred_at` | 발생 시각. 현재 보조 함수는 occurred_at→timestamp→received_at→queued_at 순서로 선택 | 실제 발생/수신/보강 시각의 출처 구분·잘못된 시간 검증 |
| `device`, `gateway` | 카메라/센서/게이트웨이 식별·부가정보 보존 | 현장·구역 등록, 미등록 장치 처리 |
| `event` | 종류·중요도·출처·신뢰도 및 생산자가 제공한 확장 필드 보존 | 중요도와 장치 실행 순서는 P1.3에서 확정 |
| `decoded` | 해석된 측정값 보존. 모듈 보강 과정에서 빈 값으로 덮어쓰지 않음 | 공통 측정항목·단위·품질·관측 ID |
| `raw`, `media` | 원본 근거·영상 참조. 기존 raw의 동일 키를 보강용 기본값보다 우선 | 원본 보관·접근·용량 정책, 무제한 저장을 의미하지 않음 |

기존 MQTT/Kuiper 호환용 `camera_id`, `type`, `timestamp`, `metadata` 등을 바로
삭제하지 않는다. 센서에서 `camera_id`가 장치 ID를 담는 기존 호환 형태를 실제
카메라라고 해석하지 않는다. `device`의 식별자를 우선 읽는 기존 accessor를 재사용한다.
새 필드가 필요한 경우 현재 소비자와 데이터 이행 영향을 확인한 뒤 작은 변경으로 적용한다.

첫 수정은 **schema_version 또는 event가 없는 부분 형식의 정보 보존**이었다.
명령·측정값의 유효성 검사기가 아니며, 실제 센서 경로 전체를 공통화한 것도 아니다.
동일 중첩 객체의 충돌은 생산자 값을 우선한다. `raw.metadata` 등 그 안쪽까지
재귀적으로 합치지 않는다. 입력 객체를 수정하지 않으며 재보강에도 결과를 유지한다.

## 3. 시간과 ID

- 현재 숫자 시각은 초/ms/μs 크기 구분 후 KST ISO 문자열로 바뀐다. 문자열 시각은
  검증·시간대 통일 없이 보존되는 경우가 있다. 이를 시간 검증 완료로 표현하지 않는다.
- 목표 계약은 시간대가 있는 발생 시각과 수신 시각을 구분하고 출처를 남기는 것이다.
  발생 시각이 없을 때 수신 시각을 사용했다는 사실을 감추지 않는다. 필드·실패 처리와
  [관측 계약](OBSERVATION_CONTRACT.md)에 센서 측정/수신과 영상 처리/촬영 시각을
  구분했다. 기존 부모 필드의 호환/오류 처리와 데이터 이행은 남아 있다.
- 현재 자동 event ID는 장치·종류·시간과 message_id 또는 payload 기반 해시다.
  원본에 시각/ID가 없으면 재수신 시 보강 시각이 달라질 수 있으므로 중복 제거를
  보장하지 않는다. 같은 순간의 다른 시간 문자열·입력 형태도 같은 ID를 보장하지 않는다.
- 이번 보존 수정으로 **ID가 없는 부분 payload**의 해시 입력에는 이전에 유실되던
  값이 포함될 수 있다. 이미 부여된 ID와 기존 DB 기록을 재생성/이행하지 않는다.
  과거 ID 없는 원본의 재처리·중복 대응은 저장 설계(P1.2)에서 함께 검토한다.
- 명령 ID는 기존 `event_id → command_id(event_id:device) → request_id(command_id:device_id)`
  관계를 재사용한다. 논리 장치 종류와 실제 장치 ID를 혼동하지 않는다. 재시도/응답
  유실·부분 성공 규칙은 P1.3의 별도 완료 조건이며 정확히 1회 출력 보장은 하지 않는다.

## 4. 센서 단위 대응 조사

아래는 **현재 코드 키가 나타내는 단위**다. 실제 보유 센서의 최신 TLV 버전·배율·
부호·범위·결측 표현·축 방향은 센서 부서의 패킷과 대조해야 한다(D02).

| 현재 TLV 해석 키 | 단위 의미 | 현재 저장 경로 |
|---|---|---|
| `temperature_c` | °C | T34954/T34957의 `temperature` |
| `humidity_pct` | % | T34954의 `humidity` |
| `angle_x_deg`, `angle_y_deg` | degree, 축 기준 미확정 | T34955/T34957의 `angle_x`, `angle_y` |
| `reporting_period_s` | second | 해당 종류의 `reporting_period` |

공통 조회에서 단위를 드러내되 기존 테이블을 일괄 교체하지 않는다. 정상 관측값과
임계값 경보는 구분하고 모두 보존 경로를 가진다. 값 없음/해석 실패를 정상값 0으로
합치지 않는 관측 변환은 외부 MQTT 원본에 구현·검증했다. 기존 직접 TLV 저장 경로와
모든 조회 경로의 품질 보존·이행은 남아 있다.

## 5. 앞선 정보 보존 검증

- 보존 결함: 수정 전 3개 재현 실패, 기존 4개 통과. 수정 후 발행 경로까지 포함한
  6개 테스트 파일 **32 PASS**, Ruff PASS. [실패 증거](evidence/canonical_preservation_red_20260918.xml),
  [수정 후 증거](evidence/canonical_preservation_green_20260918.xml).
- 회귀에는 부분 형식의 장치/게이트웨이/측정/원본/확장 필드, 기존 ID, 입력 불변,
  재보강 결과 유지, 영상 발행 시 모델/스켈레톤 근거 보존을 포함했다. 발행 콜백 모형
  시험이며 실제 MQTT·DB·장치·관제의 종단 시험은 아니다.
- 별도 입력 점검에서 잘못된 시간 문자열, 장치 누락(`unknown`), 미지원 버전,
  빈 event_id가 거절되지 않는 것을 확인해 I12로 등록했다. 아래 후속 작업에서
  외부 MQTT 경계를 검증했다. 기존 영상 보강 함수 자체는 여전히 검증기가 아니다.

## 6. 외부 MQTT 검증·오류 보관 — 로컬 구현/검증 완료

09-18 후속 작업. 공통 `src/event_validation.py`와 외부 입력 어댑터에 적용했다.
현재 어댑터의 최상위 장치 ID 별칭/LoRa 형태가 대상이다. 모든 canonical·직접 TLV
입력 형태를 지원하는 통합 어댑터라고 해석하지 않는다.

| 입력 상황 | 처리 규칙 |
|---|---|
| 장치 ID | camera_id→cameraId→source_id→device_id→deviceId→dev_eui→camera 중 **처음 존재하는 키** 사용. 비어 있음/null/unknown/bool/객체 등은 오류. 기존 문자열·정수 ID 지원 |
| 발생 시각 | occurred_at→timestamp→time→ts 중 처음 존재하는 키, 없으면 첫 LoRa rx_metadata의 gw_recv_time→time→timestamp 사용 |
| 시각 형식 | 유한한 숫자·숫자 문자열(기존 초/ms/μs 변환), 시간대가 있는 ISO 문자열 허용. epoch 0도 정상 시각 |
| 시각 오류 | 명시한 null/빈 값·잘못된 날짜·시간대 없음·NaN/Infinity·표현 범위 초과·bool/객체는 오류. 다른 시각/현재 시각으로 덮어 정상화하지 않음 |
| 시각 자체가 없음 | 기존 입력 호환을 위해 서비스 수신 시각 사용. `received_at`와 `occurred_at_source=received_at`를 기록 |
| 원본 시각 있음 | `occurred_at_source`에 선택 키 또는 `rx_metadata[0]`를 기록. gateway 시각과 실제 센서 발생 시각은 구분 필요 |
| event_id 명시 | 비어 있지 않은 문자열인지 검사. 정상 입력의 **기존 자동 ID 생성은 유지**하며 원본 ID는 `raw.raw_payload.event_id`에 보존 |
| event_id 미제공 | 기존 생성 규칙 유지. 시각/생산자 ID가 모두 없으면 재전송 동일 ID를 보장하지 않음 |
| schema_version | 미제공은 기존 1.0 보강. 명시한 값이 `1.0`이 아니면 오류 |
| JSON/측정 구조 | 최상위 객체가 아니거나 decoded/telemetry가 객체 또는 null이 아니면 오류. JSON/UTF-8 해석 실패도 오류 보관 |

검증 결과는 실패 필드와 코드로 표현한다. 예를 들어
`{"field":"occurred_at","code":"timezone_required"}`다. 저장 형식:

```json
{
  "event_id": "rejected_<수신별 UUID>",
  "message_type": "invalid_external_input",
  "received_at": "2026-09-18T11:00:00+09:00",
  "validation": {
    "status": "invalid",
    "errors": [{"field": "occurred_at", "code": "invalid_timestamp"}]
  }
}
```

기존 `ingest_events`의 `normalized_payload`에 위 기록, `raw_payload`에 입력 원본,
`republished=0`을 저장한다. JSON으로 읽지 못한 원본 바이트는 base64로 보관한다.
DB 스키마는 그대로다. 격리 기록 ID는 명령용 이벤트 ID가 아니며 입력의 잘못된/중복
ID 때문에 서로 다른 오류 원본이 사라지지 않도록 수신마다 새 ID를 부여한다.
격리 데이터는 정상 이벤트 발행기에 전달하지 않는다. 이후 조회/재처리에서는
`message_type=invalid_external_input`을 일반 이벤트와 구분해야 한다.

저장 성공 후에만 `quarantined_count`를 증가시킨다. DB 예외는 핸들러에서 전파하며
정상 성공/재발행으로 처리하지 않는다. 다만 상위 `MqttTopicSubscriber._on_message`는
예외를 로그로 처리하므로 **브로커 재전송·DB 장애 중 무손실은 아직 보장하지 않는다**(I13).
정상 입력도 기존 발행→DB 순서를 유지한다. 이 접수/전달 경계와 오류 보관 용량·
보관 기간·재처리 절차는 P1.2/P2에서 다룬다.

외부 입력에서 예전에는 허용되던 시간대 없는 문자열, null/빈 시간, 장치 미상 값은
적용 후 정상 재발행되지 않는다. 생산자 패킷을 대조하고 격리 기록을 확인한 뒤
운영 적용한다. 기존 DB 기록을 재작성하거나 기존 자동 ID를 변경하지 않는다.

검증: 초기 재현 31 FAIL / 6 PASS. 후속 LoRa 사례 포함 외부 입력 41개와 인접
경로 27개, **총 68 PASS**. 임시 SQLite·모형 발행기에서 원본 보존, 재발행 차단,
저장 실패, 오류 다음 정상 입력 처리, 기존 정상 입력 ID·시각 호환을 확인했다.
[초기 실패](evidence/external_validation_red_20260918.xml),
[최종 회귀](evidence/external_validation_green_20260918.xml).
Ruff(변경 Python 3개)·Black(신규 모듈/시험 파일) PASS. 운영 재시작·실제 MQTT
브로커/장치/관제 시험은 하지 않았다. API 경로·DB 스키마·배포 설정 변경은 없으며
외부 이벤트 JSON 시각 출처 필드와 내부 통계 키가 추가됐다.

## 7. 관측 규칙 후속 구현과 계약 완료 판정

09-18 후속 작업에서 [관측 계약](OBSERVATION_CONTRACT.md)의 센서 범위를 구현했다.
외부 입력의 t34954/t34955/t34957에 observation을 추가하고 ID·단위·실제 0/결측/오류,
측정/수신 시각 구분을 확인했다(당시 101 PASS). 이후 DetectionEvent 발행에도
영상 observation을 연결했다. 기존 ID 생성 뒤 추가해 ID/토픽을 유지하고 트랙 0,
동시 인물, false/unknown/결측/오류와 모델 참조를 검증했다. 최종 인접 회귀 포함
**147 PASS**이며 [영상 변환·적용 범위와 증거](OBSERVATION_CONTRACT.md#4-영상-관측--detectionevent-경로-로컬-구현)에 기록했다.

영상 timestamp는 현재 처리 시각이며 촬영 시각은 null로 남긴다. 생산자가 이미
False로 대체한 결측은 복원하지 못하므로 I16으로 추적한다. 직접 TLV·dict 영상
진입점·실제 운영 반영·단위의 실물 규격 검증과 기존 DB 이행은 별도다.


[진입점 대응·접수/실패·재처리 계약과 완료 판정](INGEST_FAILURE_CONTRACT.md)을 추가했다.
현재 코드에 대한 합성 재현 9개에서 정상 대조 2개 충족, 실패 보관 기준 7개 미충족을
확인하고 I01/I12/I13/I17의 후속 구현과 연결했다. **P1.1 계약 산출물은 DONE**이며,
이것이 오류 보관·무손실 구현의 완료는 아니다. 앞선 정상 회귀와 이번 결함 재현을
서로 상쇄하지 않는다. 후속 [P1.2 저장 책임·조회·이행 설계](STORAGE_DESIGN.md)도 완료했으며,
후속 [P2.1 센서 저장 실패 복구](SENSOR_RECOVERY.md)도 로컬 검증을 완료했다. 다음은
[P2.2 공통 이벤트·전달 대기 영속화](WBS.md)다.
I12 전체와 실제 센서 규격 D02, 외부 관제 D06은 미해결로 유지한다.
