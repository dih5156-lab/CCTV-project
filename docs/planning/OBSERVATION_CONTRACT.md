# P1.1 — 관측 ID·시간·단위·결측 계약

2026-09-18 · 내부 형식 v1.0 · **센서 및 DetectionEvent 관측 변환·발행 로컬 검증 완료**

온습도·기울기의 원본 해석값을 공통 관측 레코드로 만든다. 구현은
`src/sensor_observation.py`이며 외부 MQTT 정상 이벤트 JSON의 `observation`에 연결했다.
영상은 `src/video_observation.py`와 DetectionEvent 발행 경로에 연결했다. 기존 이벤트
필드·ID 생성·DB 테이블을 유지하고 원본도 그대로 보관한다. 직접 TLV 서비스·기존
센서 테이블·모든 영상 진입점·실행 중인 서비스·관제 제품에 적용한 상태는 아니다.

2026-09-21 후속: [직접 TLV 공통 관측·기록 전달](TLV_RECORD_DELIVERY.md)을 로컬 검증했다.
해당 선택 경로는 namespace=`tlv_mqtt`, source_record_id=`ingest_id`이며 기존 관측 산식을
그대로 쓴다. 고정된 해석 원본의 품질을 보존하고 가짜 event_id를 만들지 않는다.
아래 외부 MQTT·영상 어댑터 설명은 P1.1 당시 범위다. 운영/기존 PG 컬럼의 일괄 전환은 아니다.

## 1. ID와 버전

| 필드 | 내부 규칙 |
|---|---|
| schema_version | 관측 형식 `1.0`. 부모 이벤트의 버전과 별도로 기록 |
| message_type | 센서 `sensor_observation`, 영상 `video_observation` |
| source_namespace | 입력 경로 구분. 센서 `external_mqtt`, DetectionEvent `video_detection` |
| source_record_id | 변환 전 원본 기록의 ID. 입력/저장 경계에서 한 번 정하고 재사용 |
| device_id | 입력 경계에서 식별한 센서/카메라 ID |
| profile | 센서 `t34954`, `t34955`, `t34957`; 영상 `detection` |
| observation_id | `obs_` + SHA256(JSON 배열 `[namespace, device_id, profile, source_record_id]`) |

JSON 배열은 UTF-8, `ensure_ascii=False`, `separators=(",", ":")`로 인코딩한다.
필드 구분자 충돌을 피하고 동일 원본 ID·장치·프로필·입력 경로에 대해 재변환 결과의
ID를 유지한다. 조회 시각·키 순서·측정값으로 새 ID를 만들지 않는다. 같은 원본 ID를
다른 실제 관측에 재사용하지 않는 책임은 입력/영속 접수 경계에 있다.

이번 어댑터의 source_record_id는 기존 외부 이벤트의 event_id다. 여기서 이벤트는
기존 전송 레코드의 이름이며 모든 관측이 위험 경보라는 뜻은 아니다. 원본 시각과
생산자 ID가 없는 외부 메시지는 재수신 시 기존 event_id부터 달라질 수 있다.
따라서 **관측 변환의 안정적인 ID와 브로커 재전송 중복 제거는 별개**다(I13/P1.2/P2).

관측은 명령과 별도다. 향후 경보 이벤트가 근거 관측을 참조하도록 연결하되 이번에
기존 `event_id → command_id → request_id` 규칙을 바꾸거나 명령을 추가하지 않았다.
과거 DB 기록을 재작성하지 않으며 관측 보정/중복 기록의 저장 정책은 P1.2에서 정한다.

## 2. 센서 시각의 의미

| 필드 | 구현 규칙 |
|---|---|
| received_at | 외부 입력 서비스가 받은 시각. 시간대가 있는 ISO 문자열 필수 |
| observed_at | TLV 해석 데이터의 created_at을 **epoch millisecond 정수**로 해석한 측정 시각. 기존 프로젝트와 같은 KST ISO 형태 |
| observed_at_source | `data.created_at` 또는 `decoded.created_at`. 키가 없으면 null |
| observed_at_quality | `valid` 변환 가능한 정수 ms, `missing` 키 없음/null, `invalid` 형식·범위 오류 |

측정 시각이 없거나 잘못됐으면 observed_at은 null이다. 수신 시각으로 대체하지 않는다.
epoch 0도 정상값이다. 문자열/소수/bool을 측정 시각으로 추측 변환하지 않는다.
이는 현재 `SensorService._get_created_at`의 정수 ms 의미를 근거로 한 계약이다.
센서 실제 시계 오차·타임스탬프 발생 위치·최신 TLV 버전은 D02 확인이 필요하다.

부모 이벤트의 occurred_at은 기존 호환 규칙에 따라 수신 시각으로 보강될 수 있다.
**실제 측정 시각은 observation.observed_at와 품질을 함께 사용**한다. 원본 패킷의
received_at/게이트웨이 시각은 raw에 남지만 서비스 수신 시각과 자동 혼합하지 않는다.

## 3. 측정값과 단위

data 키가 있으면 그 값을, 없으면 decoded를 사용한다. 프로필은 최상위 table 또는
tableName, 없으면 해석 데이터의 tableName에서 확인한다. 모르는 프로필에는 단위를
추정해 붙이지 않으며 observation을 추가하지 않는다.

| 프로필 | 원본 필드 → 공통 측정 이름 | 단위 코드 |
|---|---|---|
| t34954 온습도 | temperature_c → temperature, humidity_pct → humidity | degC, % |
| t34955 기울기 | angle_x_deg → angle_x, angle_y_deg → angle_y | deg, deg |
| t34957 온도·기울기 | temperature_c → temperature, angle_x_deg → angle_x, angle_y_deg → angle_y | degC, deg, deg |

각 측정은 `{value, unit, quality, source_field}`를 가진다. 해당 프로필의 기대 필드를
모두 포함하므로 humidity가 없다는 사실도 확인할 수 있다. 다른 원본 필드와 원본
잘못된 값은 상위 이벤트의 raw/기존 raw_payload DB 열에서 추적한다.

| 입력 값 | value | quality |
|---|---|---|
| 유한한 int/float, 실제 0 포함 | 수치 | valid |
| 필드 없음 또는 null | null | missing |
| bool, 문자열(숫자 문자열 포함), 객체/배열, NaN/Infinity, float 변환 범위 초과 | null | invalid |
| data/decoded 자체가 객체가 아닌 경우 | 해당 프로필 측정값 모두 null | invalid |

valid는 **형식상 사용 가능한 수치**라는 뜻이다. 교정된 물리값·센서 정상·위험도 정상
판정이 아니다. 단위 코드는 현재 파서 키의 의미를 따른 것이며 배율·축 방향·범위·
현장 센서 규격 검증은 D02/P4.1에서 한다. 공통 변환은 원본을 변경하지 않고 새 측정을
생성한다. 특히 기존 `_get_float`를 거쳐 결측이 0으로 바뀐 값에서 품질을 복원하지 않는다.

측정 품질과 기존 `status/severity` 경보 분류는 별도다. 이번 코드가 모든 잘못된
측정값의 경보 실행을 차단하는 것은 아니다. 품질에 따른 경보 정책은 P1.3/P4.2에 연결한다.

## 4. 영상 관측 — DetectionEvent 경로 로컬 구현

`normalize_event_queue_item`이 **DetectionEvent 객체**를 받은 경우, 기존 canonical
변환과 event_id 생성 후 observation을 추가한다. 문자열 camera_id가 비어 있거나
unknown이면 추가하지 않고 기존 경로를 유지한다. 장치 미상 입력의 오류 격리는
I12로 남아 있으며, observation이 없다고 정상으로 판정하지 않는다.

DeepStream 검출 큐는 DetectionEvent를 사용한다. OpenCV의 `_InferencePipeline`은
검출을 dict로 바꿔 적재하며 구역/외부 입력도 dict 형태일 수 있다. 이번에는 dict를
영상으로 추측해 변환하지 않는다. 이미 observation이 있는 dict는 그대로 보존한다.
센서·영상 변환을 하나의 복잡한 조건문으로 합치지 않고 각 어댑터를 분리했다.

### 시간·트랙·모델 대응

| 필드 | 의미와 규칙 |
|---|---|
| source_record_id | 새 관측을 붙이기 **전에** 생성한 기존 event_id. 기존 토픽·ID 해시 입력 유지 |
| observed_at/source/quality | 현재 검증된 촬영 시각이 없으므로 null/null/missing. 처리·수신 시각을 촬영 시각으로 사용하지 않음 |
| processed_at | 원본 timestamp의 유한한 epoch **seconds** 숫자를 KST ISO로 변환. 크기로 ms/μs를 추측하지 않음 |
| processed_at_source/quality | 키가 있으면 timestamp. null/누락은 missing, bool/문자열/NaN/Infinity/범위 오류는 invalid |
| received_at | 발행 정규화 단계에서 기록한 시각. 카메라 수신·촬영·DB 저장 시각과 다름 |
| track_id | 원본 object_id를 `{value, quality, source_field}`로 표현. 정수 0 포함 유효, null/누락 missing, 음수/uint64 최댓값 unknown, bool/문자열/소수/uint64 초과 invalid |
| frame_num | metadata.frame_num 원본 참조. 없으면 null; 이 값만으로 전역 프레임 ID를 만들지 않음 |
| model | metadata의 backend/model_task/gie_id/model_version/attribute_backend를 backend/task/gie_id/version/attribute_backend로 대응. backend만 최상위 값으로 보완, 다른 누락은 null |

근거: `_deepstream_event_factory`의 object/tensor 변환은 `timestamp_factory=time.time`,
외형 매칭도 `time.time()`을 사용한다. 따라서 timestamp는 처리 시각이다. 현재 원본
시각 0은 관측에서는 보존하지만 부모 canonical의 truthy fallback 문제까지 고치지는
않았다(I12). 잘못된 시각으로 부모 정규화가 먼저 실패하면 관측 변환까지 도달하지
못하므로, 순수 변환기의 invalid 표시는 큐 실패 원본의 영속 보관을 보장하지 않는다.

트랙 번호는 카메라/트래커 세션 범위다. 재시작·재접속 뒤 재사용될 수 있고 합성
트랙도 있으므로 영구 인물 ID로 사용하지 않는다. 현재 세션 ID·캡처/PTS 기준과 모델
버전은 모두 보장되지 않는다. 이 근거 연결은 P5.1에서 보완한다. 같은 프레임/시각의
서로 다른 트랙은 기존 event_id로 구별하며, 관측 ID 안정성은 같은 source_record_id를
재사용할 때의 보장이다. 재전송 중복 제거·트랙 재식별 보장은 아니다.

종류·class_name·confidence·bbox·keypoints 및 속성 점수/추가 메타데이터는 부모
이벤트와 raw에 남긴다. model 참조가 있다고 실제 엔진/버전 로딩을 검증한 것은 아니다.
추가 observation이 MQTT JSON 크기에 미치는 실제 처리량 영향은 아직 측정하지 않았다.
`project_edgex_event`의 기존 경량 Reading에는 observation을 추가하지 않았다.
따라서 MQTT payload의 확장이 EdgeX·외부 관제까지 동일하게 전달된다는 뜻은 아니며,
해당 조회/전달 계약은 P1.4/P2.3에서 확인한다.

### 속성 품질

불리언: has_helmet, has_backpack, has_handbag, has_suitcase.
범주: upper_color, lower_color, helmet_color, gender, age_group, face_name.
각 항목은 `{value, unit: null, quality, source_field}`다. 최상위 attributes에 키가
있으면 null도 우선하고, 없으면 평탄한 metadata의 같은 키를 읽는다. 읽어야 할
컨테이너가 객체가 아니면 invalid다. 원본을 수정하지 않는다.

| 입력 | 공통 값·품질 |
|---|---|
| 실제 bool true/false | 그대로 / valid. 숫자 0/1이나 문자열로부터 변환하지 않음 |
| 비어 있지 않은 범주 문자열 | 앞뒤 공백 제거 / valid. 라벨 지원 여부·모델 정확도 판정은 아님 |
| 문자열 unknown (대소문자·앞뒤 공백 무시) | null / unknown |
| 누락 또는 null | null / missing |
| 잘못된 타입·빈 범주·숫자/문자열 불리언 | null / invalid |

valid는 **생산자가 보낸 형식이 유효함**을 뜻한다. `_appearance_analyzer`는 너무 작은
crop에서 헬멧/가방을 False로 반환하며 `_appearance_pipeline`에도 bool/default 변환이
있다(I16). 이미 이 단계에서 합쳐진 미지원·결측·판정 불가를 관측 어댑터가 복구할 수
없다. **has_helmet=false를 확정 미착용 경보로 바로 해석하지 않는다.** 능력/가시성·
근거 품질과 경보 정책은 P5.3/P5.4/P1.3에서 검증하고 연결한다.

신규 동작 분류의 시간창/인물 계약은 P5.6에서 확장한다. 현재 어댑터가 싸움·폭행
분류 모델이나 시간창을 구현한 것은 아니다.

## 5. 검증과 적용 한계

- 실패 재현: 외부 수신 저장/발행 시험에서 observation 누락 1 FAIL, 비대상 호환 2 PASS.
  [초기 XML](evidence/observation_red_20260918.xml).
- 최종 관측 단위 26개·외부 입력 44개·인접 경로 31개 **101 PASS (3.70초)**.
  [최종 XML](evidence/observation_green_20260918.xml). Ruff 변경 Python 4개,
  Black 신규 모듈/시험 3개 PASS.
- [재현 가능한 JSON 예제 3개](evidence/observation_examples_20260918.json)는 코드로 생성한
  정상/결측/오류 예제다. 실제 센서 패킷·현장 합격 증거가 아니다.
- 임시 SQLite·모형 발행기에서 원본 보존과 observation 저장을 확인했다. 운영 배포,
  실장비/실브로커·GPU·외부 관제 시험은 하지 않았다. 서비스 구독 설정을 바꾸지 않아
  실제 TLV 토픽이 이 경로로 들어오는지도 별도 검증해야 한다.
- I14: 기존 직접 TLV 숫자 helper가 결측/잘못된 값을 0.0으로 바꾸는 경로는 남아 있다.
  원본 해석값에서 분리한 이번 관측 변환으로 기존 테이블의 문제가 해결된 것은 아니다.
- I15: 현재 EdgeXForwarder의 Core Data 프로필 매핑에는 t34954가 없다. 온습도 관측
  변환 지원과 EdgeX Core Data 전송 지원은 구분한다. MQTT/Alert 전달 경로와도 별개다.

영상 후속 검증:

- 초기 발행 시험 **2 FAIL / 8 PASS**로 observation 누락을 재현했다.
  [초기 XML](evidence/video_observation_red_20260918.xml).
- 최종 영상/센서 변환·외부 입력·발행·factory/dispatcher 등 11개 파일
  **147 PASS (6.59초)**. [최종 XML](evidence/video_observation_green_20260918.xml).
  Ruff 변경 4개·Black 신규 모듈/시험 3개 PASS.
- 기존 event_id·토픽·원본 보존, 재정규화 유지, 트랙 0·동시 인물 구분, unknown/결측/
  false/잘못된 값, 시간 출처, 미상 카메라·dict 경로의 적용 제외를 확인했다.
- [영상 JSON 예제 3개](evidence/video_observation_examples_20260918.json)는 순수 변환기
  입력/출력의 합성 사례다. 실제 영상·모델 추론 성적이나 잘못된 영상 입력의 저장
  성공 증거가 아니다. GPU·서비스 재시작·실제 출력·외부 관제는 시험하지 않았다.

이후 [입력별 오류 보관·재처리 계약](INGEST_FAILURE_CONTRACT.md)을 확정해 P1.1은
계약 작업으로 DONE 처리했다. DB 통합·이행·영속 접수/오류 보관 구현은 P1.2/P2,
실제 센서 검증은 P4에서 이어간다. 미적용 경로가 해결된 것은 아니다. 실장비 적용은
[단계별 적용 순서](POC_SCOPE.md#31-모듈별-실장비-적용-시점)를 따른다.
