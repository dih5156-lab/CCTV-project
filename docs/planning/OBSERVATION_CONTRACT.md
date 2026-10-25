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
| track_id | 원본 object_id를 `{value, quality, source_field}`로 표현. 10-12 I63 개정: 정수 또는 정규 10진 문자열 입력, 유효 value는 문자열. 0 포함 유효, null/누락 missing, 음수 정수/uint64 최댓값 unknown, bool/소수/잘못된 문자열/uint64 초과 invalid. [추적 ID 규격](../features/EVENT_SCHEMA_STANDARD.md#추적-id-계약--2026-10-12--i63) |
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

신규3종의 시간창/인물·평가 계약은 아래 P5.6 절을 따른다. 이 계약과 모형 전달 검증은
싸움·월담 모델 구현이나 운영 경로 활성화를 뜻하지 않는다.

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


<a id="behavior-contract-20261015"></a>

## 6. P5.6 — 싸움·월담·쓰러짐 사건·평가·JSON 계약

10-15 제공 [사양서와 내부 기준](POC_SCOPE.md#10-15-ai-모델링-사양서-반영)을 적용한다.
아래는 **내부 개발 계약1.0**이다. KISA의 공식 정답/시간 허용창·월담 항목 대응·SA 형식과
발주처의 성능 확인서 수용은 P7.4, 실제 관제 수신 규격은 P0.2/P6.1에서 확인한다.
인증 일정은 미정이며12월POC와 구분한다. 기존 `schema_version=1.0`·원본 ID·낙상 유형을 유지한다.

### 6.1 사건 정의와 경계

P5.7 자료 대응(10-15): AI Hub171은 일방적 `assault`와 상호 `fight`를 구분하지만, 아래 내부 싸움 정의는 둘 다 검수 대상이다. 원본 클래스는 보존하고 사람쌍/시간창의 실제 행동을 확인한 후 내부 정답에 연결한다. 위협·무태그 구간을 싸움/정상으로 자동 변환하지 않는다. 이 대응은 기존 사건 범위를 명시한 것이며 JSON/API 유형 변경은 없다. [공식 가이드 대조](MODEL_INFERENCE_METRICS.md#fight-group-guide-20261015)

| 사양서 | JSON 유형 | 내부 양성 사건 | 정상·유사 상황과 판정 불가 |
|---|---|---|---|
| 쓰러짐 | 기존 `fall_detected` | 서거나 이동하던 사람이 균형/지지를 잃고 낮은 자세/바닥으로 전환하는 장면과 그 지속 근거 | 자발적 착석·눕기·물건 줍기·무릎 꿇기는 정상 대조. 처음부터 누운 사람은 전환 미관측으로 별도 상태이며 쓰러짐 발생을 소급 확정하지 않음 |
| 싸움 | 예약 `fight_detected` | 둘 이상의 사람 사이에서 관측되는 때리기·차기·격한 밀치기 등 공격적 신체 상호작용. 일방적 공격도 피해자의 반격 유무만으로 제외하지 않음 | 인사·포옹·장난·운동·도움 주기·군중 밀집을 구분. 가까이 있거나 손을 드는 한 프레임만으로 확정하지 않음. 가림으로 상대/행동이 불명확하면 판정 불가 |
| 월담 | 예약 `fence_climb_detected` | 설정된 담장/울타리 경계를 몸이 넘어가는 동작과 양쪽 위치 전환이 확인되는 사건 | 문/정상 통로 이용·담장 옆 보행·기대기·반대편 원근 겹침은 별도 대조. 단순 구역 진입 `intrusion`과 구분. 오르려는 시도만 있고 실제 넘음이 미관측이면 월담 완료로 세지 않음 |

싸움의 행동 강도/시작·종료, 월담 시도와 완료의 현장 수용 범위는 평가 전 라벨 지침으로
고정하고 애매한 표본은 사유와 함께 검토한다. 위 정의는 내부 기본값이며 공식 인증 정의를
대신하지 않는다. 사고 유형만으로 의도·범죄·법적 책임을 판정하지 않는다.

**P5.7 내부 검수 지침 `fight-local-boundary-v0.2` (10-15):** 접촉 전환 후보와 공격 사건 정답을
구분한다. 손을 들기·몸에 닿기·상대 몸이 밀리거나 당겨지기를 앞뒤 연속 프레임에서 대조한다.
경계가 흐리면 단일 시각 대신 실제 PTS의 후보 범위를 기록한다. 시작이 가려진 경우 최초 시작은
미확정으로 두고 처음 보이는 접촉 시각을 따로 기록한다. 그런 영상은 시작 지연 평가의 제외 사유와
분모를 공개하며, 정상으로 바꾸거나 사건 F1 대상에서 조용히 제거하지 않는다. 제지·도움·시야 이탈·
영상 끝은 실제 공격 종료를 대신하지 않는다. 원래 당사자 간 분리와 제지자의 접촉을 구분하고,
일어나는 중의 등 접촉은 도움/재공격 여부를 별도 검수한다. 화면 밖 이탈 후 재등장하면 미관측
구간을 파일 끝까지 확장하지 않는다. 실제 종료 미확정은 null로 남기고 종료 지연 지표의 제외 사유와
분모를 공개한다. 사건 F1에서 조용히 삭제하거나 무라벨 구간을 정상으로 자동 승인하지 않는다. 후보의 범위 축소만으로 학습/채점에 승격하지 않고 사건 강도·전체 문맥·
정상 대조·촬영 그룹을 검수한 최종 정답 목록을 별도로 고정한다. 대표2초 구간에서 타격이 안 보인다고 정상으로 바꾸지 않는다. 파일명/XML 장소가 다르면 두 값을 보존하고 미해결로 기록하며, 어느 쪽도 단독 학습/평가 분리 키로 쓰지 않는다. 이는 공식 인증 규칙이 아니다.

기존 `EventType`에는 신규2종이 없고 `from_string()`은 미등록값을 `other`로 바꾼다.
따라서 이번 모형은 기존 **dict 입력 경로**로 검증한다. P5.7/P5.8 실제 생산자 연결 때
Enum·분류 매핑·디바운스·유형별 정책·전광판/음성·eKuiper/관제 필터와 해당 시험을 함께
확인한다. `unsafe_behavior` 또는 `intrusion`으로 조용히 치환하지 않는다. 모형 JSON이
저장됐다는 이유로 신규 유형이 운영에서 탐지/출력된다고 표시하지 않는다.

### 6.2 영상·시간창·사람 식별

- 내부 기본 영상은1920×1080 전체화면/30FPS, 사람/Pose 검출 기준0.5, 실제 추론15회/초 목표다.
  모델 입력 크기·동작 결정 임계값·지속시간은 별도이며3종에 같은 값을 강제하지 않는다.
- 사람 검출 점수, 동작 모델 점수/규칙 점수, F1/특이도를 서로 바꾸어 쓰지 않는다.
  `confidence`는 생산자 버전에 따른 이벤트 점수이며 정답률이나 보정된 확률을 보장하지 않는다.
- 판단 근거는 **같은 카메라·스트림 세션의 영상PTS 초**로 기록한다. 처리 속도·FPS로
  PTS를 역산하거나 처리시간을 촬영시각으로 쓰지 않는다. `timestamp`는 기존 처리시각 epoch초,
  `occurred_at`은 그 정규화값을 유지한다. 실제 촬영시각이 없으면 관측 시각은 missing이다.
- 신규 생산자의 `stream_session_id`는 파이프라인 생성/입력 재연결·PTS 역행으로 새 시간축이
  시작될 때 바꾼다. 해당 시점에 자세 이력·동작 창·디바운스 상태도 초기화한다. 단순 정수
  tracker ID나 재기동마다1로 시작하는 generation만으로 세션 유일성을 보장하지 않는다.
- `0 <= window_start_pts_seconds <= window_end_pts_seconds <= decision_pts_seconds`이며 모두
  유한한 초 단위 숫자다. 첫 값은 **실제로 사용한 입력 창의 시작**이며 정답 사건 시작을 뜻하지 않는다.
  미관측 구간·키포인트 부족·재접속을 반복 프레임/0 좌표로 채워 새 행동 근거로 쓰지 않는다.
  허용 입력 간격/유효 프레임 비율은 모델·정책 버전별로 평가 전에 고정한다.
- 각 참여자의 식별 범위는 `(camera_id, stream_session_id, object_id)`다. 숫자 ID의 크기와
  관계없이 wire에서는 정규10진 문자열/null이다. 미추적은 null이며 다른 사람과 합치지 않는다.
  두 카메라의 같은 번호, 재접속 전후 같은 번호를 같은 사람으로 간주하지 않는다.
- 싸움은 사건1건 안에 참여자 배열을 둔다. 여러 사람이 보인다고 같은 사건을 인원 수만큼
  알리지 않는다. 단일 사람의 기존 top-level `object_id`는 유지하고 다인 사건에서는 null로 둔다.
  사건 bbox는 참여자 범위, 참여자 bbox는 각각 전체화면 기준의 좌상단 x/y·width/height 픽셀이다.
  좌표는 영상 범위 내 양의 크기여야 하며 crop 좌표·모델960 좌표와 혼용하지 않는다.
- 원래 사람의 추적이 끊기면 자동으로 새 번호에 시간창을 넘기지 않는다. 추적 재연결의 허용
  조건/근거를 별도 검증하며 세션을 넘는 인물 재식별은 이 계약의 보장이 아니다.

### 6.3 JSON 확장과 전달 책임

[합성3종 예제](../../tests/fixtures/behavior_event_contract.json)를 계약 예제로 사용한다.
`metadata.synthetic=true`로 시험 자료임을 명시한다.
실제 모델 점수·촬영 자료가 아니며 실행 토픽/장치로 전송하지 않는다.

| 위치 | 계약 |
|---|---|
| `event_id` | 사건 확정 시 생산자가 한 번 생성·영속화한 고유 문자열. 재전송은 ID와 원본 전체를 재사용. 서로 다른 사건/카메라/세션에는 새 ID. 참여자 순서·트랙 번호만으로 ID를 만들지 않음 |
| `type` / `event.event_type` | 위3종 이름을 일치시킴. 기존 낙상의 방향은 metadata로 유지 |
| `camera_id`, `timestamp`, `confidence`, `bbox`, `object_id` | 기존 형식/의미 보존. 기존 소비자 필드 삭제/단위 변경 없음 |
| `metadata.behavior.contract_version` | 신규 시간창 계약 `1.0`; 부모 schema 버전과 별도 |
| `metadata.behavior.stream_session_id` | 신규 생산자가 보장할 시간축 세션 문자열. 현재 모든 운영 이벤트에 있다고 가정하지 않음 |
| `window_start_pts_seconds`, `window_end_pts_seconds`, `decision_pts_seconds` | 위 세션의 입력 창/결정 시각. 경보 지연은 정답 onset과 결정PTS를 대조하여 별도 계산 |
| `frame_size`, `participants` | 입력 크기 및 참여자별 object_id/bbox. 배열 순서를 영구 역할/신원으로 해석하지 않음 |
| `model_version`, `decision_policy_version` | 재현 가능한 모델 manifest/설정 버전 참조. 실제 시험 때 엔진·설정 SHA와 연결하며 문자열 존재만으로 실제 적재 증명하지 않음 |
| `person_confidence_threshold` | 내부 검출 기준0.5를 기록. confidence 출력값이나 행동 판정 임계값과 구분 |
| `boundary` — 월담 | 경계 id·설정 version·`outside_to_inside`/`inside_to_outside` 방향. 경계 형상/좌표는 고정된 카메라 설정에서 조회. 경계 미설정이면 월담 양성 확정 보류 |

신규2종도 기존 `canonicalize_event_payload`와 `VideoEventIngest`의 원본 저장→정규화→발행을
재사용한다. 같은 event_id/같은 내용은 중복, 같은 ID/다른 내용은 충돌로 보존하고 덮어쓰지 않는다.
장기 사건의 추가 관측을 보내려면 별도 관측/업데이트 계약이 필요하며 같은 확정 경보 ID의
시각/점수를 매번 바꿔 보내지 않는다. 재기동 전 미발행 경보는 기존 held/대조 절차를 따른다.

부모 JSON의 유효성·유한 시각·카메라 검증과 영속 저장은 기존 코드가 담당한다.
**새 behavior 내부의 PTS 순서·참여자/경계 유효성은 아직 런타임에서 강제되지 않는다.**
P5.7/P5.8 생산자 연결 시 검증을 구현하고 잘못된 입력은 사유와 함께 기록해야 한다.
이 계약 추가만으로 기존 처리기가 의미상 잘못된 메타데이터를 거부한다고 주장하지 않는다.

관제에는 기존 record 전용 envelope를 재사용해 원본 사건 전체를 보낸다. MQTT 발행 콜백
성공은 소비자 저장 ACK나 장치 출력 완료가 아니다. EdgeX 경량 Reading의 일부 필드만으로
참여자/시간창 보존을 가정하지 않으며 실제 외부 수신 대조는 P6.1에서 수행한다. 사양서의
SA **또는** JSON을 모두 필수 구현으로 확대하지 않는다. 기관이 SA를 요구할 경우 최신 형식·
클립명/시간축/유형 매핑을 확보한 뒤 고정 사건 로그에서 변환하고 검증한다(P7.4).

### 6.4 평가 데이터와 지표 계산 단위

| 대상 | 고정할 기준 |
|---|---|
| 분리 | 원본 영상/장면·장소·인물 묶음 단위로 학습/보정/독립 평가 분리. crop·해상도 파생물·다른 클립 구간을 새 독립 원본으로 세지 않음. 평가셋/라벨/설정 hash와 배정 이유 보존 |
| 정답 | `case_id`, 원본/파생 참조·변환 시각 대응, 카메라/세션, 유형, GT 사건 id·주석 인물/공간 범위, onset/end PTS초, 주석 상태·검토자·버전. GT 인물 id를 모델 tracker ID와 동일시하지 않음 |
| 정상 대조 | 이벤트 없는 **사전 고정10초 비중첩 구간**을 내부 정상 평가 단위로 사용. 유형별 normal-window ID·길이·출처 고정. 마지막 불완전 구간/제외 이유·총 정상 노출시간 별도 기록. 공식 정상 단위로 단정하지 않음 |
| 매칭 | 같은 case/카메라/세션/유형이고 주석 인물/공간 근거가 맞으며 정답 `[onset,end]` 안의 최초 경보1건을 TP로 대응. GT1건↔경보1건. 내부 예비 허용창은 추가 여유0초; 공식 시간창 확정 후 별도 프로파일로 비교 |
| 반복·오류 | 동일 event_id/동일 원본 재전달만 먼저 제거. 새 ID의 추가 경보는 unmatched FP로 센다. 조기/늦은/다른 사람·유형 경보는 FP, 대응 없는 GT는 FN. 일찍 울렸다는 이유로 유효 TP로 바꾸지 않음 |
| 입력 실패 | 시작/끝 미관측·디코드/추론 실패·자료 부족은 `NO_RESULT`, 라벨 불명은 `UNLABELED`. 정상(TN)으로 세지 않으며 원래 시험 분모/누락률·사유를 보고한다. 필수 평가에 미해결 사례가 있으면 합격 판정 보류 |
| 정상 판정 | 해당 유형의 정상 창에서 양성 경보가 하나라도 있으면 FP_normal_window1, 없고 전체 관측이 끝났으면 TN_normal_window1. 한 창의 반복 경보 수는 별도 보존 |
| 지연/성능 | 매칭된 사건별 최초 decisionPTS−GT onset, 조기/늦은 건수, 정상 영상시간당 오경보, 실제 추론/처리 FPS·지연 증가·메모리/온도 함께 보고. 영상PTS 지연과 벽시계 종단 지연은 별도 |

- 사건 지표: `Precision=TP_event/(TP_event+FP_event)`, `Recall=TP_event/(TP_event+FN_event)`,
  `F1=2TP_event/(2TP_event+FP_event+FN_event)`.
- 정상 창 지표: `Specificity=TN_normal_window/(TN_normal_window+FP_normal_window)`.
  사건 FP 개수를 정상 창 분모에 섞지 않는다. 한 정상 창의 경보3건은 사건 FP3·정상 FP창1이다.
- 분모0은 `null/산정 불가`로 표시한다.3종별 TP/FP/FN·정상 창의TN/FP·표본수·노출시간을
  각각 보고하며 평균만으로 한 유형의 미달을 숨기지 않는다. 보정에 사용한20개 낙상 성적은 개발 결과다.
- [KISA 신규 안내](https://www.ksecurity.or.kr/user/bbs/kisis/26/95/bbsDataView/36981.do)의 F1/특이도
  각90점은 준비 목표로 참조한다. 위 내부 매칭/정상10초 단위로 계산한 값은 **공식 인증 성적이 아니다**.
  KISA의 단위/허용창·제출 형식이 확정되면 프로파일과 결과를 별도로 보존한다.

판정 예: 정답 사건2건 중1건만 정상 매칭, 조기 경보1건·동일 사건의 새 ID 반복1건·정상 창의
경보2건이면 TP_event1/FN_event1/FP_event4다. 정상10초 창2개 중 경보가 난 창이1개면
TN_normal_window1/FP_normal_window1이다. 사건F1은2/7, 특이도는1/2이며 서로 다른 단위다.
입력 실패한 정상 창1개가 더 있으면 NO_RESULT1로 따로 기록하고 전체 합격을 보류한다.

### 6.5 구현·검증 범위와 다음 연결

이번은 합성3종의 원본/ID/시간/참여자 보존·같은 사건 재전달·충돌·재시작 held·큰 추적ID/미추적·
유효하지 않은 부모 시간/카메라 격리를 **실제 로컬 SQLite와 기존 관제 직렬화/수신 함수**로 검증한다.
네트워크 콜백은 모형이며 실제 MQTT/관제 화면/실물 출력/모델 추론은 수행하지 않는다.
실제 성적 산출기·주석 대응/세션 생성·behavior 의미 검증은 신규 생산자와 함께 P5.7/P5.8에서
연결한다. 낙상 운영 생산자에 신규 필드를 소급 보장하지 않으며 P5.1/P5.2 검증에 연결한다.
P5.6은 내부 계약/모형 검증과 외부·실제 생산자 검증을 구분해 상태를 보고한다.
