# EdgeX·eKuiper 상세 인수인계서

최종 확인 기준: 2026-09-03

## 1. 결론부터

이 프로젝트에서 EdgeX와 eKuiper는 같은 역할을 하지 않는다.

- **EdgeX**: 장치·리소스·Reading·Command를 표준화하고 서비스 사이를 연결하는 장치 플랫폼
- **eKuiper**: MQTT/EdgeX에서 들어온 데이터를 SQL 스트림과 룰로 필터·집계·라우팅하는 규칙 엔진
- **Mosquitto MQTT**: CCTV·센서·Action Layer가 사용하는 프로젝트 메시지 브로커
- **EdgeX Redis MessageBus**: EdgeX 내부 서비스 통신용 버스

특히 Mosquitto MQTT와 EdgeX Redis MessageBus는 서로 다른 계층이다. MQTT topic을 EdgeX 내부 MessageBus topic으로 직접 간주하면 장애 원인을 잘못 찾게 된다.

## 2. 전체 구조

```text
외부 CCTV / LoRa 센서
  ├─ RTSP → cctv-ai-engine → cctv/ai/events/...
  └─ MQTT uplink → aiot-parser → aiot/sensors/...

Mosquitto MQTT
  ├─ cctv-edgex-adapter → EdgeX Device Service/Core Data
  ├─ eKuiper → cctv/rules/intrusion/* 또는 aiot/rules/sensor/*
  └─ Action Layer → 스피커·전광판·사이렌

EdgeX
  ├─ Core Metadata: 장치·프로파일·리소스
  ├─ Core Data: Reading 저장
  ├─ Core Command: Command 라우팅
  ├─ Redis MessageBus: 내부 이벤트
  └─ Device Service: 실제 장치 프로토콜 변환
```

## 3. EdgeX 구성 요소별 역할

| 구성 | 기본 컨테이너/포트 | 역할 |
|---|---|---|
| Consul | `edgex-core-consul` | 서비스 등록·공통 설정 저장 |
| Redis | `edgex-redis` | EdgeX DB와 MessageBus |
| Core Metadata | `edgex-core-metadata:59881` | 장치·profile·resource 등록 |
| Core Data | `edgex-core-data:59880` | Event/Reading 저장·조회 |
| Core Command | `edgex-core-command:59882` | 장치 Command 실행 경로 |
| Device REST | `edgex-device-rest:59986` 계열 | 검증용 REST 장치 서비스 |
| Device Virtual | `edgex-device-virtual` | 실제 장치 없이 계약 검증 |
| eKuiper | `edgex-kuiper:59720` | 스트림·룰 실행 |
| CCTV EdgeX Adapter | `cctv-edgex-adapter` | AI/센서 이벤트와 EdgeX 연결 |
| Dabit Device Service | `cctv-device-dabit:59990` | EdgeX Command를 Dabit TCP로 변환 |

포트는 Compose profile과 실행 파일에 따라 달라질 수 있으므로 최종값은 `docker compose config`와 health endpoint로 확인한다.

## 4. EdgeX의 핵심 개념

### Device

센서, 카메라, 전광판처럼 EdgeX가 관리하는 논리 장치다. `deviceName`, `serviceName`, `profileName`이 연결되어야 한다.

### Device Profile

장치가 제공하는 리소스와 자료형을 정의한다. 예를 들어 전광판 profile은 `display`, `clear`, `power` Command와 입력 파라미터를 선언한다.

### Resource

장치의 하나의 측정값 또는 제어 기능이다. 센서 temperature, camera fall_detection, signboard display가 resource가 될 수 있다.

### Event / Reading

Device Service가 만든 측정 이벤트와 실제 값이다. 이 프로젝트에서는 AI/센서 이벤트를 EdgeX가 이해할 수 있는 `device`, `resource`, `value`, `timestamp`, `origin` 의미로 투영한다.

### Command

장치에 값을 쓰거나 동작을 실행하는 요청이다. Core Command가 profile과 device service를 통해 실제 HTTP/TCP 장치 명령으로 전달한다.

## 5. CCTV AI 이벤트의 EdgeX 연결

1. AI Engine이 `cctv/ai/events/{camera_id}/{event_type}`에 JSON을 발행한다.
2. `cctv-edgex-adapter`가 MQTT topic을 구독한다.
3. canonical event의 event type을 EdgeX resource로 매핑한다.
4. 예: `fall_detected` → `fall_detection`, `helmet` → `helmet_detection`.
5. Core Data에 Reading을 저장하거나 EdgeX MessageBus로 전달한다.
6. eKuiper 또는 다른 소비자가 규칙 처리·조회·외부 전달을 수행한다.

EdgeX 투영 예시:

```json
{
  "event_id": "evt-20260903-0001",
  "schema_version": "1.0",
  "type": "fall_detected",
  "resource": "fall_detection",
  "device": "camera_1",
  "device_type": "cctv",
  "confidence": 0.86,
  "severity": "critical",
  "occurred_at": "2026-09-03T14:00:00+09:00"
}
```

원본 bbox·keypoint·fall reason 전체를 EdgeX Reading의 단일 값으로 무리하게 넣지 않고, 표준 Reading에는 핵심값을 넣고 원본 이벤트/DB에서 상세값을 조회하는 방식이 안전하다.

## 6. AIoT 센서의 EdgeX 연결

```text
LoRa uplink
  → aiot-parser Base64/TLV decode
  → aiot/sensors/{dev_eui}/{table}
  → EdgeX forwarder / cctv-edgex-adapter
  → Core Data Reading
  → eKuiper 또는 Sensor Rule Bridge
```

센서 원본은 `device_id`, `dev_eui`, `table`, `data`, `received_at`을 유지한다. EdgeX로 보낼 때는 장치와 resource를 분명히 매핑한다. 원본 MQTT가 정상이어도 Metadata profile이나 forwarder outbox가 잘못되면 Core Data에 쌓이지 않을 수 있다.

## 7. Dabit 전광판 Device Service

### 등록 관계

```text
cctv-signboard-dabit-profile.yaml
  → Core Metadata 등록
  → cctv-device-dabit service
  → Core Command
  → PUT /api/v3/device/name/{device}/{command}
  → Dabit TCP/EUC-KR
```

등록 예시:

```bash
python edgex/register_signboard_device.py \
  --metadata-url http://127.0.0.1:59881 \
  --service-url http://cctv-device-dabit:59990 \
  --device-name cctv-signboard-01
```

등록 전에 Dabit TCP host/port, Device Service health, profile의 command 이름이 같은지 확인한다. 실제 Action Layer의 기본 운영 경로는 TCP 직접 제어일 수 있으며, EdgeX Device Service 경로와 항상 동일하다고 가정하지 않는다.

## 8. eKuiper 동작 방식

eKuiper 룰은 대체로 다음 세 부분으로 구성된다.

1. **Stream**: MQTT topic과 JSON 필드 구조를 선언
2. **SQL**: 조건·집계·window로 이벤트 선별
3. **Action**: MQTT topic 등으로 결과 발행

배포 도구는 `runners/run_kuiper_rules.py`다. 기존 stream/rule이 있으면 삭제 후 최신 정의를 재생성하고, `{{TOKEN}}` 값을 환경변수로 치환한다.

## 9. CCTV eKuiper 룰

파일: `kuiper/rules/cctv_intrusion_rules.json`

### 입력 stream

```sql
CREATE STREAM ai_events_stream
(camera_id STRING, type STRING, confidence FLOAT, timestamp STRING, object_id BIGINT)
WITH (DATASOURCE="cctv/ai/events/+/+", TYPE="mqtt", FORMAT="json", SHARED="true")
```

### 룰

| 룰 ID | 조건 | 출력 |
|---|---|---|
| `intrusion_confidence_filter` | 지정 이벤트이고 confidence ≥ `INTRUSION_CONFIDENCE` | `cctv/rules/intrusion/filtered` |
| `intrusion_5s_persist` | 5초 tumbling window에서 hit count ≥ `PERSIST_HIT_COUNT` | `cctv/rules/intrusion/persisted` |
| `intrusion_high_confidence_routing` | 위험 이벤트이고 confidence ≥ `CRITICAL_CONFIDENCE` | `cctv/rules/intrusion/critical` |

기본값은 intrusion confidence `0.7`, critical confidence `0.9`, persist hit count `5`다. 이 룰의 `confidence`는 AI 이벤트 예측 점수이고, 해당 룰을 통과했다는 사실이 실제 안전사고 확정이라는 뜻은 아니다.

## 10. AIoT eKuiper 룰

파일: `kuiper/rules/aiot_sensor_rules.json`

| stream | 입력 topic | 주요 필드 |
|---|---|---|
| `aiot_tilt_stream` | `aiot/sensors/+/t34955` | angle_x, angle_y |
| `aiot_tilt_temp_stream` | `aiot/sensors/+/t34957` | temperature, angle, event_code |
| `aiot_imu_stream` | `aiot/sensors/+/t34958` | acc, gyro, event_code |

주요 출력:

| 룰 | 조건 | 결과 topic |
|---|---|---|
| 기울기 | angle 절대값 ≥ `TILT_THRESHOLD` | `aiot/rules/sensor/tilt` |
| 온도 | temperature ≥ `TEMP_HIGH_THRESHOLD` | `aiot/rules/sensor/temperature` |
| 진동/충격 | `event_code=1` | `aiot/rules/sensor/vibration` |

기본 `TILT_THRESHOLD=10.0`, `TEMP_HIGH_THRESHOLD=60.0`이다. 이 파일의 event_code 기반 진동은 센서가 이미 판정한 flag를 라우팅하는 방식이고, Python `SensorRuleBridge`의 가속도 편차 계산과는 다른 경로다.

## 11. 룰 배포·실행

Compose에서 rule loader가 다음과 같이 배포한다.

```bash
python runners/run_kuiper_rules.py \
  --kuiper-api http://127.0.0.1:59720 \
  --rules-file kuiper/rules/cctv_intrusion_rules.json

python runners/run_kuiper_rules.py \
  --kuiper-api http://127.0.0.1:59720 \
  --rules-file kuiper/rules/aiot_sensor_rules.json
```

운영 컨테이너 내부 주소는 보통 `http://edgex-kuiper:59720`이다. 배포 전에 MQTT source default config가 올바른 broker·계정·protocol 3.1.1을 사용하는지 확인한다.

## 12. 룰 수정 절차

1. 입력 topic과 실제 JSON 필드를 먼저 `mosquitto_sub`로 확인한다.
2. stream 선언의 field type과 중첩 구조를 실제 payload와 맞춘다.
3. SQL 조건과 출력 topic을 수정한다.
4. unresolved token(`{{...}}`)이 남지 않았는지 확인한다.
5. 개발용 eKuiper에 배포하고 rule 상태를 조회한다.
6. 정상·임계값 미만·임계값 초과 payload를 각각 발행한다.
7. 출력 topic, Action Layer 장치 동작, DB 이력을 확인한다.
8. 운영 승격 전 기존 rule export와 rollback 파일을 보관한다.

## 13. 장애 진단 순서

### MQTT에는 원본이 있지만 eKuiper 출력이 없음

1. stream topic wildcard가 실제 topic과 일치하는가
2. JSON 필드명·대소문자·중첩 경로가 맞는가
3. threshold token이 올바른 값으로 치환됐는가
4. eKuiper rule 상태와 로그에 SQL 오류가 없는가
5. MQTT source broker/계정 설정이 맞는가

### eKuiper 출력은 있지만 장치가 동작하지 않음

1. Action Layer가 해당 topic을 구독하는가
2. `DEFAULT_ALARM_TOPICS`에 포함되는가
3. confidence threshold·manual mode·cooldown에 걸리지 않았는가
4. 장치 `configured/reachable` 상태가 정상인가
5. 장치별 API와 command 결과가 성공인가

### EdgeX Core Data에 Reading이 없음

1. Metadata에서 device/profile/resource 등록 여부 확인
2. adapter/forwarder 로그와 outbox pending 확인
3. Core Data·Redis·Consul health 확인
4. device name과 resource name이 profile과 일치하는지 확인
5. event timestamp/origin과 value 변환 오류 확인

## 14. 운영 점검 명령

```bash
docker compose --env-file .env.jetson -f docker-compose.jetson.yml ps
docker logs --tail 200 cctv-edgex-adapter
docker logs --tail 200 edgex-kuiper
docker logs --tail 200 cctv-kuiper-rule-loader
mosquitto_sub -h <MQTT 주소> -p 1883 -t 'cctv/ai/events/#' -v
mosquitto_sub -h <MQTT 주소> -p 1883 -t 'cctv/rules/#' -v
mosquitto_sub -h <MQTT 주소> -p 1883 -t 'aiot/sensors/#' -v
mosquitto_sub -h <MQTT 주소> -p 1883 -t 'aiot/rules/sensor/#' -v
```

EdgeX API 확인:

```bash
curl -fsS http://127.0.0.1:59880/api/v3/ping
curl -fsS http://127.0.0.1:59881/api/v3/ping
curl -fsS http://127.0.0.1:59720
```

인증·포트가 현장 설정과 다르면 실제 Compose 환경값을 사용한다.

## 15. 변경 영향 범위

| 변경 | 함께 확인할 대상 |
|---|---|
| stream topic 변경 | publisher, MQTT ACL, rule, Action Layer 구독 |
| SQL threshold 변경 | 정상/경계/위험 fixture, 장치 오탐 |
| output topic 변경 | Action Layer topic set, 문서, 대시보드 |
| Device Profile 변경 | Metadata 등록, Core Command, Device Service |
| Reading resource 변경 | Core Data 조회, EdgeX adapter, eKuiper field |
| MQTT broker 변경 | parser, AI Engine, adapter, eKuiper source, Action Layer |

## 16. 인수인계 체크리스트

- [ ] MQTT와 EdgeX Redis MessageBus의 차이를 설명할 수 있는가
- [ ] Core Metadata에서 device/profile/resource를 조회했는가
- [ ] Core Data에 Reading이 쌓이는 것을 확인했는가
- [ ] eKuiper stream과 rule의 입력·출력 topic을 확인했는가
- [ ] 룰 token과 threshold의 실제 운영값을 기록했는가
- [ ] AI·센서 정상/위험 payload를 각각 테스트했는가
- [ ] Action Layer가 output topic을 구독하는지 확인했는가
- [ ] Dabit Device Service 등록·실행 경로를 확인했는가
- [ ] outbox/retry와 rollback 방법을 알고 있는가
- [ ] 실제 장비 IP·계정·담당자 정보를 별도 보안 방식으로 전달했는가

## 관련 문서

- [프로젝트 구조·파이프라인](../architecture/PROJECT_STRUCTURE_AND_PIPELINES.md)
- [EdgeX 연동 요약](../architecture/EDGEX_INTEGRATION.md)
- [센서 연동](../architecture/SENSOR_INTEGRATION.md)
- [MQTT·EdgeX 데이터 계약](MQTT_EDGEX_DATA_CONTRACT.md)
- [EdgeX 전광판 프로파일](EDGEX_SIGNBOARD_PROFILE.md)
- [현장 EdgeX 체크리스트](../guides/JETSON_EDGEX_FIELD_CHECKLIST.md)
- [운영 EdgeX 파일럿](../operations/edgex-aiot-pilot.md)

