# EdgeX 연동 구조와 수정 방법

## 1. EdgeX를 왜 사용하는가

EdgeX는 센서·장치를 장치 서비스와 표준 Reading/Command 형태로 묶는 계층이다. 이 프로젝트에서는 CCTV AI 이벤트와 AIoT telemetry를 EdgeX 데이터 흐름에 투영하고, EdgeX Core Command 또는 프로젝트 Action Layer를 통해 장치 명령을 전달한다.

## 2. 실제 구성

| 구성 | 역할 |
|---|---|
| `core-metadata` | 장치·프로파일·리소스 메타데이터 등록 |
| `core-data` | Reading 저장·조회 |
| `core-command` | 장치 명령 라우팅 |
| Redis MessageBus | EdgeX 내부 서비스 이벤트 전달 |
| `cctv-edgex-adapter` | AI/MQTT 이벤트를 EdgeX Reading으로 변환 |
| `aiot-parser` EdgeX forwarder | 센서 Reading 이벤트를 EdgeX 계약으로 생성하고 outbox 재전송 |
| `device-rest`, `device-virtual` | 개발·검증용 EdgeX 장치 서비스 |
| Dabit Device Service | 전광판 Command를 Dabit TCP로 변환 |
| ASC External HTTP | EdgeX 이벤트를 MQTT 외부 연동으로 전달하는 구성 |

## 3. 데이터가 흐르는 순서

### 센서 Reading

1. `aiot-parser`가 외부 MQTT uplink를 수신한다.
2. Base64 payload를 TLV로 해석하고 `SensorReading`을 만든다.
3. `device`, `resource`, `value`, `timestamp`, `origin` 의미로 EdgeX event를 만든다.
4. EdgeX forwarder가 MQTT/EdgeX 입력으로 전달한다.
5. 실패하면 EdgeX Outbox에 보관하고 다음 주기에 재시도한다.

### AI 이벤트

1. AI Engine이 `cctv/ai/events/{camera_id}/{event_type}`를 발행한다.
2. `cctv-edgex-adapter`가 이벤트를 EdgeX 장치·Reading 표현으로 변환한다.
3. Core Data가 Reading을 저장하고, 규칙·조회 계층이 이를 사용한다.

### 전광판 Command

1. Action Layer 또는 EdgeX Core Command가 `display`, `clear`, `power` 명령을 생성한다.
2. Dabit Device Service가 명령을 검증한다.
3. Dabit 프로토콜 `![00<payload>!]`와 EUC-KR 인코딩으로 TCP 전송한다.
4. 응답·실패·이력은 명령 결과와 Action 이력에서 확인한다.

## 4. 장치 등록

```bash
python edgex/register_aiot_devices.py --metadata-url http://localhost:59881
python edgex/register_signboard_device.py --metadata-url http://localhost:59881
```

운영 환경에서는 URL·인증·장치 ID를 환경에 맞게 바꾼다. 등록 전에 `core-metadata` health와 Device Service health를 확인한다. 이미 등록된 장치는 스크립트가 건너뛸 수 있으므로 프로파일 변경 후에는 실제 메타데이터를 조회한다.

## 5. EdgeX 수정 시 체크포인트

- 프로파일의 `deviceResources` 이름과 실제 Command 경로가 같은가
- Event의 `device`, `resource`, `value`가 비어 있지 않은가
- MQTT 브로커와 EdgeX Redis MessageBus를 혼동하지 않았는가
- 변경한 프로파일을 등록 스크립트·Compose·문서에 함께 반영했는가
- 장치가 없는 PC에서도 `device-virtual`로 계약 테스트가 가능한가

EdgeX 상세 계약은 [MQTT·EdgeX 데이터 계약](../integrations/MQTT_EDGEX_DATA_CONTRACT.md), 전광판은 [EdgeX 전광판 프로파일](../integrations/EDGEX_SIGNBOARD_PROFILE.md)을 함께 본다.

