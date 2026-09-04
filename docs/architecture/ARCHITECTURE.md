# 전체 아키텍처

## 1. 계층별 역할

| 계층 | 구성 | 책임 |
|---|---|---|
| 입력 | CCTV RTSP, LoRa Network Server | 영상·센서 원천 데이터 제공 |
| 분석 | `cctv-ai-engine`, `aiot-parser` | 영상 추론, pose 판정, TLV 디코드 |
| 메시지 | Mosquitto, EdgeX Redis MessageBus | 서비스 간 비동기 전달 |
| 규칙 | eKuiper, Sensor Rule Bridge | 센서·AI 조건을 운영 이벤트로 변환 |
| 저장 | Alert/Action/AIoT SQLite, EdgeX DB | 이벤트·측정값·명령 이력 저장 |
| 제어 | Action Layer, Dabit Device Service | 정책 적용 및 장치 명령 실행 |
| 외부 장치 | InterM 스피커·사이렌, Dabit 전광판 | 실제 경보 출력 |
| 조회 | Public API, 관제 UI, Prometheus/Grafana | 상태 조회·운영·관측성 |

## 2. 요청·이벤트 흐름

일반 조회·수동 제어는 `Public API :9000` 또는 `Alert API :8000`으로 들어간다. API는 Action Layer와 통신해 승인·거부·모드 변경·장치 명령을 수행한다.

자동 경보는 API 요청이 아니라 MQTT 이벤트로 시작한다. AI Engine 또는 센서 규칙이 이벤트를 발행하면 Event/Alert 처리 경로가 저장과 장치 동작을 분리한다. 따라서 장치가 잠시 꺼져도 이벤트 저장과 재시도 상태를 확인할 수 있다.

## 3. 핵심 식별자

| 식별자 | 의미 | 주의 |
|---|---|---|
| `camera_id` | 카메라 논리 ID | 카메라 설정·이벤트·검색에서 동일하게 사용 |
| `device_id` | 센서 또는 출력 장치 ID | 장치 프로파일·Action 대상과 일치해야 함 |
| `message_id` | MQTT 이벤트 식별자 | 재시도 시 유지해 중복 처리 방지 |
| `event_id` | 저장된 이벤트 ID | 승인·거부·장치 명령의 기준 |
| `command_id` | 장치 명령 ID | 결과·재시도·이력 추적에 사용 |

## 4. 실패 격리

- MQTT 발행 실패: Runtime Outbox에 저장 후 재시도
- EdgeX 전달 실패: `parser-python`의 EdgeX Outbox와 재시도 상태 확인
- 장치 제어 실패: Action HTTP/MQTT 결과와 cooldown을 확인
- DB·서비스 장애: health/readiness와 Docker 로그를 함께 확인

서비스 간 계약은 [MQTT·EdgeX 데이터 계약](../integrations/MQTT_EDGEX_DATA_CONTRACT.md)을 기준으로 한다. 기존 호환 필드를 제거하지 말고, 확장 시 `schema_version`을 올린다.

