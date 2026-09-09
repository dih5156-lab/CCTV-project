# 정부과제 요구사항 추적표

이 문서는 전임자 자료와 현재 구현을 연결하기 위한 양식이다. 현재 폴더에서 확인된 공식 후보 자료는 `엣지 플랫폼/(릴리즈) 개발용_서비스시나리오_도시공원_AI_Algo_231031.hwp`와 KETI CCTV 이벤트 API Postman 자료다. 최종 제출 전 과제 제안서·협약서·시험계획서 원문과 대조해 요구사항 ID와 문구를 확정해야 한다.

증빙 파일의 상세 등록은 [정부과제 증빙 등록 양식](EVIDENCE_REGISTER_TEMPLATE.md)을 사용한다.

## 상태 정의

| 상태 | 의미 |
|---|---|
| 구현 | 코드 또는 설정이 존재함 |
| 검증 | 자동 테스트·시뮬레이션으로 확인함 |
| 실증 | 실제 현장 장치·영상·네트워크로 확인함 |
| 보류 | 장치·데이터·조건이 없어 아직 판단할 수 없음 |

## 추적표

| 요구 기능 | 전임자 근거 | 현재 구현 근거 | 검증 근거 | 현재 상태 |
|---|---|---|---|---|
| CCTV 영상 입력 | `smartpole`, CCTV API 자료 | `src/core`, `src/edgex/device_service.py` | `tests/test_deepstream_pipeline_builder.py`, 운영 smoke | 구현·검증 |
| AI 이상행동 분석 | 모델 구현 폴더, 학습 결과 | `src/core/_inference_pipeline.py`, `src/core/_deepstream_pipeline_builder.py` | `tests/test_object_detection_pipeline.py`, `tests/test_run_fall_training_pipeline.py` | 진행 |
| 센서 연동 | AIoT 어댑터·KETI 자료 | `parser-python`, `src/aiot/contracts.py`, `src/services/sensor_rule_bridge.py` | `tests/test_aiot_contracts.py`, `parser-python/tests/test_edgex_outbox.py` | 구현·검증 |
| 실시간 이벤트 전달 | KETI API 자료 | `parser-python/mqtt`, `src/edgex/command_contract.py` | `parser-python/tests/test_edgex_forwarder_contract.py`, outbox 테스트 | 구현·검증 |
| 규칙 기반 판단 | 초기 설계에는 제한적 | `kuiper/`, `runners/run_kuiper_rules.py`, `src/services/sensor_rule_bridge.py` | 규칙·센서 브리지 테스트 | 진행 |
| 출력 장치 표준 제어 | 전임자 Device Manager 설계 | `src/services/action_bridge.py`, `src/services/device_command_transport.py` | `tests/test_action_bridge.py`, `tests/test_device_command_transport.py` | 구현·검증 |
| 전광판 제어 | 다빛 프로토콜 자료 | `src/edgex/dabit_device_service.py`, `runners/run_signboard_device_service.py` | `tests/test_dabit_device_service.py`, `tests/test_dabit_mqtt_device_service.py`, 실제 Core Command 로그 | 실증 진행 |
| 스피커 제어 | Inter-M API 자료 | `src/edgex/speaker_device_service.py`, `runners/run_speaker_device_service.py` | `scripts/ops/check_edgex_device_service_contracts.py`, dry-run | 실제 실증 보류 |
| 사이렌·경광등 제어 | Qlight·GPIO 자료 | `src/edgex/siren_device_service.py`, `runners/run_siren_device_service.py` | Device Service 계약·dry-run 점검 | 실제 실증 보류 |
| 다중 장치 운영 | 전임자 자료는 단일 장치 가정 | `config/output_devices.json`, `src/services/device_command_transport.py` | `tests/test_device_command_transport.py`, 장치별 풀 테스트 | 구현·검증 |
| 장치 장애 격리 | 별도 근거 부족 | 장치별 결과·미등록 차단 로직 | `tests/test_edgex_device_service_contracts.py`, 장애 격리 테스트 | 구현·검증 |
| 명령 결과·감사 | 초기 설계에는 제한적 | `src/aiot/contracts.py`, 결과 collector, SQLite 저장·Public API | `tests/test_aiot_contracts.py`, 결과 수집 테스트 | 구현·검증 |
| 현장 운영·복구 | 과거 시연 절차 | `docker-compose*.yml`, `docs/operations/`, `scripts/health/` | 운영 점검 스크립트·Jetson 체크리스트 | 진행 |

## 증빙 작성 규칙

각 항목에는 다음 정보를 함께 남긴다.

```text
요구사항 ID
설계 문서 버전
구현 파일 또는 이미지 버전
테스트 명령과 결과
실증 장치·현장·일시
로그·스크린샷·영상 경로
미완료 조건과 담당자
```

“HTTP 200”, “시뮬레이션 성공”, “장치 응답 수신”은 물리적 출력 성공과 동일하지 않다. 실제 장치 동작은 별도의 실증 증빙으로 기록한다.

## 외부 연동 계약 근거

전임자 KETI Postman 자료에는 CCTV 이벤트를 oneM2M `m2m:cin` 형식으로 보고하는 계약이 포함되어 있다. 주요 근거는 다음과 같다.

- `POST` 데이터 보고
- `Content-Type: application/json;ty=4`
- `X-M2M-RI`, `X-M2M-Origin` 헤더
- `Event Type`, `Event Name`, `Accuracy`, `Event Source`, `Event Source Type`, `URL`, `Timestamp` 필드
- 최신 데이터·이력·에러 조회 API

현재 프로젝트의 내부 MQTT·EdgeX 이벤트 계약과 외부 oneM2M 보고 계약은 서로 다른 경계다. 내부 이벤트를 외부 계약으로 변환할 때 필드 매핑과 전송 성공 결과를 별도로 증빙한다.
