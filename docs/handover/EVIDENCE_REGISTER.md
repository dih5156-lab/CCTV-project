# 정부과제 증빙 초기 등록 목록

기준일: 2026-09-04

이 문서는 현재 저장소에서 확인 가능한 증빙의 초기 목록이다. 공식 제출 전에는 과제 제안서·협약서·시험계획서의 요구사항 ID와 현장 시험 일시를 보완해야 한다.

상태는 `구현`, `자동 검증`, `dry-run`, `실제 장치 검증`, `미검증`을 구분한다.

| 증빙 ID | 대상 | 유형 | 근거 경로 | 현재 상태 | 보완 필요 |
|---|---|---|---|---|---|
| EV-ARCH-001 | 초기 Edge 플랫폼 구조 | 설계 문서·이미지 | 전임자 `7. SEdge-Platform.pdf`, `엣지 플랫폼/SEdge-Platform/src/SEdge-Platform.png` | 참고·이력 | 현재 EdgeX 구조와 변경 이력 연결 |
| EV-CCTV-001 | CCTV 영상 입력 | 코드·테스트 | `src/core`, `tests/test_deepstream_pipeline_builder.py` | 구현·자동 검증 | 현장 카메라별 시험 로그 등록 |
| EV-AI-001 | 이상행동 분석 | 모델·결과 이미지 | 전임자 `이상행동 검출 모델 구현/`, `src/core/_inference_pipeline.py` | 후보·진행 | 데이터셋·분할·현장 성능 고정 |
| EV-SENSOR-001 | AIoT 센서 파싱 | 코드·테스트 | `parser-python`, `src/aiot/contracts.py`, `tests/test_aiot_contracts.py` | 구현·자동 검증 | 실제 센서 원본과 시험 일시 등록 |
| EV-MQTT-001 | 이벤트 전달·재전송 | 코드·테스트 | `parser-python/mqtt`, `parser-python/tests/test_edgex_forwarder_contract.py` | 구현·자동 검증 | 운영 브로커 시험 로그 등록 |
| EV-RULE-001 | 규칙 처리 | 코드·설정·테스트 | `kuiper/`, `runners/run_kuiper_rules.py`, `src/services/sensor_rule_bridge.py` | 구현·진행 | eKuiper와 Python 규칙의 최종 판정 주체 확정 |
| EV-EDGEX-001 | 출력 명령 표준화 | 코드·테스트 | `src/services/action_bridge.py`, `src/edgex/command_contract.py`, `tests/test_action_bridge.py` | 구현·자동 검증 | 운영 모드 전환 승인 기록 |
| EV-SIGN-001 | 전광판 Device Service | 코드·테스트·API 계약 | `src/edgex/dabit_device_service.py`, `runners/run_signboard_device_service.py`, `tests/test_dabit_device_service.py`, `scripts/ops/check_edgex_device_service_contracts.py` | 구현·API `simulated` 응답 검증·canary 물리 출력 | 운영 전환 후 재검증 |
| EV-SIGN-002 | 전광판 EdgeX 명령 | 로그·DB·실장치 | [전광판 EdgeX 실장치 UAT](EDGEX_LIVE_UAT_20260904.md) 9월 2주차 재검증 포함 | 실제 명령·결과 DB·물리 호출 로그 확인 | 육안 촬영 사진 등록 |
| EV-SPEAKER-001 | 스피커 Device Service | 코드·dry-run | `src/edgex/speaker_device_service.py`, `scripts/ops/check_edgex_device_service_contracts.py` | dry-run | 실제 스피커 연결 후 UAT |
| EV-SIREN-001 | 사이렌·경광등 경계 | 코드·dry-run | `src/edgex/siren_device_service.py`, `runners/run_siren_device_service.py` | dry-run | 실제 장치·릴레이 확보 후 UAT |
| EV-MULTI-001 | 다중 장치 라우팅 | 설정·테스트·안전 UAT | `config/output_devices.json`, `tests/test_device_command_transport.py`, `docs/handover/EDGEX_LIVE_UAT_20260904.md#15-다중-디바이스-격리-uat-재검증` | `safe-002`에서 연결 불가 장치 제외와 전광판 simulated 결과 확인 | 2대 이상 실제 장치 장애 격리 시험 |
| EV-AUDIT-001 | 명령 결과·감사 | 코드·DB·테스트 | `src/aiot/contracts.py`, `data/runtime/edgex_command_results.db`, 관련 테스트 | 구현·자동 검증 | 운영 보존기간과 백업 증빙 |
| EV-OPS-001 | Docker·Jetson 운영 | 런북·점검 | `docs/architecture/DOCKER_AND_EXECUTION.md`, `scripts/health/`, `docs/guides/JETSON_EDGEX_FIELD_CHECKLIST.md` | 문서·점검 가능 | 실제 배포 로그와 복구 결과 등록 |
| EV-EXT-001 | 외부 CCTV 이벤트 계약 | API 자료 | 전임자 `엣지 플랫폼/KETI 연동 정보/쏘우웨이브 CCTV 이벤트 API 가이드.postman_collection.json` | 참고·계약 후보 | 공식 연동 시험 결과와 버전 확정 |

## 등록 시 주의

- 전임자 자료는 출처와 작성 시점을 함께 기록한다.
- 테스트 통과는 실제 현장 실증 통과와 분리한다.
- 인증정보·개인 연락처·내부 비밀값은 증빙 목록에 복사하지 않는다.
- 실제 장치 증빙에는 장치 ID, 명령, 응답, 물리 결과, 일시, 촬영 파일을 함께 남긴다.
- 모델 증빙에는 모델 파일 체크섬, 데이터셋 버전, 평가 임계값, Jetson 성능을 함께 남긴다.
