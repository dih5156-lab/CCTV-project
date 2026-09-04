# 디바이스 API 문서

장치 종류에 따라 아래 문서를 참고합니다.

| 문서 | 대상 | 통신 방식 |
|---|---|---|
| [스피커 API](SPEAKER_API.md) | InterM 스피커 | HTTP Digest |
| [전광판 API](SIGNBOARD_API.md) | Dabit Metrix 전광판 | TCP/EUC-KR, EdgeX HTTP 경계 |
| [경광등/사이렌 API](SIREN_API.md) | InterM 경광등 | HTTP Digest |
| [Action Layer 공통 API](ACTION_LAYER_DEVICE_API.md) | 세 장치 공통 제어 | Public API/내부 REST/MQTT |
| [이벤트 Payload 계약](EVENT_PAYLOADS.md) | AI·센서 이벤트와 장치별 변환 | MQTT/Action Layer |

## 권장 확인 순서

1. 장치별 문서에서 물리 장치의 주소, 포트, 인증 방식을 확인합니다.
2. `ACTION_LAYER_DEVICE_API.md`에서 사이트별 장치 선택과 제어 모드를 확인합니다.
3. `/api/v1/control/devices`로 설정/연결 상태를 확인합니다.
4. 테스트 이벤트를 발생시킨 뒤 `/api/v1/control/action-events`에서 장치별 결과를 확인합니다.

실제 계정, 비밀번호, API key는 문서에 기록하지 않고 운영 secret 또는 `.env.jetson`으로 전달합니다.
