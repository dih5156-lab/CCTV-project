# 기능 정리

사용자가 이용하는 기능, 외부 API, 이벤트 형식, AI 기능 연동 내용을 모았습니다.

## Public API와 이벤트

- [Public API 사용 가이드](PUBLIC_API_GUIDE.md): 인증, 상태, 이벤트, 카메라, 제어 API
- [Public API 호출 예시](PUBLIC_API_EXAMPLES.md): 바로 실행할 수 있는 `curl` 예시
- [외형 검색 상태 API](APPEARANCES_STATUS_API.md): 외형 검색 준비 상태와 응답 계약
- [이벤트 표준 스키마](EVENT_SCHEMA_STANDARD.md): 서비스 간 공통 이벤트 payload
- [외부 이벤트 수신](EXTERNAL_INGEST.md): MQTT 및 외부 입력 연동

## AI 기능

- [falldata 보조 검증](FALLDATA_INTEGRATION.md): 낙상 후보 shadow/confirm 검증 구조

설치와 활성화 방법은 [실행 및 배포 문서](../guides/README.md), 내부 구현 구조는 [모듈화 문서](../modules/README.md)를 참고하세요.
